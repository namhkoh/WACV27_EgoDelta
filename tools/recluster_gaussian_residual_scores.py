#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from change_detection.cluster_gaussians import dbscan_numpy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute candidate and DBSCAN cluster labels from an existing "
            "Gaussian residual score file without rerendering residuals."
        )
    )
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_key", default="score")
    parser.add_argument("--score_quantile", default=0.80, type=float)
    parser.add_argument("--score_min", default=0.05, type=float)
    parser.add_argument("--cluster_eps", default=0.04, type=float)
    parser.add_argument("--cluster_min_samples", default=4, type=int)
    parser.add_argument("--cluster_max_bbox_diag", default=0.35, type=float)
    parser.add_argument("--cluster_max_planarity", default=200.0, type=float, help="Reject clusters above this PCA ratio. Use <=0 to disable this filter.")
    return parser.parse_args()


def cluster_shape(xyz):
    if xyz.shape[0] < 3:
        return np.zeros(3, dtype=np.float32)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, xyz.shape[0] - 1)
    return np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)


def summarize_clusters(xyz, opacity, score, residual, visibility, candidate_indices, labels_local, args):
    summaries = []
    labels_full = np.full(xyz.shape[0], -1, dtype=np.int64)
    valid_labels = sorted(int(v) for v in np.unique(labels_local) if int(v) >= 0)

    new_id = 0
    for label in valid_labels:
        local = np.flatnonzero(labels_local == label)
        members = candidate_indices[local]
        c_xyz = xyz[members]
        eigvals = cluster_shape(c_xyz)
        bbox_min = c_xyz.min(axis=0)
        bbox_max = c_xyz.max(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
        planarity = float(eigvals[0] / (eigvals[2] + 1e-8)) if eigvals[2] > 0 else float("inf")
        if bbox_diag > args.cluster_max_bbox_diag:
            continue
        if args.cluster_max_planarity > 0.0 and planarity > args.cluster_max_planarity:
            continue

        labels_full[members] = new_id
        summaries.append(
            {
                "cluster_id": int(new_id),
                "source_label": int(label),
                "num_gaussians": int(members.size),
                "mean_score": float(score[members].mean()),
                "max_score": float(score[members].max()),
                "mean_residual": float(residual[members].mean()),
                "mean_visibility_count": float(visibility[members].mean()),
                "mean_opacity_pre": float(opacity[members].mean()),
                "centroid_xyz": c_xyz.mean(axis=0).tolist(),
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "bbox_diag": bbox_diag,
                "pca_eigvals": eigvals.tolist(),
                "planarity_ratio": planarity,
                "label": "reclustered dense multi-view residual candidate",
            }
        )
        new_id += 1
    return labels_full, summaries


def write_score_ply(path, xyz, rgb, opacity, score, residual, high_ratio, visibility, density, cluster_labels, candidate):
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    colors[candidate] = np.array([255, 60, 40], dtype=np.uint8)
    colors[cluster_labels >= 0] = np.array([255, 180, 30], dtype=np.uint8)

    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("score", "f4"),
            ("residual_mean", "f4"),
            ("high_residual_ratio", "f4"),
            ("visibility_count", "i4"),
            ("density_count", "i4"),
            ("opacity_pre", "f4"),
            ("candidate", "u1"),
            ("cluster_id", "i4"),
        ],
    )
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["nx"] = 0.0
    vertex["ny"] = 0.0
    vertex["nz"] = 0.0
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    vertex["score"] = score.astype(np.float32)
    vertex["residual_mean"] = residual.astype(np.float32)
    vertex["high_residual_ratio"] = high_ratio.astype(np.float32)
    vertex["visibility_count"] = visibility.astype(np.int32)
    vertex["density_count"] = density.astype(np.int32)
    vertex["opacity_pre"] = opacity.astype(np.float32)
    vertex["candidate"] = candidate.astype(np.uint8)
    vertex["cluster_id"] = cluster_labels.astype(np.int32)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_path = Path(args.score_npz)
    data = np.load(score_path)
    if args.score_key not in data.files:
        raise RuntimeError(f"{args.score_key} is absent from {score_path}")

    score = data[args.score_key].astype(np.float32)
    positive = score[score > 0]
    score_threshold = max(
        float(args.score_min),
        float(np.quantile(positive, args.score_quantile)) if positive.size else 1.0,
    )
    candidate = score >= score_threshold
    candidate_indices = np.flatnonzero(candidate)

    if candidate_indices.size:
        labels_local = dbscan_numpy(
            data["xyz"][candidate_indices].astype(np.float32),
            args.cluster_eps,
            args.cluster_min_samples,
        )
        cluster_id, clusters = summarize_clusters(
            data["xyz"].astype(np.float32),
            data["opacity_pre"].reshape(-1).astype(np.float32),
            score,
            data["residual_mean"].astype(np.float32),
            data["visibility_count"].astype(np.float32),
            candidate_indices,
            labels_local,
            args,
        )
    else:
        cluster_id = np.full(score.shape[0], -1, dtype=np.int64)
        clusters = []

    payload = {key: data[key] for key in data.files}
    payload["candidate"] = candidate
    payload["cluster_id"] = cluster_id
    np.savez(out_dir / "gaussian_multiview_residual_scores.npz", **payload)

    write_score_ply(
        out_dir / "gaussian_multiview_residual_scores.ply",
        data["xyz"].astype(np.float32),
        data["rgb"].astype(np.float32),
        data["opacity_pre"].reshape(-1).astype(np.float32),
        score,
        data["residual_mean"].astype(np.float32),
        data["high_residual_ratio"].astype(np.float32),
        data["visibility_count"].astype(np.int32),
        data["density_count"].astype(np.int32),
        cluster_id,
        candidate,
    )
    (out_dir / "multiview_residual_clusters.json").write_text(json.dumps(clusters, indent=2) + "\n")

    source_config_path = score_path.parent / "score_config.json"
    config = {}
    if source_config_path.exists():
        config = json.loads(source_config_path.read_text())
    config.update(
        {
            "reclustered_from": str(score_path),
            "score_key": args.score_key,
            "score_quantile": float(args.score_quantile),
            "score_min": float(args.score_min),
            "score_threshold": float(score_threshold),
            "cluster_eps": float(args.cluster_eps),
            "cluster_min_samples": int(args.cluster_min_samples),
            "cluster_max_bbox_diag": float(args.cluster_max_bbox_diag),
            "cluster_max_planarity": float(args.cluster_max_planarity),
            "num_candidates": int(candidate.sum()),
            "num_clusters": int(len(clusters)),
        }
    )
    (out_dir / "score_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"Saved {out_dir / 'gaussian_multiview_residual_scores.npz'}")
    print(f"Saved {out_dir / 'gaussian_multiview_residual_scores.ply'}")
    print(f"Saved {out_dir / 'multiview_residual_clusters.json'} ({len(clusters)} clusters)")
    print(json.dumps({"candidates": int(candidate.sum()), "clusters": clusters}, indent=2))


if __name__ == "__main__":
    main()
