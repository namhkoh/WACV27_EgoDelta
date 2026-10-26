#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


COLORS = {
    0: (255, 180, 30),
    1: (80, 170, 255),
    2: (255, 80, 180),
    3: (80, 220, 100),
    4: (210, 210, 80),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune small low-score connected components from a selected Gaussian mask."
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_key", default="optimized_candidate")
    parser.add_argument("--cluster_key", default="post_opt_cluster_id")
    parser.add_argument("--selected_clusters", default="all")
    parser.add_argument("--component_radius", default=0.018, type=float)
    parser.add_argument("--min_component_size", default=120, type=int)
    parser.add_argument("--min_component_ratio", default=0.08, type=float)
    parser.add_argument("--min_median_score", default=0.25, type=float)
    parser.add_argument("--out_npz", default="component_pruned_presence_scores.npz")
    return parser.parse_args()


def parse_cluster_ids(spec, cluster_id, selected):
    valid = sorted(int(v) for v in np.unique(cluster_id[selected]) if int(v) >= 0)
    spec = str(spec).strip().lower()
    if spec in ("all", "-1", ""):
        return valid
    requested = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    missing = sorted(set(requested) - set(valid))
    if missing:
        raise RuntimeError(f"Requested clusters {missing} are absent from selected candidates {valid}")
    return requested


def connected_components(points, radius):
    if points.shape[0] == 0:
        return []
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, radius)
    seen = np.zeros(points.shape[0], dtype=bool)
    components = []
    for seed in range(points.shape[0]):
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(key=lambda item: item.size, reverse=True)
    return components


def bbox_diag(points):
    if points.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def write_ply(path, xyz, cluster_id, pruned, removed):
    points = []
    colors = []
    labels = []
    for cid in sorted(int(v) for v in np.unique(cluster_id[pruned | removed]) if int(v) >= 0):
        color = np.asarray(COLORS.get(cid, (255, 255, 255)), dtype=np.uint8)
        kept_idx = np.flatnonzero(pruned & (cluster_id == cid))
        removed_idx = np.flatnonzero(removed & (cluster_id == cid))
        if kept_idx.size:
            points.append(xyz[kept_idx])
            colors.append(np.tile(color, (kept_idx.size, 1)))
            labels.append(np.full(kept_idx.size, cid, dtype=np.int32))
        if removed_idx.size:
            points.append(xyz[removed_idx])
            colors.append(np.tile(np.asarray((255, 40, 40), dtype=np.uint8), (removed_idx.size, 1)))
            labels.append(np.full(removed_idx.size, cid + 100, dtype=np.int32))

    all_points = np.concatenate(points, axis=0) if points else np.zeros((0, 3), dtype=np.float32)
    all_colors = np.concatenate(colors, axis=0) if colors else np.zeros((0, 3), dtype=np.uint8)
    all_labels = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int32)
    with Path(path).open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {all_points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int label\nend_header\n")
        for point, color, label in zip(all_points, all_colors, all_labels):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label)}\n"
            )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    presence = np.load(args.presence_npz)
    score_npz = np.load(args.score_npz)
    if args.mask_key not in presence.files:
        raise RuntimeError(f"{args.mask_key} is absent from {args.presence_npz}")
    if args.cluster_key not in presence.files:
        raise RuntimeError(f"{args.cluster_key} is absent from {args.presence_npz}")
    if "score" not in score_npz.files:
        raise RuntimeError(f"score is absent from {args.score_npz}")

    xyz = presence["xyz"].astype(np.float32)
    if score_npz["xyz"].shape[0] != xyz.shape[0]:
        raise RuntimeError("presence_npz and score_npz have different Gaussian counts")

    selected = presence[args.mask_key].astype(bool)
    cluster_id = presence[args.cluster_key].astype(np.int64)
    residual_score = score_npz["score"].astype(np.float32)
    cluster_ids = parse_cluster_ids(args.selected_clusters, cluster_id, selected)

    pruned = selected.copy()
    removed = np.zeros_like(selected, dtype=bool)
    cluster_summaries = []
    for cid in cluster_ids:
        cluster_mask = selected & (cluster_id == cid)
        cluster_indices = np.flatnonzero(cluster_mask)
        components = connected_components(xyz[cluster_indices], args.component_radius)
        if not components:
            continue
        largest = max(component.size for component in components)
        component_summaries = []
        for component_idx, component in enumerate(components):
            global_idx = cluster_indices[component]
            ratio = float(component.size / max(largest, 1))
            median_score = float(np.median(residual_score[global_idx]))
            should_remove = (
                component_idx > 0
                and component.size < args.min_component_size
                and ratio < args.min_component_ratio
                and median_score < args.min_median_score
            )
            if should_remove:
                pruned[global_idx] = False
                removed[global_idx] = True
            component_summaries.append(
                {
                    "component": int(component_idx),
                    "num_gaussians": int(component.size),
                    "size_ratio_to_largest": ratio,
                    "median_score": median_score,
                    "bbox_diag": bbox_diag(xyz[global_idx]),
                    "removed": bool(should_remove),
                }
            )
        cluster_summaries.append({"cluster_id": int(cid), "components": component_summaries})

    payload = {key: presence[key] for key in presence.files}
    payload[f"raw_{args.mask_key}"] = selected
    payload[args.mask_key] = pruned
    payload["component_pruned_candidate"] = pruned
    payload["component_pruned_removed"] = removed
    np.savez_compressed(out_dir / args.out_npz, **payload)

    summary = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "mask_key": args.mask_key,
        "cluster_key": args.cluster_key,
        "selected_clusters": cluster_ids,
        "component_radius": float(args.component_radius),
        "rule": {
            "remove_non_largest": True,
            "min_component_size": int(args.min_component_size),
            "min_component_ratio": float(args.min_component_ratio),
            "min_median_score": float(args.min_median_score),
        },
        "base_gaussians": int(selected.sum()),
        "pruned_gaussians": int(pruned.sum()),
        "removed_gaussians": int(removed.sum()),
        "clusters": cluster_summaries,
    }
    (out_dir / "component_pruned_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_ply(out_dir / "component_pruned_mask.ply", xyz, cluster_id, pruned, removed)
    print(f"Saved {out_dir / args.out_npz}")
    print(f"Saved {out_dir / 'component_pruned_summary.json'}")
    print(f"Saved {out_dir / 'component_pruned_mask.ply'}")
    print(json.dumps({"removed_gaussians": int(removed.sum()), "pruned_gaussians": int(pruned.sum())}, indent=2))


if __name__ == "__main__":
    main()
