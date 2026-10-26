#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split merged tomato fruit Gaussian components into per-fruit instances."
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_key", default="tomato_fruit_candidate")
    parser.add_argument("--cluster_key", default="tomato_instance_id")
    parser.add_argument("--source_cluster", default="largest")
    parser.add_argument("--num_splits", default=2, type=int)
    parser.add_argument("--auto", action="store_true", help="Infer split count from local graph connectivity.")
    parser.add_argument("--min_instance_gaussians", default=50, type=int)
    parser.add_argument("--min_instance_fraction", default=0.08, type=float)
    parser.add_argument("--min_split_fraction", default=0.12, type=float)
    parser.add_argument("--adaptive_neighbor_k", default=3, type=int)
    parser.add_argument("--adaptive_radius_quantiles", default="0.99,0.98,0.95,0.90")
    parser.add_argument("--drop_other_clusters_below", default=50, type=int)
    parser.add_argument("--iterations", default=80, type=int)
    parser.add_argument("--out_npz", default="tomato_fruit_mask.npz")
    return parser.parse_args()


def bbox_diag(points):
    if points.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def pca_init(points, k):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = centered @ axis
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = []
    for quantile in quantiles:
        target = np.quantile(projection, quantile)
        idx = int(np.argmin(np.abs(projection - target)))
        centers.append(points[idx])
    return np.asarray(centers, dtype=np.float32)


def kmeans(points, k, iterations):
    centers = pca_init(points, k)
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = points[labels == cluster]
            if members.shape[0]:
                centers[cluster] = members.mean(axis=0)
    return labels, centers


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
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in neighbors[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        components.append(np.asarray(members, dtype=np.int64))
    components.sort(key=lambda item: item.size, reverse=True)
    return components


def parse_quantiles(spec):
    return sorted({float(part.strip()) for part in str(spec).split(",") if part.strip()}, reverse=True)


def adaptive_graph_split(points, global_idx, args):
    n_points = points.shape[0]
    min_size = max(
        int(args.min_instance_gaussians),
        int(np.ceil(float(args.min_instance_fraction) * n_points)),
    )
    min_second = max(
        int(args.min_instance_gaussians),
        int(np.ceil(float(args.min_split_fraction) * n_points)),
    )
    if n_points < max(2 * min_second, int(args.adaptive_neighbor_k) + 1):
        if n_points >= min_size:
            return [global_idx], {
                "mode": "single_small_component",
                "min_size": int(min_size),
                "num_gaussians": int(n_points),
            }
        return [], {
            "mode": "rejected_too_small",
            "min_size": int(min_size),
            "num_gaussians": int(n_points),
        }

    tree = cKDTree(points)
    k = min(int(args.adaptive_neighbor_k) + 1, n_points)
    distances, _ = tree.query(points, k=k)
    scale_distances = distances[:, -1]
    trials = []
    best = None
    for quantile in parse_quantiles(args.adaptive_radius_quantiles):
        radius = float(np.quantile(scale_distances, quantile))
        if radius <= 0.0:
            continue
        components = connected_components(points, radius)
        sizable = [component for component in components if component.size >= min_second]
        coverage = float(sum(component.size for component in sizable) / max(1, n_points))
        trial = {
            "radius": radius,
            "quantile": float(quantile),
            "component_sizes": [int(component.size) for component in components[:10]],
            "sizable_sizes": [int(component.size) for component in sizable],
            "sizable_coverage": coverage,
        }
        trials.append(trial)
        if len(sizable) >= 2 and coverage >= 0.80:
            best = (components, sizable, trial)
            break

    if best is None:
        if n_points >= min_size:
            return [global_idx], {
                "mode": "single_component_no_stable_split",
                "min_size": int(min_size),
                "num_gaussians": int(n_points),
                "trials": trials,
            }
        return [], {
            "mode": "rejected_no_stable_split",
            "min_size": int(min_size),
            "num_gaussians": int(n_points),
            "trials": trials,
        }

    components, sizable, best_trial = best
    centers = np.asarray([points[component].mean(axis=0) for component in sizable], dtype=np.float32)
    assignments = [list(component.tolist()) for component in sizable]
    sizable_members = set(np.concatenate(sizable).tolist())
    for component in components:
        if component.size >= min_second:
            continue
        for local_id in component:
            if int(local_id) in sizable_members:
                continue
            nearest = int(np.argmin(np.linalg.norm(centers - points[local_id][None, :], axis=1)))
            assignments[nearest].append(int(local_id))

    split_indices = [global_idx[np.asarray(sorted(set(local_ids)), dtype=np.int64)] for local_ids in assignments]
    split_indices.sort(key=lambda item: item.size, reverse=True)
    return split_indices, {
        "mode": "adaptive_graph_split",
        "min_size": int(min_size),
        "min_second_size": int(min_second),
        "best_trial": best_trial,
        "trials": trials,
    }


def manual_split(points, global_idx, args):
    if global_idx.size < args.num_splits:
        raise RuntimeError(f"Cannot split {global_idx.size} Gaussians into {args.num_splits} instances")
    labels, _ = kmeans(points, args.num_splits, args.iterations)
    split_indices = []
    for local_label in range(args.num_splits):
        members = global_idx[labels == local_label]
        if members.size:
            split_indices.append(members)
    split_indices.sort(key=lambda item: item.size, reverse=True)
    return split_indices, {"mode": "manual_pca_initialized_kmeans", "num_splits": int(args.num_splits)}


def add_instance_summary(summaries, instance_id, source_cluster, split_label, members, xyz, split_diagnostics):
    pts = xyz[members]
    summaries.append(
        {
            "instance_id": int(instance_id),
            "source_cluster_id": int(source_cluster),
            "split_label": None if split_label is None else int(split_label),
            "fruit_gaussians": int(members.size),
            "centroid_xyz": pts.mean(axis=0).tolist(),
            "bbox_min": pts.min(axis=0).tolist(),
            "bbox_max": pts.max(axis=0).tolist(),
            "bbox_diag": bbox_diag(pts),
            "split_diagnostics": split_diagnostics,
        }
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.presence_npz)
    xyz = data["xyz"].astype(np.float32)
    selected = data[args.mask_key].astype(bool)
    source_id = data[args.cluster_key].astype(np.int64)
    source_clusters = sorted(int(v) for v in np.unique(source_id[selected]) if int(v) >= 0)
    if not source_clusters:
        raise RuntimeError("No source tomato instances are available")

    if str(args.source_cluster).lower() == "largest":
        split_source = max(source_clusters, key=lambda cid: int((selected & (source_id == cid)).sum()))
    elif str(args.source_cluster).lower() in ("all", "auto"):
        split_source = None
    else:
        split_source = int(args.source_cluster)
        if split_source not in source_clusters:
            raise RuntimeError(f"source cluster {split_source} is absent from {source_clusters}")

    instance_id = np.full(xyz.shape[0], -1, dtype=np.int64)
    final_mask = np.zeros_like(selected, dtype=bool)
    summaries = []
    dropped = []
    next_instance = 0

    largest_size = max(int((selected & (source_id == cid)).sum()) for cid in source_clusters)
    for cid in source_clusters:
        members = np.flatnonzero(selected & (source_id == cid))
        should_split = args.auto and (split_source is None or cid == split_source)
        if not should_split and cid != split_source and members.size < args.drop_other_clusters_below:
            dropped.append({"source_cluster_id": int(cid), "fruit_gaussians": int(members.size), "reason": "below_drop_threshold"})
            continue

        if should_split:
            split_parts, diagnostics = adaptive_graph_split(xyz[members], members, args)
            if not split_parts:
                dropped.append({
                    "source_cluster_id": int(cid),
                    "fruit_gaussians": int(members.size),
                    "reason": diagnostics.get("mode", "auto_rejected"),
                    "split_diagnostics": diagnostics,
                })
                continue
        elif cid == split_source:
            split_parts, diagnostics = manual_split(xyz[members], members, args)
        else:
            split_parts, diagnostics = [members], {"mode": "kept_existing_component"}

        for split_label, part in enumerate(split_parts):
            if part.size < args.min_instance_gaussians and part.size < largest_size * args.min_instance_fraction:
                dropped.append({
                    "source_cluster_id": int(cid),
                    "fruit_gaussians": int(part.size),
                    "reason": "split_part_below_min_instance_size",
                })
                continue
            instance_id[part] = next_instance
            final_mask[part] = True
            add_instance_summary(summaries, next_instance, cid, split_label, part, xyz, diagnostics)
            next_instance += 1

    if not final_mask.any():
        raise RuntimeError("All tomato fruit components were rejected")

    payload = {key: data[key] for key in data.files}
    payload[f"raw_{args.cluster_key}"] = data[args.cluster_key]
    payload[args.mask_key] = final_mask
    payload["tomato_fruit_candidate"] = final_mask
    payload["tomato_instance_id"] = instance_id
    payload["tomato_fruit_rejected_nonfruit"] = selected & ~final_mask
    np.savez_compressed(out_dir / args.out_npz, **payload)

    final_points = xyz[final_mask]
    summary = {
        "presence_npz": args.presence_npz,
        "method": "adaptive_graph_split_tomato_fruit_instances" if args.auto else "pca_initialized_kmeans_split_of_merged_tomato_component",
        "source_cluster": None if split_source is None else int(split_source),
        "auto": bool(args.auto),
        "num_splits": None if args.auto else int(args.num_splits),
        "drop_other_clusters_below": int(args.drop_other_clusters_below),
        "min_instance_gaussians": int(args.min_instance_gaussians),
        "min_instance_fraction": float(args.min_instance_fraction),
        "min_split_fraction": float(args.min_split_fraction),
        "fruit_gaussians": int(final_mask.sum()),
        "fruit_bbox_min": final_points.min(axis=0).tolist(),
        "fruit_bbox_max": final_points.max(axis=0).tolist(),
        "fruit_bbox_diag": bbox_diag(final_points),
        "tomato_instances": summaries,
        "dropped_source_clusters": dropped,
    }
    (out_dir / "tomato_fruit_mask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {out_dir / args.out_npz}")
    print(f"Saved {out_dir / 'tomato_fruit_mask_summary.json'}")
    print(json.dumps({"instances": len(summaries), "fruit": int(final_mask.sum()), "dropped": dropped}, indent=2))


if __name__ == "__main__":
    main()
