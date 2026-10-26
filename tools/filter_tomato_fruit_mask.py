#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Constrain an optimized tomato removal mask to the red fruit object. "
            "This removes nearby green stem/leaf Gaussians that entered through "
            "candidate expansion or opacity optimization."
        )
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_key", default="optimized_candidate")
    parser.add_argument("--cluster_key", default="post_opt_cluster_id")
    parser.add_argument("--selected_clusters", default="all")
    parser.add_argument("--red_min", default=0.25, type=float)
    parser.add_argument("--red_ratio_min", default=1.25, type=float)
    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument(
        "--near_red_radius",
        default=0.0,
        type=float,
        help="Optional shell radius around red fruit Gaussians for non-red highlights. Use 0 for strict fruit-red mask.",
    )
    parser.add_argument("--component_radius", default=0.018, type=float)
    parser.add_argument("--min_component_size", default=20, type=int)
    parser.add_argument(
        "--score_seeded_red_components",
        action="store_true",
        help="Use residual score clusters as seeds and keep the full connected red fruit components.",
    )
    parser.add_argument("--out_npz", default="tomato_fruit_mask.npz")
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


def red_like(rgb, opacity, args):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    return (
        (opacity >= args.opacity_min)
        & (r >= args.red_min)
        & (r >= g * args.red_ratio_min)
        & (r >= b * args.red_ratio_min)
    )


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


def bbox_diag(points):
    if points.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def assign_instances_from_score_clusters(xyz, final_mask, score_cluster_id):
    instance_id = np.full(xyz.shape[0], -1, dtype=np.int64)
    final_idx = np.flatnonzero(final_mask)
    source_clusters = sorted(int(v) for v in np.unique(score_cluster_id) if int(v) >= 0)
    if final_idx.size == 0 or not source_clusters:
        return instance_id, []

    source_points = []
    source_instance = []
    seed_counts = {}
    for instance, source_cluster in enumerate(source_clusters):
        seed_idx = np.flatnonzero(score_cluster_id == source_cluster)
        if seed_idx.size == 0:
            continue
        source_points.append(xyz[seed_idx])
        source_instance.append(np.full(seed_idx.size, instance, dtype=np.int64))
        seed_counts[instance] = int(seed_idx.size)
    if not source_points:
        return instance_id, []

    source_xyz = np.concatenate(source_points, axis=0)
    source_instance_id = np.concatenate(source_instance, axis=0)
    tree = cKDTree(source_xyz)
    _, nearest = tree.query(xyz[final_idx], k=1)
    instance_id[final_idx] = source_instance_id[nearest]

    summaries = []
    for instance in sorted(int(v) for v in np.unique(instance_id[final_idx]) if int(v) >= 0):
        members = final_idx[instance_id[final_idx] == instance]
        c_xyz = xyz[members]
        summaries.append(
            {
                "instance_id": int(instance),
                "source_cluster_id": int(source_clusters[instance]),
                "seed_gaussians": int(seed_counts.get(instance, 0)),
                "fruit_gaussians": int(members.size),
                "centroid_xyz": c_xyz.mean(axis=0).tolist(),
                "bbox_min": c_xyz.min(axis=0).tolist(),
                "bbox_max": c_xyz.max(axis=0).tolist(),
                "bbox_diag": bbox_diag(c_xyz),
            }
        )
    return instance_id, summaries


def build_score_seeded_red_components(xyz, red, score_cluster_id, radius):
    final_mask = np.zeros(xyz.shape[0], dtype=bool)
    instance_id = np.full(xyz.shape[0], -1, dtype=np.int64)
    red_idx = np.flatnonzero(red)
    if red_idx.size == 0:
        return final_mask, instance_id, [], []

    components = connected_components(xyz[red_idx], radius)
    component_summaries = []
    tomato_instances = []
    for component_id, component in enumerate(components):
        global_idx = red_idx[component]
        source_clusters = sorted(int(v) for v in np.unique(score_cluster_id[global_idx]) if int(v) >= 0)
        keep = bool(source_clusters)
        component_summaries.append(
            {
                "component": int(component_id),
                "num_gaussians": int(global_idx.size),
                "bbox_diag": bbox_diag(xyz[global_idx]),
                "source_clusters": source_clusters,
                "kept": keep,
            }
        )
        if not keep:
            continue

        final_mask[global_idx] = True
        seed_points = []
        seed_instances = []
        for source_cluster in source_clusters:
            seed_idx = np.flatnonzero(score_cluster_id == source_cluster)
            if seed_idx.size == 0:
                continue
            seed_points.append(xyz[seed_idx])
            seed_instances.append(np.full(seed_idx.size, source_cluster, dtype=np.int64))
        if seed_points:
            seed_xyz = np.concatenate(seed_points, axis=0)
            seed_instance_id = np.concatenate(seed_instances, axis=0)
            tree = cKDTree(seed_xyz)
            _, nearest = tree.query(xyz[global_idx], k=1)
            instance_id[global_idx] = seed_instance_id[nearest]

    for instance in sorted(int(v) for v in np.unique(instance_id[final_mask]) if int(v) >= 0):
        members = np.flatnonzero(final_mask & (instance_id == instance))
        c_xyz = xyz[members]
        tomato_instances.append(
            {
                "instance_id": int(instance),
                "source_cluster_id": int(instance),
                "seed_gaussians": int((score_cluster_id == instance).sum()),
                "fruit_gaussians": int(members.size),
                "centroid_xyz": c_xyz.mean(axis=0).tolist(),
                "bbox_min": c_xyz.min(axis=0).tolist(),
                "bbox_max": c_xyz.max(axis=0).tolist(),
                "bbox_diag": bbox_diag(c_xyz),
            }
        )

    return final_mask, instance_id, [{"cluster_id": -1, "components": component_summaries}], tomato_instances


def write_ply(path, xyz, fruit_mask, rejected_mask):
    points = []
    colors = []
    labels = []
    fruit_idx = np.flatnonzero(fruit_mask)
    rejected_idx = np.flatnonzero(rejected_mask)
    if fruit_idx.size:
        points.append(xyz[fruit_idx])
        colors.append(np.tile(np.asarray((235, 70, 45), dtype=np.uint8), (fruit_idx.size, 1)))
        labels.append(np.zeros(fruit_idx.size, dtype=np.int32))
    if rejected_idx.size:
        points.append(xyz[rejected_idx])
        colors.append(np.tile(np.asarray((70, 210, 95), dtype=np.uint8), (rejected_idx.size, 1)))
        labels.append(np.ones(rejected_idx.size, dtype=np.int32))
    all_points = np.concatenate(points, axis=0) if points else np.zeros((0, 3), dtype=np.float32)
    all_colors = np.concatenate(colors, axis=0) if colors else np.zeros((0, 3), dtype=np.uint8)
    all_labels = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int32)
    vertex = np.empty(
        all_points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
        ],
    )
    if all_points.shape[0]:
        vertex["x"] = all_points[:, 0]
        vertex["y"] = all_points[:, 1]
        vertex["z"] = all_points[:, 2]
        vertex["red"] = all_colors[:, 0]
        vertex["green"] = all_colors[:, 1]
        vertex["blue"] = all_colors[:, 2]
        vertex["label"] = all_labels
    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(path)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    presence = np.load(args.presence_npz)
    scores = np.load(args.score_npz)
    xyz = presence["xyz"].astype(np.float32)
    if scores["xyz"].shape[0] != xyz.shape[0]:
        raise RuntimeError("presence_npz and score_npz have different Gaussian counts")
    if args.mask_key not in presence.files:
        raise RuntimeError(f"{args.mask_key} is absent from {args.presence_npz}")
    if args.cluster_key not in presence.files:
        raise RuntimeError(f"{args.cluster_key} is absent from {args.presence_npz}")

    selected = presence[args.mask_key].astype(bool)
    cluster_id = presence[args.cluster_key].astype(np.int64)
    selected_clusters = parse_cluster_ids(args.selected_clusters, cluster_id, selected)
    selected &= np.isin(cluster_id, np.asarray(selected_clusters, dtype=np.int64))

    rgb = scores["rgb"].astype(np.float32)
    opacity = presence["opacity_pre"].reshape(-1).astype(np.float32)
    red = red_like(rgb, opacity, args)
    if "cluster_id" in scores.files:
        score_cluster_id = scores["cluster_id"].astype(np.int64)
    else:
        score_cluster_id = np.full(xyz.shape[0], -1, dtype=np.int64)

    if args.score_seeded_red_components:
        final_mask, tomato_instance_id, cluster_summaries, tomato_instances = build_score_seeded_red_components(
            xyz, red, score_cluster_id, args.component_radius
        )
        if not final_mask.any():
            raise RuntimeError("No residual-seeded red tomato components were found")
        selected = final_mask.copy()
        rejected = red & ~final_mask
        selected_clusters = sorted(int(v) for v in np.unique(score_cluster_id[final_mask]) if int(v) >= 0)

        payload = {key: presence[key] for key in presence.files}
        payload[f"raw_{args.mask_key}"] = presence[args.mask_key]
        payload[args.mask_key] = final_mask
        payload["tomato_fruit_candidate"] = final_mask
        payload["tomato_instance_id"] = tomato_instance_id
        payload["tomato_fruit_rejected_nonfruit"] = rejected
        np.savez_compressed(out_dir / args.out_npz, **payload)
        write_ply(out_dir / "tomato_fruit_mask.ply", xyz, final_mask, rejected)

        fruit_points = xyz[final_mask]
        rejected_points = xyz[rejected]
        summary = {
            "presence_npz": args.presence_npz,
            "score_npz": args.score_npz,
            "mask_key": args.mask_key,
            "cluster_key": args.cluster_key,
            "selected_clusters": selected_clusters,
            "method": "score_seeded_red_connected_components",
            "red_component_radius": float(args.component_radius),
            "input_selected_gaussians": int(selected.sum()),
            "fruit_gaussians": int(final_mask.sum()),
            "rejected_nonfruit_gaussians": int(rejected.sum()),
            "fruit_bbox_min": fruit_points.min(axis=0).tolist(),
            "fruit_bbox_max": fruit_points.max(axis=0).tolist(),
            "fruit_bbox_diag": bbox_diag(fruit_points),
            "rejected_bbox_diag": bbox_diag(rejected_points),
            "clusters": cluster_summaries,
            "tomato_instances": tomato_instances,
        }
        (out_dir / "tomato_fruit_mask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {out_dir / args.out_npz}")
        print("Saved " + str(out_dir / "tomato_fruit_mask.ply"))
        print("Saved " + str(out_dir / "tomato_fruit_mask_summary.json"))
        print(json.dumps({"input_selected": int(selected.sum()), "fruit": int(final_mask.sum()), "instances": len(tomato_instances)}, indent=2))
        return

    red_selected = selected & red
    if not red_selected.any():
        raise RuntimeError("No red selected tomato Gaussians are available for fruit filtering")

    fruit_mask = red_selected.copy()
    if args.near_red_radius > 0.0:
        tree = cKDTree(xyz[red_selected])
        dist, _ = tree.query(xyz[selected], k=1)
        selected_idx = np.flatnonzero(selected)
        near_red = np.zeros_like(selected, dtype=bool)
        near_red[selected_idx] = dist <= args.near_red_radius
        fruit_mask |= selected & near_red

    final_mask = np.zeros_like(selected, dtype=bool)
    cluster_summaries = []
    for cid in selected_clusters:
        cluster_fruit = fruit_mask & (cluster_id == cid)
        idx = np.flatnonzero(cluster_fruit)
        components = connected_components(xyz[idx], args.component_radius)
        component_summaries = []
        for component_id, component in enumerate(components):
            global_idx = idx[component]
            keep = component_id == 0 or component.size >= args.min_component_size
            if keep:
                final_mask[global_idx] = True
            component_summaries.append(
                {
                    "component": int(component_id),
                    "num_gaussians": int(component.size),
                    "bbox_diag": bbox_diag(xyz[global_idx]),
                    "kept": bool(keep),
                }
            )
        cluster_summaries.append({"cluster_id": int(cid), "components": component_summaries})

    rejected = selected & ~final_mask
    if "cluster_id" in scores.files:
        score_cluster_id = scores["cluster_id"].astype(np.int64)
    else:
        score_cluster_id = np.full(xyz.shape[0], -1, dtype=np.int64)
    tomato_instance_id, tomato_instances = assign_instances_from_score_clusters(
        xyz, final_mask, score_cluster_id
    )

    payload = {key: presence[key] for key in presence.files}
    payload[f"raw_{args.mask_key}"] = presence[args.mask_key]
    payload[args.mask_key] = final_mask
    payload["tomato_fruit_candidate"] = final_mask
    payload["tomato_instance_id"] = tomato_instance_id
    payload["tomato_fruit_rejected_nonfruit"] = rejected
    np.savez_compressed(out_dir / args.out_npz, **payload)
    write_ply(out_dir / "tomato_fruit_mask.ply", xyz, final_mask, rejected)

    fruit_points = xyz[final_mask]
    rejected_points = xyz[rejected]
    summary = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "mask_key": args.mask_key,
        "cluster_key": args.cluster_key,
        "selected_clusters": selected_clusters,
        "method": "tomato_red_fruit_mask_filter",
        "near_red_radius": float(args.near_red_radius),
        "input_selected_gaussians": int(selected.sum()),
        "fruit_gaussians": int(final_mask.sum()),
        "rejected_nonfruit_gaussians": int(rejected.sum()),
        "fruit_bbox_min": fruit_points.min(axis=0).tolist() if fruit_points.size else None,
        "fruit_bbox_max": fruit_points.max(axis=0).tolist() if fruit_points.size else None,
        "fruit_bbox_diag": bbox_diag(fruit_points),
        "rejected_bbox_diag": bbox_diag(rejected_points),
        "clusters": cluster_summaries,
        "tomato_instances": tomato_instances,
    }
    (out_dir / "tomato_fruit_mask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {out_dir / args.out_npz}")
    print(f"Saved {out_dir / 'tomato_fruit_mask.ply'}")
    print(f"Saved {out_dir / 'tomato_fruit_mask_summary.json'}")
    print(
        json.dumps(
            {
                "input_selected": int(selected.sum()),
                "fruit": int(final_mask.sum()),
                "rejected_nonfruit": int(rejected.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
