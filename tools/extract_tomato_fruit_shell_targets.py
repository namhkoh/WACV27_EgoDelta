#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


COLORS = {
    0: (235, 70, 45),
    1: (255, 170, 30),
    2: (80, 170, 255),
    3: (255, 80, 180),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Infer tomato harvest targets from red fruit instances by searching "
            "nearby pre-scene green/calyx Gaussians. The target is the green "
            "contact medoid closest to each removed fruit."
        )
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_key", default="tomato_fruit_candidate")
    parser.add_argument("--cluster_key", default="tomato_instance_id")
    parser.add_argument("--selected_clusters", default="all")

    parser.add_argument("--search_radius", default=0.09, type=float)
    parser.add_argument("--min_contact_distance", default=0.002, type=float)
    parser.add_argument("--component_radius", default=0.018, type=float)
    parser.add_argument("--min_component_size", default=2, type=int)
    parser.add_argument("--target_points", default=12, type=int)
    parser.add_argument("--other_fruit_margin", default=0.02, type=float)

    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--green_min", default=0.08, type=float)
    parser.add_argument("--green_margin", default=0.0, type=float)
    parser.add_argument("--value_min", default=0.06, type=float)
    parser.add_argument("--red_min", default=0.25, type=float)
    parser.add_argument("--red_ratio_min", default=1.25, type=float)
    return parser.parse_args()


def parse_cluster_ids(spec, cluster_id, selected):
    valid = sorted(int(v) for v in np.unique(cluster_id[selected]) if int(v) >= 0)
    spec = str(spec).strip().lower()
    if spec in ("all", "-1", ""):
        return valid
    requested = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    missing = sorted(set(requested) - set(valid))
    if missing:
        raise RuntimeError(f"Requested clusters {missing} are absent from {valid}")
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


def green_like(rgb, opacity, args):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    value = np.max(rgb, axis=1)
    strong_red = (r > g * args.red_ratio_min) & (r > b * args.red_ratio_min)
    return (
        (opacity >= args.opacity_min)
        & (g >= args.green_min)
        & (g >= r + args.green_margin)
        & (g >= b + args.green_margin)
        & (value >= args.value_min)
        & ~strong_red
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


def weighted_centroid(points, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(points.shape[0], dtype=np.float64)
    return (points.astype(np.float64) * weights[:, None]).sum(axis=0) / weights.sum()


def build_other_fruit_tree(xyz, selected_all, cluster_id, cid):
    other = selected_all & (cluster_id != cid) & (cluster_id >= 0)
    if not other.any():
        return None
    return cKDTree(xyz[other].astype(np.float64))


def choose_contact_target(xyz, rgb, opacity_drop, fruit_mask, cluster_id, cid, green, red, args):
    fruit_idx = np.flatnonzero(fruit_mask)
    if fruit_idx.size == 0:
        return None, []

    fruit_tree = cKDTree(xyz[fruit_idx].astype(np.float64))
    candidate_base = green & ~red & ~fruit_mask
    candidate_idx = np.flatnonzero(candidate_base)
    if candidate_idx.size == 0:
        return None, []

    distances, _ = fruit_tree.query(xyz[candidate_idx].astype(np.float64), k=1)
    near = (distances >= args.min_contact_distance) & (distances <= args.search_radius)
    if not near.any():
        return None, []

    shell_idx = candidate_idx[near]
    shell_distances = distances[near]
    other_tree = build_other_fruit_tree(xyz, fruit_mask | (cluster_id >= 0), cluster_id, cid)
    if other_tree is not None:
        other_distances, _ = other_tree.query(xyz[shell_idx].astype(np.float64), k=1)
        own_ok = shell_distances <= other_distances + args.other_fruit_margin
        shell_idx = shell_idx[own_ok]
        shell_distances = shell_distances[own_ok]
    if shell_idx.size == 0:
        return None, []

    components = connected_components(xyz[shell_idx].astype(np.float64), args.component_radius)
    scored = []
    for component_id, component in enumerate(components):
        global_idx = shell_idx[component]
        if global_idx.size < args.min_component_size:
            continue
        points = xyz[global_idx].astype(np.float64)
        dist, _ = fruit_tree.query(points, k=1)
        if dist.min() > args.search_radius:
            continue

        order = np.argsort(dist)[: min(args.target_points, global_idx.size)]
        contact_idx = global_idx[order]
        contact_points = xyz[contact_idx].astype(np.float64)
        contact_dist = np.maximum(dist[order], 0.003)
        green_strength = rgb[contact_idx, 1] - np.maximum(rgb[contact_idx, 0], rgb[contact_idx, 2])
        weights = (1.0 / contact_dist) * (1.0 + np.maximum(green_strength, 0.0))
        if opacity_drop is not None:
            weights *= 1.0 + np.maximum(opacity_drop[contact_idx], 0.0)
        centroid = weighted_centroid(contact_points, weights)
        medoid = int(np.argmin(np.linalg.norm(contact_points - centroid[None, :], axis=1)))
        target_idx = int(contact_idx[medoid])
        target_point = xyz[target_idx].astype(np.float64)

        component_green_strength = float(
            np.mean(rgb[global_idx, 1] - np.maximum(rgb[global_idx, 0], rgb[global_idx, 2]))
        )
        score = (
            float(np.exp(-float(dist.min()) / 0.035))
            + 0.08 * float(np.log1p(global_idx.size))
            + 0.25 * max(component_green_strength, 0.0)
        )
        scored.append(
            {
                "component_id": int(component_id),
                "indices": global_idx,
                "target_idx": target_idx,
                "target_point": target_point,
                "score": score,
                "num_gaussians": int(global_idx.size),
                "bbox_diag": bbox_diag(points),
                "min_fruit_distance_m": float(dist.min()),
                "target_fruit_distance_m": float(fruit_tree.query(target_point[None, :], k=1)[0][0]),
                "green_strength": component_green_strength,
            }
        )

    if not scored:
        return None, []

    scored.sort(key=lambda item: (item["min_fruit_distance_m"], -item["num_gaussians"], -item["score"]))
    best = scored[0]
    confidence = float(np.clip(best["score"] / 1.6, 0.0, 1.0))
    target = {
        "point_xyz": best["target_point"].tolist(),
        "method": "tomato_fruit_shell_green_contact",
        "confidence": confidence,
        "score": float(best["score"]),
        "target_gaussian_id": int(best["target_idx"]),
        "target_is_selected_gaussian": False,
        "green_component_id": int(best["component_id"]),
        "green_component_size": int(best["num_gaussians"]),
        "green_component_bbox_diag": float(best["bbox_diag"]),
        "min_fruit_distance_m": float(best["min_fruit_distance_m"]),
        "target_fruit_distance_m": float(best["target_fruit_distance_m"]),
        "green_strength": float(best["green_strength"]),
    }
    diagnostics = [
        {key: value for key, value in item.items() if key not in ("indices", "target_point")}
        for item in scored[:8]
    ]
    return target, diagnostics


def write_ply(path, xyz, instance_id, target_xyz):
    points = []
    colors = []
    labels = []
    for iid in sorted(int(v) for v in np.unique(instance_id) if int(v) >= 0):
        idx = np.flatnonzero(instance_id == iid)
        if idx.size == 0:
            continue
        color = np.asarray(COLORS.get(iid % len(COLORS), (255, 255, 255)), dtype=np.uint8)
        points.append(xyz[idx])
        colors.append(np.tile(color, (idx.size, 1)))
        labels.append(np.full(idx.size, iid, dtype=np.int32))
    if target_xyz:
        target_points = np.asarray(target_xyz, dtype=np.float32)
        points.append(target_points)
        colors.append(np.tile(np.asarray((255, 255, 255), dtype=np.uint8), (target_points.shape[0], 1)))
        labels.append(np.full(target_points.shape[0], 100, dtype=np.int32))

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
    scores = np.load(args.score_npz)
    xyz = presence["xyz"].astype(np.float32)
    rgb = scores["rgb"].astype(np.float32)
    opacity = presence["opacity_pre"].reshape(-1).astype(np.float32)
    opacity_drop = presence["opacity_drop"].reshape(-1).astype(np.float32) if "opacity_drop" in presence.files else None
    selected_all = presence[args.mask_key].astype(bool)
    cluster_id = presence[args.cluster_key].astype(np.int64)

    clusters = parse_cluster_ids(args.selected_clusters, cluster_id, selected_all)
    red = red_like(rgb, opacity, args)
    green = green_like(rgb, opacity, args)

    tomato_instance_id = np.full(xyz.shape[0], -1, dtype=np.int32)
    results = []
    target_xyz = []
    target_confidence = []
    target_instance_ids = []

    for instance_id, cid in enumerate(clusters):
        fruit_mask = selected_all & (cluster_id == cid)
        if not fruit_mask.any():
            continue
        target, diagnostics = choose_contact_target(
            xyz,
            rgb,
            opacity_drop,
            fruit_mask,
            cluster_id,
            cid,
            green,
            red,
            args,
        )
        tomato_instance_id[fruit_mask] = instance_id
        fruit_points = xyz[fruit_mask]
        result = {
            "cluster_id": int(instance_id),
            "source_cluster_id": int(cid),
            "num_gaussians": int(fruit_mask.sum()),
            "bbox_min": fruit_points.min(axis=0).tolist(),
            "bbox_max": fruit_points.max(axis=0).tolist(),
            "bbox_diag": bbox_diag(fruit_points),
            "selected_red_gaussians": int((fruit_mask & red).sum()),
            "candidate_components": diagnostics,
            "target": target,
            "target_xyz": None if target is None else target["point_xyz"],
            "target_confidence": None if target is None else float(target["confidence"]),
        }
        results.append(result)
        if target is not None:
            target_xyz.append(target["point_xyz"])
            target_confidence.append(float(target["confidence"]))
            target_instance_ids.append(instance_id)

    payload = {key: presence[key] for key in presence.files}
    payload["tomato_instance_candidate"] = tomato_instance_id >= 0
    payload["tomato_instance_id"] = tomato_instance_id
    payload["interaction_target_instance_id"] = np.asarray(target_instance_ids, dtype=np.int32)
    payload["interaction_target_xyz"] = (
        np.asarray(target_xyz, dtype=np.float32) if target_xyz else np.zeros((0, 3), dtype=np.float32)
    )
    payload["interaction_target_confidence"] = (
        np.asarray(target_confidence, dtype=np.float32) if target_confidence else np.zeros((0,), dtype=np.float32)
    )
    payload["interaction_target_method"] = np.asarray("tomato_fruit_shell_green_contact")
    np.savez_compressed(out_dir / "tomato_component_targets.npz", **payload)

    summary = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "method": "tomato_fruit_shell_green_contact",
        "method_note": (
            "The removed red fruit mask is fixed. For each tomato instance, the target "
            "is chosen from nearby pre-scene green/calyx Gaussians as the contact medoid "
            "closest to the fruit surface."
        ),
        "selected_source_clusters": clusters,
        "num_instances": int(len(results)),
        "num_targets": int(len(target_xyz)),
        "search_radius_m": float(args.search_radius),
        "component_radius_m": float(args.component_radius),
        "clusters": results,
    }
    (out_dir / "tomato_component_targets_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_ply(out_dir / "tomato_component_targets.ply", xyz, tomato_instance_id, target_xyz)

    print(f"Saved {out_dir / 'tomato_component_targets.npz'}")
    print(f"Saved {out_dir / 'tomato_component_targets.ply'}")
    print(f"Saved {out_dir / 'tomato_component_targets_summary.json'}")
    print(json.dumps({"instances": len(results), "targets": len(target_xyz)}, indent=2))


if __name__ == "__main__":
    main()
