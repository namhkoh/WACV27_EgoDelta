#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
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
        description="Trace retained-plant connection paths from fixed removed leaf masks."
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--selected_clusters", default="0,1,2,3")
    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--green_min", default=0.08, type=float)
    parser.add_argument("--green_margin", default=-0.05, type=float)
    parser.add_argument("--value_min", default=0.06, type=float)
    parser.add_argument("--red_ratio_max", default=1.25, type=float)
    parser.add_argument("--stable_opacity_drop_max", default=0.02, type=float)
    parser.add_argument("--stable_presence_drop_max", default=0.20, type=float)
    parser.add_argument("--retained_search_radius", default=0.075, type=float)
    parser.add_argument("--retained_min_along", default=0.015, type=float)
    parser.add_argument("--retained_max_along", default=0.30, type=float)
    parser.add_argument("--retained_quantile", default=0.30, type=float)
    parser.add_argument("--tube_radius", default=0.075, type=float)
    parser.add_argument("--path_edge_radius", default=0.028, type=float)
    parser.add_argument("--path_score_min", default=0.0, type=float)
    parser.add_argument("--max_path_nodes", default=25000, type=int)
    parser.add_argument("--tight_component_radius", default=0.018, type=float)
    return parser.parse_args()


def parse_cluster_ids(spec, cluster_id, selected):
    spec = str(spec).strip().lower()
    valid = sorted(int(v) for v in np.unique(cluster_id[selected]) if int(v) >= 0)
    if spec in ("all", "-1", ""):
        return valid
    requested = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    missing = sorted(set(requested) - set(valid))
    if missing:
        raise RuntimeError(f"Requested clusters {missing} are absent from optimized candidates {valid}")
    return requested


def green_like(rgb, opacity, plant_roi, score, args, score_min=None):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    value = np.max(rgb, axis=1)
    strong_red = (r > g * args.red_ratio_max) & (r > b * args.red_ratio_max)
    mask = (
        (opacity >= args.opacity_min)
        & plant_roi
        & (g >= args.green_min)
        & (g >= r + args.green_margin)
        & (g >= b + args.green_margin)
        & (value >= args.value_min)
        & ~strong_red
    )
    if score_min is not None:
        mask &= score >= float(score_min)
    return mask


def weighted_centroid(points, weights):
    if points.shape[0] == 0:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0.0:
        weights = np.ones(points.shape[0], dtype=np.float64)
    return (points.astype(np.float64) * weights[:, None]).sum(axis=0) / weights.sum()


def pca_axis(points, weights):
    center = weighted_centroid(points, weights)
    centered = points.astype(np.float64) - center[None, :]
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0.0:
        weights = np.ones(points.shape[0], dtype=np.float64)
    cov = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-8)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    return center, axis, np.maximum(vals[::-1], 0.0)


def bbox_diag(points):
    if points.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


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
            for nb in neighbors[current]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(key=lambda item: item.size, reverse=True)
    return components


def choose_endpoint(xyz, seed, weights_all, retained, args):
    seed_idx = np.flatnonzero(seed)
    if seed_idx.size == 0:
        return None
    weights = np.maximum(weights_all[seed_idx], 0.0)
    center, axis, eigvals = pca_axis(xyz[seed_idx], weights)
    projection = (xyz[seed_idx].astype(np.float64) - center[None, :]) @ axis
    choices = []
    for side, endpoint_mask in (
        ("low", projection <= np.quantile(projection, 0.05)),
        ("high", projection >= np.quantile(projection, 0.95)),
    ):
        endpoint = xyz[seed_idx[endpoint_mask]].mean(axis=0).astype(np.float64)
        direction = endpoint - center
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        rel = xyz.astype(np.float64) - endpoint[None, :]
        along = rel @ direction
        perp = np.linalg.norm(rel - along[:, None] * direction[None, :], axis=1)
        retained_cone = (
            retained
            & (along > args.retained_min_along)
            & (along < args.retained_max_along)
            & (perp <= args.retained_search_radius)
        )
        retained_count = int(retained_cone.sum())
        median_along = float(np.median(along[retained_cone])) if retained_cone.any() else 0.0
        score = retained_count * max(median_along, 1e-4)
        choices.append(
            {
                "side": side,
                "endpoint": endpoint,
                "direction": direction,
                "along": along,
                "perp": perp,
                "retained_cone": retained_cone,
                "retained_count": retained_count,
                "median_along": median_along,
                "continuation_score": float(score),
                "pca_eigvals": eigvals,
            }
        )
    choices.sort(key=lambda item: item["continuation_score"], reverse=True)
    return choices[0]


def shortest_path(xyz, node_mask, start_mask, target_mask, edge_radius, max_nodes):
    graph_indices = np.flatnonzero(node_mask)
    if graph_indices.size == 0:
        return np.asarray([], dtype=np.int64), None
    if graph_indices.size > max_nodes:
        seed_points = xyz[np.flatnonzero(start_mask)]
        seed_tree = cKDTree(seed_points)
        dist_to_seed, _ = seed_tree.query(xyz[graph_indices], k=1)
        keep = np.argsort(dist_to_seed)[:max_nodes]
        graph_indices = graph_indices[keep]

    local_of_global = {int(global_idx): local_idx for local_idx, global_idx in enumerate(graph_indices)}
    start_local = np.asarray([local_of_global[int(i)] for i in np.flatnonzero(start_mask) if int(i) in local_of_global], dtype=np.int64)
    target_local_set = {local_of_global[int(i)] for i in np.flatnonzero(target_mask) if int(i) in local_of_global}
    if start_local.size == 0 or not target_local_set:
        return np.asarray([], dtype=np.int64), None

    graph_xyz = xyz[graph_indices].astype(np.float32)
    tree = cKDTree(graph_xyz)
    dist = np.full(graph_indices.shape[0], np.inf, dtype=np.float32)
    prev = np.full(graph_indices.shape[0], -1, dtype=np.int64)
    heap = []
    for local_idx in start_local:
        dist[local_idx] = 0.0
        heapq.heappush(heap, (0.0, int(local_idx)))

    reached = None
    while heap:
        current_dist, local_idx = heapq.heappop(heap)
        if current_dist != float(dist[local_idx]):
            continue
        if local_idx in target_local_set:
            reached = local_idx
            break
        for neighbor_idx in tree.query_ball_point(graph_xyz[local_idx], edge_radius):
            if neighbor_idx == local_idx:
                continue
            edge = float(np.linalg.norm(graph_xyz[neighbor_idx] - graph_xyz[local_idx]))
            new_dist = current_dist + edge
            if new_dist < float(dist[neighbor_idx]):
                dist[neighbor_idx] = new_dist
                prev[neighbor_idx] = local_idx
                heapq.heappush(heap, (new_dist, int(neighbor_idx)))

    if reached is None:
        return np.asarray([], dtype=np.int64), None

    path = []
    current = reached
    while current >= 0:
        path.append(current)
        current = int(prev[current])
    path = np.asarray(path[::-1], dtype=np.int64)
    return graph_indices[path], {
        "path_length_m": float(dist[reached]),
        "graph_nodes": int(graph_indices.size),
        "start_nodes": int(start_local.size),
        "target_nodes": int(len(target_local_set)),
    }


def write_ply(path, xyz, base_masks, path_masks, target_points):
    points = []
    colors = []
    labels = []
    for cluster, base in base_masks.items():
        path_mask = path_masks.get(cluster, np.zeros(xyz.shape[0], dtype=bool))
        color = np.asarray(COLORS.get(cluster, (255, 255, 255)), dtype=np.uint8)
        for mask, label, scale in ((base, cluster, 1.0), (path_mask & ~base, cluster + 10, 0.55)):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            points.append(xyz[idx])
            colors.append(np.tile(np.clip(color.astype(np.float32) * scale, 0, 255).astype(np.uint8), (idx.size, 1)))
            labels.append(np.full(idx.size, label, dtype=np.int32))
    if target_points:
        target_xyz = np.asarray([item["point"] for item in target_points], dtype=np.float32)
        target_color = np.asarray([[255, 255, 255]], dtype=np.uint8)
        points.append(target_xyz)
        colors.append(np.tile(target_color, (target_xyz.shape[0], 1)))
        labels.append(np.asarray([100 + int(item["cluster_id"]) for item in target_points], dtype=np.int32))

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
    xyz = presence["xyz"].astype(np.float32)
    optimized = presence["optimized_candidate"].astype(bool)
    cluster_id = presence["post_opt_cluster_id"].astype(np.int64)
    clusters = parse_cluster_ids(args.selected_clusters, cluster_id, optimized)

    opacity = presence["opacity_pre"].astype(np.float32)
    opacity_drop = presence["opacity_drop"].astype(np.float32)
    presence_drop = presence["presence_drop"].astype(np.float32)
    plant_roi = score_npz["plant_structure_roi"].astype(bool)
    residual_score = score_npz["score"].astype(np.float32)
    rgb = score_npz["rgb"].astype(np.float32)

    plant_green = green_like(rgb, opacity, plant_roi, residual_score, args, score_min=None)
    path_green = green_like(rgb, opacity, plant_roi, residual_score, args, score_min=args.path_score_min)
    stable_retained = (
        plant_green
        & ~optimized
        & ((opacity_drop < args.stable_opacity_drop_max) | (presence_drop < args.stable_presence_drop_max))
    )

    connection_mask = np.zeros(xyz.shape[0], dtype=bool)
    connection_cluster_id = np.full(xyz.shape[0], -1, dtype=np.int32)
    target_points = []
    base_masks = {}
    path_masks = {}
    diagnostics = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "selected_clusters": clusters,
        "note": "removed_core_mask is unchanged; connection_path is for target/attachment reasoning only.",
        "clusters": [],
    }

    for cluster in clusters:
        base = optimized & (cluster_id == cluster)
        base_masks[cluster] = base
        components = connected_components(xyz[base], args.tight_component_radius)
        component_summary = []
        base_indices = np.flatnonzero(base)
        for component in components[:8]:
            global_idx = base_indices[component]
            pts = xyz[global_idx]
            component_summary.append(
                {
                    "num_gaussians": int(component.size),
                    "bbox_min": pts.min(axis=0).tolist(),
                    "bbox_max": pts.max(axis=0).tolist(),
                    "bbox_diag": bbox_diag(pts),
                    "median_residual_score": float(np.median(residual_score[global_idx])),
                    "mean_opacity_drop": float(opacity_drop[global_idx].mean()),
                }
            )

        endpoint = choose_endpoint(xyz, base, opacity_drop, stable_retained, args)
        if endpoint is None or not endpoint["retained_cone"].any():
            path_idx = np.asarray([], dtype=np.int64)
            path_meta = None
            target_idx = None
        else:
            cutoff = float(np.quantile(endpoint["along"][endpoint["retained_cone"]], args.retained_quantile))
            start = base & (endpoint["perp"] <= args.tube_radius) & (endpoint["along"] >= -0.03) & (endpoint["along"] <= 0.03)
            target = endpoint["retained_cone"] & (endpoint["along"] <= cutoff)
            scope = (
                path_green
                & (endpoint["along"] >= -0.04)
                & (endpoint["along"] <= cutoff + 0.04)
                & (endpoint["perp"] <= args.tube_radius)
            )
            scope |= start | target
            path_idx, path_meta = shortest_path(
                xyz,
                scope,
                start,
                target,
                args.path_edge_radius,
                args.max_path_nodes,
            )
            target_idx = int(path_idx[-1]) if path_idx.size else None

        path_mask = np.zeros(xyz.shape[0], dtype=bool)
        path_mask[path_idx] = True
        path_masks[cluster] = path_mask
        connection_mask |= path_mask
        connection_cluster_id[path_mask] = int(cluster)
        if target_idx is not None:
            target_points.append({"cluster_id": int(cluster), "point": xyz[target_idx].tolist()})

        diagnostics["clusters"].append(
            {
                "cluster_id": int(cluster),
                "base_gaussians": int(base.sum()),
                "base_bbox_diag": bbox_diag(xyz[base]),
                "tight_component_radius": float(args.tight_component_radius),
                "tight_components": component_summary,
                "endpoint_side": endpoint["side"] if endpoint is not None else None,
                "endpoint_xyz": endpoint["endpoint"].tolist() if endpoint is not None else None,
                "retained_cone_gaussians": int(endpoint["retained_count"]) if endpoint is not None else 0,
                "retained_cone_median_along": float(endpoint["median_along"]) if endpoint is not None else 0.0,
                "path_gaussians": int(path_mask.sum()),
                "path_new_gaussians": int((path_mask & ~base).sum()),
                "path_meta": path_meta,
                "target_xyz": xyz[target_idx].tolist() if target_idx is not None else None,
            }
        )

    payload = {key: presence[key] for key in presence.files}
    payload["connection_path_candidate"] = connection_mask
    payload["connection_cluster_id"] = connection_cluster_id
    # These fields are for existing overlay tooling only; they are not a completed removal mask.
    payload["completed_candidate"] = optimized | connection_mask
    payload["completed_cluster_id"] = np.where(connection_cluster_id >= 0, connection_cluster_id, cluster_id).astype(np.int32)
    payload["connection_added_candidate"] = connection_mask & ~optimized
    np.savez_compressed(out_dir / "leaf_connection_paths.npz", **payload)
    write_ply(out_dir / "leaf_connection_paths.ply", xyz, base_masks, path_masks, target_points)
    (out_dir / "leaf_connection_paths_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"Saved {out_dir / 'leaf_connection_paths.npz'}")
    print(f"Saved {out_dir / 'leaf_connection_paths.ply'}")
    print(f"Saved {out_dir / 'leaf_connection_paths_summary.json'}")
    print(json.dumps({"connection_gaussians": int(connection_mask.sum())}, indent=2))


if __name__ == "__main__":
    main()
