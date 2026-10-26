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
        description=(
            "Infer leaf interaction targets from a fixed removed Gaussian mask by "
            "snapping a mask endpoint ray to retained green plant geometry."
        )
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--selected_clusters", default="0,1,2,3")
    parser.add_argument(
        "--connection_summary_json",
        default=None,
        help="Seed summary from extract_leaf_connection_paths.py.",
    )
    parser.add_argument("--current_summary_json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manual_votes_json", default=None, help="Optional label-derived proxy for evaluation only.")

    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--green_min", default=0.08, type=float)
    parser.add_argument("--green_margin", default=-0.05, type=float)
    parser.add_argument("--value_min", default=0.06, type=float)
    parser.add_argument("--red_ratio_max", default=1.25, type=float)
    parser.add_argument("--stable_opacity_drop_max", default=0.02, type=float)
    parser.add_argument("--stable_presence_drop_max", default=0.20, type=float)
    parser.add_argument("--stable_visibility_min", default=4, type=int)

    parser.add_argument(
        "--target_method",
        choices=("backbone_gap_or_rise", "ray_prior"),
        default="backbone_gap_or_rise",
    )
    parser.add_argument("--target_along", default=0.10, type=float)
    parser.add_argument("--target_sigma", default=0.035, type=float)
    parser.add_argument("--ray_min_along", default=0.015, type=float)
    parser.add_argument("--ray_max_along", default=0.42, type=float)
    parser.add_argument("--tube_radius", default=0.090, type=float)
    parser.add_argument("--bin_width", default=0.015, type=float)
    parser.add_argument("--min_bin_points", default=8, type=int)
    parser.add_argument("--density_radius", default=0.030, type=float)
    parser.add_argument("--component_edge_radius", default=0.026, type=float)
    parser.add_argument("--density_peak_ratio", default=0.70, type=float)
    parser.add_argument("--entry_density_ratio", default=0.45, type=float)
    parser.add_argument("--max_retained_ply_points", default=1800, type=int)
    return parser.parse_args()


def parse_cluster_ids(spec, cluster_id, selected):
    valid = sorted(int(v) for v in np.unique(cluster_id[selected]) if int(v) >= 0)
    spec = str(spec).strip().lower()
    if spec in ("all", "-1", ""):
        return valid
    requested = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    missing = sorted(set(requested) - set(valid))
    if missing:
        raise RuntimeError(f"Requested clusters {missing} are absent from optimized candidates {valid}")
    return requested


def green_like(rgb, opacity, plant_roi, args):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    value = np.max(rgb, axis=1)
    strong_red = (r > g * args.red_ratio_max) & (r > b * args.red_ratio_max)
    return (
        (opacity >= args.opacity_min)
        & plant_roi
        & (g >= args.green_min)
        & (g >= r + args.green_margin)
        & (g >= b + args.green_margin)
        & (value >= args.value_min)
        & ~strong_red
    )


def weighted_centroid(points, weights=None):
    if points.shape[0] == 0:
        return None
    if weights is None:
        weights = np.ones(points.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        weights = np.ones(points.shape[0], dtype=np.float64)
    return (points.astype(np.float64) * weights[:, None]).sum(axis=0) / weights.sum()


def bbox_diag(points):
    if points.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def load_connection_seeds(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    seeds = {}
    for item in data.get("clusters", []):
        endpoint = item.get("endpoint_xyz")
        target = item.get("target_xyz")
        if endpoint is None or target is None:
            continue
        endpoint = np.asarray(endpoint, dtype=np.float64)
        direction = np.asarray(target, dtype=np.float64) - endpoint
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            continue
        seeds[int(item["cluster_id"])] = {
            "endpoint": endpoint,
            "direction": direction / norm,
            "source": "connection_summary",
            "seed_target_xyz": np.asarray(target, dtype=np.float64),
        }
    return seeds


def load_manual_proxy(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    proxies = {}
    for leaf, item in data.items():
        rows = item.get("top_gaussians", [])
        if rows:
            proxies[leaf] = np.asarray(rows[0]["xyz"], dtype=np.float64)
    return proxies


def ray_scope(xyz, stable_retained, endpoint, direction, args):
    rel = xyz.astype(np.float64) - endpoint[None, :]
    along = rel @ direction
    perp = np.linalg.norm(rel - along[:, None] * direction[None, :], axis=1)
    retained = (
        stable_retained
        & (along >= args.ray_min_along)
        & (along <= args.ray_max_along)
        & (perp <= args.tube_radius)
    )
    return along, perp, retained


def make_centerline_nodes(xyz, retained, endpoint, direction, along, perp, args):
    retained_idx = np.flatnonzero(retained)
    if retained_idx.size == 0:
        return []
    retained_points = xyz[retained_idx].astype(np.float64)
    retained_along = along[retained_idx]
    retained_perp = perp[retained_idx]
    bins = np.floor(retained_along / args.bin_width).astype(np.int32)
    tree = cKDTree(retained_points)

    nodes = []
    for bin_id in sorted(set(int(v) for v in bins)):
        local = np.flatnonzero(bins == bin_id)
        if local.size < args.min_bin_points:
            continue
        weights = 1.0 / np.maximum(retained_perp[local], 0.003)
        center = weighted_centroid(retained_points[local], weights)
        rel = center - endpoint
        node_along = float(rel @ direction)
        node_perp = float(np.linalg.norm(rel - node_along * direction))
        density = int(len(tree.query_ball_point(center, args.density_radius)))
        nodes.append(
            {
                "point": center,
                "along": node_along,
                "perp": node_perp,
                "bin_id": int(bin_id),
                "bin_points": int(local.size),
                "density": density,
            }
        )
    return nodes


def connected_components(points, radius):
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, radius)
    labels = np.full(points.shape[0], -1, dtype=np.int32)
    degree = np.asarray([max(0, len(item) - 1) for item in neighbors], dtype=np.int32)
    sizes = []
    component_id = 0
    for seed in range(points.shape[0]):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = component_id
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            for nb in neighbors[current]:
                if labels[nb] < 0:
                    labels[nb] = component_id
                    stack.append(int(nb))
        sizes.append(len(members))
        component_id += 1
    return labels, np.asarray(sizes, dtype=np.int32), degree


def make_backbone_nodes(xyz, retained, endpoint, direction, along, perp, visibility, args):
    retained_idx = np.flatnonzero(retained)
    if retained_idx.size == 0:
        return []
    retained_points = xyz[retained_idx].astype(np.float64)
    retained_along = along[retained_idx]
    retained_perp = perp[retained_idx]
    retained_visibility = visibility[retained_idx]

    comp_label, comp_sizes, degree = connected_components(retained_points, args.component_edge_radius)
    bins = np.floor(retained_along / args.bin_width).astype(np.int32)
    tree = cKDTree(retained_points)
    max_comp_size = int(comp_sizes.max()) if comp_sizes.size else 1
    max_degree = int(degree.max()) if degree.size else 1
    max_visibility = float(max(retained_visibility.max(), 1)) if retained_visibility.size else 1.0

    nodes = []
    for bin_id in sorted(set(int(v) for v in bins)):
        local = np.flatnonzero(bins == bin_id)
        if local.size < args.min_bin_points:
            continue
        weights = 1.0 / np.maximum(retained_perp[local], 0.003)
        center = weighted_centroid(retained_points[local], weights)
        rel = center - endpoint
        node_along = float(rel @ direction)
        node_perp = float(np.linalg.norm(rel - node_along * direction))
        density = int(len(tree.query_ball_point(center, args.density_radius)))
        labels, counts = np.unique(comp_label[local], return_counts=True)
        dominant_label = int(labels[np.argmax(counts)])
        dominant_size = int(comp_sizes[dominant_label]) if dominant_label >= 0 else 0
        nodes.append(
            {
                "point": center,
                "along": node_along,
                "perp": node_perp,
                "bin_id": int(bin_id),
                "bin_points": int(local.size),
                "density": density,
                "dominant_component": dominant_label,
                "dominant_component_size": dominant_size,
                "component_norm": np.log1p(dominant_size) / max(np.log1p(max_comp_size), 1e-6),
                "degree_norm": float(np.median(degree[local])) / max(float(max_degree), 1e-6),
                "visibility_norm": float(np.median(retained_visibility[local])) / max_visibility,
                "radial_p75": float(np.quantile(retained_perp[local], 0.75)),
                "density_norm": 0.0,
            }
        )

    if not nodes:
        return []
    max_density = max(node["density"] for node in nodes)
    max_bin_points = max(node["bin_points"] for node in nodes)
    max_radial = max(node["radial_p75"] for node in nodes)
    for node in nodes:
        node["density_norm"] = np.log1p(node["density"]) / max(np.log1p(max_density), 1e-6)
        node["count_norm"] = np.log1p(node["bin_points"]) / max(np.log1p(max_bin_points), 1e-6)
        node["radial_norm"] = node["radial_p75"] / max(max_radial, 1e-6)
        node["ray_score"] = float(np.exp(-((node["perp"] / max(args.tube_radius, 1e-6)) ** 2)))
        node["backbone_score"] = float(
            0.32 * node["component_norm"]
            + 0.24 * node["density_norm"]
            + 0.16 * node["count_norm"]
            + 0.12 * node["degree_norm"]
            + 0.10 * node["visibility_norm"]
            + 0.06 * node["radial_norm"]
        )
    return nodes


def fallback_node_from_retained(xyz, retained, endpoint, direction, along, perp, args):
    retained_idx = np.flatnonzero(retained)
    if retained_idx.size == 0:
        return None
    delta = np.abs(along[retained_idx] - args.target_along)
    window = max(args.target_sigma, args.bin_width * 2.0)
    local = retained_idx[delta <= window]
    if local.size == 0:
        order = np.argsort(delta)[: min(32, retained_idx.size)]
        local = retained_idx[order]
    weights = 1.0 / np.maximum(perp[local], 0.003)
    point = weighted_centroid(xyz[local], weights)
    rel = point - endpoint
    node_along = float(rel @ direction)
    node_perp = float(np.linalg.norm(rel - node_along * direction))
    return {
        "point": point,
        "along": node_along,
        "perp": node_perp,
        "bin_id": None,
        "bin_points": int(local.size),
        "density": int(local.size),
        "fallback": True,
    }


def choose_ray_target(nodes, xyz, retained, endpoint, direction, along, perp, args):
    if not nodes:
        fallback = fallback_node_from_retained(xyz, retained, endpoint, direction, along, perp, args)
        nodes = [fallback] if fallback is not None else []
    if not nodes:
        return None

    max_density = max(node["density"] for node in nodes)
    max_bin_points = max(node["bin_points"] for node in nodes)
    scored = []
    for node in nodes:
        perp_norm = node["perp"] / max(args.tube_radius, 1e-6)
        density_norm = np.log1p(node["density"]) / max(np.log1p(max_density), 1e-6)
        count_norm = np.log1p(node["bin_points"]) / max(np.log1p(max_bin_points), 1e-6)
        ray_score = float(np.exp(-(perp_norm**2)))
        prior_score = float(np.exp(-((node["along"] - args.target_along) / max(args.target_sigma, 1e-6)) ** 2))
        score = 1.25 * prior_score + 0.55 * density_norm + 0.35 * count_norm + 0.45 * ray_score
        item = dict(node)
        item.update(
            {
                "score": float(score),
                "density_norm": float(density_norm),
                "count_norm": float(count_norm),
                "ray_score": float(ray_score),
                "prior_score": float(prior_score),
            }
        )
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    return {
        "point_xyz": np.asarray(best["point"], dtype=np.float64).tolist(),
        "method": f"ray_prior_{args.target_along:.2f}m",
        "score": float(best["score"]),
        "confidence": float(np.clip(best["score"] / 2.6, 0.0, 1.0)),
        "along_m": float(best["along"]),
        "perp_m": float(best["perp"]),
        "density": int(best["density"]),
        "bin_points": int(best["bin_points"]),
        "fallback_node": bool(best.get("fallback", False)),
    }


def target_from_backbone_node(node, method, **extra):
    target = {
        "point_xyz": np.asarray(node["point"], dtype=np.float64).tolist(),
        "method": method,
        "score": float(node.get("backbone_score", 0.0)),
        "confidence": float(np.clip(node.get("backbone_score", 0.0), 0.0, 1.0)),
        "along_m": float(node["along"]),
        "perp_m": float(node["perp"]),
        "density": int(node["density"]),
        "bin_points": int(node["bin_points"]),
        "dominant_component_size": int(node["dominant_component_size"]),
    }
    target.update(extra)
    return target


def largest_component_label(nodes):
    sizes = {}
    for node in nodes:
        sizes[node["dominant_component"]] = max(
            sizes.get(node["dominant_component"], 0),
            node["dominant_component_size"],
        )
    return max(sizes, key=sizes.get) if sizes else None


def choose_backbone_gap_or_rise_target(nodes, args):
    if not nodes:
        return None
    label = largest_component_label(nodes)
    scoped = sorted([node for node in nodes if node["dominant_component"] == label], key=lambda item: item["along"])
    if not scoped:
        return None

    if len(scoped) >= 3:
        gaps = np.asarray(
            [scoped[i + 1]["along"] - scoped[i]["along"] for i in range(len(scoped) - 1)],
            dtype=np.float64,
        )
        median_gap = float(np.median(gaps)) if gaps.size else 0.0
        for gap_idx in np.argsort(gaps)[::-1]:
            if gaps[gap_idx] >= 2.5 * max(median_gap, args.bin_width):
                return target_from_backbone_node(
                    scoped[int(gap_idx) + 1],
                    "backbone_gap_or_rise",
                    decision="after_largest_component_gap",
                    gap_m=float(gaps[gap_idx]),
                )

    max_density = max(node["density"] for node in scoped)
    entry = scoped[0]
    if entry["density"] >= args.entry_density_ratio * max_density:
        return target_from_backbone_node(
            entry,
            "backbone_gap_or_rise",
            decision="component_entry_dense_enough",
            entry_density_ratio=float(entry["density"] / max(max_density, 1)),
        )

    candidates = [node for node in scoped if node["density"] >= args.density_peak_ratio * max_density] or scoped
    candidates.sort(key=lambda item: (item["along"], -item["density"]))
    return target_from_backbone_node(
        candidates[0],
        "backbone_gap_or_rise",
        decision="first_density_rise",
        density_threshold=float(args.density_peak_ratio * max_density),
    )


def process_cluster(xyz, cluster_id, optimized, stable_retained, visibility, connection_seeds, cid, args):
    base = optimized & (cluster_id == cid)
    if not base.any():
        return None, np.zeros(xyz.shape[0], dtype=bool)

    seed = connection_seeds.get(cid)
    if seed is None:
        return None, np.zeros(xyz.shape[0], dtype=bool)

    endpoint = np.asarray(seed["endpoint"], dtype=np.float64)
    direction = np.asarray(seed["direction"], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    along, perp, retained = ray_scope(xyz, stable_retained, endpoint, direction, args)
    ray_nodes = make_centerline_nodes(xyz, retained, endpoint, direction, along, perp, args)
    backbone_nodes = make_backbone_nodes(xyz, retained, endpoint, direction, along, perp, visibility, args)
    if args.target_method == "ray_prior":
        target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
    else:
        target = choose_backbone_gap_or_rise_target(backbone_nodes, args)
        if target is None:
            target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
            if target is not None:
                target["fallback_from"] = args.target_method
    if target is None:
        return None, retained

    return (
        {
            "cluster_id": int(cid),
            "base_gaussians": int(base.sum()),
            "base_bbox_diag": bbox_diag(xyz[base]),
            "endpoint_xyz": endpoint.tolist(),
            "direction": direction.tolist(),
            "seed_source": seed.get("source", "unknown"),
            "retained_candidates": int(retained.sum()),
            "centerline_nodes": int(len(ray_nodes)),
            "backbone_nodes": int(len(backbone_nodes)),
            "target_xyz": target["point_xyz"],
            "target_confidence": target["confidence"],
            "target": target,
        },
        retained,
    )


def add_manual_distance(result, manual_point):
    if result is None or manual_point is None:
        return None
    target = np.asarray(result["target_xyz"], dtype=np.float64)
    distance = float(np.linalg.norm(target - manual_point))
    result["manual_proxy_xyz"] = manual_point.tolist()
    result["target_distance_to_manual_proxy_m"] = distance
    result["target"]["distance_to_manual_proxy_m"] = distance
    return distance


def write_ply(path, xyz, cluster_id, optimized, retained_masks, results, max_retained_points):
    points = []
    colors = []
    labels = []
    for cid in sorted(results):
        base_idx = np.flatnonzero(optimized & (cluster_id == cid))
        if base_idx.size:
            color = np.asarray(COLORS.get(cid, (255, 255, 255)), dtype=np.uint8)
            points.append(xyz[base_idx])
            colors.append(np.tile(color, (base_idx.size, 1)))
            labels.append(np.full(base_idx.size, cid, dtype=np.int32))

        retained = retained_masks.get(cid)
        if retained is not None:
            retained_idx = np.flatnonzero(retained)
            if retained_idx.size > max_retained_points:
                rng = np.random.default_rng(cid)
                retained_idx = rng.choice(retained_idx, max_retained_points, replace=False)
            if retained_idx.size:
                points.append(xyz[retained_idx])
                colors.append(np.tile(np.asarray((70, 210, 95), dtype=np.uint8), (retained_idx.size, 1)))
                labels.append(np.full(retained_idx.size, cid + 20, dtype=np.int32))

    target_points = []
    for cid, result in sorted(results.items()):
        target_points.append(np.asarray(result["target_xyz"], dtype=np.float32))
    if target_points:
        target_xyz = np.asarray(target_points, dtype=np.float32)
        points.append(target_xyz)
        colors.append(np.tile(np.asarray((255, 255, 255), dtype=np.uint8), (target_xyz.shape[0], 1)))
        labels.append(np.asarray([100 + int(cid) for cid in sorted(results)], dtype=np.int32))

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

    connection_summary_json = args.connection_summary_json or args.current_summary_json
    if connection_summary_json is None:
        raise RuntimeError("--connection_summary_json is required for leaf ray target extraction.")
    presence = np.load(args.presence_npz)
    score_npz = np.load(args.score_npz)
    xyz = presence["xyz"].astype(np.float32)
    optimized = presence["optimized_candidate"].astype(bool)
    cluster_id = presence["post_opt_cluster_id"].astype(np.int64)
    clusters = parse_cluster_ids(args.selected_clusters, cluster_id, optimized)

    opacity = presence["opacity_pre"].reshape(-1).astype(np.float32)
    opacity_drop = presence["opacity_drop"].reshape(-1).astype(np.float32)
    presence_drop = presence["presence_drop"].reshape(-1).astype(np.float32)
    plant_roi = score_npz["plant_structure_roi"].astype(bool)
    rgb = score_npz["rgb"].astype(np.float32)
    visibility = (
        score_npz["visibility_count"].astype(np.float32)
        if "visibility_count" in score_npz.files
        else np.zeros(xyz.shape[0], dtype=np.float32)
    )

    plant_green = green_like(rgb, opacity, plant_roi, args)
    stable_retained = (
        plant_green
        & ~optimized
        & (visibility >= args.stable_visibility_min)
        & ((opacity_drop < args.stable_opacity_drop_max) | (presence_drop < args.stable_presence_drop_max))
    )
    connection_seeds = load_connection_seeds(connection_summary_json)
    manual_proxy = load_manual_proxy(args.manual_votes_json)
    missing_connection_seeds = [int(cid) for cid in clusters if cid not in connection_seeds]

    retained_masks = {}
    results = {}
    distances = []
    diagnostics = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "connection_summary_json": connection_summary_json,
        "manual_votes_json": args.manual_votes_json,
        "selected_clusters": clusters,
        "method": "leaf_interaction_target",
        "method_note": (
            "The removed Gaussian mask is fixed. The interaction target is selected from retained "
            "green plant Gaussians near the mask endpoint ray. The default method builds a local "
            "retained-plant backbone profile and picks the first stable connection evidence from "
            "the dominant component; the fixed 0.10 m along-ray prior is kept only as an optional "
            "fallback mode."
        ),
        "target_method": args.target_method,
        "target_along_m": float(args.target_along),
        "target_sigma_m": float(args.target_sigma),
        "tube_radius_m": float(args.tube_radius),
        "stable_visibility_min": int(args.stable_visibility_min),
        "missing_connection_seed_clusters": missing_connection_seeds,
        "clusters": [],
    }

    for cid in clusters:
        if cid not in connection_seeds:
            diagnostics["clusters"].append(
                {
                    "cluster_id": int(cid),
                    "failure_reason": "missing_connection_seed",
                    "target_xyz": None,
                    "target_confidence": 0.0,
                }
            )
            continue
        result, retained = process_cluster(
            xyz,
            cluster_id,
            optimized,
            stable_retained,
            visibility,
            connection_seeds,
            cid,
            args,
        )
        retained_masks[cid] = retained
        if result is None:
            diagnostics["clusters"].append({"cluster_id": int(cid), "failure_reason": "no_ray_target"})
            continue
        distance = add_manual_distance(result, manual_proxy.get(f"leaf{cid}"))
        if distance is not None:
            distances.append(distance)
        results[cid] = result
        diagnostics["clusters"].append(result)

    diagnostics["method_eval"] = {
        "num_labeled_clusters": int(len(distances)),
        "mean_distance_to_manual_proxy_m": float(np.mean(distances)) if distances else None,
        "max_distance_to_manual_proxy_m": float(np.max(distances)) if distances else None,
    }

    (out_dir / "leaf_ray_targets_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    write_ply(
        out_dir / "leaf_ray_targets.ply",
        xyz,
        cluster_id,
        optimized,
        retained_masks,
        results,
        args.max_retained_ply_points,
    )

    target_cluster_id = np.asarray(sorted(results), dtype=np.int32)
    target_xyz = (
        np.asarray([results[cid]["target_xyz"] for cid in sorted(results)], dtype=np.float32)
        if results
        else np.zeros((0, 3), dtype=np.float32)
    )
    target_confidence = (
        np.asarray([results[cid]["target_confidence"] for cid in sorted(results)], dtype=np.float32)
        if results
        else np.zeros((0,), dtype=np.float32)
    )
    payload = {key: presence[key] for key in presence.files}
    payload["ray_target_cluster_id"] = target_cluster_id
    payload["ray_target_xyz"] = target_xyz
    payload["ray_target_confidence"] = target_confidence
    payload["interaction_target_cluster_id"] = target_cluster_id
    payload["interaction_target_xyz"] = target_xyz
    payload["interaction_target_confidence"] = target_confidence
    payload["interaction_target_method"] = np.asarray(args.target_method)
    np.savez_compressed(out_dir / "leaf_ray_targets.npz", **payload)

    print(f"Saved {out_dir / 'leaf_ray_targets_summary.json'}")
    print(f"Saved {out_dir / 'leaf_ray_targets.ply'}")
    print(f"Saved {out_dir / 'leaf_ray_targets.npz'}")
    print(json.dumps(diagnostics["method_eval"], indent=2))


if __name__ == "__main__":
    main()
