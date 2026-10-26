#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extract_leaf_ray_targets import (
    green_like,
    load_connection_seeds,
    load_manual_proxy,
    make_backbone_nodes,
    parse_cluster_ids,
    ray_scope,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ray-prior leaf targets against retained-backbone intersection targets."
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--connection_summary_json", required=True)
    parser.add_argument("--manual_votes_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--selected_clusters", default="0,1,2,3")

    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--green_min", default=0.08, type=float)
    parser.add_argument("--green_margin", default=-0.05, type=float)
    parser.add_argument("--value_min", default=0.06, type=float)
    parser.add_argument("--red_ratio_max", default=1.25, type=float)
    parser.add_argument("--stable_opacity_drop_max", default=0.02, type=float)
    parser.add_argument("--stable_presence_drop_max", default=0.20, type=float)
    parser.add_argument("--stable_visibility_min", default=4, type=int)

    parser.add_argument("--target_along", default=0.10, type=float)
    parser.add_argument("--target_sigma", default=0.035, type=float)
    parser.add_argument("--ray_min_along", default=0.015, type=float)
    parser.add_argument("--ray_max_along", default=0.42, type=float)
    parser.add_argument("--tube_radius", default=0.090, type=float)
    parser.add_argument("--bin_width", default=0.015, type=float)
    parser.add_argument("--min_bin_points", default=8, type=int)
    parser.add_argument("--density_radius", default=0.030, type=float)
    parser.add_argument("--component_edge_radius", default=0.026, type=float)
    parser.add_argument("--first_strong_ratio", default=0.68, type=float)
    parser.add_argument("--density_peak_ratio", default=0.70, type=float)
    parser.add_argument("--entry_density_ratio", default=0.45, type=float)
    return parser.parse_args()


def candidate_from_node(node, method):
    return {
        "method": method,
        "point_xyz": np.asarray(node["point"], dtype=np.float64).tolist(),
        "along_m": float(node["along"]),
        "perp_m": float(node["perp"]),
        "score": float(node.get("backbone_score", 0.0)),
        "density": int(node["density"]),
        "bin_points": int(node["bin_points"]),
        "dominant_component_size": int(node["dominant_component_size"]),
    }


def choose_ray_prior(nodes, args):
    if not nodes:
        return None
    scored = []
    for node in nodes:
        prior_score = float(np.exp(-((node["along"] - args.target_along) / max(args.target_sigma, 1e-6)) ** 2))
        score = (
            1.25 * prior_score
            + 0.55 * node["density_norm"]
            + 0.35 * node["count_norm"]
            + 0.45 * node["ray_score"]
        )
        item = dict(node)
        item["score"] = float(score)
        item["prior_score"] = prior_score
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return candidate_from_node(scored[0], f"ray_prior_{args.target_along:.2f}m")


def choose_largest_component_entry(nodes):
    if not nodes:
        return None
    sizes = {}
    for node in nodes:
        sizes[node["dominant_component"]] = max(
            sizes.get(node["dominant_component"], 0),
            node["dominant_component_size"],
        )
    largest_label = max(sizes, key=sizes.get)
    scoped = [node for node in nodes if node["dominant_component"] == largest_label]
    scoped.sort(key=lambda item: (item["along"], -item["backbone_score"]))
    return candidate_from_node(scoped[0], "largest_component_entry")


def choose_first_strong_backbone(nodes, ratio):
    if not nodes:
        return None
    best_score = max(node["backbone_score"] for node in nodes)
    threshold = ratio * best_score
    scoped = [
        node
        for node in nodes
        if node["backbone_score"] >= threshold and node["component_norm"] >= 0.45 and node["count_norm"] >= 0.35
    ]
    if not scoped:
        scoped = [node for node in nodes if node["backbone_score"] >= threshold]
    if not scoped:
        return None
    scoped.sort(key=lambda item: (item["along"], -item["backbone_score"]))
    out = candidate_from_node(scoped[0], "first_strong_backbone")
    out["threshold"] = float(threshold)
    return out


def choose_density_first_peak(nodes, ratio):
    if not nodes:
        return None
    ordered = sorted(nodes, key=lambda item: item["along"])
    density = np.asarray([node["density"] for node in ordered], dtype=np.float64)
    if density.size >= 3:
        smooth = density.copy()
        smooth[1:-1] = 0.25 * density[:-2] + 0.5 * density[1:-1] + 0.25 * density[2:]
    else:
        smooth = density
    threshold = float(ratio * smooth.max())
    candidates = [node for node, value in zip(ordered, smooth) if value >= threshold]
    if not candidates:
        return None
    out = candidate_from_node(candidates[0], "density_first_peak")
    out["density_threshold"] = threshold
    return out


def choose_score_peak(nodes):
    if not nodes:
        return None
    return candidate_from_node(max(nodes, key=lambda item: item["backbone_score"]), "backbone_score_peak")


def largest_component_label(nodes):
    sizes = {}
    for node in nodes:
        sizes[node["dominant_component"]] = max(
            sizes.get(node["dominant_component"], 0),
            node["dominant_component_size"],
        )
    return max(sizes, key=sizes.get) if sizes else None


def choose_largest_component_strong_density(nodes, ratio):
    if not nodes:
        return None
    label = largest_component_label(nodes)
    scoped = [node for node in nodes if node["dominant_component"] == label]
    max_density = max(node["density"] for node in scoped)
    candidates = [node for node in scoped if node["density"] >= ratio * max_density] or scoped
    candidates.sort(key=lambda item: (item["along"], -item["density"]))
    out = candidate_from_node(candidates[0], "largest_component_strong_density")
    out["density_threshold"] = float(ratio * max_density)
    return out


def choose_largest_component_strong_score(nodes, ratio):
    if not nodes:
        return None
    label = largest_component_label(nodes)
    scoped = [node for node in nodes if node["dominant_component"] == label]
    max_score = max(node["backbone_score"] for node in scoped)
    candidates = [node for node in scoped if node["backbone_score"] >= ratio * max_score] or scoped
    candidates.sort(key=lambda item: (item["along"], -item["backbone_score"]))
    out = candidate_from_node(candidates[0], "largest_component_strong_score")
    out["score_threshold"] = float(ratio * max_score)
    return out


def choose_largest_gap_entry(nodes, args):
    if len(nodes) < 3:
        return None
    ordered = sorted(nodes, key=lambda item: item["along"])
    gaps = np.asarray(
        [ordered[i + 1]["along"] - ordered[i]["along"] for i in range(len(ordered) - 1)],
        dtype=np.float64,
    )
    median_gap = float(np.median(gaps)) if gaps.size else 0.0
    gap_idx = int(np.argmax(gaps))
    if gaps[gap_idx] < 2.5 * max(median_gap, args.bin_width):
        return None
    out = candidate_from_node(ordered[gap_idx + 1], "largest_gap_entry")
    out["gap_m"] = float(gaps[gap_idx])
    return out


def choose_largest_component_gap_or_rise(nodes, args):
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
                out = candidate_from_node(scoped[int(gap_idx) + 1], "largest_component_gap_or_rise")
                out["decision"] = "after_largest_component_gap"
                out["gap_m"] = float(gaps[gap_idx])
                return out

    max_density = max(node["density"] for node in scoped)
    entry = scoped[0]
    if entry["density"] >= args.entry_density_ratio * max_density:
        out = candidate_from_node(entry, "largest_component_gap_or_rise")
        out["decision"] = "component_entry_dense_enough"
        out["entry_density_ratio"] = float(entry["density"] / max(max_density, 1))
        return out

    candidates = [node for node in scoped if node["density"] >= args.density_peak_ratio * max_density] or scoped
    candidates.sort(key=lambda item: (item["along"], -item["density"]))
    out = candidate_from_node(candidates[0], "largest_component_gap_or_rise")
    out["decision"] = "first_density_rise"
    out["density_threshold"] = float(args.density_peak_ratio * max_density)
    return out


def choose_node_oracle(nodes, manual_point):
    if not nodes:
        return None
    best = min(
        nodes,
        key=lambda item: float(np.linalg.norm(np.asarray(item["point"], dtype=np.float64) - manual_point)),
    )
    return candidate_from_node(best, "node_oracle")


def add_distance(candidate, manual_point):
    if candidate is None:
        return None
    point = np.asarray(candidate["point_xyz"], dtype=np.float64)
    distance = float(np.linalg.norm(point - manual_point))
    candidate["distance_to_manual_proxy_m"] = distance
    return distance


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

    opacity = presence["opacity_pre"].reshape(-1).astype(np.float32)
    opacity_drop = presence["opacity_drop"].reshape(-1).astype(np.float32)
    presence_drop = presence["presence_drop"].reshape(-1).astype(np.float32)
    rgb = score_npz["rgb"].astype(np.float32)
    plant_roi = score_npz["plant_structure_roi"].astype(bool)
    visibility = score_npz["visibility_count"].astype(np.float32)

    plant_green = green_like(rgb, opacity, plant_roi, args)
    stable_retained = (
        plant_green
        & ~optimized
        & (visibility >= args.stable_visibility_min)
        & ((opacity_drop < args.stable_opacity_drop_max) | (presence_drop < args.stable_presence_drop_max))
    )
    connection_seeds = load_connection_seeds(args.connection_summary_json)
    manual_proxy = load_manual_proxy(args.manual_votes_json)

    methods = [
        "ray_prior_0.10m",
        "largest_component_entry",
        "largest_component_gap_or_rise",
        "largest_component_strong_density",
        "largest_component_strong_score",
        "largest_gap_entry",
        "first_strong_backbone",
        "density_first_peak",
        "backbone_score_peak",
        "node_oracle",
    ]
    method_distances = {method: [] for method in methods}
    summary = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "connection_summary_json": args.connection_summary_json,
        "selected_clusters": clusters,
        "note": "Backbone candidates use retained green, visible, low-opacity-change Gaussians in the endpoint ray tube.",
        "clusters": [],
    }

    for cid in clusters:
        seed = connection_seeds[cid]
        endpoint = seed["endpoint"]
        direction = seed["direction"]
        along, perp, retained = ray_scope(xyz, stable_retained, endpoint, direction, args)
        nodes = make_backbone_nodes(xyz, retained, endpoint, direction, along, perp, visibility, args)
        manual_point = manual_proxy[f"leaf{cid}"]
        candidates = {
            "ray_prior_0.10m": choose_ray_prior(nodes, args),
            "largest_component_entry": choose_largest_component_entry(nodes),
            "largest_component_gap_or_rise": choose_largest_component_gap_or_rise(nodes, args),
            "largest_component_strong_density": choose_largest_component_strong_density(nodes, args.density_peak_ratio),
            "largest_component_strong_score": choose_largest_component_strong_score(nodes, args.first_strong_ratio),
            "largest_gap_entry": choose_largest_gap_entry(nodes, args),
            "first_strong_backbone": choose_first_strong_backbone(nodes, args.first_strong_ratio),
            "density_first_peak": choose_density_first_peak(nodes, args.density_peak_ratio),
            "backbone_score_peak": choose_score_peak(nodes),
            "node_oracle": choose_node_oracle(nodes, manual_point),
        }
        for method, candidate in candidates.items():
            distance = add_distance(candidate, manual_point)
            if distance is not None:
                method_distances[method].append(distance)
        summary["clusters"].append(
            {
                "cluster_id": int(cid),
                "retained_candidates": int(retained.sum()),
                "nodes": int(len(nodes)),
                "manual_proxy_xyz": manual_point.tolist(),
                "candidates": candidates,
                "node_profile": [
                    {
                        "along_m": float(node["along"]),
                        "density": int(node["density"]),
                        "bin_points": int(node["bin_points"]),
                        "component_size": int(node["dominant_component_size"]),
                        "score": float(node["backbone_score"]),
                        "distance_to_manual_proxy_m": float(
                            np.linalg.norm(np.asarray(node["point"], dtype=np.float64) - manual_point)
                        ),
                    }
                    for node in nodes
                ],
            }
        )

    summary["method_eval"] = {
        method: {
            "num_clusters": int(len(values)),
            "mean_distance_to_manual_proxy_m": float(np.mean(values)) if values else None,
            "max_distance_to_manual_proxy_m": float(np.max(values)) if values else None,
        }
        for method, values in method_distances.items()
    }
    (out_dir / "backbone_target_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["method_eval"], indent=2))


if __name__ == "__main__":
    main()
