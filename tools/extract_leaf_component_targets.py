#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from extract_leaf_ray_targets import (
    COLORS,
    bbox_diag,
    choose_backbone_gap_or_rise_target,
    choose_ray_target,
    green_like,
    make_backbone_nodes,
    make_centerline_nodes,
    ray_scope,
    weighted_centroid,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split removed leaf Gaussian masks into physical components, reject weak "
            "false positives, and infer one interaction target per component."
        )
    )
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--selected_clusters", default="all")

    parser.add_argument("--component_radius", default=0.018, type=float)
    parser.add_argument("--min_component_gaussians", default=60, type=int)
    parser.add_argument("--min_component_fraction", default=0.015, type=float)
    parser.add_argument("--max_components_per_cluster", default=12, type=int)

    parser.add_argument("--weak_core_overlap_max", default=0.05, type=float)
    parser.add_argument("--weak_median_score_max", default=0.15, type=float)
    parser.add_argument("--tomato_near_radius", default=0.08, type=float)
    parser.add_argument("--tomato_near_frac_min", default=0.8, type=float)
    parser.add_argument("--red_min", default=0.28, type=float)
    parser.add_argument("--red_ratio_min", default=1.25, type=float)

    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--green_min", default=0.08, type=float)
    parser.add_argument("--green_margin", default=-0.05, type=float)
    parser.add_argument("--value_min", default=0.06, type=float)
    parser.add_argument("--red_ratio_max", default=1.25, type=float)
    parser.add_argument("--stable_opacity_drop_max", default=0.02, type=float)
    parser.add_argument("--stable_presence_drop_max", default=0.20, type=float)
    parser.add_argument("--stable_residual_max", default=0.12, type=float)
    parser.add_argument("--stable_visibility_min", default=4, type=int)

    parser.add_argument(
        "--target_method",
        choices=(
            "backbone_gap_or_rise",
            "ray_prior",
            "local_contact",
            "backbone_guided_local_contact",
            "validated_backbone",
            "validated_mask_endpoint",
        ),
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

    parser.add_argument("--local_contact_max_along", default=0.12, type=float)
    parser.add_argument("--local_contact_tube_radius", default=0.05, type=float)
    parser.add_argument("--local_contact_shell_radius", default=0.045, type=float)
    parser.add_argument("--local_contact_min_points", default=5, type=int)
    parser.add_argument("--local_contact_max_gap", default=0.10, type=float)
    parser.add_argument("--local_contact_midpoint_weight", default=0.65, type=float)
    parser.add_argument("--local_contact_max_selected_ahead_ratio", default=0.30, type=float)
    parser.add_argument("--allow_unguided_local_contact", action="store_true")
    parser.add_argument("--allow_reverse_probe_local_contact", action="store_true")
    parser.add_argument("--guide_max_endpoint_distance", default=0.12, type=float)
    parser.add_argument("--guide_local_agreement_max", default=0.08, type=float)
    parser.add_argument("--mask_endpoint_target_points", default=32, type=int)
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


def connected_component_indices(points, radius):
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


def pca_axis(points, weights):
    center = weighted_centroid(points, weights)
    centered = points.astype(np.float64) - center[None, :]
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if weights.sum() <= 0.0:
        weights = np.ones(points.shape[0], dtype=np.float64)
    cov = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-8)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axis = vecs[:, int(order[0])]
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    return center, axis, np.maximum(vals[order], 0.0)


def red_like(rgb, args):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    return (r >= args.red_min) & (r >= g * args.red_ratio_min) & (r >= b * args.red_ratio_min)


def tomato_near_fraction(xyz, component_idx, red_tree, args):
    if red_tree is None or component_idx.size == 0:
        return 0.0
    hits = red_tree.query_ball_point(xyz[component_idx], args.tomato_near_radius)
    return float(np.mean([len(item) > 0 for item in hits]))


def component_rejection_reason(stats, args):
    weak = (
        stats["core_overlap_ratio"] < args.weak_core_overlap_max
        and stats["median_residual_score"] < args.weak_median_score_max
    )
    tomato_adjacent = stats["tomato_near_fraction"] >= args.tomato_near_frac_min
    if weak and tomato_adjacent:
        return "weak_residual_evidence_tomato_adjacent"
    if weak:
        return "weak_residual_evidence"
    return None


def build_instances(xyz, optimized, cluster_id, candidate_core, residual_score, red_tree, clusters, args):
    leaf_instance_id = np.full(xyz.shape[0], -1, dtype=np.int32)
    leaf_instance_source_cluster_id = np.full(xyz.shape[0], -1, dtype=np.int32)
    leaf_instance_component_id = np.full(xyz.shape[0], -1, dtype=np.int32)
    accepted = []
    rejected = []
    next_instance_id = 0

    for cid in clusters:
        base_idx = np.flatnonzero(optimized & (cluster_id == cid))
        if base_idx.size == 0:
            continue
        components = connected_component_indices(xyz[base_idx].astype(np.float64), args.component_radius)
        min_size = max(args.min_component_gaussians, int(np.ceil(args.min_component_fraction * max(base_idx.size, 1))))
        for component_id, component in enumerate(components[: args.max_components_per_cluster]):
            global_idx = base_idx[component]
            stats = {
                "source_cluster_id": int(cid),
                "source_component_id": int(component_id),
                "num_gaussians": int(global_idx.size),
                "min_component_gaussians": int(min_size),
                "bbox_min": xyz[global_idx].min(axis=0).tolist(),
                "bbox_max": xyz[global_idx].max(axis=0).tolist(),
                "bbox_diag": bbox_diag(xyz[global_idx]),
                "median_residual_score": float(np.median(residual_score[global_idx])),
                "mean_residual_score": float(np.mean(residual_score[global_idx])),
                "core_overlap_ratio": float(np.mean(candidate_core[global_idx])),
                "tomato_near_fraction": tomato_near_fraction(xyz, global_idx, red_tree, args),
            }
            if global_idx.size < min_size:
                rejected.append({**stats, "rejection_reason": "small_component"})
                continue
            reason = component_rejection_reason(stats, args)
            if reason is not None:
                rejected.append({**stats, "rejection_reason": reason})
                continue

            instance_id = next_instance_id
            next_instance_id += 1
            leaf_instance_id[global_idx] = instance_id
            leaf_instance_source_cluster_id[global_idx] = int(cid)
            leaf_instance_component_id[global_idx] = int(component_id)
            accepted_stats = {**stats, "leaf_instance_id": int(instance_id)}
            accepted.append({"indices": global_idx, "stats": accepted_stats})

    return leaf_instance_id, leaf_instance_source_cluster_id, leaf_instance_component_id, accepted, rejected


def endpoint_direction_candidates(xyz, seed_idx, weights):
    pts = xyz[seed_idx]
    center, axis, eigvals = pca_axis(pts, weights[seed_idx])
    projection = (pts.astype(np.float64) - center[None, :]) @ axis
    candidates = []
    for side, quantile, sign in (("low", 0.05, -1.0), ("high", 0.95, 1.0)):
        if side == "low":
            endpoint_local = projection <= np.quantile(projection, quantile)
        else:
            endpoint_local = projection >= np.quantile(projection, quantile)
        endpoint = pts[endpoint_local].mean(axis=0).astype(np.float64)
        outward = endpoint - center
        outward /= max(float(np.linalg.norm(outward)), 1e-8)
        candidates.append(
            {
                "side": side,
                "direction_mode": "outward",
                "endpoint": endpoint,
                "endpoint_indices": seed_idx[endpoint_local],
                "direction": outward,
                "pca_center": center,
                "pca_axis": axis,
                "pca_side_sign": sign,
                "pca_eigvals": eigvals,
            }
        )
        candidates.append(
            {
                "side": side,
                "direction_mode": "reverse_probe",
                "endpoint": endpoint,
                "endpoint_indices": seed_idx[endpoint_local],
                "direction": -outward,
                "pca_center": center,
                "pca_axis": axis,
                "pca_side_sign": sign,
                "pca_eigvals": eigvals,
            }
        )
    return candidates


def choose_mask_endpoint_target(xyz, weights, candidate, local_contact_target, args):
    endpoint_idx = np.asarray(candidate.get("endpoint_indices", []), dtype=np.int64)
    if endpoint_idx.size == 0:
        return None
    endpoint_points = xyz[endpoint_idx].astype(np.float64)
    endpoint = np.asarray(candidate["endpoint"], dtype=np.float64)
    if local_contact_target is not None and local_contact_target.get("support_point_xyz") is not None:
        anchor = np.asarray(local_contact_target["support_point_xyz"], dtype=np.float64)
    else:
        anchor = endpoint

    anchor_dist = np.linalg.norm(endpoint_points - anchor[None, :], axis=1)
    keep_n = min(max(int(args.mask_endpoint_target_points), 1), endpoint_idx.size)
    keep_order = np.argsort(anchor_dist)[:keep_n]
    keep_idx = endpoint_idx[keep_order]
    keep_points = xyz[keep_idx].astype(np.float64)
    local_dist = np.maximum(anchor_dist[keep_order], 0.003)
    local_weights = 1.0 / local_dist
    local_weights *= np.maximum(weights[keep_idx], 0.0001)
    centroid = weighted_centroid(keep_points, local_weights)
    medoid = int(np.argmin(np.linalg.norm(keep_points - centroid[None, :], axis=1)))
    point = keep_points[medoid]

    contact_gap = None
    local_confidence = 0.5
    support_count = 0
    if local_contact_target is not None:
        contact_gap = local_contact_target.get("contact_gap_m")
        local_confidence = float(local_contact_target.get("confidence", local_confidence))
        support_count = int(local_contact_target.get("density", 0))
    endpoint_gap = float(np.linalg.norm(point - endpoint))
    anchor_gap = float(np.linalg.norm(point - anchor))
    confidence = local_confidence * np.exp(-endpoint_gap / 0.08)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return {
        "point_xyz": point.tolist(),
        "method": "validated_mask_endpoint",
        "score": confidence,
        "confidence": confidence,
        "along_m": 0.0,
        "perp_m": 0.0,
        "density": support_count,
        "bin_points": int(keep_idx.size),
        "endpoint_xyz": endpoint.tolist(),
        "endpoint_gap_m": endpoint_gap,
        "anchor_gap_m": anchor_gap,
        "contact_gap_m": contact_gap,
        "anchor_xyz": anchor.tolist(),
        "nearest_selected_distance_m": 0.0,
        "target_is_selected_gaussian": True,
    }


def choose_local_contact_target(xyz, stable_retained, endpoint, direction, along, perp, args):
    if not stable_retained.any():
        return None
    rel = xyz.astype(np.float64) - endpoint[None, :]
    dist = np.linalg.norm(rel, axis=1)
    cone = (
        stable_retained
        & (along >= 0.0)
        & (along <= args.local_contact_max_along)
        & (perp <= args.local_contact_tube_radius)
    )
    shell = stable_retained & (dist <= args.local_contact_shell_radius)
    candidates = np.flatnonzero(cone | shell)
    if candidates.size < args.local_contact_min_points:
        return None

    cand_along = along[candidates]
    cand_perp = perp[candidates]
    cand_dist = dist[candidates]
    forward = cand_along >= -0.005
    if forward.any():
        candidates = candidates[forward]
        cand_along = cand_along[forward]
        cand_perp = cand_perp[forward]
        cand_dist = cand_dist[forward]
    if candidates.size < args.local_contact_min_points:
        return None

    order = np.lexsort((cand_perp, np.maximum(cand_along, 0.0), cand_dist))
    keep = order[: min(max(args.local_contact_min_points, 12), order.size)]
    local_idx = candidates[keep]
    local_dist = cand_dist[keep]
    local_perp = cand_perp[keep]
    support_weights = 1.0 / np.maximum(local_dist, 0.003)
    support_weights *= np.exp(-((local_perp / max(args.local_contact_tube_radius, 1e-6)) ** 2))
    support = weighted_centroid(xyz[local_idx].astype(np.float64), support_weights)
    gap = float(np.linalg.norm(support - endpoint))
    if gap > args.local_contact_max_gap:
        return None

    target = (1.0 - args.local_contact_midpoint_weight) * endpoint + args.local_contact_midpoint_weight * support
    support_count = int(candidates.size)
    confidence = float(np.exp(-gap / max(args.local_contact_max_gap, 1e-8)))
    confidence *= float(np.clip(support_count / max(args.local_contact_min_points, 1), 0.0, 1.0))
    return {
        "point_xyz": target.tolist(),
        "method": "local_contact",
        "score": confidence,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "along_m": float(np.mean(cand_along[keep])),
        "perp_m": float(np.mean(cand_perp[keep])),
        "density": support_count,
        "bin_points": int(keep.size),
        "support_point_xyz": support.tolist(),
        "contact_gap_m": gap,
        "cone_support_points": int(cone.sum()),
        "shell_support_points": int(shell.sum()),
    }


def target_endpoint_distance(target, endpoint):
    if target is None:
        return None
    point = np.asarray(target["point_xyz"], dtype=np.float64)
    return float(np.linalg.norm(point - endpoint))


def score_direction(candidate, target, retained_count, backbone_count, selected_ahead_ratio):
    if target is None:
        base = 0.0
    else:
        base = float(target.get("confidence", 0.0))
        if target.get("method") != "local_contact":
            base += 0.05 * np.log1p(max(retained_count, 0))
            base += 0.03 * np.log1p(max(backbone_count, 0))
        distance = target_endpoint_distance(target, np.asarray(candidate["endpoint"], dtype=np.float64))
        if distance is not None:
            base -= 0.45 * max(distance - 0.08, 0.0)
    if candidate["direction_mode"] == "reverse_probe":
        base -= 0.1
    base -= 0.35 * selected_ahead_ratio
    return float(base)


def adjusted_target_confidence(raw_confidence, direction_mode, selected_ahead_ratio):
    reliability = 1.0 - 0.45 * float(selected_ahead_ratio)
    if direction_mode == "reverse_probe":
        reliability -= 0.15
    reliability = float(np.clip(reliability, 0.2, 1.0))
    return float(np.clip(raw_confidence * reliability, 0.0, 1.0)), reliability


def _add_guide_metadata(target, guide_target, endpoint):
    target["guide_method"] = guide_target.get("method")
    target["guide_decision"] = guide_target.get("decision")
    target["guide_point_xyz"] = guide_target.get("point_xyz")
    target["guide_endpoint_distance"] = target_endpoint_distance(guide_target, endpoint)


def _failure(instance_id, reason, best, attempts):
    if best is None:
        return {
            "cluster_id": int(instance_id),
            "target_xyz": None,
            "failure_reason": "no_retained_support",
            "direction_attempts": attempts,
        }, None
    result = {
        "cluster_id": int(instance_id),
        "target_xyz": None,
        "failure_reason": reason,
        "endpoint_xyz": best["endpoint_xyz"],
        "direction": best["direction"],
        "endpoint_side": best["side"],
        "direction_mode": best["direction_mode"],
        "selected_ahead_ratio": float(best["selected_ahead_ratio"]),
        "selection_score": float(best["selection_score"]),
        "direction_attempts": attempts,
    }
    return result, best["retained_mask"]


def process_instance(xyz, seed_idx, stable_retained, visibility, weights, instance_id, args):
    seed_mask = np.zeros(xyz.shape[0], dtype=bool)
    seed_mask[seed_idx] = True
    best = None
    attempts = []

    for candidate in endpoint_direction_candidates(xyz, seed_idx, weights):
        endpoint = candidate["endpoint"]
        direction = candidate["direction"]
        along, perp, retained = ray_scope(xyz, stable_retained, endpoint, direction, args)
        selected_ahead = (
            seed_mask
            & (along >= args.ray_min_along)
            & (along <= min(args.ray_max_along, 0.3))
            & (perp <= args.tube_radius)
        )
        selected_ahead_ratio = float(selected_ahead.sum() / max(seed_idx.size, 1))
        ray_nodes = make_centerline_nodes(xyz, retained, endpoint, direction, along, perp, args)
        backbone_nodes = make_backbone_nodes(xyz, retained, endpoint, direction, along, perp, visibility, args)

        local_contact_target = None
        guide_target = None
        if args.target_method in (
            "local_contact",
            "backbone_guided_local_contact",
            "validated_backbone",
            "validated_mask_endpoint",
        ):
            local_contact_target = choose_local_contact_target(
                xyz, stable_retained, endpoint, direction, along, perp, args
            )

        if args.target_method == "local_contact":
            target = local_contact_target
            score_target = target
        elif args.target_method == "backbone_guided_local_contact":
            guide_target = choose_backbone_gap_or_rise_target(backbone_nodes, args)
            if guide_target is None:
                guide_target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
                if guide_target is not None:
                    guide_target["fallback_from"] = "backbone_gap_or_rise"
            target = local_contact_target if guide_target is not None or args.allow_unguided_local_contact else None
            score_target = guide_target if target is None and guide_target is not None else target
            if target is not None and guide_target is not None:
                _add_guide_metadata(target, guide_target, endpoint)
        elif args.target_method == "validated_backbone":
            guide_target = choose_backbone_gap_or_rise_target(backbone_nodes, args)
            if guide_target is None:
                guide_target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
                if guide_target is not None:
                    guide_target["fallback_from"] = "backbone_gap_or_rise"
            target = guide_target
            score_target = guide_target
            if target is not None and local_contact_target is not None:
                guide_point = np.asarray(target["point_xyz"], dtype=np.float64)
                contact_point = np.asarray(local_contact_target["point_xyz"], dtype=np.float64)
                target["local_contact_point_xyz"] = local_contact_target["point_xyz"]
                target["local_contact_gap_m"] = local_contact_target.get("contact_gap_m")
                target["local_contact_agreement_m"] = float(np.linalg.norm(guide_point - contact_point))
        elif args.target_method == "validated_mask_endpoint":
            guide_target = choose_backbone_gap_or_rise_target(backbone_nodes, args)
            if guide_target is None:
                guide_target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
                if guide_target is not None:
                    guide_target["fallback_from"] = "backbone_gap_or_rise"
            target = (
                choose_mask_endpoint_target(xyz, weights, candidate, local_contact_target, args)
                if guide_target is not None and local_contact_target is not None
                else None
            )
            score_target = guide_target if guide_target is not None else target
            if target is not None:
                guide_point = np.asarray(guide_target["point_xyz"], dtype=np.float64)
                contact_point = np.asarray(local_contact_target["point_xyz"], dtype=np.float64)
                _add_guide_metadata(target, guide_target, endpoint)
                target["local_contact_point_xyz"] = local_contact_target["point_xyz"]
                target["local_contact_gap_m"] = local_contact_target.get("contact_gap_m")
                target["local_contact_agreement_m"] = float(np.linalg.norm(guide_point - contact_point))
        elif args.target_method == "ray_prior":
            target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
            score_target = target
        else:
            target = choose_backbone_gap_or_rise_target(backbone_nodes, args)
            if target is None:
                target = choose_ray_target(ray_nodes, xyz, retained, endpoint, direction, along, perp, args)
                if target is not None:
                    target["fallback_from"] = args.target_method
            score_target = target

        retained_count = int(retained.sum())
        score = score_direction(
            candidate,
            score_target,
            retained_count,
            len(backbone_nodes),
            selected_ahead_ratio,
        )
        attempt = {
            "side": candidate["side"],
            "direction_mode": candidate["direction_mode"],
            "endpoint_xyz": endpoint.tolist(),
            "direction": direction.tolist(),
            "retained_candidates": retained_count,
            "centerline_nodes": int(len(ray_nodes)),
            "backbone_nodes": int(len(backbone_nodes)),
            "selected_ahead_ratio": selected_ahead_ratio,
            "target_endpoint_distance": target_endpoint_distance(target, endpoint),
            "guide_endpoint_distance": target_endpoint_distance(guide_target, endpoint),
            "selection_score": score,
            "target": target,
            "retained_mask": retained,
        }
        attempts.append({key: value for key, value in attempt.items() if key != "retained_mask"})
        if best is None or score > best["selection_score"]:
            best = attempt

    if best is None or best["target"] is None:
        result = {
            "cluster_id": int(instance_id),
            "target_xyz": None,
            "failure_reason": "no_retained_support",
            "direction_attempts": attempts,
        }
        return result, np.zeros(xyz.shape[0], dtype=bool)

    if args.target_method in (
        "local_contact",
        "backbone_guided_local_contact",
        "validated_backbone",
        "validated_mask_endpoint",
    ):
        if best["selected_ahead_ratio"] > args.local_contact_max_selected_ahead_ratio:
            return _failure(instance_id, "ambiguous_endpoint_selected_ahead", best, attempts)
        if best["direction_mode"] == "reverse_probe" and not args.allow_reverse_probe_local_contact:
            return _failure(instance_id, "reverse_probe_local_contact", best, attempts)

        if args.target_method in ("validated_backbone", "validated_mask_endpoint"):
            endpoint = np.asarray(best["endpoint_xyz"], dtype=np.float64)
            guide_distance = best["target"].get("guide_endpoint_distance")
            if guide_distance is None:
                guide_distance = target_endpoint_distance(best["target"], endpoint)
            agreement = best["target"].get("local_contact_agreement_m")
            if guide_distance is None or guide_distance > args.guide_max_endpoint_distance:
                result, mask = _failure(instance_id, "guide_too_far_from_endpoint", best, attempts)
                result["guide_endpoint_distance"] = guide_distance
                return result, mask
            if agreement is None or agreement > args.guide_local_agreement_max:
                result, mask = _failure(instance_id, "guide_local_contact_disagreement", best, attempts)
                result["guide_endpoint_distance"] = guide_distance
                result["local_contact_agreement_m"] = agreement
                return result, mask

    target = best["target"]
    raw_confidence = float(target["confidence"])
    adjusted_confidence, direction_reliability = adjusted_target_confidence(
        raw_confidence,
        best["direction_mode"],
        best["selected_ahead_ratio"],
    )
    result = {
        "cluster_id": int(instance_id),
        "endpoint_xyz": best["endpoint_xyz"],
        "direction": best["direction"],
        "endpoint_side": best["side"],
        "direction_mode": best["direction_mode"],
        "retained_candidates": int(best["retained_candidates"]),
        "centerline_nodes": int(best["centerline_nodes"]),
        "backbone_nodes": int(best["backbone_nodes"]),
        "selected_ahead_ratio": float(best["selected_ahead_ratio"]),
        "target_endpoint_distance": best["target_endpoint_distance"],
        "selection_score": float(best["selection_score"]),
        "target_xyz": target["point_xyz"],
        "target_confidence": adjusted_confidence,
        "target_raw_confidence": raw_confidence,
        "target_direction_reliability": direction_reliability,
        "target": target,
        "direction_attempts": attempts,
    }
    return result, best["retained_mask"]


def write_ply(path, xyz, leaf_instance_id, retained_masks, results, max_retained_points):
    points = []
    colors = []
    labels = []
    result_by_id = {int(item["cluster_id"]): item for item in results}
    for iid in sorted(int(v) for v in np.unique(leaf_instance_id) if int(v) >= 0):
        base_idx = np.flatnonzero(leaf_instance_id == iid)
        if base_idx.size:
            color = np.asarray(COLORS.get(iid, (255, 255, 255)), dtype=np.uint8)
            points.append(xyz[base_idx])
            colors.append(np.tile(color, (base_idx.size, 1)))
            labels.append(np.full(base_idx.size, iid, dtype=np.int32))

        retained = retained_masks.get(iid)
        if retained is not None:
            retained_idx = np.flatnonzero(retained)
            if retained_idx.size > max_retained_points:
                rng = np.random.default_rng(iid)
                retained_idx = rng.choice(retained_idx, max_retained_points, replace=False)
            if retained_idx.size:
                points.append(xyz[retained_idx])
                colors.append(np.tile(np.asarray((70, 210, 95), dtype=np.uint8), (retained_idx.size, 1)))
                labels.append(np.full(retained_idx.size, iid + 20, dtype=np.int32))

    target_points = []
    for _iid, result in sorted(result_by_id.items()):
        if result.get("target_xyz") is None:
            continue
        target_points.append(np.asarray(result["target_xyz"], dtype=np.float32))
    if target_points:
        target_xyz = np.asarray(target_points, dtype=np.float32)
        points.append(target_xyz)
        colors.append(np.tile(np.asarray((255, 255, 255), dtype=np.uint8), (target_xyz.shape[0], 1)))
        labels.append(np.full(target_xyz.shape[0], 100, dtype=np.int32))

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
    candidate_core = (
        presence["candidate_core"].astype(bool)
        if "candidate_core" in presence.files
        else score_npz["candidate"].astype(bool)
    )
    residual_score = score_npz["score"].astype(np.float32)
    clusters = parse_cluster_ids(args.selected_clusters, cluster_id, optimized)

    rgb = score_npz["rgb"].astype(np.float32)
    red_mask = red_like(rgb, args)
    red_tree = cKDTree(xyz[red_mask]) if red_mask.any() else None
    (
        leaf_instance_id,
        leaf_instance_source_cluster_id,
        leaf_instance_component_id,
        accepted,
        rejected,
    ) = build_instances(xyz, optimized, cluster_id, candidate_core, residual_score, red_tree, clusters, args)
    leaf_instance_candidate = leaf_instance_id >= 0

    opacity = presence["opacity_pre"].reshape(-1).astype(np.float32)
    opacity_drop = presence["opacity_drop"].reshape(-1).astype(np.float32)
    presence_drop = presence["presence_drop"].reshape(-1).astype(np.float32)
    plant_roi = score_npz["plant_structure_roi"].astype(bool)
    visibility = (
        score_npz["visibility_count"].astype(np.float32)
        if "visibility_count" in score_npz.files
        else np.zeros(xyz.shape[0], dtype=np.float32)
    )

    plant_green = green_like(rgb, opacity, plant_roi, args)
    stable_retained = (
        plant_green
        & ~leaf_instance_candidate
        & (visibility >= args.stable_visibility_min)
        & (
            (residual_score <= args.stable_residual_max)
            | (opacity_drop < args.stable_opacity_drop_max)
            | (presence_drop < args.stable_presence_drop_max)
        )
    )

    retained_masks = {}
    results = []
    target_instance_ids = []
    target_source_cluster_ids = []
    target_xyz = []
    target_confidence = []
    weights = np.maximum(opacity_drop, 0.0) + 0.25 * np.maximum(residual_score, 0.0) + 0.0001

    for item in accepted:
        iid = int(item["stats"]["leaf_instance_id"])
        result, retained = process_instance(
            xyz,
            item["indices"],
            stable_retained,
            visibility,
            weights,
            iid,
            args,
        )
        result.update(item["stats"])
        result["base_gaussians"] = int(item["indices"].size)
        result["base_bbox_diag"] = bbox_diag(xyz[item["indices"]])
        retained_masks[iid] = retained
        results.append(result)
        if result.get("target_xyz") is not None:
            target_instance_ids.append(iid)
            target_source_cluster_ids.append(int(item["stats"]["source_cluster_id"]))
            target_xyz.append(result["target_xyz"])
            target_confidence.append(float(result["target_confidence"]))

    diagnostics = {
        "presence_npz": args.presence_npz,
        "score_npz": args.score_npz,
        "method": "leaf_component_interaction_targets",
        "method_note": (
            "The fixed removed Gaussian mask is first split into connected components. "
            "Weak residual components, including tomato-adjacent false positives, are "
            "rejected before target extraction. Each accepted component then performs "
            "a bidirectional endpoint search and snaps the target to retained green "
            "plant geometry near the best endpoint ray."
        ),
        "selected_source_clusters": clusters,
        "component_radius_m": float(args.component_radius),
        "stable_residual_max": float(args.stable_residual_max),
        "target_method": args.target_method,
        "num_accepted_instances": int(len(accepted)),
        "num_targets": int(len(target_xyz)),
        "num_rejected_components": int(len(rejected)),
        "clusters": results,
        "rejected_components": rejected,
    }

    payload = {key: presence[key] for key in presence.files}
    payload["leaf_instance_candidate"] = leaf_instance_candidate
    payload["leaf_instance_id"] = leaf_instance_id
    payload["leaf_instance_source_cluster_id"] = leaf_instance_source_cluster_id
    payload["leaf_instance_component_id"] = leaf_instance_component_id
    payload["interaction_target_instance_id"] = np.asarray(target_instance_ids, dtype=np.int32)
    payload["interaction_target_cluster_id"] = np.asarray(target_instance_ids, dtype=np.int32)
    payload["interaction_target_source_cluster_id"] = np.asarray(target_source_cluster_ids, dtype=np.int32)
    payload["interaction_target_xyz"] = (
        np.asarray(target_xyz, dtype=np.float32) if target_xyz else np.zeros((0, 3), dtype=np.float32)
    )
    payload["interaction_target_confidence"] = (
        np.asarray(target_confidence, dtype=np.float32) if target_confidence else np.zeros((0,), dtype=np.float32)
    )
    payload["interaction_target_method"] = np.asarray("component_backbone_endpoint")
    np.savez_compressed(out_dir / "leaf_component_targets.npz", **payload)
    write_ply(
        out_dir / "leaf_component_targets.ply",
        xyz,
        leaf_instance_id,
        retained_masks,
        results,
        args.max_retained_ply_points,
    )
    (out_dir / "leaf_component_targets_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    print(f"Saved {out_dir / 'leaf_component_targets.npz'}")
    print(f"Saved {out_dir / 'leaf_component_targets.ply'}")
    print(f"Saved {out_dir / 'leaf_component_targets_summary.json'}")
    print(
        json.dumps(
            {
                "accepted_instances": int(len(accepted)),
                "targets": int(len(target_xyz)),
                "rejected_components": int(len(rejected)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
