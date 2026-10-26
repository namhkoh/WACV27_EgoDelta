from __future__ import annotations

from typing import Any

import numpy as np

from tomato_cutpoints.targets.geometry import (
    endpoints_along_axis,
    nearby_persistent_points,
    nearest_distance_to_set,
    nearest_pair,
    normalize,
)


SUPPORTED_TASKS = {"harvest", "deleaf"}


def _base_target(task: str, cluster: dict[str, Any], object_type: str) -> dict[str, Any]:
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "object_type": object_type,
        "object_center_world": None,
        "leaf_center_world": None,
        "cut_point_world": None,
        "approach_direction_world": None,
        "removal_direction_world": None,
        "confidence": 0.0,
        "diagnostics": {
            "n_points": int(cluster.get("n_points", 0)),
            "mean_nn_distance": cluster.get("mean_nn_distance"),
            "mean_m3c2_distance": cluster.get("mean_m3c2_distance"),
            "fallback_used": False,
            "warnings": [],
        },
    }


def _cluster_confidence(cluster: dict[str, Any], fallback_used: bool) -> float:
    score = cluster.get("mean_combined_score")
    conf = 0.5 if score is None else float(score)
    conf = float(np.clip(conf, 0.0, 1.0))
    n_points = int(cluster.get("n_points", 0))
    size_factor = float(np.clip(np.sqrt(max(n_points, 1) / 30.0), 0.35, 1.0))
    conf *= size_factor
    if cluster.get("mean_m3c2_distance") is None:
        conf *= 0.85
    if fallback_used:
        conf *= 0.6
    return conf


def _score_cluster_for_task(task: str, cluster: dict[str, Any], priors: dict[str, Any]) -> float:
    score = float(cluster.get("mean_combined_score") or 0.5)
    diameter = float(cluster.get("approximate_diameter") or 0.0)
    compactness = float(cluster.get("compactness") or 1.0)
    linearity = float(cluster.get("linearity") or 1.0)
    n_points = int(cluster.get("n_points", 0))
    m3c2_supported = cluster.get("mean_m3c2_distance") is not None
    dims = np.asarray(cluster.get("bbox_dimensions", [0.0, 0.0, 0.0]), dtype=float)
    length = float(np.max(dims)) if dims.size else 0.0

    if task == "harvest":
        min_d = float(priors.get("min_diameter_m", 0.02))
        max_d = float(priors.get("max_diameter_m", 0.15))
        if min_d <= diameter <= max_d:
            score += 0.35
        else:
            score -= 0.25
        score += 0.2 / max(compactness, 1.0)
        if priors.get("prefer_compact", True):
            score -= min(0.20, max(0.0, compactness - 3.0) * 0.04)
            score -= min(0.15, max(0.0, linearity - 6.0) * 0.02)
        point_weight = float(priors.get("point_count_weight", 0.18))
        point_saturation = max(float(priors.get("point_count_saturation", 60.0)), 1.0)
        score += min(point_weight, np.log1p(max(n_points, 0)) / np.log1p(point_saturation) * point_weight)
        min_points = priors.get("min_points")
        if min_points is not None:
            min_points_f = max(float(min_points), 1.0)
            if n_points < min_points_f:
                score -= min(0.75, (min_points_f - n_points) / min_points_f * 0.75)
        if m3c2_supported:
            score += 0.25
        max_scene_dist = priors.get("max_scene_robust_distance")
        if max_scene_dist is not None and "scene_robust_distance" in cluster:
            scene_dist = float(cluster["scene_robust_distance"])
            if scene_dist > float(max_scene_dist):
                score -= 0.85 + min(0.65, 0.08 * (scene_dist - float(max_scene_dist)))
            else:
                score += 0.10 * (1.0 - scene_dist / max(float(max_scene_dist), 1e-6))
    elif task == "deleaf":
        if length >= float(priors.get("min_length_m", 0.04)):
            score += 0.2
        score += min(linearity / 10.0, 0.4)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return score


def rank_clusters_for_task(
    task: str,
    clusters: list[dict[str, Any]],
    config: dict[str, Any],
    post_xyz_aligned: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    del post_xyz_aligned
    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    priors = config.get("task_priors", {}).get(task, {})
    scored: list[tuple[float, dict[str, Any]]] = []
    for cluster in clusters:
        hard_min_points = priors.get("hard_min_points")
        if hard_min_points is not None and int(cluster.get("n_points", 0)) < int(hard_min_points):
            continue
        hard_max_points = priors.get("hard_max_points")
        if hard_max_points is not None and int(cluster.get("n_points", 0)) > int(hard_max_points):
            continue
        diameter = float(cluster.get("approximate_diameter") or 0.0)
        hard_min_diameter = priors.get("hard_min_diameter_m")
        if hard_min_diameter is not None and diameter < float(hard_min_diameter):
            continue
        hard_max_diameter = priors.get("hard_max_diameter_m")
        if hard_max_diameter is not None and diameter > float(hard_max_diameter):
            continue
        linearity = float(cluster.get("linearity") or 1.0)
        hard_min_linearity = priors.get("hard_min_linearity")
        if hard_min_linearity is not None and linearity < float(hard_min_linearity):
            continue
        hard_max_linearity = priors.get("hard_max_linearity")
        if hard_max_linearity is not None and linearity > float(hard_max_linearity):
            continue

        score = _score_cluster_for_task(task, cluster, priors)
        cluster["target_selection_score"] = float(score)
        scored.append((score, cluster))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cluster for _, cluster in scored]


def _select_clusters(
    task: str,
    clusters: list[dict[str, Any]],
    config: dict[str, Any],
    post_xyz_aligned: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    ranked = rank_clusters_for_task(task, clusters, config, post_xyz_aligned)
    if not ranked:
        return []
    priors = config.get("task_priors", {}).get(task, {})
    min_score = priors.get("min_target_score")
    max_targets = int(priors.get("max_targets", 1))
    if max_targets <= 0:
        max_targets = len(ranked)

    if min_score is not None:
        ranked = [cluster for cluster in ranked if float(cluster["target_selection_score"]) >= float(min_score)]
    return ranked[:max_targets]


def _target_from_cluster(task: str, cluster: dict[str, Any], post_xyz_aligned: np.ndarray) -> dict[str, Any]:
    if task == "harvest":
        return _harvest_target(cluster, post_xyz_aligned)
    if task == "deleaf":
        return _deleaf_target(cluster, post_xyz_aligned)
    raise ValueError(f"Unsupported task: {task}")


def _harvest_target(cluster: dict[str, Any], post_xyz: np.ndarray) -> dict[str, Any]:
    points = np.asarray(cluster["points"], dtype=np.float64)
    center = np.asarray(cluster["centroid"], dtype=np.float64)
    target = _base_target("harvest", cluster, "tomato")
    persistent = nearby_persistent_points(points, post_xyz, margin=0.10)
    fallback_used = False

    pair = nearest_pair(persistent, points)
    if pair is None:
        cut_point = center.copy()
        fallback_used = True
        target["diagnostics"]["warnings"].append("no_persistent_points_near_cluster")
    else:
        _, cut_point, nearest_distance = pair
        target["diagnostics"]["nearest_persistent_distance_m"] = nearest_distance

    approach = normalize(center - cut_point, fallback=np.asarray(cluster["pca_eigenvectors"])[:, -1])
    target["object_center_world"] = center.tolist()
    target["cut_point_world"] = np.asarray(cut_point, dtype=float).tolist()
    target["approach_direction_world"] = approach.tolist()
    target["removal_direction_world"] = approach.tolist()
    target["confidence"] = _cluster_confidence(cluster, fallback_used)
    target["diagnostics"]["fallback_used"] = fallback_used
    return target


def _deleaf_target(cluster: dict[str, Any], post_xyz: np.ndarray) -> dict[str, Any]:
    points = np.asarray(cluster["points"], dtype=np.float64)
    center = np.asarray(cluster["centroid"], dtype=np.float64)
    eigvecs = np.asarray(cluster["pca_eigenvectors"], dtype=np.float64)
    axis = normalize(eigvecs[:, 0])
    end_a, end_b = endpoints_along_axis(points, axis, center)
    persistent = nearby_persistent_points(points, post_xyz, margin=0.08)
    target = _base_target("deleaf", cluster, "leaf")
    fallback_used = False

    if len(persistent):
        dists = nearest_distance_to_set(np.vstack([end_a, end_b]), persistent)
        if dists[0] <= dists[1]:
            cut_point, far_endpoint = end_a, end_b
        else:
            cut_point, far_endpoint = end_b, end_a
        target["diagnostics"]["nearest_endpoint_persistent_distance_m"] = float(np.min(dists))
    else:
        cut_point, far_endpoint = end_a, end_b
        fallback_used = True
        target["diagnostics"]["warnings"].append("no_persistent_points_near_leaf")

    removal = normalize(far_endpoint - cut_point, fallback=axis)
    target["leaf_center_world"] = center.tolist()
    target["object_center_world"] = center.tolist()
    target["cut_point_world"] = np.asarray(cut_point, dtype=float).tolist()
    target["removal_direction_world"] = removal.tolist()
    target["approach_direction_world"] = removal.tolist()
    target["confidence"] = _cluster_confidence(cluster, fallback_used)
    target["diagnostics"]["fallback_used"] = fallback_used
    return target


def infer_targets(
    task: str,
    clusters: list[dict[str, Any]],
    pre_xyz: np.ndarray,
    post_xyz_aligned: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    del pre_xyz
    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    selected = _select_clusters(task, clusters, config, post_xyz_aligned)
    return [_target_from_cluster(task, cluster, post_xyz_aligned) for cluster in selected]


def build_targets_document(task: str, transform_post_to_pre: np.ndarray, targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": task,
        "coordinate_frame": "pre_action_mps_world",
        "method": "aria_mps_semidense_nn_m3c2_dbscan",
        "transforms": {"T_post_to_pre": np.asarray(transform_post_to_pre, dtype=float).tolist()},
        "targets": targets,
    }
