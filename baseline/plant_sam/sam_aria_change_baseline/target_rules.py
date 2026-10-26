from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .task_prompts import TASK_PROMPTS


def _load_points(instance: dict[str, Any]) -> np.ndarray:
    path = instance.get("points_3d_path")
    if path and Path(path).exists():
        return np.asarray(np.load(path), dtype=float).reshape((-1, 3))
    centroid = np.asarray(instance.get("centroid_3d", [0.0, 0.0, 0.0]), dtype=float)
    bbox_min = instance.get("bbox_3d_min")
    bbox_max = instance.get("bbox_3d_max")
    if bbox_min is not None and bbox_max is not None:
        return np.vstack([centroid, np.asarray(bbox_min, dtype=float), np.asarray(bbox_max, dtype=float)])
    return centroid.reshape((1, 3))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def _nearest_point_pair(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points_a = np.asarray(points_a, dtype=float).reshape((-1, 3))
    points_b = np.asarray(points_b, dtype=float).reshape((-1, 3))
    if len(points_a) == 0 or len(points_b) == 0:
        zero = np.zeros((3,), dtype=float)
        return zero, zero, float("inf")
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points_b)
        distances, indices = tree.query(points_a, k=1)
        idx = int(np.argmin(distances))
        return points_a[idx], points_b[int(indices[idx])], float(distances[idx])
    except Exception:
        distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
        idx_a, idx_b = np.unravel_index(int(np.argmin(distances)), distances.shape)
        return points_a[idx_a], points_b[idx_b], float(distances[idx_a, idx_b])


def _confidence(change_score: float, multiplier: float = 1.0, base: float = 0.15) -> float:
    return float(np.clip(base + multiplier * float(change_score), 0.0, 1.0))


def _find_instance(instances: list[dict[str, Any]], instance_id: str) -> dict[str, Any] | None:
    for instance in instances:
        if instance.get("instance_id") == instance_id:
            return instance
    return None


def _candidate_aux_instances(
    primary_instance: dict[str, Any],
    auxiliary_instances: list[dict[str, Any]],
    max_distance_m: float,
) -> list[tuple[dict[str, Any], float]]:
    primary_points = _load_points(primary_instance)
    candidates: list[tuple[dict[str, Any], float]] = []
    for aux in auxiliary_instances:
        _, _, distance = _nearest_point_pair(primary_points, _load_points(aux))
        if distance <= max_distance_m:
            candidates.append((aux, distance))
    candidates.sort(
        key=lambda item: (
            item[1],
            -int(item[0].get("num_frames_observed", 0)),
            -float(item[0].get("mean_score_2d", 0.0)),
        )
    )
    return candidates


def _has_active_deleafing_aux_prompt(instance: dict[str, Any]) -> bool:
    active_prompts = {
        str(prompt).lower() for prompt in TASK_PROMPTS.get("deleafing", {}).get("auxiliary", [])
    }
    if not active_prompts:
        return True
    prompts = [str(prompt).lower() for prompt in instance.get("source_prompts", [])]
    if not prompts and instance.get("prompt"):
        prompts = [str(instance["prompt"]).lower()]
    return any(prompt in active_prompts for prompt in prompts)


def _main_stem_geometry_stats(
    instance: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    points = _load_points(instance)
    if len(points) < 2:
        return {
            "is_reliable": False,
            "vertical_extent_m": 0.0,
            "horizontal_extent_m": 0.0,
            "vertical_ratio": 0.0,
        }
    world_up = _normalize(np.asarray(config.world_up, dtype=float))
    center = np.median(points, axis=0)
    centered = points - center.reshape((1, 3))
    vertical_coords = centered @ world_up
    if len(points) >= 5:
        vertical_extent = float(
            np.quantile(vertical_coords, 0.95) - np.quantile(vertical_coords, 0.05)
        )
    else:
        vertical_extent = float(np.max(vertical_coords) - np.min(vertical_coords))
    horizontal_vectors = centered - vertical_coords.reshape((-1, 1)) * world_up.reshape((1, 3))
    horizontal_radius = float(np.quantile(np.linalg.norm(horizontal_vectors, axis=1), 0.90))
    horizontal_extent = max(2.0 * horizontal_radius, 1e-6)
    vertical_ratio = float(vertical_extent / horizontal_extent)
    is_reliable = (
        vertical_extent >= config.deleafing_aux_min_vertical_extent_m
        and vertical_ratio >= config.deleafing_aux_min_vertical_ratio
    )
    return {
        "is_reliable": bool(is_reliable),
        "vertical_extent_m": vertical_extent,
        "horizontal_extent_m": float(horizontal_extent),
        "vertical_ratio": vertical_ratio,
    }


def _reliable_deleafing_main_stem_instances(
    auxiliary_instances: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    reliable: list[dict[str, Any]] = []
    for instance in auxiliary_instances:
        if not _has_active_deleafing_aux_prompt(instance):
            continue
        stats = _main_stem_geometry_stats(instance, config)
        if stats["is_reliable"]:
            reliable.append(instance)
    return reliable


def harvesting_target_rule(
    instance: dict[str, Any],
    candidate: dict[str, Any],
    auxiliary_instances: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, Any]:
    tomato_points = _load_points(instance)
    centroid = np.asarray(instance["centroid_3d"], dtype=float)
    aux_candidates = _candidate_aux_instances(instance, auxiliary_instances, max_distance_m=0.15)
    change_score = float(candidate.get("final_change_score", 0.0))
    if aux_candidates:
        aux = aux_candidates[0][0]
        tomato_point, stem_point, _ = _nearest_point_pair(tomato_points, _load_points(aux))
        cut_point = 0.7 * stem_point + 0.3 * tomato_point
        return {
            "task": "harvesting",
            "target_type": "tomato_harvest",
            "pre_instance_id": instance["instance_id"],
            "object_centroid_3d_world": centroid.tolist(),
            "target_point_3d_world": cut_point.tolist(),
            "target_point_type": "sam_aux_peduncle_or_stem_geometry",
            "used_auxiliary_structure": True,
            "auxiliary_instance_id": aux["instance_id"],
            "confidence": _confidence(change_score, multiplier=0.75, base=0.20),
            "limitations": "Cut point uses SAM auxiliary stem/peduncle geometry and should be verified if masks are noisy.",
        }

    world_up = _normalize(np.asarray(config.world_up, dtype=float))
    centered_heights = (tomato_points - centroid.reshape((1, 3))) @ world_up
    if len(centered_heights) == 0:
        top_surface_center = centroid
    else:
        threshold = float(np.quantile(centered_heights, config.tomato_top_quantile))
        top_points = tomato_points[centered_heights >= threshold]
        if len(top_points) == 0:
            top_points = tomato_points
        top_surface_center = np.median(top_points, axis=0)
    proxy_cut_point = top_surface_center + config.harvest_cut_offset_m * world_up
    return {
        "task": "harvesting",
        "target_type": "tomato_harvest",
        "pre_instance_id": instance["instance_id"],
        "object_centroid_3d_world": centroid.tolist(),
        "target_point_3d_world": proxy_cut_point.tolist(),
        "target_point_type": "proxy_cut_point_from_tomato_top_surface",
        "used_auxiliary_structure": False,
        "confidence": _confidence(change_score, multiplier=0.50, base=0.10),
        "limitations": (
            "No reliable peduncle/stem mask was available; target point is a proxy and is "
            "not a true peduncle cut point."
        ),
    }


def _pca_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) == 0:
        zero = np.zeros((3,), dtype=float)
        return zero, zero
    if len(points) == 1:
        return points[0], points[0]
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    coords = centered @ axis
    return points[int(np.argmin(coords))], points[int(np.argmax(coords))]


def deleafing_target_rule(
    instance: dict[str, Any],
    candidate: dict[str, Any],
    auxiliary_instances: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, Any]:
    petiole_points = _load_points(instance)
    change_score = float(candidate.get("final_change_score", 0.0))
    reliable_auxiliary_instances = _reliable_deleafing_main_stem_instances(
        auxiliary_instances, config
    )
    aux_candidates = _candidate_aux_instances(
        instance, reliable_auxiliary_instances, max_distance_m=0.12
    )
    if aux_candidates:
        aux_instance = aux_candidates[0][0]
        stem_points = _load_points(aux_instance)
        petiole_point, _, _ = _nearest_point_pair(petiole_points, stem_points)
        stem_stats = _main_stem_geometry_stats(aux_instance, config)
        return {
            "task": "deleafing",
            "target_type": "petiole_cut",
            "pre_instance_id": instance["instance_id"],
            "target_point_3d_world": petiole_point.tolist(),
            "target_point_type": "petiole_endpoint_near_main_stem",
            "used_auxiliary_structure": True,
            "auxiliary_instance_id": aux_instance["instance_id"],
            "auxiliary_main_stem_vertical_extent_m": stem_stats["vertical_extent_m"],
            "auxiliary_main_stem_vertical_ratio": stem_stats["vertical_ratio"],
            "confidence": _confidence(change_score, multiplier=0.70, base=0.18),
            "limitations": "Target uses nearest petiole point to auxiliary main-stem geometry.",
        }
    endpoint_a, endpoint_b = _pca_endpoints(petiole_points)
    return {
        "task": "deleafing",
        "target_type": "petiole_cut",
        "pre_instance_id": instance["instance_id"],
        "target_point_3d_world": endpoint_a.tolist(),
        "target_point_type": "ambiguous_petiole_endpoint",
        "candidate_target_points_3d_world": [endpoint_a.tolist(), endpoint_b.tolist()],
        "used_auxiliary_structure": False,
        "confidence": _confidence(change_score, multiplier=0.35, base=0.08),
        "limitations": (
            "No reliable vertical main-stem geometry was available; both petiole endpoints are "
            "output as candidates."
        ),
    }


def generate_task_predictions(
    change_candidates: list[dict[str, Any]],
    pre_instances: list[dict[str, Any]],
    config: PipelineConfig,
    task: str,
) -> list[dict[str, Any]]:
    auxiliary_instances = [
        instance for instance in pre_instances if instance.get("semantic_group") == "auxiliary"
    ]
    disappeared = [
        candidate for candidate in change_candidates if candidate.get("status") == "disappeared"
    ]
    disappeared.sort(key=lambda item: float(item.get("final_change_score", 0.0)), reverse=True)
    predictions: list[dict[str, Any]] = []
    for candidate in disappeared:
        instance = _find_instance(pre_instances, candidate["pre_instance_id"])
        if instance is None:
            continue
        if task == "harvesting":
            pred = harvesting_target_rule(instance, candidate, auxiliary_instances, config)
        elif task == "deleafing":
            pred = deleafing_target_rule(instance, candidate, auxiliary_instances, config)
        else:
            raise ValueError(f"Unsupported task {task!r}.")
        pred["final_change_score"] = float(candidate.get("final_change_score", 0.0))
        pred["visible_post_views"] = int(candidate.get("visible_post_views", 0))
        pred["source_prompts"] = list(instance.get("source_prompts", []))
        predictions.append(pred)
    return predictions


def run_target_rules(
    pre_instances_json: str | Path,
    change_candidates_json: str | Path,
    config: PipelineConfig,
    task: str,
    out_dir: str | Path,
    method_metadata: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    with Path(pre_instances_json).open("r") as handle:
        pre_instances = json.load(handle)
    with Path(change_candidates_json).open("r") as handle:
        change_candidates = json.load(handle)
    predictions = generate_task_predictions(change_candidates, pre_instances, config, task)
    final = {
        "method": "sam3_concept_masks_plus_aria_mps_geometry",
        "task": task,
        **method_metadata,
        "notes": [
            "No SAM fine-tuning was used.",
            "2D SAM masks were lifted into 3D using Aria MPS semi-dense points.",
            "Pre/post instances were compared in Aria world coordinates.",
        ],
        "predictions": predictions,
        "debug_stats_path": "debug_stats.json",
    }
    output_path = Path(out_dir) / "final_predictions.json"
    with output_path.open("w") as handle:
        json.dump(final, handle, indent=2)
    stats = {"num_final_predictions": int(len(predictions))}
    return output_path, final, stats
