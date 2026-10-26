"""Proxy harvesting cut point estimation from disappeared tomato geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig


LIMITATIONS_TEXT = (
    "No peduncle/keypoint model was used. This is a proxy cut point estimated "
    "from the top surface of the disappeared tomato."
)


def estimate_proxy_cut_points(
    disappearance_results: list[dict[str, Any]],
    pre_instances: list[dict[str, Any]],
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    instance_by_id = {item["instance_id"]: item for item in pre_instances}
    predictions: list[dict[str, Any]] = []
    for result in disappearance_results:
        if result.get("status") != "disappeared":
            continue
        instance = instance_by_id.get(result["pre_instance_id"])
        if instance is None:
            continue
        points = _load_instance_points(instance, out)
        predictions.append(compute_proxy_cut_point(points, instance, result, config))

    predictions.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    path = out / "proxy_cut_points.json"
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions


def compute_proxy_cut_point(
    points: np.ndarray,
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    world_up = normalize_world_up(config.world_up)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    centroid = (
        np.asarray(instance.get("centroid_3d"), dtype=float).reshape(3)
        if instance.get("centroid_3d") is not None
        else np.zeros((3,), dtype=float)
    )
    if len(points):
        centroid = np.median(points, axis=0)
        heights = (points - centroid.reshape(1, 3)) @ world_up
        threshold = float(np.quantile(heights, config.top_quantile))
        top_points = points[heights >= threshold]
    else:
        top_points = np.zeros((0, 3), dtype=float)

    if len(top_points) >= max(3, min(config.min_points_inside_mask, len(points))):
        top_surface_center = np.median(top_points, axis=0)
    else:
        radius = float(instance.get("radius_m", 0.03))
        top_surface_center = centroid + 0.6 * radius * world_up

    cut_point = top_surface_center + config.cut_offset_m * world_up
    confidence = proxy_confidence(instance, disappearance_result)
    return {
        "pre_instance_id": instance["instance_id"],
        "cut_point_type": "proxy_cut_point_from_tomato_top_surface",
        "cut_point_3d_world": [float(v) for v in cut_point],
        "tomato_centroid_3d_world": [float(v) for v in centroid],
        "top_surface_center_3d_world": [float(v) for v in top_surface_center],
        "world_up": [float(v) for v in world_up],
        "cut_offset_m": float(config.cut_offset_m),
        "confidence": float(confidence),
        "limitations": LIMITATIONS_TEXT,
        "disappearance_score": float(disappearance_result.get("final_disappearance_score", 0.0)),
        "visible_post_views": int(disappearance_result.get("visible_post_views", 0)),
        "num_frames_observed_pre": int(instance.get("num_frames_observed", 0)),
        "num_points": int(instance.get("num_points", len(points))),
    }


def build_final_predictions(
    scene_id: str,
    proxy_cut_points: list[dict[str, Any]],
) -> dict[str, Any]:
    removed = []
    for item in proxy_cut_points:
        removed.append(
            {
                "pre_instance_id": item["pre_instance_id"],
                "tomato_centroid_3d_world": item["tomato_centroid_3d_world"],
                "proxy_cut_point_3d_world": item["cut_point_3d_world"],
                "cut_point_type": item["cut_point_type"],
                "disappearance_score": float(item.get("disappearance_score", 0.0)),
                "confidence": float(item.get("confidence", 0.0)),
                "num_frames_observed_pre": int(item.get("num_frames_observed_pre", 0)),
                "visible_post_views": int(item.get("visible_post_views", 0)),
            }
        )
    removed.sort(key=lambda item: float(item["disappearance_score"]), reverse=True)
    return {
        "scene_id": scene_id,
        "method": "pretrained_yolo11_tomato_segmentation_plus_aria_mps_prepost_change",
        "note": (
            "No YOLO fine-tuning. Cut point is proxy from tomato top surface, "
            "not true peduncle detection."
        ),
        "removed_tomatoes": removed,
    }


def normalize_world_up(world_up: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(world_up, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("world_up must be a non-zero vector")
    return vector / norm


def proxy_confidence(instance: dict[str, Any], disappearance_result: dict[str, Any]) -> float:
    normalized_num_frames = min(1.0, float(instance.get("num_frames_observed", 0)) / 5.0)
    normalized_num_points = min(1.0, float(instance.get("num_points", 0)) / 100.0)
    mean_confidence_2d = min(1.0, max(0.0, float(instance.get("mean_confidence_2d", 0.0))))
    disappearance_score = min(
        1.0, max(0.0, float(disappearance_result.get("final_disappearance_score", 0.0)))
    )
    return min(
        1.0,
        0.25 * normalized_num_frames
        + 0.25 * normalized_num_points
        + 0.25 * mean_confidence_2d
        + 0.25 * disappearance_score,
    )


def _load_instance_points(instance: dict[str, Any], out_dir: Path) -> np.ndarray:
    path_value = instance.get("points_3d_path")
    if not path_value:
        return np.zeros((0, 3), dtype=float)
    path = Path(path_value)
    if not path.is_absolute():
        path = out_dir / path
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    return np.asarray(np.load(path), dtype=float).reshape(-1, 3)

