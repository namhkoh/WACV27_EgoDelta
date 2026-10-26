"""Deleafing-specific proxy cut-point estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..fusion.curve_fitting import distance_point_to_line, fit_pca_line
from ..task_config import TaskConfig, task_config_from_dict


LIMITATIONS = (
    "No trained petiole keypoint model was used. This is a proxy based on "
    "promptable petiole segmentation and 3D geometry."
)


def estimate_deleaf_cut_points(
    disappearance_results: list[dict[str, Any]],
    pre_instances: list[dict[str, Any]],
    out_dir: str | Path,
    task_config: TaskConfig,
    main_stem_axis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    instance_by_id = {item["instance_id"]: item for item in pre_instances}
    plant_centroid = estimate_plant_centroid(pre_instances, out)
    predictions: list[dict[str, Any]] = []
    for result in disappearance_results:
        if result.get("status") != "disappeared":
            continue
        instance = instance_by_id.get(result["pre_instance_id"])
        if instance is None:
            continue
        cut = estimate_single_deleaf_cut_point(
            instance=instance,
            disappearance_result=result,
            task_config=task_config,
            main_stem_axis=main_stem_axis,
            plant_centroid=plant_centroid,
        )
        predictions.append(cut)
    predictions.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    (out / "deleaf_cut_points.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions


def build_main_stem_task_config(base: TaskConfig) -> TaskConfig:
    data = base.to_dict()
    data.update(
        {
            "task_name": "deleaf_main_stem",
            "target_semantic_class": "main_stem",
            "text_prompts": [
                "main tomato stem",
                "plant main stem",
            ],
            "negative_prompts": [
                "tomato fruit",
                "leaf blade",
                "greenhouse string",
                "background pipe",
            ],
            "min_mask_area_px": 80,
            "max_mask_area_px": 50000,
            "depth_band_m": max(base.depth_band_m, 0.05),
            "lifting_dbscan_eps_m": max(base.lifting_dbscan_eps_m, 0.04),
            "instance_cluster_eps_m": max(base.instance_cluster_eps_m, 0.08),
            "instance_min_points": max(base.instance_min_points, 12),
            "action_point_type": "main_stem_axis",
        }
    )
    return task_config_from_dict(data)


def estimate_single_deleaf_cut_point(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    task_config: TaskConfig,
    main_stem_axis: dict[str, Any] | None,
    plant_centroid: np.ndarray | None,
) -> dict[str, Any]:
    endpoint_a = np.asarray(instance.get("endpoint_a_3d"), dtype=float).reshape(3)
    endpoint_b = np.asarray(instance.get("endpoint_b_3d"), dtype=float).reshape(3)
    base, tip, main_stem_used = choose_petiole_base(
        endpoint_a,
        endpoint_b,
        main_stem_axis=main_stem_axis,
        plant_centroid=plant_centroid,
    )
    axis = tip - base
    norm = float(np.linalg.norm(axis))
    if norm > 1e-9:
        cut_point = base + 0.01 * axis / norm
    else:
        cut_point = base
    return {
        "target_instance_id": instance["instance_id"],
        "pre_instance_id": instance["instance_id"],
        "task": "deleaf",
        "action_point_type": task_config.action_point_type,
        "cut_point_3d_world": [float(v) for v in cut_point],
        "proxy_cut_point_3d_world": [float(v) for v in cut_point],
        "petiole_centroid_3d_world": instance["centroid_3d"],
        "petiole_base_3d_world": [float(v) for v in base],
        "petiole_tip_3d_world": [float(v) for v in tip],
        "main_stem_used": bool(main_stem_used),
        "confidence": deleaf_confidence(instance, disappearance_result, main_stem_used),
        "limitations": LIMITATIONS,
        "disappearance_score": float(disappearance_result.get("final_disappearance_score", 0.0)),
        "visible_post_views": int(disappearance_result.get("visible_post_views", 0)),
        "num_frames_observed_pre": int(instance.get("num_frames_observed", 0)),
    }


def choose_petiole_base(
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    *,
    main_stem_axis: dict[str, Any] | None,
    plant_centroid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if main_stem_axis is not None:
        line_point = np.asarray(main_stem_axis["point_on_axis"], dtype=float).reshape(3)
        line_dir = np.asarray(main_stem_axis["direction"], dtype=float).reshape(3)
        d_a = distance_point_to_line(endpoint_a, line_point, line_dir)
        d_b = distance_point_to_line(endpoint_b, line_point, line_dir)
        if d_a <= d_b:
            return endpoint_a, endpoint_b, True
        return endpoint_b, endpoint_a, True
    if plant_centroid is not None:
        d_a = float(np.linalg.norm(endpoint_a - plant_centroid))
        d_b = float(np.linalg.norm(endpoint_b - plant_centroid))
        if d_a <= d_b:
            return endpoint_a, endpoint_b, False
        return endpoint_b, endpoint_a, False
    return endpoint_a, endpoint_b, False


def estimate_main_stem_axis_from_instances(instances: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Any] | None:
    points = load_all_instance_points(instances, Path(out_dir))
    if len(points) < 10:
        return None
    fit = fit_pca_line(points)
    return {
        "point_on_axis": [float(v) for v in np.asarray(fit["centroid"], dtype=float)],
        "direction": [float(v) for v in np.asarray(fit["axis"], dtype=float)],
        "source": "promptable_main_stem_or_fallback_points",
    }


def estimate_plant_centroid(instances: list[dict[str, Any]], out_dir: Path) -> np.ndarray | None:
    points = load_all_instance_points(instances, out_dir)
    if len(points) == 0:
        return None
    return np.median(points, axis=0)


def load_all_instance_points(instances: list[dict[str, Any]], out_dir: Path) -> np.ndarray:
    chunks = []
    for instance in instances:
        path_value = instance.get("points_3d_path")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = out_dir / path
        if path.exists():
            chunks.append(np.asarray(np.load(path), dtype=float).reshape(-1, 3))
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(chunks)


def deleaf_confidence(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    main_stem_used: bool,
) -> float:
    return min(
        1.0,
        0.25 * min(1.0, float(instance.get("num_frames_observed", 0)) / 5.0)
        + 0.20 * min(1.0, float(instance.get("num_points", 0)) / 50.0)
        + 0.20 * min(1.0, float(instance.get("mean_confidence_2d", 0.0)))
        + 0.25 * min(1.0, float(disappearance_result.get("final_disappearance_score", 0.0)))
        + (0.10 if main_stem_used else 0.0),
    )


def build_deleaf_final_predictions(
    scene_id: str,
    cut_points: list[dict[str, Any]],
) -> dict[str, Any]:
    removed = []
    for item in cut_points:
        removed.append(
            {
                "pre_instance_id": item["pre_instance_id"],
                "petiole_centroid_3d_world": item["petiole_centroid_3d_world"],
                "proxy_cut_point_3d_world": item["proxy_cut_point_3d_world"],
                "petiole_base_3d_world": item["petiole_base_3d_world"],
                "petiole_tip_3d_world": item["petiole_tip_3d_world"],
                "main_stem_used": bool(item["main_stem_used"]),
                "disappearance_score": float(item["disappearance_score"]),
                "confidence": float(item["confidence"]),
                "num_frames_observed_pre": int(item["num_frames_observed_pre"]),
                "visible_post_views": int(item["visible_post_views"]),
            }
        )
    return {
        "scene_id": scene_id,
        "task": "deleaf",
        "method": "promptable_petiole_segmentation_plus_aria_mps_prepost_change",
        "note": "No training. Petiole cut point is a proxy from 3D petiole geometry.",
        "removed_petioles": removed,
    }
