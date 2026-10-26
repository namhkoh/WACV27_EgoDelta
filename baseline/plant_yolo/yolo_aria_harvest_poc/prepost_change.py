"""Pre/post 3D tomato disappearance detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig
from .mask_lifting import load_detection_mask
from .mps_geometry import MPSGeometry, get_calibration_focal_estimate, project_world_points_to_rgb
from .task_config import TaskConfig


def detect_disappeared_tomatoes(
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    post_frames: list[dict[str, Any]],
    post_predictions: list[dict[str, Any]],
    post_geometry: MPSGeometry,
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    post_prediction_by_frame = {pred["frame_id"]: pred for pred in post_predictions}
    results: list[dict[str, Any]] = []

    for pre_instance in pre_instances:
        nearest_id, nearest_distance = nearest_post_instance(pre_instance, post_instances)
        if nearest_distance is not None and nearest_distance < config.prepost_match_radius_m:
            results.append(
                {
                    "pre_instance_id": pre_instance["instance_id"],
                    "matched_post_instance_id": nearest_id,
                    "nearest_post_distance_m": float(nearest_distance),
                    "visible_post_views": 0,
                    "yolo_absence_score": 0.0,
                    "point_disappearance_score": 0.0,
                    "final_disappearance_score": 0.0,
                    "status": "still_present",
                }
            )
            continue

        visible_views = visible_post_views(pre_instance, post_frames, post_geometry)
        yolo_absence_score = yolo_absence_in_views(
            visible_views=visible_views,
            post_prediction_by_frame=post_prediction_by_frame,
            out_dir=out,
        )
        point_score = point_disappearance_score(
            pre_instance=pre_instance,
            post_points_world=post_geometry.points_world,
            out_dir=out,
            point_match_radius_m=config.point_match_radius_m,
        )
        final_score, score_weights = combine_disappearance_scores(
            yolo_absence_score,
            point_score,
            point_disappearance_weight=config.point_disappearance_weight,
        )
        if len(visible_views) < config.min_visible_post_views:
            status = "unknown_not_enough_post_visibility"
        elif final_score >= config.disappeared_score_thres:
            status = "disappeared"
        else:
            status = "still_present_or_uncertain"

        results.append(
            {
                "pre_instance_id": pre_instance["instance_id"],
                "matched_post_instance_id": None,
                "nearest_post_distance_m": None if nearest_distance is None else float(nearest_distance),
                "visible_post_views": int(len(visible_views)),
                "yolo_absence_score": float(yolo_absence_score),
                "point_disappearance_score": float(point_score),
                "score_weights": score_weights,
                "final_disappearance_score": float(final_score),
                "status": status,
            }
        )

    results.sort(key=lambda item: float(item.get("final_disappearance_score", 0.0)), reverse=True)
    output_path = out / "disappeared_tomatoes.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def detect_disappeared_instances(
    task_config: TaskConfig,
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    post_frames: list[dict[str, Any]],
    post_predictions: list[dict[str, Any]],
    post_geometry: MPSGeometry,
    out_dir: str | Path,
    point_disappearance_weight: float,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    post_prediction_by_frame = {pred["frame_id"]: pred for pred in post_predictions}
    results: list[dict[str, Any]] = []
    matching_post = [
        item
        for item in post_instances
        if item.get("semantic_class") == task_config.target_semantic_class
    ]

    for pre_instance in pre_instances:
        nearest_id, nearest_distance = nearest_post_instance(pre_instance, matching_post)
        if nearest_distance is not None and nearest_distance < task_config.prepost_match_radius_m:
            results.append(
                {
                    "task": task_config.task_name,
                    "pre_instance_id": pre_instance["instance_id"],
                    "matched_post_instance_id": nearest_id,
                    "nearest_post_distance_m": float(nearest_distance),
                    "visible_post_views": 0,
                    "prompt_absence_score": 0.0,
                    "point_disappearance_score": 0.0,
                    "final_disappearance_score": 0.0,
                    "status": "still_present",
                }
            )
            continue

        visible_views = visible_post_views(pre_instance, post_frames, post_geometry)
        absence_score = yolo_absence_in_views(
            visible_views=visible_views,
            post_prediction_by_frame=post_prediction_by_frame,
            out_dir=out,
        )
        point_score = point_disappearance_score(
            pre_instance=pre_instance,
            post_points_world=post_geometry.points_world,
            out_dir=out,
            point_match_radius_m=task_config.point_match_radius_m,
        )
        final_score, score_weights = combine_disappearance_scores(
            absence_score,
            point_score,
            point_disappearance_weight=point_disappearance_weight,
        )
        if len(visible_views) < task_config.min_visible_post_views:
            status = "unknown_not_enough_post_visibility"
        elif final_score >= task_config.disappeared_score_thres:
            status = "disappeared"
        else:
            status = "still_present_or_uncertain"

        results.append(
            {
                "task": task_config.task_name,
                "pre_instance_id": pre_instance["instance_id"],
                "matched_post_instance_id": None,
                "nearest_post_distance_m": None if nearest_distance is None else float(nearest_distance),
                "visible_post_views": int(len(visible_views)),
                "prompt_absence_score": float(absence_score),
                "point_disappearance_score": float(point_score),
                "score_weights": {
                    "prompt_absence_weight": score_weights["yolo_absence_weight"],
                    "point_disappearance_weight": score_weights["point_disappearance_weight"],
                },
                "final_disappearance_score": float(final_score),
                "status": status,
            }
        )

    results.sort(key=lambda item: float(item.get("final_disappearance_score", 0.0)), reverse=True)
    output_name = f"disappeared_{task_config.target_semantic_class}s.json"
    if task_config.task_name == "harvest":
        output_name = "disappeared_tomatoes.json"
    elif task_config.task_name == "deleaf":
        output_name = "disappeared_petioles.json"
    (out / output_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "candidates_debug.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def combine_disappearance_scores(
    yolo_absence_score: float,
    point_disappearance_score_value: float,
    point_disappearance_weight: float,
) -> tuple[float, dict[str, float]]:
    point_weight = min(1.0, max(0.0, float(point_disappearance_weight)))
    yolo_weight = 1.0 - point_weight
    score = yolo_weight * float(yolo_absence_score) + point_weight * float(
        point_disappearance_score_value
    )
    return (
        float(score),
        {
            "yolo_absence_weight": float(yolo_weight),
            "point_disappearance_weight": float(point_weight),
        },
    )


def nearest_post_instance(
    pre_instance: dict[str, Any],
    post_instances: list[dict[str, Any]],
) -> tuple[str | None, float | None]:
    if not post_instances:
        return None, None
    pre_centroid = np.asarray(pre_instance["centroid_3d"], dtype=float)
    post_centroids = np.asarray([item["centroid_3d"] for item in post_instances], dtype=float)
    distances = np.linalg.norm(post_centroids - pre_centroid.reshape(1, 3), axis=1)
    idx = int(np.argmin(distances))
    return str(post_instances[idx]["instance_id"]), float(distances[idx])


def visible_post_views(
    pre_instance: dict[str, Any],
    post_frames: list[dict[str, Any]],
    post_geometry: MPSGeometry,
    min_projected_radius_px: float = 3.0,
) -> list[dict[str, Any]]:
    center = np.asarray(pre_instance["centroid_3d"], dtype=float).reshape(3)
    radius_m = float(pre_instance.get("radius_m", 0.03))
    visible: list[dict[str, Any]] = []
    for frame in post_frames:
        T_world_rgb = post_geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
        uv, depth, valid = project_world_points_to_rgb(
            center.reshape(1, 3),
            T_world_rgb,
            post_geometry.rgb_camera_calibration,
        )
        if not bool(valid[0]):
            continue
        radius_px = estimate_projected_radius_px(
            center, radius_m, T_world_rgb, post_geometry.rgb_camera_calibration, float(depth[0])
        )
        if radius_px < min_projected_radius_px:
            continue
        visible.append(
            {
                "frame_id": frame["frame_id"],
                "timestamp_ns": int(frame["timestamp_ns"]),
                "uv_center": [float(v) for v in uv[0]],
                "projected_radius_px": float(radius_px),
            }
        )
    return visible


def estimate_projected_radius_px(
    center_world: np.ndarray,
    radius_m: float,
    T_world_rgb: np.ndarray,
    calibration: Any,
    depth_m: float,
) -> float:
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [radius_m, 0.0, 0.0],
            [0.0, radius_m, 0.0],
            [0.0, 0.0, radius_m],
        ],
        dtype=float,
    )
    points = center_world.reshape(1, 3) + offsets
    uv, _, valid = project_world_points_to_rgb(points, T_world_rgb, calibration)
    if valid[0] and valid[1:].any():
        distances = np.linalg.norm(uv[1:][valid[1:]] - uv[0].reshape(1, 2), axis=1)
        if len(distances):
            return float(max(1.0, np.max(distances)))
    focal = get_calibration_focal_estimate(calibration)
    if depth_m <= 1e-6:
        return 0.0
    return float(focal * radius_m / depth_m)


def yolo_absence_in_views(
    visible_views: list[dict[str, Any]],
    post_prediction_by_frame: dict[str, dict[str, Any]],
    out_dir: Path,
) -> float:
    if not visible_views:
        return 0.0
    overlap_views = 0
    for view in visible_views:
        prediction = post_prediction_by_frame.get(view["frame_id"])
        if prediction is None or not prediction.get("detections"):
            continue
        if projected_region_overlaps_prediction(view, prediction, out_dir):
            overlap_views += 1
    return float(1.0 - overlap_views / max(1, len(visible_views)))


def projected_region_overlaps_prediction(
    view: dict[str, Any],
    prediction: dict[str, Any],
    out_dir: Path,
) -> bool:
    center = np.asarray(view["uv_center"], dtype=float)
    radius = max(4.0, float(view["projected_radius_px"]))
    for det in prediction.get("detections", []):
        mask = load_detection_mask(det, out_dir)
        h, w = mask.shape[:2]
        region = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            region,
            (int(round(center[0])), int(round(center[1]))),
            int(round(radius)),
            1,
            -1,
        )
        if np.logical_and(mask > 0, region > 0).any():
            return True
    return False


def point_disappearance_score(
    pre_instance: dict[str, Any],
    post_points_world: np.ndarray,
    out_dir: Path,
    point_match_radius_m: float,
) -> float:
    pre_points = _load_instance_points(pre_instance, out_dir)
    post_points = np.asarray(post_points_world, dtype=float).reshape(-1, 3)
    if len(pre_points) == 0 or len(post_points) == 0:
        return 0.0
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, _ = cKDTree(post_points).query(pre_points, k=1)
    except Exception:
        distances = _nearest_distances_numpy(pre_points, post_points)
    fraction_near = float(np.mean(distances <= point_match_radius_m))
    return float(1.0 - fraction_near)


def _nearest_distances_numpy(query: np.ndarray, reference: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    distances: list[np.ndarray] = []
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        dist = np.linalg.norm(chunk[:, None, :] - reference[None, :, :], axis=2)
        distances.append(dist.min(axis=1))
    return np.concatenate(distances, axis=0)


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
