from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .io_aria import load_frames_metadata
from .mask_postprocess import load_mask
from .mps_geometry import AriaMpsGeometry


def _load_points(instance: dict[str, Any]) -> np.ndarray:
    path = instance.get("points_3d_path")
    if path and Path(path).exists():
        return np.asarray(np.load(path), dtype=float).reshape((-1, 3))
    centroid = instance.get("centroid_3d")
    if centroid is None:
        return np.empty((0, 3), dtype=float)
    return np.asarray(centroid, dtype=float).reshape((1, 3))


def _nearest_instance(
    pre_instance: dict[str, Any],
    post_instances: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float | None]:
    if not post_instances:
        return None, None
    pre_centroid = np.asarray(pre_instance["centroid_3d"], dtype=float)
    distances = [
        float(np.linalg.norm(pre_centroid - np.asarray(post_instance["centroid_3d"], dtype=float)))
        for post_instance in post_instances
    ]
    idx = int(np.argmin(distances))
    return post_instances[idx], distances[idx]


def _project_instance_to_frame(
    instance: dict[str, Any],
    frame: dict[str, Any],
    geometry: AriaMpsGeometry,
) -> tuple[bool, np.ndarray, float]:
    center = np.asarray(instance["centroid_3d"], dtype=float)
    radius = max(float(instance.get("radius_m", 0.01)), 0.01)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=float,
    )
    points = center.reshape((1, 3)) + offsets
    uv, _, valid = geometry.project_world_points_to_frame(
        points,
        int(frame["timestamp_ns"]),
        int(frame["width"]),
        int(frame["height"]),
    )
    if not valid[0]:
        return False, uv[0], 0.0
    valid_offsets = uv[valid]
    if len(valid_offsets) <= 1:
        radius_px = 2.0
    else:
        radius_px = float(np.max(np.linalg.norm(valid_offsets - uv[0].reshape((1, 2)), axis=1)))
    return radius_px >= 1.0, uv[0], radius_px


def visible_post_frames(
    instance: dict[str, Any],
    post_frames: list[dict[str, Any]],
    post_geometry: AriaMpsGeometry,
) -> list[tuple[dict[str, Any], np.ndarray, float]]:
    visible: list[tuple[dict[str, Any], np.ndarray, float]] = []
    for frame in post_frames:
        is_visible, uv, radius_px = _project_instance_to_frame(instance, frame, post_geometry)
        if is_visible:
            visible.append((frame, uv, radius_px))
    return visible


def _sam_absence_score(
    visible_frames: list[tuple[dict[str, Any], np.ndarray, float]],
    post_predictions: list[dict[str, Any]] | None,
) -> float:
    if not visible_frames:
        return 0.0
    if not post_predictions:
        return 1.0
    prediction_by_frame = {item["frame_id"]: item for item in post_predictions}
    absent = 0
    checked = 0
    for frame, uv, radius_px in visible_frames:
        pred = prediction_by_frame.get(frame["frame_id"])
        if pred is None:
            continue
        checked += 1
        x = int(round(float(uv[0])))
        y = int(round(float(uv[1])))
        radius = int(max(4.0, radius_px * 1.5))
        x1, x2 = max(0, x - radius), min(int(frame["width"]), x + radius + 1)
        y1, y2 = max(0, y - radius), min(int(frame["height"]), y + radius + 1)
        has_overlap = False
        for mask_record in pred.get("masks", []):
            if mask_record.get("semantic_group", "primary") != "primary":
                continue
            try:
                mask = load_mask(mask_record)
            except Exception:
                continue
            if mask[y1:y2, x1:x2].any():
                has_overlap = True
                break
        if not has_overlap:
            absent += 1
    if checked == 0:
        return 1.0
    return float(absent / checked)


def _point_disappearance_score(
    pre_instance: dict[str, Any],
    post_points_world: np.ndarray | None,
    match_radius_m: float,
) -> float:
    pre_points = _load_points(pre_instance)
    if len(pre_points) == 0:
        return 1.0
    if post_points_world is None or len(post_points_world) == 0:
        return 1.0
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(post_points_world, dtype=float).reshape((-1, 3)))
        distances, _ = tree.query(pre_points, k=1)
    except Exception:
        post_points = np.asarray(post_points_world, dtype=float).reshape((-1, 3))
        distances = np.min(
            np.linalg.norm(pre_points[:, None, :] - post_points[None, :, :], axis=2), axis=1
        )
    fraction_near = float(np.mean(distances <= match_radius_m))
    return float(1.0 - fraction_near)


def detect_prepost_changes(
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    config: PipelineConfig,
    task: str,
    post_frames: list[dict[str, Any]] | None = None,
    post_geometry: AriaMpsGeometry | None = None,
    post_predictions: list[dict[str, Any]] | None = None,
    post_points_world: np.ndarray | None = None,
    point_disappearance_reliable: bool = True,
) -> list[dict[str, Any]]:
    pre_primary = [item for item in pre_instances if item.get("semantic_group", "primary") == "primary"]
    post_primary = [item for item in post_instances if item.get("semantic_group", "primary") == "primary"]
    match_radius = config.prepost_match_radius_for_task(task)
    disappeared_score_thres = config.disappeared_score_threshold_for_task(task)
    results: list[dict[str, Any]] = []
    for pre_instance in pre_primary:
        same_class_posts = [
            item
            for item in post_primary
            if item.get("target_class") == pre_instance.get("target_class")
            or pre_instance.get("target_class") is None
        ]
        nearest, distance = _nearest_instance(pre_instance, same_class_posts)
        base = {
            "pre_instance_id": pre_instance["instance_id"],
            "matched_post_instance_id": nearest["instance_id"] if nearest is not None else None,
            "nearest_post_distance_m": distance,
            "visible_post_views": 0,
            "sam_absence_score": 0.0,
            "point_disappearance_score": 0.0,
            "final_change_score": 0.0,
            "change_score_mode": config.change_score_mode,
            "point_disappearance_reliable": bool(point_disappearance_reliable),
            "status": "unknown",
        }
        if distance is not None and distance < match_radius:
            base["status"] = "still_present"
            results.append(base)
            continue

        if "visible_post_views_override" in pre_instance:
            visible_count = int(pre_instance["visible_post_views_override"])
            visible_frames_for_score: list[tuple[dict[str, Any], np.ndarray, float]] = []
        elif post_frames is not None and post_geometry is not None:
            visible_frames_for_score = visible_post_frames(pre_instance, post_frames, post_geometry)
            visible_count = len(visible_frames_for_score)
        else:
            visible_frames_for_score = []
            visible_count = 0

        base["visible_post_views"] = int(visible_count)
        if visible_count < config.min_visible_post_views:
            base["status"] = "unknown_not_enough_post_visibility"
            results.append(base)
            continue

        if visible_frames_for_score:
            sam_absence = _sam_absence_score(visible_frames_for_score, post_predictions)
        else:
            sam_absence = 1.0
        point_disappearance = _point_disappearance_score(
            pre_instance, post_points_world, config.point_match_radius_m
        )
        score_mode = config.change_score_mode
        if score_mode == "auto":
            score_mode = "average" if point_disappearance_reliable else "sam_absence_only"
        if score_mode == "sam_absence_only":
            final_score = sam_absence
        elif score_mode == "point_disappearance_only":
            final_score = point_disappearance
        elif score_mode == "average":
            final_score = 0.5 * sam_absence + 0.5 * point_disappearance
        else:
            raise ValueError(f"Unsupported change_score_mode {config.change_score_mode!r}.")
        base.update(
            {
                "sam_absence_score": float(sam_absence),
                "point_disappearance_score": float(point_disappearance),
                "final_change_score": float(final_score),
                "change_score_mode": score_mode,
                "status": (
                    "disappeared"
                    if final_score >= disappeared_score_thres
                    else "still_present_or_uncertain"
                ),
            }
        )
        results.append(base)
    return results


def run_prepost_change(
    pre_instances_json: str | Path,
    post_instances_json: str | Path,
    post_frames_json: str | Path,
    post_geometry: AriaMpsGeometry,
    post_predictions_json: str | Path,
    config: PipelineConfig,
    task: str,
    out_dir: str | Path,
    point_disappearance_reliable: bool = True,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    with Path(pre_instances_json).open("r") as handle:
        pre_instances = json.load(handle)
    with Path(post_instances_json).open("r") as handle:
        post_instances = json.load(handle)
    post_frames = load_frames_metadata(post_frames_json)
    predictions_path = Path(post_predictions_json)
    if predictions_path.exists():
        with predictions_path.open("r") as handle:
            post_predictions = json.load(handle)
    else:
        post_predictions = []
    post_points = post_geometry.load_filtered_points()
    changes = detect_prepost_changes(
        pre_instances=pre_instances,
        post_instances=post_instances,
        config=config,
        task=task,
        post_frames=post_frames,
        post_geometry=post_geometry,
        post_predictions=post_predictions,
        post_points_world=post_points,
        point_disappearance_reliable=point_disappearance_reliable,
    )
    output_path = Path(out_dir) / "change_candidates.json"
    with output_path.open("w") as handle:
        json.dump(changes, handle, indent=2)
    stats = {
        "num_candidates": int(len(changes)),
        "num_disappeared_candidates": int(sum(1 for item in changes if item.get("status") == "disappeared")),
        "num_unknown_visibility": int(
            sum(1 for item in changes if item.get("status") == "unknown_not_enough_post_visibility")
        ),
    }
    return output_path, changes, stats
