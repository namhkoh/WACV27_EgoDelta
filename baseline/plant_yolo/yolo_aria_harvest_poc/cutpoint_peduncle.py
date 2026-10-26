"""Peduncle-derived cut-point estimation with proxy fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .cutpoint_proxy import compute_proxy_cut_point, proxy_confidence


PEDUNCLE_LIMITATION_TEXT = (
    "Cut point estimated from associated peduncle evidence. This is still a perception "
    "estimate and should be validated against the actual robot tool geometry."
)


def estimate_cut_points_with_mode(
    disappearance_results: list[dict[str, Any]],
    pre_instances: list[dict[str, Any]],
    peduncle_lifted: list[dict[str, Any]],
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    mode = config.cutpoint_mode
    predictions: list[dict[str, Any]] = []
    instance_by_id = {item["instance_id"]: item for item in pre_instances}
    peduncle_by_frame = _peduncles_by_frame(peduncle_lifted)

    for result in disappearance_results:
        if result.get("status") != "disappeared":
            continue
        instance = instance_by_id.get(result["pre_instance_id"])
        if instance is None:
            continue
        tomato_points = _load_instance_points(instance, out)
        peduncle_prediction = None
        if mode in {"peduncle_keypoint", "peduncle_auto"}:
            peduncle_prediction = estimate_from_keypoints(
                instance=instance,
                disappearance_result=result,
                peduncle_candidates=peduncle_lifted,
                out_dir=out,
                config=config,
            )
        if peduncle_prediction is None and mode in {"peduncle_mask", "peduncle_auto"}:
            peduncle_prediction = estimate_from_masks(
                instance=instance,
                disappearance_result=result,
                peduncle_candidates=peduncle_lifted,
                out_dir=out,
                config=config,
            )
        if peduncle_prediction is not None:
            predictions.append(peduncle_prediction)
            continue
        if mode == "proxy" or mode == "peduncle_auto":
            proxy = compute_proxy_cut_point(tomato_points, instance, result, config)
            if mode == "peduncle_auto":
                proxy["fallback_reason"] = "no_associated_peduncle_evidence"
            predictions.append(proxy)
            continue
        predictions.append(_failed_peduncle_prediction(instance, result, mode))

    predictions.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    path = out / "cut_points.json"
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions


def estimate_from_keypoints(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    peduncle_candidates: list[dict[str, Any]],
    out_dir: Path,
    config: PipelineConfig,
) -> dict[str, Any] | None:
    tomato_centroid = np.asarray(instance["centroid_3d"], dtype=float).reshape(3)
    candidates = []
    for peduncle in _candidate_peduncles_for_instance(instance, peduncle_candidates):
        ok_keypoints = [kp for kp in peduncle.get("keypoints_3d", []) if kp.get("status") == "ok"]
        if len(ok_keypoints) < 2:
            continue
        attachment = choose_attachment_keypoint(ok_keypoints, tomato_centroid)
        axis_target = choose_axis_target_keypoint(ok_keypoints, attachment, tomato_centroid)
        if attachment is None or axis_target is None:
            continue
        attachment_xyz = np.asarray(attachment["xyz_world"], dtype=float).reshape(3)
        target_xyz = np.asarray(axis_target["xyz_world"], dtype=float).reshape(3)
        distance = distance_to_tomato_surface(attachment_xyz, instance)
        if distance > config.peduncle_association_max_distance_m:
            continue
        axis = target_xyz - attachment_xyz
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            continue
        axis = axis / norm
        score = association_score(distance, peduncle, instance, config)
        candidates.append((score, peduncle, attachment, axis_target, attachment_xyz, axis, distance))
    if not candidates:
        return None
    score, peduncle, attachment, axis_target, attachment_xyz, axis, distance = max(
        candidates, key=lambda item: item[0]
    )
    cut_point = attachment_xyz + config.peduncle_cut_offset_m * axis
    return _peduncle_prediction_payload(
        instance=instance,
        disappearance_result=disappearance_result,
        cut_point=cut_point,
        source="peduncle_keypoints",
        cut_point_type="peduncle_keypoint_derived_cut_point",
        peduncle=peduncle,
        confidence=peduncle_confidence(instance, disappearance_result, score),
        extra={
            "fruit_attachment_3d_world": [float(v) for v in attachment_xyz],
            "axis_target_keypoint": axis_target["name"],
            "attachment_keypoint": attachment["name"],
            "peduncle_axis_world": [float(v) for v in axis],
            "attachment_distance_to_tomato_surface_m": float(distance),
        },
    )


def estimate_from_masks(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    peduncle_candidates: list[dict[str, Any]],
    out_dir: Path,
    config: PipelineConfig,
) -> dict[str, Any] | None:
    tomato_centroid = np.asarray(instance["centroid_3d"], dtype=float).reshape(3)
    candidates = []
    for peduncle in _candidate_peduncles_for_instance(instance, peduncle_candidates):
        path_value = peduncle.get("mask_points_3d_path")
        if not path_value:
            continue
        points = _load_points(path_value, out_dir)
        if len(points) < config.peduncle_min_points:
            continue
        curve = fit_peduncle_curve_pca(points, tomato_centroid)
        distance = distance_to_tomato_surface(curve["tomato_side_endpoint"], instance)
        if distance > config.peduncle_association_max_distance_m:
            continue
        score = association_score(distance, peduncle, instance, config)
        candidates.append((score, peduncle, curve, distance))
    if not candidates:
        return None
    score, peduncle, curve, distance = max(candidates, key=lambda item: item[0])
    cut_point = curve["tomato_side_endpoint"] + config.peduncle_cut_offset_m * curve["axis_from_tomato"]
    return _peduncle_prediction_payload(
        instance=instance,
        disappearance_result=disappearance_result,
        cut_point=cut_point,
        source="peduncle_mask_curve",
        cut_point_type="peduncle_mask_curve_derived_cut_point",
        peduncle=peduncle,
        confidence=peduncle_confidence(instance, disappearance_result, score),
        extra={
            "fruit_attachment_3d_world": [float(v) for v in curve["tomato_side_endpoint"]],
            "peduncle_axis_world": [float(v) for v in curve["axis_from_tomato"]],
            "stem_side_endpoint_3d_world": [float(v) for v in curve["stem_side_endpoint"]],
            "attachment_distance_to_tomato_surface_m": float(distance),
        },
    )


def fit_peduncle_curve_pca(points: np.ndarray, tomato_centroid: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    center = np.median(points, axis=0)
    centered = points - center.reshape(1, 3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    endpoint_a = center + float(projections.min()) * axis
    endpoint_b = center + float(projections.max()) * axis
    if np.linalg.norm(endpoint_b - tomato_centroid) < np.linalg.norm(endpoint_a - tomato_centroid):
        tomato_side = endpoint_b
        stem_side = endpoint_a
    else:
        tomato_side = endpoint_a
        stem_side = endpoint_b
    axis_from_tomato = stem_side - tomato_side
    norm = float(np.linalg.norm(axis_from_tomato))
    if norm <= 1e-9:
        axis_from_tomato = axis
    else:
        axis_from_tomato = axis_from_tomato / norm
    return {
        "tomato_side_endpoint": tomato_side,
        "stem_side_endpoint": stem_side,
        "axis_from_tomato": axis_from_tomato,
    }


def choose_attachment_keypoint(
    keypoints: list[dict[str, Any]],
    tomato_centroid: np.ndarray,
) -> dict[str, Any] | None:
    named = [
        kp
        for kp in keypoints
        if "attachment" in kp.get("name", "").lower() or "fruit" in kp.get("name", "").lower()
    ]
    candidates = named or keypoints
    return min(
        candidates,
        key=lambda kp: np.linalg.norm(np.asarray(kp["xyz_world"], dtype=float) - tomato_centroid),
        default=None,
    )


def choose_axis_target_keypoint(
    keypoints: list[dict[str, Any]],
    attachment: dict[str, Any] | None,
    tomato_centroid: np.ndarray,
) -> dict[str, Any] | None:
    if attachment is None:
        return None
    attachment_xyz = np.asarray(attachment["xyz_world"], dtype=float)
    preferred = [
        kp
        for kp in keypoints
        if kp is not attachment
        and any(token in kp.get("name", "").lower() for token in ["stem", "cut", "mid", "peduncle"])
    ]
    candidates = preferred or [kp for kp in keypoints if kp is not attachment]
    return max(
        candidates,
        key=lambda kp: np.linalg.norm(np.asarray(kp["xyz_world"], dtype=float) - attachment_xyz),
        default=None,
    )


def distance_to_tomato_surface(point: np.ndarray, instance: dict[str, Any]) -> float:
    centroid = np.asarray(instance["centroid_3d"], dtype=float).reshape(3)
    radius = float(instance.get("radius_m", 0.0))
    return max(0.0, float(np.linalg.norm(point - centroid) - radius))


def association_score(
    attachment_distance_m: float,
    peduncle: dict[str, Any],
    instance: dict[str, Any],
    config: PipelineConfig,
) -> float:
    distance_score = 1.0 - min(1.0, attachment_distance_m / config.peduncle_association_max_distance_m)
    confidence_score = min(1.0, max(0.0, float(peduncle.get("confidence_2d", 0.0))))
    frame_bonus = 1.0 if peduncle.get("frame_id") in set(instance.get("source_frame_ids", [])) else 0.5
    return float(0.55 * distance_score + 0.30 * confidence_score + 0.15 * frame_bonus)


def peduncle_confidence(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    association_score_value: float,
) -> float:
    return min(
        1.0,
        0.30 * association_score_value
        + 0.25 * min(1.0, float(instance.get("num_frames_observed", 0)) / 5.0)
        + 0.20 * min(1.0, float(instance.get("num_points", 0)) / 100.0)
        + 0.25 * min(1.0, float(disappearance_result.get("final_disappearance_score", 0.0))),
    )


def build_final_predictions(
    scene_id: str,
    cut_points: list[dict[str, Any]],
) -> dict[str, Any]:
    removed = []
    for item in cut_points:
        cut_point = item.get("cut_point_3d_world")
        if cut_point is None:
            continue
        removed.append(
            {
                "pre_instance_id": item["pre_instance_id"],
                "tomato_centroid_3d_world": item["tomato_centroid_3d_world"],
                "cut_point_3d_world": cut_point,
                "proxy_cut_point_3d_world": cut_point,
                "cut_point_type": item["cut_point_type"],
                "cut_point_source": item.get("cut_point_source", item.get("source", "proxy")),
                "disappearance_score": float(item.get("disappearance_score", 0.0)),
                "confidence": float(item.get("confidence", 0.0)),
                "num_frames_observed_pre": int(item.get("num_frames_observed_pre", 0)),
                "visible_post_views": int(item.get("visible_post_views", 0)),
            }
        )
    removed.sort(key=lambda item: float(item["disappearance_score"]), reverse=True)
    return {
        "scene_id": scene_id,
        "method": "yolo11_tomato_segmentation_plus_aria_mps_prepost_change_with_configurable_cutpoint",
        "note": (
            "Cut point mode is configurable. Peduncle-derived cut points are used only "
            "when associated peduncle evidence is available; otherwise proxy fallback "
            "may be used depending on --cutpoint-mode."
        ),
        "removed_tomatoes": removed,
    }


def _peduncle_prediction_payload(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    cut_point: np.ndarray,
    source: str,
    cut_point_type: str,
    peduncle: dict[str, Any],
    confidence: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    tomato_centroid = np.asarray(instance["centroid_3d"], dtype=float).reshape(3)
    return {
        "pre_instance_id": instance["instance_id"],
        "cut_point_type": cut_point_type,
        "cut_point_source": source,
        "cut_point_3d_world": [float(v) for v in cut_point],
        "tomato_centroid_3d_world": [float(v) for v in tomato_centroid],
        "associated_peduncle_det_id": peduncle["det_id"],
        "associated_peduncle_frame_id": peduncle["frame_id"],
        "confidence": float(confidence),
        "limitations": PEDUNCLE_LIMITATION_TEXT,
        "disappearance_score": float(disappearance_result.get("final_disappearance_score", 0.0)),
        "visible_post_views": int(disappearance_result.get("visible_post_views", 0)),
        "num_frames_observed_pre": int(instance.get("num_frames_observed", 0)),
        "num_points": int(instance.get("num_points", 0)),
        **extra,
    }


def _failed_peduncle_prediction(
    instance: dict[str, Any],
    disappearance_result: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    return {
        "pre_instance_id": instance["instance_id"],
        "cut_point_type": f"{mode}_unavailable",
        "cut_point_source": mode,
        "cut_point_3d_world": None,
        "tomato_centroid_3d_world": instance["centroid_3d"],
        "confidence": 0.0,
        "limitations": "Requested peduncle cut-point mode, but no associated peduncle evidence was available.",
        "disappearance_score": float(disappearance_result.get("final_disappearance_score", 0.0)),
        "visible_post_views": int(disappearance_result.get("visible_post_views", 0)),
        "num_frames_observed_pre": int(instance.get("num_frames_observed", 0)),
    }


def _candidate_peduncles_for_instance(
    instance: dict[str, Any],
    peduncle_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_frames = set(instance.get("source_frame_ids", []))
    if not source_frames:
        return [item for item in peduncle_candidates if item.get("status") == "ok"]
    same_frame = [
        item
        for item in peduncle_candidates
        if item.get("status") == "ok" and item.get("frame_id") in source_frames
    ]
    return same_frame or [item for item in peduncle_candidates if item.get("status") == "ok"]


def _peduncles_by_frame(peduncle_lifted: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in peduncle_lifted:
        out.setdefault(str(item.get("frame_id")), []).append(item)
    return out


def _load_instance_points(instance: dict[str, Any], out_dir: Path) -> np.ndarray:
    path_value = instance.get("points_3d_path")
    if not path_value:
        return np.zeros((0, 3), dtype=float)
    return _load_points(path_value, out_dir)


def _load_points(path_value: str, out_dir: Path) -> np.ndarray:
    path = Path(path_value)
    if not path.is_absolute():
        path = out_dir / path
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    return np.asarray(np.load(path), dtype=float).reshape(-1, 3)

