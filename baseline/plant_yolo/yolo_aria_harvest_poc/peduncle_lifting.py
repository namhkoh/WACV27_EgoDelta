"""Lift peduncle masks/keypoints into 3D using Aria MPS points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, relpath
from .mask_lifting import largest_dbscan_cluster, load_detection_mask, mask_contains_uv
from .mps_geometry import MPSGeometry, project_world_points_to_rgb


def lift_peduncle_detections_for_split(
    split: str,
    frames: list[dict[str, Any]],
    peduncle_predictions: list[dict[str, Any]],
    geometry: MPSGeometry,
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    out = Path(out_dir)
    points_dir = out / "lifted_3d" / "peduncle_points"
    points_dir.mkdir(parents=True, exist_ok=True)
    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    lifted: list[dict[str, Any]] = []

    for frame_pred in tqdm(peduncle_predictions, desc=f"Lift peduncle {split}", unit="frame"):
        frame = frame_by_id.get(frame_pred["frame_id"])
        if frame is None:
            continue
        T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
        uv, depth, valid_projection = project_world_points_to_rgb(
            geometry.points_world,
            T_world_rgb,
            geometry.rgb_camera_calibration,
        )
        for detection in frame_pred.get("detections", []):
            lifted.append(
                lift_single_peduncle_detection(
                    split=split,
                    frame_id=frame["frame_id"],
                    detection=detection,
                    points_world=geometry.points_world,
                    uv=uv,
                    depth=depth,
                    valid_projection=valid_projection,
                    out_dir=out,
                    points_dir=points_dir,
                    config=config,
                )
            )

    output_path = out / "lifted_3d" / f"{split}_peduncle_lifted_detections.json"
    output_path.write_text(json.dumps(lifted, indent=2), encoding="utf-8")
    return lifted


def lift_single_peduncle_detection(
    split: str,
    frame_id: str,
    detection: dict[str, Any],
    points_world: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_projection: np.ndarray,
    out_dir: Path,
    points_dir: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    lifted: dict[str, Any] = {
        "det_id": detection["det_id"],
        "frame_id": frame_id,
        "split": split,
        "confidence_2d": float(detection.get("confidence", 0.0)),
        "bbox_xyxy": detection.get("bbox_xyxy"),
        "keypoints_3d": [],
        "mask_points_3d_path": None,
        "num_mask_points": 0,
        "mask_centroid_3d": None,
        "mask_bbox_3d_min": None,
        "mask_bbox_3d_max": None,
        "status": "no_3d_support",
        "warnings": [],
    }

    for keypoint in detection.get("keypoints_2d", []):
        if not keypoint.get("visible", True):
            continue
        lifted_keypoint = lift_keypoint_2d(
            keypoint=keypoint,
            points_world=points_world,
            uv=uv,
            depth=depth,
            valid_projection=valid_projection,
            config=config,
        )
        lifted["keypoints_3d"].append(lifted_keypoint)

    if detection.get("mask_path") or detection.get("mask"):
        mask = load_detection_mask(detection, out_dir)
        mask_points = lift_peduncle_mask_points(
            mask=mask,
            points_world=points_world,
            uv=uv,
            depth=depth,
            valid_projection=valid_projection,
            config=config,
        )
        if len(mask_points) >= config.peduncle_min_points:
            points_path = points_dir / f"{detection['det_id']}_mask_points.npy"
            np.save(points_path, mask_points.astype(np.float32))
            lifted.update(
                {
                    "mask_points_3d_path": relpath(points_path, out_dir),
                    "num_mask_points": int(len(mask_points)),
                    "mask_centroid_3d": [float(v) for v in np.median(mask_points, axis=0)],
                    "mask_bbox_3d_min": [float(v) for v in mask_points.min(axis=0)],
                    "mask_bbox_3d_max": [float(v) for v in mask_points.max(axis=0)],
                }
            )

    ok_keypoints = [kp for kp in lifted["keypoints_3d"] if kp.get("status") == "ok"]
    if ok_keypoints or lifted["mask_points_3d_path"] is not None:
        lifted["status"] = "ok"
    return lifted


def lift_keypoint_2d(
    keypoint: dict[str, Any],
    points_world: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_projection: np.ndarray,
    config: PipelineConfig,
) -> dict[str, Any]:
    xy = np.asarray(keypoint["xy"], dtype=float).reshape(2)
    distances_px = np.linalg.norm(uv - xy.reshape(1, 2), axis=1)
    near = valid_projection & np.isfinite(distances_px) & (
        distances_px <= config.peduncle_keypoint_lift_radius_px
    )
    points_near = points_world[near]
    depths_near = depth[near]
    if len(points_near) < config.peduncle_min_points:
        return {
            **keypoint,
            "xyz_world": None,
            "num_support_points": int(len(points_near)),
            "status": "no_3d_support",
        }
    z_front = float(np.quantile(depths_near, 0.20))
    foreground = depths_near <= z_front + config.peduncle_depth_band_m
    points_front = points_near[foreground]
    if len(points_front) < config.peduncle_min_points:
        points_front = points_near
    xyz = np.median(points_front, axis=0)
    return {
        **keypoint,
        "xyz_world": [float(v) for v in xyz],
        "num_support_points": int(len(points_front)),
        "status": "ok",
    }


def lift_peduncle_mask_points(
    mask: np.ndarray,
    points_world: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_projection: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    inside = valid_projection & mask_contains_uv(mask, uv)
    points_inside = points_world[inside]
    depths_inside = depth[inside]
    if len(points_inside) < config.peduncle_min_points:
        return np.zeros((0, 3), dtype=float)
    z_front = float(np.quantile(depths_inside, 0.20))
    foreground = depths_inside <= z_front + config.peduncle_depth_band_m
    points_front = points_inside[foreground]
    if len(points_front) < config.peduncle_min_points:
        return np.zeros((0, 3), dtype=float)
    cluster_points, _, _, _ = largest_dbscan_cluster(
        points_front,
        eps=config.peduncle_dbscan_eps_m,
        min_samples=config.peduncle_dbscan_min_samples,
    )
    if len(cluster_points) < config.peduncle_min_points:
        return points_front
    return cluster_points

