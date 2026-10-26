"""Lift 2D YOLO tomato masks into 3D using projected Aria MPS points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, relpath, should_write_visualization
from .mps_geometry import MPSGeometry, project_world_points_to_rgb
from .visualization import draw_lifting_debug


def mask_contains_uv(mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Return whether each floating pixel coordinate falls inside a binary mask."""

    mask = np.asarray(mask)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    height, width = mask.shape[:2]
    contained = np.zeros((len(uv),), dtype=bool)
    finite = np.isfinite(uv).all(axis=1)
    if not finite.any():
        return contained
    x = np.floor(uv[finite, 0]).astype(int)
    y = np.floor(uv[finite, 1]).astype(int)
    in_bounds = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    finite_indices = np.where(finite)[0]
    valid_indices = finite_indices[in_bounds]
    contained[valid_indices] = mask[y[in_bounds], x[in_bounds]] > 0
    return contained


def lift_detections_for_split(
    split: str,
    frames: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    geometry: MPSGeometry,
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for mask lifting debug images.") from exc

    out = Path(out_dir)
    points_dir = out / "lifted_3d" / "points"
    debug_dir = out / "visualizations" / "lifting_debug" / split
    points_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    lifted: list[dict[str, Any]] = []

    for frame_index, frame_pred in enumerate(tqdm(predictions, desc=f"Lift {split}", unit="frame")):
        frame = frame_by_id.get(frame_pred["frame_id"])
        if frame is None:
            continue
        T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
        uv, depth, valid_projection = project_world_points_to_rgb(
            geometry.points_world,
            T_world_rgb,
            geometry.rgb_camera_calibration,
        )
        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((frame["original_height"], frame["original_width"], 3), dtype=np.uint8)

        for det in frame_pred.get("detections", []):
            mask = load_detection_mask(det, out)
            lifted_det, uv_inside, uv_foreground = lift_single_detection(
                split=split,
                frame_id=frame["frame_id"],
                detection=det,
                mask=mask,
                points_world=geometry.points_world,
                uv=uv,
                depth=depth,
                valid_projection=valid_projection,
                out_dir=out,
                points_dir=points_dir,
                config=config,
            )
            lifted.append(lifted_det)
            if should_write_visualization(frame_index, config.visualization_stride):
                debug = draw_lifting_debug(image, mask, uv_inside, uv_foreground)
                cv2.imwrite(str(debug_dir / f"{det['det_id']}.jpg"), debug)

    output_path = out / "lifted_3d" / f"{split}_lifted_detections.json"
    output_path.write_text(json.dumps(lifted, indent=2), encoding="utf-8")
    return lifted


def lift_single_detection(
    split: str,
    frame_id: str,
    detection: dict[str, Any],
    mask: np.ndarray,
    points_world: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_projection: np.ndarray,
    out_dir: Path,
    points_dir: Path,
    config: PipelineConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    inside = valid_projection & mask_contains_uv(mask, uv)
    uv_inside = uv[inside]
    points_inside = points_world[inside]
    depths_inside = depth[inside]

    base = {
        "det_id": detection["det_id"],
        "frame_id": frame_id,
        "split": split,
        "confidence_2d": float(detection.get("confidence", 0.0)),
        "num_points_inside_mask": int(len(points_inside)),
        "num_points_after_depth_filter": 0,
        "num_points_after_cluster": 0,
        "centroid_3d": None,
        "bbox_3d_min": None,
        "bbox_3d_max": None,
        "points_3d_path": None,
        "status": "no_3d_support",
        "warnings": [],
    }

    if len(points_inside) < config.min_points_inside_mask:
        return base, uv_inside, np.zeros((0, 2), dtype=float)

    z_front = float(np.quantile(depths_inside, 0.15))
    foreground = depths_inside <= z_front + config.tomato_depth_band_m
    points_foreground = points_inside[foreground]
    uv_foreground = uv_inside[foreground]
    base["num_points_after_depth_filter"] = int(len(points_foreground))

    if len(points_foreground) < config.min_points_inside_mask:
        return base, uv_inside, uv_foreground

    cluster_points, cluster_mask, status, warning = largest_dbscan_cluster(
        points_foreground,
        eps=config.tomato_dbscan_eps_m,
        min_samples=config.tomato_dbscan_min_samples,
    )
    uv_cluster = uv_foreground[cluster_mask] if len(cluster_mask) == len(uv_foreground) else uv_foreground
    base["status"] = status
    if warning:
        base["warnings"].append(warning)

    base["num_points_after_cluster"] = int(len(cluster_points))
    if len(cluster_points) == 0:
        base["status"] = "no_3d_support"
        return base, uv_inside, uv_cluster

    centroid = np.median(cluster_points, axis=0)
    bbox_min = cluster_points.min(axis=0)
    bbox_max = cluster_points.max(axis=0)
    points_path = points_dir / f"{detection['det_id']}.npy"
    np.save(points_path, cluster_points.astype(np.float32))

    base.update(
        {
            "centroid_3d": [float(v) for v in centroid],
            "bbox_3d_min": [float(v) for v in bbox_min],
            "bbox_3d_max": [float(v) for v in bbox_max],
            "points_3d_path": relpath(points_path, out_dir),
        }
    )
    return base, uv_inside, uv_cluster


def largest_dbscan_cluster(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points, np.zeros((0,), dtype=bool), "no_3d_support", None
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    except Exception:
        return (
            points,
            np.ones((len(points),), dtype=bool),
            "dbscan_failed",
            "DBSCAN unavailable or failed; kept all depth-filtered points.",
        )

    non_noise = labels[labels >= 0]
    if len(non_noise) == 0:
        return (
            points,
            np.ones((len(points),), dtype=bool),
            "dbscan_failed",
            "DBSCAN found no non-noise cluster; kept all depth-filtered points.",
        )
    unique, counts = np.unique(non_noise, return_counts=True)
    keep_label = int(unique[np.argmax(counts)])
    keep = labels == keep_label
    return points[keep], keep, "ok", None


def load_detection_mask(detection: dict[str, Any], out_dir: str | Path) -> np.ndarray:
    path_value = detection.get("mask_path") or detection.get("mask")
    if path_value is None:
        raise ValueError(f"Detection {detection.get('det_id')} is missing mask_path")
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(out_dir) / path
    data = np.load(path)
    if "mask" not in data:
        raise ValueError(f"Mask file {path} does not contain a 'mask' array")
    return np.asarray(data["mask"], dtype=bool)
