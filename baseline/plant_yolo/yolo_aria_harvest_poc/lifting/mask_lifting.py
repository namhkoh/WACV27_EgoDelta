"""Task-generic mask-to-3D lifting using Aria MPS points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..config import PipelineConfig, relpath, should_write_visualization
from ..mask_lifting import largest_dbscan_cluster, load_detection_mask, mask_contains_uv
from ..mps_geometry import MPSGeometry, project_world_points_to_rgb
from ..task_config import TaskConfig
from ..visualization import draw_lifting_debug
from .point_filters import front_surface_filter

MAX_DBSCAN_POINTS = 8000


def lift_segmentation_for_split(
    split: str,
    frames: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    geometry: MPSGeometry,
    out_dir: str | Path,
    task_config: TaskConfig,
    pipeline_config: PipelineConfig,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for mask lifting debug images.") from exc

    out = Path(out_dir)
    points_dir = out / "lifted_3d" / task_config.task_name / "points"
    debug_dir = out / "visualizations" / task_config.task_name / f"lifting_debug_{split}"
    points_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    work_items = [
        (frame_index, frame_pred, frame_by_id.get(frame_pred["frame_id"]))
        for frame_index, frame_pred in enumerate(predictions)
    ]
    lifted_by_frame: list[list[dict[str, Any]]] = [[] for _ in predictions]
    desc = f"Lift {task_config.task_name} {split}"

    for frame_index, frame_pred, frame in tqdm(work_items, desc=desc, unit="frame"):
        _, frame_lifted = _lift_segmentation_frame(
            split,
            frame_index,
            frame_pred,
            frame,
            geometry,
            out,
            points_dir,
            debug_dir,
            task_config,
            pipeline_config,
        )
        lifted_by_frame[frame_index] = frame_lifted

    lifted = [det for frame_lifted in lifted_by_frame for det in frame_lifted]

    output_path = out / "lifted_3d" / task_config.task_name / f"{split}_lifted_detections.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lifted, indent=2), encoding="utf-8")
    return lifted


def _lift_segmentation_frame(
    split: str,
    frame_index: int,
    frame_pred: dict[str, Any],
    frame: dict[str, Any] | None,
    geometry: MPSGeometry,
    out: Path,
    points_dir: Path,
    debug_dir: Path,
    task_config: TaskConfig,
    pipeline_config: PipelineConfig,
) -> tuple[int, list[dict[str, Any]]]:
    if frame is None:
        return frame_index, []

    import cv2  # type: ignore

    T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
    uv, depth, valid_projection = project_world_points_to_rgb(
        geometry.points_world,
        T_world_rgb,
        geometry.rgb_camera_calibration,
    )

    write_debug = should_write_visualization(frame_index, pipeline_config.visualization_stride)
    image = None
    if write_debug:
        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((frame["original_height"], frame["original_width"], 3), dtype=np.uint8)

    frame_lifted: list[dict[str, Any]] = []
    for det in frame_pred.get("detections", []):
        mask = load_detection_mask(det, out)
        lifted_det, uv_inside, uv_foreground = lift_single_segmentation_detection(
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
            task_config=task_config,
        )
        frame_lifted.append(lifted_det)
        if write_debug and image is not None:
            debug = draw_lifting_debug(image, mask, uv_inside, uv_foreground)
            cv2.imwrite(str(debug_dir / f"{det['det_id']}.jpg"), debug)

    return frame_index, frame_lifted


def lift_single_segmentation_detection(
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
    task_config: TaskConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    inside = projected_points_inside_detection_mask(mask, uv, valid_projection, detection.get("bbox_xyxy"))
    uv_inside = uv[inside]
    points_inside = points_world[inside]
    depths_inside = depth[inside]
    min_points_inside = max(3, int(task_config.lifting_dbscan_min_samples))

    base = {
        "det_id": detection["det_id"],
        "frame_id": frame_id,
        "split": split,
        "semantic_class": detection.get("semantic_class", task_config.target_semantic_class),
        "confidence_2d": float(detection.get("confidence", 0.0)),
        "prompt_used": detection.get("prompt_used"),
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

    if len(points_inside) < min_points_inside:
        return base, uv_inside, np.zeros((0, 2), dtype=float)

    points_foreground, foreground_mask = front_surface_filter(
        points_inside,
        depths_inside,
        depth_band_m=task_config.depth_band_m,
    )
    uv_foreground = uv_inside[foreground_mask]
    base["num_points_after_depth_filter"] = int(len(points_foreground))
    if len(points_foreground) < min_points_inside:
        return base, uv_inside, uv_foreground

    cluster_input, input_indices = bounded_dbscan_input(points_foreground, max_points=MAX_DBSCAN_POINTS)
    cluster_points, cluster_input_mask, status, warning = largest_dbscan_cluster(
        cluster_input,
        eps=task_config.lifting_dbscan_eps_m,
        min_samples=task_config.lifting_dbscan_min_samples,
    )
    if len(input_indices) == len(points_foreground) and len(cluster_input_mask) == len(uv_foreground):
        uv_cluster = uv_foreground[cluster_input_mask]
    elif len(cluster_input_mask) == len(input_indices):
        uv_cluster = uv_foreground[input_indices[cluster_input_mask]]
    else:
        uv_cluster = uv_foreground
    base["status"] = status
    if warning:
        base["warnings"].append(warning)
    if len(points_foreground) > len(cluster_input):
        base["warnings"].append(
            f"Subsampled {len(points_foreground)} foreground points to {len(cluster_input)} before DBSCAN."
        )
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


def projected_points_inside_detection_mask(
    mask: np.ndarray,
    uv: np.ndarray,
    valid_projection: np.ndarray,
    bbox_xyxy: Any,
) -> np.ndarray:
    """Return projected points inside a mask, using bbox prefiltering for speed."""

    candidate = np.asarray(valid_projection, dtype=bool).copy()
    bbox = _parse_bbox_xyxy(bbox_xyxy)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        candidate &= uv[:, 0] >= x1 - 1.0
        candidate &= uv[:, 0] <= x2 + 1.0
        candidate &= uv[:, 1] >= y1 - 1.0
        candidate &= uv[:, 1] <= y2 + 1.0
    candidate_indices = np.flatnonzero(candidate)
    inside = np.zeros_like(candidate, dtype=bool)
    if len(candidate_indices) == 0:
        return inside
    inside_subset = mask_contains_uv(mask, uv[candidate_indices])
    inside[candidate_indices[inside_subset]] = True
    return inside


def _parse_bbox_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def bounded_dbscan_input(points: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) <= max_points:
        indices = np.arange(len(points), dtype=int)
        return points, indices
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return points[indices], indices
