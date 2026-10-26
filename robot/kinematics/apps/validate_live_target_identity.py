from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR
    ]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.task_frame import (
    DEFAULT_TARGET_DIR,
    DEFAULT_TARGET_PATTERN,
    HarvestTaskFrame,
    discover_harvest_task_frame_paths,
    load_harvest_task_frame,
)
from robot.kinematics.utils.capture_live_realsense import (
    build_point_cloud,
    capture_realsense,
    default_transform_json,
    write_capture_artifacts,
)


DEFAULT_DATASET = "a270c038"
TASK_NAME = "target_identity_validation"


@dataclass(frozen=True)
class LiveFruitDetection:
    detection_id: int
    component_label: int
    pixel_uv: np.ndarray
    point_realsense_m: np.ndarray
    point_robot_m: np.ndarray
    depth_m: float
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "detection_id": int(self.detection_id),
            "component_label": int(self.component_label),
            "pixel_uv": self.pixel_uv.tolist(),
            "point_realsense_m": self.point_realsense_m.tolist(),
            "point_robot_m": self.point_robot_m.tolist(),
            "depth_m": float(self.depth_m),
            "area_px": int(self.area_px),
            "bbox_xyxy": [int(value) for value in self.bbox_xyxy],
        }


def load_transforms(transform_json: Path) -> dict[str, np.ndarray]:
    data = json.loads(transform_json.read_text(encoding="utf-8"))
    return {
        "T_robot_from_realsense": np.asarray(data["T_robot_from_realsense"], dtype=float).reshape(4, 4),
        "T_realsense_from_aria": np.asarray(data["T_realsense_from_aria"], dtype=float).reshape(4, 4),
    }


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = np.asarray(point, dtype=float).reshape(3)
    return (transform @ homogeneous)[:3]


def project_robot_point(
    point_robot_m: np.ndarray,
    *,
    T_realsense_from_robot: np.ndarray,
    intrinsics: dict[str, Any],
) -> dict[str, Any]:
    point_realsense = transform_point(T_realsense_from_robot, point_robot_m)
    z = float(point_realsense[2])
    if z <= 0.0:
        return {"in_front": False, "in_image": False, "point_realsense_m": point_realsense.tolist()}
    u = float(intrinsics["fx"]) * float(point_realsense[0]) / z + float(intrinsics["ppx"])
    v = float(intrinsics["fy"]) * float(point_realsense[1]) / z + float(intrinsics["ppy"])
    in_image = 0.0 <= u < float(intrinsics["width"]) and 0.0 <= v < float(intrinsics["height"])
    return {
        "in_front": True,
        "in_image": bool(in_image),
        "pixel_uv": [u, v],
        "depth_m": z,
        "point_realsense_m": point_realsense.tolist(),
    }


def red_fruit_mask(color_bgr: np.ndarray, *, min_saturation: int, min_value: int) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(hsv, (0, int(min_saturation), int(min_value)), (12, 255, 255))
    upper_red = cv2.inRange(hsv, (170, int(min_saturation), int(min_value)), (179, 255, 255))
    mask = cv2.bitwise_or(lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_live_fruit(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    T_robot_from_realsense: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    min_area_px: int,
    max_area_px: int,
    min_saturation: int,
    min_value: int,
) -> tuple[list[LiveFruitDetection], np.ndarray, np.ndarray]:
    mask = red_fruit_mask(color_bgr, min_saturation=min_saturation, min_value=min_value)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections: list[LiveFruitDetection] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area_px) or area > int(max_area_px):
            continue
        component = labels == label
        valid = component & np.isfinite(depth_m) & (depth_m >= float(min_depth_m)) & (depth_m <= float(max_depth_m))
        if int(np.count_nonzero(valid)) < max(16, int(min_area_px) // 20):
            continue
        rows, cols = np.nonzero(valid)
        z = np.asarray(depth_m[valid], dtype=float)
        x = (cols.astype(float) - float(intrinsics["ppx"])) * z / float(intrinsics["fx"])
        y = (rows.astype(float) - float(intrinsics["ppy"])) * z / float(intrinsics["fy"])
        point_realsense = np.asarray([np.median(x), np.median(y), np.median(z)], dtype=float)
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        detections.append(
            LiveFruitDetection(
                detection_id=len(detections),
                component_label=int(label),
                pixel_uv=np.asarray(centroids[label], dtype=float),
                point_realsense_m=point_realsense,
                point_robot_m=transform_point(T_robot_from_realsense, point_realsense),
                depth_m=float(np.median(z)),
                area_px=area,
                bbox_xyxy=(x0, y0, x0 + width - 1, y0 + height - 1),
            )
        )
    return detections, mask, labels


def task_frame_mask_npz(paths: tuple[Path, ...], override: Path | None) -> Path:
    if override is not None:
        return override
    candidates: list[Path] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("mask_npz")
        if value:
            candidates.append(Path(value))
    unique = sorted({str(path) for path in candidates})
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one mask_npz across task frames, found {unique}")
    return Path(unique[0])


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(_REPO_ROOT) / path


def load_mask_instances(mask_npz: Path) -> dict[int, np.ndarray]:
    data = np.load(mask_npz)
    xyz = np.asarray(data["xyz"], dtype=float).reshape(-1, 3)
    instance_ids = np.asarray(data["tomato_instance_id"], dtype=int).reshape(-1)
    return {
        int(instance_id): xyz[instance_ids == int(instance_id)]
        for instance_id in sorted(np.unique(instance_ids))
        if int(instance_id) >= 0
    }


def project_mask_instance(
    points_aria_m: np.ndarray,
    *,
    T_realsense_from_aria: np.ndarray,
    intrinsics: dict[str, Any],
    image_shape: tuple[int, int],
    radius_px: int,
) -> dict[str, Any]:
    height, width = image_shape
    points = np.asarray(points_aria_m, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return {"point_count": 0, "in_image_count": 0, "mask": np.zeros((height, width), dtype=np.uint8)}
    homogeneous = np.ones((points.shape[0], 4), dtype=float)
    homogeneous[:, :3] = points
    points_realsense = (T_realsense_from_aria @ homogeneous.T).T[:, :3]
    z = points_realsense[:, 2]
    in_front = z > 0.0
    u = np.full(points.shape[0], np.nan, dtype=float)
    v = np.full(points.shape[0], np.nan, dtype=float)
    u[in_front] = float(intrinsics["fx"]) * points_realsense[in_front, 0] / z[in_front] + float(intrinsics["ppx"])
    v[in_front] = float(intrinsics["fy"]) * points_realsense[in_front, 1] / z[in_front] + float(intrinsics["ppy"])
    in_image = in_front & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels = np.column_stack((u[in_image], v[in_image]))
    for pixel in pixels:
        cv2.circle(mask, tuple(np.round(pixel).astype(int)), int(radius_px), 255, -1)
    bbox = None
    if pixels.size:
        bbox = [
            int(np.floor(np.min(pixels[:, 0]))),
            int(np.floor(np.min(pixels[:, 1]))),
            int(np.ceil(np.max(pixels[:, 0]))),
            int(np.ceil(np.max(pixels[:, 1]))),
        ]
    return {
        "point_count": int(points.shape[0]),
        "in_front_count": int(np.count_nonzero(in_front)),
        "in_image_count": int(np.count_nonzero(in_image)),
        "in_image_fraction": float(np.count_nonzero(in_image)) / float(points.shape[0]),
        "pixel_centroid_uv": None if not pixels.size else np.median(pixels, axis=0).tolist(),
        "pixel_bbox_xyxy": bbox,
        "mask": mask,
    }


def live_depth_target_from_overlap(
    *,
    projected_mask: np.ndarray,
    detection: LiveFruitDetection,
    label_image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    T_robot_from_realsense: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, Any] | None:
    overlap = (projected_mask > 0) & (label_image == detection.component_label)
    valid = overlap & np.isfinite(depth_m) & (depth_m >= float(min_depth_m)) & (depth_m <= float(max_depth_m))
    if not np.any(valid):
        return None
    rows, cols = np.nonzero(valid)
    z = np.asarray(depth_m[valid], dtype=float)
    x = (cols.astype(float) - float(intrinsics["ppx"])) * z / float(intrinsics["fx"])
    y = (rows.astype(float) - float(intrinsics["ppy"])) * z / float(intrinsics["fy"])
    point_realsense = np.asarray([np.median(x), np.median(y), np.median(z)], dtype=float)
    return {
        "overlap_pixels": int(np.count_nonzero(overlap)),
        "valid_depth_pixels": int(z.size),
        "point_realsense_m": point_realsense.tolist(),
        "point_robot_m": transform_point(T_robot_from_realsense, point_realsense).tolist(),
        "pixel_uv": [float(np.median(cols)), float(np.median(rows))],
        "depth_m": float(np.median(z)),
    }


def frame_projection_summary(
    frame: HarvestTaskFrame,
    *,
    T_realsense_from_robot: np.ndarray,
    T_realsense_from_aria: np.ndarray,
    intrinsics: dict[str, Any],
    mask_instances: dict[int, np.ndarray],
    image_shape: tuple[int, int],
    mask_projection_radius_px: int,
) -> dict[str, Any]:
    target_projection = project_robot_point(
        frame.target_position_robot_m,
        T_realsense_from_robot=T_realsense_from_robot,
        intrinsics=intrinsics,
    )
    center_projection = project_robot_point(
        frame.fruit_center_robot_m,
        T_realsense_from_robot=T_realsense_from_robot,
        intrinsics=intrinsics,
    )
    mask_projection = project_mask_instance(
        mask_instances.get(int(frame.target_index), np.empty((0, 3), dtype=float)),
        T_realsense_from_aria=T_realsense_from_aria,
        intrinsics=intrinsics,
        image_shape=image_shape,
        radius_px=mask_projection_radius_px,
    )
    mask = mask_projection.pop("mask")
    return {
        "target_name": frame.target_name,
        "target_index": int(frame.target_index),
        "source_path": str(frame.source_path),
        "target_robot_m": frame.target_position_robot_m.tolist(),
        "fruit_center_robot_m": frame.fruit_center_robot_m.tolist(),
        "target_projection": target_projection,
        "fruit_center_projection": center_projection,
        "mask_projection": mask_projection,
        "_projected_mask": mask,
    }


def match_score(
    frame_summary: dict[str, Any],
    detection: LiveFruitDetection,
    *,
    label_image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    T_robot_from_realsense: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    max_center_distance_m: float,
    max_pixel_distance_px: float,
    max_depth_error_m: float,
    min_mask_overlap_pixels: int,
) -> dict[str, Any]:
    center_robot = np.asarray(frame_summary["fruit_center_robot_m"], dtype=float)
    center_projection = frame_summary["fruit_center_projection"]
    live_target = live_depth_target_from_overlap(
        projected_mask=frame_summary["_projected_mask"],
        detection=detection,
        label_image=label_image,
        depth_m=depth_m,
        intrinsics=intrinsics,
        T_robot_from_realsense=T_robot_from_realsense,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    center_distance_m = float(np.linalg.norm(detection.point_robot_m - center_robot))
    pixel_distance_px = None
    depth_error_m = None
    if center_projection.get("in_image", False):
        projected_pixel = np.asarray(center_projection["pixel_uv"], dtype=float)
        pixel_distance_px = float(np.linalg.norm(detection.pixel_uv - projected_pixel))
        depth_error_m = abs(float(detection.depth_m) - float(center_projection["depth_m"]))
    overlap_pixels = 0 if live_target is None else int(live_target["overlap_pixels"])
    valid_depth_pixels = 0 if live_target is None else int(live_target["valid_depth_pixels"])
    passes = center_distance_m <= float(max_center_distance_m)
    passes = passes and valid_depth_pixels >= int(min_mask_overlap_pixels)
    if pixel_distance_px is not None:
        passes = passes and pixel_distance_px <= float(max_pixel_distance_px)
    if depth_error_m is not None:
        passes = passes and depth_error_m <= float(max_depth_error_m)
    normalized = center_distance_m / float(max_center_distance_m)
    if pixel_distance_px is not None:
        normalized += pixel_distance_px / float(max_pixel_distance_px)
    if depth_error_m is not None:
        normalized += depth_error_m / float(max_depth_error_m)
    if valid_depth_pixels > 0:
        normalized += 1.0 / float(valid_depth_pixels)
    else:
        normalized += 1000.0
    return {
        "detection_id": int(detection.detection_id),
        "passes": bool(passes),
        "score": float(normalized),
        "center_distance_m": center_distance_m,
        "pixel_distance_px": pixel_distance_px,
        "depth_error_m": depth_error_m,
        "mask_overlap_pixels": overlap_pixels,
        "valid_depth_pixels": valid_depth_pixels,
        "live_depth_target": live_target,
    }


def resolve_assignments(
    frames: list[dict[str, Any]],
    detections: list[LiveFruitDetection],
    *,
    label_image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    T_robot_from_realsense: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    max_center_distance_m: float,
    max_pixel_distance_px: float,
    max_depth_error_m: float,
    min_mask_overlap_pixels: int,
    min_assignment_margin_m: float,
) -> tuple[list[dict[str, Any]], bool]:
    score_rows = [
        [
            match_score(
                frame,
                detection,
                label_image=label_image,
                depth_m=depth_m,
                intrinsics=intrinsics,
                T_robot_from_realsense=T_robot_from_realsense,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                max_center_distance_m=max_center_distance_m,
                max_pixel_distance_px=max_pixel_distance_px,
                max_depth_error_m=max_depth_error_m,
                min_mask_overlap_pixels=min_mask_overlap_pixels,
            )
            for detection in detections
        ]
        for frame in frames
    ]
    reports: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        row = score_rows[frame_index]
        sorted_row = sorted(row, key=lambda item: item["center_distance_m"])
        best = sorted_row[0] if sorted_row else None
        second = sorted_row[1] if len(sorted_row) > 1 else None
        margin = None
        if best is not None and second is not None:
            margin = float(second["center_distance_m"] - best["center_distance_m"])
        ok = bool(best is not None and best["passes"] and (margin is None or margin >= float(min_assignment_margin_m)))
        reports.append(
            {
                "target_name": frame["target_name"],
                "target_index": frame["target_index"],
                "source_path": frame["source_path"],
                "ok": ok,
                "assigned_detection_id": None if best is None else int(best["detection_id"]),
                "best_match": best,
                "second_best_match": second,
                "assignment_margin_m": margin,
                "all_detection_scores": row,
            }
        )
    enforce_live_target_separation(reports, min_distance_m=min_assignment_margin_m)
    return reports, bool(reports and all(report["ok"] for report in reports))


def enforce_live_target_separation(reports: list[dict[str, Any]], *, min_distance_m: float) -> None:
    live_points: list[tuple[int, np.ndarray]] = []
    for index, report in enumerate(reports):
        target = (report.get("best_match") or {}).get("live_depth_target")
        if target is not None:
            live_points.append((index, np.asarray(target["point_robot_m"], dtype=float).reshape(3)))
    for (i, point_i), (j, point_j) in itertools.combinations(live_points, 2):
        distance = float(np.linalg.norm(point_i - point_j))
        reports[i].setdefault("live_target_separation_m", {})[reports[j]["target_name"]] = distance
        reports[j].setdefault("live_target_separation_m", {})[reports[i]["target_name"]] = distance
        if distance < float(min_distance_m):
            reports[i]["ok"] = False
            reports[j]["ok"] = False
            reports[i]["separation_reject_reason"] = f"live target separation {distance:.6f} m is below threshold"
            reports[j]["separation_reject_reason"] = f"live target separation {distance:.6f} m is below threshold"


def draw_overlay(
    color_bgr: np.ndarray,
    detections: list[LiveFruitDetection],
    frames: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> np.ndarray:
    overlay = color_bgr.copy()
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_xyxy
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
        center = tuple(np.round(detection.pixel_uv).astype(int))
        cv2.circle(overlay, center, 5, (0, 255, 255), -1)
        cv2.putText(overlay, f"live:{detection.detection_id}", (center[0] + 6, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    assignment_by_target = {item["target_name"]: item for item in assignments}
    for frame in frames:
        mask_projection = frame.get("mask_projection", {})
        bbox = mask_projection.get("pixel_bbox_xyxy")
        if bbox is not None:
            cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), 1)
        projection = frame["fruit_center_projection"]
        if not projection.get("in_image", False):
            continue
        point = tuple(np.round(projection["pixel_uv"]).astype(int))
        report = assignment_by_target.get(frame["target_name"], {})
        color = (0, 255, 0) if report.get("ok", False) else (0, 0, 255)
        cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(overlay, frame["target_name"], (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        live_target = (report.get("best_match") or {}).get("live_depth_target")
        if live_target is not None:
            live_point = tuple(np.round(live_target["pixel_uv"]).astype(int))
            cv2.circle(overlay, live_point, 6, color, 2)
    return overlay


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    transform_json = args.transform_json or default_transform_json(args.dataset_name)
    transform_json = resolve_repo_path(transform_json)
    transforms = load_transforms(transform_json)
    T_robot_from_realsense = transforms["T_robot_from_realsense"]
    T_realsense_from_aria = transforms["T_realsense_from_aria"]
    T_realsense_from_robot = np.linalg.inv(T_robot_from_realsense)
    target_dir = resolve_repo_path(args.target_dir)
    paths = tuple(resolve_repo_path(path) for path in args.task_frame_jsons) if args.task_frame_jsons else discover_harvest_task_frame_paths(target_dir, args.pattern)
    task_frames = [load_harvest_task_frame(path) for path in paths]
    mask_npz = resolve_repo_path(task_frame_mask_npz(paths, args.mask_npz))
    mask_instances = load_mask_instances(mask_npz)
    capture = capture_realsense(
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        serial=args.serial,
    )
    point_cloud = build_point_cloud(
        color_bgr=capture["color_bgr"],
        depth_m=capture["depth_m"],
        intrinsics=capture["intrinsics"],
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        stride=args.pointcloud_stride,
    )
    detections, red_mask, label_image = detect_live_fruit(
        color_bgr=capture["color_bgr"],
        depth_m=capture["depth_m"],
        intrinsics=capture["intrinsics"],
        T_robot_from_realsense=T_robot_from_realsense,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
    )
    frame_summaries = [
        frame_projection_summary(
            frame,
            T_realsense_from_robot=T_realsense_from_robot,
            T_realsense_from_aria=T_realsense_from_aria,
            intrinsics=capture["intrinsics"],
            mask_instances=mask_instances,
            image_shape=capture["depth_m"].shape,
            mask_projection_radius_px=args.mask_projection_radius_px,
        )
        for frame in task_frames
    ]
    assignments, ok = resolve_assignments(
        frame_summaries,
        detections,
        label_image=label_image,
        depth_m=capture["depth_m"],
        intrinsics=capture["intrinsics"],
        T_robot_from_realsense=T_robot_from_realsense,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        max_center_distance_m=args.max_center_distance_m,
        max_pixel_distance_px=args.max_pixel_distance_px,
        max_depth_error_m=args.max_depth_error_m,
        min_mask_overlap_pixels=args.min_mask_overlap_pixels,
        min_assignment_margin_m=args.min_assignment_margin_m,
    )
    run_label = args.run_label or "identity_validation_" + time.strftime("%H%M%S")
    public_frame_summaries = [
        {key: value for key, value in frame.items() if not key.startswith("_")}
        for frame in frame_summaries
    ]
    metadata = {
        "method": "live_realsense_task_frame_identity_validation",
        "dataset_name": args.dataset_name,
        "run_label": run_label,
        "ok": ok,
        "transform_json": str(transform_json),
        "mask_npz": str(mask_npz),
        "target_dir": str(target_dir),
        "pattern": args.pattern,
        "task_frame_count": len(task_frames),
        "detection_count": len(detections),
        "thresholds": {
            "max_center_distance_m": float(args.max_center_distance_m),
            "max_pixel_distance_px": float(args.max_pixel_distance_px),
            "max_depth_error_m": float(args.max_depth_error_m),
            "min_mask_overlap_pixels": int(args.min_mask_overlap_pixels),
            "min_assignment_margin_m": float(args.min_assignment_margin_m),
            "mask_projection_radius_px": int(args.mask_projection_radius_px),
            "min_area_px": int(args.min_area_px),
            "max_area_px": int(args.max_area_px),
            "min_saturation": int(args.min_saturation),
            "min_value": int(args.min_value),
        },
        "frame_metadata": capture["frame_metadata"],
        "intrinsics": capture["intrinsics"],
        "task_frames": public_frame_summaries,
        "detections": [detection.to_json() for detection in detections],
        "assignments": assignments,
    }
    capture_paths = write_capture_artifacts(
        dataset_name=args.dataset_name,
        run_label=run_label,
        runs_root=resolve_repo_path(args.runs_root),
        capture=capture,
        point_cloud=point_cloud,
        metadata=metadata,
    )
    overlay = draw_overlay(capture["color_bgr"], detections, frame_summaries, assignments)
    runs_root = resolve_repo_path(args.runs_root)
    overlay_path = artifact_path(TASK_NAME, f"{args.dataset_name}_{run_label}_overlay", suffix=".jpg", runs_root=runs_root)
    mask_path = artifact_path(TASK_NAME, f"{args.dataset_name}_{run_label}_red_mask", suffix=".png", runs_root=runs_root)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), red_mask)
    json_path = artifact_path(TASK_NAME, f"{args.dataset_name}_{run_label}", runs_root=runs_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    metadata["artifacts"] = {
        "json": str(json_path),
        "overlay_jpg": str(overlay_path),
        "red_mask_png": str(mask_path),
        "capture": capture_paths,
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EgoDelta task-frame identity against the current live RealSense view.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--pattern", default=DEFAULT_TARGET_PATTERN)
    parser.add_argument("--task-frame-jsons", type=Path, nargs="*", default=None)
    parser.add_argument("--mask-npz", type=Path, default=None)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--min-area-px", type=int, default=500)
    parser.add_argument("--max-area-px", type=int, default=25000)
    parser.add_argument("--min-saturation", type=int, default=80)
    parser.add_argument("--min-value", type=int, default=50)
    parser.add_argument("--max-center-distance-m", type=float, default=0.06)
    parser.add_argument("--max-pixel-distance-px", type=float, default=140.0)
    parser.add_argument("--max-depth-error-m", type=float, default=0.08)
    parser.add_argument("--mask-projection-radius-px", type=int, default=12)
    parser.add_argument("--min-mask-overlap-pixels", type=int, default=20)
    parser.add_argument("--min-assignment-margin-m", type=float, default=0.02)
    args = parser.parse_args()
    payload = run_validation(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
