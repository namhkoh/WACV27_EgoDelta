from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def validate_target_artifact_goal(
    artifact_path: Path,
    requested_goal_robot_m: np.ndarray,
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    selected = payload.get("selected_point")
    metadata: dict[str, Any] = {
        "artifact": str(Path(artifact_path).resolve()),
        "point_semantics": "visible_fruit_surface",
        "artifact_goal_checked": False,
    }
    if not isinstance(selected, dict) or selected.get("point_robot_m") is None:
        return metadata

    requested = np.asarray(requested_goal_robot_m, dtype=float).reshape(3)
    artifact_goal = np.asarray(selected["point_robot_m"], dtype=float).reshape(3)
    error = float(np.linalg.norm(requested - artifact_goal))
    if error > float(tolerance_m):
        raise RuntimeError(
            f"planner goal disagrees with selected target artifact by {error:.4f}m; "
            f"maximum is {float(tolerance_m):.4f}m"
        )
    metadata.update(
        {
            "point_semantics": str(selected.get("point_semantics", "visible_fruit_surface")),
            "artifact_goal_checked": True,
            "artifact_goal_error_m": error,
            "artifact_goal_tolerance_m": float(tolerance_m),
            "selected_pixel_xy": selected.get("pixel_xy"),
            "raw_click_pixel_xy": selected.get("raw_click_pixel_xy"),
            "selection_method": selected.get("selection_method"),
        }
    )
    for key in (
        "selected_peak_radius_px",
        "estimated_fruit_radius_m",
        "estimated_fruit_diameter_m",
        "metric_radius_method",
        "metric_radius_focal_px",
    ):
        if selected.get(key) is not None:
            metadata[key] = selected[key]
    return metadata


def _red_mask(
    color_bgr: np.ndarray,
    *,
    max_low_hue: int,
    min_high_hue: int,
    min_saturation: int,
    min_value: int,
) -> np.ndarray:
    image = np.asarray(color_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"color_bgr must have shape HxWx3, got {image.shape}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    mask = (
        ((hue <= int(max_low_hue)) | (hue >= int(min_high_hue)))
        & (hsv[:, :, 1] >= int(min_saturation))
        & (hsv[:, :, 2] >= int(min_value))
    ).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


def _component_for_click(
    mask: np.ndarray,
    click_xy: np.ndarray,
    *,
    max_click_distance_px: float,
    min_component_area_px: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x = int(round(float(click_xy[0])))
    y = int(round(float(click_xy[1])))
    height, width = mask.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        raise RuntimeError(f"clicked pixel {(x, y)} lies outside image {width}x{height}")

    label = int(labels[y, x])
    snapped_click = np.asarray([x, y], dtype=int)
    click_distance = 0.0
    if label == 0:
        red_y, red_x = np.where(labels > 0)
        if red_x.size == 0:
            raise RuntimeError("capture contains no red fruit pixels")
        distances_sq = (red_x - x) ** 2 + (red_y - y) ** 2
        nearest = int(np.argmin(distances_sq))
        click_distance = float(np.sqrt(distances_sq[nearest]))
        if click_distance > float(max_click_distance_px):
            raise RuntimeError(
                f"click is {click_distance:.1f}px from the nearest red fruit; "
                f"maximum is {float(max_click_distance_px):.1f}px"
            )
        snapped_click = np.asarray([int(red_x[nearest]), int(red_y[nearest])], dtype=int)
        label = int(labels[snapped_click[1], snapped_click[0]])

    area = int(stats[label, cv2.CC_STAT_AREA]) if 0 < label < count else 0
    if area < int(min_component_area_px):
        raise RuntimeError(
            f"selected red component is too small: area={area}px, minimum={int(min_component_area_px)}px"
        )
    return labels == label, snapped_click, click_distance


def _instance_peaks(
    component_mask: np.ndarray,
    *,
    min_peak_radius_px: float,
    peak_window_px: int,
    min_peak_separation_px: float,
) -> list[tuple[float, int, int]]:
    distance = cv2.distanceTransform(component_mask.astype(np.uint8), cv2.DIST_L2, 5)
    window = max(3, int(peak_window_px))
    if window % 2 == 0:
        window += 1
    local_max = cv2.dilate(distance, np.ones((window, window), dtype=np.uint8))
    ys, xs = np.where((distance >= local_max - 1e-5) & (distance >= float(min_peak_radius_px)))
    candidates = sorted(
        ((float(distance[y, x]), int(x), int(y)) for x, y in zip(xs, ys)),
        reverse=True,
    )
    peaks: list[tuple[float, int, int]] = []
    separation_sq = float(min_peak_separation_px) ** 2
    for candidate in candidates:
        _, x, y = candidate
        if all((x - other_x) ** 2 + (y - other_y) ** 2 >= separation_sq for _, other_x, other_y in peaks):
            peaks.append(candidate)
    if not peaks:
        raise RuntimeError("selected red component has no fruit-sized interior peak")
    return peaks


def refine_red_fruit_center_surface(
    capture: dict[str, np.ndarray],
    click_xy: np.ndarray | list[float] | tuple[float, float],
    *,
    intrinsics: dict[str, Any] | None = None,
    max_low_hue: int = 12,
    min_high_hue: int = 170,
    min_saturation: int = 80,
    min_value: int = 50,
    max_click_distance_px: float = 20.0,
    min_component_area_px: int = 300,
    min_peak_radius_px: float = 8.0,
    peak_window_px: int = 21,
    min_peak_separation_px: float = 25.0,
    center_support_radius_px: float = 6.0,
    min_depth_support_points: int = 20,
    min_assignment_margin_px: float = 5.0,
) -> dict[str, Any]:
    """Use a click to select a red fruit instance, then return its centered visible surface point."""

    required = ("color_bgr", "points", "uv")
    missing = [key for key in required if key not in capture]
    if missing:
        raise RuntimeError(f"RGB-D capture is missing arrays required for fruit centering: {missing}")
    click = np.asarray(click_xy, dtype=float).reshape(2)
    mask = _red_mask(
        capture["color_bgr"],
        max_low_hue=max_low_hue,
        min_high_hue=min_high_hue,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    component, snapped_click, click_to_mask_distance = _component_for_click(
        mask,
        click,
        max_click_distance_px=max_click_distance_px,
        min_component_area_px=min_component_area_px,
    )
    peaks = _instance_peaks(
        component,
        min_peak_radius_px=min_peak_radius_px,
        peak_window_px=peak_window_px,
        min_peak_separation_px=min_peak_separation_px,
    )
    peak_xy = np.asarray([[item[1], item[2]] for item in peaks], dtype=float)
    click_distances = np.linalg.norm(peak_xy - snapped_click[None, :], axis=1)
    order = np.argsort(click_distances)
    selected_peak = int(order[0])
    assignment_margin = (
        float(click_distances[order[1]] - click_distances[order[0]])
        if len(order) > 1
        else float("inf")
    )
    if assignment_margin < float(min_assignment_margin_px):
        raise RuntimeError(
            f"fruit click is ambiguous between touching instances: margin={assignment_margin:.1f}px, "
            f"minimum={float(min_assignment_margin_px):.1f}px"
        )

    component_y, component_x = np.where(component)
    point_pixels = np.column_stack((component_x, component_y)).astype(float)
    assignments = np.argmin(
        np.sum((point_pixels[:, None, :] - peak_xy[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    selected_pixels = point_pixels[assignments == selected_peak]
    if selected_pixels.shape[0] < int(min_component_area_px):
        raise RuntimeError(
            f"selected fruit instance is too small after separating touching fruit: {selected_pixels.shape[0]}px"
        )
    center_xy = np.median(selected_pixels, axis=0)

    uv = np.asarray(capture["uv"], dtype=float).reshape(-1, 2)
    points = np.asarray(capture["points"], dtype=float).reshape(-1, 3)
    if uv.shape[0] != points.shape[0]:
        raise RuntimeError(f"RGB-D capture uv/points length mismatch: {uv.shape[0]} vs {points.shape[0]}")
    uv_round = np.rint(uv).astype(int)
    height, width = component.shape
    in_image = (
        (uv_round[:, 0] >= 0)
        & (uv_round[:, 0] < width)
        & (uv_round[:, 1] >= 0)
        & (uv_round[:, 1] < height)
    )
    support = in_image & (
        np.sum((uv - center_xy[None, :]) ** 2, axis=1) <= float(center_support_radius_px) ** 2
    )
    valid_indices = np.flatnonzero(support)
    if valid_indices.size:
        pixels = uv_round[valid_indices]
        belongs_to_component = component[pixels[:, 1], pixels[:, 0]]
        valid_indices = valid_indices[belongs_to_component]
    finite = np.all(np.isfinite(points[valid_indices]), axis=1) if valid_indices.size else np.zeros(0, dtype=bool)
    valid_indices = valid_indices[finite]
    if valid_indices.size < int(min_depth_support_points):
        raise RuntimeError(
            f"selected fruit center has insufficient depth support: {valid_indices.size} points, "
            f"minimum={int(min_depth_support_points)}"
        )

    point_realsense = np.median(points[valid_indices], axis=0)
    nearest_index = int(valid_indices[np.argmin(np.linalg.norm(points[valid_indices] - point_realsense, axis=1))])
    center_pixel = [int(round(float(center_xy[0]))), int(round(float(center_xy[1])))]
    selected_peak_radius_px = float(peaks[selected_peak][0])
    metric_geometry: dict[str, Any] = {}
    if intrinsics is not None:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(f"camera focal lengths must be positive, got fx={fx}, fy={fy}")
        focal_px = float(np.sqrt(fx * fy))
        estimated_radius_m = selected_peak_radius_px * float(point_realsense[2]) / focal_px
        metric_geometry = {
            "estimated_fruit_radius_m": estimated_radius_m,
            "estimated_fruit_diameter_m": 2.0 * estimated_radius_m,
            "metric_radius_method": "distance_transform_inscribed_radius_times_depth_over_focal",
            "metric_radius_focal_px": focal_px,
        }
    return {
        "point_index": nearest_index,
        "pixel_xy": center_pixel,
        "point_realsense_m": [float(value) for value in point_realsense],
        "point_semantics": "selected_fruit_center_visible_surface",
        "selection_method": "manual_click_red_instance_center",
        "raw_click_pixel_xy": [int(round(float(click[0]))), int(round(float(click[1])))],
        "mask_click_pixel_xy": [int(value) for value in snapped_click],
        "raw_to_center_pixel_distance_px": float(np.linalg.norm(center_xy - click)),
        "click_to_mask_distance_px": click_to_mask_distance,
        "selected_instance_area_px": int(selected_pixels.shape[0]),
        "selected_peak_pixel_xy": [int(value) for value in peak_xy[selected_peak]],
        "selected_peak_radius_px": selected_peak_radius_px,
        "detected_instance_peak_count": len(peaks),
        "assignment_margin_px": assignment_margin,
        "depth_support_points": int(valid_indices.size),
        "center_support_radius_px": float(center_support_radius_px),
        **metric_geometry,
    }
