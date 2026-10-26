from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Camera:
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    center: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_json(cls, item: dict) -> "Camera":
        width = int(item["width"])
        height = int(item["height"])
        name = str(item.get("img_name", item.get("image_name", "")))
        if not name:
            raise ValueError("Camera entry is missing img_name")
        return cls(
            image_name=name if name.lower().endswith((".jpg", ".jpeg", ".png")) else f"{name}.jpg",
            width=width,
            height=height,
            fx=float(item["fx"]),
            fy=float(item["fy"]),
            cx=float(item.get("cx", (width - 1.0) * 0.5)),
            cy=float(item.get("cy", (height - 1.0) * 0.5)),
            center=np.asarray(item["position"], dtype=np.float64).reshape(3),
            rotation=np.asarray(item["rotation"], dtype=np.float64).reshape(3, 3),
        )


@dataclass(frozen=True)
class ProjectedComponent:
    label: int
    uv: np.ndarray
    depth: np.ndarray
    world_scale: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray

    @property
    def visible_count(self) -> int:
        return int(np.count_nonzero(self.valid))


def project_points(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera_points = (points - camera.center[None, :]) @ camera.rotation
    depth = camera_points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    positive = depth > 1.0e-5
    uv[positive, 0] = camera.fx * camera_points[positive, 0] / depth[positive] + camera.cx
    uv[positive, 1] = camera.fy * camera_points[positive, 1] / depth[positive] + camera.cy
    valid = (
        positive
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.height)
    )
    return uv, depth, valid


def projection_counts(components: Iterable[dict], camera: Camera) -> list[int]:
    counts: list[int] = []
    for component in components:
        _, _, valid = project_points(component["xyz"], camera)
        counts.append(int(np.count_nonzero(valid)))
    return counts


def select_temporal_views(
    cameras: list[Camera],
    counts_by_view: list[list[int]],
    count: int,
    min_visible_points: int = 20,
) -> list[int]:
    """Choose mask-visible views while preserving coverage through the sequence."""

    if count < 1:
        raise ValueError("count must be positive")
    if len(cameras) != len(counts_by_view):
        raise ValueError("cameras and counts_by_view must have equal length")
    if len(cameras) < count:
        raise ValueError(f"Requested {count} views from only {len(cameras)} cameras")

    def frame_number(camera: Camera) -> int:
        match = re.search(r"(\d+)(?!.*\d)", camera.image_name)
        return int(match.group(1)) if match else 0

    def quality(index: int) -> tuple[float, int, int, str]:
        values = np.asarray(counts_by_view[index], dtype=np.float64)
        covered = int(np.count_nonzero(values))
        balanced = float(np.log1p(values).sum())
        return balanced, covered, int(values.sum()), cameras[index].image_name

    ordered = sorted(range(len(cameras)), key=lambda index: (frame_number(cameras[index]), cameras[index].image_name))
    eligible = [index for index in ordered if sum(counts_by_view[index]) >= min_visible_points]
    if len(eligible) < count:
        eligible = ordered

    selected: list[int] = []
    for bucket in np.array_split(np.asarray(eligible, dtype=np.int64), count):
        if len(bucket):
            selected.append(max((int(value) for value in bucket), key=quality))

    if len(selected) < count:
        remaining = [index for index in ordered if index not in set(selected)]
        remaining.sort(key=quality, reverse=True)
        selected.extend(remaining[: count - len(selected)])

    selected = sorted(selected[:count], key=lambda index: (frame_number(cameras[index]), cameras[index].image_name))
    return selected


def build_scene_depth(points: np.ndarray, camera: Camera, dilation_radius: int = 2) -> np.ndarray:
    """Build a conservative center-depth buffer from the complete pre-action model."""

    uv, depth, valid = project_points(points, camera)
    depth_map = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    if np.any(valid):
        u = np.rint(uv[valid, 0]).astype(np.int32).clip(0, camera.width - 1)
        v = np.rint(uv[valid, 1]).astype(np.int32).clip(0, camera.height - 1)
        np.minimum.at(depth_map, (v, u), depth[valid].astype(np.float32))
    if dilation_radius <= 0:
        return depth_map

    radius = int(dilation_radius)
    padded = np.pad(depth_map, radius, mode="constant", constant_values=np.inf)
    expanded = depth_map.copy()
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            expanded = np.minimum(
                expanded,
                padded[dy : dy + camera.height, dx : dx + camera.width],
            )
    return expanded


def project_component(component: dict, label: int, camera: Camera) -> ProjectedComponent:
    uv, depth, valid = project_points(component["xyz"], camera)
    return ProjectedComponent(
        label=label,
        uv=uv,
        depth=depth,
        world_scale=np.asarray(component["world_scale"], dtype=np.float64),
        alpha=np.asarray(component["alpha"], dtype=np.float64),
        valid=valid,
    )


def rasterize_components(
    components: list[dict],
    camera: Camera,
    scene_depth: np.ndarray | None = None,
    footprint_sigma: float = 2.25,
    max_radius_px: int = 14,
    min_alpha: float = 0.02,
    occlusion_tolerance_m: float = 0.012,
    occlusion_relative_tolerance: float = 0.025,
) -> tuple[Image.Image, dict]:
    """Create an editable instance label map from projected Gaussian components.

    Labels are ``component_id + 1`` and background is zero. This projection is
    intentionally conservative and is only an initialization for human review.
    """

    if len(components) > 254:
        raise ValueError("8-bit label maps support at most 254 components")
    image = Image.new("L", (camera.width, camera.height), color=0)
    draw = ImageDraw.Draw(image)
    projected = [project_component(component, index + 1, camera) for index, component in enumerate(components)]

    strokes: list[tuple[float, int, float, float, int]] = []
    per_component: list[dict] = []
    focal = 0.5 * (camera.fx + camera.fy)
    for item in projected:
        valid = item.valid & (item.alpha >= min_alpha)
        if scene_depth is not None and np.any(valid):
            indices = np.flatnonzero(valid)
            u = np.rint(item.uv[indices, 0]).astype(np.int32).clip(0, camera.width - 1)
            v = np.rint(item.uv[indices, 1]).astype(np.int32).clip(0, camera.height - 1)
            reference_depth = scene_depth[v, u]
            tolerance = np.maximum(
                occlusion_tolerance_m,
                occlusion_relative_tolerance * item.depth[indices],
            )
            visible = ~np.isfinite(reference_depth) | (item.depth[indices] <= reference_depth + tolerance)
            filtered = np.zeros_like(valid)
            filtered[indices[visible]] = True
            valid = filtered

        indices = np.flatnonzero(valid)
        if len(indices):
            radii = np.ceil(
                footprint_sigma * focal * item.world_scale[indices] / np.maximum(item.depth[indices], 1.0e-5)
            )
            radii = np.clip(radii, 1, max_radius_px).astype(np.int32)
            for source_index, radius in zip(indices, radii, strict=True):
                strokes.append(
                    (
                        float(item.depth[source_index]),
                        item.label,
                        float(item.uv[source_index, 0]),
                        float(item.uv[source_index, 1]),
                        int(radius),
                    )
                )
        per_component.append(
            {
                "label": item.label,
                "projected_points": int(np.count_nonzero(item.valid)),
                "occlusion_filtered_points": int(len(indices)),
            }
        )

    # Painter's order approximates a z-buffer between exported components.
    strokes.sort(key=lambda row: row[0], reverse=True)
    for _, label, u, v, radius in strokes:
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=label)

    array = np.asarray(image, dtype=np.uint8)
    pixel_counts = {str(label): int(np.count_nonzero(array == label)) for label in range(1, len(components) + 1)}
    metadata = {
        "visible_pixels": int(np.count_nonzero(array)),
        "pixel_counts": pixel_counts,
        "components": per_component,
    }
    return image, metadata


def camera_to_manifest(camera: Camera) -> dict:
    world_to_camera_rotation = camera.rotation.T
    world_to_camera_translation = -(world_to_camera_rotation @ camera.center)
    return {
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "camera_center_world": camera.center.tolist(),
        "rotation_camera_to_world": camera.rotation.tolist(),
        "rotation_world_to_camera": world_to_camera_rotation.tolist(),
        "translation_world_to_camera": world_to_camera_translation.tolist(),
    }
