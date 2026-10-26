"""Visibility helpers for task-agnostic pre/post change detection."""

from __future__ import annotations

from typing import Any

from .mps_geometry import MPSGeometry
from .prepost_change import visible_post_views


def count_visible_views(
    instance: dict[str, Any],
    post_frames: list[dict[str, Any]],
    post_geometry: MPSGeometry,
    min_projected_radius_px: float = 3.0,
) -> list[dict[str, Any]]:
    return visible_post_views(
        instance,
        post_frames,
        post_geometry,
        min_projected_radius_px=min_projected_radius_px,
    )
