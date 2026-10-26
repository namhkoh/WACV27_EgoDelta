from __future__ import annotations

from typing import Any

import numpy as np

from .config import PipelineConfig


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm


def _robust_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape((-1,))
    if len(values) == 0:
        return 0.0
    if len(values) < 5:
        return float(np.max(values) - np.min(values))
    return float(np.quantile(values, 0.95) - np.quantile(values, 0.05))


def point_extent_stats(points: np.ndarray, world_up: tuple[float, float, float]) -> dict[str, float]:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) < 2:
        return {
            "major_extent_m": 0.0,
            "horizontal_extent_m": 0.0,
            "vertical_extent_m": 0.0,
        }

    world_up_arr = _normalize(np.asarray(world_up, dtype=float))
    center = np.median(points, axis=0)
    centered = points - center.reshape((1, 3))

    vertical_coords = centered @ world_up_arr
    vertical_extent = _robust_range(vertical_coords)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    major_coords = centered @ vh[0]
    major_extent = _robust_range(major_coords)

    horizontal = centered - vertical_coords.reshape((-1, 1)) * world_up_arr.reshape((1, 3))
    horizontal_norm = np.linalg.norm(horizontal, axis=1)
    if float(np.max(horizontal_norm)) < 1e-9:
        horizontal_extent = 0.0
    else:
        _, _, h_vh = np.linalg.svd(horizontal, full_matrices=False)
        horizontal_coords = horizontal @ h_vh[0]
        horizontal_extent = _robust_range(horizontal_coords)

    return {
        "major_extent_m": float(major_extent),
        "horizontal_extent_m": float(horizontal_extent),
        "vertical_extent_m": float(vertical_extent),
    }


def deleafing_petiole_size_reject_reason(
    points: np.ndarray,
    config: PipelineConfig,
) -> tuple[str | None, dict[str, Any]]:
    stats = point_extent_stats(points, config.world_up)
    major = stats["major_extent_m"]

    if major > config.petiole_max_3d_major_extent_m:
        return "petiole_3d_major_extent_too_large", stats
    if major < config.petiole_min_3d_major_extent_m:
        return "petiole_3d_major_extent_too_small", stats
    return None, stats
