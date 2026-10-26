"""Simple curve and line fitting utilities."""

from __future__ import annotations

import numpy as np


def fit_pca_line(points: np.ndarray) -> dict[str, np.ndarray | float]:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        zero = np.zeros((3,), dtype=float)
        return {
            "centroid": zero,
            "axis": np.array([0.0, 0.0, 1.0], dtype=float),
            "endpoint_a": zero,
            "endpoint_b": zero,
            "length_m": 0.0,
            "thinness": 0.0,
        }
    centroid = np.median(points, axis=0)
    centered = points - centroid.reshape(1, 3)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    lo = float(np.quantile(projections, 0.05))
    hi = float(np.quantile(projections, 0.95))
    endpoint_a = centroid + lo * axis
    endpoint_b = centroid + hi * axis
    length = float(np.linalg.norm(endpoint_b - endpoint_a))
    if len(singular_values) >= 2 and float(singular_values[0]) > 1e-9:
        thinness = float(1.0 - min(1.0, singular_values[1] / singular_values[0]))
    else:
        thinness = 0.0
    return {
        "centroid": centroid,
        "axis": axis,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "length_m": length,
        "thinness": thinness,
    }


def distance_point_to_line(point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
    point = np.asarray(point, dtype=float).reshape(3)
    line_point = np.asarray(line_point, dtype=float).reshape(3)
    line_dir = np.asarray(line_dir, dtype=float).reshape(3)
    norm = float(np.linalg.norm(line_dir))
    if norm <= 1e-9:
        return float(np.linalg.norm(point - line_point))
    direction = line_dir / norm
    return float(np.linalg.norm(np.cross(point - line_point, direction)))
