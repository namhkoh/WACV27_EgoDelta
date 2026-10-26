from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normalize(fallback, np.array([0.0, 0.0, 1.0], dtype=np.float64)) if np.linalg.norm(fallback) > 1e-12 else fallback


def endpoints_along_axis(points: np.ndarray, axis: np.ndarray, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    axis = normalize(axis)
    if center is None:
        center = points.mean(axis=0)
    projections = (points - center) @ axis
    return points[int(np.argmin(projections))], points[int(np.argmax(projections))]


def nearby_persistent_points(
    cluster_points: np.ndarray,
    post_xyz: np.ndarray,
    margin: float = 0.08,
) -> np.ndarray:
    cluster_points = np.asarray(cluster_points, dtype=np.float64)
    post_xyz = np.asarray(post_xyz, dtype=np.float64)
    if len(cluster_points) == 0 or len(post_xyz) == 0:
        return np.empty((0, 3), dtype=np.float64)

    mins = cluster_points.min(axis=0) - margin
    maxs = cluster_points.max(axis=0) + margin
    in_region = np.all((post_xyz >= mins) & (post_xyz <= maxs), axis=1)
    candidates = post_xyz[in_region]
    if len(candidates) == 0:
        return candidates

    center = cluster_points.mean(axis=0)
    diameter = float(np.linalg.norm(cluster_points.max(axis=0) - cluster_points.min(axis=0)))
    keep = np.linalg.norm(candidates - center, axis=1) > max(0.01, 0.20 * diameter)
    return candidates[keep]


def nearest_pair(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if len(source_points) == 0 or len(target_points) == 0:
        return None
    tree = cKDTree(target_points)
    distances, indices = tree.query(source_points, k=1, workers=-1)
    source_idx = int(np.argmin(distances))
    target_idx = int(indices[source_idx])
    return source_points[source_idx], target_points[target_idx], float(distances[source_idx])


def nearest_distance_to_set(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    if len(reference) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    tree = cKDTree(reference)
    distances, _ = tree.query(points, k=1, workers=-1)
    return distances
