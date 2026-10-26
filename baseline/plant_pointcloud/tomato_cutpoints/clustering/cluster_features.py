from __future__ import annotations

from typing import Any

import numpy as np


def pca_features(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 3:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    centered = xyz - xyz.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    return values, vectors


def compute_cluster_features(
    cluster_id: int,
    points: np.ndarray,
    original_indices: np.ndarray,
    nn_distances: np.ndarray | None = None,
    m3c2_distances: np.ndarray | None = None,
    combined_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = maxs - mins
    centroid = points.mean(axis=0)
    eigvals, eigvecs = pca_features(points)
    eps = 1e-12
    bbox_diagonal = float(np.linalg.norm(dims))
    approximate_diameter = float(np.max(dims))
    compactness = float(np.max(dims) / max(np.min(dims[dims > eps]) if np.any(dims > eps) else eps, eps))

    feature: dict[str, Any] = {
        "cluster_id": int(cluster_id),
        "n_points": int(len(points)),
        "centroid": centroid,
        "min_xyz": mins,
        "max_xyz": maxs,
        "bbox_dimensions": dims,
        "pca_eigenvalues": eigvals,
        "pca_eigenvectors": eigvecs,
        "linearity": float(eigvals[0] / (eigvals[1] + eps)),
        "planarity": float(eigvals[1] / (eigvals[2] + eps)),
        "compactness": compactness,
        "bbox_diagonal": bbox_diagonal,
        "approximate_diameter": approximate_diameter,
        "point_indices": np.asarray(original_indices, dtype=np.int64),
        "points": points,
    }
    if nn_distances is not None and len(nn_distances):
        feature["mean_nn_distance"] = float(np.nanmean(nn_distances))
    else:
        feature["mean_nn_distance"] = None
    if m3c2_distances is not None and len(m3c2_distances):
        finite = m3c2_distances[np.isfinite(m3c2_distances)]
        feature["mean_m3c2_distance"] = float(np.nanmean(finite)) if len(finite) else None
    else:
        feature["mean_m3c2_distance"] = None
    if combined_scores is not None and len(combined_scores):
        feature["mean_combined_score"] = float(np.nanmean(combined_scores))
    else:
        feature["mean_combined_score"] = None
    return feature


def serializable_cluster_feature(feature: dict[str, Any]) -> dict[str, Any]:
    out = dict(feature)
    out.pop("points", None)
    out.pop("point_indices", None)
    out.pop("target_selection_score", None)
    for key in list(out):
        if key.startswith("_"):
            out.pop(key)
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
    return out
