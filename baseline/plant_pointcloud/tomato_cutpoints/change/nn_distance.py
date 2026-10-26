from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def compute_pre_to_post_nn(pre_xyz: np.ndarray, post_xyz: np.ndarray) -> np.ndarray:
    pre_xyz = np.asarray(pre_xyz, dtype=np.float64)
    post_xyz = np.asarray(post_xyz, dtype=np.float64)
    if len(pre_xyz) == 0:
        return np.empty(0, dtype=np.float64)
    if len(post_xyz) == 0:
        return np.full(len(pre_xyz), np.inf, dtype=np.float64)
    tree = cKDTree(post_xyz)
    dists, _ = tree.query(pre_xyz, k=1, workers=-1)
    return dists.astype(np.float64)


def auto_nn_threshold(
    distances: np.ndarray,
    stable_mask: np.ndarray | None,
    base_threshold: float,
    k_mad: float,
) -> tuple[float, dict[str, Any]]:
    distances = np.asarray(distances, dtype=np.float64)
    if stable_mask is None or stable_mask.sum() < 10:
        return float(base_threshold), {
            "threshold_m": float(base_threshold),
            "auto_used": False,
            "reason": "no_stable_roi_or_too_few_points",
        }

    stable_dist = distances[stable_mask]
    stable_dist = stable_dist[np.isfinite(stable_dist)]
    if len(stable_dist) < 10:
        return float(base_threshold), {
            "threshold_m": float(base_threshold),
            "auto_used": False,
            "reason": "too_few_finite_stable_distances",
        }
    median = float(np.median(stable_dist))
    mad = float(np.median(np.abs(stable_dist - median)))
    auto = median + float(k_mad) * 1.4826 * mad
    threshold = max(float(base_threshold), auto)
    return threshold, {
        "threshold_m": threshold,
        "auto_used": True,
        "base_threshold_m": float(base_threshold),
        "stable_median_m": median,
        "stable_mad_m": mad,
        "auto_threshold_m": auto,
        "n_stable": int(len(stable_dist)),
    }
