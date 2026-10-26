from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _radius_counts(tree: cKDTree, points: np.ndarray, radius: float) -> np.ndarray:
    try:
        return np.asarray(tree.query_ball_point(points, r=radius, return_length=True, workers=-1), dtype=np.int64)
    except TypeError:
        return np.asarray([len(indices) for indices in tree.query_ball_point(points, r=radius)], dtype=np.int64)


def compute_density_drop_score(
    pre_xyz: np.ndarray,
    post_xyz: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score pre points whose local neighborhood loses occupancy in post.

    NN distance is intentionally pointwise. A removed object can disappear while
    nearby plant structure remains close enough to keep NN distance small. This
    score compares local pre support with local post support around each pre point.
    """

    pre_xyz = np.asarray(pre_xyz, dtype=np.float64)
    post_xyz = np.asarray(post_xyz, dtype=np.float64)
    cfg = config.get("density", config)

    if len(pre_xyz) == 0:
        return np.empty(0, dtype=np.float64), {
            "enabled": bool(cfg.get("enabled", False)),
            "radius_m": float(cfg.get("radius_m", 0.04)),
            "n_density_candidates": 0,
        }
    if len(post_xyz) == 0:
        return np.ones(len(pre_xyz), dtype=np.float64), {
            "enabled": bool(cfg.get("enabled", False)),
            "radius_m": float(cfg.get("radius_m", 0.04)),
            "n_density_candidates": int(len(pre_xyz)),
            "reason": "empty_post_cloud",
        }

    radius = float(cfg.get("radius_m", 0.04))
    min_pre_neighbors = int(cfg.get("min_pre_neighbors", 3))
    min_drop_fraction = float(cfg.get("min_drop_fraction", 0.35))
    min_drop_count = int(cfg.get("min_drop_count", 2))
    max_post_neighbors = int(cfg.get("max_post_neighbors", 2))

    pre_tree = cKDTree(pre_xyz)
    post_tree = cKDTree(post_xyz)
    pre_counts = _radius_counts(pre_tree, pre_xyz, radius)
    post_counts = _radius_counts(post_tree, pre_xyz, radius)

    support = pre_counts >= min_pre_neighbors
    pre_counts_f = np.maximum(pre_counts.astype(np.float64), 1.0)
    ratios = post_counts[support].astype(np.float64) / np.maximum(pre_counts[support].astype(np.float64), 1.0)
    if len(ratios):
        expected_post_ratio = float(np.median(ratios))
    else:
        expected_post_ratio = float(len(post_xyz) / max(len(pre_xyz), 1))
    expected_post_ratio = max(expected_post_ratio, float(cfg.get("min_expected_post_ratio", 0.25)))

    expected_post_counts = pre_counts_f * expected_post_ratio
    drop_count = np.maximum(expected_post_counts - post_counts.astype(np.float64), 0.0)
    drop_fraction = drop_count / np.maximum(expected_post_counts, 1.0)

    fraction_score = (drop_fraction - min_drop_fraction) / max(1.0 - min_drop_fraction, 1e-6)
    fraction_score = np.clip(fraction_score, 0.0, 1.0)

    count_score = (drop_count - float(min_drop_count) + 1.0) / np.maximum(pre_counts_f, 1.0)
    count_score = np.clip(count_score, 0.0, 1.0)

    sparse_post_score = np.zeros(len(pre_xyz), dtype=np.float64)
    sparse_mask = post_counts <= max_post_neighbors
    sparse_post_score[sparse_mask] = 1.0 - (post_counts[sparse_mask] / max(float(max_post_neighbors + 1), 1.0))

    score = np.maximum.reduce([fraction_score, count_score, sparse_post_score])
    score[~support] = 0.0

    threshold = float(cfg.get("candidate_threshold", 0.25))
    diagnostics = {
        "enabled": bool(cfg.get("enabled", False)),
        "radius_m": radius,
        "min_pre_neighbors": min_pre_neighbors,
        "min_drop_fraction": min_drop_fraction,
        "min_drop_count": min_drop_count,
        "max_post_neighbors": max_post_neighbors,
        "expected_post_ratio": expected_post_ratio,
        "candidate_threshold": threshold,
        "n_density_candidates": int(np.sum(score >= threshold)),
        "pre_neighbor_count_median": float(np.median(pre_counts)),
        "post_neighbor_count_median": float(np.median(post_counts)),
        "drop_fraction_median": float(np.median(drop_fraction)),
    }
    return score.astype(np.float64), diagnostics
