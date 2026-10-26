from __future__ import annotations

from typing import Any

import numpy as np


def _normalized_excess(distance: np.ndarray, threshold: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    if threshold <= 0:
        return np.zeros_like(distance, dtype=np.float64)
    score = np.clip((distance - threshold) / threshold, 0.0, 3.0) / 3.0
    score[~np.isfinite(score)] = 1.0
    return score


def combine_change_scores(
    nn_dist: np.ndarray,
    nn_threshold: float,
    m3c2_dist: np.ndarray | None,
    m3c2_threshold: float,
    config: dict[str, Any],
    density_score: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scfg = config.get("scoring", config)
    nn_score = _normalized_excess(nn_dist, nn_threshold)
    nn_weight = float(scfg.get("nn_weight", 0.55))
    diagnostics: dict[str, Any] = {
        "nn_threshold_m": float(nn_threshold),
        "m3c2_available": m3c2_dist is not None,
        "density_available": density_score is not None,
    }

    combined = nn_weight * nn_score
    weight_sum = np.full_like(nn_score, nn_weight, dtype=np.float64)
    component_scores = [nn_score]

    m3c2_weight = float(scfg.get("m3c2_weight", 0.45))
    if m3c2_dist is not None and m3c2_weight > 0:
        m3c2_score = _normalized_excess(np.abs(m3c2_dist), m3c2_threshold)
        m3c2_available = np.isfinite(m3c2_dist)
        m3c2_score[~m3c2_available] = 0.0
        component_scores.append(m3c2_score)
        combined += m3c2_weight * m3c2_score
        weight_sum[m3c2_available] += m3c2_weight
        diagnostics["m3c2_threshold_m"] = float(m3c2_threshold)
        diagnostics["n_points_with_m3c2_score"] = int(m3c2_available.sum())
    elif m3c2_dist is not None:
        diagnostics["m3c2_threshold_m"] = float(m3c2_threshold)
        diagnostics["n_points_with_m3c2_score"] = int(np.isfinite(m3c2_dist).sum())

    density_weight = float(scfg.get("density_weight", 0.0))
    if density_score is not None and density_weight > 0:
        density = np.asarray(density_score, dtype=np.float64)
        if density.shape != nn_score.shape:
            raise ValueError("density_score must have same shape as nn_dist")
        density = np.clip(density, 0.0, 1.0)
        component_scores.append(density)
        combined += density_weight * density
        weight_sum += density_weight
        diagnostics["n_points_with_density_score"] = int(np.isfinite(density).sum())

    if scfg.get("fusion", "weighted_average") == "max":
        combined = np.maximum.reduce(component_scores)
    else:
        combined = combined / np.maximum(weight_sum, 1e-12)

    min_score = float(scfg.get("min_combined_score", 0.5))
    removed_mask = combined >= min_score
    diagnostics["min_combined_score"] = min_score
    diagnostics["n_removed_candidates"] = int(removed_mask.sum())
    return combined, diagnostics
