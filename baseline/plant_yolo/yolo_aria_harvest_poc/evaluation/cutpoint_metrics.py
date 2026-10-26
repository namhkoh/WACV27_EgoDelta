"""Thresholded one-to-one metrics for unordered 3D cut-point predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CutPointMatch:
    pred_index: int
    gt_index: int
    distance_m: float

    def to_dict(self, pred_points: np.ndarray, gt_points: np.ndarray) -> dict[str, Any]:
        return {
            "pred_index": int(self.pred_index),
            "gt_index": int(self.gt_index),
            "distance_m": float(self.distance_m),
            "distance_cm": float(self.distance_m * 100.0),
            "pred_point_world": pred_points[self.pred_index].tolist(),
            "gt_point_world": gt_points[self.gt_index].tolist(),
        }


def compute_cutpoint_metrics(
    pred_points: np.ndarray | list[list[float]],
    gt_points: np.ndarray | list[list[float]],
    tau_m: float = 0.02,
) -> dict[str, Any]:
    """Compute one-scene cut-point detection and localization metrics.

    Points are unordered. A true positive is a one-to-one prediction/GT match
    whose Euclidean distance is at most ``tau_m``.
    """

    pred = _as_points(pred_points)
    gt = _as_points(gt_points)
    matches = match_cutpoints(pred, gt, tau_m=tau_m)
    matched_pred = {match.pred_index for match in matches}
    matched_gt = {match.gt_index for match in matches}

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_f1(precision, recall)
    tau_suffix = _tau_key_suffix(tau_m)
    matched_distances = np.asarray([match.distance_m for match in matches], dtype=float)
    gt_nearest_distances = gt_to_nearest_distances(pred, gt)

    metrics: dict[str, Any] = {
        "tau_m": float(tau_m),
        "tau_cm": float(tau_m * 100.0),
        "num_gt": int(len(gt)),
        "num_pred": int(len(pred)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision_at_tau": precision,
        "recall_at_tau": recall,
        "f1_at_tau": f1,
        "matched_mean_distance_m": _mean_or_none(matched_distances),
        "matched_mean_distance_cm": _mean_or_none(matched_distances * 100.0),
        "gt_to_nearest_mean_distance_m": _mean_or_none(gt_nearest_distances),
        "gt_to_nearest_mean_distance_cm": _mean_or_none(gt_nearest_distances * 100.0),
        "matches": [match.to_dict(pred, gt) for match in matches],
        "unmatched_pred_indices": [idx for idx in range(len(pred)) if idx not in matched_pred],
        "unmatched_gt_indices": [idx for idx in range(len(gt)) if idx not in matched_gt],
        "gt_to_nearest_distances_m": _json_number_list(gt_nearest_distances),
        "gt_to_nearest_distances_cm": _json_number_list(gt_nearest_distances * 100.0),
    }
    metrics[f"precision_at_{tau_suffix}"] = precision
    metrics[f"recall_at_{tau_suffix}"] = recall
    metrics[f"f1_at_{tau_suffix}"] = f1
    return metrics


def match_cutpoints(pred_points: np.ndarray, gt_points: np.ndarray, tau_m: float) -> list[CutPointMatch]:
    pred = _as_points(pred_points)
    gt = _as_points(gt_points)
    if len(pred) == 0 or len(gt) == 0:
        return []
    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    valid = distances <= float(tau_m)
    if not valid.any():
        return []

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        large_cost = max(float(tau_m) * 1_000.0, float(np.nanmax(distances)) + 1.0, 1.0)
        cost = np.where(valid, distances, large_cost)
        pred_indices, gt_indices = linear_sum_assignment(cost)
        matches = [
            CutPointMatch(int(pred_idx), int(gt_idx), float(distances[pred_idx, gt_idx]))
            for pred_idx, gt_idx in zip(pred_indices, gt_indices)
            if valid[pred_idx, gt_idx]
        ]
        matches.sort(key=lambda match: (match.gt_index, match.pred_index))
        return matches
    except Exception:
        return _greedy_thresholded_matching(distances, valid)


def gt_to_nearest_distances(pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    pred = _as_points(pred_points)
    gt = _as_points(gt_points)
    if len(gt) == 0:
        return np.zeros((0,), dtype=float)
    if len(pred) == 0:
        return np.full((len(gt),), np.inf, dtype=float)
    distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    return distances.min(axis=1)


def aggregate_cutpoint_metrics(scene_metrics: list[dict[str, Any]], tau_m: float = 0.02) -> dict[str, Any]:
    total_tp = int(sum(int(item.get("TP", 0)) for item in scene_metrics))
    total_fp = int(sum(int(item.get("FP", 0)) for item in scene_metrics))
    total_fn = int(sum(int(item.get("FN", 0)) for item in scene_metrics))
    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_f1(precision, recall)
    tau_suffix = _tau_key_suffix(tau_m)

    matched_distances_cm = np.asarray(
        [
            float(match["distance_cm"])
            for item in scene_metrics
            for match in item.get("matches", [])
            if np.isfinite(float(match["distance_cm"]))
        ],
        dtype=float,
    )
    gt_nearest_cm = np.asarray(
        [
            float(distance)
            for item in scene_metrics
            for distance in item.get("gt_to_nearest_distances_cm", [])
            if distance is not None and np.isfinite(float(distance))
        ],
        dtype=float,
    )

    metrics: dict[str, Any] = {
        "tau_m": float(tau_m),
        "tau_cm": float(tau_m * 100.0),
        "num_scenes": int(len(scene_metrics)),
        "total_gt": int(sum(int(item.get("num_gt", 0)) for item in scene_metrics)),
        "total_pred": int(sum(int(item.get("num_pred", 0)) for item in scene_metrics)),
        "total_TP": total_tp,
        "total_FP": total_fp,
        "total_FN": total_fn,
        "micro_precision_at_tau": precision,
        "micro_recall_at_tau": recall,
        "micro_f1_at_tau": f1,
        "matched_localization_mean_cm": _mean_or_none(matched_distances_cm),
        "matched_localization_median_cm": _median_or_none(matched_distances_cm),
        "gt_to_nearest_mean_cm": _mean_or_none(gt_nearest_cm),
        "gt_to_nearest_median_cm": _median_or_none(gt_nearest_cm),
    }
    metrics[f"micro_precision_at_{tau_suffix}"] = precision
    metrics[f"micro_recall_at_{tau_suffix}"] = recall
    metrics[f"micro_f1_at_{tau_suffix}"] = f1
    return metrics


def _greedy_thresholded_matching(distances: np.ndarray, valid: np.ndarray) -> list[CutPointMatch]:
    pairs = [
        (float(distances[pred_idx, gt_idx]), pred_idx, gt_idx)
        for pred_idx in range(distances.shape[0])
        for gt_idx in range(distances.shape[1])
        if valid[pred_idx, gt_idx]
    ]
    pairs.sort()
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[CutPointMatch] = []
    for distance, pred_idx, gt_idx in pairs:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append(CutPointMatch(int(pred_idx), int(gt_idx), float(distance)))
    matches.sort(key=lambda match: (match.gt_index, match.pred_index))
    return matches


def _as_points(points: np.ndarray | list[list[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.zeros((0, 3), dtype=float)
    array = array.reshape(-1, 3)
    if not np.isfinite(array).all():
        raise ValueError("Cut-point arrays must contain only finite coordinates")
    return array


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    return float(np.mean(finite))


def _median_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def _json_number_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values, dtype=float).reshape(-1):
        out.append(float(value) if np.isfinite(value) else None)
    return out


def _tau_key_suffix(tau_m: float) -> str:
    cm = float(tau_m) * 100.0
    if abs(cm - round(cm)) < 1e-9:
        return f"{int(round(cm))}cm"
    text = f"{cm:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}cm"
