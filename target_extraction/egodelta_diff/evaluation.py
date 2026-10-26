"""One-to-one Euclidean target evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_points(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold_m: float = 0.03,
) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.float64).reshape((-1, 3))
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape((-1, 3))
    if not len(predictions):
        return _metrics(0, 0, len(ground_truth), [])
    if not len(ground_truth):
        return _metrics(0, len(predictions), 0, [])

    distance = np.linalg.norm(
        predictions[:, None, :] - ground_truth[None, :, :], axis=2
    )
    invalid = float(distance.max() + 2.0)
    rows, columns = linear_sum_assignment(
        np.where(distance <= threshold_m, distance, invalid)
    )
    valid = distance[rows, columns] <= threshold_m
    tp = int(valid.sum())
    errors = distance[rows[valid], columns[valid]].tolist()
    return _metrics(tp, len(predictions) - tp, len(ground_truth) - tp, errors)


def aggregate_evaluations(evaluations: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(evaluations)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    errors = [float(value) for row in rows for value in row["matched_errors_m"]]
    return _metrics(tp, fp, fn, errors)


def _metrics(tp: int, fp: int, fn: int, errors: list[float]) -> dict[str, object]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "predictions": tp + fp,
        "gt": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_error_cm": float(np.median(errors) * 100.0) if errors else None,
        "matched_errors_m": errors,
    }
