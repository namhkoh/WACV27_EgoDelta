from __future__ import annotations

import pytest

from tomato_cutpoints.eval.metrics import aggregate_cut_point_metrics, evaluate_cut_points


def test_duplicate_predictions_near_one_gt_count_as_false_positives() -> None:
    metrics = evaluate_cut_points(
        pred_points=[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]],
        gt_points=[[0.0, 0.0, 0.0]],
        threshold_m=0.02,
    )

    assert metrics["TP"] == 1
    assert metrics["FP"] == 2
    assert metrics["FN"] == 0
    assert metrics["Precision@2cm"] == pytest.approx(1.0 / 3.0)
    assert metrics["Recall@2cm"] == pytest.approx(1.0)
    assert metrics["F1@2cm"] == pytest.approx(0.5)
    assert metrics["matched_mean_distance"] == pytest.approx(0.0)


def test_predictions_outside_threshold_are_not_true_positives() -> None:
    metrics = evaluate_cut_points(
        pred_points=[[0.03, 0.0, 0.0]],
        gt_points=[[0.0, 0.0, 0.0]],
        threshold_m=0.02,
    )

    assert metrics["TP"] == 0
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["F1@2cm"] == pytest.approx(0.0)
    assert metrics["matched_mean_distance"] is None
    assert metrics["gt_to_nearest_mean_distance"] == pytest.approx(0.03)


def test_aggregate_uses_micro_counts_and_all_distances() -> None:
    scene_a = evaluate_cut_points(
        pred_points=[[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]],
        gt_points=[[0.0, 0.0, 0.0]],
        threshold_m=0.02,
    )
    scene_b = evaluate_cut_points(
        pred_points=[],
        gt_points=[[1.0, 0.0, 0.0]],
        threshold_m=0.02,
    )

    agg = aggregate_cut_point_metrics([scene_a, scene_b], threshold_m=0.02)

    assert agg["TP"] == 1
    assert agg["FP"] == 1
    assert agg["FN"] == 1
    assert agg["micro Precision@2cm"] == pytest.approx(0.5)
    assert agg["micro Recall@2cm"] == pytest.approx(0.5)
    assert agg["micro F1@2cm"] == pytest.approx(0.5)
