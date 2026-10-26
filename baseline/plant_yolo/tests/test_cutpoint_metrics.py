import numpy as np

from yolo_aria_harvest_poc.evaluation import aggregate_cutpoint_metrics, compute_cutpoint_metrics


def test_thresholded_one_to_one_matching_penalizes_duplicate_predictions():
    gt = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    pred = np.array(
        [
            [0.005, 0.0, 0.0],
            [0.010, 0.0, 0.0],
            [0.101, 0.0, 0.0],
        ]
    )

    metrics = compute_cutpoint_metrics(pred, gt, tau_m=0.02)

    assert metrics["TP"] == 2
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0
    assert metrics["precision_at_2cm"] == 2 / 3
    assert metrics["recall_at_2cm"] == 1.0
    assert len(metrics["matches"]) == 2


def test_predictions_outside_two_centimeters_are_not_true_positives():
    gt = np.array([[0.0, 0.0, 0.0]])
    pred = np.array([[0.021, 0.0, 0.0]])

    metrics = compute_cutpoint_metrics(pred, gt, tau_m=0.02)

    assert metrics["TP"] == 0
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["precision_at_2cm"] == 0.0
    assert metrics["recall_at_2cm"] == 0.0
    assert metrics["gt_to_nearest_mean_distance_cm"] == 2.1


def test_empty_predictions_have_null_gt_to_nearest_distance():
    gt = np.array([[0.0, 0.0, 0.0]])
    pred = np.zeros((0, 3))

    metrics = compute_cutpoint_metrics(pred, gt, tau_m=0.02)

    assert metrics["TP"] == 0
    assert metrics["FP"] == 0
    assert metrics["FN"] == 1
    assert metrics["gt_to_nearest_mean_distance_cm"] is None
    assert metrics["gt_to_nearest_distances_cm"] == [None]


def test_aggregate_uses_micro_counts_and_distance_distributions():
    scene_a = compute_cutpoint_metrics(
        pred_points=np.array([[0.005, 0.0, 0.0], [0.20, 0.0, 0.0]]),
        gt_points=np.array([[0.0, 0.0, 0.0]]),
        tau_m=0.02,
    )
    scene_b = compute_cutpoint_metrics(
        pred_points=np.array([[0.0, 0.0, 0.0]]),
        gt_points=np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]),
        tau_m=0.02,
    )

    aggregate = aggregate_cutpoint_metrics([scene_a, scene_b], tau_m=0.02)

    assert aggregate["total_TP"] == 2
    assert aggregate["total_FP"] == 1
    assert aggregate["total_FN"] == 1
    assert aggregate["micro_precision_at_2cm"] == 2 / 3
    assert aggregate["micro_recall_at_2cm"] == 2 / 3
    assert aggregate["micro_f1_at_2cm"] == 2 / 3
    assert aggregate["matched_localization_mean_cm"] == 0.25


def test_non_integer_tolerance_uses_dynamic_metric_keys():
    metrics = compute_cutpoint_metrics(
        pred_points=np.array([[0.034, 0.0, 0.0]]),
        gt_points=np.array([[0.0, 0.0, 0.0]]),
        tau_m=0.035,
    )
    aggregate = aggregate_cutpoint_metrics([metrics], tau_m=0.035)

    assert metrics["precision_at_3p5cm"] == 1.0
    assert metrics["recall_at_3p5cm"] == 1.0
    assert metrics["f1_at_3p5cm"] == 1.0
    assert aggregate["micro_precision_at_3p5cm"] == 1.0
