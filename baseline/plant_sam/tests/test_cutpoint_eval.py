import json

import numpy as np

from sam_aria_change_baseline.evaluate_cut_points import (
    evaluate_cut_point_sets,
    evaluate_scene,
    extract_prediction_points,
    load_prediction_points,
    match_cut_points,
    normalize_scene_id,
    parse_gt_label_file,
)


def test_one_to_one_matching_counts_extra_close_predictions_as_fp():
    gt_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    pred_points = [
        [0.005, 0.0, 0.0],
        [0.006, 0.0, 0.0],
        [1.01, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]

    metrics = evaluate_scene("scene0", pred_points, gt_points, tau_m=0.02)

    assert metrics["TP"] == 2
    assert metrics["FP"] == 2
    assert metrics["FN"] == 0
    assert metrics["Precision@2cm"] == 0.5
    assert metrics["Recall@2cm"] == 1.0
    assert np.isclose(metrics["F1@2cm"], 2.0 / 3.0)
    assert np.isclose(metrics["matched_mean_distance"], 0.0075)


def test_match_cut_points_rejects_pairs_outside_tau():
    result = match_cut_points([[0.03, 0.0, 0.0]], [[0.0, 0.0, 0.0]], tau_m=0.02)

    assert result["matches"] == []
    assert result["matched_distances"] == []


def test_gt_to_nearest_distance_is_averaged_over_all_gt_points():
    metrics = evaluate_scene(
        "scene0",
        pred_points=[[0.01, 0.0, 0.0]],
        gt_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        tau_m=0.02,
    )

    assert metrics["TP"] == 1
    assert metrics["FN"] == 1
    assert np.isclose(metrics["gt_to_nearest_mean_distance"], 0.5)


def test_parse_gt_label_file_uses_requested_task_and_tolerates_malformed_point_lines(tmp_path):
    label_path = tmp_path / "3d points labels.txt"
    label_path.write_text(
        "\n".join(
            [
                "1. mps_scene-a_vrs",
                "deleafing",
                "[0.1, -0.2, 0.3]",
                "[0.4, 0.5, 0.6",
                "harvesting",
                "[9.0, 9.0, 9.0]",
                "2. scene-b.vrs",
                "deleafing",
                "[1.0, 2.0, 3.0]",
            ]
        ),
        encoding="utf-8",
    )

    gt = parse_gt_label_file(label_path, task="deleafing")

    assert set(gt) == {"scene-a", "scene-b"}
    assert gt["scene-a"] == [[0.1, -0.2, 0.3], [0.4, 0.5, 0.6]]
    assert gt["scene-b"] == [[1.0, 2.0, 3.0]]


def test_extract_prediction_points_uses_target_points_by_default():
    data = {
        "predictions": [
            {
                "task": "deleafing",
                "target_point_3d_world": [0.0, 0.0, 0.0],
                "candidate_target_points_3d_world": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            },
            {"task": "harvesting", "target_point_3d_world": [9.0, 9.0, 9.0]},
        ]
    }

    assert extract_prediction_points(data, task="deleafing") == [[0.0, 0.0, 0.0]]
    assert extract_prediction_points(data, task="deleafing", include_candidate_points=True) == [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_load_prediction_points_maps_scene_id_from_parent_directory(tmp_path):
    scene_dir = tmp_path / "scene-001"
    scene_dir.mkdir()
    pred_path = scene_dir / "final_predictions.json"
    pred_path.write_text(
        json.dumps({"predictions": [{"task": "deleafing", "target_point_3d_world": [0, 0, 0]}]}),
        encoding="utf-8",
    )

    pred_by_scene, warnings = load_prediction_points(tmp_path, task="deleafing")

    assert warnings == []
    assert pred_by_scene == {"scene-001": [[0.0, 0.0, 0.0]]}


def test_aggregate_metrics_use_micro_counts():
    gt_by_scene = {
        "scene0": [[0.0, 0.0, 0.0]],
        "scene1": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    }
    pred_by_scene = {
        "scene0": [[0.01, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "scene1": [[1.5, 0.0, 0.0]],
    }

    per_scene, aggregate = evaluate_cut_point_sets(gt_by_scene, pred_by_scene, tau_m=0.02)

    assert len(per_scene) == 2
    assert aggregate["total_TP"] == 1
    assert aggregate["total_FP"] == 2
    assert aggregate["total_FN"] == 2
    assert np.isclose(aggregate["micro_Precision@2cm"], 1.0 / 3.0)
    assert np.isclose(aggregate["micro_Recall@2cm"], 1.0 / 3.0)
    assert np.isclose(aggregate["micro_F1@2cm"], 1.0 / 3.0)


def test_metric_labels_follow_threshold():
    metrics = evaluate_scene(
        "scene0",
        pred_points=[[0.03, 0.0, 0.0]],
        gt_points=[[0.0, 0.0, 0.0]],
        tau_m=0.035,
    )

    assert metrics["Precision@3.5cm"] == 1.0
    assert metrics["Recall@3.5cm"] == 1.0
    assert metrics["F1@3.5cm"] == 1.0


def test_normalize_scene_id_handles_benchmark_path_variants():
    assert normalize_scene_id("/tmp/mps_abc-123_vrs") == "abc-123"
    assert normalize_scene_id("/tmp/abc-123.vrs") == "abc-123"
    assert normalize_scene_id("/tmp/abc-123.json") == "abc-123"
