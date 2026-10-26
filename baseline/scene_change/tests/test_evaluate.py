import json

import numpy as np
from PIL import Image

from egoscene.evaluate import confusion, evaluate_predictions, metrics_from_confusion


def test_binary_metrics():
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    gt = np.array([[1, 0], [1, 0]], dtype=bool)
    assert confusion(pred, gt) == (1, 1, 1, 1)
    metrics = metrics_from_confusion(1, 1, 1, 1)
    assert metrics["iou"] == 1 / 3
    assert metrics["f1"] == 0.5


def test_empty_masks_are_correct():
    metrics = metrics_from_confusion(0, 0, 0, 16)
    assert metrics["iou"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_incomplete_evaluation_scores_only_completed_scenes(tmp_path):
    gt_root = tmp_path / "gt"
    pred_root = tmp_path / "pred"
    tasks = []
    for scene_id in ("complete", "pending"):
        relative = f"label_maps/{scene_id}/frame.png"
        path = gt_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2), 255).save(path)
        tasks.append(
            {
                "task_id": f"{scene_id}__frame",
                "scene_id": scene_id,
                "task_kind": "deleafing",
                "image_name": "frame.jpg",
                "label_map": relative,
            }
        )
    (gt_root / "dataset.json").write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    complete_root = pred_root / "complete"
    (complete_root / "masks").mkdir(parents=True)
    Image.new("L", (2, 2), 255).save(complete_root / "masks" / "frame.png")
    (complete_root / "metadata.json").write_text("{}", encoding="utf-8")

    summary = evaluate_predictions(gt_root, pred_root, allow_incomplete_predictions=True)

    assert summary["image_count"] == 1
    assert summary["expected_image_count"] == 2
    assert summary["scene_intervention_count"] == 1
    assert summary["expected_scene_intervention_count"] == 2
    assert summary["prediction_coverage"] == 0.5
    assert summary["views_per_scene_intervention"] == 1
