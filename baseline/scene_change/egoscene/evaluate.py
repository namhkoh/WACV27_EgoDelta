from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def confusion(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    union = tp + fp + fn
    predicted = tp + fp
    positive = tp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": 1.0 if union == 0 else tp / union,
        "f1": 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn),
        "precision": 1.0 if predicted == 0 and positive == 0 else (0.0 if predicted == 0 else tp / predicted),
        "recall": 1.0 if positive == 0 else tp / positive,
    }


def evaluate_predictions(
    gt_export: str | Path,
    predictions_root: str | Path,
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    allow_incomplete_predictions: bool = False,
) -> dict[str, Any]:
    gt_export = Path(gt_export).expanduser().resolve()
    predictions_root = Path(predictions_root).expanduser().resolve()
    dataset = json.loads((gt_export / "dataset.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[int]] = {}
    missing_tasks: list[str] = []
    for task in dataset["tasks"]:
        scene_root = predictions_root / task["scene_id"]
        if allow_incomplete_predictions and not (scene_root / "metadata.json").is_file():
            missing_tasks.append(task["task_id"])
            continue
        gt = np.asarray(Image.open(gt_export / task["label_map"]).convert("L")) > 0
        pred_path = scene_root / "masks" / f"{Path(task['image_name']).stem}.png"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")
        pred = np.asarray(Image.open(pred_path).convert("L")) > 0
        counts = confusion(pred, gt)
        metric = metrics_from_confusion(*counts)
        rows.append(
            {
                "task_id": task["task_id"],
                "scene_id": task["scene_id"],
                "task_kind": task["task_kind"],
                "image_name": task["image_name"],
                **metric,
            }
        )
        accum = aggregate.setdefault(task["scene_id"], [0, 0, 0, 0])
        for i, value in enumerate(counts):
            accum[i] += value
    if not rows:
        raise ValueError("No completed predictions overlap the ground-truth export")

    scene_rows = []
    scene_kind = {row["scene_id"]: row["task_kind"] for row in rows}
    for scene_id, counts in sorted(aggregate.items()):
        scene_rows.append({"scene_id": scene_id, "task_kind": scene_kind[scene_id], **metrics_from_confusion(*counts)})
    views_per_scene = [sum(row["scene_id"] == scene["scene_id"] for row in rows) for scene in scene_rows]
    unique_view_counts = sorted(set(views_per_scene))
    expected_scene_count = len({task["scene_id"] for task in dataset["tasks"]})
    summary = {
        "schema_version": 1,
        "evaluation_unit": (
            "scene-intervention (metrics pool the available annotated views within each unit, "
            "then macro-average units)"
        ),
        "views_per_scene_intervention": (
            unique_view_counts[0] if len(unique_view_counts) == 1 else unique_view_counts
        ),
        "expected_image_count": len(dataset["tasks"]),
        "image_count": len(rows),
        "missing_prediction_count": len(missing_tasks),
        "prediction_coverage": len(rows) / len(dataset["tasks"]),
        "expected_scene_intervention_count": expected_scene_count,
        "scene_intervention_count": len(scene_rows),
        "macro_scene": _macro(scene_rows),
        "macro_image": _macro(rows),
        "by_task_kind": {
            kind: _macro([row for row in scene_rows if row["task_kind"] == kind])
            for kind in sorted({row["task_kind"] for row in scene_rows})
        },
        "scenes": scene_rows,
    }
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: float("nan") for key in ("iou", "f1", "precision", "recall")}
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in ("iou", "f1", "precision", "recall")}
