#!/usr/bin/env python3
"""Evaluate unordered 3D cut-point predictions for harvest or deleafing."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from yolo_aria_harvest_poc.evaluation import aggregate_cutpoint_metrics, compute_cutpoint_metrics


DEFAULT_LABELS = "/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt"

TASK_SPECS = {
    "harvesting": {
        "label_key": "harvesting",
        "task_dir": "harvest",
        "prediction_file": "final_predictions.json",
        "list_key": "removed_tomatoes",
        "point_keys": ("cut_point_3d_world", "proxy_cut_point_3d_world"),
    },
    "deleafing": {
        "label_key": "deleafing",
        "task_dir": "deleaf",
        "prediction_file": "final_predictions_deleaf.json",
        "list_key": "removed_petioles",
        "point_keys": ("proxy_cut_point_3d_world", "cut_point_3d_world"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--task", choices=["harvesting", "harvest", "deleafing", "deleaf"], required=True)
    parser.add_argument(
        "--pred-root",
        action="append",
        required=True,
        help="Prediction root. Can be passed multiple times, e.g. shard0 and shard1.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--tau-m", action="append", type=float, default=None)
    parser.add_argument("--recording-id", action="append")
    parser.add_argument(
        "--missing-pred",
        choices=["empty", "skip", "error"],
        default="empty",
        help="How to handle a scene without a prediction JSON.",
    )
    args = parser.parse_args()

    task = canonical_task(args.task)
    spec = TASK_SPECS[task]
    taus = args.tau_m or [0.02, 0.035, 0.05]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.labels)
    recording_ids = list(labels)
    if args.recording_id:
        wanted = set(args.recording_id)
        recording_ids = [recording_id for recording_id in recording_ids if recording_id in wanted]

    pred_index = build_prediction_index([Path(root) for root in args.pred_root], spec)
    results_by_tau: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for tau_m in taus:
        tau_key = tau_key_suffix(tau_m)
        per_scene: list[dict[str, Any]] = []
        for recording_id in recording_ids:
            gt_points = np.asarray(labels[recording_id].get(spec["label_key"], []), dtype=float).reshape(-1, 3)
            pred_path = pred_index.get(recording_id)
            if pred_path is None:
                message = f"Missing prediction file for {recording_id}"
                if args.missing_pred == "error":
                    raise FileNotFoundError(message)
                if args.missing_pred == "skip":
                    warnings.append(f"Skipped scene at {tau_key}. {message}")
                    continue
                warnings.append(f"Using empty predictions at {tau_key}. {message}")
                pred_points = np.zeros((0, 3), dtype=float)
            else:
                pred_points = load_prediction_points(pred_path, spec)

            metrics = compute_cutpoint_metrics(pred_points=pred_points, gt_points=gt_points, tau_m=float(tau_m))
            metrics.update(
                {
                    "recording_id": recording_id,
                    "prediction_path": str(pred_path) if pred_path is not None else None,
                    "gt_points_world": gt_points.tolist(),
                    "pred_points_world": pred_points.tolist(),
                }
            )
            per_scene.append(metrics)

        results_by_tau[tau_key] = {
            "tau_m": float(tau_m),
            "tau_cm": float(tau_m * 100.0),
            "aggregate": aggregate_cutpoint_metrics(per_scene, tau_m=float(tau_m)),
            "per_scene": per_scene,
        }

    result = {
        "task": task,
        "metric": "thresholded_3d_cutpoint_detection",
        "label_frame": "full_mps_world",
        "labels_path": args.labels,
        "pred_roots": args.pred_root,
        "thresholds": results_by_tau,
        "warnings": sorted(set(warnings)),
    }
    json_path = out_dir / f"{task}_cutpoint_metrics.json"
    per_scene_csv = out_dir / f"{task}_cutpoint_metrics_per_scene.csv"
    aggregate_csv = out_dir / f"{task}_cutpoint_metrics_aggregate.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_per_scene_csv(results_by_tau, per_scene_csv)
    write_aggregate_csv(results_by_tau, aggregate_csv)

    print(json.dumps({"json_path": str(json_path), "aggregate_csv": str(aggregate_csv), "thresholds": {
        key: value["aggregate"] for key, value in results_by_tau.items()
    }}, indent=2))
    return 0


def canonical_task(task: str) -> str:
    if task == "harvest":
        return "harvesting"
    if task == "deleaf":
        return "deleafing"
    return task


def parse_labels(path: str | Path) -> dict[str, dict[str, list[list[float]]]]:
    entries: dict[str, dict[str, list[list[float]]]] = {}
    current_id: str | None = None
    current_task: str | None = None
    header_re = re.compile(r"^\s*\d+\.\s+([^\s]+)\.vrs\s*$")
    point_re = re.compile(r"\[([^\]]+)\]")
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = header_re.match(line)
        if header:
            current_id = header.group(1)
            entries[current_id] = {"deleafing": [], "harvesting": []}
            current_task = None
            continue
        if line in {"deleafing", "harvesting"}:
            current_task = line
            continue
        match = point_re.search(line)
        if match and current_id and current_task:
            values = [float(item.strip()) for item in match.group(1).split(",")]
            if len(values) != 3:
                raise ValueError(f"Expected a 3D point in {line}")
            entries[current_id][current_task].append(values)
    return entries


def build_prediction_index(pred_roots: list[Path], spec: dict[str, Any]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    prediction_file = spec["prediction_file"]
    task_dir = spec["task_dir"]
    for root in pred_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(f"**/{task_dir}/{prediction_file}")):
            recording_id = path.parent.parent.name
            index.setdefault(recording_id, path)
    return index


def load_prediction_points(path: str | Path, spec: dict[str, Any]) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = []
    for item in data.get(spec["list_key"], []):
        for key in spec["point_keys"]:
            value = item.get(key)
            if value is not None:
                points.append(value)
                break
    return np.asarray(points, dtype=float).reshape(-1, 3)


def write_per_scene_csv(results_by_tau: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "tau_cm",
        "recording_id",
        "num_gt",
        "num_pred",
        "TP",
        "FP",
        "FN",
        "precision_at_tau",
        "recall_at_tau",
        "f1_at_tau",
        "matched_mean_distance_cm",
        "gt_to_nearest_mean_distance_cm",
        "prediction_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for threshold in results_by_tau.values():
            for item in threshold["per_scene"]:
                row = {field: item.get(field) for field in fields}
                row["tau_cm"] = threshold["tau_cm"]
                writer.writerow(row)


def write_aggregate_csv(results_by_tau: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "tau_cm",
        "num_scenes",
        "total_gt",
        "total_pred",
        "total_TP",
        "total_FP",
        "total_FN",
        "micro_precision_at_tau",
        "micro_recall_at_tau",
        "micro_f1_at_tau",
        "matched_localization_mean_cm",
        "matched_localization_median_cm",
        "gt_to_nearest_mean_cm",
        "gt_to_nearest_median_cm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for threshold in results_by_tau.values():
            aggregate = threshold["aggregate"]
            row = {field: aggregate.get(field) for field in fields}
            row["tau_cm"] = threshold["tau_cm"]
            writer.writerow(row)


def tau_key_suffix(tau_m: float) -> str:
    cm = float(tau_m) * 100.0
    if abs(cm - round(cm)) < 1e-9:
        return f"{int(round(cm))}cm"
    text = f"{cm:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}cm"


if __name__ == "__main__":
    raise SystemExit(main())
