#!/usr/bin/env python3
"""Evaluate deleafing 3D cut-point predictions with thresholded matching."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument(
        "--pred-root",
        required=True,
        help="Root containing per-scene prediction outputs.",
    )
    parser.add_argument(
        "--pred-pattern",
        default="{recording_id}/deleaf/final_predictions_deleaf.json",
        help=(
            "Path pattern under --pred-root. Available key: {recording_id}. "
            "Default matches outputs from scripts/run_eval_benchmark.py."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--tau-m", type=float, default=0.02)
    parser.add_argument("--recording-id", action="append")
    parser.add_argument(
        "--missing-pred",
        choices=["empty", "skip", "error"],
        default="empty",
        help="How to handle a scene without final_predictions_deleaf.json.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.labels)
    recording_ids = list(labels)
    if args.recording_id:
        wanted = set(args.recording_id)
        recording_ids = [recording_id for recording_id in recording_ids if recording_id in wanted]

    per_scene: list[dict[str, Any]] = []
    warnings: list[str] = []
    for recording_id in recording_ids:
        gt_points = np.asarray(labels[recording_id].get("deleafing", []), dtype=float).reshape(-1, 3)
        pred_path = Path(args.pred_root) / args.pred_pattern.format(recording_id=recording_id)
        if not pred_path.exists():
            message = f"Missing prediction file for {recording_id}: {pred_path}"
            if args.missing_pred == "error":
                raise FileNotFoundError(message)
            if args.missing_pred == "skip":
                warnings.append(f"Skipped scene. {message}")
                continue
            warnings.append(f"Using empty predictions. {message}")
            pred_points = np.zeros((0, 3), dtype=float)
        else:
            pred_points = load_deleaf_prediction_points(pred_path)

        metrics = compute_cutpoint_metrics(pred_points=pred_points, gt_points=gt_points, tau_m=args.tau_m)
        metrics.update(
            {
                "recording_id": recording_id,
                "prediction_path": str(pred_path),
                "gt_points_world": gt_points.tolist(),
                "pred_points_world": pred_points.tolist(),
            }
        )
        per_scene.append(metrics)

    aggregate = aggregate_cutpoint_metrics(per_scene, tau_m=args.tau_m)
    result = {
        "task": "deleafing",
        "metric": "thresholded_3d_cutpoint_detection",
        "label_frame": "full_mps_world",
        "labels_path": args.labels,
        "pred_root": args.pred_root,
        "pred_pattern": args.pred_pattern,
        "tau_m": float(args.tau_m),
        "tau_cm": float(args.tau_m * 100.0),
        "aggregate": aggregate,
        "per_scene": per_scene,
        "warnings": warnings,
    }
    json_path = out_dir / "deleafing_cutpoint_metrics.json"
    csv_path = out_dir / "deleafing_cutpoint_metrics.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(per_scene, csv_path, tau_m=float(args.tau_m))
    print(json.dumps({"json_path": str(json_path), "csv_path": str(csv_path), "aggregate": aggregate}, indent=2))
    return 0


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
                raise ValueError(f"Expected 3D point, got {line}")
            entries[current_id][current_task].append(values)
    return entries


def load_deleaf_prediction_points(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = []
    for item in data.get("removed_petioles", []):
        value = item.get("proxy_cut_point_3d_world") or item.get("cut_point_3d_world")
        if value is not None:
            points.append(value)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def write_csv(per_scene: list[dict[str, Any]], path: Path, tau_m: float) -> None:
    tau_suffix = tau_key_suffix(tau_m)
    fields = [
        "recording_id",
        "num_gt",
        "num_pred",
        "TP",
        "FP",
        "FN",
        f"precision_at_{tau_suffix}",
        f"recall_at_{tau_suffix}",
        f"f1_at_{tau_suffix}",
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
        for item in per_scene:
            writer.writerow({field: item.get(field) for field in fields})


def tau_key_suffix(tau_m: float) -> str:
    cm = float(tau_m) * 100.0
    if abs(cm - round(cm)) < 1e-9:
        return f"{int(round(cm))}cm"
    text = f"{cm:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}cm"


if __name__ == "__main__":
    raise SystemExit(main())
