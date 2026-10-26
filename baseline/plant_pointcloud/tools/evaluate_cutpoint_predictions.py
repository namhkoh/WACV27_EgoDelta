from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tomato_cutpoints.eval.metrics import aggregate_cut_point_metrics, evaluate_cut_points, threshold_label_cm
from tools.evaluate_cutpoints import TASK_SPECS, jsonable, parse_labels, parse_task_names, read_predictions, write_csv


PER_SCENE_BASE_FIELDS = [
    "uuid",
    "task",
    "num_gt",
    "num_pred",
    "TP",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
    "matched_mean_distance",
    "matched_mean_distance_cm",
    "gt_to_nearest_mean_distance",
    "gt_to_nearest_mean_distance_cm",
    "run_dir",
]

AGGREGATE_BASE_FIELDS = [
    "task",
    "threshold_m",
    "num_scenes",
    "num_gt",
    "num_pred",
    "TP",
    "FP",
    "FN",
    "micro Precision",
    "micro Recall",
    "micro F1",
    "matched_mean_distance",
    "matched_median_distance",
    "matched_mean_distance_cm",
    "matched_median_distance_cm",
    "gt_to_nearest_mean_distance",
    "gt_to_nearest_median_distance",
    "gt_to_nearest_mean_distance_cm",
    "gt_to_nearest_median_distance_cm",
]


def per_scene_fields(threshold_label: str) -> list[str]:
    fields = PER_SCENE_BASE_FIELDS.copy()
    insert_at = fields.index("matched_mean_distance")
    fields[insert_at:insert_at] = [
        f"Precision@{threshold_label}",
        f"Recall@{threshold_label}",
        f"F1@{threshold_label}",
    ]
    return fields


def aggregate_fields(threshold_label: str) -> list[str]:
    fields = AGGREGATE_BASE_FIELDS.copy()
    insert_at = fields.index("matched_mean_distance")
    fields[insert_at:insert_at] = [
        f"micro Precision@{threshold_label}",
        f"micro Recall@{threshold_label}",
        f"micro F1@{threshold_label}",
    ]
    return fields


def read_gt_points(run_dir: Path, labels: dict[str, Any], uuid: str, label_key: str) -> np.ndarray:
    transformed_path = run_dir / "gt_points_prediction_frame.json"
    if transformed_path.exists():
        doc = json.loads(transformed_path.read_text(encoding="utf-8"))
        points = doc.get("gt_points_prediction_frame", [])
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.asarray(labels[uuid][label_key], dtype=np.float64).reshape(-1, 3)


def resolve_run_dir(eval_root: Path, uuid: str, task: str) -> Path:
    runs_layout = eval_root / "runs" / uuid / task
    if runs_layout.exists():
        return runs_layout
    return eval_root / uuid / task


def select_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing cut-point predictions with thresholded matching.")
    parser.add_argument("--labels", type=Path, default=Path("3d points labels.txt"))
    parser.add_argument("--eval-root", type=Path, required=True, help="Evaluation root containing runs/<uuid>/<task>/targets.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to --eval-root.")
    parser.add_argument("--threshold-m", type=float, default=0.02)
    parser.add_argument(
        "--tasks",
        type=parse_task_names,
        default=list(TASK_SPECS),
        help="Comma-separated tasks to evaluate: deleaf, harvest, or deleaf,harvest.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-complete", action="store_true", help="Skip task runs without targets.json.")
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    uuids = list(labels.keys())
    if args.limit is not None:
        uuids = uuids[: args.limit]

    out_dir = args.out_dir or args.eval_root
    threshold_label = threshold_label_cm(args.threshold_m)
    per_scene_fieldnames = per_scene_fields(threshold_label)
    aggregate_fieldnames = aggregate_fields(threshold_label)
    scene_docs: list[dict[str, Any]] = []
    per_scene_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    gt_nearest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for uuid in uuids:
        for task in args.tasks:
            spec = TASK_SPECS[task]
            run_dir = resolve_run_dir(args.eval_root, uuid, task)
            targets_path = run_dir / "targets.json"
            if args.only_complete and not targets_path.exists():
                skipped.append({"uuid": uuid, "task": task, "reason": "missing targets.json"})
                continue

            gt = read_gt_points(run_dir, labels, uuid, spec["label_key"])
            pred = read_predictions(targets_path)
            metrics = evaluate_cut_points(pred, gt, threshold_m=args.threshold_m)
            scene_doc = {"uuid": uuid, "task": task, **metrics}
            scene_docs.append(scene_doc)

            per_scene_rows.append(
                select_fields(
                    {
                        "uuid": uuid,
                        "task": task,
                        "run_dir": str(run_dir),
                        **metrics,
                    },
                    per_scene_fieldnames,
                )
            )
            for row in metrics["matches"]:
                match_rows.append({"uuid": uuid, "task": task, **row})
            for row in metrics["gt_to_nearest"]:
                gt_nearest_rows.append({"uuid": uuid, "task": task, **row})

    aggregate_rows = [{"task": "all", **aggregate_cut_point_metrics(scene_docs, threshold_m=args.threshold_m)}]
    for task in args.tasks:
        task_docs = [row for row in scene_docs if row["task"] == task]
        aggregate_rows.append({"task": task, **aggregate_cut_point_metrics(task_docs, threshold_m=args.threshold_m)})

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "cutpoint_metrics_per_scene.csv", per_scene_rows)
    write_csv(
        out_dir / "cutpoint_metrics_aggregate.csv",
        [select_fields(row, aggregate_fieldnames) for row in aggregate_rows],
    )
    write_csv(out_dir / "cutpoint_tp_matches.csv", match_rows)
    write_csv(out_dir / "cutpoint_gt_to_nearest.csv", gt_nearest_rows)

    summary = {
        "labels": args.labels,
        "eval_root": args.eval_root,
        "threshold_m": args.threshold_m,
        "tasks": args.tasks,
        "n_label_entries": len(labels),
        "n_evaluated_scenes": len(scene_docs),
        "aggregates": aggregate_rows,
        "skipped": skipped,
    }
    (out_dir / "cutpoint_metrics_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    print("Aggregate summary:")
    for row in aggregate_rows:
        print(select_fields(row, aggregate_fieldnames))
    if skipped:
        print(f"Skipped {len(skipped)} incomplete task runs")


if __name__ == "__main__":
    main()
