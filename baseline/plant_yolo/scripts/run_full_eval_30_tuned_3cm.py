#!/usr/bin/env python3
"""Run tuned full benchmark evaluation for harvesting and deleafing at 3 cm.

The script launches one benchmark shard per GPU, then recomputes combined
cut-point metrics across all shard outputs at the requested threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_cutpoints import parse_labels, tau_key_suffix  # noqa: E402


DEFAULT_LABELS = "/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt"
DEFAULT_FULL_ROOT = "/path/to/data/mps/egodelta_benchmark"
DEFAULT_PAIRS_ROOT = "/path/to/data/mps/egodelta_benchmark_data/pairs_fast"
DEFAULT_OUT = "/tmp/egodelta/plant_yolo/full_eval_30_tuned_yoloe26s_3cm"

TASK_TO_PAIR_DIR = {
    "harvesting": "tomato_pair",
    "deleafing": "leaf_pair",
}


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.eval_threshold_m <= 0.0:
        raise SystemExit("--eval-threshold-m must be positive")
    if args.visualization_stride < 1:
        raise SystemExit("--visualization-stride must be >= 1")
    if args.segmentation_batch_size < 1:
        raise SystemExit("--segmentation-batch-size must be >= 1")

    tasks = parse_tasks(args.tasks)
    gpus = parse_gpus(args.gpus)
    labels = parse_labels(args.labels)
    recording_ids = select_recording_ids(
        labels=labels,
        pairs_root=Path(args.pairs_root),
        tasks=tasks,
        expected_samples=args.expected_samples,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "recording_ids_30.txt").write_text("\n".join(recording_ids) + "\n", encoding="utf-8")
    write_run_config(args, tasks, gpus, recording_ids, out_root)

    print(f"Running {len(recording_ids)} recordings, tasks={','.join(tasks)}, threshold={args.eval_threshold_m} m")
    print(f"Output root: {out_root}")

    shard_roots = run_shards(args, tasks, gpus, recording_ids, out_root)
    metric_paths = {}
    for task in tasks:
        metric_paths[task] = run_metric_eval(args, task, recording_ids, shard_roots, out_root)

    summary = build_summary(args, tasks, recording_ids, shard_roots, metric_paths, out_root)
    summary_path = out_root / "summary_3cm.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== Full 30-Sample 3cm Summary =====")
    for task in tasks:
        aggregate = summary["tasks"][task]["aggregate"]
        print_metric_line(task, aggregate)
    print(f"summary: {summary_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--full-root", default=DEFAULT_FULL_ROOT)
    parser.add_argument("--pairs-root", default=DEFAULT_PAIRS_ROOT)
    parser.add_argument("--tasks", default="harvesting,deleafing")
    parser.add_argument("--expected-samples", type=int, default=30)
    parser.add_argument("--eval-threshold-m", type=float, default=0.03)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--yoloe-model", default="yoloe-26s-seg.pt")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--harvest-imgsz", type=int, default=720)
    parser.add_argument("--deleaf-imgsz", type=int, default=720)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--segmentation-batch-size", type=int, default=64)
    parser.add_argument("--visualization-stride", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--coordinate-check",
        action="store_true",
        help="Run pair/full-MPS coordinate checks. By default this uses the previously verified identity alignment.",
    )
    return parser


def parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(tasks).difference(TASK_TO_PAIR_DIR))
    if unsupported:
        raise SystemExit(f"Unsupported tasks: {', '.join(unsupported)}")
    if not tasks:
        raise SystemExit("--tasks must contain at least one task")
    return tasks


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    return gpus or ["cpu"]


def select_recording_ids(
    *,
    labels: dict[str, dict[str, list[list[float]]]],
    pairs_root: Path,
    tasks: list[str],
    expected_samples: int,
) -> list[str]:
    recording_ids = []
    missing_by_task: dict[str, list[str]] = {task: [] for task in tasks}
    for recording_id in labels:
        has_all = True
        for task in tasks:
            manifest = pairs_root / recording_id / TASK_TO_PAIR_DIR[task] / "manifest.json"
            if not manifest.exists():
                missing_by_task[task].append(recording_id)
                has_all = False
        if has_all:
            recording_ids.append(recording_id)

    if len(recording_ids) != expected_samples:
        message = [
            f"Expected {expected_samples} labeled recordings with all requested pair manifests, found {len(recording_ids)}.",
            "This usually means a GT/pairs mismatch is still present.",
        ]
        for task, missing in missing_by_task.items():
            if missing:
                message.append(f"Missing {TASK_TO_PAIR_DIR[task]} manifests for {task}:")
                message.extend(f"  {recording_id}" for recording_id in missing)
        raise SystemExit("\n".join(message))
    return recording_ids


def write_run_config(
    args: argparse.Namespace,
    tasks: list[str],
    gpus: list[str],
    recording_ids: list[str],
    out_root: Path,
) -> None:
    config = {
        "labels": args.labels,
        "full_root": args.full_root,
        "pairs_root": args.pairs_root,
        "tasks": tasks,
        "recording_ids": recording_ids,
        "gpus": gpus,
        "eval_threshold_m": args.eval_threshold_m,
        "yoloe_model": args.yoloe_model,
        "frame_stride": args.frame_stride,
        "harvest_imgsz": args.harvest_imgsz,
        "deleaf_imgsz": args.deleaf_imgsz,
        "max_det": args.max_det,
        "segmentation_batch_size": args.segmentation_batch_size,
        "visualization_stride": args.visualization_stride,
        "force": args.force,
        "coordinate_check": args.coordinate_check,
    }
    (out_root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_shards(
    args: argparse.Namespace,
    tasks: list[str],
    gpus: list[str],
    recording_ids: list[str],
    out_root: Path,
) -> list[Path]:
    shard_roots = [out_root / f"shard_{idx}" for idx in range(len(gpus))]

    def run_one(shard_index: int, gpu: str) -> dict[str, Any]:
        shard_root = shard_roots[shard_index]
        shard_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / f"shard_{shard_index}.log"
        cmd = [
            args.python,
            str(PROJECT_ROOT / "scripts" / "run_eval_benchmark.py"),
            "--labels",
            args.labels,
            "--full-root",
            args.full_root,
            "--pairs-root",
            args.pairs_root,
            "--out",
            str(shard_root),
            "--tasks",
            ",".join(tasks),
            "--yoloe-model",
            args.yoloe_model,
            "--frame-stride",
            str(args.frame_stride),
            "--harvest-imgsz",
            str(args.harvest_imgsz),
            "--deleaf-imgsz",
            str(args.deleaf_imgsz),
            "--max-det",
            str(args.max_det),
            "--segmentation-batch-size",
            str(args.segmentation_batch_size),
            "--visualization-stride",
            str(args.visualization_stride),
            "--eval-threshold-m",
            str(args.eval_threshold_m),
            "--num-shards",
            str(len(gpus)),
            "--shard-index",
            str(shard_index),
        ]
        if not args.coordinate_check:
            cmd.append("--skip-coordinate-check")
        if args.force:
            cmd.append("--force")
        for recording_id in recording_ids:
            cmd.extend(["--recording-id", recording_id])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        if gpu.lower() != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = gpu

        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("CMD " + " ".join(cmd) + "\n")
            log_file.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
            log_file.flush()
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {
            "shard_index": shard_index,
            "gpu": gpu,
            "root": str(shard_root),
            "log_path": str(log_path),
            "returncode": completed.returncode,
            "elapsed_s": time.time() - start,
        }

    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        future_to_shard = {
            executor.submit(run_one, shard_index, gpu): shard_index
            for shard_index, gpu in enumerate(gpus)
        }
        for future in as_completed(future_to_shard):
            result = future.result()
            results.append(result)
            print(
                f"shard {result['shard_index']} gpu={result['gpu']} "
                f"returncode={result['returncode']} log={result['log_path']}",
                flush=True,
            )

    results.sort(key=lambda item: item["shard_index"])
    (out_root / "shard_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [item for item in results if item["returncode"] != 0]
    if failed:
        details = "\n".join(f"  shard {item['shard_index']}: {item['log_path']}" for item in failed)
        raise SystemExit(f"One or more benchmark shards failed:\n{details}")
    return shard_roots


def run_metric_eval(
    args: argparse.Namespace,
    task: str,
    recording_ids: list[str],
    shard_roots: list[Path],
    out_root: Path,
) -> Path:
    metric_out = out_root / "metrics_3cm" / task
    metric_out.mkdir(parents=True, exist_ok=True)
    log_path = out_root / f"metrics_{task}_3cm.log"
    cmd = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "evaluate_cutpoints.py"),
        "--labels",
        args.labels,
        "--task",
        task,
        "--out",
        str(metric_out),
        "--tau-m",
        str(args.eval_threshold_m),
        "--missing-pred",
        "error",
    ]
    for root in shard_roots:
        cmd.extend(["--pred-root", str(root)])
    for recording_id in recording_ids:
        cmd.extend(["--recording-id", recording_id])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("CMD " + " ".join(cmd) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"Metric evaluation failed for {task}; see {log_path}")
    return metric_out / f"{task}_cutpoint_metrics.json"


def build_summary(
    args: argparse.Namespace,
    tasks: list[str],
    recording_ids: list[str],
    shard_roots: list[Path],
    metric_paths: dict[str, Path],
    out_root: Path,
) -> dict[str, Any]:
    tau_key = tau_key_suffix(args.eval_threshold_m)
    summary: dict[str, Any] = {
        "num_samples": len(recording_ids),
        "recording_ids": recording_ids,
        "eval_threshold_m": args.eval_threshold_m,
        "eval_threshold_key": tau_key,
        "shard_roots": [str(path) for path in shard_roots],
        "tasks": {},
    }
    for task in tasks:
        data = json.loads(metric_paths[task].read_text(encoding="utf-8"))
        threshold = data["thresholds"][tau_key]
        summary["tasks"][task] = {
            "metrics_json": str(metric_paths[task]),
            "aggregate": threshold["aggregate"],
        }
    return summary


def print_metric_line(task: str, aggregate: dict[str, Any]) -> None:
    print(
        f"{task}: scenes={aggregate['num_scenes']} "
        f"GT={aggregate['total_gt']} pred={aggregate['total_pred']} "
        f"TP={aggregate['total_TP']} FP={aggregate['total_FP']} FN={aggregate['total_FN']} "
        f"P={aggregate['micro_precision_at_tau']:.4f} "
        f"R={aggregate['micro_recall_at_tau']:.4f} "
        f"F1={aggregate['micro_f1_at_tau']:.4f} "
        f"gtNN={aggregate.get('gt_to_nearest_mean_cm')}cm"
    )


if __name__ == "__main__":
    raise SystemExit(main())
