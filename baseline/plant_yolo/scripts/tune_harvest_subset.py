#!/usr/bin/env python3
"""Tune tomato harvesting config on the fixed EgoDelta 8-scene subset.

This script keeps the existing harvesting method: YOLOE prompt segmentation,
Aria MPS lifting, instance fusion, pre/post disappearance scoring, and proxy
tomato-top cut points. Only prompts and numeric thresholds are varied.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_cutpoints import TASK_SPECS, load_prediction_points, parse_labels, tau_key_suffix  # noqa: E402
from yolo_aria_harvest_poc.evaluation import aggregate_cutpoint_metrics, compute_cutpoint_metrics  # noqa: E402
from yolo_aria_harvest_poc.task_config import load_task_config  # noqa: E402


DEFAULT_LABELS = "/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt"
DEFAULT_PAIRS_ROOT = "/path/to/data/mps/egodelta_benchmark_data/pairs_fast"
DEFAULT_OUT = "/tmp/egodelta/plant_yolo/harvest_tune8_yoloe26s_20trials"

DEFAULT_RECORDINGS = [
    "35aa89c0-e852-4750-9608-3f79788f7ee9",
    "3b0e7a93-8e13-443f-b729-80e0926dcc42",
    "44fe03cd-d0a7-408a-8ca2-b9b302c912a0",
    "47d0e394-ae62-49e1-bcd6-2f9f236972d0",
    "4d5ebec9-3ebd-4c65-b01d-8d2a0c668df2",
    "5479d13e-b1ed-4539-aeb4-c42ce20dbf8f",
    "5d9ae0cf-5e76-43c8-af38-24547c88e5e5",
    "5ede33ac-5cf2-4a8d-88cc-ef38f9b1e5ff",
]

BASE_TOMATO_PROMPTS = ["tomato fruit", "ripe tomato"]
COLOR_TOMATO_PROMPTS = ["tomato fruit", "ripe tomato", "red tomato", "green tomato", "unripe tomato"]
TOMATO_OBJECT_PROMPTS = ["tomato", "tomato fruit", "tomato cluster", "ripe tomato", "red tomato"]
RIPE_TOMATO_PROMPTS = ["ripe tomato", "red tomato", "tomato fruit"]
GREENHOUSE_TOMATO_PROMPTS = ["tomato fruit on plant", "greenhouse tomato", "ripe tomato", "red tomato"]
COMPACT_TOMATO_PROMPTS = ["tomato fruit"]


def trial(
    name: str,
    *,
    imgsz: int = 720,
    pipeline_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "imgsz": imgsz,
        "overrides": overrides,
        "pipeline_overrides": pipeline_overrides or {},
    }


TRIAL_SPECS = [
    trial("base_yoloe_720", text_prompts=BASE_TOMATO_PROMPTS),
    trial(
        "base_low_conf",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_2d_confidence=0.15,
        min_mask_area_px=200,
        disappeared_score_thres=0.5,
    ),
    trial(
        "base_high_conf",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_2d_confidence=0.35,
        min_mask_area_px=300,
        disappeared_score_thres=0.6,
    ),
    trial(
        "color_prompts",
        text_prompts=COLOR_TOMATO_PROMPTS,
        min_2d_confidence=0.20,
        min_mask_area_px=250,
    ),
    trial(
        "color_low_conf",
        text_prompts=COLOR_TOMATO_PROMPTS,
        min_2d_confidence=0.12,
        min_mask_area_px=160,
        disappeared_score_thres=0.45,
        min_visible_post_views=2,
    ),
    trial(
        "object_prompts",
        text_prompts=TOMATO_OBJECT_PROMPTS,
        min_2d_confidence=0.18,
        min_mask_area_px=220,
        max_mask_area_px=70000,
    ),
    trial(
        "ripe_prompts_strict",
        text_prompts=RIPE_TOMATO_PROMPTS,
        min_2d_confidence=0.25,
        min_mask_area_px=300,
        disappeared_score_thres=0.55,
    ),
    trial(
        "greenhouse_prompts",
        text_prompts=GREENHOUSE_TOMATO_PROMPTS,
        min_2d_confidence=0.18,
        min_mask_area_px=220,
    ),
    trial(
        "compact_tomato_lowbar",
        text_prompts=COMPACT_TOMATO_PROMPTS,
        min_2d_confidence=0.12,
        min_mask_area_px=160,
        disappeared_score_thres=0.45,
        min_visible_post_views=2,
    ),
    trial(
        "base_wider_depth",
        text_prompts=BASE_TOMATO_PROMPTS,
        depth_band_m=0.10,
        lifting_dbscan_eps_m=0.06,
        instance_cluster_eps_m=0.12,
    ),
    trial(
        "base_narrow_depth",
        text_prompts=BASE_TOMATO_PROMPTS,
        depth_band_m=0.05,
        lifting_dbscan_eps_m=0.035,
        instance_cluster_eps_m=0.08,
    ),
    trial(
        "base_more_visible_relaxed_score",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_visible_post_views=2,
        disappeared_score_thres=0.45,
        prepost_match_radius_m=0.10,
    ),
    trial(
        "base_strict_prepost",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_visible_post_views=3,
        disappeared_score_thres=0.65,
        prepost_match_radius_m=0.10,
    ),
    trial(
        "base_loose_prepost",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_visible_post_views=2,
        disappeared_score_thres=0.45,
        prepost_match_radius_m=0.15,
    ),
    trial(
        "base_minframes1",
        text_prompts=BASE_TOMATO_PROMPTS,
        instance_min_frames=1,
        instance_min_points=10,
        min_visible_post_views=2,
        disappeared_score_thres=0.45,
    ),
    trial(
        "color_minframes1_wide",
        text_prompts=COLOR_TOMATO_PROMPTS,
        min_2d_confidence=0.12,
        min_mask_area_px=160,
        depth_band_m=0.10,
        lifting_dbscan_eps_m=0.06,
        instance_cluster_eps_m=0.12,
        instance_min_frames=1,
        instance_min_points=8,
        min_visible_post_views=2,
        disappeared_score_thres=0.45,
    ),
    trial(
        "highres960_base",
        imgsz=960,
        text_prompts=BASE_TOMATO_PROMPTS,
        min_2d_confidence=0.20,
        min_mask_area_px=220,
    ),
    trial(
        "highres960_color_low_conf",
        imgsz=960,
        text_prompts=COLOR_TOMATO_PROMPTS,
        min_2d_confidence=0.12,
        min_mask_area_px=160,
        min_visible_post_views=2,
        disappeared_score_thres=0.45,
    ),
    trial(
        "cutpoint_lower_offset",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_2d_confidence=0.20,
        min_mask_area_px=220,
        pipeline_overrides={"top_quantile": 0.70, "cut_offset_m": 0.015},
    ),
    trial(
        "cutpoint_higher_offset",
        text_prompts=BASE_TOMATO_PROMPTS,
        min_2d_confidence=0.20,
        min_mask_area_px=220,
        pipeline_overrides={"top_quantile": 0.85, "cut_offset_m": 0.025},
    ),
]


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.trials > len(TRIAL_SPECS):
        raise SystemExit(f"--trials={args.trials} requested, but only {len(TRIAL_SPECS)} trials are defined")
    if args.visualization_stride < 1:
        raise SystemExit("--visualization-stride must be >= 1")
    if args.segmentation_batch_size < 1:
        raise SystemExit("--segmentation-batch-size must be >= 1")

    recordings = load_recording_ids(args)
    labels = parse_labels(args.labels)
    validate_inputs(args, recordings, labels)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    base_config = load_task_config(resolve_project_path(args.base_config)).to_dict()

    results: list[dict[str, Any]] = []
    for trial_index, spec in enumerate(TRIAL_SPECS[: args.trials], start=1):
        trial_root = out_root / f"trial_{trial_index:02d}_{spec['name']}"
        summary_path = trial_root / "trial_summary.json"
        if summary_path.exists() and not args.force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print_trial_line(summary, reused=True)
        else:
            summary = run_trial(
                args=args,
                trial_index=trial_index,
                spec=spec,
                trial_root=trial_root,
                base_config=base_config,
                recordings=recordings,
                labels=labels,
                gpus=gpus,
            )
            if summary["status"] != "ok" and args.stop_on_error:
                raise SystemExit(f"Trial {trial_index:02d} failed; see {trial_root}")
        results.append(summary)
        write_search_outputs(out_root, args, recordings, results)

    best = best_result(results, args.rank_by)
    if best is None:
        raise SystemExit("No successful trials completed")

    print("\n===== Best Harvest Tuning =====", flush=True)
    print(f"rank_by: {args.rank_by}", flush=True)
    print(f"trial: {best['trial_index']:02d} {best['trial_name']}", flush=True)
    print_metric_line(best["aggregate_3p5cm"], prefix="3.5cm")
    print(f"config: {best['config_path']}", flush=True)
    print(f"trial_root: {best['root']}", flush=True)
    print(f"best_json: {out_root / 'best_trial.json'}", flush=True)
    print(f"leaderboard: {out_root / 'leaderboard.csv'}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--pairs-root", default=DEFAULT_PAIRS_ROOT)
    parser.add_argument("--base-config", default="configs/harvest.yaml")
    parser.add_argument("--yoloe-model", default="yoloe-26s-seg.pt")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for pipeline subprocesses")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated CUDA IDs. Use cpu to avoid CUDA pinning.")
    parser.add_argument("--recording-id", action="append", help="Recording UUID or .vrs name. Defaults to the fixed 8.")
    parser.add_argument("--recording-list", help="Optional newline-delimited UUID/.vrs list.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--segmentation-batch-size", type=int, default=64)
    parser.add_argument("--visualization-stride", type=int, default=30)
    parser.add_argument("--tau-m", action="append", type=float, default=None)
    parser.add_argument("--rank-by", choices=["f1", "tp", "recall", "precision", "gt-nearest"], default="f1")
    parser.add_argument("--force", action="store_true", help="Rerun existing trial/sample outputs")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def load_recording_ids(args: argparse.Namespace) -> list[str]:
    raw: list[str] = []
    if args.recording_list:
        raw.extend(
            line.strip()
            for line in Path(args.recording_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if args.recording_id:
        raw.extend(args.recording_id)
    if not raw:
        raw = list(DEFAULT_RECORDINGS)
    recordings = [normalize_recording_id(item) for item in raw]
    seen: set[str] = set()
    unique = []
    for recording_id in recordings:
        if recording_id not in seen:
            seen.add(recording_id)
            unique.append(recording_id)
    return unique


def normalize_recording_id(value: str) -> str:
    name = Path(value).name.strip()
    if name.endswith(".vrs"):
        name = name[:-4]
    return name


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def parse_gpus(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or ["cpu"]


def validate_inputs(args: argparse.Namespace, recordings: list[str], labels: dict[str, Any]) -> None:
    missing_labels = [recording_id for recording_id in recordings if recording_id not in labels]
    if missing_labels:
        joined = "\n  ".join(missing_labels)
        raise SystemExit(f"Missing GT labels for requested recordings:\n  {joined}")

    pairs_root = Path(args.pairs_root)
    missing_pairs = [
        recording_id
        for recording_id in recordings
        if not (pairs_root / recording_id / "tomato_pair" / "manifest.json").exists()
    ]
    if missing_pairs:
        joined = "\n  ".join(missing_pairs)
        raise SystemExit(f"Missing tomato_pair manifests for requested recordings:\n  {joined}")


def run_trial(
    *,
    args: argparse.Namespace,
    trial_index: int,
    spec: dict[str, Any],
    trial_root: Path,
    base_config: dict[str, Any],
    recordings: list[str],
    labels: dict[str, Any],
    gpus: list[str],
) -> dict[str, Any]:
    start = time.time()
    trial_root.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(base_config)
    config.update(spec["overrides"])
    config["segmentation_backend"] = "yoloe_prompt"
    config_path = trial_root / "harvest_config.yaml"
    write_yaml(config, config_path)

    sample_runs = run_samples_for_trial(args, spec, trial_root, config_path, recordings, gpus)
    failed = [item for item in sample_runs if item["returncode"] != 0]
    status = "failed" if failed else "ok"
    thresholds = evaluate_trial(args, trial_root, recordings, labels) if status == "ok" else {}
    aggregate_3p5 = thresholds.get("3p5cm", {}).get("aggregate", {})
    per_scene_3p5 = thresholds.get("3p5cm", {}).get("per_scene", [])
    summary = {
        "trial_index": trial_index,
        "trial_name": spec["name"],
        "status": status,
        "root": str(trial_root),
        "config_path": str(config_path),
        "imgsz": int(spec["imgsz"]),
        "overrides": spec["overrides"],
        "pipeline_overrides": spec["pipeline_overrides"],
        "aggregate_3p5cm": aggregate_3p5,
        "per_scene_3p5cm": per_scene_3p5,
        "thresholds": thresholds,
        "sample_runs": sample_runs,
        "elapsed_s": time.time() - start,
    }
    (trial_root / "trial_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )
    print_trial_line(summary)
    return summary


def run_samples_for_trial(
    args: argparse.Namespace,
    spec: dict[str, Any],
    trial_root: Path,
    config_path: Path,
    recordings: list[str],
    gpus: list[str],
) -> list[dict[str, Any]]:
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    max_workers = min(len(gpus), len(recordings))
    results: list[dict[str, Any]] = []

    def run_with_gpu(recording_id: str) -> dict[str, Any]:
        gpu = gpu_queue.get()
        try:
            return run_one_sample(args, spec, trial_root, config_path, recording_id, gpu)
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_recording = {
            executor.submit(run_with_gpu, recording_id): recording_id
            for recording_id in recordings
        }
        for future in as_completed(future_to_recording):
            results.append(future.result())

    return sorted(results, key=lambda item: recordings.index(item["recording_id"]))


def run_one_sample(
    args: argparse.Namespace,
    spec: dict[str, Any],
    trial_root: Path,
    config_path: Path,
    recording_id: str,
    gpu: str,
) -> dict[str, Any]:
    sample_out = trial_root / recording_id / "harvest"
    final_path = sample_out / "final_predictions.json"
    log_path = trial_root / f"{recording_id}.log"
    if final_path.exists() and not args.force:
        return {
            "recording_id": recording_id,
            "gpu": gpu,
            "status": "reused",
            "returncode": 0,
            "elapsed_s": 0.0,
            "log_path": str(log_path),
            "prediction_path": str(final_path),
        }

    pair_manifest = Path(args.pairs_root) / recording_id / "tomato_pair" / "manifest.json"
    cmd = [
        args.python,
        "-m",
        "yolo_aria_harvest_poc.run_pipeline",
        "--task",
        "harvest",
        "--task-config",
        str(config_path),
        "--mode",
        "extracted_frames",
        "--pair-manifest",
        str(pair_manifest),
        "--segmentation-backend",
        "yoloe_prompt",
        "--yoloe-model",
        args.yoloe_model,
        "--image-view",
        "pinhole",
        "--frame-stride",
        str(args.frame_stride),
        "--imgsz",
        str(spec["imgsz"]),
        "--max-det",
        str(args.max_det),
        "--segmentation-batch-size",
        str(args.segmentation_batch_size),
        "--visualization-stride",
        str(args.visualization_stride),
        "--world-up",
        "0,0,1",
        "--out",
        str(sample_out),
        "--debug",
    ]
    add_pipeline_override_args(cmd, spec.get("pipeline_overrides", {}))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if gpu.lower() != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu

    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        "recording_id": recording_id,
        "gpu": gpu,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "elapsed_s": time.time() - start,
        "log_path": str(log_path),
        "prediction_path": str(final_path),
    }


def add_pipeline_override_args(cmd: list[str], overrides: dict[str, Any]) -> None:
    mapping = {
        "point_disappearance_weight": "--point-disappearance-weight",
        "top_quantile": "--top-quantile",
        "cut_offset_m": "--cut-offset-m",
    }
    for key, flag in mapping.items():
        if key in overrides:
            cmd.extend([flag, str(overrides[key])])


def evaluate_trial(
    args: argparse.Namespace,
    trial_root: Path,
    recordings: list[str],
    labels: dict[str, Any],
) -> dict[str, Any]:
    spec = TASK_SPECS["harvesting"]
    taus = args.tau_m or [0.02, 0.035, 0.05]
    thresholds: dict[str, Any] = {}
    for tau_m in taus:
        per_scene = []
        for recording_id in recordings:
            gt_points = np.asarray(labels[recording_id].get(spec["label_key"], []), dtype=float).reshape(-1, 3)
            pred_path = trial_root / recording_id / spec["task_dir"] / spec["prediction_file"]
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction file: {pred_path}")
            pred_points = load_prediction_points(pred_path, spec)
            metrics = compute_cutpoint_metrics(pred_points=pred_points, gt_points=gt_points, tau_m=float(tau_m))
            metrics.update(
                {
                    "recording_id": recording_id,
                    "prediction_path": str(pred_path),
                    "gt_points_world": gt_points.tolist(),
                    "pred_points_world": pred_points.tolist(),
                }
            )
            per_scene.append(metrics)
        key = tau_key_suffix(float(tau_m))
        thresholds[key] = {
            "tau_m": float(tau_m),
            "tau_cm": float(tau_m * 100.0),
            "aggregate": aggregate_cutpoint_metrics(per_scene, tau_m=float(tau_m)),
            "per_scene": per_scene,
        }
    return thresholds


def best_result(results: list[dict[str, Any]], rank_by: str) -> dict[str, Any] | None:
    ok = [item for item in results if item.get("status") == "ok" and item.get("aggregate_3p5cm")]
    if not ok:
        return None
    return sorted(ok, key=lambda item: rank_key(item, rank_by), reverse=True)[0]


def rank_key(item: dict[str, Any], rank_by: str) -> tuple[float, ...]:
    aggregate = item["aggregate_3p5cm"]
    tp = float(aggregate.get("total_TP", 0))
    pred = float(aggregate.get("total_pred", 0))
    precision = float(aggregate.get("micro_precision_at_tau", 0.0))
    recall = float(aggregate.get("micro_recall_at_tau", 0.0))
    f1 = float(aggregate.get("micro_f1_at_tau", 0.0))
    gt_nearest = aggregate.get("gt_to_nearest_mean_cm")
    gt_nearest_score = -float(gt_nearest) if gt_nearest is not None else float("-inf")
    if rank_by == "tp":
        return (tp, f1, precision, gt_nearest_score, -pred)
    if rank_by == "recall":
        return (recall, f1, precision, gt_nearest_score, -pred)
    if rank_by == "precision":
        return (precision, tp, f1, gt_nearest_score, -pred)
    if rank_by == "gt-nearest":
        return (gt_nearest_score, f1, tp, precision, -pred)
    return (f1, tp, precision, recall, gt_nearest_score, -pred)


def write_search_outputs(
    out_root: Path,
    args: argparse.Namespace,
    recordings: list[str],
    results: list[dict[str, Any]],
) -> None:
    sorted_results = sorted(
        [item for item in results if item.get("status") == "ok"],
        key=lambda item: rank_key(item, args.rank_by),
        reverse=True,
    )
    best = sorted_results[0] if sorted_results else None
    summary = {
        "rank_by": args.rank_by,
        "recording_ids": recordings,
        "trials_requested": args.trials,
        "trials_completed": len(results),
        "best_trial": best,
        "results": results,
    }
    (out_root / "tuning_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )
    if best is not None:
        (out_root / "best_trial.json").write_text(
            json.dumps(json_ready(best), indent=2),
            encoding="utf-8",
        )
    write_leaderboard_csv(out_root / "leaderboard.csv", results, args.rank_by)


def write_leaderboard_csv(path: Path, results: list[dict[str, Any]], rank_by: str) -> None:
    fields = [
        "rank",
        "trial_index",
        "trial_name",
        "status",
        "imgsz",
        "total_gt",
        "total_pred",
        "total_TP",
        "total_FP",
        "total_FN",
        "precision_3p5cm",
        "recall_3p5cm",
        "f1_3p5cm",
        "matched_mean_cm",
        "gt_to_nearest_mean_cm",
        "elapsed_s",
        "config_path",
        "root",
        "overrides_json",
        "pipeline_overrides_json",
    ]
    ordered = sorted(
        results,
        key=lambda item: (item.get("status") == "ok", rank_key(item, rank_by) if item.get("aggregate_3p5cm") else ()),
        reverse=True,
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(ordered, start=1):
            aggregate = item.get("aggregate_3p5cm") or {}
            writer.writerow(
                {
                    "rank": rank if item.get("status") == "ok" else "",
                    "trial_index": item.get("trial_index"),
                    "trial_name": item.get("trial_name"),
                    "status": item.get("status"),
                    "imgsz": item.get("imgsz"),
                    "total_gt": aggregate.get("total_gt"),
                    "total_pred": aggregate.get("total_pred"),
                    "total_TP": aggregate.get("total_TP"),
                    "total_FP": aggregate.get("total_FP"),
                    "total_FN": aggregate.get("total_FN"),
                    "precision_3p5cm": aggregate.get("micro_precision_at_tau"),
                    "recall_3p5cm": aggregate.get("micro_recall_at_tau"),
                    "f1_3p5cm": aggregate.get("micro_f1_at_tau"),
                    "matched_mean_cm": aggregate.get("matched_localization_mean_cm"),
                    "gt_to_nearest_mean_cm": aggregate.get("gt_to_nearest_mean_cm"),
                    "elapsed_s": item.get("elapsed_s"),
                    "config_path": item.get("config_path"),
                    "root": item.get("root"),
                    "overrides_json": json.dumps(item.get("overrides", {}), sort_keys=True),
                    "pipeline_overrides_json": json.dumps(item.get("pipeline_overrides", {}), sort_keys=True),
                }
            )


def print_trial_line(summary: dict[str, Any], *, reused: bool = False) -> None:
    state = "REUSED" if reused else "TRIAL"
    if summary.get("status") != "ok":
        print(
            f"{state} {summary['trial_index']:02d} {summary['trial_name']}: "
            f"status={summary.get('status')} root={summary.get('root')}",
            flush=True,
        )
        return
    aggregate = summary["aggregate_3p5cm"]
    print(
        f"{state} {summary['trial_index']:02d} {summary['trial_name']}: "
        f"TP={aggregate['total_TP']} FP={aggregate['total_FP']} FN={aggregate['total_FN']} "
        f"P={aggregate['micro_precision_at_tau']:.4f} "
        f"R={aggregate['micro_recall_at_tau']:.4f} "
        f"F1={aggregate['micro_f1_at_tau']:.4f} "
        f"gtNN={aggregate.get('gt_to_nearest_mean_cm')}cm "
        f"root={summary['root']}",
        flush=True,
    )


def print_metric_line(aggregate: dict[str, Any], *, prefix: str) -> None:
    print(
        f"{prefix}: TP={aggregate['total_TP']} FP={aggregate['total_FP']} FN={aggregate['total_FN']} "
        f"P={aggregate['micro_precision_at_tau']:.4f} "
        f"R={aggregate['micro_recall_at_tau']:.4f} "
        f"F1={aggregate['micro_f1_at_tau']:.4f} "
        f"gtNN={aggregate.get('gt_to_nearest_mean_cm')}cm",
        flush=True,
    )


def write_yaml(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    if yaml is not None:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return

    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            if key == "world_up":
                lines.append(f"{key}: [{', '.join(format_scalar(item) for item in value)}]")
            else:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {format_scalar(item)}")
        else:
            lines.append(f"{key}: {format_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
