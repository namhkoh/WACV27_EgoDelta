#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PY = Path("/tmp/egodelta_1cm/manual_front_yolo_032/vendor_py")
if VENDOR_PY.exists():
    sys.path.insert(0, str(VENDOR_PY))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_manual_front_yolo_032 import (  # noqa: E402
    PIXEL_THRESHOLDS,
    METER_THRESHOLDS,
    backproject_np,
    blockers_from_audit,
    metric_definition,
    resolve_samples,
    result_to_detections,
    summarize_values,
)


TASK_ID = "yolo26l_box32_2dgt_selectors_036"
TASKS = ("deleafing", "harvesting")
DEFAULT_LABELS = Path("/tmp/egodelta/front/labels.jsonl")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/yolo26l_box32_2dgt_improve_036/selector_diagnostics")


DEPLOYABLE_SELECTORS = (
    "confidence",
    "center_prior",
    "train_mean_prior",
    "train_median_prior",
    "recording_median_prior",
    "nearest_time_train_prior",
    "nearest_highconf_train_median",
    "confidence_weighted_train_median",
    "confidence_weighted_time_prior",
)
ORACLE_SELECTORS = ("oracle_nearest_top5", "oracle_nearest_top20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26L box32 candidate selectors on the manual 2D-GT test set."
    )
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--tasks", default="deleafing,harvesting")
    parser.add_argument("--weights", type=Path, default=None, help="Shared weights for all requested tasks.")
    parser.add_argument("--deleafing-weights", type=Path, default=None)
    parser.add_argument("--harvesting-weights", type=Path, default=None)
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--imgsz-list", default="640")
    parser.add_argument("--conf-list", default="0.001,0.0003,0.0001")
    parser.add_argument("--max-det-list", default="20,100")
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--miss-penalty-m", type=float, default=1.0)
    parser.add_argument("--high-conf-ratio", type=float, default=0.25)
    parser.add_argument("--high-conf-floor", type=float, default=0.01)
    parser.add_argument("--conf-prior-weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    set_thread_limits()
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    args.out_root.mkdir(parents=True, exist_ok=True)
    accepted_labels = write_accepted_only_labels(args.labels_jsonl, args.out_root / "labels_accepted_only.jsonl")
    samples, audit = resolve_samples(accepted_labels, args.front_root, "nearest-source-target")
    weights_by_task = weights_map(args)
    missing_weights = [task for task in tasks if weights_by_task.get(task) is None]
    if missing_weights:
        raise SystemExit(f"Missing YOLO weights for tasks: {', '.join(missing_weights)}")
    unresolved = [
        sample
        for sample in samples
        if sample.get("task") in tasks and sample.get("split") == args.split and sample.get("metric_ready") is not True
    ]
    if unresolved:
        raise SystemExit(f"{len(unresolved)} selected samples lack metric-ready camera/depth")

    priors = build_priors(samples, tasks)
    write_json(args.out_root / "run_config.json", build_run_config(args, tasks, accepted_labels, weights_by_task))
    write_json(args.out_root / "data_audit.json", audit)
    write_json(args.out_root / "priors.json", priors)

    all_rows: list[dict[str, Any]] = []
    for task in tasks:
        weights = weights_by_task[task]
        if weights is None:
            continue
        task_samples = [
            sample
            for sample in samples
            if sample.get("task") == task and sample.get("split") == args.split and sample.get("metric_ready") is True
        ]
        task_rows = evaluate_task_grid(task, task_samples, weights, priors[task], args)
        all_rows.extend(task_rows)

    write_jsonl(args.out_root / "per_sample_metrics.jsonl", all_rows)
    summary = summarize_all_rows(all_rows, audit, args)
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve())}, indent=2))
    return 0


def set_thread_limits() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def parse_tasks(text: str) -> list[str]:
    tasks = [part.strip() for part in str(text).split(",") if part.strip()]
    bad = [task for task in tasks if task not in TASKS]
    if bad:
        raise ValueError(f"Unknown task(s): {bad}; allowed={TASKS}")
    return tasks


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def weights_map(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "deleafing": args.deleafing_weights or args.weights,
        "harvesting": args.harvesting_weights or args.weights,
    }


def write_accepted_only_labels(labels_path: Path, out_path: Path) -> Path:
    rows = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("status")) == "accepted":
                rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out_path


def build_priors(samples: list[dict[str, Any]], tasks: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in tasks:
        train_xy = np.asarray(
            [
                sample["target_2d"]
                for sample in samples
                if sample.get("task") == task
                and sample.get("split") == "train"
                and sample.get("metric_ready") is True
                and sample.get("target_2d") is not None
            ],
            dtype=np.float64,
        )
        if train_xy.size == 0:
            raise RuntimeError(f"No train samples available for {task} prior")
        mean_xy = np.mean(train_xy, axis=0)
        median_xy = np.median(train_xy, axis=0)
        spread = np.linalg.norm(train_xy - median_xy[None, :], axis=1)
        scale = float(np.percentile(spread, 75)) if spread.size else 128.0
        out[task] = {
            "train_count": int(train_xy.shape[0]),
            "train_mean_xy": [float(mean_xy[0]), float(mean_xy[1])],
            "train_median_xy": [float(median_xy[0]), float(median_xy[1])],
            "train_prior_scale_px": max(32.0, scale),
            "recording_phase_medians": recording_phase_medians(samples, task),
            "train_temporal_points": train_temporal_points(samples, task),
        }
    return out


def recording_phase_medians(samples: list[dict[str, Any]], task: str) -> dict[str, list[float]]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for sample in samples:
        if sample.get("task") != task or sample.get("split") != "train" or sample.get("target_2d") is None:
            continue
        grouped[recording_phase_key(sample)].append([float(v) for v in sample["target_2d"]])
    out: dict[str, list[float]] = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        med = np.median(arr, axis=0)
        out[key] = [float(med[0]), float(med[1])]
    return out


def train_temporal_points(samples: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    points = []
    for sample in samples:
        if sample.get("task") != task or sample.get("split") != "train" or sample.get("target_2d") is None:
            continue
        points.append(
            {
                "recording_phase_key": recording_phase_key(sample),
                "relative_time_s": sample.get("relative_time_s"),
                "timestamp_ns": sample.get("timestamp_ns"),
                "target_2d": [float(v) for v in sample["target_2d"]],
            }
        )
    return points


def build_run_config(
    args: argparse.Namespace,
    tasks: list[str],
    accepted_labels: Path,
    weights_by_task: dict[str, Path | None],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "accepted_only_labels_jsonl": str(accepted_labels.resolve()),
        "front_root": str(args.front_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "tasks": tasks,
        "split": args.split,
        "checkpoint_tag": args.checkpoint_tag,
        "weights": {
            task: str(path.resolve()) if path is not None else None for task, path in weights_by_task.items()
        },
        "prediction_grid": {
            "imgsz_list": parse_int_list(args.imgsz_list),
            "conf_list": parse_float_list(args.conf_list),
            "max_det_list": parse_int_list(args.max_det_list),
            "iou": float(args.iou),
            "batch": int(args.batch),
            "device": str(args.device),
        },
        "selectors": {
            "deployable": list(DEPLOYABLE_SELECTORS),
            "diagnostic_oracle": list(ORACLE_SELECTORS),
            "nearest_highconf_train_median": {
                "high_conf_ratio_of_top": float(args.high_conf_ratio),
                "high_conf_floor": float(args.high_conf_floor),
            },
            "confidence_weighted_train_median": {"conf_prior_weight": float(args.conf_prior_weight)},
        },
        "thread_limits": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "metric_definition": metric_definition("nearest-source-target"),
    }


def evaluate_task_grid(
    task: str,
    samples: list[dict[str, Any]],
    weights: Path,
    priors: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not weights.exists():
        raise FileNotFoundError(weights)
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for YOLO prediction.") from exc

    model = YOLO(str(weights))
    image_paths = [sample["image_path"] for sample in samples]
    rows: list[dict[str, Any]] = []
    for imgsz in parse_int_list(args.imgsz_list):
        for conf in parse_float_list(args.conf_list):
            for max_det in parse_int_list(args.max_det_list):
                results_iter = model.predict(
                    source=image_paths,
                    imgsz=int(imgsz),
                    conf=float(conf),
                    iou=float(args.iou),
                    max_det=int(max_det),
                    batch=int(args.batch),
                    device=str(args.device),
                    stream=True,
                    save=False,
                    verbose=False,
                )
                for sample, result in zip(samples, results_iter):
                    detections = result_to_detections(result)
                    rows.extend(score_all_selectors(task, sample, detections, priors, imgsz, conf, max_det, args))
    return rows


def score_all_selectors(
    task: str,
    sample: dict[str, Any],
    detections: list[dict[str, Any]],
    priors: dict[str, Any],
    imgsz: int,
    conf: float,
    max_det: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected = {
        "confidence": choose_confidence(detections),
        "center_prior": choose_nearest_point(detections, image_center(sample)),
        "train_mean_prior": choose_nearest_point(detections, np.asarray(priors["train_mean_xy"], dtype=np.float64)),
        "train_median_prior": choose_nearest_point(detections, np.asarray(priors["train_median_xy"], dtype=np.float64)),
        "recording_median_prior": choose_nearest_point(detections, recording_median_prior(sample, priors)),
        "nearest_time_train_prior": choose_nearest_point(detections, nearest_time_prior(sample, priors)),
        "nearest_highconf_train_median": choose_nearest_highconf_train_median(detections, priors, args),
        "confidence_weighted_train_median": choose_confidence_weighted_train_median(detections, priors, args),
        "confidence_weighted_time_prior": choose_confidence_weighted_point(
            detections, nearest_time_prior(sample, priors), max(1.0, float(priors["train_prior_scale_px"])), args
        ),
        "oracle_nearest_top5": choose_oracle_nearest(detections, sample, 5),
        "oracle_nearest_top20": choose_oracle_nearest(detections, sample, 20),
    }
    rows = []
    for selector, detection in selected.items():
        rows.append(
            score_selection(
                task=task,
                sample=sample,
                detections=detections,
                detection=detection,
                selector=selector,
                deployable=selector in DEPLOYABLE_SELECTORS,
                imgsz=imgsz,
                conf=conf,
                max_det=max_det,
                miss_penalty_m=float(args.miss_penalty_m),
            )
        )
    return rows


def choose_confidence(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    return detections[0] if detections else None


def choose_nearest_point(detections: list[dict[str, Any]], point_xy: np.ndarray) -> dict[str, Any] | None:
    if not detections:
        return None
    return min(
        detections,
        key=lambda det: float(np.linalg.norm(np.asarray(det["center_xy"], dtype=np.float64) - point_xy)),
    )


def choose_nearest_highconf_train_median(
    detections: list[dict[str, Any]], priors: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any] | None:
    if not detections:
        return None
    top_conf = max(float(det["conf"]) for det in detections)
    dynamic_floor = max(float(args.high_conf_floor), float(args.high_conf_ratio) * top_conf)
    candidates = [det for det in detections if float(det["conf"]) >= dynamic_floor]
    if not candidates:
        candidates = [det for det in detections if float(det["conf"]) >= float(args.high_conf_ratio) * top_conf]
    prior_xy = np.asarray(priors["train_median_xy"], dtype=np.float64)
    return choose_nearest_point(candidates or detections, prior_xy)


def choose_confidence_weighted_train_median(
    detections: list[dict[str, Any]], priors: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any] | None:
    return choose_confidence_weighted_point(
        detections,
        np.asarray(priors["train_median_xy"], dtype=np.float64),
        max(1.0, float(priors["train_prior_scale_px"])),
        args,
    )


def choose_confidence_weighted_point(
    detections: list[dict[str, Any]], prior_xy: np.ndarray, scale: float, args: argparse.Namespace
) -> dict[str, Any] | None:
    if not detections:
        return None
    weight = float(args.conf_prior_weight)

    def score(det: dict[str, Any]) -> float:
        dist = float(np.linalg.norm(np.asarray(det["center_xy"], dtype=np.float64) - prior_xy))
        conf = max(float(det["conf"]), 1.0e-9)
        return dist / scale - weight * math.log(conf)

    return min(detections, key=score)


def recording_median_prior(sample: dict[str, Any], priors: dict[str, Any]) -> np.ndarray:
    key = recording_phase_key(sample)
    xy = priors.get("recording_phase_medians", {}).get(key) or priors["train_median_xy"]
    return np.asarray(xy, dtype=np.float64)


def nearest_time_prior(sample: dict[str, Any], priors: dict[str, Any]) -> np.ndarray:
    key = recording_phase_key(sample)
    rel = sample.get("relative_time_s")
    ts = sample.get("timestamp_ns")
    same_key = [point for point in priors.get("train_temporal_points", []) if point.get("recording_phase_key") == key]
    if not same_key:
        return recording_median_prior(sample, priors)

    def distance(point: dict[str, Any]) -> float:
        point_rel = point.get("relative_time_s")
        if rel is not None and point_rel is not None:
            return abs(float(rel) - float(point_rel))
        point_ts = point.get("timestamp_ns")
        if ts is not None and point_ts is not None:
            return abs(float(ts) - float(point_ts)) / 1.0e9
        return 0.0

    nearest = min(same_key, key=distance)
    return np.asarray(nearest["target_2d"], dtype=np.float64)


def recording_phase_key(sample: dict[str, Any]) -> str:
    return f"{sample.get('recording_id', '')}|{sample.get('phase', '')}"


def choose_oracle_nearest(
    detections: list[dict[str, Any]], sample: dict[str, Any], top_k: int
) -> dict[str, Any] | None:
    if not detections:
        return None
    target_xy = np.asarray(sample["target_2d"], dtype=np.float64)
    candidates = detections[: int(top_k)]
    return min(
        candidates,
        key=lambda det: float(np.linalg.norm(np.asarray(det["center_xy"], dtype=np.float64) - target_xy)),
    )


def image_center(sample: dict[str, Any]) -> np.ndarray:
    width = float(sample.get("width") or sample.get("camera", {}).get("width") or 720.0)
    height = float(sample.get("height") or sample.get("camera", {}).get("height") or 720.0)
    return np.asarray([0.5 * width, 0.5 * height], dtype=np.float64)


def score_selection(
    task: str,
    sample: dict[str, Any],
    detections: list[dict[str, Any]],
    detection: dict[str, Any] | None,
    selector: str,
    deployable: bool,
    imgsz: int,
    conf: float,
    max_det: int,
    miss_penalty_m: float,
) -> dict[str, Any]:
    target_xy = np.asarray(sample["target_2d"], dtype=np.float64)
    target_depth = float(sample["target_depth"])
    camera = sample["camera"]
    gt_3d = np.asarray(
        sample.get("target_3d_camera") or backproject_np(target_xy, target_depth, camera),
        dtype=np.float64,
    )
    base = {
        "task": task,
        "split": sample["split"],
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "target_2d": sample["target_2d"],
        "target_depth": target_depth,
        "metric_depth_source": sample.get("metric_depth_source"),
        "selector": selector,
        "deployable_selector": bool(deployable),
        "imgsz": int(imgsz),
        "conf_threshold": float(conf),
        "max_det": int(max_det),
        "num_detections": len(detections),
    }
    if detection is None:
        return {
            **base,
            "missing_detection": True,
            "selected_rank": None,
            "pred_center_xy": None,
            "pred_xyxy": None,
            "pred_conf": None,
            "center_error_px": None,
            "error_3d_m": float(miss_penalty_m),
            "error_3d_m_covered_only": None,
        }
    pred_xy = np.asarray(detection["center_xy"], dtype=np.float64)
    pred_3d = backproject_np(pred_xy, target_depth, camera)
    err_3d = float(np.linalg.norm(pred_3d - gt_3d))
    return {
        **base,
        "missing_detection": False,
        "selected_rank": int(detection["rank"]),
        "pred_center_xy": [float(pred_xy[0]), float(pred_xy[1])],
        "pred_xyxy": detection["xyxy"],
        "pred_conf": float(detection["conf"]),
        "center_error_px": float(np.linalg.norm(pred_xy - target_xy)),
        "error_3d_m": err_3d,
        "error_3d_m_covered_only": err_3d,
    }


def summarize_all_rows(rows: list[dict[str, Any]], audit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["task"]),
                str(row["selector"]),
                int(row["imgsz"]),
                float(row["conf_threshold"]),
                int(row["max_det"]),
            )
        ].append(row)
    results = []
    for (task, selector, imgsz, conf, max_det), group_rows in sorted(groups.items()):
        metrics = summarize_rows(group_rows)
        results.append(
            {
                "task": task,
                "selector": selector,
                "deployable_selector": bool(group_rows[0].get("deployable_selector")),
                "imgsz": int(imgsz),
                "conf": float(conf),
                "max_det": int(max_det),
                "metrics": metrics,
            }
        )
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "checkpoint_tag": args.checkpoint_tag,
        "metric_definition": metric_definition("nearest-source-target"),
        "metric_caveat": (
            "Manual labels are 2D-only. Direct 2D center error is measured against the manual click; "
            "meter error is oracle-depth lateral 3D error with a miss penalty."
        ),
        "data_audit": audit,
        "results": results,
        "best_deployable_by_task": best_by_task(results, deployable_only=True),
        "best_oracle_by_task": best_by_task(results, deployable_only=False, oracle_only=True),
        "blockers": blockers_from_audit(audit),
        "artifacts": {
            "run_config": str((args.out_root / "run_config.json").resolve()),
            "data_audit": str((args.out_root / "data_audit.json").resolve()),
            "priors": str((args.out_root / "priors.json").resolve()),
            "per_sample_metrics": str((args.out_root / "per_sample_metrics.jsonl").resolve()),
            "summary": str((args.out_root / "summary.json").resolve()),
        },
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [row for row in rows if not row.get("missing_detection")]
    return {
        "num_samples": len(rows),
        "num_covered": len(covered),
        "num_missing_detection": len(rows) - len(covered),
        "miss_rate": rate(len(rows) - len(covered), len(rows)),
        "selected_rank": summarize_values([row.get("selected_rank") for row in covered], (1, 3, 5, 20), "rank"),
        "center_error_px": summarize_values([row.get("center_error_px") for row in covered], PIXEL_THRESHOLDS, "px"),
        "error_3d_m_covered_only": summarize_values(
            [row.get("error_3d_m_covered_only") for row in covered], METER_THRESHOLDS, "m"
        ),
        "error_3d_m_with_miss_penalty": summarize_values([row.get("error_3d_m") for row in rows], METER_THRESHOLDS, "m"),
    }


def best_by_task(
    results: list[dict[str, Any]], deployable_only: bool, oracle_only: bool = False
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_results = [result for result in results if result.get("task") == task]
        if deployable_only:
            task_results = [result for result in task_results if result.get("deployable_selector") is True]
        if oracle_only:
            task_results = [result for result in task_results if result.get("deployable_selector") is False]
        if not task_results:
            continue
        out[task] = compact_result(min(task_results, key=result_sort_key))
    return out


def result_sort_key(result: dict[str, Any]) -> tuple[float, float, float]:
    metrics = result.get("metrics", {})
    m = metrics.get("error_3d_m_with_miss_penalty", {}).get("mean")
    px = metrics.get("center_error_px", {}).get("mean")
    miss = metrics.get("miss_rate")
    return (
        float(m) if m is not None else float("inf"),
        float(px) if px is not None else float("inf"),
        float(miss) if miss is not None else float("inf"),
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    px = metrics.get("center_error_px", {})
    m = metrics.get("error_3d_m_with_miss_penalty", {})
    return {
        "task": result.get("task"),
        "selector": result.get("selector"),
        "deployable_selector": result.get("deployable_selector"),
        "imgsz": result.get("imgsz"),
        "conf": result.get("conf"),
        "max_det": result.get("max_det"),
        "num_samples": metrics.get("num_samples"),
        "num_missing_detection": metrics.get("num_missing_detection"),
        "miss_rate": metrics.get("miss_rate"),
        "center_error_px_mean": px.get("mean"),
        "center_error_px_median": px.get("median"),
        "center_error_px_p90": px.get("p90"),
        "oracle_depth_error_m_mean": m.get("mean"),
        "oracle_depth_error_m_median": m.get("median"),
        "oracle_depth_error_m_p90": m.get("p90"),
        "le_0p02m": m.get("le_0p02m"),
    }


def rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
