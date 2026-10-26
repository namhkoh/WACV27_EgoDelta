#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TASK_ID = "manual_front_yolo_eval_032"
DEFAULT_LABELS = Path("/tmp/egodelta/front/labels.jsonl")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/manual_front_yolo_032/eval")
TASKS = ("deleafing", "harvesting")
SPLITS = ("train", "test")
METER_THRESHOLDS = (0.01, 0.02, 0.05)
PIXEL_THRESHOLDS = (5.0, 10.0, 20.0, 50.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate task-specific YOLO bbox-center predictions on the manually "
            "labeled front-view 2D point dataset. This script does not train."
        )
    )
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "test", "all"))
    parser.add_argument("--tasks", default="deleafing,harvesting")
    parser.add_argument("--deleafing-weights", type=Path, default=None)
    parser.add_argument("--harvesting-weights", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="Optional shared weights for all requested tasks.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--miss-penalty-m", type=float, default=1.0)
    parser.add_argument("--overlay-count", type=int, default=80)
    parser.add_argument(
        "--depth-policy",
        default="nearest-source-target",
        choices=("manual-only", "nearest-source-target"),
        help=(
            "manual-only requires target_depth/camera in labels.jsonl. "
            "nearest-source-target recovers camera/depth from the nearest old source "
            "target on the same VRS frame and records the approximation diagnostics."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true", help="Resolve/audit samples, but do not load YOLO.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    splits = list(SPLITS) if args.split == "all" else [args.split]
    args.out_root.mkdir(parents=True, exist_ok=True)
    samples, audit = resolve_samples(args.labels_jsonl, args.front_root, args.depth_policy)
    selected = [s for s in samples if s["task"] in tasks and s["split"] in splits]
    write_json(args.out_root / "run_config.json", build_run_config(args, tasks, splits))
    write_json(args.out_root / "data_audit.json", audit)
    write_jsonl(args.out_root / "resolved_samples.jsonl", samples)
    if args.prepare_only:
        summary = {
            "task_id": TASK_ID,
            "created_at_utc": utc_now(),
            "mode": "prepare_only",
            "selected_samples": len(selected),
            "data_audit": audit,
            "blockers": blockers_from_audit(audit),
            "artifacts": artifact_payload(args.out_root),
        }
        write_json(args.out_root / "summary.json", summary)
        print(json.dumps({"status": "prepare_only_complete", "summary": str(args.out_root / "summary.json")}, indent=2))
        return 0

    weights_by_task = weights_map(args)
    missing_weights = [task for task in tasks if weights_by_task.get(task) is None]
    if missing_weights:
        raise SystemExit(f"Missing YOLO weights for tasks: {', '.join(missing_weights)}")
    unresolved = [s for s in selected if s.get("metric_ready") is not True]
    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} selected samples lack metric-ready camera/depth. "
            "Use --depth-policy nearest-source-target or add target_depth/camera to labels."
        )

    all_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_samples = [s for s in selected if s["task"] == task]
        if not task_samples:
            continue
        rows = evaluate_task(task, task_samples, weights_by_task[task], args)
        all_rows.extend(rows)
    write_jsonl(args.out_root / "per_sample_metrics.jsonl", all_rows)
    summary = summarize_eval(all_rows, audit, args)
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps({"status": "complete", "summary": str(args.out_root / "summary.json")}, indent=2))
    return 0


def parse_tasks(text: str) -> list[str]:
    tasks = [part.strip() for part in str(text).split(",") if part.strip()]
    bad = [task for task in tasks if task not in TASKS]
    if bad:
        raise ValueError(f"Unknown task(s): {bad}")
    return tasks


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_run_config(args: argparse.Namespace, tasks: list[str], splits: list[str]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "front_root": str(args.front_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "tasks": tasks,
        "splits": splits,
        "depth_policy": args.depth_policy,
        "prepare_only": bool(args.prepare_only),
        "prediction": {
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "batch": int(args.batch),
            "device": str(args.device),
        },
        "weights": {
            "shared": str(args.weights.resolve()) if args.weights else None,
            "deleafing": str(args.deleafing_weights.resolve()) if args.deleafing_weights else None,
            "harvesting": str(args.harvesting_weights.resolve()) if args.harvesting_weights else None,
        },
        "metric_definition": metric_definition(args.depth_policy),
    }


def metric_definition(depth_policy: str) -> dict[str, Any]:
    return {
        "2d_error_px": "Euclidean distance between YOLO bbox center and manual clicked point in rectified 720x720 RGB.",
        "3d_error_m": (
            "Backproject predicted bbox center and manual clicked point with the same target depth, then compute "
            "Euclidean camera-frame distance. This measures lateral 3D error of the 2D localizer."
        ),
        "depth_source": (
            "labels.jsonl target_depth/camera if present; otherwise nearest old source target on the same VRS frame"
            if depth_policy == "nearest-source-target"
            else "manual labels only; samples without target_depth/camera are unresolved"
        ),
        "strictness_caveat": (
            "A strict 3D point metric requires each manual click to carry its own target_depth_m or target_3d_camera. "
            "Nearest-source-target depth is an approximation when manual clicks moved away from old projected labels."
        ),
    }


def weights_map(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "deleafing": args.deleafing_weights or args.weights,
        "harvesting": args.harvesting_weights or args.weights,
    }


def resolve_samples(labels_path: Path, front_root: Path, depth_policy: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = load_labels(labels_path)
    manifest_rows = load_companion_manifests(front_root)
    by_rel = {rel_from_front(Path(row["output_path"]), front_root): row for row in manifest_rows}
    source_cache: dict[str, list[dict[str, Any]]] = {}
    samples: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    ambiguity: Counter[str] = Counter()
    nearest_distances: dict[str, list[float]] = defaultdict(list)
    missing_join: list[str] = []
    rejected = 0
    accepted = 0
    for label in labels:
        rel = str(label.get("image", ""))
        task = str(label.get("task", ""))
        status = str(label.get("status", ""))
        row = by_rel.get(rel)
        if row is None:
            counters["missing_companion_manifest"] += 1
            missing_join.append(rel)
            continue
        base = {
            "sample_id": sample_id_from_rel(rel),
            "task": task,
            "status": status,
            "split": str(row.get("split", "")),
            "image_path": str((front_root / rel).resolve()),
            "source_image_path": row.get("source_image_path"),
            "recording_id": row.get("recording_id"),
            "phase": row.get("phase"),
            "timestamp_ns": safe_int(row.get("timestamp_ns")),
            "relative_time_s": safe_float(row.get("relative_time_s")),
            "source_manifest": row.get("source_manifest"),
            "width": int(label.get("image_width") or 0),
            "height": int(label.get("image_height") or 0),
        }
        if status != "accepted":
            rejected += 1
            base["metric_ready"] = False
            samples.append(base)
            continue
        accepted += 1
        x = safe_float(label.get("x"))
        y = safe_float(label.get("y"))
        if x is None or y is None:
            counters["accepted_missing_xy"] += 1
            base["metric_ready"] = False
            samples.append(base)
            continue
        base["target_2d"] = [float(x), float(y)]
        metric = metric_fields_from_label(label)
        if metric is None and depth_policy == "nearest-source-target":
            metric = metric_fields_from_source(row, [float(x), float(y)], source_cache)
        if metric is None:
            counters["missing_metric_depth_or_camera"] += 1
            base["metric_ready"] = False
        else:
            base.update(metric)
            base["metric_ready"] = True
            ambiguity[f"{task}_source_match_count_{metric.get('source_match_count', 0)}"] += 1
            if metric.get("source_nearest_target_px") is not None:
                nearest_distances[task].append(float(metric["source_nearest_target_px"]))
        samples.append(base)
    audit = {
        "labels_path": str(labels_path.resolve()),
        "front_root": str(front_root.resolve()),
        "num_label_rows": len(labels),
        "num_manifest_rows": len(manifest_rows),
        "accepted_rows": accepted,
        "rejected_rows": rejected,
        "accepted_by_task": dict(Counter(s["task"] for s in samples if s.get("status") == "accepted")),
        "accepted_by_task_split": dict(
            stringify_counter(Counter((s["task"], s["split"]) for s in samples if s.get("status") == "accepted"))
        ),
        "metric_ready_rows": sum(1 for s in samples if s.get("metric_ready") is True),
        "metric_ready_by_task_split": dict(
            stringify_counter(Counter((s["task"], s["split"]) for s in samples if s.get("metric_ready") is True))
        ),
        "counters": dict(counters),
        "source_join_ambiguity": dict(ambiguity),
        "source_nearest_target_px": {
            task: summarize_values(values, PIXEL_THRESHOLDS, "px")
            for task, values in nearest_distances.items()
        },
        "missing_companion_examples": missing_join[:10],
        "metric_definition": metric_definition(depth_policy),
    }
    return samples, audit


def load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_companion_manifests(front_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in TASKS:
        path = front_root / f"{task}_manifest.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def rel_from_front(path: Path, front_root: Path) -> str:
    try:
        return path.resolve().relative_to(front_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sample_id_from_rel(rel: str) -> str:
    return Path(rel).with_suffix("").as_posix().replace("/", "__")


def metric_fields_from_label(label: dict[str, Any]) -> dict[str, Any] | None:
    camera = label.get("camera")
    depth = label.get("target_depth", label.get("target_depth_m"))
    target_3d = label.get("target_3d_camera")
    if camera is None or depth is None:
        return None
    try:
        depth_f = float(depth)
        camera_d = normalize_camera(camera)
    except (TypeError, ValueError):
        return None
    out = {
        "target_depth": depth_f,
        "camera": camera_d,
        "metric_depth_source": "manual_label",
        "source_match_count": None,
        "source_nearest_target_px": None,
    }
    if target_3d is not None:
        out["target_3d_camera"] = [float(v) for v in target_3d]
    return out


def metric_fields_from_source(
    row: dict[str, str],
    manual_xy: list[float],
    source_cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    path = row.get("source_manifest")
    if not path:
        return None
    if path not in source_cache:
        source_cache[path] = json.loads(Path(path).read_text(encoding="utf-8")).get("samples", [])
    matches = [
        sample
        for sample in source_cache[path]
        if sample.get("image_path") == row.get("source_image_path")
        and str(sample.get("timestamp_ns")) == str(row.get("timestamp_ns"))
        and str(sample.get("recording_id")) == str(row.get("recording_id"))
        and str(sample.get("phase")) == str(row.get("phase"))
    ]
    scored = []
    for sample in matches:
        target = sample.get("target_2d")
        if target is None:
            continue
        dist = math.hypot(float(manual_xy[0]) - float(target[0]), float(manual_xy[1]) - float(target[1]))
        scored.append((dist, sample))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    nearest_dist, sample = scored[0]
    camera = sample.get("camera")
    depth = sample.get("target_depth")
    if camera is None or depth is None:
        return None
    return {
        "target_depth": float(depth),
        "camera": normalize_camera(camera),
        "metric_depth_source": "nearest_source_target_depth",
        "source_match_count": len(matches),
        "source_nearest_target_px": float(nearest_dist),
        "source_sample_id": sample.get("sample_id"),
        "source_target_2d": [float(v) for v in sample.get("target_2d", [])],
        "source_target_3d_camera": [float(v) for v in sample.get("target_3d_camera", [])]
        if sample.get("target_3d_camera") is not None
        else None,
    }


def normalize_camera(camera: dict[str, Any]) -> dict[str, float]:
    return {
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
        "width": float(camera.get("width", 720)),
        "height": float(camera.get("height", 720)),
    }


def evaluate_task(task: str, samples: list[dict[str, Any]], weights: Path | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    if weights is None:
        raise RuntimeError(f"No weights for {task}")
    if not weights.exists():
        raise FileNotFoundError(weights)
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for YOLO prediction.") from exc

    model = YOLO(str(weights))
    image_paths = [sample["image_path"] for sample in samples]
    results_iter = model.predict(
        source=image_paths,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        batch=int(args.batch),
        device=args.device if not isinstance(args.device, str) else str(args.device),
        stream=True,
        save=False,
        verbose=False,
    )
    rows = []
    overlay_budget = int(args.overlay_count)
    overlay_dir = args.out_root / "prediction_overlays" / task
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for sample, result in zip(samples, results_iter):
        detections = result_to_detections(result)
        row = score_sample(sample, detections, float(args.miss_penalty_m))
        rows.append(row)
        if overlay_budget > 0:
            write_overlay(sample, detections, row, overlay_dir)
            overlay_budget -= 1
    return rows


def result_to_detections(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
    order = np.argsort(-conf)
    detections: list[dict[str, Any]] = []
    for rank, idx in enumerate(order.tolist(), start=1):
        x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
        detections.append(
            {
                "rank": int(rank),
                "xyxy": [x1, y1, x2, y2],
                "center_xy": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
                "conf": float(conf[idx]),
            }
        )
    return detections


def score_sample(sample: dict[str, Any], detections: list[dict[str, Any]], miss_penalty_m: float) -> dict[str, Any]:
    target_xy = np.asarray(sample["target_2d"], dtype=np.float64)
    target_depth = float(sample["target_depth"])
    camera = sample["camera"]
    gt_3d = np.asarray(
        sample.get("target_3d_camera") or backproject_np(target_xy, target_depth, camera),
        dtype=np.float64,
    )
    if not detections:
        return {
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "split": sample["split"],
            "image_path": sample["image_path"],
            "target_2d": sample["target_2d"],
            "target_depth": target_depth,
            "metric_depth_source": sample.get("metric_depth_source"),
            "missing_detection": True,
            "num_detections": 0,
            "pred_center_xy": None,
            "pred_xyxy": None,
            "pred_conf": None,
            "center_error_px": None,
            "error_3d_m": float(miss_penalty_m),
            "error_3d_m_covered_only": None,
        }
    top = detections[0]
    pred_xy = np.asarray(top["center_xy"], dtype=np.float64)
    pred_3d = backproject_np(pred_xy, target_depth, camera)
    center_error = float(np.linalg.norm(pred_xy - target_xy))
    err_3d = float(np.linalg.norm(pred_3d - gt_3d))
    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "split": sample["split"],
        "image_path": sample["image_path"],
        "source_image_path": sample.get("source_image_path"),
        "target_2d": sample["target_2d"],
        "target_depth": target_depth,
        "metric_depth_source": sample.get("metric_depth_source"),
        "source_match_count": sample.get("source_match_count"),
        "source_nearest_target_px": sample.get("source_nearest_target_px"),
        "missing_detection": False,
        "num_detections": len(detections),
        "pred_center_xy": [float(pred_xy[0]), float(pred_xy[1])],
        "pred_xyxy": top["xyxy"],
        "pred_conf": float(top["conf"]),
        "center_error_px": center_error,
        "error_3d_m": err_3d,
        "error_3d_m_covered_only": err_3d,
    }


def backproject_np(xy: np.ndarray, depth: float, camera: dict[str, Any]) -> np.ndarray:
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    z = float(depth)
    return np.asarray([(float(xy[0]) - cx) * z / fx, (float(xy[1]) - cy) * z / fy, z], dtype=np.float64)


def write_overlay(sample: dict[str, Any], detections: list[dict[str, Any]], row: dict[str, Any], out_dir: Path) -> None:
    image = Image.open(sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    tx, ty = sample["target_2d"]
    draw_cross(draw, float(tx), float(ty), color=(255, 220, 0), radius=12, width=3)
    draw.text((8, 8), f"GT {sample['task']} {sample['split']}", fill=(255, 255, 255), font=font)
    if detections:
        top = detections[0]
        x1, y1, x2, y2 = top["xyxy"]
        px, py = top["center_xy"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=3)
        draw_cross(draw, float(px), float(py), color=(0, 255, 255), radius=10, width=3)
        label = f"{row['center_error_px']:.1f}px {row['error_3d_m']:.3f}m conf {row['pred_conf']:.3f}"
    else:
        label = "miss"
    draw.rectangle([8, image.height - 28, min(image.width - 8, 440), image.height - 8], fill=(0, 0, 0))
    draw.text((14, image.height - 24), label, fill=(255, 255, 255), font=font)
    out_path = out_dir / f"{safe_stem(sample['sample_id'])}.jpg"
    image.save(out_path, quality=92)
    row["overlay_path"] = str(out_path)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int, width: int) -> None:
    draw.line([x - radius, y, x + radius, y], fill=color, width=width)
    draw.line([x, y - radius, x, y + radius], fill=color, width=width)
    draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=width)


def safe_stem(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(keep)[:180]


def summarize_eval(rows: list[dict[str, Any]], audit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"combined": rows}
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task") == task]
        if task_rows:
            groups[task] = task_rows
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "metric_definition": metric_definition(args.depth_policy),
        "data_audit": audit,
        "metrics": {name: summarize_rows(group_rows) for name, group_rows in groups.items()},
        "artifacts": artifact_payload(args.out_root),
        "blockers": blockers_from_audit(audit),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [row for row in rows if not row.get("missing_detection")]
    return {
        "num_samples": len(rows),
        "num_covered": len(covered),
        "num_missing_detection": len(rows) - len(covered),
        "miss_rate": rate(len(rows) - len(covered), len(rows)),
        "center_error_px": summarize_values([row.get("center_error_px") for row in covered], PIXEL_THRESHOLDS, "px"),
        "error_3d_m_covered_only": summarize_values(
            [row.get("error_3d_m_covered_only") for row in covered], METER_THRESHOLDS, "m"
        ),
        "error_3d_m_with_miss_penalty": summarize_values([row.get("error_3d_m") for row in rows], METER_THRESHOLDS, "m"),
    }


def summarize_values(values: list[Any], thresholds: tuple[float, ...], suffix: str) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }
    for threshold in thresholds:
        key = f"le_{format_threshold(threshold)}{suffix}"
        out[key] = float(np.mean(arr <= float(threshold)))
    return out


def format_threshold(value: float) -> str:
    if value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{value:.0f}"


def blockers_from_audit(audit: dict[str, Any]) -> list[str]:
    blockers = []
    if audit.get("metric_ready_rows", 0) < audit.get("accepted_rows", 0):
        blockers.append(
            "Some accepted manual labels lack metric-ready camera/depth. Add target_depth/camera or use an approximate depth policy."
        )
    source_px = audit.get("source_nearest_target_px", {})
    if source_px:
        blockers.append(
            "Strict 3D GT is not present in labels.jsonl. nearest-source-target depth is approximate; manual clicks are often far from old source targets."
        )
    return blockers


def artifact_payload(out_root: Path) -> dict[str, str]:
    return {
        "run_config": str((out_root / "run_config.json").resolve()),
        "data_audit": str((out_root / "data_audit.json").resolve()),
        "resolved_samples": str((out_root / "resolved_samples.jsonl").resolve()),
        "per_sample_metrics": str((out_root / "per_sample_metrics.jsonl").resolve()),
        "summary": str((out_root / "summary.json").resolve()),
        "prediction_overlays": str((out_root / "prediction_overlays").resolve()),
    }


def safe_float(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value == "" or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def rate(num: int, den: int) -> float | None:
    return float(num) / float(den) if den else None


def stringify_counter(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {f"{key[0]}/{key[1]}": int(value) for key, value in counter.items()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
