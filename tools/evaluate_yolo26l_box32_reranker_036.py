#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
    metric_definition,
    resolve_samples,
    result_to_detections,
    summarize_values,
)


TASK_ID = "yolo26l_box32_reranker_036"
TASKS = ("deleafing", "harvesting")
DEFAULT_LABELS = Path("/tmp/egodelta/front/labels.jsonl")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/yolo26l_box32_2dgt_improve_036/reranker_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a train-only candidate reranker for YOLO box32.")
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--task", default="deleafing", choices=TASKS)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.00001)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--miss-penalty-m", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--crop-size", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    set_thread_limits()
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    accepted_labels = write_accepted_only_labels(args.labels_jsonl, args.out_root / "labels_accepted_only.jsonl")
    samples, audit = resolve_samples(accepted_labels, args.front_root, "nearest-source-target")
    task_samples = [s for s in samples if s.get("task") == args.task and s.get("metric_ready") is True]
    global _SAMPLE_LOOKUP
    _SAMPLE_LOOKUP = {sample["sample_id"]: sample for sample in task_samples}
    train_samples = [s for s in task_samples if s.get("split") == "train"]
    test_samples = [s for s in task_samples if s.get("split") == "test"]
    priors = build_priors(train_samples)

    train_candidates = predict_candidates(train_samples, args)
    test_candidates = predict_candidates(test_samples, args)
    model = train_logistic_reranker(train_candidates, priors, args)
    rows = []
    rows.extend(score_split("train", train_samples, train_candidates, priors, model, args))
    rows.extend(score_split("test", test_samples, test_candidates, priors, model, args))

    write_json(args.out_root / "run_config.json", build_run_config(args, accepted_labels))
    write_json(args.out_root / "data_audit.json", audit)
    write_json(args.out_root / "priors.json", priors)
    write_json(args.out_root / "reranker_model.json", model_json(model))
    write_jsonl(args.out_root / "per_sample_metrics.jsonl", rows)
    summary = summarize(rows, audit, args)
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve())}, indent=2))
    return 0


def set_thread_limits() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


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


def build_priors(train_samples: list[dict[str, Any]]) -> dict[str, Any]:
    xy = np.asarray([sample["target_2d"] for sample in train_samples], dtype=np.float64)
    median_xy = np.median(xy, axis=0)
    mean_xy = np.mean(xy, axis=0)
    recording: dict[str, list[list[float]]] = {}
    for sample in train_samples:
        recording.setdefault(recording_key(sample), []).append([float(v) for v in sample["target_2d"]])
    recording_medians = {}
    for key, values in recording.items():
        arr = np.asarray(values, dtype=np.float64)
        med = np.median(arr, axis=0)
        recording_medians[key] = [float(med[0]), float(med[1])]
    return {
        "train_mean_xy": [float(mean_xy[0]), float(mean_xy[1])],
        "train_median_xy": [float(median_xy[0]), float(median_xy[1])],
        "recording_medians": recording_medians,
    }


def predict_candidates(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for YOLO prediction.") from exc
    model = YOLO(str(args.weights))
    results_iter = model.predict(
        source=[sample["image_path"] for sample in samples],
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        batch=int(args.batch),
        device=str(args.device),
        stream=True,
        save=False,
        verbose=False,
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for sample, result in zip(samples, results_iter):
        out[sample["sample_id"]] = result_to_detections(result)
    return out


def train_logistic_reranker(
    candidates_by_sample: dict[str, list[dict[str, Any]]],
    priors: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows_x = []
    rows_y = []
    rows_w = []
    for sample_id, detections in candidates_by_sample.items():
        if not detections:
            continue
        sample = sample_from_candidate_id(sample_id)
        target_xy = np.asarray(sample["target_2d"], dtype=np.float64)
        distances = [float(np.linalg.norm(np.asarray(det["center_xy"], dtype=np.float64) - target_xy)) for det in detections]
        best_idx = int(np.argmin(distances))
        for idx, det in enumerate(detections):
            rows_x.append(feature_vector(det, sample, priors, args))
            positive = idx == best_idx or distances[idx] <= 6.0
            rows_y.append(1.0 if positive else 0.0)
            rows_w.append(4.0 if positive else 1.0)
    if not rows_x:
        raise RuntimeError("No train candidates available for reranker")
    x = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    weights = np.asarray(rows_w, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1.0e-6] = 1.0
    xz = (x - mean[None, :]) / std[None, :]
    xz = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float64)], axis=1)
    beta = np.zeros((xz.shape[1],), dtype=np.float64)
    for _ in range(int(args.epochs)):
        logits = np.clip(xz @ beta, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = (xz.T @ ((pred - y) * weights)) / max(1.0, float(np.sum(weights)))
        grad[:-1] += float(args.l2) * beta[:-1]
        beta -= float(args.lr) * grad
    return {
        "feature_names": feature_names(int(args.crop_size)),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "beta": beta.tolist(),
        "train_num_candidates": int(len(rows_y)),
        "train_positive_rate": float(np.mean(y)),
        "crop_size": int(args.crop_size),
    }


def score_split(
    split: str,
    samples: list[dict[str, Any]],
    candidates_by_sample: dict[str, list[dict[str, Any]]],
    priors: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        detections = candidates_by_sample.get(sample["sample_id"], [])
        rows.append(score_one(sample, detections, choose_confidence(detections), split, "confidence", args))
        rows.append(score_one(sample, detections, choose_oracle(sample, detections, 20), split, "oracle_top20", args))
        rows.append(score_one(sample, detections, choose_logistic(sample, detections, priors, model), split, "train_logistic", args))
    return rows


def choose_confidence(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    return detections[0] if detections else None


def choose_oracle(sample: dict[str, Any], detections: list[dict[str, Any]], top_k: int) -> dict[str, Any] | None:
    if not detections:
        return None
    target_xy = np.asarray(sample["target_2d"], dtype=np.float64)
    return min(
        detections[:top_k],
        key=lambda det: float(np.linalg.norm(np.asarray(det["center_xy"], dtype=np.float64) - target_xy)),
    )


def choose_logistic(
    sample: dict[str, Any], detections: list[dict[str, Any]], priors: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any] | None:
    if not detections:
        return None
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    beta = np.asarray(model["beta"], dtype=np.float64)

    def score(det: dict[str, Any]) -> float:
        x = feature_vector(det, sample, priors, argparse.Namespace(crop_size=model.get("crop_size", 96)))
        z = (x - mean) / std
        zb = np.concatenate([z, np.ones((1,), dtype=np.float64)])
        return float(zb @ beta)

    return max(detections, key=score)


def feature_names(crop_size: int) -> list[str]:
    names = [
        "log_conf",
        "rank",
        "x",
        "y",
        "w",
        "h",
        "area",
        "dist_center",
        "dist_train_median",
        "dist_recording_median",
    ]
    if int(crop_size) > 0:
        names.extend([f"crop_gray_{idx:02d}" for idx in range(64)])
        names.extend(["crop_r_mean", "crop_g_mean", "crop_b_mean", "crop_r_std", "crop_g_std", "crop_b_std"])
    return names


def feature_vector(
    det: dict[str, Any], sample: dict[str, Any], priors: dict[str, Any], args: argparse.Namespace
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in det["xyxy"]]
    cx, cy = [float(v) for v in det["center_xy"]]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    image_center = np.asarray([360.0, 360.0], dtype=np.float64)
    train_median = np.asarray(priors["train_median_xy"], dtype=np.float64)
    rec_median = np.asarray(priors.get("recording_medians", {}).get(recording_key(sample), priors["train_median_xy"]))
    point = np.asarray([cx, cy], dtype=np.float64)
    geom = np.asarray(
        [
            math.log(max(float(det.get("conf", 0.0)), 1.0e-9)),
            float(det.get("rank", 999)),
            cx / 720.0,
            cy / 720.0,
            w / 720.0,
            h / 720.0,
            (w * h) / (720.0 * 720.0),
            float(np.linalg.norm(point - image_center)) / 720.0,
            float(np.linalg.norm(point - train_median)) / 720.0,
            float(np.linalg.norm(point - rec_median)) / 720.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([geom, crop_features(det, sample, int(args.crop_size))], axis=0)


_IMAGE_CACHE: dict[str, Image.Image] = {}


def crop_features(det: dict[str, Any], sample: dict[str, Any], crop_size: int) -> np.ndarray:
    if int(crop_size) <= 0:
        return np.zeros((0,), dtype=np.float64)
    path = str(sample["image_path"])
    image = _IMAGE_CACHE.get(path)
    if image is None:
        image = Image.open(path).convert("RGB")
        _IMAGE_CACHE[path] = image
    cx, cy = [float(v) for v in det["center_xy"]]
    half = max(8.0, 0.5 * float(crop_size))
    crop = image.crop((cx - half, cy - half, cx + half, cy + half)).resize((8, 8), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.float64) / 255.0
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).reshape(-1)
    mean = np.mean(arr, axis=(0, 1))
    std = np.std(arr, axis=(0, 1))
    return np.concatenate([gray, mean, std], axis=0).astype(np.float64)


def score_one(
    sample: dict[str, Any],
    detections: list[dict[str, Any]],
    detection: dict[str, Any] | None,
    split: str,
    selector: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_xy = np.asarray(sample["target_2d"], dtype=np.float64)
    target_depth = float(sample["target_depth"])
    camera = sample["camera"]
    gt_3d = np.asarray(sample.get("target_3d_camera") or backproject_np(target_xy, target_depth, camera), dtype=np.float64)
    base = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "split": split,
        "selector": selector,
        "num_detections": len(detections),
        "target_2d": sample["target_2d"],
        "target_depth": target_depth,
        "image_path": sample["image_path"],
    }
    if detection is None:
        return {
            **base,
            "missing_detection": True,
            "selected_rank": None,
            "pred_conf": None,
            "pred_center_xy": None,
            "center_error_px": None,
            "error_3d_m": float(args.miss_penalty_m),
            "error_3d_m_covered_only": None,
        }
    pred_xy = np.asarray(detection["center_xy"], dtype=np.float64)
    pred_3d = backproject_np(pred_xy, target_depth, camera)
    err_3d = float(np.linalg.norm(pred_3d - gt_3d))
    return {
        **base,
        "missing_detection": False,
        "selected_rank": int(detection["rank"]),
        "pred_conf": float(detection["conf"]),
        "pred_center_xy": [float(pred_xy[0]), float(pred_xy[1])],
        "center_error_px": float(np.linalg.norm(pred_xy - target_xy)),
        "error_3d_m": err_3d,
        "error_3d_m_covered_only": err_3d,
    }


def sample_from_candidate_id(sample_id: str) -> dict[str, Any]:
    sample = _SAMPLE_LOOKUP.get(sample_id)
    if sample is None:
        raise KeyError(sample_id)
    return sample


_SAMPLE_LOOKUP: dict[str, dict[str, Any]] = {}


def summarize(rows: list[dict[str, Any]], audit: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    groups = {}
    for row in rows:
        key = f"{row['split']}/{row['selector']}"
        groups.setdefault(key, []).append(row)
    metrics = {key: summarize_rows(group_rows) for key, group_rows in sorted(groups.items())}
    test_results = [
        {
            "selector": key.split("/", 1)[1],
            "metrics": value,
        }
        for key, value in metrics.items()
        if key.startswith("test/")
    ]
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "checkpoint_tag": args.checkpoint_tag,
        "metric_definition": metric_definition("nearest-source-target"),
        "metric_caveat": "Manual labels are 2D-only; meter error is oracle-depth lateral 3D error with miss penalty.",
        "data_audit": audit,
        "metrics": metrics,
        "best_test_deployable": compact_result(min([r for r in test_results if r["selector"] != "oracle_top20"], key=sort_key)),
        "test_oracle": compact_result([r for r in test_results if r["selector"] == "oracle_top20"][0]),
        "artifacts": {
            "run_config": str((args.out_root / "run_config.json").resolve()),
            "reranker_model": str((args.out_root / "reranker_model.json").resolve()),
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
        "miss_rate": (len(rows) - len(covered)) / len(rows) if rows else None,
        "selected_rank": summarize_values([row.get("selected_rank") for row in covered], (1, 3, 5, 20), "rank"),
        "center_error_px": summarize_values([row.get("center_error_px") for row in covered], PIXEL_THRESHOLDS, "px"),
        "error_3d_m_covered_only": summarize_values(
            [row.get("error_3d_m_covered_only") for row in covered], METER_THRESHOLDS, "m"
        ),
        "error_3d_m_with_miss_penalty": summarize_values([row.get("error_3d_m") for row in rows], METER_THRESHOLDS, "m"),
    }


def sort_key(result: dict[str, Any]) -> tuple[float, float]:
    metrics = result["metrics"]
    return (
        float(metrics["error_3d_m_with_miss_penalty"].get("mean", float("inf"))),
        float(metrics["center_error_px"].get("mean", float("inf"))),
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    px = metrics["center_error_px"]
    m = metrics["error_3d_m_with_miss_penalty"]
    return {
        "selector": result["selector"],
        "num_samples": metrics["num_samples"],
        "miss_rate": metrics["miss_rate"],
        "num_missing_detection": metrics["num_missing_detection"],
        "center_error_px_mean": px.get("mean"),
        "center_error_px_median": px.get("median"),
        "center_error_px_p90": px.get("p90"),
        "oracle_depth_error_m_mean": m.get("mean"),
        "oracle_depth_error_m_median": m.get("median"),
        "oracle_depth_error_m_p90": m.get("p90"),
        "le_0p02m": m.get("le_0p02m"),
    }


def build_run_config(args: argparse.Namespace, accepted_labels: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "accepted_only_labels_jsonl": str(accepted_labels.resolve()),
        "front_root": str(args.front_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "task": args.task,
        "weights": str(args.weights.resolve()),
        "checkpoint_tag": args.checkpoint_tag,
        "prediction": {
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "max_det": int(args.max_det),
            "iou": float(args.iou),
            "batch": int(args.batch),
            "device": str(args.device),
        },
        "reranker": {"epochs": int(args.epochs), "lr": float(args.lr), "l2": float(args.l2)},
    }


def model_json(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if key != "samples"}


def recording_key(sample: dict[str, Any]) -> str:
    return f"{sample.get('recording_id', '')}|{sample.get('phase', '')}"


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
