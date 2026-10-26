#!/usr/bin/env python3
"""Run full-frame benchmark evaluation for all labeled recordings."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_one_sample import (  # noqa: E402
    EVAL_THRESHOLDS_M,
    evaluate_task,
    load_deleaf_predictions,
    load_harvest_predictions,
    parse_labels,
    write_combined_overlay,
)
from verify_pairs_fast_alignment import verify_recording_alignment  # noqa: E402
from yolo_aria_harvest_poc.config import PipelineConfig  # noqa: E402
from yolo_aria_harvest_poc.mps_geometry import MPSGeometry  # noqa: E402


TASKS = ("harvesting", "deleafing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt")
    parser.add_argument("--split-root", default="/path/to/data/mps/egodelta_benchmark_splitted")
    parser.add_argument("--full-root", default="/path/to/data/mps/egodelta_benchmark")
    parser.add_argument(
        "--pairs-root",
        help="Root containing recording_id/{tomato_pair,leaf_pair}/manifest.json with pre-extracted frames.",
    )
    parser.add_argument("--out", default="outputs/eval_benchmark_full_mps")
    parser.add_argument("--tasks", default="harvesting,deleafing")
    parser.add_argument("--model", default="best_tomato.pt", help="Unused when harvest is run with YOLOE.")
    parser.add_argument("--yoloe-model", default="yoloe-11s-seg.pt")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--image-view", choices=["raw", "egodelta_pinhole", "pinhole"], default="raw")
    parser.add_argument("--aria-camera-label", default="camera-rgb")
    parser.add_argument("--rect-size", type=int, default=720)
    parser.add_argument("--rect-focal", type=float, default=380.0)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--harvest-imgsz", type=int, default=720)
    parser.add_argument("--deleaf-imgsz", type=int, default=720)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--segmentation-batch-size", type=int, default=1)
    parser.add_argument("--visualization-stride", type=int, default=30)
    parser.add_argument(
        "--eval-threshold-m",
        action="append",
        type=float,
        default=None,
        help="3D matching threshold in meters. Can be passed multiple times. Defaults to 0.02, 0.035, 0.05.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recording-id", action="append")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-coordinate-check", action="store_true")
    parser.add_argument("--alignment-max-translation-m", type=float, default=0.02)
    parser.add_argument("--alignment-max-rotation-deg", type=float, default=1.0)
    parser.add_argument("--alignment-max-frames-per-dataset", type=int, default=0)
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.segmentation_batch_size < 1:
        raise SystemExit("--segmentation-batch-size must be >= 1")
    if args.visualization_stride < 1:
        raise SystemExit("--visualization-stride must be >= 1")
    eval_thresholds_m = tuple(args.eval_threshold_m) if args.eval_threshold_m else EVAL_THRESHOLDS_M
    if args.pairs_root:
        args.image_view = "pinhole"

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.labels)
    requested_tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    for task in requested_tasks:
        if task not in TASKS:
            raise SystemExit(f"Unsupported task {task}; choose from {TASKS}")

    pairs_root = Path(args.pairs_root) if args.pairs_root else None
    recording_ids = list(labels)
    if pairs_root is not None and not args.recording_id:
        recording_ids = [
            recording_id
            for recording_id in recording_ids
            if (pairs_root / recording_id / "tomato_pair" / "manifest.json").exists()
            or (pairs_root / recording_id / "leaf_pair" / "manifest.json").exists()
        ]
    if args.recording_id:
        wanted = set(args.recording_id)
        recording_ids = [recording_id for recording_id in recording_ids if recording_id in wanted]
    if args.num_shards > 1:
        recording_ids = [
            recording_id
            for idx, recording_id in enumerate(recording_ids)
            if idx % args.num_shards == args.shard_index
        ]
    if args.limit is not None:
        recording_ids = recording_ids[: args.limit]

    sample_summaries: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    for sample_index, recording_id in enumerate(recording_ids, start=1):
        print(f"===== [{sample_index}/{len(recording_ids)}] {recording_id} =====", flush=True)
        timing: dict[str, Any] = {}
        vrs_path = Path(args.full_root) / f"{recording_id}.vrs"
        mps_dir = Path(args.full_root) / f"mps_{recording_id}_vrs"
        if pairs_root is None:
            timing = load_timing_json(Path(args.split_root), recording_id)
            vrs_path = Path(timing.get("vrs_path") or vrs_path)
            mps_dir = Path(timing.get("mps_dir") or mps_dir)
        sample_out = out_root / recording_id
        sample_out.mkdir(parents=True, exist_ok=True)
        alignment_verification = None
        if pairs_root is not None and not args.skip_coordinate_check:
            alignment_verification = verify_recording_alignment(
                recording_id=recording_id,
                pairs_root=pairs_root,
                full_root=Path(args.full_root),
                tasks=requested_tasks,
                rgb_stream_id=args.rgb_stream_id,
                aria_camera_label=args.aria_camera_label,
                rect_size=args.rect_size,
                rect_focal=args.rect_focal,
                max_frames_per_dataset=args.alignment_max_frames_per_dataset,
                max_translation_m=args.alignment_max_translation_m,
                max_rotation_deg=args.alignment_max_rotation_deg,
            )
            eval_dir = sample_out / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "coordinate_alignment_verification.json").write_text(
                json.dumps(alignment_verification, indent=2),
                encoding="utf-8",
            )
            if not alignment_verification["matched"]:
                raise RuntimeError(
                    f"pairs_fast coordinates do not match full-MPS world for {recording_id}. "
                    f"max translation={alignment_verification['max_translation_m']:.4f} m, "
                    f"max rotation={alignment_verification['max_rotation_deg']:.4f} deg"
                )

        if "harvesting" in requested_tasks:
            harvest_out = sample_out / "harvest"
            if should_run(harvest_out / "final_predictions.json", args.force, args.skip_pipeline):
                if pairs_root is not None:
                    run_pair_pipeline(args, "harvest", pair_manifest_path(pairs_root, recording_id, "harvesting"), harvest_out)
                else:
                    run_harvest_pipeline(args, timing, vrs_path, mps_dir, harvest_out)
        if "deleafing" in requested_tasks:
            deleaf_out = sample_out / "deleaf"
            if should_run(deleaf_out / "final_predictions_deleaf.json", args.force, args.skip_pipeline):
                if pairs_root is not None:
                    run_pair_pipeline(args, "deleaf", pair_manifest_path(pairs_root, recording_id, "deleafing"), deleaf_out)
                else:
                    run_deleaf_pipeline(args, timing, vrs_path, mps_dir, deleaf_out)

        sample_eval, combined_overlay = evaluate_sample(
            recording_id=recording_id,
            labels=labels[recording_id],
            vrs_path=vrs_path,
            mps_dir=mps_dir,
            sample_out=sample_out,
            tasks=requested_tasks,
            rgb_stream_id=args.rgb_stream_id,
            image_view=args.image_view,
            aria_camera_label=args.aria_camera_label,
            rect_size=args.rect_size,
            rect_focal=args.rect_focal,
            pairs_root=pairs_root,
            alignment_verification=alignment_verification,
            eval_thresholds_m=eval_thresholds_m,
        )
        sample_summaries.append(sample_eval)
        if combined_overlay is not None:
            overlay_paths.append(combined_overlay)
        (sample_out / "eval" / "one_sample_eval.json").write_text(
            json.dumps(sample_eval, indent=2),
            encoding="utf-8",
        )

    aggregate = aggregate_results(sample_summaries)
    aggregate["num_samples"] = len(sample_summaries)
    aggregate["recording_ids"] = [item["recording_id"] for item in sample_summaries]
    aggregate_path = out_root / "aggregate_eval.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_aggregate_csv(sample_summaries, out_root / "aggregate_table.csv")
    contact_sheet = write_overlay_contact_sheet(overlay_paths, out_root / "aggregate_overlay_contact_sheet.jpg")
    if contact_sheet is not None:
        aggregate["aggregate_overlay_contact_sheet"] = str(contact_sheet)
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


def should_run(final_path: Path, force: bool, skip_pipeline: bool) -> bool:
    if skip_pipeline:
        return False
    return force or not final_path.exists()


def load_timing_json(split_root: Path, recording_id: str) -> dict[str, Any]:
    path = split_root / "timing_json" / f"{recording_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing timing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def segment_window(timing: dict[str, Any], phase: str) -> tuple[float, float]:
    for segment in timing.get("segments", []):
        if segment.get("phase") == phase or segment.get("label") == phase:
            return float(segment["start_s"]), float(segment["end_s"])
    raise KeyError(f"Could not find segment {phase} in timing metadata")


def pair_manifest_path(pairs_root: Path, recording_id: str, task: str) -> Path:
    pair_name = {"harvesting": "tomato_pair", "deleafing": "leaf_pair"}[task]
    path = pairs_root / recording_id / pair_name / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing extracted pair manifest: {path}")
    return path


def run_pair_pipeline(args: argparse.Namespace, task: str, pair_manifest: Path, out: Path) -> None:
    task_config = "configs/harvest.yaml" if task == "harvest" else "configs/deleaf.yaml"
    imgsz = args.harvest_imgsz if task == "harvest" else args.deleaf_imgsz
    cmd = [
        sys.executable,
        "-m",
        "yolo_aria_harvest_poc.run_pipeline",
        "--task",
        task,
        "--task-config",
        task_config,
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
        str(imgsz),
        "--max-det",
        str(args.max_det),
        "--segmentation-batch-size",
        str(args.segmentation_batch_size),
        "--visualization-stride",
        str(args.visualization_stride),
        "--world-up",
        "0,0,1",
        "--out",
        str(out),
        "--debug",
    ]
    run_command(cmd)


def run_harvest_pipeline(args: argparse.Namespace, timing: dict[str, Any], vrs_path: Path, mps_dir: Path, out: Path) -> None:
    pre_start, pre_end = segment_window(timing, "post_deleafing_pre_harvesting")
    post_start, post_end = segment_window(timing, "post_harvesting")
    cmd = [
        sys.executable,
        "-m",
        "yolo_aria_harvest_poc.run_pipeline",
        "--task",
        "harvest",
        "--task-config",
        "configs/harvest.yaml",
        "--mode",
        "single_recording",
        "--vrs",
        str(vrs_path),
        "--mps",
        str(mps_dir),
        "--pre-start-s",
        str(pre_start),
        "--pre-end-s",
        str(pre_end),
        "--post-start-s",
        str(post_start),
        "--post-end-s",
        str(post_end),
        "--segmentation-backend",
        "yoloe_prompt",
        "--yoloe-model",
        args.yoloe_model,
        "--rgb-stream-id",
        args.rgb_stream_id,
        "--image-view",
        args.image_view,
        "--aria-camera-label",
        args.aria_camera_label,
        "--rect-size",
        str(args.rect_size),
        "--rect-focal",
        str(args.rect_focal),
        "--frame-stride",
        str(args.frame_stride),
        "--imgsz",
        str(args.harvest_imgsz),
        "--max-det",
        str(args.max_det),
        "--segmentation-batch-size",
        str(args.segmentation_batch_size),
        "--visualization-stride",
        str(args.visualization_stride),
        "--world-up",
        "0,0,1",
        "--out",
        str(out),
        "--debug",
    ]
    run_command(cmd)


def run_deleaf_pipeline(args: argparse.Namespace, timing: dict[str, Any], vrs_path: Path, mps_dir: Path, out: Path) -> None:
    pre_start, pre_end = segment_window(timing, "pre_deleafing")
    post_start, post_end = segment_window(timing, "post_deleafing_pre_harvesting")
    cmd = [
        sys.executable,
        "-m",
        "yolo_aria_harvest_poc.run_pipeline",
        "--task",
        "deleaf",
        "--task-config",
        "configs/deleaf.yaml",
        "--mode",
        "single_recording",
        "--vrs",
        str(vrs_path),
        "--mps",
        str(mps_dir),
        "--pre-start-s",
        str(pre_start),
        "--pre-end-s",
        str(pre_end),
        "--post-start-s",
        str(post_start),
        "--post-end-s",
        str(post_end),
        "--segmentation-backend",
        "yoloe_prompt",
        "--yoloe-model",
        args.yoloe_model,
        "--rgb-stream-id",
        args.rgb_stream_id,
        "--image-view",
        args.image_view,
        "--aria-camera-label",
        args.aria_camera_label,
        "--rect-size",
        str(args.rect_size),
        "--rect-focal",
        str(args.rect_focal),
        "--frame-stride",
        str(args.frame_stride),
        "--imgsz",
        str(args.deleaf_imgsz),
        "--max-det",
        str(args.max_det),
        "--segmentation-batch-size",
        str(args.segmentation_batch_size),
        "--visualization-stride",
        str(args.visualization_stride),
        "--world-up",
        "0,0,1",
        "--out",
        str(out),
        "--debug",
    ]
    run_command(cmd)


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate_sample(
    recording_id: str,
    labels: dict[str, list[list[float]]],
    vrs_path: Path,
    mps_dir: Path,
    sample_out: Path,
    tasks: tuple[str, ...],
    rgb_stream_id: str,
    image_view: str,
    aria_camera_label: str,
    rect_size: int,
    rect_focal: float,
    pairs_root: Path | None = None,
    alignment_verification: dict[str, Any] | None = None,
    eval_thresholds_m: tuple[float, ...] = EVAL_THRESHOLDS_M,
) -> tuple[dict[str, Any], Path | None]:
    eval_out = sample_out / "eval"
    eval_out.mkdir(parents=True, exist_ok=True)
    geometry_config = PipelineConfig(
        rgb_stream_id=rgb_stream_id,
        image_view="pinhole" if pairs_root is not None else image_view,
        aria_camera_label=aria_camera_label,
        rect_size=rect_size,
        rect_focal=rect_focal,
    )
    shared_geometry = None
    if pairs_root is None:
        shared_geometry = MPSGeometry.from_paths(mps_dir, vrs_path, geometry_config)
    task_results: dict[str, Any] = {}
    overlays: list[Path] = []
    if "harvesting" in tasks:
        geometry = (
            MPSGeometry.from_pair_manifest(pair_manifest_path(pairs_root, recording_id, "harvesting"), geometry_config)
            if pairs_root is not None
            else shared_geometry
        )
        result = evaluate_task(
            task="harvesting",
            gt_points=np.asarray(labels.get("harvesting", []), dtype=float).reshape(-1, 3),
            pred_points=load_harvest_predictions(sample_out / "harvest"),
            frames_path=sample_out / "harvest" / "frames" / "pre_frames.json",
            geometry=geometry,
            out_dir=eval_out,
            thresholds_m=eval_thresholds_m,
        )
        task_results["harvesting"] = result
        if result.get("overlay_path"):
            overlays.append(Path(result["overlay_path"]))
    if "deleafing" in tasks:
        geometry = (
            MPSGeometry.from_pair_manifest(pair_manifest_path(pairs_root, recording_id, "deleafing"), geometry_config)
            if pairs_root is not None
            else shared_geometry
        )
        result = evaluate_task(
            task="deleafing",
            gt_points=np.asarray(labels.get("deleafing", []), dtype=float).reshape(-1, 3),
            pred_points=load_deleaf_predictions(sample_out / "deleaf"),
            frames_path=sample_out / "deleaf" / "frames" / "pre_frames.json",
            geometry=geometry,
            out_dir=eval_out,
            thresholds_m=eval_thresholds_m,
        )
        task_results["deleafing"] = result
        if result.get("overlay_path"):
            overlays.append(Path(result["overlay_path"]))
    combined_overlay = write_combined_overlay(overlays, eval_out / "one_sample_eval_overlay.jpg")
    sample = {
        "recording_id": recording_id,
        "label_frame": "full_mps_world",
        "prediction_frame": "pairs_fast_colmap_world" if pairs_root is not None else "full_mps_world",
        "coordinate_alignment": {
            "transform_applied_to_predictions_before_evaluation": "identity",
            "verified_against_full_mps": bool(alignment_verification and alignment_verification.get("matched")),
            "verification_path": (
                str(sample_out / "eval" / "coordinate_alignment_verification.json")
                if alignment_verification is not None
                else None
            ),
            "note": (
                "pairs_fast predictions are compared directly to GT labels after verifying "
                "the pair manifest COLMAP camera poses against full-MPS camera poses."
                if pairs_root is not None
                else "Predictions and GT are both evaluated in the full-MPS world frame."
            ),
        },
        "metric": "euclidean_absolute_error_cm",
        "tasks": task_results,
    }
    if combined_overlay is not None:
        sample["combined_overlay_path"] = str(combined_overlay)
    return sample, combined_overlay


def aggregate_results(samples: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"tasks": {}}
    for task in TASKS:
        errors: list[float] = []
        num_gt = num_pred = num_matched = num_missed = num_extra = 0
        threshold_totals: dict[str, dict[str, Any]] = {}
        sample_rows = []
        for sample in samples:
            result = sample.get("tasks", {}).get(task)
            if result is None:
                continue
            task_errors = [float(match["error_cm"]) for match in result.get("matches", [])]
            errors.extend(task_errors)
            num_gt += int(result["num_gt"])
            num_pred += int(result["num_pred"])
            num_matched += int(result["num_matched"])
            num_missed += len(result.get("missed_gt_indices", []))
            num_extra += len(result.get("unmatched_pred_indices", []))
            for threshold_key, threshold_result in result.get("threshold_metrics", {}).items():
                totals = threshold_totals.setdefault(
                    threshold_key,
                    {
                        "threshold_m": threshold_result.get("threshold_m"),
                        "threshold_cm": threshold_result.get("threshold_cm"),
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "matched_distances_cm": [],
                        "gt_to_nearest_distances_cm": [],
                        "gt_to_nearest_undefined_count": 0,
                    },
                )
                totals["TP"] += int(threshold_result.get("TP", 0))
                totals["FP"] += int(threshold_result.get("FP", 0))
                totals["FN"] += int(threshold_result.get("FN", 0))
                totals["matched_distances_cm"].extend(
                    float(match["error_cm"]) for match in threshold_result.get("matches", [])
                )
                nearest = threshold_result.get("gt_to_nearest_distances_cm") or []
                totals["gt_to_nearest_distances_cm"].extend(float(value) for value in nearest)
                if result["num_gt"] and not nearest:
                    totals["gt_to_nearest_undefined_count"] += int(result["num_gt"])
            sample_rows.append(
                {
                    "recording_id": sample["recording_id"],
                    "num_gt": result["num_gt"],
                    "num_pred": result["num_pred"],
                    "num_matched": result["num_matched"],
                    "mean_error_cm": result["mean_error_cm"],
                    "median_error_cm": result["median_error_cm"],
                    "max_error_cm": result["max_error_cm"],
                    "num_missed": len(result.get("missed_gt_indices", [])),
                    "num_extra": len(result.get("unmatched_pred_indices", [])),
                }
            )
        threshold_metrics: dict[str, Any] = {}
        for threshold_key, totals in threshold_totals.items():
            tp = int(totals["TP"])
            fp = int(totals["FP"])
            fn = int(totals["FN"])
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and (precision + recall) > 0.0
                else None
            )
            matched_distances = totals["matched_distances_cm"]
            nearest_distances = totals["gt_to_nearest_distances_cm"]
            threshold_metrics[threshold_key] = {
                "threshold_m": totals["threshold_m"],
                "threshold_cm": totals["threshold_cm"],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_mean_distance_cm": float(np.mean(matched_distances)) if matched_distances else None,
                "matched_median_distance_cm": float(np.median(matched_distances)) if matched_distances else None,
                "gt_to_nearest_mean_distance_cm": float(np.mean(nearest_distances)) if nearest_distances else None,
                "gt_to_nearest_median_distance_cm": float(np.median(nearest_distances)) if nearest_distances else None,
                "gt_to_nearest_undefined_count": totals["gt_to_nearest_undefined_count"],
            }
        aggregate["tasks"][task] = {
            "num_gt": num_gt,
            "num_pred": num_pred,
            "num_matched": num_matched,
            "num_missed": num_missed,
            "num_extra_predictions": num_extra,
            "mean_error_cm": float(np.mean(errors)) if errors else None,
            "median_error_cm": float(np.median(errors)) if errors else None,
            "max_error_cm": float(np.max(errors)) if errors else None,
            "threshold_metrics": threshold_metrics,
            "sample_results": sample_rows,
        }
    return aggregate


def write_aggregate_csv(samples: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for sample in samples:
        for task, result in sample.get("tasks", {}).items():
            rows.append(
                {
                    "recording_id": sample["recording_id"],
                    "task": task,
                    "num_gt": result["num_gt"],
                    "num_pred": result["num_pred"],
                    "num_matched": result["num_matched"],
                    "mean_error_cm": result["mean_error_cm"],
                    "median_error_cm": result["median_error_cm"],
                    "max_error_cm": result["max_error_cm"],
                    "num_missed": len(result.get("missed_gt_indices", [])),
                    "num_extra": len(result.get("unmatched_pred_indices", [])),
                    "overlay_path": result.get("overlay_path"),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recording_id",
                "task",
                "num_gt",
                "num_pred",
                "num_matched",
                "mean_error_cm",
                "median_error_cm",
                "max_error_cm",
                "num_missed",
                "num_extra",
                "overlay_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_overlay_contact_sheet(paths: list[Path], out_path: Path) -> Path | None:
    if not paths:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (640, 240))
        cv2.putText(image, path.parent.parent.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        images.append(image)
    if not images:
        return None
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    canvas = np.full((rows * 240, cols * 640, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        canvas[r * 240 : (r + 1) * 240, c * 640 : (c + 1) * 640] = image
    cv2.imwrite(str(out_path), canvas)
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
