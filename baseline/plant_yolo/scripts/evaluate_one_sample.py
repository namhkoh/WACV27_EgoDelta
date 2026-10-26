#!/usr/bin/env python3
"""Evaluate one benchmark recording against full-MPS-frame 3D labels."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yolo_aria_harvest_poc.config import PipelineConfig
from yolo_aria_harvest_poc.mps_geometry import MPSGeometry, project_world_points_to_rgb


@dataclass
class Match:
    pred_index: int
    gt_index: int
    error_m: float


EVAL_THRESHOLDS_M = (0.02, 0.035, 0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mps", required=True)
    parser.add_argument("--vrs", required=True)
    parser.add_argument("--harvest-out")
    parser.add_argument("--deleaf-out")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--image-view", choices=["raw", "egodelta_pinhole"], default=None)
    parser.add_argument("--aria-camera-label", default=None)
    parser.add_argument("--rect-size", type=int, default=None)
    parser.add_argument("--rect-focal", type=float, default=None)
    parser.add_argument(
        "--eval-threshold-m",
        action="append",
        type=float,
        default=None,
        help="3D matching threshold in meters. Can be passed multiple times. Defaults to 0.02, 0.035, 0.05.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.labels)
    if args.recording_id not in labels:
        raise SystemExit(f"Recording {args.recording_id} was not found in {args.labels}")

    inferred_view = infer_camera_view_from_outputs(args.harvest_out, args.deleaf_out)
    config = PipelineConfig(
        rgb_stream_id=args.rgb_stream_id,
        image_view=args.image_view or inferred_view.get("image_view", "raw"),
        aria_camera_label=args.aria_camera_label or inferred_view.get("camera_label", "camera-rgb"),
        rect_size=int(args.rect_size or inferred_view.get("rect_size") or 720),
        rect_focal=float(args.rect_focal or inferred_view.get("rect_focal") or 380.0),
    )
    geometry = MPSGeometry.from_paths(args.mps, args.vrs, config)
    sample_labels = labels[args.recording_id]
    eval_thresholds_m = tuple(args.eval_threshold_m) if args.eval_threshold_m else EVAL_THRESHOLDS_M

    task_results: dict[str, Any] = {}
    overlays: list[Path] = []
    if args.harvest_out:
        result = evaluate_task(
            task="harvesting",
            gt_points=np.asarray(sample_labels.get("harvesting", []), dtype=float).reshape(-1, 3),
            pred_points=load_harvest_predictions(Path(args.harvest_out)),
            frames_path=Path(args.harvest_out) / "frames" / "pre_frames.json",
            geometry=geometry,
            out_dir=out_dir,
            thresholds_m=eval_thresholds_m,
        )
        task_results["harvesting"] = result
        if result.get("overlay_path"):
            overlays.append(Path(result["overlay_path"]))
    if args.deleaf_out:
        result = evaluate_task(
            task="deleafing",
            gt_points=np.asarray(sample_labels.get("deleafing", []), dtype=float).reshape(-1, 3),
            pred_points=load_deleaf_predictions(Path(args.deleaf_out)),
            frames_path=Path(args.deleaf_out) / "frames" / "pre_frames.json",
            geometry=geometry,
            out_dir=out_dir,
            thresholds_m=eval_thresholds_m,
        )
        task_results["deleafing"] = result
        if result.get("overlay_path"):
            overlays.append(Path(result["overlay_path"]))

    summary = {
        "recording_id": args.recording_id,
        "label_frame": "full_mps_world",
        "metric": "euclidean_absolute_error_cm",
        "tasks": task_results,
    }
    summary_path = out_dir / "one_sample_eval.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    combined_path = write_combined_overlay(overlays, out_dir / "one_sample_eval_overlay.jpg")
    if combined_path is not None:
        summary["combined_overlay_path"] = str(combined_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
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


def load_harvest_predictions(out_dir: Path) -> np.ndarray:
    path = out_dir / "final_predictions.json"
    data = load_json(path, default={"removed_tomatoes": []})
    points = []
    for item in data.get("removed_tomatoes", []):
        value = item.get("cut_point_3d_world") or item.get("proxy_cut_point_3d_world")
        if value is not None:
            points.append(value)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def load_deleaf_predictions(out_dir: Path) -> np.ndarray:
    path = out_dir / "final_predictions_deleaf.json"
    data = load_json(path, default={"removed_petioles": []})
    points = []
    for item in data.get("removed_petioles", []):
        value = item.get("proxy_cut_point_3d_world") or item.get("cut_point_3d_world")
        if value is not None:
            points.append(value)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def infer_camera_view_from_outputs(*out_dirs: str | None) -> dict[str, Any]:
    for out_dir_text in out_dirs:
        if not out_dir_text:
            continue
        out_dir = Path(out_dir_text)
        for path in [
            out_dir / "frames" / "single_recording_camera_view.json",
            out_dir / "frames" / "pre_camera_view.json",
            out_dir / "frames" / "pre_frames.json",
        ]:
            if not path.exists():
                continue
            data = load_json(path, default={})
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict) and data.get("image_view"):
                return data
    return {"image_view": "raw"}


def evaluate_task(
    task: str,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    frames_path: Path,
    geometry: MPSGeometry,
    out_dir: Path,
    thresholds_m: tuple[float, ...] = EVAL_THRESHOLDS_M,
) -> dict[str, Any]:
    matches = match_points(pred_points, gt_points)
    errors_cm = [100.0 * match.error_m for match in matches]
    threshold_metrics = compute_threshold_metrics(pred_points, gt_points, thresholds_m)
    frames = load_json(frames_path, default=[])
    overlay_path = write_task_overlay(
        task=task,
        frames=frames,
        gt_points=gt_points,
        pred_points=pred_points,
        matches=matches,
        geometry=geometry,
        out_path=out_dir / f"{task}_eval_overlay.jpg",
    )
    matched_pred = {match.pred_index for match in matches}
    matched_gt = {match.gt_index for match in matches}
    return {
        "num_gt": int(len(gt_points)),
        "num_pred": int(len(pred_points)),
        "num_matched": int(len(matches)),
        "mean_error_cm": float(np.mean(errors_cm)) if errors_cm else None,
        "median_error_cm": float(np.median(errors_cm)) if errors_cm else None,
        "max_error_cm": float(np.max(errors_cm)) if errors_cm else None,
        "matches": [
            {
                "pred_index": match.pred_index,
                "gt_index": match.gt_index,
                "pred_point_world": pred_points[match.pred_index].tolist(),
                "gt_point_world": gt_points[match.gt_index].tolist(),
                "error_cm": 100.0 * match.error_m,
            }
            for match in matches
        ],
        "threshold_metrics": threshold_metrics,
        "missed_gt_indices": [idx for idx in range(len(gt_points)) if idx not in matched_gt],
        "unmatched_pred_indices": [idx for idx in range(len(pred_points)) if idx not in matched_pred],
        "overlay_path": str(overlay_path) if overlay_path is not None else None,
    }


def match_points(pred_points: np.ndarray, gt_points: np.ndarray) -> list[Match]:
    if len(pred_points) == 0 or len(gt_points) == 0:
        return []
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        pred_indices, gt_indices = linear_sum_assignment(distances)
        return [
            Match(int(pred_idx), int(gt_idx), float(distances[pred_idx, gt_idx]))
            for pred_idx, gt_idx in zip(pred_indices, gt_indices)
        ]
    except Exception:
        pairs = [
            (float(distances[pred_idx, gt_idx]), pred_idx, gt_idx)
            for pred_idx in range(distances.shape[0])
            for gt_idx in range(distances.shape[1])
        ]
        pairs.sort()
        used_pred: set[int] = set()
        used_gt: set[int] = set()
        matches: list[Match] = []
        for distance, pred_idx, gt_idx in pairs:
            if pred_idx in used_pred or gt_idx in used_gt:
                continue
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            matches.append(Match(int(pred_idx), int(gt_idx), float(distance)))
        return matches


def match_points_with_threshold(pred_points: np.ndarray, gt_points: np.ndarray, threshold_m: float) -> list[Match]:
    if len(pred_points) == 0 or len(gt_points) == 0:
        return []
    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    large_cost = float(max(threshold_m + 1.0, distances.max(initial=0.0) + 1.0))
    costs = np.where(distances <= threshold_m, distances, large_cost)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        pred_indices, gt_indices = linear_sum_assignment(costs)
        return [
            Match(int(pred_idx), int(gt_idx), float(distances[pred_idx, gt_idx]))
            for pred_idx, gt_idx in zip(pred_indices, gt_indices)
            if distances[pred_idx, gt_idx] <= threshold_m
        ]
    except Exception:
        pairs = [
            (float(distances[pred_idx, gt_idx]), pred_idx, gt_idx)
            for pred_idx in range(distances.shape[0])
            for gt_idx in range(distances.shape[1])
            if distances[pred_idx, gt_idx] <= threshold_m
        ]
        pairs.sort()
        used_pred: set[int] = set()
        used_gt: set[int] = set()
        matches: list[Match] = []
        for distance, pred_idx, gt_idx in pairs:
            if pred_idx in used_pred or gt_idx in used_gt:
                continue
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
            matches.append(Match(int(pred_idx), int(gt_idx), float(distance)))
        return matches


def compute_threshold_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds_m: tuple[float, ...] = EVAL_THRESHOLDS_M,
) -> dict[str, Any]:
    if len(gt_points) and len(pred_points):
        nearest_gt_distances_m = np.min(
            np.linalg.norm(gt_points[:, None, :] - pred_points[None, :, :], axis=2),
            axis=1,
        )
        gt_to_nearest_mean_cm = float(100.0 * np.mean(nearest_gt_distances_m))
        gt_to_nearest_median_cm = float(100.0 * np.median(nearest_gt_distances_m))
    else:
        nearest_gt_distances_m = np.asarray([], dtype=float)
        gt_to_nearest_mean_cm = None
        gt_to_nearest_median_cm = None

    metrics: dict[str, Any] = {}
    for threshold_m in thresholds_m:
        threshold_cm = 100.0 * threshold_m
        matches = match_points_with_threshold(pred_points, gt_points, threshold_m)
        tp = len(matches)
        fp = len(pred_points) - tp
        fn = len(gt_points) - tp
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0.0
            else None
        )
        matched_distances_cm = [100.0 * match.error_m for match in matches]
        key = f"{threshold_cm:g}cm"
        metrics[key] = {
            "threshold_m": threshold_m,
            "threshold_cm": threshold_cm,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_mean_distance_cm": float(np.mean(matched_distances_cm)) if matched_distances_cm else None,
            "matched_median_distance_cm": float(np.median(matched_distances_cm)) if matched_distances_cm else None,
            "gt_to_nearest_mean_distance_cm": gt_to_nearest_mean_cm,
            "gt_to_nearest_median_distance_cm": gt_to_nearest_median_cm,
            "gt_to_nearest_distances_cm": (100.0 * nearest_gt_distances_m).tolist(),
            "matches": [
                {
                    "pred_index": match.pred_index,
                    "gt_index": match.gt_index,
                    "error_cm": 100.0 * match.error_m,
                }
                for match in matches
            ],
        }
    return metrics


def write_task_overlay(
    task: str,
    frames: list[dict[str, Any]],
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    matches: list[Match],
    geometry: MPSGeometry,
    out_path: Path,
) -> Path | None:
    if not frames:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    all_points = np.vstack([arr for arr in [gt_points, pred_points] if len(arr)]) if (
        len(gt_points) or len(pred_points)
    ) else np.zeros((0, 3), dtype=float)
    if len(all_points) == 0:
        return None

    best_frame = None
    best_projection = None
    best_count = -1
    for frame in frames:
        T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
        uv, _, valid = project_world_points_to_rgb(all_points, T_world_rgb, geometry.rgb_camera_calibration)
        count = int(valid.sum())
        if count > best_count:
            best_count = count
            best_frame = frame
            best_projection = (uv, valid)
    if best_frame is None or best_projection is None:
        return None

    image = cv2.imread(str(best_frame["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return None

    uv_all, valid_all = best_projection
    num_gt = len(gt_points)
    gt_uv = uv_all[:num_gt]
    gt_valid = valid_all[:num_gt]
    pred_uv = uv_all[num_gt:]
    pred_valid = valid_all[num_gt:]

    overlay = image.copy()
    cv2.putText(
        overlay,
        f"{task} eval frame={best_frame['frame_id']} visible={best_count}/{len(all_points)}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for idx, uv in enumerate(gt_uv):
        if not gt_valid[idx]:
            continue
        point = tuple(np.round(uv).astype(int).tolist())
        cv2.drawMarker(overlay, point, (0, 255, 0), cv2.MARKER_CROSS, 24, 3)
        cv2.putText(overlay, f"GT{idx}", (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    for idx, uv in enumerate(pred_uv):
        if not pred_valid[idx]:
            continue
        point = tuple(np.round(uv).astype(int).tolist())
        cv2.circle(overlay, point, 9, (0, 0, 255), 2)
        cv2.putText(overlay, f"P{idx}", (point[0] + 6, point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    for match in matches:
        if not (gt_valid[match.gt_index] and pred_valid[match.pred_index]):
            continue
        gt_point = tuple(np.round(gt_uv[match.gt_index]).astype(int).tolist())
        pred_point = tuple(np.round(pred_uv[match.pred_index]).astype(int).tolist())
        mid = ((gt_point[0] + pred_point[0]) // 2, (gt_point[1] + pred_point[1]) // 2)
        cv2.line(overlay, gt_point, pred_point, (0, 255, 255), 2)
        cv2.putText(
            overlay,
            f"{100.0 * match.error_m:.1f}cm",
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), overlay)
    return out_path


def write_combined_overlay(paths: list[Path], out_path: Path) -> Path | None:
    if not paths:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return None
    target_h = min(image.shape[0] for image in images)
    resized = []
    for image in images:
        scale = target_h / image.shape[0]
        resized.append(cv2.resize(image, (int(round(image.shape[1] * scale)), target_h)))
    combined = cv2.hconcat(resized) if len(resized) > 1 else resized[0]
    cv2.imwrite(str(out_path), combined)
    return out_path


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
