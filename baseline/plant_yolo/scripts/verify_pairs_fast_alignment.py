#!/usr/bin/env python3
"""Verify pairs_fast COLMAP poses are in the full-MPS world frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_aria_harvest_poc.mps_geometry import (  # noqa: E402
    _load_colmap_image_poses,
    find_mps_file,
    load_closed_loop_trajectory,
    load_rgb_calibration,
    nearest_pose_index,
)


PAIR_BY_TASK = {
    "harvesting": "tomato_pair",
    "deleafing": "leaf_pair",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-root", required=True)
    parser.add_argument("--full-root", default="/path/to/data/mps/egodelta_benchmark")
    parser.add_argument("--recording-id", action="append")
    parser.add_argument("--tasks", default="harvesting,deleafing")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--aria-camera-label", default="camera-rgb")
    parser.add_argument("--rect-size", type=int, default=720)
    parser.add_argument("--rect-focal", type=float, default=380.0)
    parser.add_argument("--max-frames-per-dataset", type=int, default=0)
    parser.add_argument("--max-translation-m", type=float, default=0.02)
    parser.add_argument("--max-rotation-deg", type=float, default=1.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    pairs_root = Path(args.pairs_root)
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    recording_ids = args.recording_id or sorted(path.name for path in pairs_root.iterdir() if path.is_dir())
    results = []
    for recording_id in recording_ids:
        result = verify_recording_alignment(
            recording_id=recording_id,
            pairs_root=pairs_root,
            full_root=Path(args.full_root),
            tasks=tasks,
            rgb_stream_id=args.rgb_stream_id,
            aria_camera_label=args.aria_camera_label,
            rect_size=args.rect_size,
            rect_focal=args.rect_focal,
            max_frames_per_dataset=args.max_frames_per_dataset,
            max_translation_m=args.max_translation_m,
            max_rotation_deg=args.max_rotation_deg,
        )
        results.append(result)
        status = "OK" if result["matched"] else "MISMATCH"
        print(
            f"{status} {recording_id}: "
            f"max_t={result['max_translation_m']:.6f} m, "
            f"max_r={result['max_rotation_deg']:.6f} deg, "
            f"n={result['num_compared']}",
            flush=True,
        )

    aggregate = aggregate_alignment_results(results, args.max_translation_m, args.max_rotation_deg)
    payload = {"aggregate": aggregate, "recordings": results}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0 if aggregate["matched"] else 1


def verify_recording_alignment(
    recording_id: str,
    pairs_root: Path,
    full_root: Path,
    tasks: tuple[str, ...],
    rgb_stream_id: str = "214-1",
    aria_camera_label: str = "camera-rgb",
    rect_size: int = 720,
    rect_focal: float = 380.0,
    max_frames_per_dataset: int = 0,
    max_translation_m: float = 0.02,
    max_rotation_deg: float = 1.0,
) -> dict[str, Any]:
    vrs_path = full_root / f"{recording_id}.vrs"
    mps_dir = full_root / f"mps_{recording_id}_vrs"
    trajectory_path = find_mps_file(mps_dir, ["closed_loop_trajectory.csv"])
    trajectory = load_closed_loop_trajectory(trajectory_path)
    _, T_device_rgb = load_rgb_calibration(
        vrs_path,
        rgb_stream_id,
        image_view="egodelta_pinhole",
        rect_size=rect_size,
        rect_focal=rect_focal,
        aria_camera_label=aria_camera_label,
    )

    task_results = []
    for task in tasks:
        pair_name = PAIR_BY_TASK.get(task)
        if pair_name is None:
            raise ValueError(f"Unsupported task for alignment verification: {task}")
        pair_manifest = pairs_root / recording_id / pair_name / "manifest.json"
        if not pair_manifest.exists():
            continue
        task_results.append(
            verify_pair_manifest_alignment(
                pair_manifest=pair_manifest,
                trajectory=trajectory,
                T_device_rgb=T_device_rgb,
                max_frames_per_dataset=max_frames_per_dataset,
            )
        )
    if not task_results:
        raise FileNotFoundError(f"No pair manifests found for {recording_id} in {pairs_root}")

    translation_errors = [v for task in task_results for v in task["translation_errors_m"]]
    rotation_errors = [v for task in task_results for v in task["rotation_errors_deg"]]
    time_deltas = [v for task in task_results for v in task["nearest_pose_time_delta_ms"]]
    max_translation = float(np.max(translation_errors)) if translation_errors else float("inf")
    max_rotation = float(np.max(rotation_errors)) if rotation_errors else float("inf")
    matched = max_translation <= max_translation_m and max_rotation <= max_rotation_deg
    return {
        "recording_id": recording_id,
        "matched": bool(matched),
        "thresholds": {
            "max_translation_m": float(max_translation_m),
            "max_rotation_deg": float(max_rotation_deg),
        },
        "num_compared": int(len(translation_errors)),
        "mean_translation_m": _mean_or_none(translation_errors),
        "median_translation_m": _median_or_none(translation_errors),
        "max_translation_m": max_translation,
        "mean_rotation_deg": _mean_or_none(rotation_errors),
        "median_rotation_deg": _median_or_none(rotation_errors),
        "max_rotation_deg": max_rotation,
        "max_nearest_pose_time_delta_ms": float(np.max(time_deltas)) if time_deltas else None,
        "tasks": task_results,
    }


def verify_pair_manifest_alignment(
    pair_manifest: Path,
    trajectory: list[Any],
    T_device_rgb: np.ndarray,
    max_frames_per_dataset: int = 0,
) -> dict[str, Any]:
    manifest = json.loads(pair_manifest.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    time_deltas: list[float] = []
    dataset_results = []
    for dataset in datasets:
        sparse_dir = Path(str(dataset["dataset_dir"])) / "sparse" / "0"
        colmap_poses = _load_colmap_image_poses(sparse_dir / "images.txt")
        frame_timestamps = load_frame_timestamp_map(Path(str(dataset["source_root"])))
        items = sorted(colmap_poses.items())
        if max_frames_per_dataset > 0:
            step = max(1, len(items) // max_frames_per_dataset)
            items = items[::step][:max_frames_per_dataset]
        ds_t: list[float] = []
        ds_r: list[float] = []
        ds_dt: list[float] = []
        for frame_idx, T_pair_world_rgb in items:
            if frame_idx not in frame_timestamps:
                continue
            timestamp_ns = int(frame_timestamps[frame_idx])
            pose_idx = nearest_pose_index(trajectory, timestamp_ns)
            nearest = trajectory[pose_idx]
            T_full_world_rgb = nearest.T_world_device @ T_device_rgb
            ds_t.append(float(np.linalg.norm(T_pair_world_rgb[:3, 3] - T_full_world_rgb[:3, 3])))
            ds_r.append(rotation_error_deg(T_pair_world_rgb[:3, :3], T_full_world_rgb[:3, :3]))
            ds_dt.append(abs(int(nearest.timestamp_ns) - timestamp_ns) / 1e6)
        translation_errors.extend(ds_t)
        rotation_errors.extend(ds_r)
        time_deltas.extend(ds_dt)
        dataset_results.append(
            {
                "label": dataset.get("label"),
                "dataset_dir": dataset.get("dataset_dir"),
                "num_compared": int(len(ds_t)),
                "mean_translation_m": _mean_or_none(ds_t),
                "max_translation_m": float(np.max(ds_t)) if ds_t else None,
                "mean_rotation_deg": _mean_or_none(ds_r),
                "max_rotation_deg": float(np.max(ds_r)) if ds_r else None,
                "max_nearest_pose_time_delta_ms": float(np.max(ds_dt)) if ds_dt else None,
            }
        )
    return {
        "pair_manifest": str(pair_manifest),
        "num_compared": int(len(translation_errors)),
        "translation_errors_m": translation_errors,
        "rotation_errors_deg": rotation_errors,
        "nearest_pose_time_delta_ms": time_deltas,
        "datasets": dataset_results,
    }


def load_frame_timestamp_map(source_root: Path) -> dict[int, int]:
    index_path = source_root / "frame_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing frame index CSV: {index_path}")
    out: dict[int, int] = {}
    with index_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            out[int(row["frame_idx"])] = int(row["timestamp_ns"])
    return out


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    value = float((np.trace(R_delta) - 1.0) * 0.5)
    value = max(-1.0, min(1.0, value))
    return float(math.degrees(math.acos(value)))


def aggregate_alignment_results(
    results: list[dict[str, Any]],
    max_translation_m: float,
    max_rotation_deg: float,
) -> dict[str, Any]:
    translations = [v for result in results for task in result["tasks"] for v in task["translation_errors_m"]]
    rotations = [v for result in results for task in result["tasks"] for v in task["rotation_errors_deg"]]
    return {
        "matched": bool(all(result["matched"] for result in results)),
        "num_recordings": len(results),
        "num_compared": len(translations),
        "thresholds": {
            "max_translation_m": float(max_translation_m),
            "max_rotation_deg": float(max_rotation_deg),
        },
        "mean_translation_m": _mean_or_none(translations),
        "median_translation_m": _median_or_none(translations),
        "max_translation_m": float(np.max(translations)) if translations else None,
        "mean_rotation_deg": _mean_or_none(rotations),
        "median_rotation_deg": _median_or_none(rotations),
        "max_rotation_deg": float(np.max(rotations)) if rotations else None,
        "mismatched_recordings": [result["recording_id"] for result in results if not result["matched"]],
    }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
