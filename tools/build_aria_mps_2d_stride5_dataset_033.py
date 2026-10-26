#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, TextIO

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from projectaria_tools.core import data_provider

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from manual_cut_point_gt_tool import (  # noqa: E402
    image_record_timestamp_ns,
    load_trajectory,
    make_rectified_camera,
    pose_at,
    rectify_image,
)


TASK_ID = "aria_mps_2d_stride5_data_033"
TASKS = ("deleafing", "harvesting")
TASK_TO_PHASE = {
    "deleafing": "pre_deleafing",
    "harvesting": "post_deleafing_pre_harvesting",
}
PHASE_TO_TASK = {phase: task for task, phase in TASK_TO_PHASE.items()}
PRIOR_TEST_RECORDINGS = {
    "5479d13e-b1ed-4539-aeb4-c42ce20dbf8f",
    "7830a118-0c98-44f6-b856-1d4205da228a",
    "7e2bd97a-c200-47ce-8080-48b3dfee93ba",
    "dd6fab2f-6fc2-4093-b385-79329e121e36",
}
RECORDING_LINE_RE = re.compile(
    r"^\s*(?P<ordinal>\d+)\.\s*(?P<recording>[0-9a-fA-F-]{36})\.vrs\s*$"
)
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/aria_mps_2d_stride5_033")
DEFAULT_LABELS_PATH = REPO_ROOT / "3d_points_labels.txt"
DEFAULT_REPORT_PATH = REPO_ROOT / "research" / "REPORTS" / "aria_mps_2d_stride5_data_033.md"
DEFAULT_SPLIT_ROOT = Path("/path/to/data/mps/egodelta_benchmark_splitted")
DEFAULT_MULTI_SLAM_ROOT = Path("/path/to/data/mps/mps_multi_slam_split")
BOX_SIZES = (32, 48, 64)
LABEL_DECIMALS = 8


@dataclass(frozen=True)
class LabelPoint:
    recording_order: int
    recording_id: str
    task: str
    label_index: int
    point_world: tuple[float, float, float]
    line_number: int

    @property
    def phase(self) -> str:
        return TASK_TO_PHASE[self.task]

    @property
    def label_id(self) -> str:
        return f"{self.recording_id}_{self.task}_label{self.label_index:03d}"


@dataclass(frozen=True)
class PhaseAssets:
    recording_id: str
    phase: str
    vrs_path: Path
    mps_dir: Path
    phase_times_path: Path
    source: str


@dataclass
class PhaseContext:
    assets: PhaseAssets
    provider: Any
    rgb_id: Any
    camera: dict[str, Any]
    trajectory: Any
    phase_start_ns: int | None
    phase_end_ns: int | None
    n_source_frames: int


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.handle.close()


class YoloExporter:
    def __init__(self, out_root: Path, box_sizes: list[int], copy_images: bool):
        self.out_root = out_root
        self.box_sizes = box_sizes
        self.copy_images = copy_images
        self.records: dict[tuple[str, int], TextIO] = {}
        self.label_counts: Counter[tuple[str, int, str]] = Counter()
        self.image_counts: Counter[tuple[str, int, str]] = Counter()
        self.image_label_counts: Counter[tuple[str, int, str, str]] = Counter()
        self.shrunk_counts: Counter[tuple[str, int, str]] = Counter()
        self.dataset_paths: dict[str, dict[str, Any]] = {}

        for task in TASKS:
            for box_size in self.box_sizes:
                dataset_dir = self.dataset_dir(task, box_size)
                for split in ("train", "test"):
                    (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                    (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
                handle = (dataset_dir / "box_records.jsonl").open("w", encoding="utf-8")
                self.records[(task, box_size)] = handle

    def dataset_dir(self, task: str, box_size: int) -> Path:
        return self.out_root / "datasets" / f"{task}_box{box_size}"

    def export_frame(self, task: str, split: str, image_path: Path, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        export_name = yolo_image_export_name(samples[0], image_path)
        for box_size in self.box_sizes:
            dataset_dir = self.dataset_dir(task, box_size)
            dst = dataset_dir / "images" / split / export_name
            link_or_copy(image_path, dst, self.copy_images)
            label_path = dataset_dir / "labels" / split / f"{Path(export_name).stem}.txt"
            lines: list[str] = []
            for sample in samples:
                record = make_box_record(task, split, box_size, sample, image_path, export_name, label_path)
                lines.append(
                    f"0 {record['x_center_norm']:.{LABEL_DECIMALS}f} "
                    f"{record['y_center_norm']:.{LABEL_DECIMALS}f} "
                    f"{record['w_norm']:.{LABEL_DECIMALS}f} {record['h_norm']:.{LABEL_DECIMALS}f}"
                )
                self.records[(task, box_size)].write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
                self.label_counts[(task, box_size, split)] += 1
                if record["shrink_w_px"] > 1e-6 or record["shrink_h_px"] > 1e-6:
                    self.shrunk_counts[(task, box_size, split)] += 1
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.image_counts[(task, box_size, split)] += 1
            self.image_label_counts[(task, box_size, split, export_name)] = len(samples)

    def close(self) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for handle in self.records.values():
            handle.close()
        for task in TASKS:
            for box_size in self.box_sizes:
                dataset_dir = self.dataset_dir(task, box_size)
                data_yaml = dataset_dir / "data.yaml"
                data_yaml.write_text(
                    "\n".join(
                        [
                            f"path: {dataset_dir.resolve()}",
                            "train: images/train",
                            "val: images/test",
                            "test: images/test",
                            "names:",
                            f"  0: {task}_target",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                by_split: dict[str, Any] = {}
                for split in ("train", "test"):
                    group_sizes = [
                        count
                        for (g_task, g_box, g_split, _name), count in self.image_label_counts.items()
                        if g_task == task and g_box == box_size and g_split == split
                    ]
                    by_split[split] = {
                        "num_labels": int(self.label_counts[(task, box_size, split)]),
                        "num_images": int(self.image_counts[(task, box_size, split)]),
                        "multi_label_images": int(sum(1 for count in group_sizes if count > 1)),
                        "max_labels_per_image": int(max(group_sizes) if group_sizes else 0),
                        "shrunk_box_labels": int(self.shrunk_counts[(task, box_size, split)]),
                    }
                summary = {
                    "task": task,
                    "box_size_px": int(box_size),
                    "dataset_dir": str(dataset_dir.resolve()),
                    "data_yaml": str(data_yaml.resolve()),
                    "box_records_jsonl": str((dataset_dir / "box_records.jsonl").resolve()),
                    "box_policy": "center_preserving_shrink_at_edges",
                    "splits": by_split,
                }
                write_json(dataset_dir / "dataset_summary.json", summary)
                summaries[f"{task}_box{box_size}"] = summary
        return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a broad Aria/MPS 2D point dataset by projecting every 3D point label "
            "onto rectified RGB VRS frames sampled by RGB stream index stride."
        )
    )
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--multi-slam-root", type=Path, default=DEFAULT_MULTI_SLAM_ROOT)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--box-sizes", default=",".join(str(value) for value in BOX_SIZES))
    parser.add_argument("--rect-size", type=int, default=720)
    parser.add_argument("--rect-focal", type=float, default=380.0)
    parser.add_argument("--camera-label", default="camera-rgb")
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument(
        "--max-frames-per-phase",
        type=int,
        default=0,
        help="Debug throttle only. 0 means full stride-5 extraction.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    box_sizes = parse_box_sizes(args.box_sizes)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    manifest_root = out_root / "manifests" / "task_specific"
    manifest_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    labels = parse_labels(args.labels_path)
    labels_by_recording_task: dict[tuple[str, str], list[LabelPoint]] = defaultdict(list)
    for label in labels:
        labels_by_recording_task[(label.recording_id, label.task)].append(label)
    split_payload = make_split_payload(sorted({label.recording_id for label in labels}))

    all_manifest_path = out_root / "manifest_all.jsonl"
    all_writer = JsonlWriter(all_manifest_path)
    valid_writers: dict[tuple[str, str], JsonlWriter] = {}
    for task in TASKS:
        for split in ("train", "test", "all"):
            valid_writers[(task, split)] = JsonlWriter(manifest_root / f"{task}_{split}.jsonl")

    yolo = YoloExporter(out_root, box_sizes, bool(args.copy_images))

    counters: dict[str, Counter[Any]] = {
        "attempts_by_task": Counter(),
        "attempts_by_task_split": Counter(),
        "attempts_by_recording": Counter(),
        "status": Counter(),
        "invalid_reason": Counter(),
        "invalid_by_task_reason": Counter(),
        "valid_by_task_split": Counter(),
        "valid_by_recording_task_split": Counter(),
        "images_saved_by_task_split": Counter(),
        "frames_by_phase": Counter(),
        "frames_missing_pose_by_phase": Counter(),
        "frames_with_valid_by_phase": Counter(),
    }
    valid_label_ids: set[str] = set()
    valid_recordings: set[str] = set()
    phase_summaries: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    run_config = {
        "task_id": TASK_ID,
        "started_at_utc": utc_now(),
        "command": command_string(),
        "thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "labels_path": str(args.labels_path.resolve()),
        "out_root": str(out_root.resolve()),
        "split_root": str(args.split_root),
        "multi_slam_root": str(args.multi_slam_root),
        "stride": int(args.stride),
        "rectification": {
            "camera_label": args.camera_label,
            "rect_size": int(args.rect_size),
            "rect_focal": float(args.rect_focal),
            "method": "Project Aria calibration.distort_by_calibration, cw90 rotated pinhole view",
        },
        "box_sizes_px": box_sizes,
        "filter_policy": "no semantic/front-view/manual filtering; YOLO labels include only valid in-image projections",
        "split_policy": split_payload["policy"],
        "max_frames_per_phase": int(args.max_frames_per_phase),
    }
    write_json(out_root / "run_config.json", run_config)
    write_json(out_root / "parsed_labels.json", [label_to_dict(label) for label in labels])

    try:
        for (recording_id, task), task_labels in sorted(labels_by_recording_task.items()):
            phase = TASK_TO_PHASE[task]
            split = split_payload["recording_to_split"].get(recording_id, "train")
            phase_summary = process_recording_phase(
                args=args,
                labels=task_labels,
                recording_id=recording_id,
                task=task,
                phase=phase,
                split=split,
                all_writer=all_writer,
                valid_writers=valid_writers,
                yolo=yolo,
                counters=counters,
                valid_label_ids=valid_label_ids,
                valid_recordings=valid_recordings,
            )
            phase_summaries.append(phase_summary)
            if phase_summary.get("blocker"):
                blockers.append(phase_summary["blocker"])
            print(
                json.dumps(
                    {
                        "recording_id": recording_id,
                        "task": task,
                        "phase": phase,
                        "frames_seen": phase_summary.get("frames_seen", 0),
                        "attempts": phase_summary.get("attempts", 0),
                        "valid": phase_summary.get("valid", 0),
                        "invalid": phase_summary.get("invalid", 0),
                        "reason": phase_summary.get("skip_reason"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        all_writer.close()
        for writer in valid_writers.values():
            writer.close()
        yolo_summary = yolo.close()

    summary = build_summary(
        args=args,
        labels=labels,
        split_payload=split_payload,
        all_manifest_path=all_manifest_path,
        manifest_root=manifest_root,
        counters=counters,
        valid_label_ids=valid_label_ids,
        valid_recordings=valid_recordings,
        phase_summaries=phase_summaries,
        blockers=blockers,
        yolo_summary=yolo_summary,
        box_sizes=box_sizes,
    )
    summary_path = out_root / "summary.json"
    write_json(summary_path, summary)
    write_manifest_metadata(args, summary, manifest_root)
    write_report(args.report_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "out_root": str(out_root.resolve()),
                "summary": str(summary_path.resolve()),
                "manifest_all_jsonl": str(all_manifest_path.resolve()),
                "report": str(args.report_path.resolve()),
                "counts": summary["counts"],
                "invalid_counts": summary["invalid_counts"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def process_recording_phase(
    args: argparse.Namespace,
    labels: list[LabelPoint],
    recording_id: str,
    task: str,
    phase: str,
    split: str,
    all_writer: JsonlWriter,
    valid_writers: dict[tuple[str, str], JsonlWriter],
    yolo: YoloExporter,
    counters: dict[str, Counter[Any]],
    valid_label_ids: set[str],
    valid_recordings: set[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "recording_id": recording_id,
        "task": task,
        "phase": phase,
        "split": split,
        "labels": len(labels),
        "frames_seen": 0,
        "frames_with_pose": 0,
        "frames_missing_pose": 0,
        "frames_with_valid": 0,
        "images_saved": 0,
        "attempts": 0,
        "valid": 0,
        "invalid": 0,
    }
    assets, asset_error = resolve_phase_assets(recording_id, phase, args)
    if asset_error is not None:
        reason = asset_error["reason"]
        for label in labels:
            row = make_base_row(label, split)
            row.update(
                {
                    "status": "invalid",
                    "reason": reason,
                    "attempt_scope": "label_phase",
                    "vrs_path": asset_error.get("vrs_path"),
                    "mps_dir": asset_error.get("mps_dir"),
                    "missing_paths": asset_error.get("missing_paths", []),
                    "projectable_in_image": False,
                    "yolo_included": False,
                }
            )
            write_attempt(row, all_writer, counters)
            summary["attempts"] += 1
            summary["invalid"] += 1
        summary["skip_reason"] = reason
        summary["blocker"] = {
            "type": reason,
            "recording_id": recording_id,
            "task": task,
            "phase": phase,
            "detail": asset_error.get("detail", ""),
            "missing_paths": asset_error.get("missing_paths", []),
        }
        return summary

    assert assets is not None
    try:
        context = load_phase_context(assets, args)
    except Exception as exc:
        reason = classify_load_error(exc)
        for label in labels:
            row = make_base_row(label, split)
            row.update(
                {
                    "status": "invalid",
                    "reason": reason,
                    "attempt_scope": "label_phase",
                    "vrs_path": str(assets.vrs_path),
                    "mps_dir": str(assets.mps_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                    "projectable_in_image": False,
                    "yolo_included": False,
                }
            )
            write_attempt(row, all_writer, counters)
            summary["attempts"] += 1
            summary["invalid"] += 1
        summary["skip_reason"] = reason
        summary["blocker"] = {
            "type": reason,
            "recording_id": recording_id,
            "task": task,
            "phase": phase,
            "detail": f"{type(exc).__name__}: {exc}",
            "vrs_path": str(assets.vrs_path),
            "mps_dir": str(assets.mps_dir),
        }
        return summary

    selected_indices = list(range(0, context.n_source_frames, int(args.stride)))
    if args.max_frames_per_phase > 0:
        selected_indices = selected_indices[: int(args.max_frames_per_phase)]
    summary["asset_source"] = assets.source
    summary["vrs_path"] = str(assets.vrs_path)
    summary["mps_dir"] = str(assets.mps_dir)
    summary["n_source_frames"] = int(context.n_source_frames)
    summary["selected_indices"] = len(selected_indices)
    camera_public = public_camera(context.camera)
    image_dir = args.out_root / "images" / recording_id / phase

    for local_i, frame_idx in enumerate(selected_indices, start=1):
        summary["frames_seen"] += 1
        counters["frames_by_phase"][(task, phase)] += 1
        try:
            image_data, record = context.provider.get_image_data_by_index(context.rgb_id, int(frame_idx))
            timestamp_ns = int(image_record_timestamp_ns(record))
        except Exception as exc:
            reason = "frame_read_error"
            for label in labels:
                row = make_frame_row(label, split, assets, frame_idx, None, None, camera_public, reason)
                row["error"] = f"{type(exc).__name__}: {exc}"
                write_attempt(row, all_writer, counters)
                summary["attempts"] += 1
                summary["invalid"] += 1
            continue

        T_world_device = pose_at(context.trajectory, timestamp_ns // 1000)
        rel_s = relative_time_s(timestamp_ns, context)
        planned_image = image_dir / f"frame_{frame_idx:06d}_{timestamp_ns}.jpg"
        if T_world_device is None:
            summary["frames_missing_pose"] += 1
            counters["frames_missing_pose_by_phase"][(task, phase)] += 1
            for label in labels:
                row = make_frame_row(label, split, assets, frame_idx, timestamp_ns, rel_s, camera_public, "missing_pose")
                row["image_path"] = str(planned_image.resolve())
                write_attempt(row, all_writer, counters)
                summary["attempts"] += 1
                summary["invalid"] += 1
            continue

        summary["frames_with_pose"] += 1
        T_world_camera = np.asarray(T_world_device @ context.camera["T_device_camera"], dtype=np.float64)
        frame_valid_samples: list[dict[str, Any]] = []
        frame_attempt_rows: list[dict[str, Any]] = []
        for label in labels:
            row, sample = project_label(label, split, assets, frame_idx, timestamp_ns, rel_s, camera_public, T_world_camera)
            row["image_path"] = str(planned_image.resolve())
            frame_attempt_rows.append(row)
            if sample is not None:
                frame_valid_samples.append(sample)

        image_saved = False
        if frame_valid_samples:
            if not planned_image.exists():
                planned_image.parent.mkdir(parents=True, exist_ok=True)
                image = rectify_image(image_data, context.camera)
                image.save(planned_image, quality=int(args.jpg_quality))
            image_saved = True
            summary["frames_with_valid"] += 1
            summary["images_saved"] += 1
            counters["frames_with_valid_by_phase"][(task, phase)] += 1
            counters["images_saved_by_task_split"][(task, split)] += 1
            for sample in frame_valid_samples:
                sample["image_path"] = str(planned_image.resolve())
                sample["image_name"] = planned_image.name

        for row in frame_attempt_rows:
            row["image_saved"] = bool(image_saved)
            write_attempt(row, all_writer, counters)
            summary["attempts"] += 1
            if row["status"] == "valid":
                summary["valid"] += 1
                valid_label_ids.add(str(row["label_id"]))
                valid_recordings.add(recording_id)
            else:
                summary["invalid"] += 1

        if frame_valid_samples:
            for sample in frame_valid_samples:
                valid_writers[(task, split)].write(sample)
                valid_writers[(task, "all")].write(sample)
                counters["valid_by_task_split"][(task, split)] += 1
                counters["valid_by_recording_task_split"][(recording_id, task, split)] += 1
            yolo.export_frame(task, split, planned_image, frame_valid_samples)

        if int(args.progress_every) > 0 and local_i % int(args.progress_every) == 0:
            print(
                json.dumps(
                    {
                        "progress": f"{recording_id}/{phase}",
                        "frames": local_i,
                        "selected": len(selected_indices),
                        "valid": summary["valid"],
                        "invalid": summary["invalid"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return summary


def resolve_phase_assets(recording_id: str, phase: str, args: argparse.Namespace) -> tuple[PhaseAssets | None, dict[str, Any] | None]:
    phase_dir = args.split_root / recording_id / phase
    vrs_path = phase_dir / f"{phase}.vrs"
    phase_times_path = phase_dir / "phase_times.json"
    split_mps = phase_dir / f"mps_{phase}_vrs"
    split_traj = split_mps / "slam" / "closed_loop_trajectory.csv"
    if vrs_path.exists() and phase_times_path.exists() and split_traj.exists():
        return PhaseAssets(recording_id, phase, vrs_path, split_mps, phase_times_path, "split_phase_mps"), None

    multi_map_path = args.multi_slam_root / recording_id / "vrs_to_multi_slam.json"
    if vrs_path.exists() and phase_times_path.exists() and multi_map_path.exists():
        mapping = json.loads(multi_map_path.read_text(encoding="utf-8"))
        key = str(vrs_path.resolve())
        rel = mapping.get(key) or mapping.get(str(vrs_path))
        if rel is not None:
            multi_mps = args.multi_slam_root / recording_id / str(rel)
            if (multi_mps / "slam" / "closed_loop_trajectory.csv").exists():
                return PhaseAssets(recording_id, phase, vrs_path, multi_mps, phase_times_path, "multi_slam_split_mps"), None

    missing_paths: list[str] = []
    reason = "missing_mps"
    if not vrs_path.exists():
        missing_paths.append(str(vrs_path))
        reason = "missing_vrs"
    if not phase_times_path.exists():
        missing_paths.append(str(phase_times_path))
        if reason != "missing_vrs":
            reason = "missing_phase_times"
    if not split_traj.exists():
        missing_paths.append(str(split_traj))
    if not missing_paths and multi_map_path.exists():
        missing_paths.append(f"usable MPS mapping in {multi_map_path}")
    return None, {
        "reason": reason,
        "detail": f"No usable phase assets for {recording_id}/{phase}",
        "vrs_path": str(vrs_path),
        "mps_dir": str(split_mps),
        "missing_paths": missing_paths,
    }


def load_phase_context(assets: PhaseAssets, args: argparse.Namespace) -> PhaseContext:
    provider = data_provider.create_vrs_data_provider(str(assets.vrs_path))
    if not provider:
        raise RuntimeError(f"Could not open VRS provider: {assets.vrs_path}")
    rgb_id = provider.get_stream_id_from_label(args.camera_label)
    camera = make_rectified_camera(provider, args.camera_label, int(args.rect_size), float(args.rect_focal))
    trajectory = load_trajectory(assets.mps_dir / "slam" / "closed_loop_trajectory.csv")
    times = json.loads(assets.phase_times_path.read_text(encoding="utf-8"))
    return PhaseContext(
        assets=assets,
        provider=provider,
        rgb_id=rgb_id,
        camera=camera,
        trajectory=trajectory,
        phase_start_ns=int(times["start_device_time_ns"]) if "start_device_time_ns" in times else None,
        phase_end_ns=int(times["end_device_time_ns"]) if "end_device_time_ns" in times else None,
        n_source_frames=int(provider_num_data(provider, rgb_id)),
    )


def provider_num_data(provider: Any, stream_id: Any) -> int:
    if hasattr(provider, "get_num_data"):
        return int(provider.get_num_data(stream_id))
    return int(provider.getNumData(stream_id))


def project_label(
    label: LabelPoint,
    split: str,
    assets: PhaseAssets,
    frame_idx: int,
    timestamp_ns: int,
    relative_s: float | None,
    camera: dict[str, Any],
    T_world_camera: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row = make_frame_row(label, split, assets, frame_idx, timestamp_ns, relative_s, camera, None)
    p_cam = world_to_camera(label.point_world, T_world_camera)
    uv = [math.nan, math.nan]
    ray = [math.nan, math.nan, math.nan]
    reason = None
    if not finite_all(p_cam):
        reason = "nonfinite_camera_point"
    elif float(p_cam[2]) <= 1e-9:
        reason = "behind_camera"
    else:
        uv = [
            float(camera["fx"]) * float(p_cam[0]) / float(p_cam[2]) + float(camera["cx"]),
            float(camera["fy"]) * float(p_cam[1]) / float(p_cam[2]) + float(camera["cy"]),
        ]
        ray = normalized_ray_from_uv(uv, camera)
        if not finite_all(uv) or not finite_all(ray):
            reason = "nonfinite_projection"
        elif not (0.0 <= uv[0] < float(camera["width"]) and 0.0 <= uv[1] < float(camera["height"])):
            reason = "out_of_image"

    valid = reason is None
    row.update(
        {
            "status": "valid" if valid else "invalid",
            "reason": "valid" if valid else reason,
            "target_2d": floats_or_none(uv),
            "target_depth": float(p_cam[2]) if math.isfinite(float(p_cam[2])) else None,
            "target_3d_camera": floats_or_none(p_cam),
            "target_ray_camera_unit": floats_or_none(ray),
            "target_3d_world": list(label.point_world),
            "projectable_in_image": bool(valid),
            "yolo_included": bool(valid),
        }
    )
    if not valid:
        return row, None

    sample_id = sample_id_for(label, frame_idx, timestamp_ns)
    sample = {
        "sample_id": sample_id,
        "recording_id": label.recording_id,
        "vrs_filename": f"{label.recording_id}.vrs",
        "task": label.task,
        "phase": label.phase,
        "split": split,
        "label_id": label.label_id,
        "label_index": int(label.label_index),
        "source_label_line_number": int(label.line_number),
        "frame_idx": int(frame_idx),
        "frame_id": f"{label.recording_id}|{label.phase}|{frame_idx}|{timestamp_ns}",
        "timestamp_ns": int(timestamp_ns),
        "relative_time_s": relative_s,
        "image_name": "",
        "image_path": "",
        "camera": camera,
        "target_2d": row["target_2d"],
        "target_depth": row["target_depth"],
        "target_3d_camera": row["target_3d_camera"],
        "target_ray_camera_unit": row["target_ray_camera_unit"],
        "target_3d_world": list(label.point_world),
        "vrs_path": str(assets.vrs_path),
        "mps_dir": str(assets.mps_dir),
        "asset_source": assets.source,
        "filter_policy": "valid in-image projection only; no semantic/front-view/manual quality filtering",
    }
    return row, sample


def make_base_row(label: LabelPoint, split: str) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "recording_id": label.recording_id,
        "vrs_filename": f"{label.recording_id}.vrs",
        "task": label.task,
        "phase": label.phase,
        "split": split,
        "label_id": label.label_id,
        "label_index": int(label.label_index),
        "source_label_line_number": int(label.line_number),
        "target_3d_world": list(label.point_world),
    }


def make_frame_row(
    label: LabelPoint,
    split: str,
    assets: PhaseAssets,
    frame_idx: int,
    timestamp_ns: int | None,
    relative_s: float | None,
    camera: dict[str, Any] | None,
    reason: str | None,
) -> dict[str, Any]:
    row = make_base_row(label, split)
    row.update(
        {
            "attempt_scope": "frame_label",
            "status": "invalid" if reason else "pending",
            "reason": reason,
            "frame_idx": int(frame_idx),
            "frame_id": None if timestamp_ns is None else f"{label.recording_id}|{label.phase}|{frame_idx}|{timestamp_ns}",
            "timestamp_ns": timestamp_ns,
            "relative_time_s": relative_s,
            "image_path": None,
            "image_saved": False,
            "vrs_path": str(assets.vrs_path),
            "mps_dir": str(assets.mps_dir),
            "asset_source": assets.source,
            "camera": camera,
            "target_2d": None,
            "target_depth": None,
            "target_3d_camera": None,
            "target_ray_camera_unit": None,
            "projectable_in_image": False,
            "yolo_included": False,
        }
    )
    return row


def write_attempt(row: dict[str, Any], all_writer: JsonlWriter, counters: dict[str, Counter[Any]]) -> None:
    all_writer.write(row)
    task = str(row.get("task", ""))
    split = str(row.get("split", ""))
    recording = str(row.get("recording_id", ""))
    status = str(row.get("status", "invalid"))
    reason = str(row.get("reason") or status)
    counters["attempts_by_task"][task] += 1
    counters["attempts_by_task_split"][(task, split)] += 1
    counters["attempts_by_recording"][recording] += 1
    counters["status"][status] += 1
    if status != "valid":
        counters["invalid_reason"][reason] += 1
        counters["invalid_by_task_reason"][(task, reason)] += 1


def world_to_camera(point_world: tuple[float, float, float], T_world_camera: np.ndarray) -> list[float]:
    rotation = T_world_camera[:3, :3]
    translation = T_world_camera[:3, 3]
    delta = np.asarray(point_world, dtype=np.float64) - translation
    return (rotation.T @ delta).astype(float).tolist()


def normalized_ray_from_uv(uv: list[float], camera: dict[str, Any]) -> list[float]:
    fx = max(abs(float(camera["fx"])), 1e-12)
    fy = max(abs(float(camera["fy"])), 1e-12)
    ray = [
        (float(uv[0]) - float(camera["cx"])) / fx,
        (float(uv[1]) - float(camera["cy"])) / fy,
        1.0,
    ]
    norm = math.sqrt(sum(value * value for value in ray))
    if not math.isfinite(norm) or norm <= 1e-12:
        return [math.nan, math.nan, math.nan]
    return [value / norm for value in ray]


def public_camera(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "rectified_pinhole",
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
    }


def make_box_record(
    task: str,
    split: str,
    box_size: int,
    sample: dict[str, Any],
    image_path: Path,
    export_image_name: str,
    label_path: Path,
) -> dict[str, Any]:
    camera = sample["camera"]
    width = int(camera["width"])
    height = int(camera["height"])
    target_x = float(sample["target_2d"][0])
    target_y = float(sample["target_2d"][1])
    half = float(box_size) / 2.0
    half_w = min(half, max(target_x, 0.0), max(float(width) - target_x, 0.0))
    half_h = min(half, max(target_y, 0.0), max(float(height) - target_y, 0.0))
    if half_w <= 0.0 or half_h <= 0.0:
        raise ValueError(f"Degenerate center-preserving box for {sample.get('sample_id')}")
    x0 = target_x - half_w
    x1 = target_x + half_w
    y0 = target_y - half_h
    y1 = target_y + half_h
    actual_w = x1 - x0
    actual_h = y1 - y0
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    serialized_x = float(f"{center_x / float(width):.{LABEL_DECIMALS}f}") * float(width)
    serialized_y = float(f"{center_y / float(height):.{LABEL_DECIMALS}f}") * float(height)
    return {
        "task": task,
        "split": split,
        "box_size_px": int(box_size),
        "sample_id": str(sample["sample_id"]),
        "recording_id": str(sample["recording_id"]),
        "label_id": str(sample["label_id"]),
        "frame_idx": int(sample["frame_idx"]),
        "timestamp_ns": int(sample["timestamp_ns"]),
        "image_path": str(image_path.resolve()),
        "export_image_name": export_image_name,
        "label_path": str(label_path.resolve()),
        "target_x": target_x,
        "target_y": target_y,
        "target_depth_m": float(sample["target_depth"]),
        "width": width,
        "height": height,
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "x_center_norm": center_x / float(width),
        "y_center_norm": center_y / float(height),
        "w_norm": actual_w / float(width),
        "h_norm": actual_h / float(height),
        "box_x0": x0,
        "box_y0": y0,
        "box_x1": x1,
        "box_y1": y1,
        "actual_w_px": actual_w,
        "actual_h_px": actual_h,
        "center_error_px": math.hypot(center_x - target_x, center_y - target_y),
        "serialized_center_error_px": math.hypot(serialized_x - target_x, serialized_y - target_y),
        "shrink_w_px": float(box_size) - actual_w,
        "shrink_h_px": float(box_size) - actual_h,
    }


def yolo_image_export_name(sample: dict[str, Any], image_path: Path) -> str:
    digest = hashlib.sha1(
        f"{sample['recording_id']}|{sample['phase']}|{sample['frame_idx']}|{sample['timestamp_ns']}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{sample['recording_id']}_{sample['phase']}_f{int(sample['frame_idx']):06d}_{digest}{image_path.suffix or '.jpg'}"


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_labels(path: Path) -> list[LabelPoint]:
    labels: list[LabelPoint] = []
    current_recording: tuple[int, str] | None = None
    current_task: str | None = None
    label_counts: Counter[tuple[str, str]] = Counter()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = RECORDING_LINE_RE.match(line)
        if match:
            current_recording = (int(match.group("ordinal")), match.group("recording").lower())
            current_task = None
            continue
        lowered = line.lower()
        if lowered in TASKS:
            if current_recording is None:
                raise ValueError(f"Task before recording at {path}:{line_number}")
            current_task = lowered
            continue
        if line.startswith("["):
            if current_recording is None or current_task is None:
                raise ValueError(f"Point before recording/task at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"Expected [x,y,z] at {path}:{line_number}: {line}")
            point = tuple(float(v) for v in value)
            if not finite_all(point):
                raise ValueError(f"Non-finite point at {path}:{line_number}: {line}")
            order, recording_id = current_recording
            label_index = int(label_counts[(recording_id, current_task)])
            label_counts[(recording_id, current_task)] += 1
            labels.append(LabelPoint(order, recording_id, current_task, label_index, point, line_number))
            continue
        raise ValueError(f"Unrecognized line at {path}:{line_number}: {line}")
    if not labels:
        raise ValueError(f"No labels parsed from {path}")
    return labels


def make_split_payload(recordings: list[str]) -> dict[str, Any]:
    test = sorted(recording for recording in recordings if recording in PRIOR_TEST_RECORDINGS)
    train = sorted(recording for recording in recordings if recording not in set(test))
    return {
        "policy": "preserve prior four-recording test set from repo scripts; all other labeled recordings train; no validation split",
        "train": train,
        "test": test,
        "recording_to_split": {**{recording: "train" for recording in train}, **{recording: "test" for recording in test}},
    }


def build_summary(
    args: argparse.Namespace,
    labels: list[LabelPoint],
    split_payload: dict[str, Any],
    all_manifest_path: Path,
    manifest_root: Path,
    counters: dict[str, Counter[Any]],
    valid_label_ids: set[str],
    valid_recordings: set[str],
    phase_summaries: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    yolo_summary: dict[str, Any],
    box_sizes: list[int],
) -> dict[str, Any]:
    input_by_task = Counter(label.task for label in labels)
    input_by_recording = Counter(label.recording_id for label in labels)
    valid_by_task_split = nested_counter(counters["valid_by_task_split"], 2)
    attempts_total = int(sum(counters["status"].values()))
    valid_total = int(counters["status"].get("valid", 0))
    invalid_total = attempts_total - valid_total
    task_manifests = {
        task: {
            split: str((manifest_root / f"{task}_{split}.jsonl").resolve())
            for split in ("train", "test", "all")
        }
        for task in TASKS
    }
    return {
        "task_id": TASK_ID,
        "status": "complete_with_blockers" if blockers else "complete",
        "created_at_utc": utc_now(),
        "command": command_string(),
        "labels_path": str(args.labels_path.resolve()),
        "output_root": str(args.out_root.resolve()),
        "stride": int(args.stride),
        "rectification": {
            "camera_label": args.camera_label,
            "rect_size": int(args.rect_size),
            "rect_focal": float(args.rect_focal),
            "image_space": "rectified_pinhole_rgb",
        },
        "split": split_payload,
        "input_labels": {
            "labels": len(labels),
            "recordings": len(input_by_recording),
            "by_task": dict(sorted(input_by_task.items())),
            "by_recording": dict(sorted(input_by_recording.items())),
        },
        "counts": {
            "attempted_label_frame_pairs": attempts_total,
            "valid_samples": valid_total,
            "invalid_samples": invalid_total,
            "valid_labels": len(valid_label_ids),
            "valid_recordings": len(valid_recordings),
            "by_status": dict(sorted((str(k), int(v)) for k, v in counters["status"].items())),
            "attempts_by_task": dict(sorted((str(k), int(v)) for k, v in counters["attempts_by_task"].items())),
            "attempts_by_recording": dict(sorted((str(k), int(v)) for k, v in counters["attempts_by_recording"].items())),
            "valid_by_task_split": valid_by_task_split,
            "valid_by_recording_task_split": nested_counter(counters["valid_by_recording_task_split"], 3),
            "images_saved_by_task_split": nested_counter(counters["images_saved_by_task_split"], 2),
        },
        "invalid_counts": {
            "by_reason": dict(sorted((str(k), int(v)) for k, v in counters["invalid_reason"].items())),
            "by_task_reason": nested_counter(counters["invalid_by_task_reason"], 2),
        },
        "frames": {
            "by_task_phase": nested_counter(counters["frames_by_phase"], 2),
            "missing_pose_by_task_phase": nested_counter(counters["frames_missing_pose_by_phase"], 2),
            "with_valid_by_task_phase": nested_counter(counters["frames_with_valid_by_phase"], 2),
            "phase_summaries": phase_summaries,
        },
        "artifacts": {
            "manifest_all_jsonl": str(all_manifest_path.resolve()),
            "parsed_labels_json": str((args.out_root / "parsed_labels.json").resolve()),
            "task_specific_manifests": task_manifests,
            "images_root": str((args.out_root / "images").resolve()),
            "datasets_root": str((args.out_root / "datasets").resolve()),
            "yolo_datasets": yolo_summary,
            "summary_json": str((args.out_root / "summary.json").resolve()),
            "run_config_json": str((args.out_root / "run_config.json").resolve()),
            "report_path": str(args.report_path.resolve()),
        },
        "yolo": {
            "box_sizes_px": box_sizes,
            "label_policy": "include only valid in-image projections; one class per task-specific dataset",
            "datasets": yolo_summary,
        },
        "blockers": blockers,
    }


def write_manifest_metadata(args: argparse.Namespace, summary: dict[str, Any], manifest_root: Path) -> None:
    for task in TASKS:
        for split in ("train", "test", "all"):
            path = manifest_root / f"{task}_{split}.json"
            payload = {
                "version": 1,
                "type": "aria_mps_2d_stride5_task_manifest_jsonl",
                "task_id": TASK_ID,
                "task": task,
                "split": split,
                "created_at_utc": summary["created_at_utc"],
                "samples_jsonl": str((manifest_root / f"{task}_{split}.jsonl").resolve()),
                "coordinate_frame": "rectified_vrs_rgb_camera",
                "image_space": "rectified_pinhole_rgb",
                "labels_path": str(args.labels_path.resolve()),
                "filter_policy": "valid in-image projection only; no semantic/front-view/manual quality filtering",
                "split_policy": summary["split"]["policy"],
                "required_fields": ["image_path", "target_2d", "target_depth", "camera", "split", "task"],
            }
            write_json(path, payload)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    invalid = summary["invalid_counts"]
    lines = [
        "# Aria MPS 2D Stride-5 Dataset 033",
        "",
        f"Worker role: `{TASK_ID}`",
        f"Status: `{summary['status']}`",
        f"Output root: `{summary['output_root']}`",
        "",
        "## Command",
        "",
        "```bash",
        summary["command"],
        "```",
        "",
        "## Protocol",
        "",
        "- Labels are parsed from `3d_points_labels.txt`.",
        "- RGB frames are sampled by VRS RGB stream index with stride 5.",
        "- Images saved for training are rectified pinhole RGB views.",
        "- No semantic, front-view, manual, or quality filtering is applied.",
        "- `manifest_all.jsonl` logs every attempted label/frame projection, including invalid reasons.",
        "- YOLO datasets include only valid in-image projections.",
        "- Train/test split preserves the prior four-recording test set from repo scripts; no validation split is assigned.",
        "",
        "## Counts",
        "",
        f"- Input labels: `{summary['input_labels']['labels']}` across `{summary['input_labels']['recordings']}` recordings",
        f"- Attempted label/frame pairs: `{counts['attempted_label_frame_pairs']}`",
        f"- Valid samples: `{counts['valid_samples']}`",
        f"- Invalid samples: `{counts['invalid_samples']}`",
        f"- Valid labels represented: `{counts['valid_labels']}`",
        f"- Valid recordings represented: `{counts['valid_recordings']}`",
        "",
        "| Task | Train valid | Test valid | Total valid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        task_counts = counts["valid_by_task_split"].get(task, {})
        train = int(task_counts.get("train", 0))
        test = int(task_counts.get("test", 0))
        lines.append(f"| {task} | {train} | {test} | {train + test} |")
    lines.extend(["", "## Invalid Counts", "", "| Reason | Count |", "| --- | ---: |"])
    for reason, count in invalid["by_reason"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## YOLO Datasets", "", "| Dataset | data.yaml | Train labels/images | Test labels/images |", "| --- | --- | ---: | ---: |"])
    for name, item in summary["yolo"]["datasets"].items():
        train = item["splits"]["train"]
        test = item["splits"]["test"]
        lines.append(
            f"| {name} | `{item['data_yaml']}` | {train['num_labels']}/{train['num_images']} | {test['num_labels']}/{test['num_images']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest all JSONL: `{summary['artifacts']['manifest_all_jsonl']}`",
            f"- Task manifests root: `{Path(summary['artifacts']['task_specific_manifests']['deleafing']['all']).parent}`",
            f"- Rectified images root: `{summary['artifacts']['images_root']}`",
            f"- YOLO datasets root: `{summary['artifacts']['datasets_root']}`",
            f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
            "",
            "## Blockers",
            "",
        ]
    )
    if summary["blockers"]:
        for blocker in summary["blockers"]:
            lines.append(
                f"- `{blocker['type']}` {blocker['recording_id']} / {blocker['phase']}: {blocker.get('detail', '')}"
            )
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_box_sizes(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("No box sizes provided")
    if any(value <= 0 for value in values):
        raise ValueError("Box sizes must be positive")
    return values


def label_to_dict(label: LabelPoint) -> dict[str, Any]:
    return {
        "recording_order": int(label.recording_order),
        "recording_id": label.recording_id,
        "vrs_filename": f"{label.recording_id}.vrs",
        "task": label.task,
        "phase": label.phase,
        "label_index": int(label.label_index),
        "label_id": label.label_id,
        "target_3d_world": list(label.point_world),
        "source_line_number": int(label.line_number),
    }


def relative_time_s(timestamp_ns: int, context: PhaseContext) -> float | None:
    if context.phase_start_ns is None:
        return None
    return float((int(timestamp_ns) - int(context.phase_start_ns)) / 1e9)


def sample_id_for(label: LabelPoint, frame_idx: int, timestamp_ns: int) -> str:
    digest = hashlib.sha1(
        f"{label.recording_id}|{label.task}|{label.label_index}|{frame_idx}|{timestamp_ns}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{label.recording_id}_{label.task}_l{label.label_index:03d}_f{int(frame_idx):06d}_{digest}"


def finite_all(values: Any) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def floats_or_none(values: Any) -> list[float] | None:
    try:
        out = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in out):
        return None
    return out


def nested_counter(counter: Counter[Any], width: int) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for key, value in sorted(counter.items()):
        if width == 1:
            root[str(key)] = int(value)
            continue
        if not isinstance(key, tuple) or len(key) != width:
            root[str(key)] = int(value)
            continue
        cursor = root
        for part in key[:-1]:
            cursor = cursor.setdefault(str(part), {})
        cursor[str(key[-1])] = int(value)
    return root


def classify_load_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "vrs" in text or "provider" in text:
        return "vrs_open_failed"
    if "trajectory" in text or "closed_loop" in text:
        return "missing_mps"
    return "phase_load_error"


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
