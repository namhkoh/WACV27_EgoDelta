#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.manifest import write_json  # noqa: E402


TASK_ID = "manual_front_yolo_dataset_032"
TASKS = ("deleafing", "harvesting")
SPLITS = ("train", "test")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
DEFAULT_LABELS = DEFAULT_FRONT_ROOT / "labels.jsonl"
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/manual_front_yolo_032")
DEFAULT_REPORT = REPO_ROOT / "research" / "REPORTS" / "manual_front_yolo_dataset_032.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task-specific YOLO/evaluation manifests from manually accepted front-view 2D labels."
    )
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--accepted-status", default="accepted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = args.out_root / "manifests"
    task_specific_root = manifest_root / "task_specific"
    task_specific_root.mkdir(parents=True, exist_ok=True)

    source_rows = load_front_manifests(args.front_root)
    labels = load_labels(args.labels_jsonl, args.front_root)
    source_cache: dict[Path, dict[str, Any]] = {}
    by_task_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected_counts: Counter[tuple[str, str]] = Counter()
    missing_rows: list[dict[str, Any]] = []
    missing_source_samples: list[dict[str, Any]] = []

    for idx, label in enumerate(labels, start=1):
        task = str(label.get("task", ""))
        status = str(label.get("status", ""))
        if task not in TASKS:
            continue
        if status != str(args.accepted_status):
            rejected_counts[(task, status)] += 1
            continue
        rel_image = str(label["image"])
        front_row = source_rows.get(rel_image)
        if front_row is None:
            missing_rows.append({"label_index": idx, "image": rel_image, "task": task})
            continue
        split = str(front_row["split"])
        if split not in SPLITS:
            missing_rows.append({"label_index": idx, "image": rel_image, "task": task, "split": split})
            continue
        source_manifest_path = Path(front_row["source_manifest"])
        source_manifest = source_cache.setdefault(source_manifest_path, read_json(source_manifest_path))
        source_sample = find_source_sample(source_manifest, front_row)
        if source_sample is None:
            missing_source_samples.append(
                {
                    "label_index": idx,
                    "image": rel_image,
                    "task": task,
                    "source_image_path": front_row.get("source_image_path"),
                    "source_manifest": str(source_manifest_path),
                }
            )
            continue
        sample = make_sample(idx, label, front_row, source_sample, args.front_root)
        by_task_split[(task, split)].append(sample)

    if missing_rows or missing_source_samples:
        write_json(
            args.out_root / "join_failures.json",
            {"missing_manifest_rows": missing_rows, "missing_source_samples": missing_source_samples},
        )
        raise RuntimeError(
            f"Manual label join incomplete: missing_rows={len(missing_rows)} "
            f"missing_source_samples={len(missing_source_samples)}"
        )

    manifest_paths: dict[str, dict[str, str]] = {task: {} for task in TASKS}
    counts: dict[str, dict[str, int]] = {task: {} for task in TASKS}
    recording_counts: dict[str, dict[str, int]] = {task: {} for task in TASKS}
    for task in TASKS:
        all_samples: list[dict[str, Any]] = []
        for split in SPLITS:
            samples = sorted(by_task_split[(task, split)], key=sample_sort_key)
            all_samples.extend(samples)
            manifest = make_manifest(task, split, samples, args, source_rows)
            path = manifest_root / f"{task}_{split}.json"
            task_specific_path = task_specific_root / f"{task}_{split}.json"
            write_json(path, manifest)
            write_json(task_specific_path, manifest)
            manifest_paths[task][split] = str(task_specific_path.resolve())
            counts[task][split] = len(samples)
            recording_counts[task][split] = len({sample.get("recording_id") for sample in samples})
        all_manifest = make_manifest(task, "all", all_samples, args, source_rows)
        write_json(manifest_root / f"{task}_all.json", all_manifest)
        write_json(task_specific_root / f"{task}_all.json", all_manifest)
        manifest_paths[task]["all"] = str((task_specific_root / f"{task}_all.json").resolve())
        counts[task]["all"] = len(all_samples)
        recording_counts[task]["all"] = len({sample.get("recording_id") for sample in all_samples})

    summary = {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "front_root": str(args.front_root.resolve()),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "out_root": str(args.out_root.resolve()),
        "manifest_root": str(manifest_root.resolve()),
        "task_specific_manifest_root": str(task_specific_root.resolve()),
        "manifest_paths": manifest_paths,
        "accepted_counts": counts,
        "unique_recording_counts": recording_counts,
        "rejected_or_ignored_label_counts": {f"{task}:{status}": count for (task, status), count in sorted(rejected_counts.items())},
        "protocol": {
            "model_input": "rectified RGB image only",
            "training_target": "fixed-size synthetic YOLO bbox centered on manual 2D label",
            "evaluation_target_2d": "manual 2D label from labels.jsonl",
            "evaluation_depth": "source VRS manifest target_depth reused only to convert 2D error to oracle-depth 3D error",
            "validation_set": "none; original manifest split is preserved as train/test only",
        },
    }
    write_json(args.out_root / "summary.json", summary)
    write_report(args.report_path, summary)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve())}, indent=2))
    return 0


def load_front_manifests(front_root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for task in TASKS:
        path = front_root / f"{task}_manifest.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in csv.DictReader(path.open("r", encoding="utf-8")):
            rel = str(Path(row["task"]) / Path(row["output_path"]).name)
            rows[rel] = row
    return rows


def load_labels(path: Path, front_root: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    labels = []
    for line_no, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rel = str(row.get("image", ""))
        image_path = front_root / rel
        if not image_path.exists():
            raise FileNotFoundError(f"Line {line_no} image does not exist: {image_path}")
        row["_line_no"] = line_no
        labels.append(row)
    return labels


def find_source_sample(manifest: dict[str, Any], front_row: dict[str, str]) -> dict[str, Any] | None:
    source_image = str(front_row.get("source_image_path", ""))
    timestamp = str(front_row.get("timestamp_ns", ""))
    recording = str(front_row.get("recording_id", ""))
    for sample in manifest.get("samples", []):
        if str(sample.get("image_path", "")) == source_image:
            return sample
    for sample in manifest.get("samples", []):
        if str(sample.get("recording_id", "")) == recording and str(sample.get("timestamp_ns", "")) == timestamp:
            return sample
    return None


def make_sample(idx: int, label: dict[str, Any], front_row: dict[str, str], source: dict[str, Any], front_root: Path) -> dict[str, Any]:
    task = str(label["task"])
    split = str(front_row["split"])
    x = float(label["x"])
    y = float(label["y"])
    image_path = front_root / str(label["image"])
    camera = dict(source["camera"])
    depth = float(source["target_depth"])
    target_3d = backproject([x, y], depth, camera)
    sample_hash = hashlib.sha1(f"{task}|{split}|{image_path}|{x:.3f}|{y:.3f}".encode("utf-8")).hexdigest()[:12]
    sample = {
        "camera": camera,
        "frame_id": source.get("frame_id"),
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "label_id": f"manual_front_{idx:04d}_{sample_hash}",
        "label_index": idx,
        "manual_label_line_no": int(label.get("_line_no", idx)),
        "manual_label_status": str(label.get("status", "")),
        "manual_label_updated_at_utc": label.get("updated_at_utc"),
        "phase": front_row.get("phase") or source.get("phase"),
        "recording_id": front_row.get("recording_id") or source.get("recording_id"),
        "relative_time_s": float(front_row["relative_time_s"]) if front_row.get("relative_time_s") else source.get("relative_time_s"),
        "sample_id": f"manual_front_{task}_{idx:04d}_{sample_hash}",
        "scene_id": source.get("scene_id"),
        "source_frame_manifest": source.get("source_frame_manifest"),
        "source_frame_sample_id": source.get("source_frame_sample_id"),
        "source_image_path": front_row.get("source_image_path"),
        "source_manifest": front_row.get("source_manifest"),
        "source_sample_id": source.get("sample_id"),
        "split": split,
        "target_2d": [x, y],
        "target_3d_camera": target_3d,
        "target_depth": depth,
        "target_ray_camera_unit": pixel_ray_unit([x, y], camera),
        "task": task,
        "timestamp_ns": int(front_row["timestamp_ns"]) if front_row.get("timestamp_ns") else source.get("timestamp_ns"),
        "vrs_filename": source.get("vrs_filename"),
    }
    width = int(camera.get("width", label.get("image_width", 0)))
    height = int(camera.get("height", label.get("image_height", 0)))
    if not (0.0 <= x < float(width) and 0.0 <= y < float(height)):
        raise ValueError(f"Manual label outside image: {sample['sample_id']} target={[x, y]} size={(width, height)}")
    return sample


def make_manifest(task: str, split: str, samples: list[dict[str, Any]], args: argparse.Namespace, source_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "version": 1,
        "task_id": TASK_ID,
        "task": task,
        "split": split,
        "created_at_utc": utc_now(),
        "num_samples": len(samples),
        "samples": samples,
        "source_labels_jsonl": str(args.labels_jsonl.resolve()),
        "source_front_root": str(args.front_root.resolve()),
        "num_front_manifest_rows": len(source_rows),
        "coordinate_frame": "rectified_vrs_rgb_camera",
        "required_fields": ["image_path", "target_2d", "target_depth", "camera", "split", "task"],
        "filter_policy": {
            "accepted_status_only": str(args.accepted_status),
            "front_view": "provided by manually labeled /tmp/egodelta/front dataset",
            "validation_set": "none",
        },
    }


def backproject(point_xy: list[float], depth: float, camera: dict[str, Any]) -> list[float]:
    fx = max(float(camera["fx"]), 1e-9)
    fy = max(float(camera["fy"]), 1e-9)
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    z = float(depth)
    return [(float(point_xy[0]) - cx) * z / fx, (float(point_xy[1]) - cy) * z / fy, z]


def pixel_ray_unit(point_xy: list[float], camera: dict[str, Any]) -> list[float]:
    ray = [
        (float(point_xy[0]) - float(camera["cx"])) / max(float(camera["fx"]), 1e-9),
        (float(point_xy[1]) - float(camera["cy"])) / max(float(camera["fy"]), 1e-9),
        1.0,
    ]
    norm = max(math.sqrt(sum(v * v for v in ray)), 1e-12)
    return [v / norm for v in ray]


def sample_sort_key(sample: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(sample.get("recording_id", "")),
        int(sample.get("label_index", -1)),
        int(sample.get("timestamp_ns", -1)),
        str(sample.get("sample_id", "")),
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Manual Front YOLO Dataset 032",
        "",
        f"Date: {summary['created_at_utc']}",
        "Role: worker/manual_front_yolo_data_032",
        f"Output root: `{summary['out_root']}`",
        "",
        "## Counts",
        "",
        "| Task | Train | Test | All | Train recordings | Test recordings |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        counts = summary["accepted_counts"][task]
        recs = summary["unique_recording_counts"][task]
        lines.append(
            f"| {task} | {counts['train']} | {counts['test']} | {counts['all']} | {recs['train']} | {recs['test']} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Input to the trained model is RGB only.",
            "- Manual accepted 2D points are converted to synthetic YOLO boxes at training time.",
            "- Source VRS target depth is retained only for oracle-depth conversion from 2D error to meters.",
            "- There is no validation split; only train/test is exported.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
