#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.manifest import read_json, write_json  # noqa: E402


TASK_ID = "yolo_bbox_center_dataset_eval_026"
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/yolo_bbox_center_dataset_eval_026")
DEFAULT_REPORT = REPO_ROOT / "research" / "REPORTS" / "yolo_bbox_center_dataset_eval_026.md"
REFINED_DELEAF_ROOT = Path("/tmp/egodelta_1cm/vrs_deleaf_visible_cutpoint_refined_024/manifests")
ORIGINAL_DELEAF_ROOT = Path("/tmp/egodelta_1cm/vrs_all_labels_camera_frame_019/manifests/task_specific")
HARVEST_ROOT = Path("/tmp/egodelta_1cm/vrs_harvest_frontview_cutpoint_refined_021/manifests")
BOX_SIZES = (32, 48, 64, 96)
SPLITS = ("train", "test")
TASKS = ("deleafing", "harvesting")
LABEL_DECIMALS = 8


@dataclass(frozen=True)
class BoxRecord:
    task: str
    split: str
    box_size_px: int
    sample_id: str
    image_path: str
    export_image_name: str
    label_path: str
    target_x: float
    target_y: float
    target_depth_m: float
    width: int
    height: int
    fx: float
    fy: float
    x_center_norm: float
    y_center_norm: float
    w_norm: float
    h_norm: float
    box_x0: float
    box_y0: float
    box_x1: float
    box_y1: float
    actual_w_px: float
    actual_h_px: float
    center_error_px: float
    serialized_center_error_px: float
    naive_clip_center_error_px: float
    shrink_w_px: float
    shrink_h_px: float
    px_budget_2cm: float
    m_per_px_conservative: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit synthetic GT-centered YOLO bbox datasets.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--deleafing-root", type=Path, default=None)
    parser.add_argument("--harvesting-root", type=Path, default=HARVEST_ROOT)
    parser.add_argument("--box-sizes", default=",".join(str(v) for v in BOX_SIZES))
    parser.add_argument("--overlay-count", type=int, default=24)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--contact-cols", type=int, default=4)
    parser.add_argument("--contact-thumb", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = datetime.now().isoformat(timespec="seconds")
    box_sizes = parse_ints(args.box_sizes)
    deleaf_root = choose_deleaf_root(args.deleafing_root)
    manifests = {
        ("deleafing", "train"): deleaf_root / "deleafing_train.json",
        ("deleafing", "test"): deleaf_root / "deleafing_test.json",
        ("harvesting", "train"): args.harvesting_root / "harvesting_train.json",
        ("harvesting", "test"): args.harvesting_root / "harvesting_test.json",
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    samples_by_key = load_all_samples(manifests)
    write_json(
        args.out_root / "run_config.json",
        {
            "task_id": TASK_ID,
            "started_at": started,
            "out_root": str(args.out_root.resolve()),
            "report_path": str(args.report_path.resolve()),
            "deleafing_root": str(deleaf_root.resolve()),
            "harvesting_root": str(args.harvesting_root.resolve()),
            "box_sizes_px": box_sizes,
            "label_decimals": LABEL_DECIMALS,
            "box_policy": "center_preserving_shrink_at_edges",
            "copy_images": bool(args.copy_images),
        },
    )

    all_box_records: list[BoxRecord] = []
    dataset_summaries: dict[str, Any] = {}
    overlay_paths: list[str] = []
    for task in TASKS:
        for box_size in box_sizes:
            records, summary = export_task_box_dataset(
                task=task,
                box_size=int(box_size),
                samples_by_key=samples_by_key,
                out_root=args.out_root,
                copy_images=bool(args.copy_images),
            )
            all_box_records.extend(records)
            dataset_summaries[f"{task}_box{box_size}"] = summary
            overlay_paths.extend(
                write_overlays(
                    task=task,
                    box_size=int(box_size),
                    samples_by_key=samples_by_key,
                    records=records,
                    out_root=args.out_root,
                    overlay_count=int(args.overlay_count),
                    cols=int(args.contact_cols),
                    thumb=int(args.contact_thumb),
                )
            )

    write_box_records(args.out_root / "box_records.csv", all_box_records)
    metrics = {
        "sample_counts": summarize_sample_counts(samples_by_key),
        "metric_conversion": summarize_metric_conversion(samples_by_key),
        "datasets": dataset_summaries,
        "oracle_and_label_audit": summarize_box_records(all_box_records),
        "overlays": overlay_paths,
        "recommendations": build_recommendations(samples_by_key, all_box_records),
    }
    write_json(args.out_root / "summary.json", metrics)
    write_report(args.report_path, args.out_root, metrics)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve()), "report": str(args.report_path.resolve())}, indent=2))
    return 0


def choose_deleaf_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    if (REFINED_DELEAF_ROOT / "deleafing_train.json").exists() and (REFINED_DELEAF_ROOT / "deleafing_test.json").exists():
        return REFINED_DELEAF_ROOT
    return ORIGINAL_DELEAF_ROOT


def parse_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("No box sizes provided")
    return values


def load_all_samples(manifests: dict[tuple[str, str], Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, path in manifests.items():
        task, split = key
        manifest = read_json(path)
        samples = []
        for sample in manifest.get("samples", []):
            if str(sample.get("task") or sample.get("task_type") or manifest.get("task")) != task:
                continue
            if str(sample.get("split") or manifest.get("split")) != split:
                continue
            if not valid_sample(sample):
                continue
            samples.append(sample)
        samples.sort(key=sample_sort_key)
        if not samples:
            raise RuntimeError(f"No valid samples for {task}/{split}: {path}")
        out[key] = samples
    return out


def valid_sample(sample: dict[str, Any]) -> bool:
    if not sample.get("image_path") or not Path(sample["image_path"]).exists():
        return False
    if sample.get("target_2d") is None or sample.get("target_depth") is None:
        return False
    camera = sample.get("camera") or {}
    return all(k in camera for k in ("fx", "fy", "width", "height"))


def sample_sort_key(sample: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(sample.get("recording_id", "")),
        safe_int(sample.get("label_index"), -1),
        safe_int(sample.get("timestamp_ns"), -1),
        str(sample.get("sample_id", "")),
    )


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def export_task_box_dataset(
    task: str,
    box_size: int,
    samples_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    out_root: Path,
    copy_images: bool,
) -> tuple[list[BoxRecord], dict[str, Any]]:
    dataset_dir = out_root / "datasets" / f"{task}_box{box_size}"
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    records: list[BoxRecord] = []
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], BoxRecord]]] = defaultdict(list)
    for split in SPLITS:
        for sample in samples_by_key[(task, split)]:
            record = make_box_record(task, split, box_size, sample, dataset_dir)
            records.append(record)
            grouped[(split, record.export_image_name)].append((sample, record))

    for (split, export_name), items in grouped.items():
        src = Path(items[0][0]["image_path"])
        dst = dataset_dir / "images" / split / export_name
        link_or_copy(src, dst, copy_images)
        label_path = dataset_dir / "labels" / split / f"{Path(export_name).stem}.txt"
        lines = [
            f"0 {r.x_center_norm:.{LABEL_DECIMALS}f} {r.y_center_norm:.{LABEL_DECIMALS}f} {r.w_norm:.{LABEL_DECIMALS}f} {r.h_norm:.{LABEL_DECIMALS}f}"
            for _, r in items
        ]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    write_records_jsonl(dataset_dir / "box_records.jsonl", records)
    summary = dataset_summary(task, box_size, dataset_dir, records, grouped)
    write_json(dataset_dir / "dataset_summary.json", summary)
    return records, summary


def make_box_record(task: str, split: str, box_size: int, sample: dict[str, Any], dataset_dir: Path) -> BoxRecord:
    camera = sample["camera"]
    width = int(camera.get("width") or sample.get("width"))
    height = int(camera.get("height") or sample.get("height"))
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
    center_error = math.hypot(center_x - target_x, center_y - target_y)

    x_center_norm = center_x / float(width)
    y_center_norm = center_y / float(height)
    w_norm = actual_w / float(width)
    h_norm = actual_h / float(height)
    serialized_x = float(f"{x_center_norm:.{LABEL_DECIMALS}f}") * float(width)
    serialized_y = float(f"{y_center_norm:.{LABEL_DECIMALS}f}") * float(height)
    serialized_error = math.hypot(serialized_x - target_x, serialized_y - target_y)
    naive_x0 = min(max(target_x - half, 0.0), float(width))
    naive_x1 = min(max(target_x + half, 0.0), float(width))
    naive_y0 = min(max(target_y - half, 0.0), float(height))
    naive_y1 = min(max(target_y + half, 0.0), float(height))
    naive_center_error = math.hypot(((naive_x0 + naive_x1) / 2.0) - target_x, ((naive_y0 + naive_y1) / 2.0) - target_y)

    fx = float(camera["fx"])
    fy = float(camera["fy"])
    depth = float(sample["target_depth"])
    f_conservative = min(max(fx, 1e-9), max(fy, 1e-9))
    m_per_px = depth / f_conservative
    px_budget_2cm = 0.020 / max(m_per_px, 1e-12)
    export_name = image_export_name(sample)
    label_path = dataset_dir / "labels" / split / f"{Path(export_name).stem}.txt"
    return BoxRecord(
        task=task,
        split=split,
        box_size_px=int(box_size),
        sample_id=str(sample.get("sample_id")),
        image_path=str(sample.get("image_path")),
        export_image_name=export_name,
        label_path=str(label_path.resolve()),
        target_x=target_x,
        target_y=target_y,
        target_depth_m=depth,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        x_center_norm=x_center_norm,
        y_center_norm=y_center_norm,
        w_norm=w_norm,
        h_norm=h_norm,
        box_x0=x0,
        box_y0=y0,
        box_x1=x1,
        box_y1=y1,
        actual_w_px=actual_w,
        actual_h_px=actual_h,
        center_error_px=center_error,
        serialized_center_error_px=serialized_error,
        naive_clip_center_error_px=naive_center_error,
        shrink_w_px=float(box_size) - actual_w,
        shrink_h_px=float(box_size) - actual_h,
        px_budget_2cm=px_budget_2cm,
        m_per_px_conservative=m_per_px,
    )


def image_export_name(sample: dict[str, Any]) -> str:
    src = Path(str(sample["image_path"]))
    suffix = src.suffix if src.suffix else ".jpg"
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = Path(str(sample.get("image_name") or src.stem)).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return f"{digest}_{safe_stem}{suffix}"


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


def dataset_summary(
    task: str,
    box_size: int,
    dataset_dir: Path,
    records: list[BoxRecord],
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], BoxRecord]]],
) -> dict[str, Any]:
    by_split = {}
    for split in SPLITS:
        split_records = [r for r in records if r.split == split]
        group_sizes = [len(items) for (group_split, _), items in grouped.items() if group_split == split]
        by_split[split] = {
            "num_labels": len(split_records),
            "num_images": len(group_sizes),
            "multi_label_images": int(sum(1 for value in group_sizes if value > 1)),
            "max_labels_per_image": int(max(group_sizes) if group_sizes else 0),
            "shrunk_box_labels": int(sum(1 for r in split_records if r.shrink_w_px > 1e-6 or r.shrink_h_px > 1e-6)),
            "naive_clipping_would_shift_center": int(sum(1 for r in split_records if r.naive_clip_center_error_px > 1e-6)),
            "label_bounds_ok": bool(all(0.0 <= r.x_center_norm <= 1.0 and 0.0 <= r.y_center_norm <= 1.0 and 0.0 < r.w_norm <= 1.0 and 0.0 < r.h_norm <= 1.0 for r in split_records)),
            "center_error_px": summarize([r.center_error_px for r in split_records]),
            "serialized_center_error_px": summarize([r.serialized_center_error_px for r in split_records]),
            "actual_box_w_px": summarize([r.actual_w_px for r in split_records]),
            "actual_box_h_px": summarize([r.actual_h_px for r in split_records]),
            "naive_clip_center_error_px": summarize([r.naive_clip_center_error_px for r in split_records]),
        }
    return {
        "task": task,
        "box_size_px": int(box_size),
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml": str((dataset_dir / "data.yaml").resolve()),
        "box_policy": "center_preserving_shrink_at_edges",
        "splits": by_split,
    }


def write_overlays(
    task: str,
    box_size: int,
    samples_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    records: list[BoxRecord],
    out_root: Path,
    overlay_count: int,
    cols: int,
    thumb: int,
) -> list[str]:
    paths: list[str] = []
    records_by_image: dict[tuple[str, str], list[BoxRecord]] = defaultdict(list)
    for record in records:
        records_by_image[(record.split, record.image_path)].append(record)
    for split in SPLITS:
        overlay_dir = out_root / "overlays" / f"{task}_box{box_size}" / split
        overlay_dir.mkdir(parents=True, exist_ok=True)
        selected_paths = []
        seen = set()
        for sample in samples_by_key[(task, split)]:
            image_path = str(sample["image_path"])
            if image_path in seen:
                continue
            seen.add(image_path)
            selected_paths.append(image_path)
            if len(selected_paths) >= int(overlay_count):
                break
        tiles = []
        for idx, image_path in enumerate(selected_paths):
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for record in records_by_image[(split, image_path)]:
                draw.rectangle([record.box_x0, record.box_y0, record.box_x1, record.box_y1], outline=(0, 255, 0), width=3)
                draw.line([record.target_x - 6, record.target_y, record.target_x + 6, record.target_y], fill=(255, 0, 0), width=2)
                draw.line([record.target_x, record.target_y - 6, record.target_x, record.target_y + 6], fill=(255, 0, 0), width=2)
            draw.text((8, 8), f"{task} {split} box{box_size}", fill=(255, 255, 0))
            out_path = overlay_dir / f"{idx:03d}_{Path(image_path).stem}.jpg"
            image.save(out_path, quality=90)
            paths.append(str(out_path.resolve()))
            tiles.append(image)
        if tiles:
            contact = make_contact_sheet(tiles, cols=max(1, int(cols)), thumb=int(thumb))
            contact_path = out_root / "overlays" / f"{task}_box{box_size}_{split}_contact.jpg"
            contact.save(contact_path, quality=92)
            paths.append(str(contact_path.resolve()))
    return paths


def make_contact_sheet(images: list[Image.Image], cols: int, thumb: int) -> Image.Image:
    rows = int(math.ceil(len(images) / float(cols)))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (24, 24, 24))
    for idx, image in enumerate(images):
        tile = image.copy()
        tile.thumbnail((thumb, thumb))
        x = (idx % cols) * thumb + (thumb - tile.width) // 2
        y = (idx // cols) * thumb + (thumb - tile.height) // 2
        sheet.paste(tile, (x, y))
    return sheet


def summarize_sample_counts(samples_by_key: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        task: {
            split: {
                "num_labels": len(samples_by_key[(task, split)]),
                "num_unique_images": len({sample["image_path"] for sample in samples_by_key[(task, split)]}),
            }
            for split in SPLITS
        }
        for task in TASKS
    }


def summarize_metric_conversion(samples_by_key: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASKS:
        task_samples = []
        for split in SPLITS:
            samples = samples_by_key[(task, split)]
            out[f"{task}_{split}"] = metric_conversion_for_samples(samples)
            task_samples.extend(samples)
        out[f"{task}_all"] = metric_conversion_for_samples(task_samples)
    all_samples = [sample for samples in samples_by_key.values() for sample in samples]
    out["combined_all"] = metric_conversion_for_samples(all_samples)
    return out


def metric_conversion_for_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    budgets = []
    depths = []
    m_per_px = []
    angular_deg_per_px = []
    for sample in samples:
        camera = sample["camera"]
        depth = float(sample["target_depth"])
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        f = min(max(fx, 1e-9), max(fy, 1e-9))
        depths.append(depth)
        mpp = depth / f
        m_per_px.append(mpp)
        budgets.append(0.020 / max(mpp, 1e-12))
        angular_deg_per_px.append(math.degrees(math.atan(1.0 / f)))
    return {
        "num_samples": len(samples),
        "depth_m": summarize(depths),
        "oracle_m_per_px_conservative": summarize(m_per_px),
        "pixel_radius_for_2cm_oracle": summarize(budgets),
        "angular_deg_per_px_at_center": summarize(angular_deg_per_px),
    }


def summarize_box_records(records: list[BoxRecord]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASKS:
        for split in SPLITS:
            for box_size in sorted({r.box_size_px for r in records}):
                subset = [r for r in records if r.task == task and r.split == split and r.box_size_px == box_size]
                out[f"{task}_{split}_box{box_size}"] = box_record_summary(subset)
    return out


def box_record_summary(records: list[BoxRecord]) -> dict[str, Any]:
    return {
        "num_labels": len(records),
        "center_error_px": summarize([r.center_error_px for r in records]),
        "serialized_center_error_px": summarize([r.serialized_center_error_px for r in records]),
        "naive_clip_center_error_px": summarize([r.naive_clip_center_error_px for r in records]),
        "shrunk_label_count": int(sum(1 for r in records if r.shrink_w_px > 1e-6 or r.shrink_h_px > 1e-6)),
        "naive_clip_shift_count": int(sum(1 for r in records if r.naive_clip_center_error_px > 1e-6)),
        "px_budget_2cm": summarize([r.px_budget_2cm for r in records]),
        "actual_box_w_px": summarize([r.actual_w_px for r in records]),
        "actual_box_h_px": summarize([r.actual_h_px for r in records]),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_recommendations(samples_by_key: dict[tuple[str, str], list[dict[str, Any]]], records: list[BoxRecord]) -> list[str]:
    recs = []
    for task in TASKS:
        all_samples = samples_by_key[(task, "train")] + samples_by_key[(task, "test")]
        budget = metric_conversion_for_samples(all_samples)["pixel_radius_for_2cm_oracle"]
        recs.append(
            f"{task}: median 2cm budget is {budget['median']:.2f}px and p10 budget is {budget['p10']:.2f}px; center predictions need roughly single-digit pixel accuracy."
        )
    for box_size in sorted({r.box_size_px for r in records}):
        shrink = sum(1 for r in records if r.box_size_px == box_size and (r.shrink_w_px > 1e-6 or r.shrink_h_px > 1e-6))
        total = sum(1 for r in records if r.box_size_px == box_size)
        recs.append(f"box{box_size}: {shrink}/{total} labels require edge shrinking; label center remains GT by construction.")
    recs.append("Use 64px as the default smoke/training box size: it provides more YOLO object extent than 32/48 while keeping edge shrinkage modest compared with 96.")
    recs.append("Use imgsz 640 as the first detector setting; source-pixel half-step from resize is about 0.56px for 720px images, far below the 2cm budget.")
    return recs


def write_records_jsonl(path: Path, records: list[BoxRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")


def write_box_records(path: Path, records: list[BoxRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BoxRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_report(path: Path, out_root: Path, metrics: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# YOLO Bbox Center Dataset Eval 026")
    lines.append("")
    lines.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("Worker role: yolo_bbox_center_dataset_eval_026")
    lines.append(f"Output root: `{out_root.resolve()}`")
    lines.append("")
    lines.append("## Dataset Counts")
    lines.append("")
    lines.append("| Task | Split | Labels | Unique images |")
    lines.append("| --- | --- | ---: | ---: |")
    for task, splits in metrics["sample_counts"].items():
        for split, item in splits.items():
            lines.append(f"| {task} | {split} | {item['num_labels']} | {item['num_unique_images']} |")
    lines.append("")
    lines.append("## YOLO Datasets")
    lines.append("")
    lines.append("| Dataset | data.yaml | Train labels/images | Test labels/images | Shrunk labels train/test |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for name, summary in metrics["datasets"].items():
        train = summary["splits"]["train"]
        test = summary["splits"]["test"]
        lines.append(
            f"| {name} | `{summary['data_yaml']}` | {train['num_labels']}/{train['num_images']} | "
            f"{test['num_labels']}/{test['num_images']} | {train['shrunk_box_labels']}/{test['shrunk_box_labels']} |"
        )
    lines.append("")
    lines.append("## 2cm Pixel Budget")
    lines.append("")
    lines.append("| Group | N | Depth mean m | 2cm px median | 2cm px p10 | 2cm px p90 | m/px median |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group, conv in metrics["metric_conversion"].items():
        px = conv["pixel_radius_for_2cm_oracle"]
        depth = conv["depth_m"]
        mpp = conv["oracle_m_per_px_conservative"]
        lines.append(
            f"| {group} | {conv['num_samples']} | {fmt(depth['mean'])} | {fmt(px['median'], 2)} | "
            f"{fmt(px['p10'], 2)} | {fmt(px['p90'], 2)} | {fmt(mpp['median'], 6)} |"
        )
    lines.append("")
    lines.append("## Oracle And Label Sanity")
    lines.append("")
    lines.append("- Exported bbox centers are exactly the GT 2D point by construction; edge cases are handled by symmetric shrink, not center-shifting clipping.")
    lines.append(f"- YOLO label text uses {LABEL_DECIMALS} decimal places; serialized center errors are at micro-pixel scale in the audit.")
    lines.append("- The `naive_clip_center_error_px` column in `box_records.csv` reports how much center error would have been introduced by clipping fixed-size xyxy boxes at the image boundary.")
    lines.append("")
    lines.append("| Group | Labels | Center err max px | Serialized err max px | Naive clip shifted | Naive clip err max px |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for group, audit in metrics["oracle_and_label_audit"].items():
        lines.append(
            f"| {group} | {audit['num_labels']} | {fmt(audit['center_error_px']['max'], 8)} | "
            f"{fmt(audit['serialized_center_error_px']['max'], 8)} | {audit['naive_clip_shift_count']} | "
            f"{fmt(audit['naive_clip_center_error_px']['max'], 4)} |"
        )
    lines.append("")
    lines.append("## Overlays")
    lines.append("")
    for overlay in metrics["overlays"][:40]:
        lines.append(f"- `{overlay}`")
    if len(metrics["overlays"]) > 40:
        lines.append(f"- ... {len(metrics['overlays']) - 40} additional overlay files under `{out_root / 'overlays'}`")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for rec in metrics["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Summary: `{(out_root / 'summary.json').resolve()}`")
    lines.append(f"- Box records CSV: `{(out_root / 'box_records.csv').resolve()}`")
    lines.append(f"- Datasets root: `{(out_root / 'datasets').resolve()}`")
    lines.append(f"- Overlays root: `{(out_root / 'overlays').resolve()}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
