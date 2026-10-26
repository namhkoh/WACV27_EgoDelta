from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path

from action_points.manifest import iter_samples, sample_split, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def export_yolo_detection_dataset(
    manifests: list[dict],
    out_dir: str | Path,
    scene_split: dict[str, str] | None = None,
    class_mode: str = "unified",
    unified_class_name: str = "target",
    copy_images: bool = False,
    allow_single_scene_debug_split: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    scene_split = scene_split or {}
    samples = list(iter_samples(manifests))
    if not samples:
        raise RuntimeError("No action-point samples to export")

    class_names = _resolve_class_names(samples, class_mode, unified_class_name)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        split = sample_split(sample, scene_split)
        grouped[(split, _image_export_name(sample))].append(sample)

    splits_present = sorted({split for split, _name in grouped.keys()})
    if "val" not in splits_present and allow_single_scene_debug_split:
        for (split, name), group_samples in list(grouped.items()):
            if split == "train":
                grouped[("val", name)] = group_samples
        splits_present = sorted({split for split, _name in grouped.keys()})

    for split in splits_present:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_exports: dict[tuple[str, str], Path] = {}
    for key, group_samples in grouped.items():
        split, export_name = key
        src = Path(group_samples[0]["image_path"])
        dst = out_dir / "images" / split / export_name
        if key not in image_exports:
            _link_or_copy(src, dst, copy_images)
            image_exports[key] = dst
        label_path = out_dir / "labels" / split / f"{Path(export_name).stem}.txt"
        label_lines = []
        for sample in group_samples:
            class_name = unified_class_name if class_mode == "unified" else str(sample.get("task_type") or sample.get("class_name"))
            cls = class_to_id[class_name]
            xywh = sample["bbox_xywh_norm"]
            label_lines.append(f"{cls} {xywh[0]:.8f} {xywh[1]:.8f} {xywh[2]:.8f} {xywh[3]:.8f}")
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    yaml_path = out_dir / "data.yaml"
    train_path = out_dir / "images" / "train"
    val_path = out_dir / "images" / ("val" if "val" in splits_present else "train")
    test_path = out_dir / "images" / "test"
    yaml_text = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        f"val: {val_path.relative_to(out_dir)}",
    ]
    if test_path.exists():
        yaml_text.append("test: images/test")
    yaml_text.append("names:")
    for idx, name in enumerate(class_names):
        yaml_text.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(yaml_text) + "\n", encoding="utf-8")

    summary = {
        "out_dir": str(out_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "class_mode": class_mode,
        "class_names": class_names,
        "num_samples": len(samples),
        "num_images": len(image_exports),
        "splits": {
            split: {
                "images": sum(1 for key in image_exports if key[0] == split),
                "labels": sum(len(group) for (group_split, _), group in grouped.items() if group_split == split),
            }
            for split in splits_present
        },
        "single_scene_debug_val": bool("val" in splits_present and allow_single_scene_debug_split),
    }
    write_json(out_dir / "export_summary.json", summary)
    return summary


def _resolve_class_names(samples: list[dict], class_mode: str, unified_class_name: str) -> list[str]:
    if class_mode == "unified":
        return [unified_class_name]
    if class_mode != "task":
        raise ValueError("--class-mode must be unified or task")
    names = sorted({str(sample.get("task_type") or sample.get("class_name")) for sample in samples})
    return names or [unified_class_name]


def _image_export_name(sample: dict) -> str:
    src = Path(sample["image_path"])
    suffix = src.suffix if src.suffix in IMAGE_EXTS else ".jpg"
    scene = str(sample.get("scene_id", "scene"))
    stem = Path(str(sample.get("image_name") or src.name)).stem
    return f"{scene}_{stem}{suffix}"


def _link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)
