#!/usr/bin/env python3
"""Convert the WUR/4TU tomato peduncle keypoint dataset to YOLO pose format."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SRC = "datasets/wur_peduncle_keypoints/keypoints_dataset"
DEFAULT_TRAIN_JSON = "peduncle_train_503/peduncle_train_final_503_final.json"
DEFAULT_VAL_JSON = "test/test_rgb/peduncle_test_new.json"


def main() -> int:
    args = build_arg_parser().parse_args()
    src = Path(args.src)
    out = Path(args.out)
    train_json = resolve_json_path(src, args.train_json)
    val_json = resolve_json_path(src, args.val_json)

    train_stats = convert_split(
        split="train",
        annotation_json=train_json,
        image_dir=train_json.parent,
        out_dir=out,
        copy_images=args.copy_images,
        overwrite=args.overwrite,
        bbox_pad_px=args.bbox_pad_px,
    )
    val_stats = convert_split(
        split="val",
        annotation_json=val_json,
        image_dir=val_json.parent,
        out_dir=out,
        copy_images=args.copy_images,
        overwrite=args.overwrite,
        bbox_pad_px=args.bbox_pad_px,
    )

    keypoint_names = train_stats["keypoint_names"] or val_stats["keypoint_names"]
    write_dataset_yaml(
        yaml_path=Path(args.yaml_out),
        dataset_dir=out,
        keypoint_names=keypoint_names,
    )
    summary = {
        "source": str(src),
        "output": str(out),
        "yaml": str(Path(args.yaml_out)),
        "keypoint_names": keypoint_names,
        "train": train_stats,
        "val": val_stats,
    }
    summary_path = out / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Converted WUR peduncle dataset to YOLO pose format")
    print(f"  train images: {train_stats['num_images']}")
    print(f"  train labels: {train_stats['num_annotations']}")
    print(f"  val images: {val_stats['num_images']}")
    print(f"  val labels: {val_stats['num_annotations']}")
    print(f"  keypoints: {', '.join(keypoint_names)}")
    print(f"  dataset: {out}")
    print(f"  yaml: {Path(args.yaml_out)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Wageningen/4TU tomato peduncle COCO-style keypoint "
            "annotations into Ultralytics YOLO pose labels."
        )
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="Root of the unzipped keypoints_dataset")
    parser.add_argument(
        "--train-json",
        default=DEFAULT_TRAIN_JSON,
        help="Train annotation JSON path, absolute or relative to --src",
    )
    parser.add_argument(
        "--val-json",
        default=DEFAULT_VAL_JSON,
        help="Validation annotation JSON path, absolute or relative to --src",
    )
    parser.add_argument("--out", default="datasets/peduncle_pose_yolo")
    parser.add_argument("--yaml-out", default="data/peduncle_pose.yaml")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking them into the YOLO dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing image links/files in the output dataset",
    )
    parser.add_argument(
        "--bbox-pad-px",
        type=float,
        default=4.0,
        help="Padding added to annotation boxes before normalization",
    )
    return parser


def resolve_json_path(src: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = src / path
    if not path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {path}")
    return path


def convert_split(
    split: str,
    annotation_json: Path,
    image_dir: Path,
    out_dir: Path,
    copy_images: bool,
    overwrite: bool,
    bbox_pad_px: float,
) -> dict[str, Any]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    keypoint_names = infer_keypoint_names(data)
    num_keypoints = len(keypoint_names)

    image_out_dir = out_dir / "images" / split
    label_out_dir = out_dir / "labels" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_annotations = 0
    skipped_missing_images: list[str] = []

    for image_id in sorted(annotations_by_image):
        image_info = images.get(image_id)
        if image_info is None:
            continue
        file_name = image_info["file_name"]
        src_image = image_dir / file_name
        if not src_image.exists():
            skipped_missing_images.append(str(src_image))
            continue

        dst_image = image_out_dir / file_name
        link_or_copy_image(src_image, dst_image, copy_images=copy_images, overwrite=overwrite)

        width = int(image_info["width"])
        height = int(image_info["height"])
        label_lines = [
            annotation_to_yolo_pose_line(
                annotation=annotation,
                width=width,
                height=height,
                num_keypoints=num_keypoints,
                bbox_pad_px=bbox_pad_px,
            )
            for annotation in annotations_by_image[image_id]
        ]
        label_path = label_out_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        num_images += 1
        num_annotations += len(label_lines)

    if skipped_missing_images:
        preview = "\n".join(f"  - {item}" for item in skipped_missing_images[:10])
        raise FileNotFoundError(
            f"{len(skipped_missing_images)} images referenced by {annotation_json} were missing:\n"
            f"{preview}"
        )

    return {
        "split": split,
        "annotation_json": str(annotation_json),
        "image_dir": str(image_dir),
        "num_images": num_images,
        "num_annotations": num_annotations,
        "keypoint_names": keypoint_names,
    }


def infer_keypoint_names(data: dict[str, Any]) -> list[str]:
    categories = data.get("categories", [])
    if categories:
        keypoints = categories[0].get("keypoints") or []
        if keypoints:
            return [str(item) for item in keypoints]
    max_keypoints = 0
    for annotation in data.get("annotations", []):
        max_keypoints = max(max_keypoints, len(annotation.get("keypoints", [])) // 3)
    return [f"kp_{idx:02d}" for idx in range(max_keypoints)]


def annotation_to_yolo_pose_line(
    annotation: dict[str, Any],
    width: int,
    height: int,
    num_keypoints: int,
    bbox_pad_px: float,
) -> str:
    bbox = annotation.get("bbox")
    if bbox and len(bbox) >= 4 and float(bbox[2]) > 0.0 and float(bbox[3]) > 0.0:
        x_min, y_min, box_w, box_h = [float(v) for v in bbox[:4]]
        x_max = x_min + box_w
        y_max = y_min + box_h
    else:
        x_min, y_min, x_max, y_max = bbox_from_keypoints(annotation.get("keypoints", []))

    x_min = clamp(x_min - bbox_pad_px, 0.0, float(width - 1))
    y_min = clamp(y_min - bbox_pad_px, 0.0, float(height - 1))
    x_max = clamp(x_max + bbox_pad_px, 0.0, float(width - 1))
    y_max = clamp(y_max + bbox_pad_px, 0.0, float(height - 1))
    if x_max <= x_min:
        x_max = min(float(width - 1), x_min + 1.0)
    if y_max <= y_min:
        y_max = min(float(height - 1), y_min + 1.0)

    cx = ((x_min + x_max) * 0.5) / width
    cy = ((y_min + y_max) * 0.5) / height
    bw = (x_max - x_min) / width
    bh = (y_max - y_min) / height

    values: list[float | int] = [0, cx, cy, bw, bh]
    keypoints = list(annotation.get("keypoints", []))
    for idx in range(num_keypoints):
        base = idx * 3
        if base + 2 >= len(keypoints):
            values.extend([0.0, 0.0, 0])
            continue
        x = float(keypoints[base])
        y = float(keypoints[base + 1])
        visibility = int(keypoints[base + 2])
        visible = visibility > 0 and 0.0 <= x < width and 0.0 <= y < height
        if not visible:
            values.extend([0.0, 0.0, 0])
            continue
        values.extend([clamp(x / width, 0.0, 1.0), clamp(y / height, 0.0, 1.0), visibility])

    return " ".join(format_yolo_value(item) for item in values)


def bbox_from_keypoints(keypoints: list[Any]) -> tuple[float, float, float, float]:
    xy = []
    for idx in range(0, len(keypoints), 3):
        if idx + 2 >= len(keypoints):
            continue
        if float(keypoints[idx + 2]) <= 0.0:
            continue
        xy.append((float(keypoints[idx]), float(keypoints[idx + 1])))
    if not xy:
        return 0.0, 0.0, 1.0, 1.0
    xs = [item[0] for item in xy]
    ys = [item[1] for item in xy]
    return min(xs), min(ys), max(xs), max(ys)


def link_or_copy_image(
    src_image: Path,
    dst_image: Path,
    copy_images: bool,
    overwrite: bool,
) -> None:
    if dst_image.exists() or dst_image.is_symlink():
        if not overwrite:
            return
        dst_image.unlink()
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src_image, dst_image)
    else:
        dst_image.symlink_to(src_image.resolve())


def write_dataset_yaml(yaml_path: Path, dataset_dir: Path, keypoint_names: list[str]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    flip_idx = ", ".join(str(idx) for idx in range(len(keypoint_names)))
    keypoint_comment = "\n".join(f"# {idx} {name}" for idx, name in enumerate(keypoint_names))
    text = f"""# Ultralytics YOLO pose dataset config for WUR/4TU tomato peduncle keypoints.
# Generated by scripts/convert_wur_peduncle_to_yolo_pose.py.

path: {dataset_dir.resolve()}
train: images/train
val: images/val

names:
  0: peduncle

kpt_shape: [{len(keypoint_names)}, 3]
flip_idx: [{flip_idx}]

# Keypoint order:
{keypoint_comment}
"""
    yaml_path.write_text(text, encoding="utf-8")


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def format_yolo_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
