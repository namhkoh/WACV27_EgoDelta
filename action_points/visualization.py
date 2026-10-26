from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bbox": (255, 170, 30),
    "component_bbox": (80, 170, 255),
    "target": (70, 20, 110),
    "points": (80, 220, 100),
    "prediction": (70, 20, 110),
}

BBOX_WIDTH = 4
COMPONENT_BBOX_WIDTH = 3
TARGET_WIDTH = 5


def draw_label_sample(sample: dict, out_path: str | Path, crop: bool = False) -> None:
    draw_label_samples([sample], out_path, crop=crop)


def draw_label_samples(samples: list[dict], out_path: str | Path, crop: bool = False) -> None:
    if not samples:
        return
    image_path = samples[0]["image_path"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for sample in samples:
        for point in sample.get("projected_points_preview", []):
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=COLORS["points"])
        _draw_bbox(draw, sample.get("component_bbox_xyxy"), COLORS["component_bbox"], width=COMPONENT_BBOX_WIDTH)
        _draw_bbox(draw, sample.get("bbox_xyxy"), COLORS["bbox"], width=BBOX_WIDTH)
        _draw_cross(draw, sample.get("target_2d"), COLORS["target"], radius=8, width=TARGET_WIDTH)
        label = label_for_sample(sample)
        bbox = sample.get("bbox_xyxy", [4, 24, 0, 0])
        draw.text((float(bbox[0]) + 4, max(4.0, float(bbox[1]) - 14)), label, fill=COLORS["target"], font=font)
    if crop:
        union = union_bbox([sample["bbox_xyxy"] for sample in samples if sample.get("bbox_xyxy") is not None])
        if union is not None:
            image = image.crop(tuple(int(round(v)) for v in union))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def label_for_sample(sample: dict) -> str:
    instance = sample.get("instance_id")
    prefix = f"i{instance}" if instance is not None else sample.get("class_name", "target")
    return f"{prefix} {sample.get('target_confidence', 1.0):.2f}"


def union_bbox(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    x0 = min(float(box[0]) for box in boxes)
    y0 = min(float(box[1]) for box in boxes)
    x1 = max(float(box[2]) for box in boxes)
    y1 = max(float(box[3]) for box in boxes)
    return [x0, y0, x1, y1]


def draw_prediction_overlay(
    image_path: str | Path,
    predictions: list[dict],
    out_path: str | Path,
    ground_truth: list[dict] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if ground_truth:
        for sample in ground_truth:
            _draw_bbox(draw, sample.get("bbox_xyxy"), COLORS["component_bbox"], width=COMPONENT_BBOX_WIDTH)
            _draw_cross(draw, sample.get("target_2d"), COLORS["target"], radius=7, width=TARGET_WIDTH)
    for pred in predictions:
        _draw_bbox(draw, pred.get("bbox_xyxy"), COLORS["bbox"], width=BBOX_WIDTH)
        _draw_cross(draw, pred.get("action_point_2d"), COLORS["prediction"], radius=8, width=TARGET_WIDTH)
        bbox = pred.get("bbox_xyxy", [4, 24, 0, 0])
        label = f"{pred.get('class_name', 'target')} {float(pred.get('confidence', 0.0)):.2f}"
        draw.text((float(bbox[0]) + 4, max(4.0, float(bbox[1]) - 14)), label, fill=COLORS["prediction"], font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def make_contact_sheet(paths: Iterable[str | Path], out_path: str | Path, cols: int = 3, thumb_w: int = 420) -> None:
    paths = [Path(path) for path in paths]
    if not paths:
        return
    font = ImageFont.load_default()
    label_h = 18
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        thumb_h = max(1, int(round(thumb_w * img.height / max(img.width, 1))))
        thumbs.append((path, img.resize((thumb_w, thumb_h))))
    thumb_h = max(img.height for _, img in thumbs)
    rows = int(math.ceil(len(thumbs) / max(cols, 1)))
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (path, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + thumb_h), path.stem[:58], fill=(235, 235, 235), font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox: list[float] | None, color: tuple[int, int, int], width: int) -> None:
    if bbox is None:
        return
    x0, y0, x1, y1 = [float(v) for v in bbox]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)


def _draw_cross(draw: ImageDraw.ImageDraw, point: list[float] | None, color: tuple[int, int, int], radius: int, width: int) -> None:
    if point is None:
        return
    x, y = float(point[0]), float(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)
    draw.line((x - radius - 4, y, x + radius + 4, y), fill=color, width=width)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill=color, width=width)
