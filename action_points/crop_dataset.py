from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from action_points.manifest import filter_samples, iter_samples, load_scene_split


class ActionPointCropDataset:
    def __init__(
        self,
        manifests: Iterable[dict],
        split: str | None = None,
        scenes: set[str] | None = None,
        scene_split_json: str | Path | None = None,
        image_size: int = 224,
        context_pad: float = 0.10,
        augment: bool = False,
        crop_jitter_shift: float = 0.08,
        crop_jitter_scale: float = 0.25,
        color_jitter: float = 0.20,
        blur_prob: float = 0.10,
    ) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError("ActionPointCropDataset requires torch.") from exc

        self.image_size = int(image_size)
        self.context_pad = float(context_pad)
        self.augment = bool(augment)
        self.crop_jitter_shift = float(crop_jitter_shift)
        self.crop_jitter_scale = float(crop_jitter_scale)
        self.color_jitter = float(color_jitter)
        self.blur_prob = float(blur_prob)
        scene_split = load_scene_split(scene_split_json)
        self.samples = filter_samples(iter_samples(manifests), split=split, scenes=scenes, scene_split=scene_split)
        if not self.samples:
            raise RuntimeError("No samples matched the requested split/scenes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        import torch

        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        crop_box = padded_crop_box(sample["bbox_xyxy"], sample["width"], sample["height"], self.context_pad)
        if self.augment:
            crop_box = jitter_crop_box(crop_box, sample["target_2d"], sample["width"], sample["height"], self.crop_jitter_shift, self.crop_jitter_scale)
        crop = image.crop(tuple(int(round(v)) for v in crop_box)).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.augment:
            crop = augment_crop_appearance(crop, self.color_jitter, self.blur_prob)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        target_rel = relabel_target_for_crop(sample["target_2d"], crop_box)
        return {
            "image": tensor,
            "bbox_features": torch.tensor(bbox_features(sample["bbox_xyxy"], sample["width"], sample["height"]), dtype=torch.float32),
            "target_xy": torch.tensor(target_rel, dtype=torch.float32),
            "target_depth": torch.tensor([float(sample.get("target_depth", 0.0))], dtype=torch.float32),
            "sample": sample,
        }


def bbox_features(bbox_xyxy: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    return [
        x0 / w,
        y0 / h,
        x1 / w,
        y1 / h,
        bw / w,
        bh / h,
        (bw * bh) / (w * h),
        bw / max(bh, 1.0),
    ]


def padded_crop_box(bbox_xyxy: list[float], width: int, height: int, pad_fraction: float) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    pad = float(pad_fraction) * max(bw, bh)
    return [
        max(0.0, x0 - pad),
        max(0.0, y0 - pad),
        min(float(width - 1), x1 + pad),
        min(float(height - 1), y1 + pad),
    ]


def relabel_target_for_crop(target_xy: list[float], crop_box: list[float]) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in crop_box]
    return [
        float((float(target_xy[0]) - x0) / max(x1 - x0, 1e-6)),
        float((float(target_xy[1]) - y0) / max(y1 - y0, 1e-6)),
    ]


def jitter_crop_box(
    crop_box: list[float],
    target_xy: list[float],
    width: int,
    height: int,
    shift_fraction: float,
    scale_fraction: float,
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in crop_box]
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    crop_w = max(x1 - x0, 1.0)
    crop_h = max(y1 - y0, 1.0)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    for _ in range(10):
        scale = 1.0 + random.uniform(-scale_fraction, scale_fraction)
        scale = max(0.75, min(1.50, scale))
        new_w = crop_w * scale
        new_h = crop_h * scale
        dx = random.uniform(-shift_fraction, shift_fraction) * crop_w
        dy = random.uniform(-shift_fraction, shift_fraction) * crop_h
        candidate = clamp_box(
            [
                center_x + dx - 0.5 * new_w,
                center_y + dy - 0.5 * new_h,
                center_x + dx + 0.5 * new_w,
                center_y + dy + 0.5 * new_h,
            ],
            width,
            height,
        )
        if candidate[0] <= target_x <= candidate[2] and candidate[1] <= target_y <= candidate[3]:
            return candidate
    return crop_box


def clamp_box(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    if x1 <= x0:
        x1 = min(float(width - 1), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(height - 1), y0 + 1.0)
    return [x0, y0, x1, y1]


def augment_crop_appearance(crop: Image.Image, color_jitter: float, blur_prob: float) -> Image.Image:
    jitter = max(float(color_jitter), 0.0)
    if jitter > 0.0:
        crop = ImageEnhance.Brightness(crop).enhance(random.uniform(max(0.1, 1.0 - jitter), 1.0 + jitter))
        crop = ImageEnhance.Contrast(crop).enhance(random.uniform(max(0.1, 1.0 - jitter), 1.0 + jitter))
        crop = ImageEnhance.Color(crop).enhance(random.uniform(max(0.1, 1.0 - 0.5 * jitter), 1.0 + 0.5 * jitter))
        crop = ImageEnhance.Sharpness(crop).enhance(random.uniform(max(0.1, 1.0 - jitter), 1.0 + jitter))
    if random.random() < max(float(blur_prob), 0.0):
        crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
    return crop
