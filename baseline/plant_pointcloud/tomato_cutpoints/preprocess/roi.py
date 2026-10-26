from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_roi_json(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {"crop_boxes": [], "exclude_boxes": [], "stable_boxes": []}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "crop_boxes": list(data.get("crop_boxes", [])),
        "exclude_boxes": list(data.get("exclude_boxes", [])),
        "stable_boxes": list(data.get("stable_boxes", [])),
    }


def merge_roi_config(config_roi: dict[str, Any] | None, roi_json: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    config_roi = config_roi or {}
    roi_json = roi_json or {}
    return {
        "crop_boxes": list(config_roi.get("crop_boxes", [])) + list(roi_json.get("crop_boxes", [])),
        "exclude_boxes": list(config_roi.get("exclude_boxes", [])) + list(roi_json.get("exclude_boxes", [])),
        "stable_boxes": list(config_roi.get("stable_boxes", [])) + list(roi_json.get("stable_boxes", [])),
    }


def points_in_box(xyz: np.ndarray, box: dict[str, Any]) -> np.ndarray:
    mins = np.asarray(box["min"], dtype=float)
    maxs = np.asarray(box["max"], dtype=float)
    if mins.shape != (3,) or maxs.shape != (3,):
        raise ValueError(f"ROI box must contain 3D min/max: {box}")
    return np.all((xyz >= mins) & (xyz <= maxs), axis=1)


def points_in_any_box(xyz: np.ndarray, boxes: list[dict[str, Any]]) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(len(xyz), dtype=bool)
    mask = np.zeros(len(xyz), dtype=bool)
    for box in boxes:
        mask |= points_in_box(xyz, box)
    return mask


def apply_roi_filter(xyz: np.ndarray, roi: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.ones(len(xyz), dtype=bool)
    report: dict[str, Any] = {"n_input": int(len(xyz))}

    crop_boxes = roi.get("crop_boxes", [])
    if crop_boxes:
        crop_mask = points_in_any_box(xyz, crop_boxes)
        mask &= crop_mask
        report["crop_kept"] = int(crop_mask.sum())
    else:
        report["crop_kept"] = int(len(xyz))

    exclude_boxes = roi.get("exclude_boxes", [])
    if exclude_boxes:
        exclude_mask = points_in_any_box(xyz, exclude_boxes)
        mask &= ~exclude_mask
        report["exclude_removed"] = int(exclude_mask.sum())
    else:
        report["exclude_removed"] = 0

    report["n_output"] = int(mask.sum())
    return mask, report


def stable_mask(xyz: np.ndarray, roi: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    return points_in_any_box(xyz, roi.get("stable_boxes", []))
