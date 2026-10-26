#!/usr/bin/env python3
"""Score official 3DGS-CD Mustard outputs under all documented mask roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L")) > 0


def _metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    for prediction, target in pairs:
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
        true_positive += int(np.logical_and(prediction, target).sum())
        false_positive += int(np.logical_and(prediction, ~target).sum())
        false_negative += int(np.logical_and(~prediction, target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(np.float64).eps)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predicted_root = args.scene / "masks_new"
    target_root = args.scene / "masks_gt"
    target_files = sorted(target_root.glob("frame_*.png"))
    if not target_files:
        raise FileNotFoundError(f"No ground-truth masks in {target_root}")

    variants: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "move_out": [],
        "move_in": [],
        "union": [],
    }
    for target_file in target_files:
        image_id = int(target_file.stem.rsplit("_", 1)[-1])
        move_out_file = predicted_root / f"mask_{image_id:05d}.png"
        move_in_file = predicted_root / f"mask_new_{image_id:05d}.png"
        if not move_out_file.is_file() or not move_in_file.is_file():
            raise FileNotFoundError(f"Missing prediction pair for frame {image_id}")
        target = _read(target_file)
        move_out = _read(move_out_file)
        move_in = _read(move_in_file)
        variants["move_out"].append((move_out, target))
        variants["move_in"].append((move_in, target))
        variants["union"].append((np.logical_or(move_out, move_in), target))

    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "scene": args.scene.name,
        "image_count": len(target_files),
        "variants": {},
    }
    for variant, pairs in variants.items():
        destination = args.output / variant
        destination.mkdir(parents=True, exist_ok=True)
        for target_file, (prediction, _) in zip(target_files, pairs):
            Image.fromarray(prediction.astype(np.uint8) * 255).save(destination / target_file.name)
        result["variants"][variant] = _metrics(pairs)

    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
