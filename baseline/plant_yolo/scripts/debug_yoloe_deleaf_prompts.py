#!/usr/bin/env python
"""Run YOLOE deleafing prompts on one image and save raw 2D debug outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_yoloe_leaf_prompts import draw_overlay, result_to_detections


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug YOLOE petiole/deleafing segmentation")
    parser.add_argument("--image", required=True, help="Input RGB/BGR image path")
    parser.add_argument("--task-config", default="configs/deleaf.yaml")
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="YOLOE segmentation checkpoint")
    parser.add_argument("--out", default="outputs/yoloe_deleaf_debug", help="Output directory")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=1000)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLOE  # type: ignore
    from yolo_aria_harvest_poc.task_config import load_task_config

    task_config = load_task_config(args.task_config)
    image_path = Path(args.image)
    out_dir = Path(args.out)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    positive_prompts = list(task_config.text_prompts)
    negative_prompts = list(task_config.negative_prompts)
    all_prompts = positive_prompts + negative_prompts
    conf = float(task_config.min_2d_confidence if args.conf is None else args.conf)

    model = YOLOE(args.model)
    model.set_classes(all_prompts)
    result = model.predict(
        source=image,
        conf=conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        verbose=False,
    )[0]

    detections = result_to_detections(
        result,
        all_prompts,
        set(negative_prompts),
        image.shape[:2],
        masks_dir,
    )
    overlay = draw_overlay(image, detections, masks_dir)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")
    (out_dir / "prompts.json").write_text(
        json.dumps(
            {
                "positive_prompts": positive_prompts,
                "negative_prompts": negative_prompts,
                "all_prompts": all_prompts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(args.model, image_path, positive_prompts, negative_prompts, detections, out_dir)
    return 0


def print_summary(
    model: str,
    image_path: Path,
    positive_prompts: list[str],
    negative_prompts: list[str],
    detections: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    num_negative = sum(1 for item in detections if item.get("is_negative_prompt"))
    print(f"YOLOE model: {model}")
    print(f"Image: {image_path}")
    print(f"Positive deleaf prompts: {positive_prompts}")
    print(f"Negative distractors: {negative_prompts}")
    print(f"Detections: {len(detections)}")
    print(f"Negative-prompt detections: {num_negative}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
