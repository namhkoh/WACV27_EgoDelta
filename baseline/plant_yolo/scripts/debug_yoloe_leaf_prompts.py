#!/usr/bin/env python
"""Run YOLOE leaf prompts on one image and save raw 2D debug outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PROMPTS = [
    "leaf",
    "leaves",
    "tomato leaf",
    "tomato leaves",
    "green leaf",
    "green tomato leaves",
    "tomato plant leaf",
    "leaf blade",
    "tomato leaf blade",
    "compound tomato leaf",
    "tomato leaflet",
    "green tomato leaflet",
    "green foliage",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "tomato fruit",
    "tomato",
    "ripe tomato",
    "green tomato",
    "unripe tomato",
    "plant stem",
    "tomato stem",
    "main tomato stem",
    "vertical tomato stem",
    "background pipe",
    "greenhouse pipe",
    "irrigation tube",
    "support pole",
    "greenhouse string",
    "support string",
    "trellis string",
    "wire",
    "hand",
    "glove",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug YOLOE open-vocabulary leaf segmentation")
    parser.add_argument("--image", required=True, help="Input RGB/BGR image path")
    parser.add_argument("--model", default="yoloe-26l-seg.pt", help="YOLOE segmentation checkpoint")
    parser.add_argument("--out", default="outputs/yoloe_leaf_debug", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="Positive YOLOE text prompt class list",
    )
    parser.add_argument(
        "--negative-prompts",
        nargs="+",
        default=DEFAULT_NEGATIVE_PROMPTS,
        help="Distractor YOLOE class list. Detections assigned to these prompts are marked negative.",
    )
    args = parser.parse_args()

    from ultralytics import YOLOE  # type: ignore

    image_path = Path(args.image)
    out_dir = Path(args.out)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    model = YOLOE(args.model)
    all_prompts = list(args.prompts) + list(args.negative_prompts)
    model.set_classes(all_prompts)
    result = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        verbose=False,
    )[0]

    detections = result_to_detections(
        result,
        all_prompts,
        set(args.negative_prompts),
        image.shape[:2],
        masks_dir,
    )
    overlay = draw_overlay(image, detections, masks_dir)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2), encoding="utf-8")
    (out_dir / "prompts.json").write_text(
        json.dumps(
            {
                "positive_prompts": args.prompts,
                "negative_prompts": args.negative_prompts,
                "all_prompts": all_prompts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    num_negative = sum(1 for item in detections if item.get("is_negative_prompt"))
    print(f"YOLOE model: {args.model}")
    print(f"Image: {image_path}")
    print(f"Positive prompts: {args.prompts}")
    print(f"Negative prompts: {args.negative_prompts}")
    print(f"Detections: {len(detections)}")
    print(f"Negative-prompt detections: {num_negative}")
    print(f"Output: {out_dir}")
    return 0


def result_to_detections(
    result: object,
    prompts: list[str],
    negative_prompts: set[str],
    image_shape: tuple[int, int],
    masks_dir: Path,
) -> list[dict[str, object]]:
    boxes = getattr(result, "boxes", None)
    masks_obj = getattr(result, "masks", None)
    if boxes is None or masks_obj is None or getattr(masks_obj, "data", None) is None:
        return []

    xyxy = tensor_to_numpy(getattr(boxes, "xyxy", np.zeros((0, 4))))
    conf = tensor_to_numpy(getattr(boxes, "conf", np.zeros((len(xyxy),))))
    cls = tensor_to_numpy(getattr(boxes, "cls", np.zeros((len(xyxy),)))).astype(int)
    masks = tensor_to_numpy(masks_obj.data)
    if masks.ndim == 2:
        masks = masks[None, ...]

    height, width = image_shape
    detections: list[dict[str, object]] = []
    for idx in range(min(len(xyxy), len(masks))):
        mask = (masks[idx] > 0.5).astype(np.uint8)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        prompt = prompts[int(cls[idx])] if 0 <= int(cls[idx]) < len(prompts) else f"class_{int(cls[idx])}"
        is_negative = prompt in negative_prompts
        mask_path = masks_dir / f"det_{idx:03d}.npz"
        np.savez_compressed(mask_path, mask=mask)
        detections.append(
            {
                "det_id": f"det_{idx:03d}",
                "prompt": prompt,
                "confidence": float(conf[idx]) if idx < len(conf) else None,
                "is_negative_prompt": bool(is_negative),
                "bbox_xyxy": [float(v) for v in xyxy[idx].tolist()],
                "mask_area_px": int(mask.sum()),
                "mask_path": str(mask_path),
            }
        )
    detections.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    return detections


def draw_overlay(
    image_bgr: np.ndarray,
    detections: list[dict[str, object]],
    masks_dir: Path,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colors = [
        (0, 220, 255),
        (0, 180, 0),
        (255, 80, 0),
        (255, 0, 180),
        (180, 80, 255),
        (80, 220, 80),
    ]
    for idx, det in enumerate(detections):
        mask_path = Path(str(det["mask_path"]))
        if not mask_path.is_absolute():
            mask_path = masks_dir.parent / mask_path.name
        if not mask_path.exists():
            mask_path = Path(str(det["mask_path"]))
        mask = np.load(mask_path)["mask"].astype(np.uint8)
        color = (80, 80, 80) if det.get("is_negative_prompt") else colors[idx % len(colors)]
        colored = np.zeros_like(overlay)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        prefix = "NEG " if det.get("is_negative_prompt") else ""
        label = f"{prefix}{det['prompt']} {float(det.get('confidence') or 0.0):.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
