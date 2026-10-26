#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.crop_dataset import bbox_features, padded_crop_box
from action_points.manifest import load_action_manifests, write_json
from action_points.models import ActionPointModelConfig, build_action_point_model
from action_points.visualization import draw_prediction_overlay


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector + action-point head inference on RGB images.")
    parser.add_argument("--detector", required=True, help="YOLO detector checkpoint.")
    parser.add_argument("--action_checkpoint", required=True, help="Action-point head best.pt/last.pt.")
    parser.add_argument("--image", action="append", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--manifest", action="append", default=None, help="Action-point manifest JSON; runs on its unique image paths.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--context_pad", type=float, default=0.10)
    parser.add_argument("--save_vis", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Inference requires torch, pillow, and ultralytics.") from exc

    image_paths = collect_images(args.image, args.image_dir, args.manifest)
    if not image_paths:
        raise RuntimeError("No input images found")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.detector)
    ckpt = torch.load(args.action_checkpoint, map_location=device)
    model_config = ActionPointModelConfig(**ckpt.get("model_config", {}))
    action_model = build_action_point_model(model_config).to(device)
    action_model.load_state_dict(ckpt["model_state"])
    action_model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_image = []
    names = getattr(detector, "names", {}) or {}
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        result = detector.predict(source=str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        image_predictions = []
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = tensor_to_list(boxes.xyxy)
            confs = tensor_to_list(boxes.conf)
            classes = [int(v) for v in tensor_to_list(boxes.cls)]
            for det_idx, (bbox, det_conf, cls) in enumerate(zip(xyxy, confs, classes)):
                pred = predict_for_bbox(
                    action_model,
                    image,
                    bbox,
                    width,
                    height,
                    model_config.image_size,
                    args.context_pad,
                    device,
                )
                class_name = str(names.get(cls, names.get(str(cls), cls)))
                item = {
                    "det_id": f"{Path(image_path).stem}_{det_idx:03d}",
                    "class_id": int(cls),
                    "class_name": class_name,
                    "detector_confidence": float(det_conf),
                    "confidence": float(det_conf),
                    "bbox_xyxy": [float(v) for v in bbox],
                    "action_point_2d": [float(pred["xy"][0]), float(pred["xy"][1])],
                }
                if pred.get("depth") is not None:
                    item["predicted_depth_model_units"] = float(pred["depth"])
                image_predictions.append(item)
        record = {
            "image_path": str(Path(image_path).resolve()),
            "width": width,
            "height": height,
            "predictions": image_predictions,
        }
        predictions_by_image.append(record)
        if args.save_vis:
            draw_prediction_overlay(
                image_path,
                image_predictions,
                out_dir / "visualizations" / f"{Path(image_path).stem}_action_points.jpg",
            )
    write_json(out_dir / "predictions.json", {"images": predictions_by_image})
    print(f"Saved predictions: {out_dir / 'predictions.json'}")
    return 0


def collect_images(image_args: list[str] | None, image_dir: str | None, manifest_args: list[str] | None = None) -> list[Path]:
    paths = [Path(path) for path in image_args or []]
    if image_dir is not None:
        paths.extend(sorted(path for path in Path(image_dir).iterdir() if path.suffix in IMAGE_EXTS))
    for manifest in load_action_manifests(manifest_args or []):
        paths.extend(Path(sample["image_path"]) for sample in manifest["samples"])

    unique_paths = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths


def predict_for_bbox(model, image, bbox, width: int, height: int, image_size: int, context_pad: float, device: str):
    import numpy as np
    import torch

    crop_box = padded_crop_box([float(v) for v in bbox], width, height, context_pad)
    crop = image.crop(tuple(int(round(v)) for v in crop_box)).resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    feats = torch.tensor([bbox_features([float(v) for v in bbox], width, height)], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(tensor, feats)
        rel = out["xy"][0].detach().cpu().numpy().tolist()
        depth = None
        if "depth" in out:
            depth = float(out["depth"][0].detach().cpu().reshape(-1)[0])
    x0, y0, x1, y1 = crop_box
    return {
        "xy": [x0 + float(rel[0]) * max(x1 - x0, 1e-6), y0 + float(rel[1]) * max(y1 - y0, 1e-6)],
        "depth": depth,
    }


def tensor_to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
