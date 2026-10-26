"""YOLO11 tomato segmentation inference wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, relpath, should_write_visualization
from .visualization import draw_yolo_overlay


def run_yolo_for_split(
    split: str,
    frames: list[dict[str, Any]],
    model_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Run pretrained tomato YOLO segmentation on one split and save masks/JSON."""

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO inference. Install requirements.txt first."
        ) from exc

    out = Path(out_dir)
    pred_dir = out / "yolo_predictions"
    mask_dir = pred_dir / "masks"
    vis_dir = out / "visualizations" / f"yolo_{split}"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    predictions: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(tqdm(frames, desc=f"YOLO {split}", unit="frame")):
        image_path = Path(frame["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image for YOLO inference: {image_path}")
        height, width = image.shape[:2]

        result = model.predict(
            source=image,
            conf=config.conf_thres,
            imgsz=config.imgsz,
            retina_masks=config.retina_masks,
            max_det=config.max_det,
            verbose=False,
        )[0]

        frame_predictions, masks = _result_to_frame_predictions(
            result=result,
            names=names,
            frame=frame,
            width=width,
            height=height,
            mask_dir=mask_dir,
            out_dir=out,
        )
        predictions.append(
            {
                "frame_id": frame["frame_id"],
                "timestamp_ns": int(frame["timestamp_ns"]),
                "image_path": frame["image_path"],
                "detections": frame_predictions,
            }
        )

        if should_write_visualization(frame_index, config.visualization_stride):
            overlay = draw_yolo_overlay(image, frame_predictions, masks)
            cv2.imwrite(str(vis_dir / f"{frame['frame_id']}.jpg"), overlay)

    output_path = pred_dir / f"{split}_predictions.json"
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions


def _result_to_frame_predictions(
    result: Any,
    names: dict[int, str] | dict[str, str],
    frame: dict[str, Any],
    width: int,
    height: int,
    mask_dir: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    boxes = getattr(result, "boxes", None)
    masks_obj = getattr(result, "masks", None)
    if boxes is None or masks_obj is None or getattr(masks_obj, "data", None) is None:
        return [], []

    xyxy = _tensor_to_numpy(boxes.xyxy)
    conf = _tensor_to_numpy(boxes.conf)
    cls = _tensor_to_numpy(boxes.cls).astype(int)
    masks_data = _tensor_to_numpy(masks_obj.data)
    if masks_data.ndim == 2:
        masks_data = masks_data[None, ...]

    detections: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    count = min(len(xyxy), len(masks_data))
    for det_index in range(count):
        mask = (masks_data[det_index] > 0.5).astype(np.uint8)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_area = int(mask.sum())
        if mask_area == 0:
            continue
        class_id = int(cls[det_index])
        raw_name = _class_name(names, class_id)
        det_id = f"{frame['frame_id']}_det_{len(detections):03d}"
        mask_path = mask_dir / f"{det_id}.npz"
        np.savez_compressed(mask_path, mask=mask.astype(bool))
        detection = {
            "det_id": det_id,
            "class_id": class_id,
            "class_name_raw": raw_name,
            "semantic_class": "tomato",
            "confidence": float(conf[det_index]),
            "bbox_xyxy": [float(v) for v in xyxy[det_index].tolist()],
            "mask": relpath(mask_path, out_dir),
            "mask_path": relpath(mask_path, out_dir),
            "mask_area_px": mask_area,
        }
        detections.append(detection)
        masks.append(mask)
    return detections, masks


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _class_name(names: dict[int, str] | dict[str, str], class_id: int) -> str:
    raw = names.get(class_id, names.get(str(class_id), str(class_id)))  # type: ignore[arg-type]
    return str(raw).replace("-", "_").replace(" ", "_")
