"""Tomato YOLO segmentation backend for the harvest task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..task_config import TaskConfig
from ..yolo_inference import _class_name, _tensor_to_numpy
from .base import SegmentationDetection, make_detection


class YoloTomatoBackend:
    name = "yolo_tomato"

    def __init__(self, model_path: str | Path | None) -> None:
        if not model_path:
            raise ValueError("--model is required for --segmentation-backend yolo_tomato")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError("ultralytics is required for tomato YOLO inference.") from exc
        self.model = YOLO(str(model_path))
        self.names = getattr(self.model, "names", {}) or {}

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        frame: dict[str, Any],
        task_config: TaskConfig,
        pipeline_config: PipelineConfig,
        mask_dir: Path,
        out_dir: Path,
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        import cv2  # type: ignore

        height, width = image_bgr.shape[:2]
        result = self.model.predict(
            source=image_bgr,
            conf=task_config.min_2d_confidence,
            imgsz=pipeline_config.imgsz,
            retina_masks=pipeline_config.retina_masks,
            max_det=pipeline_config.max_det,
            verbose=False,
        )[0]
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

        detections: list[SegmentationDetection] = []
        masks: list[np.ndarray] = []
        for det_index in range(min(len(xyxy), len(masks_data))):
            mask = (masks_data[det_index] > 0.5).astype(np.uint8)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) == 0:
                continue
            detection, saved_mask = make_detection(
                frame=frame,
                det_index=len(detections),
                semantic_class="tomato",
                raw_class_name=_class_name(self.names, int(cls[det_index])),
                confidence=float(conf[det_index]),
                bbox_xyxy=[float(v) for v in xyxy[det_index].tolist()],
                mask=mask,
                backend=self.name,
                prompt_used=None,
                mask_dir=mask_dir,
                out_dir=out_dir,
            )
            detections.append(detection)
            masks.append(saved_mask)
        return detections, masks
