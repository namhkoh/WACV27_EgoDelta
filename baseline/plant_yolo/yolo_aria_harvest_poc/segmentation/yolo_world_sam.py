"""YOLO-World boxes plus SAM/SAM2 box-prompted masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..task_config import TaskConfig
from .base import SegmentationDetection, make_detection, mask_to_bbox_xyxy
from .sam3_concept import _tensor_to_numpy


class YoloWorldSAMBackend:
    name = "yolo_world_sam"

    def __init__(
        self,
        yolo_world_model: str | Path | None = None,
        sam_model: str | Path | None = None,
    ) -> None:
        try:
            from ultralytics import SAM, YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError("yolo_world_sam requires ultralytics YOLO and SAM.") from exc
        self.detector = YOLO(str(yolo_world_model or "yolov8l-world.pt"))
        self.segmenter = SAM(str(sam_model or "sam2_b.pt"))

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        frame: dict[str, Any],
        task_config: TaskConfig,
        pipeline_config: PipelineConfig,
        mask_dir: Path,
        out_dir: Path,
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        try:
            self.detector.set_classes(task_config.text_prompts)
        except Exception:
            pass
        det_result = self.detector.predict(
            source=image_bgr,
            conf=task_config.min_2d_confidence,
            imgsz=pipeline_config.imgsz,
            max_det=pipeline_config.max_det,
            verbose=False,
        )[0]
        boxes = getattr(det_result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return [], []
        xyxy = _tensor_to_numpy(boxes.xyxy)
        conf = _tensor_to_numpy(getattr(boxes, "conf", np.ones((len(xyxy),))))
        cls = _tensor_to_numpy(getattr(boxes, "cls", np.zeros((len(xyxy),)))).astype(int)
        if len(xyxy) == 0:
            return [], []

        detections: list[SegmentationDetection] = []
        masks_out: list[np.ndarray] = []
        for idx, box in enumerate(xyxy):
            prompt = task_config.text_prompts[int(cls[idx])] if int(cls[idx]) < len(task_config.text_prompts) else None
            try:
                seg_result = self.segmenter.predict(
                    source=image_bgr,
                    bboxes=[box.tolist()],
                    imgsz=pipeline_config.imgsz,
                    verbose=False,
                )[0]
            except TypeError:
                seg_result = self.segmenter.predict(
                    source=image_bgr,
                    boxes=[box.tolist()],
                    imgsz=pipeline_config.imgsz,
                    verbose=False,
                )[0]
            mask = first_mask_from_result(seg_result, image_bgr.shape[:2])
            if mask is None:
                continue
            bbox = mask_to_bbox_xyxy(mask)
            detection, saved_mask = make_detection(
                frame=frame,
                det_index=len(detections),
                semantic_class=task_config.target_semantic_class,
                raw_class_name=prompt,
                confidence=float(conf[idx]),
                bbox_xyxy=bbox,
                mask=mask,
                backend=self.name,
                prompt_used=prompt,
                mask_dir=mask_dir,
                out_dir=out_dir,
            )
            detections.append(detection)
            masks_out.append(saved_mask)
        return detections, masks_out


def first_mask_from_result(result: Any, image_shape: tuple[int, int]) -> np.ndarray | None:
    import cv2  # type: ignore

    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or getattr(masks_obj, "data", None) is None:
        return None
    masks_data = _tensor_to_numpy(masks_obj.data)
    if masks_data.ndim == 2:
        masks_data = masks_data[None, ...]
    if len(masks_data) == 0:
        return None
    height, width = image_shape
    mask = (masks_data[0] > 0.5).astype(np.uint8)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask
