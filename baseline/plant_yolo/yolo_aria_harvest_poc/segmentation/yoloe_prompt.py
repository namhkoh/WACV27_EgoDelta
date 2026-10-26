"""YOLOE text-prompted zero-shot instance segmentation backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..task_config import TaskConfig
from .base import SegmentationDetection, make_detection, mask_to_bbox_xyxy, refine_mask_for_task


class YOLOEPromptBackend:
    name = "yoloe_prompt"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = str(model_path or "yoloe-11s-seg.pt")
        try:
            from ultralytics import YOLOE  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YOLOE zero-shot segmentation requires an Ultralytics version with YOLOE support. "
                "Install/upgrade ultralytics and use a YOLOE segmentation checkpoint such as "
                "yoloe-11s-seg.pt or yoloe-26s-seg.pt."
            ) from exc
        self.model = YOLOE(self.model_path)
        self._classes: tuple[str, ...] | None = None

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        frame: dict[str, Any],
        task_config: TaskConfig,
        pipeline_config: PipelineConfig,
        mask_dir: Path,
        out_dir: Path,
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        return self.predict_batch(
            images_bgr=[image_bgr],
            frames=[frame],
            task_config=task_config,
            pipeline_config=pipeline_config,
            mask_dir=mask_dir,
            out_dir=out_dir,
        )[0]

    def predict_batch(
        self,
        images_bgr: list[np.ndarray],
        frames: list[dict[str, Any]],
        task_config: TaskConfig,
        pipeline_config: PipelineConfig,
        mask_dir: Path,
        out_dir: Path,
    ) -> list[tuple[list[SegmentationDetection], list[np.ndarray]]]:
        import cv2  # type: ignore

        positive_prompts = tuple(task_config.text_prompts)
        # YOLOE often routes true petiole-like structures into "negative" text classes.
        # Keep open-vocabulary inference positive-only and let downstream filters reject pipes/background.
        prompts = positive_prompts
        if not positive_prompts:
            return [([], []) for _ in frames]
        if prompts != self._classes:
            self.model.set_classes(list(prompts))
            self._classes = prompts

        results = self.model.predict(
            source=images_bgr,
            conf=task_config.min_2d_confidence,
            imgsz=pipeline_config.imgsz,
            retina_masks=pipeline_config.retina_masks,
            max_det=pipeline_config.max_det,
            batch=max(1, int(pipeline_config.segmentation_batch_size)),
            verbose=False,
        )

        batch_outputs: list[tuple[list[SegmentationDetection], list[np.ndarray]]] = []
        for result, image_bgr, frame in zip(results, images_bgr, frames):
            batch_outputs.append(
                self._detections_from_result(
                    result=result,
                    image_bgr=image_bgr,
                    frame=frame,
                    task_config=task_config,
                    mask_dir=mask_dir,
                    out_dir=out_dir,
                    prompts=prompts,
                    cv2_module=cv2,
                )
            )
        return batch_outputs

    def _detections_from_result(
        self,
        *,
        result: Any,
        image_bgr: np.ndarray,
        frame: dict[str, Any],
        task_config: TaskConfig,
        mask_dir: Path,
        out_dir: Path,
        prompts: tuple[str, ...],
        cv2_module: Any,
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        boxes = getattr(result, "boxes", None)
        masks_obj = getattr(result, "masks", None)
        if boxes is None or masks_obj is None or getattr(masks_obj, "data", None) is None:
            return [], []

        xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", np.zeros((0, 4))))
        conf = _tensor_to_numpy(getattr(boxes, "conf", np.zeros((len(xyxy),))))
        cls = _tensor_to_numpy(getattr(boxes, "cls", np.zeros((len(xyxy),)))).astype(int)
        masks_data = _tensor_to_numpy(masks_obj.data)
        if masks_data.ndim == 2:
            masks_data = masks_data[None, ...]

        height, width = image_bgr.shape[:2]
        detections: list[SegmentationDetection] = []
        masks: list[np.ndarray] = []
        for idx in range(min(len(xyxy), len(masks_data))):
            mask = (masks_data[idx] > 0.5).astype(np.uint8)
            if mask.shape[:2] != (height, width):
                mask = cv2_module.resize(mask, (width, height), interpolation=cv2_module.INTER_NEAREST)
            if int(mask.sum()) == 0:
                continue
            class_id = int(cls[idx])
            prompt = prompts[class_id] if 0 <= class_id < len(prompts) else f"class_{class_id}"
            mask = refine_mask_for_task(
                mask=mask,
                image_bgr=image_bgr,
                task_config=task_config,
                prompt_used=prompt,
            )
            if int(mask.sum()) == 0:
                continue
            bbox = mask_to_bbox_xyxy(mask)
            detection, saved_mask = make_detection(
                frame=frame,
                det_index=len(detections),
                semantic_class=task_config.target_semantic_class,
                raw_class_name=prompt,
                confidence=float(conf[idx]) if idx < len(conf) else 0.5,
                bbox_xyxy=bbox,
                mask=mask,
                backend=self.name,
                prompt_used=prompt,
                mask_dir=mask_dir,
                out_dir=out_dir,
            )
            detections.append(detection)
            masks.append(saved_mask)
        return detections, masks


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)
