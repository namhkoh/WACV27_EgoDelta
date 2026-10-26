"""SAM 3 text/concept segmentation backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..task_config import TaskConfig
from .base import SegmentationDetection, make_detection, mask_to_bbox_xyxy


class SAM3ConceptBackend:
    name = "sam3_concept"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = str(model_path or "sam3.pt")
        try:
            from ultralytics import SAM  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "SAM 3 concept segmentation requires an Ultralytics SAM build. "
                "If SAM 3 is unavailable, rerun with --segmentation-backend yolo_world_sam."
            ) from exc
        try:
            self.model = SAM(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load SAM 3 model '{self.model_path}'. Provide --sam3-model "
                "or use --segmentation-backend yolo_world_sam."
            ) from exc

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        frame: dict[str, Any],
        task_config: TaskConfig,
        pipeline_config: PipelineConfig,
        mask_dir: Path,
        out_dir: Path,
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        detections: list[SegmentationDetection] = []
        masks_out: list[np.ndarray] = []
        for prompt in task_config.text_prompts:
            result = self._predict_prompt(image_bgr, prompt, pipeline_config)
            prompt_dets, prompt_masks = self._result_to_detections(
                result=result,
                frame=frame,
                task_config=task_config,
                prompt=prompt,
                mask_dir=mask_dir,
                out_dir=out_dir,
                start_index=len(detections),
                image_shape=image_bgr.shape[:2],
            )
            detections.extend(prompt_dets)
            masks_out.extend(prompt_masks)
        return detections, masks_out

    def _predict_prompt(self, image_bgr: np.ndarray, prompt: str, pipeline_config: PipelineConfig) -> Any:
        errors: list[str] = []
        attempts = [
            {"texts": [prompt]},
            {"text": prompt},
            {"prompts": [prompt]},
            {"labels": [prompt]},
        ]
        for kwargs in attempts:
            try:
                return self.model.predict(
                    source=image_bgr,
                    imgsz=pipeline_config.imgsz,
                    verbose=False,
                    **kwargs,
                )[0]
            except Exception as exc:
                errors.append(str(exc))
                continue
        raise RuntimeError(
            "The installed Ultralytics SAM backend did not accept text prompts. "
            "Use --segmentation-backend yolo_world_sam. Last errors: "
            + " | ".join(errors[-2:])
        )

    def _result_to_detections(
        self,
        result: Any,
        frame: dict[str, Any],
        task_config: TaskConfig,
        prompt: str,
        mask_dir: Path,
        out_dir: Path,
        start_index: int,
        image_shape: tuple[int, int],
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        import cv2  # type: ignore

        height, width = image_shape
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or getattr(masks_obj, "data", None) is None:
            return [], []
        masks_data = _tensor_to_numpy(masks_obj.data)
        if masks_data.ndim == 2:
            masks_data = masks_data[None, ...]
        boxes = getattr(result, "boxes", None)
        conf = None
        xyxy = None
        if boxes is not None:
            conf = _tensor_to_numpy(getattr(boxes, "conf", np.ones((len(masks_data),))))
            xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", np.zeros((len(masks_data), 4))))

        detections: list[SegmentationDetection] = []
        masks: list[np.ndarray] = []
        for idx, raw_mask in enumerate(masks_data):
            mask = (raw_mask > 0.5).astype(np.uint8)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) == 0:
                continue
            bbox = mask_to_bbox_xyxy(mask) if xyxy is None else [float(v) for v in xyxy[idx].tolist()]
            confidence = 0.5 if conf is None or idx >= len(conf) else float(conf[idx])
            detection, saved_mask = make_detection(
                frame=frame,
                det_index=start_index + len(detections),
                semantic_class=task_config.target_semantic_class,
                raw_class_name=prompt,
                confidence=confidence,
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
