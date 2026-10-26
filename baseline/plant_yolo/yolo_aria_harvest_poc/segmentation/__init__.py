"""Task segmentation backends."""

from __future__ import annotations

from typing import Any

from .base import SegmentationBackend, SegmentationDetection, run_segmentation_for_split


def make_segmentation_backend(name: str, **kwargs: Any) -> SegmentationBackend:
    if name == "yolo_tomato":
        from .yolo_tomato import YoloTomatoBackend

        return YoloTomatoBackend(model_path=kwargs.get("model_path"))
    if name == "sam3_concept":
        from .sam3_concept import SAM3ConceptBackend

        return SAM3ConceptBackend(model_path=kwargs.get("sam3_model"))
    if name == "yolo_world_sam":
        from .yolo_world_sam import YoloWorldSAMBackend

        return YoloWorldSAMBackend(
            yolo_world_model=kwargs.get("yolo_world_model"),
            sam_model=kwargs.get("sam_model"),
        )
    if name == "yoloe_prompt":
        from .yoloe_prompt import YOLOEPromptBackend

        return YOLOEPromptBackend(model_path=kwargs.get("yoloe_model"))
    raise ValueError(f"Unknown segmentation backend: {name}")


__all__ = [
    "SegmentationBackend",
    "SegmentationDetection",
    "make_segmentation_backend",
    "run_segmentation_for_split",
]
