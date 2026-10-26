"""Common segmentation backend schema and runners."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm

from ..config import PipelineConfig, relpath, should_write_visualization
from ..task_config import TaskConfig


@dataclass
class SegmentationDetection:
    frame_id: str
    timestamp_ns: int
    det_id: str
    semantic_class: str
    raw_class_name: str | None
    confidence: float
    bbox_xyxy: list[float]
    mask_path: str
    mask_area_px: int
    backend: str
    prompt_used: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mask"] = self.mask_path
        return data


class SegmentationBackend(Protocol):
    name: str

    def predict_frame(
        self,
        image_bgr: np.ndarray,
        frame: dict[str, Any],
        task_config: TaskConfig,
        pipeline_config: PipelineConfig,
        mask_dir: Path,
        out_dir: Path,
    ) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
        ...


def run_segmentation_for_split(
    split: str,
    frames: list[dict[str, Any]],
    backend: SegmentationBackend,
    task_config: TaskConfig,
    pipeline_config: PipelineConfig,
    out_dir: str | Path,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for segmentation overlays.") from exc

    out = Path(out_dir)
    pred_dir = out / "segmentation_predictions" / task_config.task_name
    mask_dir = pred_dir / "masks"
    rejected_dir = pred_dir / "rejected_masks"
    vis_dir = out / "visualizations" / task_config.task_name / f"prompt_seg_{split}"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    batch_size = max(1, int(pipeline_config.segmentation_batch_size))
    supports_batch = hasattr(backend, "predict_batch")
    with tqdm(total=len(frames), desc=f"{backend.name} {split}", unit="frame") as progress:
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            batch_images = []
            for frame in batch_frames:
                image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not read image for segmentation: {frame['image_path']}")
                batch_images.append(image)

            if batch_size > 1 and supports_batch:
                batch_outputs = backend.predict_batch(  # type: ignore[attr-defined]
                    images_bgr=batch_images,
                    frames=batch_frames,
                    task_config=task_config,
                    pipeline_config=pipeline_config,
                    mask_dir=mask_dir,
                    out_dir=out,
                )
            else:
                batch_outputs = [
                    backend.predict_frame(
                        image_bgr=image,
                        frame=frame,
                        task_config=task_config,
                        pipeline_config=pipeline_config,
                        mask_dir=mask_dir,
                        out_dir=out,
                    )
                    for image, frame in zip(batch_images, batch_frames)
                ]

            for offset, (frame, image, output) in enumerate(zip(batch_frames, batch_images, batch_outputs)):
                detections, masks = output
                kept, kept_masks, rejected_frame = filter_and_nms_detections(
                    detections=detections,
                    masks=masks,
                    image_shape=image.shape[:2],
                    task_config=task_config,
                    image_bgr=image,
                )
                for item in rejected_frame:
                    item["frame_id"] = frame["frame_id"]
                rejected.extend(rejected_frame)

                prediction = {
                    "frame_id": frame["frame_id"],
                    "timestamp_ns": int(frame["timestamp_ns"]),
                    "image_path": frame["image_path"],
                    "detections": [det.to_dict() for det in kept],
                }
                predictions.append(prediction)
                if should_write_visualization(start + offset, pipeline_config.visualization_stride):
                    overlay = draw_segmentation_overlay(image, [det.to_dict() for det in kept], kept_masks)
                    cv2.imwrite(str(vis_dir / f"{frame['frame_id']}.jpg"), overlay)
            progress.update(len(batch_frames))

    output_path = pred_dir / f"{split}_predictions.json"
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    rejected_path = pred_dir / f"{split}_rejected_masks.json"
    rejected_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    return predictions


def make_detection(
    *,
    frame: dict[str, Any],
    det_index: int,
    semantic_class: str,
    raw_class_name: str | None,
    confidence: float,
    bbox_xyxy: list[float],
    mask: np.ndarray,
    backend: str,
    prompt_used: str | None,
    mask_dir: Path,
    out_dir: Path,
    det_suffix: str | None = None,
) -> tuple[SegmentationDetection, np.ndarray]:
    det_id = f"{frame['frame_id']}_det_{det_index:03d}" if det_suffix is None else det_suffix
    mask = np.asarray(mask > 0, dtype=bool)
    mask_path = mask_dir / f"{det_id}.npz"
    np.savez_compressed(mask_path, mask=mask)
    detection = SegmentationDetection(
        frame_id=frame["frame_id"],
        timestamp_ns=int(frame["timestamp_ns"]),
        det_id=det_id,
        semantic_class=semantic_class,
        raw_class_name=raw_class_name,
        confidence=float(confidence),
        bbox_xyxy=[float(v) for v in bbox_xyxy],
        mask_path=relpath(mask_path, out_dir),
        mask_area_px=int(mask.sum()),
        backend=backend,
        prompt_used=prompt_used,
    )
    return detection, mask


def detections_from_masks(
    *,
    masks: list[np.ndarray],
    frame: dict[str, Any],
    semantic_class: str,
    backend: str,
    prompt_used: str | None,
    mask_dir: Path,
    out_dir: Path,
    confidence: float = 0.5,
    raw_class_name: str | None = None,
) -> tuple[list[SegmentationDetection], list[np.ndarray]]:
    detections: list[SegmentationDetection] = []
    kept_masks: list[np.ndarray] = []
    for mask in masks:
        mask_bool = np.asarray(mask > 0, dtype=bool)
        if not mask_bool.any():
            continue
        bbox = mask_to_bbox_xyxy(mask_bool)
        det, saved_mask = make_detection(
            frame=frame,
            det_index=len(detections),
            semantic_class=semantic_class,
            raw_class_name=raw_class_name,
            confidence=confidence,
            bbox_xyxy=bbox,
            mask=mask_bool,
            backend=backend,
            prompt_used=prompt_used,
            mask_dir=mask_dir,
            out_dir=out_dir,
        )
        detections.append(det)
        kept_masks.append(saved_mask)
    return detections, kept_masks


def refine_mask_for_task(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    task_config: TaskConfig,
    prompt_used: str | None = None,
) -> np.ndarray:
    """Refine raw open-vocabulary masks before they are saved.

    YOLOE often captures the full leaf/petiole cluster but includes black trellis
    pipes or strings crossing the cluster. For deleafing, keep green/yellow plant
    pixels inside the predicted mask and drop dark/gray support structures.
    """
    mask_bool = np.asarray(mask > 0, dtype=bool)
    if task_config.task_name != "deleaf" or not mask_bool.any():
        return mask_bool

    try:
        import cv2  # type: ignore
    except Exception:
        return mask_bool

    image_bgr = np.asarray(image_bgr)
    if image_bgr.ndim != 3 or image_bgr.shape[:2] != mask_bool.shape:
        return mask_bool

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    green_plant = (hue >= 28) & (hue <= 98) & (sat >= 16) & (val >= 35)
    yellow_pale_plant = (hue >= 16) & (hue <= 55) & (sat >= 10) & (val >= 48)
    not_tomato_red = ~(((hue <= 18) | (hue >= 165)) & (sat >= 50) & (val >= 45))
    not_dark_support = ~((gray < 62) & (sat < 100))
    plant_pixels = (green_plant | yellow_pale_plant) & not_tomato_red & not_dark_support

    refined = mask_bool & plant_pixels
    if not refined.any():
        return mask_bool

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined_u8 = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    refined_u8 = refined_u8 & mask_bool.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_u8, connectivity=8)
    cleaned = np.zeros_like(refined_u8, dtype=bool)
    raw_area = int(mask_bool.sum())
    component_min_area = 30 if raw_area < 2500 else 60
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= component_min_area:
            cleaned[labels == label] = True

    cleaned_area = int(cleaned.sum())
    is_cluster_prompt = _is_deleaf_leaf_cluster_prompt_text(prompt_used)
    min_replace_area = 350 if is_cluster_prompt else int(task_config.min_mask_area_px)
    if cleaned_area >= min_replace_area:
        return cleaned
    return mask_bool


def filter_and_nms_detections(
    detections: list[SegmentationDetection],
    masks: list[np.ndarray],
    image_shape: tuple[int, int],
    task_config: TaskConfig,
    image_bgr: np.ndarray | None = None,
    iou_threshold: float = 0.6,
) -> tuple[list[SegmentationDetection], list[np.ndarray], list[dict[str, Any]]]:
    height, width = image_shape
    candidates: list[tuple[SegmentationDetection, np.ndarray]] = []
    rejected: list[dict[str, Any]] = []
    for det, mask in zip(detections, masks):
        reason = rejection_reason(det, mask, height, width, task_config, image_bgr=image_bgr)
        if reason:
            rejected.append(
                {
                    "det_id": det.det_id,
                    "prompt_used": det.prompt_used,
                    "confidence": det.confidence,
                    "mask_area_px": det.mask_area_px,
                    "reason": reason,
                }
            )
        else:
            candidates.append((det, mask))

    order = sorted(
        range(len(candidates)),
        key=lambda idx: candidates[idx][0].confidence,
        reverse=True,
    )
    keep_indices: list[int] = []
    for idx in order:
        det, mask = candidates[idx]
        duplicate = False
        for kept_idx in keep_indices:
            _, kept_mask = candidates[kept_idx]
            if mask_iou(mask, kept_mask) > iou_threshold:
                duplicate = True
                rejected.append(
                    {
                        "det_id": det.det_id,
                        "prompt_used": det.prompt_used,
                        "confidence": det.confidence,
                        "mask_area_px": det.mask_area_px,
                        "reason": "duplicate_mask_nms",
                    }
                )
                break
        if not duplicate:
            keep_indices.append(idx)

    kept = [candidates[idx][0] for idx in keep_indices]
    kept_masks = [candidates[idx][1] for idx in keep_indices]
    return kept, kept_masks, rejected


def rejection_reason(
    det: SegmentationDetection,
    mask: np.ndarray,
    height: int,
    width: int,
    task_config: TaskConfig,
    image_bgr: np.ndarray | None = None,
) -> str | None:
    if _is_negative_prompt_detection(det, task_config):
        return "negative_prompt"
    if det.confidence < task_config.min_2d_confidence:
        return "low_confidence"
    if det.mask_area_px < task_config.min_mask_area_px:
        return "too_small"
    if task_config.max_mask_area_px is not None and det.mask_area_px > task_config.max_mask_area_px:
        return "too_large"
    visual_reason = deleaf_visual_rejection_reason(det, mask, task_config, image_bgr)
    if visual_reason:
        return visual_reason
    if (
        task_config.task_name == "deleaf"
        and _requires_deleaf_elongatedness(det)
        and not _is_deleaf_leaf_cluster_prompt(det)
        and elongatedness(det.bbox_xyxy) < 1.5
    ):
        return "not_elongated"
    if task_config.task_name == "deleaf":
        shape_reason = deleaf_shape_rejection_reason(det)
        if shape_reason:
            return shape_reason
    return None


def deleaf_shape_rejection_reason(det: SegmentationDetection) -> str | None:
    x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    bbox_area = width * height
    fill_ratio = float(det.mask_area_px) / bbox_area
    aspect = max(width, height) / max(min(width, height), 1.0)
    leaf_cluster_prompt = _is_deleaf_leaf_cluster_prompt(det)

    if det.mask_area_px > (45000 if leaf_cluster_prompt else 18000):
        return "too_large_deleaf_target"
    if leaf_cluster_prompt and det.mask_area_px < 2000:
        return "too_small_leaf_cluster"
    if bbox_area > 90000 and fill_ratio > 0.18:
        return "likely_large_background_mask"
    if aspect < 1.65 and not leaf_cluster_prompt:
        return "not_elongated"
    if max(width, height) >= 200 and aspect >= 4.5 and det.mask_area_px >= 2000 and fill_ratio < 0.50:
        return "likely_long_support_or_pipe"
    if fill_ratio > 0.78 and det.mask_area_px > 800 and leaf_cluster_prompt:
        return "likely_leaf_blade_or_blob"
    if fill_ratio > 0.62 and det.mask_area_px > 800 and aspect < 2.4 and not leaf_cluster_prompt:
        return "likely_leaf_blade_or_blob"
    if fill_ratio > 0.50 and det.mask_area_px > 2500 and aspect < 2.1:
        return "likely_leaf_blade_or_blob"
    return None


def deleaf_visual_rejection_reason(
    det: SegmentationDetection,
    mask: np.ndarray,
    task_config: TaskConfig,
    image_bgr: np.ndarray | None,
) -> str | None:
    if task_config.task_name != "deleaf" or image_bgr is None:
        return None
    mask_bool = np.asarray(mask > 0, dtype=bool)
    if not mask_bool.any():
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    image_bgr = np.asarray(image_bgr)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv_pixels = hsv[mask_bool]
    gray_pixels = gray[mask_bool]
    if len(gray_pixels) == 0:
        return None

    median_value = float(np.median(hsv_pixels[:, 2]))
    mean_value = float(np.mean(hsv_pixels[:, 2]))
    mean_saturation = float(np.mean(hsv_pixels[:, 1]))
    dark_fraction = float(np.mean(gray_pixels < 55))
    low_value_fraction = float(np.mean(hsv_pixels[:, 2] < 65))
    x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)
    bbox_aspect = max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1.0)
    bbox_long_side = max(bbox_width, bbox_height)
    green_fraction = float(
        np.mean(
            (hsv_pixels[:, 0] >= 30)
            & (hsv_pixels[:, 0] <= 95)
            & (hsv_pixels[:, 1] >= 25)
            & (hsv_pixels[:, 2] >= 35)
        )
    )
    yellow_green_fraction = float(
        np.mean(
            (hsv_pixels[:, 0] >= 20)
            & (hsv_pixels[:, 0] <= 50)
            & (hsv_pixels[:, 1] >= 25)
            & (hsv_pixels[:, 2] >= 45)
        )
    )
    red_orange_fraction = float(
        np.mean(
            (
                (hsv_pixels[:, 0] <= 25)
                | (hsv_pixels[:, 0] >= 165)
            )
            & (hsv_pixels[:, 1] >= 55)
            & (hsv_pixels[:, 2] >= 45)
        )
    )
    gray_dark_fraction = float(np.mean((gray_pixels < 75) & (hsv_pixels[:, 1] < 90)))

    if red_orange_fraction >= 0.28 and green_fraction < 0.65:
        return "likely_tomato_fruit"
    if dark_fraction >= 0.65 and median_value <= 60:
        return "likely_dark_pipe_or_frame"
    if low_value_fraction >= 0.82 and mean_value <= 58:
        return "likely_dark_pipe_or_frame"
    if gray_dark_fraction >= 0.70 and green_fraction < 0.65 and mean_saturation < 95:
        return "likely_dark_pipe_or_frame"
    if (
        det.mask_area_px >= 1500
        and dark_fraction >= 0.40
        and yellow_green_fraction < 0.12
        and mean_saturation < 75
        and mean_value < 95
    ):
        return "likely_green_support_or_pipe"
    if (
        det.mask_area_px >= 3000
        and bbox_aspect >= 3.0
        and yellow_green_fraction < 0.08
        and green_fraction < 0.55
    ):
        return "likely_long_support_or_glare"
    if (
        det.mask_area_px >= 3000
        and bbox_aspect >= 5.0
        and bbox_long_side >= 220
        and dark_fraction >= 0.55
        and mean_value < 75
    ):
        return "likely_long_support_or_glare"
    if det.mask_area_px >= 18000 and green_fraction < 0.55 and dark_fraction >= 0.35:
        return "likely_background_structure"
    return None


def _is_negative_prompt_detection(det: SegmentationDetection, task_config: TaskConfig) -> bool:
    prompt = (det.prompt_used or det.raw_class_name or "").strip().lower()
    negative_prompts = {item.strip().lower() for item in task_config.negative_prompts}
    return bool(prompt and prompt in negative_prompts)


def _requires_deleaf_elongatedness(det: SegmentationDetection) -> bool:
    prompt = (det.prompt_used or det.raw_class_name or "").lower()
    if any(
        token in prompt
        for token in ["petiole", "stem", "stalk", "branch", "rachis", "midrib", "central", "connecting"]
    ):
        return True
    if "leaf" in prompt:
        return False
    return True


def _is_deleaf_leaf_cluster_prompt(det: SegmentationDetection) -> bool:
    prompt = det.prompt_used or det.raw_class_name or ""
    return _is_deleaf_leaf_cluster_prompt_text(prompt)


def _is_deleaf_leaf_cluster_prompt_text(prompt: str | None) -> bool:
    prompt = (prompt or "").lower()
    cluster_terms = [
        "with leaves",
        "with leaflets",
        "leaf cluster",
        "compound tomato leaf branch",
        "tomato compound leaf branch",
    ]
    return any(term in prompt for term in cluster_terms)


def mask_to_bbox_xyxy(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a > 0, dtype=bool)
    b = np.asarray(mask_b > 0, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def elongatedness(bbox_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return float(max(w, h) / max(1.0, min(w, h)))


def count_border_touches(mask: np.ndarray, height: int, width: int) -> int:
    mask_bool = np.asarray(mask > 0, dtype=bool)
    return int(mask_bool[0, :].any()) + int(mask_bool[-1, :].any()) + int(
        mask_bool[:, 0].any()
    ) + int(mask_bool[:, width - 1].any())


def draw_segmentation_overlay(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    masks: list[np.ndarray],
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    for det, mask in zip(detections, masks):
        color = color_for_semantic(det.get("semantic_class", "target"))
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get('semantic_class')} {float(det.get('confidence', 0.0)):.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        tint = np.zeros_like(overlay)
        tint[mask_u8 > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.18, 0.0)
    return overlay


def color_for_semantic(semantic_class: str) -> tuple[int, int, int]:
    if semantic_class == "petiole":
        return (0, 150, 255)
    if semantic_class == "main_stem":
        return (190, 70, 190)
    return (0, 0, 255)
