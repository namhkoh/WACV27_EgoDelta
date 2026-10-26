"""Peduncle/stem model inference for cut-point estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, relpath, should_write_visualization


def run_peduncle_for_split(
    split: str,
    frames: list[dict[str, Any]],
    model_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Run an optional peduncle model on frames.

    The model may be an Ultralytics segmentation model, pose/keypoint model, or
    a model that outputs both masks and keypoints.
    """

    try:
        import cv2  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("Peduncle inference requires ultralytics and opencv-python.") from exc

    out = Path(out_dir)
    pred_dir = out / "peduncle_predictions"
    mask_dir = pred_dir / "masks"
    vis_dir = out / "visualizations" / f"peduncle_{split}"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    predictions: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(tqdm(frames, desc=f"Peduncle {split}", unit="frame")):
        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image for peduncle inference: {frame['image_path']}")
        height, width = image.shape[:2]
        result = model.predict(
            source=image,
            conf=config.peduncle_conf_thres,
            imgsz=config.peduncle_imgsz,
            retina_masks=True,
            verbose=False,
        )[0]
        detections, masks = _result_to_peduncle_predictions(
            result=result,
            names=names,
            frame=frame,
            width=width,
            height=height,
            mask_dir=mask_dir,
            out_dir=out,
            keypoint_names=config.peduncle_keypoint_names,
        )
        predictions.append(
            {
                "frame_id": frame["frame_id"],
                "timestamp_ns": int(frame["timestamp_ns"]),
                "image_path": frame["image_path"],
                "detections": detections,
            }
        )
        if should_write_visualization(frame_index, config.visualization_stride):
            overlay = draw_peduncle_overlay(image, detections, masks)
            cv2.imwrite(str(vis_dir / f"{frame['frame_id']}.jpg"), overlay)

    output_path = pred_dir / f"{split}_peduncle_predictions.json"
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions


def _result_to_peduncle_predictions(
    result: Any,
    names: dict[int, str] | dict[str, str],
    frame: dict[str, Any],
    width: int,
    height: int,
    mask_dir: Path,
    out_dir: Path,
    keypoint_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], []

    xyxy = _tensor_to_numpy(boxes.xyxy)
    conf = _tensor_to_numpy(boxes.conf)
    cls = _tensor_to_numpy(boxes.cls).astype(int)
    masks_data = _extract_masks(result)
    keypoints_data = _extract_keypoints(result)

    detections: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for det_index in range(len(xyxy)):
        det_id = f"{frame['frame_id']}_ped_{len(detections):03d}"
        det_mask = np.zeros((height, width), dtype=np.uint8)
        detection: dict[str, Any] = {
            "det_id": det_id,
            "frame_id": frame["frame_id"],
            "class_id": int(cls[det_index]),
            "class_name_raw": _class_name(names, int(cls[det_index])),
            "semantic_class": "peduncle",
            "confidence": float(conf[det_index]),
            "bbox_xyxy": [float(v) for v in xyxy[det_index].tolist()],
            "mask": None,
            "mask_path": None,
            "mask_area_px": 0,
            "keypoints_2d": [],
        }

        if masks_data is not None and det_index < len(masks_data):
            mask = (masks_data[det_index] > 0.5).astype(np.uint8)
            if mask.shape[:2] != (height, width):
                import cv2  # type: ignore

                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) > 0:
                mask_path = mask_dir / f"{det_id}.npz"
                np.savez_compressed(mask_path, mask=mask.astype(bool))
                detection["mask"] = relpath(mask_path, out_dir)
                detection["mask_path"] = relpath(mask_path, out_dir)
                detection["mask_area_px"] = int(mask.sum())
                det_mask = mask

        if keypoints_data is not None and det_index < len(keypoints_data):
            detection["keypoints_2d"] = _format_keypoints(
                keypoints_data[det_index],
                keypoint_names=keypoint_names,
                width=width,
                height=height,
            )

        if detection["mask_path"] is not None or detection["keypoints_2d"]:
            detections.append(detection)
            masks.append(det_mask)
    return detections, masks


def draw_peduncle_overlay(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    masks: list[np.ndarray],
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    for det, mask in zip(detections, masks):
        if mask.size and mask.sum() > 0:
            mask_u8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 180, 255), 2)
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)
        for keypoint in det.get("keypoints_2d", []):
            if keypoint.get("visible", True):
                xy = keypoint["xy"]
                cv2.circle(overlay, (int(round(xy[0])), int(round(xy[1]))), 5, (255, 80, 0), -1)
                cv2.putText(
                    overlay,
                    keypoint["name"],
                    (int(round(xy[0])) + 6, int(round(xy[1])) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 80, 0),
                    1,
                    cv2.LINE_AA,
                )
    return overlay


def _extract_masks(result: Any) -> np.ndarray | None:
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or getattr(masks_obj, "data", None) is None:
        return None
    masks = _tensor_to_numpy(masks_obj.data)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks


def _extract_keypoints(result: Any) -> np.ndarray | None:
    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None:
        return None
    data = getattr(keypoints_obj, "data", None)
    if data is None:
        xy = getattr(keypoints_obj, "xy", None)
        conf = getattr(keypoints_obj, "conf", None)
        if xy is None:
            return None
        xy_np = _tensor_to_numpy(xy)
        if conf is None:
            conf_np = np.ones(xy_np.shape[:2] + (1,), dtype=float)
        else:
            conf_np = _tensor_to_numpy(conf)[..., None]
        return np.concatenate([xy_np, conf_np], axis=2)
    keypoints = _tensor_to_numpy(data)
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    return keypoints


def _format_keypoints(
    keypoints: np.ndarray,
    keypoint_names: tuple[str, ...],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    formatted = []
    for idx, item in enumerate(np.asarray(keypoints, dtype=float).reshape(-1, keypoints.shape[-1])):
        if len(item) < 2:
            continue
        x, y = float(item[0]), float(item[1])
        conf = float(item[2]) if len(item) >= 3 else 1.0
        visible = conf > 0.0 and 0 <= x < width and 0 <= y < height
        name = keypoint_names[idx] if idx < len(keypoint_names) else f"kp_{idx:02d}"
        formatted.append(
            {
                "index": idx,
                "name": name,
                "xy": [x, y],
                "confidence": conf,
                "visible": bool(visible),
            }
        )
    return formatted


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
