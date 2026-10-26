from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .io_aria import load_frames_metadata
from .sam_backend import SamBackend
from .task_prompts import prompts_for_task, target_class_for_group
from .visualization import save_sam_overlay


def _load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _save_mask_npz(mask: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=np.asarray(mask, dtype=bool))


def load_exemplar_config(path: str | Path | None, task: str) -> dict[str, Any]:
    if path is None:
        return {"positive_exemplar_crops": [], "positive_exemplar_boxes": []}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("pyyaml is required to load --exemplar-config.") from exc
    with Path(path).open("r") as handle:
        data = yaml.safe_load(handle) or {}
    task_data = data.get(task, {})
    return {
        "positive_exemplar_crops": list(task_data.get("positive_exemplar_crops", [])),
        "positive_exemplar_boxes": list(task_data.get("positive_exemplar_boxes", [])),
    }


def _load_exemplar_images(exemplar_config: dict[str, Any]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for path in exemplar_config.get("positive_exemplar_crops", []):
        crops.append(_load_image(path))
    return crops


def _load_exemplar_boxes(exemplar_config: dict[str, Any]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for item in exemplar_config.get("positive_exemplar_boxes", []):
        if isinstance(item, dict) and "box_xyxy" in item:
            boxes.append([float(x) for x in item["box_xyxy"]])
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            boxes.append([float(x) for x in item])
    return boxes


def run_sam_inference_for_split(
    frames_json: str | Path,
    split: str,
    task: str,
    backend: SamBackend,
    config: PipelineConfig,
    out_dir: str | Path,
    use_aux_prompts: bool = False,
    exemplar_config: dict[str, Any] | None = None,
    save_visualizations: bool = True,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    if config.image_size is not None:
        print(
            "Warning: config.image_size is ignored in v1 so masks remain aligned with raw Aria "
            "RGB calibration."
        )

    frames = load_frames_metadata(frames_json)
    prompt_pairs = prompts_for_task(task, use_auxiliary=use_aux_prompts)
    primary_prompts = [prompt for prompt, group in prompt_pairs if group == "primary"]
    auxiliary_prompts = [prompt for prompt, group in prompt_pairs if group == "auxiliary"]
    prompt_to_group = {prompt: group for prompt, group in prompt_pairs}
    masks_dir = Path(out_dir) / "sam_predictions" / "masks"
    vis_dir = Path(out_dir) / "visualizations" / f"sam_{split}_raw"
    aux_vis_dir = Path(out_dir) / "visualizations" / f"sam_{split}_auxiliary"
    all_vis_dir = Path(out_dir) / "visualizations" / f"sam_{split}_all"
    prediction_path = Path(out_dir) / "sam_predictions" / f"{split}_predictions.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    exemplar_config = exemplar_config or {"positive_exemplar_crops": [], "positive_exemplar_boxes": []}
    exemplar_images = _load_exemplar_images(exemplar_config)
    exemplar_boxes = _load_exemplar_boxes(exemplar_config)
    if (exemplar_images or exemplar_boxes) and not backend.supports_exemplars:
        print(
            f"Warning: backend {backend.name} does not support exemplar prompts in this adapter; "
            "continuing with text-only prompts."
        )
        exemplar_images = []
        exemplar_boxes = []

    predictions: list[dict[str, Any]] = []
    stats = {"frames": len(frames), "raw_masks": 0, "per_prompt_stats": {}}
    batch_size = max(1, int(config.sam_batch_size))
    for batch_start in range(0, len(frames), batch_size):
        batch_frames = frames[batch_start : batch_start + batch_size]
        batch_images = [_load_image(frame["image_path"]) for frame in batch_frames]
        batch_primary_masks: list[list[Any]] = [[] for _ in batch_frames]
        batch_auxiliary_masks: list[list[Any]] = [[] for _ in batch_frames]
        if primary_prompts:
            batch_primary_masks = backend.predict_images(
                images=batch_images,
                text_prompts=primary_prompts,
                exemplar_boxes=exemplar_boxes or None,
                exemplar_images=exemplar_images or None,
                conf_thres=config.score_threshold_for_group("primary", task),
            )
        if auxiliary_prompts:
            batch_auxiliary_masks = backend.predict_images(
                images=batch_images,
                text_prompts=auxiliary_prompts,
                exemplar_boxes=exemplar_boxes or None,
                exemplar_images=exemplar_images or None,
                conf_thres=config.score_threshold_for_group("auxiliary", task),
            )

        for batch_offset, (frame, image) in enumerate(zip(batch_frames, batch_images)):
            frame_idx = batch_start + batch_offset
            sam_masks = []
            sam_masks.extend(batch_primary_masks[batch_offset])
            sam_masks.extend(batch_auxiliary_masks[batch_offset])
            frame_masks: list[dict[str, Any]] = []
            primary_overlay_masks: list[np.ndarray] = []
            primary_overlay_labels: list[str] = []
            auxiliary_overlay_masks: list[np.ndarray] = []
            auxiliary_overlay_labels: list[str] = []
            all_overlay_masks: list[np.ndarray] = []
            all_overlay_labels: list[str] = []
            for mask_idx, sam_mask in enumerate(sam_masks):
                prompt = sam_mask.prompt
                semantic_group = prompt_to_group.get(prompt, "primary")
                mask_id = f"{frame['frame_id']}_mask_{mask_idx:03d}"
                mask_path = masks_dir / f"{mask_id}.npz"
                _save_mask_npz(sam_mask.mask, mask_path)
                area_frac = float(sam_mask.area_px / max(1, image.shape[0] * image.shape[1]))
                source = sam_mask.source
                if (exemplar_images or exemplar_boxes) and source == "text":
                    source = "text+exemplar"
                record = {
                    "mask_id": mask_id,
                    "prompt": prompt,
                    "semantic_group": semantic_group,
                    "target_class": target_class_for_group(task, semantic_group),
                    "score": float(sam_mask.score),
                    "bbox_xyxy": [float(x) for x in sam_mask.bbox_xyxy],
                    "area_px": int(sam_mask.area_px),
                    "area_frac": area_frac,
                    "mask_path": str(mask_path),
                    "source": source,
                    "raw_class_name": sam_mask.raw_class_name,
                }
                frame_masks.append(record)
                label = f"{prompt} {sam_mask.score:.2f}"
                if save_visualizations:
                    if semantic_group == "auxiliary":
                        auxiliary_overlay_masks.append(sam_mask.mask)
                        auxiliary_overlay_labels.append(label)
                    else:
                        primary_overlay_masks.append(sam_mask.mask)
                        primary_overlay_labels.append(label)
                    all_overlay_masks.append(sam_mask.mask)
                    all_overlay_labels.append(f"{semantic_group}:{label}")
                prompt_stats = stats["per_prompt_stats"].setdefault(prompt, {"raw_masks": 0})
                prompt_stats["raw_masks"] += 1

            stats["raw_masks"] += len(frame_masks)
            frame_pred = {
                "frame_id": frame["frame_id"],
                "timestamp_ns": int(frame["timestamp_ns"]),
                "image_path": frame["image_path"],
                "width": int(frame.get("width", image.shape[1])),
                "height": int(frame.get("height", image.shape[0])),
                "task": task,
                "split": split,
                "masks": frame_masks,
            }
            predictions.append(frame_pred)
            if save_visualizations:
                save_sam_overlay(
                    image=image,
                    masks=primary_overlay_masks,
                    labels=primary_overlay_labels,
                    out_path=vis_dir / f"{frame['frame_id']}.jpg",
                )
                if auxiliary_prompts:
                    save_sam_overlay(
                        image=image,
                        masks=auxiliary_overlay_masks,
                        labels=auxiliary_overlay_labels,
                        out_path=aux_vis_dir / f"{frame['frame_id']}.jpg",
                    )
                    save_sam_overlay(
                        image=image,
                        masks=all_overlay_masks,
                        labels=all_overlay_labels,
                        out_path=all_vis_dir / f"{frame['frame_id']}.jpg",
                    )
            if (frame_idx + 1) % 25 == 0:
                print(f"SAM inference {split}: processed {frame_idx + 1}/{len(frames)} frames")

    with prediction_path.open("w") as handle:
        json.dump(predictions, handle, indent=2)
    return prediction_path, predictions, stats


def run_sam_inference(
    pre_frames_json: str | Path,
    post_frames_json: str | Path,
    task: str,
    backend: SamBackend,
    config: PipelineConfig,
    out_dir: str | Path,
    use_aux_prompts: bool = False,
    exemplar_config_path: str | Path | None = None,
    save_visualizations: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    exemplar_config = load_exemplar_config(exemplar_config_path, task)
    pre_path, _, pre_stats = run_sam_inference_for_split(
        frames_json=pre_frames_json,
        split="pre",
        task=task,
        backend=backend,
        config=config,
        out_dir=out_dir,
        use_aux_prompts=use_aux_prompts,
        exemplar_config=exemplar_config,
        save_visualizations=save_visualizations,
    )
    post_path, _, post_stats = run_sam_inference_for_split(
        frames_json=post_frames_json,
        split="post",
        task=task,
        backend=backend,
        config=config,
        out_dir=out_dir,
        use_aux_prompts=use_aux_prompts,
        exemplar_config=exemplar_config,
        save_visualizations=save_visualizations,
    )
    return pre_path, post_path, {"pre": pre_stats, "post": post_stats}
