from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .task_prompts import TASK_PROMPTS
from .visualization import save_sam_overlay


def load_mask(mask_record: dict[str, Any]) -> np.ndarray:
    if isinstance(mask_record.get("mask"), np.ndarray):
        return np.asarray(mask_record["mask"], dtype=bool)
    path = mask_record.get("mask_path")
    if path is None:
        raise KeyError("Mask record must contain either an in-memory 'mask' or 'mask_path'.")
    data = np.load(path)
    if "mask" in data:
        return np.asarray(data["mask"], dtype=bool)
    first_key = list(data.keys())[0]
    return np.asarray(data[first_key], dtype=bool)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=np.asarray(mask, dtype=bool))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        import cv2

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32), int(count - 1)
    except Exception:
        pass
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        return labels.astype(np.int32), int(count)
    except Exception:
        pass
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def cleanup_connected_components(
    mask: np.ndarray,
    task: str,
    config: PipelineConfig | None = None,
    semantic_group: str = "primary",
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    labels, count = _connected_components(mask)
    if count <= 1:
        return mask, {"num_components": int(count), "kept_components": int(count)}
    areas = np.array([(labels == idx).sum() for idx in range(1, count + 1)], dtype=int)
    largest_label = int(np.argmax(areas) + 1)
    if task == "deleafing":
        min_keep = max(5, int(0.15 * areas.max()))
        keep_labels = [idx + 1 for idx, area in enumerate(areas) if area >= min_keep]
        axis_rejected_labels: list[int] = []
        if (
            semantic_group == "primary"
            and config is not None
            and config.deleafing_primary_reject_vertical_axis
        ):
            retained_labels = []
            for label in keep_labels:
                component = labels == label
                component_stats = _mask_shape_stats(component)
                axis_alignment = _principal_axis_alignment(
                    component,
                    config.deleafing_primary_vertical_axis_image_direction,
                )
                if (
                    component_stats["thinness"]
                    <= config.deleafing_primary_vertical_axis_max_thinness
                    and axis_alignment >= config.deleafing_primary_vertical_axis_thres
                ):
                    axis_rejected_labels.append(label)
                else:
                    retained_labels.append(label)
            keep_labels = retained_labels
        cleaned = np.isin(labels, keep_labels)
    else:
        keep_labels = [largest_label]
        axis_rejected_labels = []
        cleaned = labels == largest_label
    return cleaned, {
        "num_components": int(count),
        "kept_components": int(len(keep_labels)),
        "largest_component_area_px": int(areas.max()),
        "axis_rejected_components": int(len(axis_rejected_labels)),
    }


def _mask_shape_stats(mask: np.ndarray) -> dict[str, float]:
    bbox = bbox_from_mask(mask)
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = float(mask.sum())
    thinness = min(width, height) / max(width, height)
    extent = area / max(1.0, width * height)
    return {
        "bbox_width": width,
        "bbox_height": height,
        "thinness": float(thinness),
        "extent": float(extent),
    }


def _principal_axis_verticality(mask: np.ndarray) -> float:
    return _principal_axis_alignment(mask, "image_vertical")


def _principal_axis_alignment(mask: np.ndarray, image_direction: str = "image_vertical") -> float:
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    if len(coords) > 5000:
        step = max(1, len(coords) // 5000)
        coords = coords[::step]
    coords -= np.mean(coords, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(coords, full_matrices=False)
    axis = vh[0]
    axis = axis / max(1e-12, np.linalg.norm(axis))
    if image_direction == "image_horizontal":
        return float(abs(axis[0]))
    return float(abs(axis[1]))


def _touches_border(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _active_prompt_reject_reason(record: dict[str, Any], task: str) -> str | None:
    semantic_group = record.get("semantic_group", "primary")
    prompt = str(record.get("prompt", "")).lower()
    if task == "deleafing" and semantic_group == "primary":
        active_prompts = {
            str(prompt).lower() for prompt in TASK_PROMPTS.get("deleafing", {}).get("primary", [])
        }
        if prompt not in active_prompts:
            return "deleafing_prompt_not_active"
    if task == "deleafing" and semantic_group == "auxiliary":
        active_prompts = {
            str(prompt).lower()
            for prompt in TASK_PROMPTS.get("deleafing", {}).get("auxiliary", [])
        }
        if prompt not in active_prompts:
            return "deleafing_aux_prompt_not_active"
    return None


def _heuristic_reject_reason(
    mask: np.ndarray,
    record: dict[str, Any],
    task: str,
    config: PipelineConfig,
    area_frac: float,
) -> str | None:
    semantic_group = record.get("semantic_group", "primary")
    prompt = str(record.get("prompt", "")).lower()
    stats = _mask_shape_stats(mask)

    prompt_reason = _active_prompt_reject_reason(record, task)
    if prompt_reason is not None:
        return prompt_reason

    if task == "harvesting" and semantic_group == "primary":
        if stats["thinness"] < 0.12 and area_frac > 0.002:
            return "tomato_primary_too_thin"
        if _touches_border(mask) and area_frac > 0.05:
            return "tomato_large_border_touch"
    elif task == "deleafing":
        if semantic_group == "primary":
            if area_frac > config.deleafing_max_mask_area_frac:
                return "deleafing_primary_too_large"
            if config.deleafing_primary_reject_vertical_axis:
                axis_alignment = _principal_axis_alignment(
                    mask,
                    config.deleafing_primary_vertical_axis_image_direction,
                )
                if (
                    stats["thinness"] <= config.deleafing_primary_vertical_axis_max_thinness
                    and axis_alignment >= config.deleafing_primary_vertical_axis_thres
                ):
                    return "deleafing_primary_vertical_axis_like_main_stem"
            if (
                stats["bbox_width"] > config.deleafing_long_horizontal_width_frac * mask.shape[1]
                and stats["bbox_width"] / max(1.0, stats["bbox_height"])
                > config.deleafing_long_horizontal_aspect_ratio
            ):
                return "deleafing_primary_long_horizontal_support_like"
            long_side = max(stats["bbox_width"], stats["bbox_height"])
            short_side = max(1.0, min(stats["bbox_width"], stats["bbox_height"]))
            image_long_side = max(mask.shape)
            if (
                long_side > config.deleafing_pipe_like_long_side_frac * image_long_side
                and long_side / short_side > config.deleafing_pipe_like_aspect_ratio
            ):
                return "deleafing_primary_long_pipe_like"
            if (
                int(mask.sum()) >= config.deleafing_primary_leaf_blob_min_area_px
                and stats["thinness"] >= config.deleafing_primary_leaf_blob_thinness
                and stats["extent"] >= config.deleafing_primary_leaf_blob_extent
            ):
                return "deleafing_primary_compact_leaf_blob"
        if semantic_group == "auxiliary" and "stem" in prompt:
            if area_frac > min(0.20, config.max_mask_area_frac):
                return "main_stem_aux_too_large"
            if stats["thinness"] > 0.55 and area_frac > 0.003:
                return "main_stem_aux_not_elongated"
            image_height, image_width = mask.shape
            height_frac = stats["bbox_height"] / max(1.0, float(image_height))
            width_frac = stats["bbox_width"] / max(1.0, float(image_width))
            aspect_ratio = stats["bbox_height"] / max(1.0, stats["bbox_width"])
            if height_frac < config.deleafing_main_stem_min_height_frac:
                return "main_stem_aux_too_short"
            if width_frac > config.deleafing_main_stem_max_width_frac:
                return "main_stem_aux_too_wide"
            if aspect_ratio < config.deleafing_main_stem_min_aspect_ratio:
                return "main_stem_aux_not_tall_enough"
            verticality = _principal_axis_verticality(mask)
            if verticality < config.deleafing_main_stem_min_verticality:
                return "main_stem_aux_not_vertical"
        if "leaf" in prompt and "stem" not in prompt and area_frac > min(0.25, config.max_mask_area_frac):
            return "leaf_prompt_too_large"
    return None


def _record_score(record: dict[str, Any]) -> float:
    return float(record.get("score", record.get("score_2d", 0.0)))


def _component_reject_reason(
    component: np.ndarray,
    record: dict[str, Any],
    task: str,
    config: PipelineConfig,
    image_shape: tuple[int, int],
) -> str | None:
    area_px = int(component.sum())
    area_frac = float(area_px / max(1, image_shape[0] * image_shape[1]))
    min_area_px = config.min_mask_area_for_group(record.get("semantic_group", "primary"), task)
    if area_px < min_area_px:
        return "component_area_below_threshold"
    if area_frac > config.max_mask_area_frac:
        return "component_area_above_threshold"
    reason = _heuristic_reject_reason(component, record, task, config, area_frac)
    if reason is not None:
        return reason
    if task == "deleafing" and record.get("semantic_group", "primary") != "auxiliary":
        stats = _mask_shape_stats(component)
        if (
            stats["thinness"] >= config.deleafing_primary_leaf_blob_thinness
            and stats["extent"] >= config.deleafing_primary_leaf_blob_extent
        ):
            return "deleafing_primary_component_compact_leaf_blob"
    return None


def _final_filter_deleafing_primary_components(
    record: dict[str, Any],
    config: PipelineConfig,
    task: str,
    image_shape: tuple[int, int],
    stats: dict[str, Any],
) -> dict[str, Any] | None:
    if (
        task != "deleafing"
        or record.get("semantic_group", "primary") == "auxiliary"
        or not config.deleafing_primary_final_component_filter
    ):
        return record
    mask = np.asarray(record.get("_mask_array"), dtype=bool)
    labels, count = _connected_components(mask)
    if count <= 1:
        return record

    kept_labels: list[int] = []
    rejected_by_reason: dict[str, int] = {}
    component_areas = [(idx, int((labels == idx).sum())) for idx in range(1, count + 1)]
    for label, _ in component_areas:
        component = labels == label
        reason = _component_reject_reason(component, record, task, config, image_shape)
        if reason is None:
            kept_labels.append(label)
        else:
            rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1

    cleaned = np.isin(labels, kept_labels)
    min_area_px = config.min_mask_area_for_group(record.get("semantic_group", "primary"), task)
    if int(cleaned.sum()) < min_area_px:
        for reason, count_rejected in rejected_by_reason.items():
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + count_rejected
        stats["rejected"]["all_components_removed_by_final_filter"] = (
            stats["rejected"].get("all_components_removed_by_final_filter", 0) + 1
        )
        return None

    for reason, count_rejected in rejected_by_reason.items():
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + count_rejected
    record["_mask_array"] = cleaned
    record["area_px"] = int(cleaned.sum())
    record["area_frac"] = float(record["area_px"] / max(1, image_shape[0] * image_shape[1]))
    record["bbox_xyxy"] = bbox_from_mask(cleaned)
    cleanup_stats = dict(record.get("cleanup_stats", {}))
    cleanup_stats.update(
        {
            "final_component_filter_enabled": True,
            "final_num_components": int(count),
            "final_kept_components": int(len(kept_labels)),
            "final_rejected_components": int(count - len(kept_labels)),
            "final_component_reject_reasons": rejected_by_reason,
        }
    )
    record["cleanup_stats"] = cleanup_stats
    return record


def _deleafing_primary_rank(record: dict[str, Any]) -> float:
    score = _record_score(record)
    mask = record.get("_mask_array")
    if isinstance(mask, np.ndarray):
        stats = _mask_shape_stats(mask)
        elongated = 1.0 - float(stats["thinness"])
        sparse_box = 1.0 - min(1.0, float(stats["extent"]))
    else:
        bbox = record.get("bbox_xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = [float(x) for x in bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        elongated = 1.0 - min(width, height) / max(width, height)
        sparse_box = 0.5
    return score + 0.30 * elongated + 0.10 * sparse_box


def _prepare_record(
    record: dict[str, Any],
    mask: np.ndarray,
    image_shape: tuple[int, int],
    cleanup_stats: dict[str, Any],
    filtered_mask_dir: Path | None,
) -> dict[str, Any]:
    prepared = {k: v for k, v in record.items() if k != "mask"}
    area_px = int(mask.sum())
    prepared["area_px"] = area_px
    prepared["area_frac"] = float(area_px / max(1, image_shape[0] * image_shape[1]))
    prepared["bbox_xyxy"] = bbox_from_mask(mask)
    prepared["score"] = _record_score(record)
    prepared["source_prompts"] = list(record.get("source_prompts", [record.get("prompt", "")]))
    prepared["cleanup_stats"] = cleanup_stats
    if filtered_mask_dir is not None:
        original_id = prepared.get("mask_id", "mask")
        mask_path = filtered_mask_dir / f"{original_id}.npz"
        save_mask(mask, mask_path)
        prepared["mask_path"] = str(mask_path)
    prepared["_mask_array"] = mask
    return prepared


def filter_and_deduplicate_frame_masks(
    mask_records: list[dict[str, Any]],
    config: PipelineConfig,
    task: str,
    image_shape: tuple[int, int] | None = None,
    filtered_mask_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "raw_masks": int(len(mask_records)),
        "kept_after_score_area_heuristics": 0,
        "kept_after_dedup": 0,
        "rejected": {},
    }
    prepared_records: list[dict[str, Any]] = []
    out_dir = Path(filtered_mask_dir) if filtered_mask_dir is not None else None

    for record in mask_records:
        try:
            mask = load_mask(record)
        except Exception as exc:
            reason = f"mask_load_failed:{exc}"
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        if image_shape is None:
            image_shape = mask.shape
        area_px = int(mask.sum())
        area_frac = float(area_px / max(1, image_shape[0] * image_shape[1]))
        score = _record_score(record)
        semantic_group = record.get("semantic_group", "primary")
        score_threshold = config.score_threshold_for_group(semantic_group, task)
        min_mask_area_px = config.min_mask_area_for_group(semantic_group, task)
        reason = None
        if score < score_threshold:
            reason = "score_below_threshold"
        elif area_px < min_mask_area_px:
            reason = "area_below_threshold"
        elif area_frac > config.max_mask_area_frac:
            reason = "area_above_threshold"
        else:
            reason = _heuristic_reject_reason(mask, record, task, config, area_frac)
        if reason is not None:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        cleaned, cleanup_stats = cleanup_connected_components(mask, task, config, semantic_group)
        cleaned_area = int(cleaned.sum())
        if cleaned_area < min_mask_area_px:
            reason = (
                "deleafing_primary_vertical_axis_like_main_stem"
                if cleanup_stats.get("axis_rejected_components", 0) > 0
                else "area_below_threshold_after_cleanup"
            )
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        prepared_records.append(_prepare_record(record, cleaned, image_shape, cleanup_stats, None))

    stats["kept_after_score_area_heuristics"] = int(len(prepared_records))

    deduped: list[dict[str, Any]] = []
    for semantic_group in ("primary", "auxiliary"):
        group_records = [
            record
            for record in prepared_records
            if record.get("semantic_group", "primary") == semantic_group
        ]
        group_records.sort(key=_record_score, reverse=True)
        for record in group_records:
            mask = record["_mask_array"]
            merged = False
            for kept in deduped:
                if kept.get("semantic_group", "primary") != semantic_group:
                    continue
                iou = mask_iou(mask, kept["_mask_array"])
                if iou > config.mask_dedup_iou_thres:
                    prompts = kept.setdefault("source_prompts", [])
                    for prompt in record.get("source_prompts", [record.get("prompt", "")]):
                        if prompt and prompt not in prompts:
                            prompts.append(prompt)
                    kept["dedup_iou_max"] = max(float(kept.get("dedup_iou_max", 0.0)), float(iou))
                    merged = True
                    break
            if not merged:
                deduped.append(record)

    if task == "deleafing" and config.deleafing_max_aux_masks_per_frame >= 0:
        primary = [record for record in deduped if record.get("semantic_group", "primary") != "auxiliary"]
        auxiliary = [record for record in deduped if record.get("semantic_group", "primary") == "auxiliary"]
        if config.deleafing_max_primary_masks_per_frame >= 0:
            primary.sort(key=_deleafing_primary_rank, reverse=True)
            max_primary = int(config.deleafing_max_primary_masks_per_frame)
            if len(primary) > max_primary:
                stats["rejected"]["deleafing_primary_mask_cap"] = (
                    stats["rejected"].get("deleafing_primary_mask_cap", 0)
                    + len(primary)
                    - max_primary
                )
                primary = primary[:max_primary]
        auxiliary.sort(key=_record_score, reverse=True)
        max_aux = int(config.deleafing_max_aux_masks_per_frame)
        if len(auxiliary) > max_aux:
            stats["rejected"]["deleafing_aux_mask_cap"] = (
                stats["rejected"].get("deleafing_aux_mask_cap", 0) + len(auxiliary) - max_aux
            )
            auxiliary = auxiliary[:max_aux]
        deduped = primary + auxiliary

    if task == "deleafing":
        final_records: list[dict[str, Any]] = []
        assert image_shape is not None
        for record in deduped:
            filtered_record = _final_filter_deleafing_primary_components(
                record=record,
                config=config,
                task=task,
                image_shape=image_shape,
                stats=stats,
            )
            if filtered_record is not None:
                final_records.append(filtered_record)
        deduped = final_records

    if out_dir is not None:
        for record in deduped:
            mask = record.get("_mask_array")
            if isinstance(mask, np.ndarray):
                mask_id = record.get("mask_id", "mask")
                mask_path = out_dir / f"{mask_id}.npz"
                save_mask(mask, mask_path)
                record["mask_path"] = str(mask_path)
    for record in deduped:
        record.pop("_mask_array", None)
    stats["kept_after_dedup"] = int(len(deduped))
    return deduped, stats


def _write_overlay_records(
    image: np.ndarray,
    records: list[dict[str, Any]],
    out_path: Path,
    draw_principal_axes: bool = False,
) -> None:
    masks = [load_mask(record) for record in records]
    labels = [
        f"{record.get('prompt', '')} {float(record.get('score', 0.0)):.2f}"
        for record in records
    ]
    save_sam_overlay(
        image=image,
        masks=masks,
        labels=labels,
        out_path=out_path,
        draw_principal_axes=draw_principal_axes,
    )


def _save_filtered_overlay(frame_pred: dict[str, Any], filtered_masks: list[dict[str, Any]], out_dir: Path) -> None:
    image_path = frame_pred.get("image_path")
    if not image_path or not Path(image_path).exists():
        return
    split = frame_pred.get("split")
    if not split:
        frame_id = str(frame_pred.get("frame_id", ""))
        split = frame_id.split("_", 1)[0] if "_" in frame_id else "unknown"
    image = np.asarray(Image.open(image_path).convert("RGB"))
    frame_name = f"{frame_pred.get('frame_id', 'frame')}.jpg"
    primary = [
        record for record in filtered_masks if record.get("semantic_group", "primary") != "auxiliary"
    ]
    auxiliary = [
        record for record in filtered_masks if record.get("semantic_group", "primary") == "auxiliary"
    ]
    _write_overlay_records(
        image=image,
        records=primary,
        out_path=out_dir / "visualizations" / f"sam_{split}" / frame_name,
    )
    if primary:
        _write_overlay_records(
            image=image,
            records=primary,
            out_path=out_dir / "visualizations" / f"sam_{split}_principal_axes" / frame_name,
            draw_principal_axes=True,
        )
    if auxiliary:
        _write_overlay_records(
            image=image,
            records=auxiliary,
            out_path=out_dir / "visualizations" / f"sam_{split}_auxiliary" / frame_name,
        )
    if primary and auxiliary:
        _write_overlay_records(
            image=image,
            records=filtered_masks,
            out_path=out_dir / "visualizations" / f"sam_{split}_all_filtered" / frame_name,
        )
        _write_overlay_records(
            image=image,
            records=filtered_masks,
            out_path=out_dir / "visualizations" / f"sam_{split}_all_principal_axes" / frame_name,
            draw_principal_axes=True,
        )


def postprocess_prediction_file(
    input_json: str | Path,
    output_json: str | Path,
    config: PipelineConfig,
    task: str,
    save_visualizations: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = Path(input_json)
    if not input_path.exists():
        predictions: list[dict[str, Any]] = []
    else:
        with input_path.open("r") as handle:
            predictions = json.load(handle)
    filtered_mask_dir = input_path.parent / "filtered_masks"
    output: list[dict[str, Any]] = []
    aggregate_stats = {
        "raw_masks": 0,
        "filtered_masks": 0,
        "frames": int(len(predictions)),
        "per_frame": {},
    }

    for frame_pred in predictions:
        image_shape = None
        if frame_pred.get("height") and frame_pred.get("width"):
            image_shape = (int(frame_pred["height"]), int(frame_pred["width"]))
        frame_dir = filtered_mask_dir / frame_pred.get("frame_id", "unknown")
        filtered_masks, frame_stats = filter_and_deduplicate_frame_masks(
            frame_pred.get("masks", []),
            config=config,
            task=task,
            image_shape=image_shape,
            filtered_mask_dir=frame_dir,
        )
        new_frame = dict(frame_pred)
        new_frame["masks"] = filtered_masks
        output.append(new_frame)
        if save_visualizations:
            _save_filtered_overlay(frame_pred, filtered_masks, input_path.parent.parent)
        aggregate_stats["raw_masks"] += int(frame_stats["raw_masks"])
        aggregate_stats["filtered_masks"] += int(frame_stats["kept_after_dedup"])
        aggregate_stats["per_frame"][frame_pred.get("frame_id", "unknown")] = frame_stats

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_json).open("w") as handle:
        json.dump(output, handle, indent=2)
    return output, aggregate_stats
