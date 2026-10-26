from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .io_aria import load_frames_metadata
from .mask_postprocess import load_mask
from .mps_geometry import AriaMpsGeometry
from .size_filters import deleafing_petiole_size_reject_reason
from .task_prompts import target_class_for_group


@dataclass
class LiftResult:
    points_3d: np.ndarray
    selected_indices: np.ndarray
    uv_inside: np.ndarray
    uv_foreground: np.ndarray
    status: str
    num_points_inside_mask: int
    num_points_after_depth_filter: int
    num_points_after_cluster: int


def _run_dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=int)
    try:
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    except Exception:
        pass

    labels = np.full((len(points),), -1, dtype=int)
    visited = np.zeros((len(points),), dtype=bool)
    cluster_id = 0
    for idx in range(len(points)):
        if visited[idx]:
            continue
        visited[idx] = True
        distances = np.linalg.norm(points - points[idx], axis=1)
        neighbors = np.where(distances <= eps)[0]
        if len(neighbors) < min_samples:
            labels[idx] = -1
            continue
        labels[idx] = cluster_id
        seeds = list(neighbors)
        while seeds:
            current = seeds.pop()
            if not visited[current]:
                visited[current] = True
                current_dist = np.linalg.norm(points - points[current], axis=1)
                current_neighbors = np.where(current_dist <= eps)[0]
                if len(current_neighbors) >= min_samples:
                    seeds.extend([int(n) for n in current_neighbors if n not in seeds])
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1
    return labels


def _largest_non_noise_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray | None:
    labels = _run_dbscan(points, eps, min_samples)
    valid_labels = [label for label in np.unique(labels) if label >= 0]
    if not valid_labels:
        return None
    best_label = max(valid_labels, key=lambda label: int((labels == label).sum()))
    return labels == best_label


def lift_mask_points(
    mask: np.ndarray,
    points_world: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    config: PipelineConfig,
    task: str,
) -> LiftResult:
    mask = np.asarray(mask, dtype=bool)
    points_world = np.asarray(points_world, dtype=float).reshape((-1, 3))
    uv = np.asarray(uv, dtype=float).reshape((-1, 2))
    depth = np.asarray(depth, dtype=float).reshape((-1,))
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape((-1,))
    min_points_inside_mask = config.min_points_inside_mask_for_task(task)

    candidate_indices = np.where(valid_mask)[0]
    if len(candidate_indices) == 0:
        return LiftResult(
            points_3d=np.empty((0, 3)),
            selected_indices=np.empty((0,), dtype=int),
            uv_inside=np.empty((0, 2)),
            uv_foreground=np.empty((0, 2)),
            status="no_3d_support",
            num_points_inside_mask=0,
            num_points_after_depth_filter=0,
            num_points_after_cluster=0,
        )

    uv_candidates = uv[candidate_indices]
    xs = np.floor(uv_candidates[:, 0]).astype(int)
    ys = np.floor(uv_candidates[:, 1]).astype(int)
    inside_image = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    inside_mask = np.zeros((len(candidate_indices),), dtype=bool)
    inside_mask[inside_image] = mask[ys[inside_image], xs[inside_image]]
    inside_indices = candidate_indices[inside_mask]
    uv_inside = uv[inside_indices]

    if len(inside_indices) < min_points_inside_mask:
        return LiftResult(
            points_3d=np.empty((0, 3)),
            selected_indices=np.empty((0,), dtype=int),
            uv_inside=uv_inside,
            uv_foreground=np.empty((0, 2)),
            status="no_3d_support",
            num_points_inside_mask=int(len(inside_indices)),
            num_points_after_depth_filter=0,
            num_points_after_cluster=0,
        )

    depths_inside = depth[inside_indices]
    finite_depth = np.isfinite(depths_inside)
    inside_indices = inside_indices[finite_depth]
    depths_inside = depths_inside[finite_depth]
    if len(inside_indices) < min_points_inside_mask:
        return LiftResult(
            points_3d=np.empty((0, 3)),
            selected_indices=np.empty((0,), dtype=int),
            uv_inside=uv_inside,
            uv_foreground=np.empty((0, 2)),
            status="no_3d_support",
            num_points_inside_mask=int(len(inside_indices)),
            num_points_after_depth_filter=0,
            num_points_after_cluster=0,
        )

    z_front = float(np.quantile(depths_inside, config.depth_quantile))
    depth_band = config.depth_band_for_task(task)
    fg_keep = depths_inside <= z_front + depth_band
    foreground_indices = inside_indices[fg_keep]
    uv_foreground = uv[foreground_indices]

    if len(foreground_indices) == 0:
        return LiftResult(
            points_3d=np.empty((0, 3)),
            selected_indices=np.empty((0,), dtype=int),
            uv_inside=uv_inside,
            uv_foreground=uv_foreground,
            status="no_3d_support_depth_filtered",
            num_points_inside_mask=int(len(inside_indices)),
            num_points_after_depth_filter=0,
            num_points_after_cluster=0,
        )

    foreground_points = points_world[foreground_indices]
    cluster_mask = _largest_non_noise_cluster(
        foreground_points,
        eps=config.dbscan_eps_for_task(task),
        min_samples=config.dbscan_min_samples_for_task(task),
    )
    if cluster_mask is None:
        status = "dbscan_failed_kept_foreground"
        selected_indices = foreground_indices
        selected_points = foreground_points
    else:
        status = "ok"
        selected_indices = foreground_indices[cluster_mask]
        selected_points = points_world[selected_indices]

    return LiftResult(
        points_3d=selected_points,
        selected_indices=selected_indices,
        uv_inside=uv_inside,
        uv_foreground=uv[selected_indices] if len(selected_indices) else uv_foreground,
        status=status,
        num_points_inside_mask=int(len(inside_indices)),
        num_points_after_depth_filter=int(len(foreground_indices)),
        num_points_after_cluster=int(len(selected_points)),
    )


def _bbox_3d(points: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
    if len(points) == 0:
        return None, None
    return np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()


def _centroid(points: np.ndarray) -> list[float] | None:
    if len(points) == 0:
        return None
    return np.median(points, axis=0).tolist()


def run_mask_lifting_for_split(
    predictions_json: str | Path,
    frames_json: str | Path,
    split: str,
    geometry: AriaMpsGeometry,
    config: PipelineConfig,
    task: str,
    out_dir: str | Path,
    debug: bool = True,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    predictions_path = Path(predictions_json)
    if predictions_path.exists():
        with predictions_path.open("r") as handle:
            predictions = json.load(handle)
    else:
        predictions = []
    frame_meta = {frame["frame_id"]: frame for frame in load_frames_metadata(frames_json)}
    points_world = geometry.load_filtered_points()

    out_root = Path(out_dir) / "lifted_3d"
    points_dir = out_root / "points"
    debug_dir = Path(out_dir) / "visualizations" / "lifting_debug" / split
    lifted_path = out_root / f"{split}_lifted_masks.json"
    lifted_records: list[dict[str, Any]] = []
    stats = {
        "lifted_total": 0,
        "lifted_ok": 0,
        "no_3d_support": 0,
        "size_rejected": 0,
        "per_prompt_stats": {},
    }

    for frame_pred in predictions:
        frame_id = frame_pred["frame_id"]
        frame = frame_meta.get(frame_id, frame_pred)
        width = int(frame.get("width", frame_pred.get("width", 0)))
        height = int(frame.get("height", frame_pred.get("height", 0)))
        timestamp_ns = int(frame["timestamp_ns"])
        uv, depth, valid = geometry.project_world_points_to_frame(points_world, timestamp_ns, width, height)
        image = None
        if debug and frame.get("image_path") and Path(frame["image_path"]).exists():
            image = np.asarray(Image.open(frame["image_path"]).convert("RGB"))
        petiole_size_overlay_masks: list[np.ndarray] = []
        petiole_size_overlay_labels: list[str] = []

        for mask_record in frame_pred.get("masks", []):
            mask = load_mask(mask_record)
            result = lift_mask_points(mask, points_world, uv, depth, valid, config, task)
            point_path = None
            if len(result.points_3d):
                point_path = points_dir / f"{mask_record['mask_id']}.npy"
                point_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(point_path, result.points_3d)
            bbox_min, bbox_max = _bbox_3d(result.points_3d)
            semantic_group = mask_record.get("semantic_group", "primary")
            status = result.status
            metric_size_stats: dict[str, Any] = {}
            if (
                task == "deleafing"
                and semantic_group == "primary"
                and status in ("ok", "dbscan_failed_kept_foreground")
            ):
                reject_reason, metric_size_stats = deleafing_petiole_size_reject_reason(
                    result.points_3d, config
                )
                if reject_reason is not None:
                    status = f"rejected_{reject_reason}"
            record = {
                "mask_id": mask_record["mask_id"],
                "frame_id": frame_id,
                "split": split,
                "task": task,
                "semantic_group": semantic_group,
                "target_class": mask_record.get(
                    "target_class", target_class_for_group(task, semantic_group)
                ),
                "source_prompts": list(
                    mask_record.get("source_prompts", [mask_record.get("prompt", "")])
                ),
                "score_2d": float(mask_record.get("score", mask_record.get("score_2d", 0.0))),
                "num_points_inside_mask": result.num_points_inside_mask,
                "num_points_after_depth_filter": result.num_points_after_depth_filter,
                "num_points_after_cluster": result.num_points_after_cluster,
                "centroid_3d": _centroid(result.points_3d),
                "bbox_3d_min": bbox_min,
                "bbox_3d_max": bbox_max,
                "metric_size_stats": metric_size_stats,
                "points_3d_path": str(point_path) if point_path is not None else None,
                "status": status,
            }
            lifted_records.append(record)
            stats["lifted_total"] += 1
            if status in ("ok", "dbscan_failed_kept_foreground"):
                stats["lifted_ok"] += 1
            if status.startswith("no_3d_support"):
                stats["no_3d_support"] += 1
            if status.startswith("rejected_petiole_3d"):
                stats["size_rejected"] += 1
            for prompt in record["source_prompts"]:
                prompt_stats = stats["per_prompt_stats"].setdefault(prompt, {"lifted_ok": 0})
                if status in ("ok", "dbscan_failed_kept_foreground"):
                    prompt_stats["lifted_ok"] += 1
            if (
                debug
                and image is not None
                and task == "deleafing"
                and semantic_group == "primary"
                and status in ("ok", "dbscan_failed_kept_foreground")
            ):
                petiole_size_overlay_masks.append(mask)
                size_label = ""
                if metric_size_stats:
                    size_label = f" major={metric_size_stats['major_extent_m']:.2f}m"
                petiole_size_overlay_labels.append(
                    f"{mask_record['mask_id']} {record['source_prompts'][0]}{size_label}"
                )
            if debug and image is not None:
                from .visualization import save_lifting_debug_overlay

                save_lifting_debug_overlay(
                    image=image,
                    mask=mask,
                    uv_inside=result.uv_inside,
                    uv_foreground=result.uv_foreground,
                    out_path=debug_dir / f"{mask_record['mask_id']}.jpg",
                    label=f"{mask_record['mask_id']} {record['source_prompts']} {status}",
                )

        if debug and image is not None and task == "deleafing":
            from .visualization import save_sam_overlay

            save_sam_overlay(
                image=image,
                masks=petiole_size_overlay_masks,
                labels=petiole_size_overlay_labels,
                out_path=Path(out_dir)
                / "visualizations"
                / f"sam_{split}_3d_petiole_filtered"
                / f"{frame_id}.jpg",
            )

    lifted_path.parent.mkdir(parents=True, exist_ok=True)
    with lifted_path.open("w") as handle:
        json.dump(lifted_records, handle, indent=2)
    return lifted_path, lifted_records, stats


def run_mask_lifting(
    pre_predictions_json: str | Path,
    post_predictions_json: str | Path,
    pre_frames_json: str | Path,
    post_frames_json: str | Path,
    pre_geometry: AriaMpsGeometry,
    post_geometry: AriaMpsGeometry,
    config: PipelineConfig,
    task: str,
    out_dir: str | Path,
    debug: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    pre_path, _, pre_stats = run_mask_lifting_for_split(
        predictions_json=pre_predictions_json,
        frames_json=pre_frames_json,
        split="pre",
        geometry=pre_geometry,
        config=config,
        task=task,
        out_dir=out_dir,
        debug=debug,
    )
    post_path, _, post_stats = run_mask_lifting_for_split(
        predictions_json=post_predictions_json,
        frames_json=post_frames_json,
        split="post",
        geometry=post_geometry,
        config=config,
        task=task,
        out_dir=out_dir,
        debug=debug,
    )
    return pre_path, post_path, {"pre": pre_stats, "post": post_stats}
