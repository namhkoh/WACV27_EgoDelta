from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .mask_lifting import _run_dbscan
from .size_filters import deleafing_petiole_size_reject_reason


VALID_LIFT_STATUSES = {"ok", "dbscan_failed_kept_foreground"}


def _load_points(path: str | None) -> np.ndarray:
    if not path:
        return np.empty((0, 3), dtype=float)
    p = Path(path)
    if not p.exists():
        return np.empty((0, 3), dtype=float)
    return np.asarray(np.load(p), dtype=float).reshape((-1, 3))


def _robust_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) == 0:
        return np.zeros((3,), dtype=float)
    if len(points) < 8:
        return np.median(points, axis=0)
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    keep = distances <= np.quantile(distances, 0.90)
    return np.mean(points[keep], axis=0)


def _radius(points: np.ndarray, centroid: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    distances = np.linalg.norm(points - centroid.reshape((1, 3)), axis=1)
    return float(np.quantile(distances, 0.90))


def fuse_lifted_detections(
    lifted_records: list[dict[str, Any]],
    config: PipelineConfig,
    task: str,
    split: str,
    out_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    usable = [
        record
        for record in lifted_records
        if record.get("status") in VALID_LIFT_STATUSES and record.get("centroid_3d") is not None
    ]
    if not usable:
        return []

    instances: list[dict[str, Any]] = []
    groups = sorted(
        {
            (record.get("semantic_group", "primary"), record.get("target_class", "unknown"))
            for record in usable
        }
    )
    points_dir = Path(out_dir) / "lifted_3d" / "instance_points" if out_dir is not None else None
    if points_dir is not None:
        points_dir.mkdir(parents=True, exist_ok=True)

    for semantic_group, target_class in groups:
        group_records = [
            record
            for record in usable
            if record.get("semantic_group", "primary") == semantic_group
            and record.get("target_class", "unknown") == target_class
        ]
        centroids = np.asarray([record["centroid_3d"] for record in group_records], dtype=float)
        labels = _run_dbscan(centroids, eps=config.instance_eps_for_task(task), min_samples=1)
        for label in sorted([int(label) for label in np.unique(labels) if label >= 0]):
            cluster_records = [record for record, record_label in zip(group_records, labels) if record_label == label]
            point_chunks = [_load_points(record.get("points_3d_path")) for record in cluster_records]
            point_chunks = [points for points in point_chunks if len(points)]
            if not point_chunks:
                continue
            points = np.vstack(point_chunks)
            centroid = _robust_centroid(points)
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            metric_size_stats: dict[str, Any] = {}
            if task == "deleafing" and semantic_group == "primary":
                reject_reason, metric_size_stats = deleafing_petiole_size_reject_reason(
                    points, config
                )
                if reject_reason is not None:
                    continue
            frames = {record["frame_id"] for record in cluster_records}
            mean_score = float(np.mean([float(record.get("score_2d", 0.0)) for record in cluster_records]))
            prompts: list[str] = []
            for record in cluster_records:
                for prompt in record.get("source_prompts", []):
                    if prompt not in prompts:
                        prompts.append(prompt)
            if len(frames) < config.instance_min_frames_for_group(semantic_group, task):
                continue
            if len(points) < config.instance_min_points_for_group(semantic_group, task):
                continue
            if mean_score < config.score_threshold_for_group(semantic_group, task):
                continue

            instance_index = len(instances)
            instance_id = f"{split}_{task}_{semantic_group}_{instance_index:03d}"
            points_path = None
            if points_dir is not None:
                points_path = points_dir / f"{instance_id}.npy"
                np.save(points_path, points)
            instances.append(
                {
                    "instance_id": instance_id,
                    "split": split,
                    "task": task,
                    "semantic_group": semantic_group,
                    "target_class": target_class,
                    "centroid_3d": centroid.tolist(),
                    "bbox_3d_min": bbox_min.tolist(),
                    "bbox_3d_max": bbox_max.tolist(),
                    "metric_size_stats": metric_size_stats,
                    "radius_m": _radius(points, centroid),
                    "num_frames_observed": int(len(frames)),
                    "num_points": int(len(points)),
                    "mean_score_2d": mean_score,
                    "source_prompts": prompts,
                    "points_3d_path": str(points_path) if points_path is not None else None,
                    "source_mask_ids": [record["mask_id"] for record in cluster_records],
                    "source_frame_ids": sorted(frames),
                }
            )
    return instances


def fuse_lifted_file(
    lifted_json: str | Path,
    split: str,
    task: str,
    config: PipelineConfig,
    out_dir: str | Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    lifted_path = Path(lifted_json)
    if lifted_path.exists():
        with lifted_path.open("r") as handle:
            lifted_records = json.load(handle)
    else:
        lifted_records = []
    instances = fuse_lifted_detections(
        lifted_records=lifted_records,
        config=config,
        task=task,
        split=split,
        out_dir=out_dir,
    )
    output_path = Path(out_dir) / "lifted_3d" / f"{split}_instances.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(instances, handle, indent=2)
    stats = {
        "num_lifted_records": int(len(lifted_records)),
        "num_instances": int(len(instances)),
        "num_primary_instances": int(
            sum(1 for instance in instances if instance.get("semantic_group") == "primary")
        ),
        "num_auxiliary_instances": int(
            sum(1 for instance in instances if instance.get("semantic_group") == "auxiliary")
        ),
    }
    return output_path, instances, stats


def fuse_prepost(
    pre_lifted_json: str | Path,
    post_lifted_json: str | Path,
    task: str,
    config: PipelineConfig,
    out_dir: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    pre_path, _, pre_stats = fuse_lifted_file(pre_lifted_json, "pre", task, config, out_dir)
    post_path, _, post_stats = fuse_lifted_file(post_lifted_json, "post", task, config, out_dir)
    return pre_path, post_path, {"pre": pre_stats, "post": post_stats}
