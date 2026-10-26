"""Task-generic multi-view lifted detection fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import relpath
from ..instance_fusion import cluster_centroids, robust_centroid
from ..task_config import TaskConfig
from .curve_fitting import fit_pca_line


def fuse_task_instances_for_split(
    split: str,
    lifted_detections: list[dict[str, Any]],
    out_dir: str | Path,
    task_config: TaskConfig,
) -> list[dict[str, Any]]:
    instances = fuse_task_lifted_detections(
        split=split,
        lifted_detections=lifted_detections,
        out_dir=Path(out_dir),
        task_config=task_config,
    )
    output_path = Path(out_dir) / "lifted_3d" / task_config.task_name / f"{split}_instances.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(instances, indent=2), encoding="utf-8")
    return instances


def fuse_task_lifted_detections(
    split: str,
    lifted_detections: list[dict[str, Any]],
    out_dir: Path,
    task_config: TaskConfig,
) -> list[dict[str, Any]]:
    ok_detections = [
        det
        for det in lifted_detections
        if det.get("status") == "ok" and det.get("centroid_3d") is not None
    ]
    if not ok_detections:
        return []

    centroids = np.asarray([det["centroid_3d"] for det in ok_detections], dtype=float)
    labels = cluster_centroids(
        centroids,
        eps_m=task_config.instance_cluster_eps_m,
        min_samples=max(1, task_config.instance_min_frames),
    )
    instances: list[dict[str, Any]] = []
    points_dir = out_dir / "lifted_3d" / task_config.task_name / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    for label in sorted(set(int(v) for v in labels if int(v) >= 0)):
        cluster_dets = [det for det, det_label in zip(ok_detections, labels) if int(det_label) == label]
        points_list = [_load_points(det, out_dir) for det in cluster_dets]
        points_list = [points for points in points_list if points.size > 0]
        if not points_list:
            continue
        points = np.vstack(points_list)
        frame_ids = sorted({str(det["frame_id"]) for det in cluster_dets})
        det_ids = [str(det["det_id"]) for det in cluster_dets]
        mean_conf = float(np.mean([float(det.get("confidence_2d", 0.0)) for det in cluster_dets]))
        if len(frame_ids) < task_config.instance_min_frames or len(points) < task_config.instance_min_points:
            continue
        if mean_conf < task_config.min_2d_confidence:
            continue

        centroid = robust_centroid(points)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        distances = np.linalg.norm(points - centroid.reshape(1, 3), axis=1)
        radius = float(np.percentile(distances, 75)) if len(distances) else 0.0
        instance_id = f"{split}_{task_config.target_semantic_class}_{len(instances):03d}"
        points_path = points_dir / f"{instance_id}_points.npy"
        np.save(points_path, points.astype(np.float32))
        instance: dict[str, Any] = {
            "instance_id": instance_id,
            "split": split,
            "semantic_class": task_config.target_semantic_class,
            "centroid_3d": [float(v) for v in centroid],
            "bbox_3d_min": [float(v) for v in bbox_min],
            "bbox_3d_max": [float(v) for v in bbox_max],
            "radius_m": radius,
            "num_frames_observed": int(len(frame_ids)),
            "num_points": int(len(points)),
            "mean_confidence_2d": mean_conf,
            "points_3d_path": relpath(points_path, out_dir),
            "source_det_ids": det_ids,
            "source_frame_ids": frame_ids,
        }
        if task_config.task_name == "deleaf":
            instance.update(petiole_geometry_payload(points))
        instances.append(instance)
    return instances


def petiole_geometry_payload(points: np.ndarray) -> dict[str, Any]:
    fit = fit_pca_line(points)
    axis = np.asarray(fit["axis"], dtype=float)
    endpoint_a = np.asarray(fit["endpoint_a"], dtype=float)
    endpoint_b = np.asarray(fit["endpoint_b"], dtype=float)
    return {
        "endpoint_a_3d": [float(v) for v in endpoint_a],
        "endpoint_b_3d": [float(v) for v in endpoint_b],
        "principal_axis_3d": [float(v) for v in axis],
        "length_m": float(fit["length_m"]),
        "thinness_score": float(fit["thinness"]),
    }


def _load_points(det: dict[str, Any], out_dir: Path) -> np.ndarray:
    path_value = det.get("points_3d_path")
    if not path_value:
        return np.zeros((0, 3), dtype=float)
    path = Path(path_value)
    if not path.is_absolute():
        path = out_dir / path
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    return np.asarray(np.load(path), dtype=float).reshape(-1, 3)
