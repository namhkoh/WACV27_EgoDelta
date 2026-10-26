"""Fuse per-frame lifted tomato detections into 3D tomato instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, relpath


def fuse_instances_for_split(
    split: str,
    lifted_detections: list[dict[str, Any]],
    out_dir: str | Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    instances = fuse_lifted_detections(
        split=split,
        lifted_detections=lifted_detections,
        out_dir=Path(out_dir),
        eps_m=config.instance_cluster_eps_m,
        min_cluster_samples=max(1, config.instance_min_frames),
        min_frames=config.instance_min_frames,
        min_points=config.instance_min_points,
        min_confidence=config.conf_thres,
    )
    output_path = Path(out_dir) / "lifted_3d" / f"{split}_instances.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(instances, indent=2), encoding="utf-8")
    return instances


def fuse_lifted_detections(
    split: str,
    lifted_detections: list[dict[str, Any]],
    out_dir: Path,
    eps_m: float,
    min_cluster_samples: int,
    min_frames: int,
    min_points: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    ok_detections = [
        det
        for det in lifted_detections
        if det.get("status") == "ok" and det.get("centroid_3d") is not None
    ]
    if not ok_detections:
        return []

    centroids = np.asarray([det["centroid_3d"] for det in ok_detections], dtype=float)
    labels = cluster_centroids(centroids, eps_m=eps_m, min_samples=min_cluster_samples)
    instances: list[dict[str, Any]] = []
    points_dir = out_dir / "lifted_3d" / "points"
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
        if len(frame_ids) < min_frames or len(points) < min_points or mean_conf < min_confidence:
            continue

        centroid = robust_centroid(points)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        distances = np.linalg.norm(points - centroid.reshape(1, 3), axis=1)
        radius = float(np.percentile(distances, 75)) if len(distances) else 0.0

        instance_id = f"{split}_tomato_{len(instances):03d}"
        points_path = points_dir / f"{instance_id}_points.npy"
        np.save(points_path, points.astype(np.float32))
        instances.append(
            {
                "instance_id": instance_id,
                "split": split,
                "semantic_class": "tomato",
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
        )
    return instances


def cluster_centroids(centroids: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 3)
    if len(centroids) == 0:
        return np.zeros((0,), dtype=int)
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        return DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(centroids)
    except Exception:
        return _fallback_radius_clusters(centroids, eps_m=eps_m, min_samples=min_samples)


def robust_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros((3,), dtype=float)
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median.reshape(1, 3), axis=1)
    keep_threshold = np.percentile(distances, 85)
    trimmed = points[distances <= keep_threshold]
    if len(trimmed) == 0:
        return median
    return np.mean(trimmed, axis=0)


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


def _fallback_radius_clusters(centroids: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    labels = np.full((len(centroids),), -1, dtype=int)
    current_label = 0
    for idx in range(len(centroids)):
        if labels[idx] >= 0:
            continue
        distances = np.linalg.norm(centroids - centroids[idx], axis=1)
        neighbors = np.where(distances <= eps_m)[0]
        if len(neighbors) < min_samples:
            continue
        stack = list(neighbors)
        labels[neighbors] = current_label
        while stack:
            node = stack.pop()
            node_distances = np.linalg.norm(centroids - centroids[node], axis=1)
            node_neighbors = np.where(node_distances <= eps_m)[0]
            if len(node_neighbors) >= min_samples:
                for neighbor in node_neighbors:
                    if labels[neighbor] < 0:
                        labels[neighbor] = current_label
                        stack.append(int(neighbor))
        current_label += 1
    return labels

