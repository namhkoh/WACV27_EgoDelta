from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from tomato_cutpoints.clustering.cluster_features import compute_cluster_features


@dataclass
class ClusterResult:
    labels: np.ndarray
    features: list[dict[str, Any]]
    cluster_points: pd.DataFrame
    all_cluster_points: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.all_cluster_points is None:
            self.all_cluster_points = self.cluster_points.copy()


def cluster_removed_points(
    removed_xyz: np.ndarray,
    original_indices: np.ndarray,
    nn_distances: np.ndarray | None,
    m3c2_distances: np.ndarray | None,
    combined_scores: np.ndarray | None,
    config: dict[str, Any],
) -> ClusterResult:
    ccfg = config.get("dbscan", config)
    removed_xyz = np.asarray(removed_xyz, dtype=np.float64)
    original_indices = np.asarray(original_indices, dtype=np.int64)

    if len(removed_xyz) == 0:
        empty = pd.DataFrame(columns=["cluster_id", "original_index", "x", "y", "z"])
        return ClusterResult(np.empty(0, dtype=np.int64), [], empty, empty.copy())

    raw_labels = DBSCAN(eps=float(ccfg.get("eps_m", 0.03)), min_samples=int(ccfg.get("min_samples", 8))).fit_predict(removed_xyz)
    all_cluster_points = pd.DataFrame(
        {
            "cluster_id": raw_labels,
            "original_index": original_indices,
            "x": removed_xyz[:, 0],
            "y": removed_xyz[:, 1],
            "z": removed_xyz[:, 2],
        }
    )
    all_cluster_points = all_cluster_points.loc[all_cluster_points["cluster_id"] >= 0].reset_index(drop=True)

    labels = np.full(len(removed_xyz), -1, dtype=np.int64)
    features: list[dict[str, Any]] = []
    next_id = 0
    min_points = int(ccfg.get("min_cluster_points", 20))
    max_points = int(ccfg.get("max_cluster_points", 200_000))

    raw_components: list[np.ndarray] = []
    for raw_label in sorted(int(v) for v in np.unique(raw_labels) if v >= 0):
        positions = np.flatnonzero(raw_labels == raw_label)
        raw_components.append(positions)

    component_groups = raw_components
    if ccfg.get("merge_components", False) and raw_components:
        centroids = np.asarray([removed_xyz[positions].mean(axis=0) for positions in raw_components], dtype=np.float64)
        merge_eps = float(ccfg.get("merge_eps_m", 0.09))
        merge_labels = DBSCAN(eps=merge_eps, min_samples=1).fit_predict(centroids)
        merge_max_diameter = ccfg.get("merge_max_diameter_m")
        merge_max_diagonal = ccfg.get("merge_max_bbox_diagonal_m")
        merged_groups: list[np.ndarray] = []
        for merge_label in sorted(int(v) for v in np.unique(merge_labels) if v >= 0):
            component_ids = np.flatnonzero(merge_labels == merge_label)
            merged_positions = np.concatenate([raw_components[int(i)] for i in component_ids])
            points = removed_xyz[merged_positions]
            dims = points.max(axis=0) - points.min(axis=0)
            diameter_ok = merge_max_diameter is None or float(np.max(dims)) <= float(merge_max_diameter)
            diagonal_ok = merge_max_diagonal is None or float(np.linalg.norm(dims)) <= float(merge_max_diagonal)
            count_ok = len(merged_positions) <= max_points
            if diameter_ok and diagonal_ok and count_ok:
                merged_groups.append(np.sort(merged_positions))
            else:
                for component_id in component_ids:
                    merged_groups.append(raw_components[int(component_id)])
        component_groups = merged_groups

    for positions in component_groups:
        n = int(len(positions))
        if n < min_points or n > max_points:
            continue
        labels[positions] = next_id
        idx = original_indices[positions]
        feature = compute_cluster_features(
            next_id,
            removed_xyz[positions],
            idx,
            nn_distances[positions] if nn_distances is not None else None,
            m3c2_distances[positions] if m3c2_distances is not None else None,
            combined_scores[positions] if combined_scores is not None else None,
        )
        features.append(feature)
        next_id += 1

    cluster_points = pd.DataFrame(
        {
            "cluster_id": labels,
            "original_index": original_indices,
            "x": removed_xyz[:, 0],
            "y": removed_xyz[:, 1],
            "z": removed_xyz[:, 2],
        }
    )
    keep = cluster_points["cluster_id"] >= 0
    cluster_points = cluster_points.loc[keep].reset_index(drop=True)
    return ClusterResult(labels, features, cluster_points, all_cluster_points)
