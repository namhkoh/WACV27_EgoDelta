from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tomato_cutpoints.clustering.cluster_features import compute_cluster_features
from tomato_cutpoints.clustering.dbscan import ClusterResult


def _metric_subset(
    subset_indices: np.ndarray,
    original_indices: np.ndarray,
    values: np.ndarray | None,
) -> np.ndarray | None:
    if values is None:
        return None
    positions = {int(idx): pos for pos, idx in enumerate(np.asarray(original_indices, dtype=np.int64))}
    subset_positions = np.asarray([positions[int(idx)] for idx in subset_indices], dtype=np.int64)
    return np.asarray(values)[subset_positions]


def _best_kmeans_split(points: np.ndarray, priors: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    min_points = int(priors.get("split_min_points", priors.get("min_points", 20)))
    max_parts = int(priors.get("split_max_parts", 3))
    min_center_distance = float(priors.get("split_min_center_distance_m", 0.055))
    min_silhouette = float(priors.get("split_min_silhouette", 0.35))
    max_subcluster_diameter = float(priors.get("split_max_subcluster_diameter_m", priors.get("max_diameter_m", 0.16)))

    best_labels: np.ndarray | None = None
    best_diag: dict[str, Any] = {"accepted": False, "candidates": []}
    best_objective = -np.inf

    for k in range(2, max_parts + 1):
        if n_points < k * min_points:
            continue
        labels = KMeans(n_clusters=k, random_state=13, n_init=20).fit_predict(points)
        sizes = np.asarray([int(np.sum(labels == label)) for label in range(k)], dtype=np.int64)
        if np.any(sizes < min_points):
            continue

        centers = np.asarray([points[labels == label].mean(axis=0) for label in range(k)], dtype=np.float64)
        center_distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        center_distances[center_distances == 0] = np.inf
        min_sep = float(np.min(center_distances))
        subcluster_diameters = np.asarray(
            [float(np.max(np.ptp(points[labels == label], axis=0))) for label in range(k)],
            dtype=np.float64,
        )
        if min_sep < min_center_distance or np.any(subcluster_diameters > max_subcluster_diameter):
            continue

        sil = float(silhouette_score(points, labels))
        candidate_diag = {
            "k": k,
            "silhouette": sil,
            "sizes": sizes.tolist(),
            "min_center_distance_m": min_sep,
            "subcluster_diameters_m": subcluster_diameters.tolist(),
        }
        best_diag["candidates"].append(candidate_diag)
        if sil < min_silhouette:
            continue

        # Prefer clear, simple splits. Extra parts need noticeably better separation.
        objective = sil - 0.05 * (k - 2)
        if objective > best_objective:
            best_objective = objective
            best_labels = labels
            best_diag.update(candidate_diag)
            best_diag["accepted"] = True

    return best_labels, best_diag


def split_merged_harvest_clusters(
    result: ClusterResult,
    original_indices: np.ndarray,
    nn_distances: np.ndarray | None,
    m3c2_distances: np.ndarray | None,
    combined_scores: np.ndarray | None,
    config: dict[str, Any],
) -> tuple[ClusterResult, dict[str, Any]]:
    priors = config.get("task_priors", {}).get("harvest", {})
    if not priors.get("split_merged", False) or not result.features:
        return result, {"enabled": bool(priors.get("split_merged", False)), "splits": []}

    trigger_diameter = float(priors.get("split_trigger_diameter_m", priors.get("max_diameter_m", 0.15)))
    new_features: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    next_cluster_id = 0

    original_indices = np.asarray(original_indices, dtype=np.int64)
    for feature in result.features:
        points = np.asarray(feature["points"], dtype=np.float64)
        point_indices = np.asarray(feature["point_indices"], dtype=np.int64)
        diameter = float(feature.get("approximate_diameter") or 0.0)
        if diameter < trigger_diameter:
            new_feature = compute_cluster_features(
                next_cluster_id,
                points,
                point_indices,
                _metric_subset(point_indices, original_indices, nn_distances),
                _metric_subset(point_indices, original_indices, m3c2_distances),
                _metric_subset(point_indices, original_indices, combined_scores),
            )
            new_features.append(new_feature)
            next_cluster_id += 1
            continue

        labels, split_diag = _best_kmeans_split(points, priors)
        report = {
            "source_cluster_id": int(feature["cluster_id"]),
            "source_n_points": int(feature.get("n_points", len(points))),
            "source_diameter_m": diameter,
            **split_diag,
        }
        split_reports.append(report)

        if labels is None:
            new_feature = compute_cluster_features(
                next_cluster_id,
                points,
                point_indices,
                _metric_subset(point_indices, original_indices, nn_distances),
                _metric_subset(point_indices, original_indices, m3c2_distances),
                _metric_subset(point_indices, original_indices, combined_scores),
            )
            new_features.append(new_feature)
            next_cluster_id += 1
            continue

        for split_label in sorted(int(v) for v in np.unique(labels)):
            mask = labels == split_label
            split_indices = point_indices[mask]
            split_feature = compute_cluster_features(
                next_cluster_id,
                points[mask],
                split_indices,
                _metric_subset(split_indices, original_indices, nn_distances),
                _metric_subset(split_indices, original_indices, m3c2_distances),
                _metric_subset(split_indices, original_indices, combined_scores),
            )
            split_feature["split_parent_cluster_id"] = int(feature["cluster_id"])
            new_features.append(split_feature)
            next_cluster_id += 1

    if not new_features:
        empty = pd.DataFrame(columns=["cluster_id", "original_index", "x", "y", "z"])
        return ClusterResult(np.empty(0, dtype=np.int64), [], empty), {"enabled": True, "splits": split_reports}

    rows: list[pd.DataFrame] = []
    label_by_original_index: dict[int, int] = {}
    for feature in new_features:
        points = np.asarray(feature["points"], dtype=np.float64)
        point_indices = np.asarray(feature["point_indices"], dtype=np.int64)
        cluster_id = int(feature["cluster_id"])
        for idx in point_indices:
            label_by_original_index[int(idx)] = cluster_id
        rows.append(
            pd.DataFrame(
                {
                    "cluster_id": cluster_id,
                    "original_index": point_indices,
                    "x": points[:, 0],
                    "y": points[:, 1],
                    "z": points[:, 2],
                }
            )
        )

    labels = np.asarray([label_by_original_index.get(int(idx), -1) for idx in original_indices], dtype=np.int64)
    cluster_points = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ClusterResult(labels, new_features, cluster_points), {"enabled": True, "splits": split_reports}
