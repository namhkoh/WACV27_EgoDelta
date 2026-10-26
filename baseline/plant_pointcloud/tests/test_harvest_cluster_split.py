from __future__ import annotations

import numpy as np
import pandas as pd

from tomato_cutpoints.clustering.cluster_features import compute_cluster_features
from tomato_cutpoints.clustering.dbscan import ClusterResult
from tomato_cutpoints.clustering.split import split_merged_harvest_clusters


def test_split_merged_harvest_cluster_separates_adjacent_tomatoes():
    rng = np.random.default_rng(31)
    left = rng.normal([0.0, 0.0, 0.0], 0.018, size=(90, 3))
    right = rng.normal([0.10, 0.0, 0.0], 0.018, size=(90, 3))
    bridge = rng.normal([0.05, 0.0, 0.0], 0.006, size=(12, 3))
    points = np.vstack([left, right, bridge])
    original_indices = np.arange(len(points), dtype=np.int64)
    feature = compute_cluster_features(
        0,
        points,
        original_indices,
        np.full(len(points), 0.08),
        np.full(len(points), 0.05),
        np.full(len(points), 0.8),
    )
    result = ClusterResult(
        labels=np.zeros(len(points), dtype=np.int64),
        features=[feature],
        cluster_points=pd.DataFrame(
            {
                "cluster_id": np.zeros(len(points), dtype=np.int64),
                "original_index": original_indices,
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
            }
        ),
    )
    config = {
        "task_priors": {
            "harvest": {
                "split_merged": True,
                "split_trigger_diameter_m": 0.10,
                "split_max_parts": 3,
                "split_min_points": 40,
                "split_min_center_distance_m": 0.05,
                "split_min_silhouette": 0.3,
                "split_max_subcluster_diameter_m": 0.12,
            }
        }
    }

    split, diagnostics = split_merged_harvest_clusters(
        result,
        original_indices,
        np.full(len(points), 0.08),
        np.full(len(points), 0.05),
        np.full(len(points), 0.8),
        config,
    )

    assert len(split.features) == 2
    assert diagnostics["splits"][0]["accepted"] is True
    sizes = sorted(int(f["n_points"]) for f in split.features)
    assert sizes[0] >= 90
    assert sizes[1] >= 90
