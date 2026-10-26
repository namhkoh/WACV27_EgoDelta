from __future__ import annotations

import numpy as np

from tomato_cutpoints.targets.rules import infer_targets


def _cluster(cluster_id: int, center: list[float], score: float) -> dict[str, object]:
    center_np = np.asarray(center, dtype=np.float64)
    offsets = np.asarray(
        [
            [-0.02, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, -0.02, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, -0.02],
            [0.0, 0.0, 0.02],
        ],
        dtype=np.float64,
    )
    points = center_np.reshape(1, 3) + offsets
    return {
        "cluster_id": cluster_id,
        "n_points": len(points),
        "points": points,
        "centroid": center_np,
        "bbox_dimensions": points.max(axis=0) - points.min(axis=0),
        "pca_eigenvectors": np.eye(3),
        "approximate_diameter": 0.04,
        "compactness": 1.0,
        "linearity": 1.0,
        "mean_combined_score": score,
        "mean_nn_distance": 0.05,
        "mean_m3c2_distance": 0.05,
    }


def test_infer_targets_returns_multiple_ranked_targets_when_configured():
    config = {
        "task_priors": {
            "harvest": {
                "max_targets": 2,
                "min_target_score": 0.0,
                "min_diameter_m": 0.02,
                "max_diameter_m": 0.08,
            }
        }
    }
    clusters = [
        _cluster(0, [0.0, 0.0, 0.0], 0.8),
        _cluster(1, [0.2, 0.0, 0.0], 0.9),
        _cluster(2, [0.4, 0.0, 0.0], 0.7),
    ]

    targets = infer_targets("harvest", clusters, np.empty((0, 3)), np.empty((0, 3)), config)

    assert [target["cluster_id"] for target in targets] == [1, 0]
