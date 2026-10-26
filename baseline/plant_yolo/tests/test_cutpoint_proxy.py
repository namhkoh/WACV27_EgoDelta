import numpy as np

from yolo_aria_harvest_poc.config import PipelineConfig
from yolo_aria_harvest_poc.cutpoint_proxy import compute_proxy_cut_point, estimate_proxy_cut_points


def test_proxy_cut_point_is_above_tomato_centroid():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(500, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    instance = {
        "instance_id": "pre_tomato_000",
        "centroid_3d": [0.0, 0.0, 0.0],
        "radius_m": 1.0,
        "num_frames_observed": 5,
        "num_points": len(points),
        "mean_confidence_2d": 0.9,
    }
    disappearance = {
        "final_disappearance_score": 0.8,
        "visible_post_views": 4,
    }

    result = compute_proxy_cut_point(points, instance, disappearance, PipelineConfig())

    assert result["cut_point_type"] == "proxy_cut_point_from_tomato_top_surface"
    assert result["cut_point_3d_world"][2] > result["tomato_centroid_3d_world"][2]
    assert result["top_surface_center_3d_world"][2] > 0


def test_proxy_cut_points_are_estimated_per_disappeared_tomato(tmp_path):
    points_a = np.array(
        [
            [-0.02, 0.00, -0.01],
            [0.00, 0.02, 0.02],
            [0.02, 0.00, 0.04],
            [0.00, -0.02, 0.05],
        ],
        dtype=float,
    )
    points_b = points_a + np.array([0.30, 0.10, 0.00])
    np.save(tmp_path / "tomato_a.npy", points_a)
    np.save(tmp_path / "tomato_b.npy", points_b)
    instances = [
        {
            "instance_id": "pre_tomato_000",
            "centroid_3d": [0.0, 0.0, 0.0],
            "radius_m": 0.04,
            "num_frames_observed": 4,
            "num_points": len(points_a),
            "mean_confidence_2d": 0.8,
            "points_3d_path": "tomato_a.npy",
        },
        {
            "instance_id": "pre_tomato_001",
            "centroid_3d": [0.3, 0.1, 0.0],
            "radius_m": 0.04,
            "num_frames_observed": 4,
            "num_points": len(points_b),
            "mean_confidence_2d": 0.8,
            "points_3d_path": "tomato_b.npy",
        },
    ]
    disappearance = [
        {
            "pre_instance_id": "pre_tomato_000",
            "status": "disappeared",
            "final_disappearance_score": 0.9,
            "visible_post_views": 5,
        },
        {
            "pre_instance_id": "pre_tomato_001",
            "status": "disappeared",
            "final_disappearance_score": 0.9,
            "visible_post_views": 5,
        },
    ]

    cut_points = estimate_proxy_cut_points(disappearance, instances, tmp_path, PipelineConfig())

    assert {item["pre_instance_id"] for item in cut_points} == {
        "pre_tomato_000",
        "pre_tomato_001",
    }
    assert len({tuple(item["cut_point_3d_world"]) for item in cut_points}) == 2
