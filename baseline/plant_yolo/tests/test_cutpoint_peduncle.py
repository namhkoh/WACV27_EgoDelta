import numpy as np

from yolo_aria_harvest_poc.config import PipelineConfig
from yolo_aria_harvest_poc.cutpoint_peduncle import (
    estimate_from_keypoints,
    fit_peduncle_curve_pca,
)


def test_peduncle_curve_cut_axis_moves_away_from_tomato():
    tomato_centroid = np.array([0.0, 0.0, 0.0])
    points = np.array([[x, 0.0, 0.0] for x in np.linspace(0.08, 0.20, 20)])

    curve = fit_peduncle_curve_pca(points, tomato_centroid)

    assert curve["tomato_side_endpoint"][0] < curve["stem_side_endpoint"][0]
    assert curve["axis_from_tomato"][0] > 0


def test_keypoint_cutpoint_uses_attachment_and_axis():
    instance = {
        "instance_id": "pre_tomato_000",
        "centroid_3d": [0.0, 0.0, 0.0],
        "radius_m": 0.05,
        "num_frames_observed": 3,
        "num_points": 100,
        "source_frame_ids": ["pre_000001"],
    }
    disappearance = {"final_disappearance_score": 1.0, "visible_post_views": 4}
    peduncle = {
        "det_id": "ped_000",
        "frame_id": "pre_000001",
        "status": "ok",
        "confidence_2d": 0.9,
        "keypoints_3d": [
            {
                "name": "fruit_attachment",
                "xyz_world": [0.06, 0.0, 0.0],
                "status": "ok",
            },
            {
                "name": "stem_attachment",
                "xyz_world": [0.16, 0.0, 0.0],
                "status": "ok",
            },
        ],
    }
    config = PipelineConfig(peduncle_cut_offset_m=0.02, peduncle_association_max_distance_m=0.08)

    result = estimate_from_keypoints(instance, disappearance, [peduncle], None, config)

    assert result is not None
    assert result["cut_point_type"] == "peduncle_keypoint_derived_cut_point"
    assert np.allclose(result["cut_point_3d_world"], [0.08, 0.0, 0.0])
