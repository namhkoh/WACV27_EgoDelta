import numpy as np

from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.mask_lifting import lift_mask_points
from sam_aria_change_baseline.mps_geometry import AriaMpsGeometry, SimplePinholeCamera


def test_synthetic_mask_with_projected_points_lifts_points():
    config = PipelineConfig(
        min_points_inside_mask=3,
        depth_band_m=0.06,
        dbscan_min_samples=2,
        tomato_dbscan_eps_m=0.08,
    )
    camera = SimplePinholeCamera(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
    points = np.array(
        [
            [0.00, 0.00, 1.00],
            [0.01, 0.00, 1.00],
            [0.00, 0.01, 1.00],
            [0.01, 0.01, 1.00],
        ]
    )
    geometry = AriaMpsGeometry.from_synthetic(points, camera, config=config)
    uv, depth, valid = geometry.project_world_points_to_frame(points, 0, 100, 100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:60, 45:60] = True

    result = lift_mask_points(mask, points, uv, depth, valid, config, "harvesting")

    assert result.status == "ok"
    assert result.num_points_after_cluster == 4


def test_background_depth_points_removed_by_foreground_filter():
    config = PipelineConfig(
        min_points_inside_mask=3,
        depth_band_m=0.05,
        dbscan_min_samples=2,
        tomato_dbscan_eps_m=0.08,
    )
    camera = SimplePinholeCamera(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
    foreground = np.array([[0.00, 0.00, 1.00], [0.01, 0.00, 1.00], [0.00, 0.01, 1.00]])
    background = np.array([[0.00, 0.00, 1.30], [0.01, 0.00, 1.30], [0.00, 0.01, 1.30]])
    points = np.vstack([foreground, background])
    geometry = AriaMpsGeometry.from_synthetic(points, camera, config=config)
    uv, depth, valid = geometry.project_world_points_to_frame(points, 0, 100, 100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:60, 45:60] = True

    result = lift_mask_points(mask, points, uv, depth, valid, config, "harvesting")

    assert result.num_points_inside_mask == 6
    assert result.num_points_after_depth_filter == 3
    assert np.all(result.points_3d[:, 2] <= 1.05)

