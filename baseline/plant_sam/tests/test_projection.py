import numpy as np

from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.mps_geometry import AriaMpsGeometry, SimplePinholeCamera


def test_synthetic_points_in_front_project_valid():
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 2.0]])
    camera = SimplePinholeCamera(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
    geometry = AriaMpsGeometry.from_synthetic(points, camera, config=PipelineConfig())

    uv, depth, valid = geometry.project_world_points_to_frame(points, 0, 100, 100)

    assert valid.tolist() == [True, True, True]
    assert np.allclose(uv[0], [50, 50])
    assert np.all(depth > 0)


def test_points_behind_camera_invalid():
    points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    camera = SimplePinholeCamera(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
    geometry = AriaMpsGeometry.from_synthetic(points, camera, config=PipelineConfig())

    _, _, valid = geometry.project_world_points_to_frame(points, 0, 100, 100)

    assert valid.tolist() == [False, True]

