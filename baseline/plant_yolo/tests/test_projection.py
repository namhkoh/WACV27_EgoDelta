import numpy as np

from yolo_aria_harvest_poc.mps_geometry import PinholeCalibration, project_world_points_to_rgb


def test_projection_marks_behind_and_out_of_bounds_invalid():
    calibration = PinholeCalibration(width=100, height=100, fx=100, fy=100, cx=50, cy=50)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [10.0, 0.0, 1.0],
        ]
    )

    uv, depth, valid = project_world_points_to_rgb(points, np.eye(4), calibration)

    assert np.allclose(uv[0], [50.0, 50.0])
    assert np.allclose(uv[1], [70.0, 50.0])
    assert depth[2] < 0
    assert valid.tolist() == [True, True, False, False]

