import numpy as np

from yolo_aria_harvest_poc.mps_geometry import PinholeCalibration
from yolo_aria_harvest_poc.visibility import count_visible_views


class FakeGeometry:
    rgb_camera_calibration = PinholeCalibration(width=100, height=100, fx=100, fy=100, cx=50, cy=50)

    def get_rgb_camera_pose_at_timestamp(self, _timestamp_ns):
        return np.eye(4)


def test_visibility_counts_inside_and_rejects_outside():
    frames = [{"frame_id": "post_000001", "timestamp_ns": 1}]

    inside = count_visible_views(
        {"centroid_3d": [0.0, 0.0, 1.0], "radius_m": 0.05},
        frames,
        FakeGeometry(),
    )
    outside = count_visible_views(
        {"centroid_3d": [10.0, 0.0, 1.0], "radius_m": 0.05},
        frames,
        FakeGeometry(),
    )

    assert len(inside) == 1
    assert len(outside) == 0
