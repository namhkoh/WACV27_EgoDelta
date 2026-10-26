import json
from pathlib import Path

import numpy as np

from egoscene.prepare import _nerfstudio_frame


def test_nerfstudio_pose_flips_opencv_yz_axes():
    camera = {
        "rotation": np.eye(3).tolist(),
        "position": [1.0, 2.0, 3.0],
    }
    matrix = np.asarray(_nerfstudio_frame("rgb/frame.jpg", camera)["transform_matrix"])
    assert np.allclose(matrix[:3, 0], [1, 0, 0])
    assert np.allclose(matrix[:3, 1], [0, -1, 0])
    assert np.allclose(matrix[:3, 2], [0, 0, -1])
    assert np.allclose(matrix[:3, 3], [1, 2, 3])
