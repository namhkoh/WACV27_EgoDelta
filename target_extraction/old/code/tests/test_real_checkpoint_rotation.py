from pathlib import Path

import numpy as np
import pytest

from egodelta_diff.data import load_2dgs_ply


REAL_CHECKPOINT = Path(
    "/path/to/data/mps/egodelta_benchmark_data/pre_models/"
    "egodelta_16b2d1bf9fa9_leaf/point_cloud/iteration_30000/point_cloud.ply"
)


@pytest.mark.integration
def test_real_wxyz_quaternions_produce_orthonormal_rotations_and_unit_normals():
    if not REAL_CHECKPOINT.exists():
        pytest.skip("real benchmark checkpoint is unavailable")
    gaussian = load_2dgs_ply(REAL_CHECKPOINT, device="cpu")
    rotation = gaussian.rotation.double().numpy()
    identity = np.broadcast_to(np.eye(3), rotation.shape)
    np.testing.assert_allclose(rotation.transpose(0, 2, 1) @ rotation, identity, atol=2e-5)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(gaussian.normal.numpy(), axis=1), 1.0, atol=2e-5)
    np.testing.assert_allclose(gaussian.normal.numpy(), gaussian.rotation[:, :, 2].numpy(), atol=1e-6)
