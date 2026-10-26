from types import SimpleNamespace

import numpy as np
import pytest

from egoscene.egodelta import projection_components


def test_projection_components_maps_cache_indices_to_original_ply_rows() -> None:
    fields = {
        "x": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        "y": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "z": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "world_scale": np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        "alpha": np.asarray([0.5, 0.6, 0.7], dtype=np.float32),
    }
    cache_mu = np.asarray([[2.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    target = SimpleNamespace(primitive_indices=np.asarray([1, 0], dtype=np.int64))

    result = projection_components([target], cache_mu, np.asarray([2, 0]), fields)

    np.testing.assert_allclose(result[0]["xyz"][:, 0], [0.0, 2.0])
    np.testing.assert_allclose(result[0]["world_scale"], [0.01, 0.03])


def test_projection_components_rejects_a_stale_source_mapping() -> None:
    fields = {
        "x": np.asarray([0.0], dtype=np.float32),
        "y": np.asarray([0.0], dtype=np.float32),
        "z": np.asarray([1.0], dtype=np.float32),
        "world_scale": np.asarray([0.01], dtype=np.float32),
        "alpha": np.asarray([1.0], dtype=np.float32),
    }
    target = SimpleNamespace(primitive_indices=np.asarray([0], dtype=np.int64))

    with pytest.raises(ValueError, match="source-index mapping"):
        projection_components(
            [target],
            np.asarray([[0.1, 0.0, 1.0]], dtype=np.float32),
            np.asarray([0]),
            fields,
        )
