from __future__ import annotations

import numpy as np


def voxel_downsample_indices(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(xyz) == 0:
        return np.empty(0, dtype=np.int64)
    if voxel_size <= 0:
        return np.arange(len(xyz), dtype=np.int64)
    voxels = np.floor(np.asarray(xyz, dtype=float) / voxel_size).astype(np.int64)
    _, first_indices = np.unique(voxels, axis=0, return_index=True)
    return np.sort(first_indices.astype(np.int64))
