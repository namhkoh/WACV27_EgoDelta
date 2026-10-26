from __future__ import annotations

from pathlib import Path

import numpy as np


def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def load_transform(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform must be a 4 x 4 matrix: {path}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"Transform last row must be [0, 0, 0, 1]: {path}")
    return matrix


def apply_transform(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return xyz.reshape(0, 3).copy()
    homog = np.ones((len(xyz), 4), dtype=np.float64)
    homog[:, :3] = xyz
    return (homog @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def best_fit_transform(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both be N x 3 arrays")
    if len(source_xyz) < 3:
        return identity_transform()

    source_centroid = source_xyz.mean(axis=0)
    target_centroid = target_xyz.mean(axis=0)
    src_centered = source_xyz - source_centroid
    tgt_centered = target_xyz - target_centroid
    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_centroid - r @ source_centroid
    transform = identity_transform()
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform
