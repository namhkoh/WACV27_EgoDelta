from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def normalize(vector: Array, *, eps: float = 1e-9) -> Array:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise ValueError("cannot normalize near-zero vector")
    return value / norm


def skew(vector: Array) -> Array:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def assert_transform(transform: Array, *, name: str = "transform", atol: float = 1e-6) -> None:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        raise ValueError(f"{name} bottom row is not homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError(f"{name} rotation is not orthonormal")
    det = float(np.linalg.det(rotation))
    if abs(det - 1.0) > atol:
        raise ValueError(f"{name} rotation determinant is {det:.6f}, expected 1")


def transform_from_rotation_translation(rotation: Array, translation: Array) -> Array:
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform


def axis_angle_from_rotation(rotation: Array) -> Array:
    """Return the rotation-vector log map for a 3x3 rotation matrix."""

    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    cos_angle = max(-1.0, min(1.0, float((np.trace(R) - 1.0) / 2.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)
    if abs(math.pi - angle) < 1e-5:
        axis = np.array(
            [
                math.sqrt(max(0.0, (R[0, 0] + 1.0) / 2.0)),
                math.sqrt(max(0.0, (R[1, 1] + 1.0) / 2.0)),
                math.sqrt(max(0.0, (R[2, 2] + 1.0) / 2.0)),
            ],
            dtype=float,
        )
        pivot = int(np.argmax(axis))
        if axis[pivot] <= 1e-9:
            return np.array([math.pi, 0.0, 0.0], dtype=float)
        if pivot == 0:
            axis[1] = (R[0, 1] + R[1, 0]) / (4.0 * axis[0])
            axis[2] = (R[0, 2] + R[2, 0]) / (4.0 * axis[0])
        elif pivot == 1:
            axis[0] = (R[0, 1] + R[1, 0]) / (4.0 * axis[1])
            axis[2] = (R[1, 2] + R[2, 1]) / (4.0 * axis[1])
        else:
            axis[0] = (R[0, 2] + R[2, 0]) / (4.0 * axis[2])
            axis[1] = (R[1, 2] + R[2, 1]) / (4.0 * axis[2])
        return normalize(axis) * angle
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    )
    return axis * (angle / (2.0 * math.sin(angle)))


def rotation_from_axis_angle(axis_angle: Array) -> Array:
    """Exp map: a rotation vector (axis-angle) back to a 3x3 rotation matrix.

    Inverse of axis_angle_from_rotation, via Rodrigues' formula. Used by the EEF-space
    action decode and orientation-error evaluation.
    """

    omega = np.asarray(axis_angle, dtype=float).reshape(3)
    angle = float(np.linalg.norm(omega))
    if angle < 1e-9:
        return np.eye(3, dtype=float)
    K = skew(omega / angle)
    return np.eye(3, dtype=float) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def orientation_residual(current_rotation: Array, target_rotation: Array) -> Array:
    """Body-frame axis-angle orientation error.

    This returns log(R_target.T @ R_current). The convention is deliberately
    documented because analytic Jacobians must use the same body-frame
    orientation error. The current finite-difference IK path only requires this
    residual to be smooth and zero at the target.
    """

    R_current = np.asarray(current_rotation, dtype=float)
    R_target = np.asarray(target_rotation, dtype=float)
    return axis_angle_from_rotation(R_target.T @ R_current)
