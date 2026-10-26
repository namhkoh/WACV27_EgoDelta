from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
ArmName = Literal["left", "right"]


@dataclass(frozen=True)
class Pose:
    """Rigid pose represented as a 4x4 homogeneous transform."""

    T: Array

    def __post_init__(self) -> None:
        matrix = np.asarray(self.T, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"pose matrix must be 4x4, got {matrix.shape}")
        object.__setattr__(self, "T", matrix)

    @property
    def rotation(self) -> Array:
        return self.T[:3, :3]

    @property
    def translation(self) -> Array:
        return self.T[:3, 3]


@dataclass(frozen=True)
class ToolOffset:
    """TCP offset expressed in the corresponding end-effector link frame."""

    link_name: str
    offset_m: Array

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset_m, dtype=float).reshape(3)
        object.__setattr__(self, "offset_m", offset)

    @property
    def T_link_from_tcp(self) -> Array:
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = self.offset_m
        return transform


@dataclass(frozen=True)
class LockedRobotState:
    """Joint values that stay fixed while one arm is optimized."""

    joint_names: tuple[str, ...]
    full_q_rad: Array
    locked_joint_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        q = np.asarray(self.full_q_rad, dtype=float).reshape(-1)
        if len(self.joint_names) != q.size:
            raise ValueError("joint_names and full_q_rad length mismatch")
        object.__setattr__(self, "full_q_rad", q)


@dataclass(frozen=True)
class ActiveArm:
    """Active arm joint subset for IK."""

    arm: ArmName
    joint_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    eef_link_name: str

    def __post_init__(self) -> None:
        if len(self.joint_indices) != len(self.joint_names):
            raise ValueError("joint_indices and joint_names length mismatch")


@dataclass(frozen=True)
class HandPrior:
    """Confidence-weighted action prior extracted from MPS hand tracking."""

    approach_axis_robot: Array
    jaw_axis_robot: Array
    pull_axis_robot: Array
    confidence: float
    source: str

    def __post_init__(self) -> None:
        for name in ("approach_axis_robot", "jaw_axis_robot", "pull_axis_robot"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(3)
            norm = float(np.linalg.norm(value))
            if norm <= 1e-9:
                raise ValueError(f"{name} must be non-zero")
            object.__setattr__(self, name, value / norm)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class TaskFrameCandidate:
    """Candidate robot-frame grasp/action frame before IK solving."""

    target_position_robot_m: Array
    rotation_robot_from_task: Array
    score: float
    source: str
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.target_position_robot_m, dtype=float).reshape(3)
        rotation = np.asarray(self.rotation_robot_from_task, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_robot_from_task must be 3x3")
        object.__setattr__(self, "target_position_robot_m", position)
        object.__setattr__(self, "rotation_robot_from_task", rotation)

    @property
    def pose(self) -> Pose:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = self.rotation_robot_from_task
        transform[:3, 3] = self.target_position_robot_m
        return Pose(transform)


@dataclass(frozen=True)
class IKSolverConfig:
    """Weights and limits for bounded least-squares IK."""

    position_weight: float = 20.0
    orientation_weight: float = 1.0
    posture_weight: float = 0.01
    smoothness_weight: float = 0.02
    clearance_weight: float = 2.0
    clearance_margin_m: float = 0.03
    clearance_pair_count: int = 10
    joint_limit_margin_rad: float = 0.05
    max_iterations: int = 100
    ftol: float = 1e-6
    xtol: float = 1e-6
    gtol: float = 1e-6


@dataclass(frozen=True)
class IKResult:
    """IK result for one waypoint."""

    ok: bool
    q_arm_rad: Array
    residual_norm: float
    position_error_m: float
    orientation_error_rad: float
    message: str
    iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_arm_rad", np.asarray(self.q_arm_rad, dtype=float).reshape(-1))
