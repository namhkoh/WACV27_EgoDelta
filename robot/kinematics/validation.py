from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.types import Array, LockedRobotState


@dataclass(frozen=True)
class TCPStateReport:
    tcp_position_robot_m: tuple[float, float, float]
    eef_position_robot_m: tuple[float, float, float]
    tool_offset_norm_m: float


@dataclass(frozen=True)
class JacobianValidationReport:
    ok: bool
    max_abs_error: float
    rms_error: float
    jacobian_shape: tuple[int, int]
    note: str


def tcp_state_report(kinematics: RBY1Kinematics, q_full_rad: Array) -> TCPStateReport:
    eef_pose = kinematics.eef_pose(q_full_rad)
    tcp_pose = kinematics.tcp_pose(q_full_rad)
    return TCPStateReport(
        tcp_position_robot_m=tuple(float(x) for x in tcp_pose[:3, 3]),
        eef_position_robot_m=tuple(float(x) for x in eef_pose[:3, 3]),
        tool_offset_norm_m=float(np.linalg.norm(tcp_pose[:3, 3] - eef_pose[:3, 3])),
    )


def validate_finite_difference_tcp_jacobian(
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    q_arm_rad: Array,
    *,
    step_rad: float = 1e-5,
    max_abs_tolerance: float = 1e-4,
) -> JacobianValidationReport:
    """Check finite-difference TCP pose Jacobian stability across step sizes."""

    coarse = kinematics.finite_difference_tcp_pose_jacobian(
        locked_state,
        q_arm_rad,
        step_rad=step_rad,
    )
    fine = kinematics.finite_difference_tcp_pose_jacobian(
        locked_state,
        q_arm_rad,
        step_rad=step_rad * 0.5,
    )
    diff = coarse - fine
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff * diff)))
    return JacobianValidationReport(
        ok=max_abs <= float(max_abs_tolerance),
        max_abs_error=max_abs,
        rms_error=rms,
        jacobian_shape=tuple(int(x) for x in coarse.shape),
        note=(
            "This validates the 6D finite-difference TCP Jacobian used by the "
            "first IK solver: position rows plus body-frame axis-angle "
            "orientation residual rows. SDK analytic Jacobian conventions are "
            "validated separately before analytic derivatives are enabled."
        ),
    )
