from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from robot.kinematics.clearance import clearance_residuals
from robot.kinematics.geometry import assert_transform, orientation_residual
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.types import Array, IKResult, IKSolverConfig, LockedRobotState


def joint_limits_with_margin(
    lower_rad: Array,
    upper_rad: Array,
    margin_rad: float,
) -> tuple[Array, Array]:
    lower = np.asarray(lower_rad, dtype=float).reshape(-1)
    upper = np.asarray(upper_rad, dtype=float).reshape(-1)
    margin = float(margin_rad)
    if margin < 0.0:
        raise ValueError("joint-limit margin must be non-negative")
    bounded_lower = lower + margin
    bounded_upper = upper - margin
    invalid = np.flatnonzero(bounded_lower >= bounded_upper)
    if invalid.size:
        raise ValueError(
            "joint-limit margin removes the feasible range for joints "
            + ",".join(str(int(index)) for index in invalid)
        )
    return bounded_lower, bounded_upper


def solve_tcp_pose_ik(
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    target_tcp_pose: Array,
    *,
    seed_q_arm_rad: Array | None = None,
    reference_q_arm_rad: Array | None = None,
    config: IKSolverConfig | None = None,
    enforce_joint_limit_margin: bool = False,
) -> IKResult:
    """Solve one bounded TCP pose IK waypoint for the active arm.

    The first implementation intentionally relies on scipy's finite-difference
    derivatives over the full calibrated TCP FK chain. This is slower than an
    analytic Jacobian, but it avoids frame/order mistakes while the solver is
    being validated.
    """

    cfg = config or IKSolverConfig()
    target = np.asarray(target_tcp_pose, dtype=float)
    assert_transform(target, name="target_tcp_pose")

    home_q = np.asarray(locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)], dtype=float)
    if seed_q_arm_rad is None:
        seed = home_q.copy()
    else:
        seed = np.asarray(seed_q_arm_rad, dtype=float).reshape(home_q.shape)
    if reference_q_arm_rad is None:
        reference = home_q.copy()
    else:
        reference = np.asarray(reference_q_arm_rad, dtype=float).reshape(home_q.shape)

    lower, upper = kinematics.active_joint_limits(locked_state.full_q_rad)
    if enforce_joint_limit_margin:
        lower, upper = joint_limits_with_margin(lower, upper, cfg.joint_limit_margin_rad)
    seed = np.clip(seed, lower, upper)

    def residual(q_arm: Array) -> Array:
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        current = kinematics.tcp_pose(q_full)
        position_error = current[:3, 3] - target[:3, 3]
        rotation_error = orientation_residual(current[:3, :3], target[:3, :3])
        posture_error = q_arm - home_q
        smoothness_error = q_arm - reference
        return np.concatenate(
            [
                cfg.position_weight * position_error,
                cfg.orientation_weight * rotation_error,
                cfg.posture_weight * posture_error,
                cfg.smoothness_weight * smoothness_error,
                cfg.clearance_weight
                * clearance_residuals(
                    kinematics,
                    q_full,
                    margin_m=cfg.clearance_margin_m,
                    pair_count=cfg.clearance_pair_count,
                ),
            ]
        )

    result = least_squares(
        residual,
        seed,
        bounds=(lower, upper),
        method="trf",
        max_nfev=int(cfg.max_iterations) * (seed.size + 1),
        ftol=float(cfg.ftol),
        xtol=float(cfg.xtol),
        gtol=float(cfg.gtol),
    )
    q_solution = np.asarray(result.x, dtype=float)
    final_pose = kinematics.tcp_pose(kinematics.q_full_from_arm(locked_state, q_solution))
    position_error_m = float(np.linalg.norm(final_pose[:3, 3] - target[:3, 3]))
    orientation_error_rad = float(np.linalg.norm(orientation_residual(final_pose[:3, :3], target[:3, :3])))
    return IKResult(
        ok=bool(result.success),
        q_arm_rad=q_solution,
        residual_norm=float(np.linalg.norm(result.fun)),
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        message=str(result.message),
        iterations=int(result.nfev),
    )


def solve_tcp_task_axis_ik(
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    target_tcp_pose: Array,
    *,
    jaw_axis_robot: Array,
    backoff_axis_robot: Array,
    gripper_jaw_axis_local: str | Array,
    gripper_backoff_axis_local: str | Array,
    seed_q_arm_rad: Array | None = None,
    reference_q_arm_rad: Array | None = None,
    config: IKSolverConfig | None = None,
) -> IKResult:
    """Solve IK with position plus task-relevant gripper axes.

    This is used for round-fruit contact waypoints where wrist roll is not a
    physical task constraint. The jaw and backoff axes are still explicit, so
    the gripper does not become orientation-free.
    """

    cfg = config or IKSolverConfig()
    target = np.asarray(target_tcp_pose, dtype=float)
    assert_transform(target, name="target_tcp_pose")
    target_jaw_axis = _unit_axis(jaw_axis_robot, "jaw_axis_robot")
    target_backoff_axis = _unit_axis(backoff_axis_robot, "backoff_axis_robot")

    home_q = np.asarray(locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)], dtype=float)
    seed = home_q.copy() if seed_q_arm_rad is None else np.asarray(seed_q_arm_rad, dtype=float).reshape(home_q.shape)
    reference = (
        home_q.copy()
        if reference_q_arm_rad is None
        else np.asarray(reference_q_arm_rad, dtype=float).reshape(home_q.shape)
    )

    lower, upper = kinematics.active_joint_limits(locked_state.full_q_rad)
    seed = np.clip(seed, lower, upper)

    def residual(q_arm: Array) -> Array:
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        current = kinematics.tcp_pose(q_full)
        current_rotation = current[:3, :3]
        position_error = current[:3, 3] - target[:3, 3]
        jaw_axis_error = np.cross(
            _tcp_local_axis(current_rotation, gripper_jaw_axis_local),
            target_jaw_axis,
        )
        backoff_axis_error = np.cross(
            _tcp_local_axis(current_rotation, gripper_backoff_axis_local),
            target_backoff_axis,
        )
        posture_error = q_arm - home_q
        smoothness_error = q_arm - reference
        return np.concatenate(
            [
                cfg.position_weight * position_error,
                cfg.orientation_weight * jaw_axis_error,
                cfg.orientation_weight * backoff_axis_error,
                cfg.posture_weight * posture_error,
                cfg.smoothness_weight * smoothness_error,
                cfg.clearance_weight
                * clearance_residuals(
                    kinematics,
                    q_full,
                    margin_m=cfg.clearance_margin_m,
                    pair_count=cfg.clearance_pair_count,
                ),
            ]
        )

    result = least_squares(
        residual,
        seed,
        bounds=(lower, upper),
        method="trf",
        max_nfev=int(cfg.max_iterations) * (seed.size + 1),
        ftol=float(cfg.ftol),
        xtol=float(cfg.xtol),
        gtol=float(cfg.gtol),
    )
    q_solution = np.asarray(result.x, dtype=float)
    final_pose = kinematics.tcp_pose(kinematics.q_full_from_arm(locked_state, q_solution))
    final_rotation = final_pose[:3, :3]
    position_error_m = float(np.linalg.norm(final_pose[:3, 3] - target[:3, 3]))
    orientation_error_rad = max(
        _axis_error_rad(_tcp_local_axis(final_rotation, gripper_jaw_axis_local), target_jaw_axis),
        _axis_error_rad(_tcp_local_axis(final_rotation, gripper_backoff_axis_local), target_backoff_axis),
    )
    return IKResult(
        ok=bool(result.success),
        q_arm_rad=q_solution,
        residual_norm=float(np.linalg.norm(result.fun)),
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        message=str(result.message),
        iterations=int(result.nfev),
    )


def _tcp_local_axis(rotation_robot_from_tcp: Array, local_axis: str | Array) -> Array:
    rotation = np.asarray(rotation_robot_from_tcp, dtype=float).reshape(3, 3)
    if not isinstance(local_axis, str):
        return rotation @ _unit_axis(local_axis, "TCP local axis")
    axis_name = local_axis.lower().strip()
    mapping = {
        "plus-x": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "minus-x": np.asarray([-1.0, 0.0, 0.0], dtype=float),
        "plus-y": np.asarray([0.0, 1.0, 0.0], dtype=float),
        "minus-y": np.asarray([0.0, -1.0, 0.0], dtype=float),
        "plus-z": np.asarray([0.0, 0.0, 1.0], dtype=float),
        "minus-z": np.asarray([0.0, 0.0, -1.0], dtype=float),
    }
    if axis_name not in mapping:
        raise ValueError(f"unsupported TCP local axis {local_axis!r}")
    return rotation @ mapping[axis_name]


def _unit_axis(vector: Array, name: str) -> Array:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def _axis_error_rad(actual: Array, expected: Array) -> float:
    dot = float(np.dot(_unit_axis(actual, "actual axis"), _unit_axis(expected, "expected axis")))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))
