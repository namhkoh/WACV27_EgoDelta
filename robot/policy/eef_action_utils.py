"""Runtime decode for EEF-space action-chunk policies (phase B of the EEF pipeline).

The EEF policy emits absolute future TCP poses (position + axis-angle) in the robot
frame. This module converts each predicted pose into a 7-DoF arm joint target using
the RB-Y1 inverse-kinematics layer (robot.kinematics.ik_solver.solve_tcp_pose_ik,
which solves over RBY1Kinematics FK and already enforces joint limits and
self-clearance). It does NOT hand-roll a Jacobian, so it makes no assumption about
SDK twist conventions; it reuses the validated bounded-least-squares solver.

The IK solver is imported lazily so that pose_from_eef_action stays usable without
the RB-Y1 SDK present.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot.kinematics.geometry import rotation_from_axis_angle


EEF_ACTION_DIM = 6  # [pos_x, pos_y, pos_z (m), aa_rx, aa_ry, aa_rz (rad)]
ARM_DOF = 7


def pose_from_eef_action(action_step: np.ndarray) -> np.ndarray:
    """Reconstruct a 4x4 robot-frame TCP pose from one 6-dim EEF action step."""
    a = np.asarray(action_step, dtype=float).reshape(EEF_ACTION_DIM)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rotation_from_axis_angle(a[3:6])
    pose[:3, 3] = a[:3]
    return pose


def decode_eef_chunk_to_joints(
    *,
    kinematics: Any,
    locked_state: Any,
    q_arm_now: np.ndarray,
    eef_chunk: np.ndarray,
    config: Any | None = None,
    position_tolerance_m: float = 0.01,
    orientation_tolerance_rad: float = 0.15,
    max_steps: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Decode an absolute EEF-pose chunk into an arm joint chunk via per-step IK.

    Each predicted TCP pose is solved with solve_tcp_pose_ik, warm-seeded from the
    previous solved joint configuration so the joint path stays continuous. Returns
    the (chunk_size, 7) joint chunk and a per-step report (IK convergence and
    pose error). Downstream guards in the executor still validate the joint chunk.

    When max_steps is set, only the first max_steps poses are IK-solved (the executor
    only executes/guards a prefix); the remaining rows are padded with the last solved
    configuration so the returned shape is unchanged. This avoids solving the full
    chunk every receding iteration (the dominant per-iteration cost for long chunks).
    """
    from robot.kinematics.ik_solver import solve_tcp_pose_ik
    from robot.kinematics.types import IKSolverConfig

    cfg = config or IKSolverConfig()
    poses = np.asarray(eef_chunk, dtype=float).reshape(-1, EEF_ACTION_DIM)
    total_steps = int(poses.shape[0])
    n_solve = total_steps if max_steps is None else max(1, min(int(max_steps), total_steps))
    q_prev = np.asarray(q_arm_now, dtype=float).reshape(ARM_DOF)

    q_rows: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for step in range(n_solve):
        action = poses[step]
        target_pose = pose_from_eef_action(action)
        result = solve_tcp_pose_ik(
            kinematics,
            locked_state,
            target_pose,
            seed_q_arm_rad=q_prev,
            reference_q_arm_rad=q_prev,
            config=cfg,
        )
        q_solved = np.asarray(result.q_arm_rad, dtype=float).reshape(ARM_DOF)
        step_ok = bool(
            result.ok
            and result.position_error_m <= position_tolerance_m
            and result.orientation_error_rad <= orientation_tolerance_rad
        )
        reports.append(
            {
                "step": int(step),
                "ik_ok": bool(result.ok),
                "within_tolerance": step_ok,
                "position_error_m": float(result.position_error_m),
                "orientation_error_rad": float(result.orientation_error_rad),
                "iterations": int(result.iterations),
            }
        )
        q_rows.append(q_solved)
        q_prev = q_solved  # warm-seed the next step from this solution

    # Pad the un-solved tail (beyond the executed/guarded prefix) with the last solved config.
    while len(q_rows) < total_steps:
        q_rows.append(q_prev.copy())

    q_chunk = np.vstack(q_rows).astype(float)
    return q_chunk, reports


def decode_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-step decode reports into a compact summary for logging/guards."""
    if not reports:
        return {"steps": 0, "all_within_tolerance": False}
    pos = [r["position_error_m"] for r in reports]
    rot = [r["orientation_error_rad"] for r in reports]
    return {
        "steps": len(reports),
        "all_ik_ok": all(r["ik_ok"] for r in reports),
        "all_within_tolerance": all(r["within_tolerance"] for r in reports),
        "max_position_error_m": float(max(pos)),
        "max_orientation_error_rad": float(max(rot)),
    }
