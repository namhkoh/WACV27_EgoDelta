"""Cartesian retargeting of pi05_droid (Franka) joint-velocity actions to RB-Y1.

pi05_droid / pi0_fast_droid emit an 8-D action = [7 Franka joint velocities,
1 gripper], because they were fine-tuned on DROID (a Franka Panda). RB-Y1's
7-DoF arm has a *different* kinematic chain, so commanding those joint velocities
on RB-Y1 joints is geometrically meaningless and runaway-prone.

This module implements the agreed bridge: interpret the model output through a
**Franka Panda kinematic model** to recover the *intended end-effector twist*
(the task-space motion a Panda would produce), then hand that twist to RB-Y1's
own IK (``solve_tcp_pose_ik``) as an absolute target TCP pose. Task-space intent
is embodiment-agnostic in a way joint angles are not.

ASSUMPTIONS / KNOWN APPROXIMATIONS (this is a zero-shot heuristic, not exact):
  * The 7 joint angles we feed the model as ``observation/joint_position`` are
    RB-Y1's, interpreted as if they were Panda joints; the Jacobian is evaluated
    there. This is inherent to running a Franka policy on a non-Franka arm.
  * The recovered twist is expressed in the Panda base frame and applied directly
    in the RB-Y1 base frame, i.e. we assume the two base frames share orientation
    (z-up, x-forward). Verify/relax before trusting orientation on hardware.
  * The DROID action space is a *normalized* joint velocity in [-1, 1]; the true
    rad/s depends on the DROID controller's velocity scaling, which is not public
    here. ``joint_vel_scale`` is an explicit, conservative knob -- tune it (and
    keep the per-step clamps) before any motion.

FK uses the documented Franka Panda modified-DH parameters; the Jacobian is
computed numerically from FK (convention-robust). Rotations via scipy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation


# Franka Panda modified-DH (Craig) parameters: (a_{i-1}, alpha_{i-1}, d_i) for
# joints 1..7, plus a fixed flange transform (joint 8, theta=0). Source: Franka
# Emika Panda kinematics (frankaemika.github.io control parameters).
_PANDA_MDH = [
    (0.0, 0.0, 0.333),
    (0.0, -np.pi / 2, 0.0),
    (0.0, np.pi / 2, 0.316),
    (0.0825, np.pi / 2, 0.0),
    (-0.0825, -np.pi / 2, 0.384),
    (0.0, np.pi / 2, 0.0),
    (0.088, np.pi / 2, 0.0),
    (0.0, 0.0, 0.107),  # flange (fixed, theta = 0)
]


def _mdh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def panda_fk(q_arm_rad: np.ndarray) -> np.ndarray:
    """Forward kinematics: 4x4 T_base_from_flange for a Franka Panda at q (7,)."""
    q = np.asarray(q_arm_rad, dtype=float).reshape(7)
    T = np.eye(4)
    thetas = list(q) + [0.0]  # flange theta = 0
    for (a, alpha, d), theta in zip(_PANDA_MDH, thetas):
        T = T @ _mdh_transform(a, alpha, d, theta)
    return T


def panda_jacobian(q_arm_rad: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical geometric Jacobian (6x7) at q: rows [vx,vy,vz, wx,wy,wz] = J @ qdot.

    Linear part from finite-differencing the flange origin; angular part from the
    small-rotation skew of dR = R(q+eps) R(q)^T. Base-frame Jacobian.
    """
    q = np.asarray(q_arm_rad, dtype=float).reshape(7)
    T0 = panda_fk(q)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    J = np.zeros((6, 7))
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        Ti = panda_fk(dq)
        J[:3, i] = (Ti[:3, 3] - p0) / eps
        dR = Ti[:3, :3] @ R0.T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2.0 * eps)
    return J


def joint_vel_to_ee_twist(q_arm_rad: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """Map joint velocities (7,) to an end-effector twist (6,) = [v(3); w(3)]."""
    return panda_jacobian(q_arm_rad) @ np.asarray(qdot, dtype=float).reshape(7)


@dataclasses.dataclass
class RetargetResult:
    target_tcp_pose: np.ndarray  # 4x4 absolute target TCP pose, RB-Y1 base frame
    twist_franka: np.ndarray  # 6-vec [v; w] recovered from the Franka model
    delta_trans_m: np.ndarray  # applied per-step translation (after clamp)
    delta_rot_rad: np.ndarray  # applied per-step rotation vector (after clamp)
    trans_clamped: bool
    rot_clamped: bool


def retarget_step(
    *,
    current_tcp_pose: np.ndarray,
    q_arm_rad: np.ndarray,
    joint_velocity: np.ndarray,
    dt_s: float,
    joint_vel_scale: float,
    max_trans_m: float,
    max_rot_rad: float,
) -> RetargetResult:
    """Convert one model joint-velocity step into an absolute target TCP pose.

    current_tcp_pose: 4x4 T_base_from_tcp of RB-Y1 *now* (from FK).
    q_arm_rad: the 7 joint angles fed to the model this tick.
    joint_velocity: model output, 7 normalized joint velocities (the action's
        first 7 dims, already clipped to [-1, 1] by the caller).
    dt_s: control period for one chunk step (DROID ~ 1/15 s).
    joint_vel_scale: rad/s per unit of normalized velocity (conservative knob).
    max_trans_m / max_rot_rad: hard per-step clamps (safety).
    """
    qdot = np.asarray(joint_velocity, dtype=float).reshape(7) * float(joint_vel_scale)
    twist = joint_vel_to_ee_twist(q_arm_rad, qdot)  # [v(m/s); w(rad/s)] in (Franka) base
    v, w = twist[:3], twist[3:]

    delta_trans = v * dt_s
    delta_rot = w * dt_s

    trans_norm = float(np.linalg.norm(delta_trans))
    trans_clamped = trans_norm > max_trans_m
    if trans_clamped and trans_norm > 1e-12:
        delta_trans = delta_trans * (max_trans_m / trans_norm)

    rot_norm = float(np.linalg.norm(delta_rot))
    rot_clamped = rot_norm > max_rot_rad
    if rot_clamped and rot_norm > 1e-12:
        delta_rot = delta_rot * (max_rot_rad / rot_norm)

    current_tcp_pose = np.asarray(current_tcp_pose, dtype=float).reshape(4, 4)
    target = np.eye(4)
    # Apply the (clamped) twist in the RB-Y1 base frame: rotate then translate.
    dR = Rotation.from_rotvec(delta_rot).as_matrix()
    target[:3, :3] = dR @ current_tcp_pose[:3, :3]
    target[:3, 3] = current_tcp_pose[:3, 3] + delta_trans

    return RetargetResult(
        target_tcp_pose=target,
        twist_franka=twist,
        delta_trans_m=delta_trans,
        delta_rot_rad=delta_rot,
        trans_clamped=trans_clamped,
        rot_clamped=rot_clamped,
    )


if __name__ == "__main__":
    # Sanity check: FK reach and Jacobian shape at the Panda "ready" pose.
    q_ready = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
    T = panda_fk(q_ready)
    J = panda_jacobian(q_ready)
    print("FK flange position (m):", np.round(T[:3, 3], 4))
    print("Jacobian shape:", J.shape)
    twist = joint_vel_to_ee_twist(q_ready, np.array([0.1, 0, 0, 0, 0, 0, 0]))
    print("twist for qdot1=0.1:", np.round(twist, 4))
