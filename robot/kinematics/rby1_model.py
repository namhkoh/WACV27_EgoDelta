from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby

from robot.kinematics.geometry import assert_transform, orientation_residual
from robot.kinematics.types import ActiveArm, ArmName, Array, LockedRobotState, ToolOffset


BASE_LINK = "base"
LEFT_EEF_LINK = "ee_left"
RIGHT_EEF_LINK = "ee_right"


@dataclass(frozen=True)
class RBY1Kinematics:
    """Thin RB-Y1 SDK wrapper for FK, limits, and Jacobians."""

    dyn_robot: Any
    joint_names: tuple[str, ...]
    active_arm: ActiveArm
    tool_offset: ToolOffset

    @classmethod
    def from_live_robot(
        cls,
        *,
        address: str,
        model_name: str,
        arm: ArmName,
        toolfit_path: Path,
    ) -> "RBY1Kinematics":
        robot = rby.create_robot(address, model_name)
        if not robot.connect():
            raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
        model = robot.model()
        return cls.from_sdk_model(robot.get_dynamics(), model, arm=arm, toolfit_path=toolfit_path)

    @classmethod
    def from_urdf(
        cls,
        *,
        urdf_path: Path,
        model_name: str,
        arm: ArmName,
        toolfit_path: Path,
        base_link_name: str = BASE_LINK,
    ) -> "RBY1Kinematics":
        model = sdk_model_from_name(model_name)
        configuration = rby.dynamics.load_robot_from_urdf(str(Path(urdf_path).resolve()), base_link_name)
        dyn_robot = rby.dynamics.Robot(configuration)
        return cls.from_sdk_model(dyn_robot, model, arm=arm, toolfit_path=toolfit_path)

    @classmethod
    def from_sdk_model(
        cls,
        dyn_robot: Any,
        model: Any,
        *,
        arm: ArmName,
        toolfit_path: Path,
    ) -> "RBY1Kinematics":
        joint_names = tuple(str(name) for name in model.robot_joint_names)
        if arm == "left":
            indices = tuple(int(i) for i in model.left_arm_idx)
            eef_link = LEFT_EEF_LINK
        else:
            indices = tuple(int(i) for i in model.right_arm_idx)
            eef_link = RIGHT_EEF_LINK
        active_arm = ActiveArm(
            arm=arm,
            joint_indices=indices,
            joint_names=tuple(joint_names[i] for i in indices),
            eef_link_name=eef_link,
        )
        tool_offset = load_tool_offset(toolfit_path, expected_link=eef_link)
        return cls(
            dyn_robot=dyn_robot,
            joint_names=joint_names,
            active_arm=active_arm,
            tool_offset=tool_offset,
        )

    def locked_state_from_home(self, home_path: Path) -> LockedRobotState:
        home = json.loads(home_path.read_text(encoding="utf-8"))
        q = np.asarray(home["all_joints"]["positions_rad"], dtype=float)
        locked = tuple(i for i in range(q.size) if i not in set(self.active_arm.joint_indices))
        return LockedRobotState(
            joint_names=tuple(home["joint_names"]),
            full_q_rad=q,
            locked_joint_indices=locked,
        )

    def q_full_from_arm(self, locked_state: LockedRobotState, q_arm_rad: Array) -> Array:
        q_full = np.asarray(locked_state.full_q_rad, dtype=float).copy()
        q_arm = np.asarray(q_arm_rad, dtype=float).reshape(len(self.active_arm.joint_indices))
        for idx, value in zip(self.active_arm.joint_indices, q_arm):
            q_full[idx] = float(value)
        return q_full

    def make_state(self, q_full_rad: Array, links: tuple[str, ...] | None = None) -> Any:
        link_names = links or (BASE_LINK, self.active_arm.eef_link_name)
        state = self.dyn_robot.make_state(list(link_names), list(self.joint_names))
        state.set_q(np.asarray(q_full_rad, dtype=float))
        self.dyn_robot.compute_forward_kinematics(state)
        return state

    def eef_pose(self, q_full_rad: Array) -> Array:
        links = (BASE_LINK, self.active_arm.eef_link_name)
        state = self.make_state(q_full_rad, links=links)
        transform = np.asarray(self.dyn_robot.compute_transformation(state, 0, 1), dtype=float)
        assert_transform(transform, name=f"T_{BASE_LINK}_from_{self.active_arm.eef_link_name}")
        return transform

    def tcp_pose(self, q_full_rad: Array) -> Array:
        transform = self.eef_pose(q_full_rad) @ self.tool_offset.T_link_from_tcp
        assert_transform(transform, name=f"T_{BASE_LINK}_from_tcp")
        return transform

    def active_joint_limits(self, q_full_rad: Array | None = None) -> tuple[Array, Array]:
        if q_full_rad is None:
            q_full_rad = np.zeros(len(self.joint_names), dtype=float)
        state = self.make_state(q_full_rad)
        lower = np.asarray(self.dyn_robot.get_limit_q_lower(state), dtype=float).reshape(-1)
        upper = np.asarray(self.dyn_robot.get_limit_q_upper(state), dtype=float).reshape(-1)
        idx = list(self.active_arm.joint_indices)
        return lower[idx], upper[idx]

    def sdk_space_jacobian_eef(self, q_full_rad: Array) -> Array:
        links = (BASE_LINK, self.active_arm.eef_link_name)
        state = self.make_state(q_full_rad, links=links)
        jacobian = np.asarray(self.dyn_robot.compute_space_jacobian(state, 0, 1), dtype=float)
        return jacobian[:, list(self.active_arm.joint_indices)]

    def finite_difference_tcp_position_jacobian(
        self,
        locked_state: LockedRobotState,
        q_arm_rad: Array,
        *,
        step_rad: float = 1e-5,
    ) -> Array:
        q_arm = np.asarray(q_arm_rad, dtype=float).reshape(len(self.active_arm.joint_indices))
        jacobian = np.zeros((3, q_arm.size), dtype=float)
        for col in range(q_arm.size):
            dq = np.zeros_like(q_arm)
            dq[col] = float(step_rad)
            plus = self.tcp_pose(self.q_full_from_arm(locked_state, q_arm + dq))[:3, 3]
            minus = self.tcp_pose(self.q_full_from_arm(locked_state, q_arm - dq))[:3, 3]
            jacobian[:, col] = (plus - minus) / (2.0 * float(step_rad))
        return jacobian

    def finite_difference_tcp_pose_jacobian(
        self,
        locked_state: LockedRobotState,
        q_arm_rad: Array,
        *,
        target_tcp_pose: Array | None = None,
        step_rad: float = 1e-5,
    ) -> Array:
        q_arm = np.asarray(q_arm_rad, dtype=float).reshape(len(self.active_arm.joint_indices))
        if target_tcp_pose is None:
            target = self.tcp_pose(self.q_full_from_arm(locked_state, q_arm))
        else:
            target = np.asarray(target_tcp_pose, dtype=float)
            assert_transform(target, name="target_tcp_pose")
        jacobian = np.zeros((6, q_arm.size), dtype=float)
        for col in range(q_arm.size):
            dq = np.zeros_like(q_arm)
            dq[col] = float(step_rad)
            plus_pose = self.tcp_pose(self.q_full_from_arm(locked_state, q_arm + dq))
            minus_pose = self.tcp_pose(self.q_full_from_arm(locked_state, q_arm - dq))
            plus = np.concatenate(
                [
                    plus_pose[:3, 3] - target[:3, 3],
                    orientation_residual(plus_pose[:3, :3], target[:3, :3]),
                ]
            )
            minus = np.concatenate(
                [
                    minus_pose[:3, 3] - target[:3, 3],
                    orientation_residual(minus_pose[:3, :3], target[:3, :3]),
                ]
            )
            jacobian[:, col] = (plus - minus) / (2.0 * float(step_rad))
        return jacobian


def load_tool_offset(toolfit_path: Path, *, expected_link: str) -> ToolOffset:
    data = json.loads(Path(toolfit_path).read_text(encoding="utf-8"))
    link_name = str(data["estimated_tool_offset_link_name"])
    if link_name != expected_link:
        raise ValueError(f"toolfit link {link_name!r} does not match expected {expected_link!r}")
    return ToolOffset(
        link_name=link_name,
        offset_m=np.asarray(data["estimated_tool_offset_m"], dtype=float),
    )


def sdk_model_from_name(model_name: str) -> Any:
    normalized = str(model_name).strip().lower()
    if normalized in ("a", "model_a"):
        return rby.Model_A()
    if normalized in ("m", "model_m"):
        return rby.Model_M()
    if normalized in ("ub", "model_ub"):
        return rby.Model_UB()
    raise ValueError(f"unsupported RB-Y1 model name: {model_name!r}")
