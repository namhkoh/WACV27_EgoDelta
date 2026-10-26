from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot.kinematics.utils import realsense_click_move_arm as motion
from robot.kinematics.apps import deleaf_click_cut, execute_harvest_plan
from robot.policy import guarded_bc_rollout


class _Joint:
    def set_minimum_time(self, value: float) -> "_Joint":
        self.minimum_time = value
        return self

    def set_position(self, value: np.ndarray) -> "_Joint":
        self.position = np.asarray(value)
        return self

    def set_velocity_limit(self, value: np.ndarray) -> "_Joint":
        self.velocity = np.asarray(value)
        return self

    def set_acceleration_limit(self, value: np.ndarray) -> "_Joint":
        self.acceleration = np.asarray(value)
        return self

    def set_command_header(self, value: object) -> "_Joint":
        self.header = value
        return self


class _Header:
    def set_control_hold_time(self, value: float) -> "_Header":
        self.hold_time = value
        return self


class _Body:
    def __init__(self) -> None:
        self.groups: dict[str, object] = {}

    def set_left_arm_command(self, command: object) -> "_Body":
        self.groups["left_arm"] = command
        return self

    def set_right_arm_command(self, command: object) -> "_Body":
        self.groups["right_arm"] = command
        return self


class _Component:
    def set_body_command(self, body: _Body) -> "_Component":
        self.body = body
        return self


class _Robot:
    def set_command(self, component: _Component) -> "_Robot":
        self.component = component
        return self


def _fake_sdk() -> SimpleNamespace:
    return SimpleNamespace(
        JointPositionCommandBuilder=_Joint,
        CommandHeaderBuilder=_Header,
        BodyComponentBasedCommandBuilder=_Body,
        ComponentBasedCommandBuilder=_Component,
        RobotCommandBuilder=_Robot,
    )


def test_command_arm_only_never_adds_fixed_or_inactive_groups(monkeypatch: object) -> None:
    monkeypatch.setattr(motion, "rby", _fake_sdk())

    left = motion.command_arm_only("left", np.zeros(7), minimum_time_s=1.0)
    right = motion.command_arm_only("right", np.zeros(7), minimum_time_s=1.0)

    assert set(left.component.body.groups) == {"left_arm"}
    assert set(right.component.body.groups) == {"right_arm"}
    assert not hasattr(left.component, "head")
    assert not hasattr(right.component, "head")

def test_direct_harvest_waypoint_commands_only_left_arm(monkeypatch: object) -> None:
    calls: list[tuple[str, np.ndarray, float]] = []

    def build(
        arm: str,
        q_arm: np.ndarray,
        *,
        minimum_time_s: float,
        velocity_limit_rad_s: float | None,
        acceleration_limit_rad_s2: float | None,
    ) -> object:
        assert velocity_limit_rad_s is None
        assert acceleration_limit_rad_s2 is None
        calls.append((arm, np.asarray(q_arm), minimum_time_s))
        return "left-only-command"

    monkeypatch.setattr(execute_harvest_plan, "command_arm_only", build)
    waypoint = {"ik_result": {"q_arm_rad": [0.0] * 7}}

    command = execute_harvest_plan._command_for_waypoint({}, waypoint, 4.0)

    assert command == "left-only-command"
    assert calls[0][0] == "left"
    assert calls[0][2] == 4.0


def test_guarded_rollout_commands_only_selected_arm(monkeypatch: object) -> None:
    calls: list[str] = []

    def build(arm: str, q_arm: np.ndarray, **_kwargs: object) -> object:
        calls.append(arm)
        return "arm-only-command"

    class _Future:
        def get(self) -> str:
            return "feedback"

    class _MotionRobot:
        def send_command(self, command: object, timeout: int) -> _Future:
            assert command == "arm-only-command"
            assert timeout >= 5
            return _Future()

    monkeypatch.setattr(guarded_bc_rollout, "command_arm_only", build)

    feedback = guarded_bc_rollout._send_arm(
        _MotionRobot(),
        {},
        "left",
        np.zeros(7),
        minimum_time_s=1.0,
        velocity_limit_rad_s=0.2,
        acceleration_limit_rad_s2=0.1,
        timeout_s=8.0,
    )

    assert feedback == "feedback"
    assert calls == ["left"]


def test_direct_deleaf_waypoint_commands_only_right_arm(monkeypatch: object) -> None:
    calls: list[str] = []

    def build(arm: str, q_arm: np.ndarray, **_kwargs: object) -> object:
        calls.append(arm)
        return "right-only-command"

    finish_ok = deleaf_click_cut.rby.RobotCommandFeedback.FinishCode.Ok

    class _Future:
        def get(self) -> SimpleNamespace:
            return SimpleNamespace(finish_code=finish_ok)

    class _MotionRobot:
        def send_command(self, command: object, timeout: int) -> _Future:
            assert command == "right-only-command"
            assert timeout >= 5
            return _Future()

    args = SimpleNamespace(
        transit_lift_time_s=1.0,
        safe_entry_time_s=1.0,
        approach_time_s=1.0,
        pre_cut_time_s=0.5,
        cut_time_s=0.04,
        post_cut_lift_time_s=1.0,
        return_transit_time_s=1.0,
        retract_time_s=1.0,
        path_waypoint_time_s=0.8,
        right_velocity_limit_rad_s=4.0,
        right_acceleration_limit_rad_s2=18.0,
        timeout_s=8.0,
        execute_until="cut",
    )
    monkeypatch.setattr(deleaf_click_cut, "command_arm_only", build)

    result = deleaf_click_cut.execute_waypoints(
        _MotionRobot(),
        {},
        [{"name": "cut", "q_arm_rad": [0.0] * 7}],
        args,
    )

    assert result[0]["ok"] is True
    assert calls == ["right"]
