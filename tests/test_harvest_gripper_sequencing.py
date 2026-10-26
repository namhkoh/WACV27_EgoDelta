from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot.kinematics.apps import execute_harvest_plan as executor


def _state_stream(states):
    remaining = list(states)

    def read(*_args, **_kwargs):
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return read


class _RobotStateStream:
    def __init__(self, states):
        self._read = _state_stream(states)

    def get_state(self):
        return self._read()


def _arm_home_and_waypoint():
    joint_names = [f"left_{index}" for index in range(7)]
    home = {
        "joint_names": joint_names,
        "all_joints": {"positions_rad": [0.0] * 7},
        "arms": {"left": {"joint_names": joint_names}},
    }
    waypoint = {
        "name": "grasp",
        "ik_result": {"q_arm_rad": [1.0] * 7},
    }
    return home, waypoint


def _arm_state(position: float, velocity: float):
    return SimpleNamespace(
        position=[position] * 7,
        velocity=[velocity] * 7,
    )


def test_open_waits_for_target_and_consecutive_settled_samples(monkeypatch) -> None:
    monkeypatch.setattr(
        executor,
        "_gripper_motor_state",
        _state_stream(
            [
                {"position": -3.9, "velocity": -7.0, "current": -0.02},
                {"position": -4.49, "velocity": 0.0, "current": -0.01},
                {"position": -4.49, "velocity": 0.0, "current": -0.01},
            ]
        ),
    )

    result = executor._wait_for_gripper_open_settled(
        "http://unused",
        arm="left",
        expected_reached_position=-4.50,
        tolerance=0.25,
        timeout_s=1.0,
        poll_s=0.001,
        request_timeout_s=0.1,
        settle_velocity=0.05,
        settle_samples=2,
    )

    assert result["ok"]
    assert result["reason"] == "open_target_settled"
    assert result["sample_count"] == 3


def test_close_does_not_complete_while_motor_is_moving(monkeypatch) -> None:
    monkeypatch.setattr(
        executor,
        "_gripper_motor_state",
        _state_stream(
            [
                {"position": 3.38, "velocity": 7.9, "current": 0.06},
                {"position": 4.56, "velocity": 0.0, "current": 0.15},
                {"position": 4.56, "velocity": 0.0, "current": 0.15},
            ]
        ),
    )

    result = executor._wait_for_gripper_close_settled(
        "http://unused",
        arm="left",
        command_position=4.57,
        before_position=-4.50,
        min_motion=0.25,
        timeout_s=1.0,
        poll_s=0.001,
        request_timeout_s=0.1,
        settle_velocity=0.05,
        settle_samples=2,
        target_tolerance=0.25,
        min_progress_fraction=0.20,
        contact_current_threshold=0.05,
    )

    assert result["ok"]
    assert result["reason"] == "close_target_settled"
    assert result["sample_count"] == 3
    assert not result["contact_detected"]


def test_close_accepts_stable_tomato_contact_before_closed_target(monkeypatch) -> None:
    monkeypatch.setattr(
        executor,
        "_gripper_motor_state",
        _state_stream(
            [
                {"position": 3.0, "velocity": 7.0, "current": 0.05},
                {"position": -2.35, "velocity": 0.0, "current": 0.20},
                {"position": -2.35, "velocity": 0.0, "current": 0.20},
            ]
        ),
    )

    result = executor._wait_for_gripper_close_settled(
        "http://unused",
        arm="left",
        command_position=4.57,
        before_position=-4.50,
        min_motion=0.25,
        timeout_s=1.0,
        poll_s=0.001,
        request_timeout_s=0.1,
        settle_velocity=0.05,
        settle_samples=2,
        target_tolerance=0.25,
        min_progress_fraction=0.20,
        contact_current_threshold=0.05,
    )

    assert result["ok"]
    assert result["reason"] == "contact_settled"
    assert result["sample_count"] == 3
    assert result["contact_detected"]


def test_contact_progress_scales_down_for_larger_live_fruit() -> None:
    policy = executor._gripper_contact_progress_policy(
        {
            "retarget_geometry_gate": {
                "template_nominal_grasp_insertion_m": 0.0216619056,
                "live_nominal_grasp_insertion_m": 0.0335444201,
            }
        },
        base_fraction=0.20,
    )

    assert policy["method"] == "template_contact_fraction_scaled_by_fruit_radius"
    assert policy["effective_min_progress_fraction"] == pytest.approx(0.129154, abs=1e-6)


def test_contact_progress_scaled_policy_accepts_trial4_contact(monkeypatch) -> None:
    policy = executor._gripper_contact_progress_policy(
        {
            "retarget_geometry_gate": {
                "template_nominal_grasp_insertion_m": 0.0216619056,
                "live_nominal_grasp_insertion_m": 0.0335444201,
            }
        },
        base_fraction=0.20,
    )
    monkeypatch.setattr(
        executor,
        "_gripper_motor_state",
        _state_stream(
            [
                {"position": -2.9759, "velocity": 0.0, "current": 0.6725},
                {"position": -2.9759, "velocity": 0.0, "current": 0.6725},
            ]
        ),
    )

    result = executor._wait_for_gripper_close_settled(
        "http://unused",
        arm="left",
        command_position=4.5710,
        before_position=-4.4884,
        min_motion=0.25,
        timeout_s=1.0,
        poll_s=0.001,
        request_timeout_s=0.1,
        settle_velocity=0.05,
        settle_samples=2,
        target_tolerance=0.25,
        min_progress_fraction=policy["effective_min_progress_fraction"],
        contact_current_threshold=0.05,
    )

    assert result["ok"]
    assert result["reason"] == "contact_settled"
    assert result["signed_motion"] / (4.5710 - -4.4884) == pytest.approx(0.167, abs=1e-3)


def test_contact_progress_falls_back_without_fruit_geometry() -> None:
    policy = executor._gripper_contact_progress_policy({}, base_fraction=0.20)

    assert policy["method"] == "fixed_fraction_fallback"
    assert policy["effective_min_progress_fraction"] == pytest.approx(0.20)


def test_strong_current_witness_accepts_contact_below_size_prediction(monkeypatch) -> None:
    monkeypatch.setattr(
        executor,
        "_gripper_motor_state",
        _state_stream(
            [
                {"position": -3.3226, "velocity": 0.0, "current": 0.6725},
                {"position": -3.3226, "velocity": 0.0, "current": 0.6725},
            ]
        ),
    )

    result = executor._wait_for_gripper_close_settled(
        "http://unused",
        arm="left",
        command_position=4.5710,
        before_position=-4.4884,
        min_motion=0.25,
        timeout_s=1.0,
        poll_s=0.001,
        request_timeout_s=0.1,
        settle_velocity=0.05,
        settle_samples=2,
        target_tolerance=0.25,
        min_progress_fraction=0.141871,
        contact_current_threshold=0.05,
        strong_contact_current_threshold=0.50,
    )

    assert result["ok"]
    assert result["reason"] == "contact_settled"
    assert result["contact_evidence"] == "strong_current_after_minimum_travel"


def test_strong_current_witness_rejects_low_current_early_stop(monkeypatch) -> None:
    monkeypatch.setattr(
        executor,
        "_gripper_motor_state",
        _state_stream(
            [
                {"position": -3.3226, "velocity": 0.0, "current": 0.10},
                {"position": -3.3226, "velocity": 0.0, "current": 0.10},
            ]
        ),
    )

    result = executor._wait_for_gripper_close_settled(
        "http://unused",
        arm="left",
        command_position=4.5710,
        before_position=-4.4884,
        min_motion=0.25,
        timeout_s=0.0,
        poll_s=0.001,
        request_timeout_s=0.1,
        settle_velocity=0.05,
        settle_samples=2,
        target_tolerance=0.25,
        min_progress_fraction=0.141871,
        contact_current_threshold=0.05,
        strong_contact_current_threshold=0.50,
    )

    assert not result["ok"]
    assert not result["contact_detected"]


def test_arm_waypoint_waits_for_position_and_velocity_settle() -> None:
    home, waypoint = _arm_home_and_waypoint()
    robot = _RobotStateStream(
        [
            _arm_state(0.95, 0.20),
            _arm_state(1.00, 0.00),
            _arm_state(1.00, 0.00),
        ]
    )

    result = executor._wait_for_arm_waypoint_settled(
        robot,
        home,
        waypoint,
        position_tolerance_rad=0.015,
        velocity_tolerance_rad_s=0.02,
        required_samples=2,
        timeout_s=1.0,
        poll_s=0.001,
    )

    assert result["ok"]
    assert result["reason"] == "arm_waypoint_settled"
    assert result["sample_count"] == 3


def test_arm_waypoint_does_not_settle_while_still_moving() -> None:
    home, waypoint = _arm_home_and_waypoint()
    robot = _RobotStateStream(
        [
            _arm_state(1.00, 0.10),
            _arm_state(1.00, 0.00),
            _arm_state(1.00, 0.00),
        ]
    )

    result = executor._wait_for_arm_waypoint_settled(
        robot,
        home,
        waypoint,
        position_tolerance_rad=0.015,
        velocity_tolerance_rad_s=0.02,
        required_samples=2,
        timeout_s=1.0,
        poll_s=0.001,
    )

    assert result["ok"]
    assert result["sample_count"] == 3


def test_explicit_schedule_closes_after_open_insertion_endpoint() -> None:
    candidate = {
        "waypoints": [
            {"name": name}
            for name in (
                "approach",
                "orient_soft_03",
                "orient_soft_07",
                "pre_grasp",
                "grasp",
                "harvest",
                "retract",
                "basket_place",
            )
        ],
        "gripper_events": [
            {"after_waypoint": "pre_grasp", "action": "open", "arm": "left"},
            {"after_waypoint": "grasp", "action": "close", "arm": "left"},
            {"after_waypoint": "basket_place", "action": "release", "arm": "left"},
        ],
    }

    schedule = executor._candidate_gripper_schedule(candidate)

    assert schedule[1]["after_waypoint"] == "grasp"


def test_schedule_rejects_close_after_harvest() -> None:
    candidate = {
        "waypoints": [
            {"name": name}
            for name in (
                "approach",
                "orient_soft_03",
                "orient_soft_07",
                "pre_grasp",
                "grasp",
                "harvest",
                "retract",
                "basket_place",
            )
        ],
        "gripper_events": [
            {"after_waypoint": "pre_grasp", "action": "open", "arm": "left"},
            {"after_waypoint": "harvest", "action": "close", "arm": "left"},
            {"after_waypoint": "basket_place", "action": "release", "arm": "left"},
        ],
    }

    with pytest.raises(RuntimeError, match="grasp insertion endpoint"):
        executor._candidate_gripper_schedule(candidate)
