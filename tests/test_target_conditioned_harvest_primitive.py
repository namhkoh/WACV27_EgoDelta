from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot.kinematics import harvest_primitive as primitive
from robot.kinematics.ik_solver import joint_limits_with_margin


def test_joint_limit_margin_is_applied_to_both_bounds() -> None:
    lower, upper = joint_limits_with_margin(
        np.asarray([-1.0, -2.0]),
        np.asarray([1.0, 2.0]),
        0.003,
    )
    assert np.allclose(lower, [-0.997, -1.997])
    assert np.allclose(upper, [0.997, 1.997])


def _candidate(names: tuple[str, ...] = primitive.HARVEST_PHASES) -> dict:
    return {
        "name": "template",
        "waypoints": [
            {
                "name": name,
                "endpoint_gate": {"effective_ok": True},
                "ik_result": {"q_arm_rad": [0.0] * 7},
            }
            for name in names
        ],
    }


def test_waypoint_shift_applies_insertion_only_to_contact_phases() -> None:
    delta = np.asarray([0.1, -0.2, 0.3])
    axis = np.asarray([1.0, 0.0, 0.0])

    assert np.allclose(
        primitive.waypoint_shift(
            "approach",
            target_delta_m=delta,
            insertion_axis=axis,
            extra_insertion_m=0.02,
            height_offset_m=0.01,
        ),
        delta,
    )
    assert np.allclose(
        primitive.waypoint_shift(
            "pre_grasp",
            target_delta_m=delta,
            insertion_axis=axis,
            extra_insertion_m=0.02,
            height_offset_m=0.01,
        ),
        delta + [0.0, 0.0, 0.01],
    )
    assert np.allclose(
        primitive.waypoint_shift(
            "grasp",
            target_delta_m=delta,
            insertion_axis=axis,
            extra_insertion_m=0.02,
            height_offset_m=0.01,
        ),
        delta + [0.02, 0.0, 0.01],
    )
    assert np.allclose(
        primitive.waypoint_shift(
            "grasp",
            target_delta_m=delta,
            insertion_axis=axis,
            extra_insertion_m=0.0,
            height_offset_m=0.0,
            size_scaled_insertion_adjustment_m=0.017,
        ),
        delta + [0.017, 0.0, 0.0],
    )


def test_live_fruit_radius_scales_demonstrated_insertion_ratio() -> None:
    candidate = {
        "manipulation_quality": {
            "nominal_grasp_insertion_m": 0.021,
        }
    }

    profile = primitive.size_normalized_insertion_profile(
        candidate,
        template_insertion_m=0.0315,
        live_nominal_insertion_m=0.033,
    )

    assert profile["mode"] == "demonstration_ratio_scaled_by_live_rgbd_radius"
    assert profile["demonstrated_ratio"] == pytest.approx(1.5)
    assert profile["scaled_insertion_m"] == pytest.approx(0.0495)
    assert profile["adjustment_m"] == pytest.approx(0.018)


def test_template_surface_is_not_confused_with_inserted_grasp_target() -> None:
    candidate = _candidate()
    pre_grasp = next(item for item in candidate["waypoints"] if item["name"] == "pre_grasp")
    grasp = next(item for item in candidate["waypoints"] if item["name"] == "grasp")
    pre_grasp["final_tcp_pose"] = np.eye(4).tolist()
    grasp_pose = np.eye(4)
    grasp_pose[0, 3] = 0.06
    grasp["final_tcp_pose"] = grasp_pose.tolist()
    grasp["surface_target_robot_m"] = [0.50, 0.0, 0.90]
    grasp["target_contact_position_robot_m"] = [0.5325, 0.0, 0.90]

    surface, axis = primitive.template_surface_and_insertion_axis(candidate)

    assert np.allclose(surface, [0.50, 0.0, 0.90])
    assert np.allclose(axis, [1.0, 0.0, 0.0])


def test_geometry_shift_keeps_surface_fixed_when_grasp_gets_extra_insertion() -> None:
    waypoint = {
        "surface_target_robot_m": [0.50, 0.0, 0.90],
        "target_contact_position_robot_m": [0.5325, 0.0, 0.90],
        "grasp_anchor_robot_m": [0.5325, 0.0, 0.90],
    }

    primitive.shift_waypoint_geometry(
        waypoint,
        pose_shift_m=np.asarray([0.12, 0.01, 0.02]),
        surface_shift_m=np.asarray([0.10, 0.01, 0.00]),
    )

    assert np.allclose(waypoint["surface_target_robot_m"], [0.60, 0.01, 0.90])
    assert np.allclose(waypoint["target_contact_position_robot_m"], [0.6525, 0.01, 0.92])
    assert np.allclose(waypoint["grasp_anchor_robot_m"], [0.6525, 0.01, 0.92])


def test_geometry_shift_preserves_demonstrated_pregrasp_offset() -> None:
    waypoint = {
        "surface_target_robot_m": [0.50, 0.0, 0.90],
        "target_contact_position_robot_m": [0.47, 0.0, 0.90],
    }

    primitive.shift_waypoint_geometry(
        waypoint,
        pose_shift_m=np.asarray([0.10, 0.02, -0.01]),
        surface_shift_m=np.asarray([0.10, 0.02, -0.01]),
    )

    assert np.allclose(waypoint["surface_target_robot_m"], [0.60, 0.02, 0.89])
    assert np.allclose(waypoint["target_contact_position_robot_m"], [0.57, 0.02, 0.89])
    assert np.allclose(
        np.asarray(waypoint["target_contact_position_robot_m"])
        - np.asarray(waypoint["surface_target_robot_m"]),
        [-0.03, 0.0, 0.0],
    )


def test_template_loader_rejects_missing_harvest_phase(tmp_path: Path) -> None:
    payload = {"best_candidate": _candidate(primitive.HARVEST_PHASES[:-1])}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="phase sequence mismatch"):
        primitive.load_template_candidate(path)


def test_validation_requires_complete_open_close_release_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    class Kinematics:
        @staticmethod
        def active_joint_limits(_q: object = None) -> tuple[np.ndarray, np.ndarray]:
            return -np.ones(7), np.ones(7)

        @staticmethod
        def q_full_from_arm(_locked: object, q: np.ndarray) -> np.ndarray:
            return np.asarray(q, dtype=float)

        @staticmethod
        def tcp_pose(_q: np.ndarray) -> np.ndarray:
            pose = np.eye(4)
            pose[:3, 3] = [0.5, 0.0, 0.9]
            return pose

    class Locked:
        full_q_rad = np.zeros(7)

    class Clearance:
        minimum_distance_m = 0.05
        ok = True

    monkeypatch.setattr(primitive, "nearest_self_clearances", lambda *args, **kwargs: Clearance())
    report = primitive.validate_harvest_candidate(
        _candidate(),
        kinematics=Kinematics(),
        locked_state=Locked(),
        start_q_arm_rad=np.zeros(7),
        joint_limit_margin_rad=0.003,
        minimum_clearance_m=0.012,
        clearance_pair_count=10,
        samples_per_segment=2,
        manipulation_bounds={"x": (0.2, 1.1), "y": (-0.75, 0.75), "z": (0.4, 1.45)},
        transit_bounds={"x": (-0.25, 1.1), "y": (-0.8, 0.8), "z": (0.35, 1.45)},
    )

    assert report["ok"] is True
    assert report["left_arm_only"] is True
    assert report["gripper_schedule"] == [
        "pre_grasp_open",
        "grasp_close",
        "basket_place_release",
    ]
    assert report["gripper_events"][1] == {
        "after_waypoint": "grasp",
        "action": "close",
        "arm": "left",
    }


def test_validation_rejects_joint_branch_discontinuity(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _candidate()
    candidate["joint_continuity_gate"] = {
        "method": "demonstration_relative_transition_envelope",
        "ok": False,
        "transitions": [
            {
                "start": "orient_soft_07",
                "end": "pre_grasp",
                "solved_l2_delta_rad": 2.124,
                "l2_limit_rad": 0.5,
                "ok": False,
            }
        ],
    }

    class Kinematics:
        @staticmethod
        def active_joint_limits(_q: object = None) -> tuple[np.ndarray, np.ndarray]:
            return -np.ones(7), np.ones(7)

        @staticmethod
        def q_full_from_arm(_locked: object, q: np.ndarray) -> np.ndarray:
            return np.asarray(q, dtype=float)

        @staticmethod
        def tcp_pose(_q: np.ndarray) -> np.ndarray:
            pose = np.eye(4)
            pose[:3, 3] = [0.5, 0.0, 0.9]
            return pose

    class Locked:
        full_q_rad = np.zeros(7)

    class Clearance:
        minimum_distance_m = 0.05
        ok = True

    monkeypatch.setattr(primitive, "nearest_self_clearances", lambda *args, **kwargs: Clearance())
    report = primitive.validate_harvest_candidate(
        candidate,
        kinematics=Kinematics(),
        locked_state=Locked(),
        start_q_arm_rad=np.zeros(7),
        joint_limit_margin_rad=0.003,
        minimum_clearance_m=0.012,
        clearance_pair_count=10,
        samples_per_segment=2,
        manipulation_bounds={"x": (0.2, 1.1), "y": (-0.75, 0.75), "z": (0.4, 1.45)},
        transit_bounds={"x": (-0.25, 1.1), "y": (-0.8, 0.8), "z": (0.35, 1.45)},
    )

    assert report["joint_continuity_ok"] is False
    assert report["ok"] is False
