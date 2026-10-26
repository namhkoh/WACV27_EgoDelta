from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby


if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR
    ]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from robot.server.gripper_status import command_targets
from robot.kinematics.utils.realsense_click_move_arm import command_arm_only


DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_HOME = Path(_REPO_ROOT) / "robot" / "utils" / "homestate.json"
DEFAULT_GRIPPER_URL = "http://192.0.2.21:8765"
DEFAULT_LEFT_OPEN_POSITION = -8.0
DEFAULT_LEFT_OPEN_REACHED_POSITION = 1.784
DEFAULT_LEFT_CLOSE_POSITION = 10.945
DEFAULT_HOME_TOLERANCE_DEG = 5.0
DEFAULT_GRIPPER_SETTLE_VELOCITY = 0.05
DEFAULT_GRIPPER_SETTLE_SAMPLES = 2
DEFAULT_GRIPPER_CLOSE_TARGET_TOLERANCE = 0.25
# This is the template-fruit witness. Execution scales it by the ratio between
# the template and live fruit radii because larger fruit contacts the jaws sooner.
DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_FRACTION = 0.20
DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_FLOOR = 0.10
DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_CEILING = 0.35
DEFAULT_GRIPPER_CONTACT_CURRENT = 0.05
DEFAULT_GRIPPER_STRONG_CONTACT_CURRENT = 0.50
DEFAULT_ARM_SETTLE_POSITION_TOLERANCE_RAD = 0.015
DEFAULT_ARM_SETTLE_VELOCITY_RAD_S = 0.02
DEFAULT_ARM_SETTLE_SAMPLES = 3
DEFAULT_ARM_SETTLE_TIMEOUT_S = 3.0
DEFAULT_ARM_SETTLE_POLL_S = 0.05
LEFT_ARM_DOF = 7


def load_plan(path: Path, candidate_name: str | None, *, allow_failed_candidate: bool = False) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not allow_failed_candidate and not payload.get("hard_gate", {}).get("ok", False):
        raise RuntimeError("refusing to execute plan artifact with hard_gate.ok != true")
    candidate = payload.get("best_candidate")
    if candidate_name is not None:
        matches = [item for item in payload.get("candidates", []) if item.get("name") == candidate_name]
        if not matches:
            raise RuntimeError(f"candidate not found in artifact: {candidate_name}")
        candidate = matches[0]
    if not candidate:
        raise RuntimeError("plan artifact has no selected candidate")
    if not allow_failed_candidate and not candidate.get("hard_gate", {}).get("ok", False):
        raise RuntimeError("selected candidate does not pass hard gates")
    return {
        "artifact": str(path),
        "payload": payload,
        "candidate": candidate,
    }


def _gripper_contact_progress_policy(
    candidate: dict[str, Any],
    *,
    base_fraction: float,
) -> dict[str, Any]:
    base = max(0.0, float(base_fraction))
    geometry = candidate.get("retarget_geometry_gate")
    if not isinstance(geometry, dict):
        geometry = {}
    template_radius = geometry.get("template_nominal_grasp_insertion_m")
    live_radius = geometry.get("live_nominal_grasp_insertion_m")
    try:
        template_radius = float(template_radius)
        live_radius = float(live_radius)
    except (TypeError, ValueError):
        template_radius = None
        live_radius = None

    if (
        template_radius is None
        or live_radius is None
        or not np.isfinite(template_radius)
        or not np.isfinite(live_radius)
        or template_radius <= 0.0
        or live_radius <= 0.0
    ):
        return {
            "method": "fixed_fraction_fallback",
            "reason": "missing_or_invalid_retarget_fruit_geometry",
            "base_fraction": base,
            "effective_min_progress_fraction": base,
            "template_fruit_radius_m": template_radius,
            "live_fruit_radius_m": live_radius,
        }

    size_ratio = template_radius / live_radius
    unclamped = base * size_ratio
    floor = min(base, DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_FLOOR)
    ceiling = max(base, DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_CEILING)
    effective = float(np.clip(unclamped, floor, ceiling))
    return {
        "method": "template_contact_fraction_scaled_by_fruit_radius",
        "base_fraction": base,
        "template_fruit_radius_m": template_radius,
        "live_fruit_radius_m": live_radius,
        "template_to_live_radius_ratio": size_ratio,
        "unclamped_min_progress_fraction": unclamped,
        "minimum_fraction_floor": floor,
        "maximum_fraction_ceiling": ceiling,
        "effective_min_progress_fraction": effective,
    }


def execute_plan(
    *,
    plan: dict[str, Any],
    home_path: Path,
    address: str,
    model: str,
    power: str,
    servo: str,
    minimum_time_s: float,
    velocity_limit_rad_s: float | None,
    acceleration_limit_rad_s2: float | None,
    timeout_s: float,
    home_tolerance_rad: float,
    skip_home_check: bool,
    skip_resume_check: bool,
    check_home_only: bool,
    start_after: str | None,
    stop_after: str | None,
    release_at_basket: bool,
    execute: bool,
    allow_prefix_diagnostic: bool,
    gripper_url: str,
    use_gripper: bool,
    gripper_target_source: str,
    left_open_position: float,
    left_open_reached_position: float,
    left_open_tolerance: float,
    left_close_position: float,
    gripper_open_current_limit: float,
    gripper_close_current_limit: float,
    gripper_hold_s: float,
    gripper_post_command_wait_s: float,
    gripper_post_command_poll_s: float,
    gripper_close_min_motion: float,
    gripper_settle_velocity: float,
    gripper_settle_samples: int,
    gripper_close_target_tolerance: float,
    gripper_close_min_progress_fraction: float,
    gripper_contact_current: float,
    gripper_strong_contact_current: float,
    arm_settle_position_tolerance_rad: float,
    arm_settle_velocity_rad_s: float,
    arm_settle_samples: int,
    arm_settle_timeout_s: float,
    arm_settle_poll_s: float,
) -> dict[str, Any]:
    home = json.loads(Path(home_path).read_text(encoding="utf-8"))
    candidate = plan["candidate"]
    gripper_schedule = _candidate_gripper_schedule(candidate)
    gripper_waypoints = {
        str(event["action"]): str(event["after_waypoint"])
        for event in gripper_schedule
    }
    selected_waypoints = _selected_waypoints(
        candidate,
        allow_prefix_diagnostic=allow_prefix_diagnostic,
        start_after=start_after,
        stop_after=stop_after,
    )
    resume_waypoint = _resume_waypoint(selected_waypoints, start_after)
    waypoints = tuple(_execution_waypoints(selected_waypoints, start_after, stop_after))
    gripper_contact_progress = _gripper_contact_progress_policy(
        candidate,
        base_fraction=gripper_close_min_progress_fraction,
    )
    summary = {
        "artifact": plan["artifact"],
        "candidate_name": candidate["name"],
        "waypoint_names": [waypoint["name"] for waypoint in waypoints],
        "execute": bool(execute),
        "use_gripper": bool(use_gripper),
        "start_after": start_after,
        "stop_after": stop_after,
        "release_at_basket": bool(release_at_basket),
        "allow_prefix_diagnostic": bool(allow_prefix_diagnostic),
        "minimum_time_s": float(minimum_time_s),
        "velocity_limit_rad_s": velocity_limit_rad_s,
        "acceleration_limit_rad_s2": acceleration_limit_rad_s2,
        "home_tolerance_rad": float(home_tolerance_rad),
        "skip_home_check": bool(skip_home_check),
        "skip_resume_check": bool(skip_resume_check),
        "check_home_only": bool(check_home_only),
        "gripper_events": [],
        "gripper_schedule": gripper_schedule,
        "gripper_contact_progress": gripper_contact_progress,
        "gripper_contact_current_thresholds": {
            "geometry_consistent_a": float(gripper_contact_current),
            "strong_witness_a": float(gripper_strong_contact_current),
        },
        "phase_events": [],
        "arm_settle": {
            "position_tolerance_rad": float(arm_settle_position_tolerance_rad),
            "velocity_tolerance_rad_s": float(arm_settle_velocity_rad_s),
            "required_samples": int(arm_settle_samples),
            "timeout_s": float(arm_settle_timeout_s),
            "poll_s": float(arm_settle_poll_s),
        },
    }
    if not execute and not check_home_only:
        return summary

    if use_gripper:
        summary["gripper_preflight"] = _require_gripper_bridge(
            gripper_url,
            timeout_s=timeout_s,
        )
        if gripper_target_source == "live_status":
            targets = command_targets(summary["gripper_preflight"]["status"], arm="left")
            left_open_position = targets.open_position
            left_open_reached_position = targets.open_position
            left_close_position = targets.close_position
            summary["gripper_command_targets"] = {
                "source": "live_status",
                "calibration_schema": targets.calibration_schema,
                "open_position": targets.open_position,
                "close_position": targets.close_position,
                "encoder_span": targets.encoder_span,
                "raw_open_position": targets.raw_open_position,
                "raw_close_position": targets.raw_close_position,
                "encoder_turn_offset": targets.encoder_turn_offset,
                "calibration_endpoint_distance": targets.calibration_endpoint_distance,
            }
        else:
            summary["gripper_command_targets"] = {
                "source": "absolute_cli",
                "open_position": float(left_open_position),
                "open_reached_position": float(left_open_reached_position),
                "close_position": float(left_close_position),
            }

    robot = _connect_for_motion(address, model, power, servo)
    if resume_waypoint is not None and not skip_resume_check:
        summary["resume_check"] = _validate_waypoint_state(
            robot,
            home,
            resume_waypoint,
            tolerance_rad=home_tolerance_rad,
        )
    elif resume_waypoint is not None:
        summary["resume_check"] = {
            "ok": True,
            "waypoint": resume_waypoint.get("name"),
            "skipped": True,
            "reason": "skip_resume_check",
        }
    elif not skip_home_check:
        summary["home_check"] = _validate_home_state(
            robot,
            home,
            tolerance_rad=home_tolerance_rad,
        )
    if check_home_only:
        return summary
    opened_for_pre_grasp = False
    execution_started = time.monotonic()
    for waypoint in waypoints:
        phase_event = {
            "name": waypoint["name"],
            "arm_started_elapsed_s": float(time.monotonic() - execution_started),
        }
        if use_gripper and waypoint["name"] == gripper_waypoints["close"]:
            phase_event["gripper_open_before_insertion"] = _require_gripper_at_position(
                gripper_url,
                arm="left",
                expected_position=left_open_reached_position,
                tolerance=left_open_tolerance,
                timeout_s=timeout_s,
                reason="insertion_requires_open_gripper",
            )
        feedback = robot.send_command(
            _command_for_waypoint(
                home,
                waypoint,
                minimum_time_s,
                velocity_limit_rad_s=velocity_limit_rad_s,
                acceleration_limit_rad_s2=acceleration_limit_rad_s2,
            ),
            int(max(float(timeout_s), float(minimum_time_s) + 5.0)),
        ).get()
        if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"waypoint {waypoint['name']} failed: {feedback.finish_code}")
        phase_event["command_feedback_elapsed_s"] = float(time.monotonic() - execution_started)
        settle = _wait_for_arm_waypoint_settled(
            robot,
            home,
            waypoint,
            position_tolerance_rad=arm_settle_position_tolerance_rad,
            velocity_tolerance_rad_s=arm_settle_velocity_rad_s,
            required_samples=arm_settle_samples,
            timeout_s=arm_settle_timeout_s,
            poll_s=arm_settle_poll_s,
        )
        phase_event["arm_settle"] = settle
        if not settle["ok"]:
            raise RuntimeError(f"left arm did not settle at waypoint {waypoint['name']}: {settle}")
        phase_event["arm_completed_elapsed_s"] = float(time.monotonic() - execution_started)
        if (
            use_gripper
            and waypoint["name"] == gripper_waypoints["open"]
            and not opened_for_pre_grasp
        ):
            event = _gripper_open(
                gripper_url,
                arm="left",
                action="open",
                position=left_open_position,
                expected_reached_position=left_open_reached_position,
                tolerance=left_open_tolerance,
                current_limit=gripper_open_current_limit,
                hold_s=gripper_hold_s,
                timeout_s=timeout_s,
                post_command_wait_s=gripper_post_command_wait_s,
                post_command_poll_s=gripper_post_command_poll_s,
                settle_velocity=gripper_settle_velocity,
                settle_samples=gripper_settle_samples,
            )
            event["ordering"] = (
                f"after_{waypoint['name']}_arm_settle_before_next_arm_phase"
            )
            summary["gripper_events"].append(event)
            phase_event["gripper_event_index"] = len(summary["gripper_events"]) - 1
            opened_for_pre_grasp = True
        if use_gripper and waypoint["name"] == gripper_waypoints["close"]:
            phase_event["gripper_open_at_insertion_endpoint"] = _require_gripper_at_position(
                gripper_url,
                arm="left",
                expected_position=left_open_reached_position,
                tolerance=left_open_tolerance,
                timeout_s=timeout_s,
                reason="close_requires_completed_open_insertion",
            )
            event = _gripper_close(
                gripper_url,
                arm="left",
                position=left_close_position,
                current_limit=gripper_close_current_limit,
                hold_s=gripper_hold_s,
                timeout_s=timeout_s,
                min_motion=gripper_close_min_motion,
                post_command_wait_s=gripper_post_command_wait_s,
                post_command_poll_s=gripper_post_command_poll_s,
                settle_velocity=gripper_settle_velocity,
                settle_samples=gripper_settle_samples,
                target_tolerance=gripper_close_target_tolerance,
                min_progress_fraction=gripper_contact_progress[
                    "effective_min_progress_fraction"
                ],
                contact_current_threshold=gripper_contact_current,
                strong_contact_current_threshold=gripper_strong_contact_current,
            )
            event["ordering"] = (
                f"after_{waypoint['name']}_arm_settle_before_next_arm_phase"
            )
            summary["gripper_events"].append(event)
            phase_event["gripper_event_index"] = len(summary["gripper_events"]) - 1
        if (
            use_gripper
            and release_at_basket
            and waypoint["name"] == gripper_waypoints["release"]
        ):
            event = _gripper_open(
                gripper_url,
                arm="left",
                action="release",
                position=left_open_position,
                expected_reached_position=left_open_reached_position,
                tolerance=left_open_tolerance,
                current_limit=gripper_open_current_limit,
                hold_s=gripper_hold_s,
                timeout_s=timeout_s,
                post_command_wait_s=gripper_post_command_wait_s,
                post_command_poll_s=gripper_post_command_poll_s,
                settle_velocity=gripper_settle_velocity,
                settle_samples=gripper_settle_samples,
            )
            event["ordering"] = f"after_{waypoint['name']}_arm_settle"
            summary["gripper_events"].append(event)
            phase_event["gripper_event_index"] = len(summary["gripper_events"]) - 1
        phase_event["phase_completed_elapsed_s"] = float(time.monotonic() - execution_started)
        summary["phase_events"].append(phase_event)
        time.sleep(0.1)

    return summary


def _selected_waypoints(
    candidate: dict[str, Any],
    *,
    allow_prefix_diagnostic: bool,
    start_after: str | None,
    stop_after: str | None,
) -> list[dict[str, Any]]:
    waypoints = candidate.get("waypoints", [])
    if not waypoints:
        raise RuntimeError("candidate contains no waypoints")
    required = {"approach", "pre_grasp", "grasp"}
    names = {waypoint.get("name") for waypoint in waypoints}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"candidate is missing required waypoints: {missing}")
    if allow_prefix_diagnostic:
        if start_after is not None:
            raise RuntimeError("prefix diagnostics do not support --start-after")
        if stop_after is None:
            raise RuntimeError("prefix diagnostics require --stop-after")
        _validate_waypoint_endpoint_gates(_truncate_waypoints(waypoints, stop_after))
        return waypoints
    _validate_waypoint_endpoint_gates(waypoints)
    return waypoints


def _candidate_gripper_schedule(candidate: dict[str, Any]) -> list[dict[str, str]]:
    raw = candidate.get("gripper_events")
    if raw is None:
        raw = [
            {"after_waypoint": "pre_grasp", "action": "open", "arm": "left"},
            {"after_waypoint": "grasp", "action": "close", "arm": "left"},
            {"after_waypoint": "basket_place", "action": "release", "arm": "left"},
        ]
    if not isinstance(raw, list):
        raise RuntimeError("candidate gripper_events must be a list")
    schedule: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("candidate gripper event must be an object")
        event = {
            "after_waypoint": str(item.get("after_waypoint", "")),
            "action": str(item.get("action", "")),
            "arm": str(item.get("arm", "")),
        }
        if event["arm"] != "left":
            raise RuntimeError(f"harvest gripper event must target left arm: {event}")
        schedule.append(event)
    actions = tuple(event["action"] for event in schedule)
    if actions != ("open", "close", "release"):
        raise RuntimeError(
            "harvest gripper events must be ordered open, close, release; "
            f"actual={actions}"
        )
    actual_contract = tuple(
        (event["after_waypoint"], event["action"], event["arm"])
        for event in schedule
    )
    expected_contract = (
        ("pre_grasp", "open", "left"),
        ("grasp", "close", "left"),
        ("basket_place", "release", "left"),
    )
    if actual_contract != expected_contract:
        raise RuntimeError(
            "harvest gripper events must open before insertion, close only after "
            "the grasp insertion endpoint, and release at basket_place; "
            f"actual={actual_contract}"
        )
    waypoint_names = {str(item.get("name")) for item in candidate.get("waypoints", [])}
    missing = [
        event["after_waypoint"]
        for event in schedule
        if event["after_waypoint"] not in waypoint_names
    ]
    if missing:
        raise RuntimeError(f"gripper events reference missing waypoints: {missing}")
    return schedule


def _validate_waypoint_endpoint_gates(waypoints: list[dict[str, Any]]) -> None:
    for waypoint in waypoints:
        gate = waypoint.get("endpoint_gate", {})
        if not gate.get("effective_ok", gate.get("ok", False)):
            raise RuntimeError(f"refusing waypoint with failed endpoint gate: {waypoint.get('name')}")


def _truncate_waypoints(waypoints: list[dict[str, Any]], stop_after: str | None) -> list[dict[str, Any]]:
    if stop_after is None:
        return waypoints
    for index, waypoint in enumerate(waypoints):
        if waypoint.get("name") == stop_after:
            return waypoints[: index + 1]
    raise RuntimeError(f"stop-after waypoint not found in candidate: {stop_after}")


def _resume_waypoint(waypoints: list[dict[str, Any]], start_after: str | None) -> dict[str, Any] | None:
    if start_after is None:
        return None
    for waypoint in waypoints:
        if waypoint.get("name") == start_after:
            return waypoint
    raise RuntimeError(f"start-after waypoint not found in candidate: {start_after}")


def _execution_waypoints(
    waypoints: list[dict[str, Any]],
    start_after: str | None,
    stop_after: str | None,
) -> list[dict[str, Any]]:
    selected = waypoints
    if start_after is not None:
        for index, waypoint in enumerate(waypoints):
            if waypoint.get("name") == start_after:
                selected = waypoints[index + 1 :]
                break
        else:
            raise RuntimeError(f"start-after waypoint not found in candidate: {start_after}")
    selected = _truncate_waypoints(selected, stop_after)
    if not selected:
        raise RuntimeError("no waypoints selected for execution")
    return selected


def _validate_home_state(robot: Any, home: dict[str, Any], *, tolerance_rad: float) -> dict[str, Any]:
    live = np.asarray(robot.get_state().position, dtype=float)
    expected = np.asarray(home["all_joints"]["positions_rad"], dtype=float)
    joint_names = list(home["joint_names"])
    if live.shape != expected.shape:
        raise RuntimeError(f"live joint vector shape {live.shape} does not match home shape {expected.shape}")
    indices = _commanded_joint_indices(home)
    delta = live[indices] - expected[indices]
    abs_delta = np.abs(delta)
    local_max_index = int(np.argmax(abs_delta))
    max_index = int(indices[local_max_index])
    max_abs_delta = float(abs_delta[local_max_index])
    if max_abs_delta > float(tolerance_rad):
        raise RuntimeError(
            "robot is not at home: "
            f"max_delta={max_abs_delta:.6f} rad "
            f"({np.rad2deg(max_abs_delta):.2f} deg) "
            f"joint={joint_names[max_index]} "
            f"tolerance={float(tolerance_rad):.6f} rad"
        )
    return {
        "ok": True,
        "max_abs_delta_rad": max_abs_delta,
        "max_abs_delta_deg": float(np.rad2deg(max_abs_delta)),
        "max_delta_joint": joint_names[max_index],
        "tolerance_rad": float(tolerance_rad),
        "checked_joint_count": int(len(indices)),
    }


def _validate_waypoint_state(
    robot: Any,
    home: dict[str, Any],
    waypoint: dict[str, Any],
    *,
    tolerance_rad: float,
) -> dict[str, Any]:
    live = np.asarray(robot.get_state().position, dtype=float)
    expected = _expected_positions_for_waypoint(home, waypoint)
    joint_names = list(home["joint_names"])
    if live.shape != expected.shape:
        raise RuntimeError(f"live joint vector shape {live.shape} does not match expected shape {expected.shape}")
    indices = _commanded_joint_indices(home)
    delta = live[indices] - expected[indices]
    abs_delta = np.abs(delta)
    local_max_index = int(np.argmax(abs_delta))
    max_index = int(indices[local_max_index])
    max_abs_delta = float(abs_delta[local_max_index])
    if max_abs_delta > float(tolerance_rad):
        raise RuntimeError(
            f"robot is not at resume waypoint {waypoint.get('name')}: "
            f"max_delta={max_abs_delta:.6f} rad "
            f"({np.rad2deg(max_abs_delta):.2f} deg) "
            f"joint={joint_names[max_index]} "
            f"tolerance={float(tolerance_rad):.6f} rad"
        )
    return {
        "ok": True,
        "waypoint": waypoint.get("name"),
        "max_abs_delta_rad": max_abs_delta,
        "max_abs_delta_deg": float(np.rad2deg(max_abs_delta)),
        "max_delta_joint": joint_names[max_index],
        "tolerance_rad": float(tolerance_rad),
        "checked_joint_count": int(len(indices)),
    }


def _commanded_joint_indices(home: dict[str, Any]) -> np.ndarray:
    joint_names = list(home["joint_names"])
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    selected: list[int] = []
    selected.extend(_indices_for_names(name_to_index, home["torso"]["joint_names"]))
    selected.extend(_indices_for_names(name_to_index, home["arms"]["right"]["joint_names"]))
    selected.extend(_indices_for_names(name_to_index, home["arms"]["left"]["joint_names"]))
    selected.extend(_indices_for_names(name_to_index, home["head"]["joint_names"]))
    return np.asarray(selected, dtype=int)


def _indices_for_names(name_to_index: dict[str, int], names: list[str]) -> list[int]:
    indices: list[int] = []
    for name in names:
        if name not in name_to_index:
            raise RuntimeError(f"home joint name not found in all_joints: {name}")
        indices.append(name_to_index[name])
    return indices


def _positions(home: dict[str, Any], group: str, side: str | None = None) -> np.ndarray:
    if side is None:
        values = home[group]["positions_rad"]
    else:
        values = home[group][side]["positions_rad"]
    return np.asarray(values, dtype=float)


def _expected_positions_for_waypoint(home: dict[str, Any], waypoint: dict[str, Any]) -> np.ndarray:
    expected = np.asarray(home["all_joints"]["positions_rad"], dtype=float).copy()
    joint_names = list(home["joint_names"])
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    left_indices = _indices_for_names(name_to_index, home["arms"]["left"]["joint_names"])
    expected[left_indices] = np.asarray(waypoint["ik_result"]["q_arm_rad"], dtype=float).reshape(LEFT_ARM_DOF)
    return expected


def _wait_for_arm_waypoint_settled(
    robot: Any,
    home: dict[str, Any],
    waypoint: dict[str, Any],
    *,
    position_tolerance_rad: float,
    velocity_tolerance_rad_s: float,
    required_samples: int,
    timeout_s: float,
    poll_s: float,
) -> dict[str, Any]:
    joint_names = list(home["joint_names"])
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    indices = np.asarray(
        _indices_for_names(name_to_index, home["arms"]["left"]["joint_names"]),
        dtype=int,
    )
    expected = _expected_positions_for_waypoint(home, waypoint)[indices]
    needed = max(1, int(required_samples))
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    settled_count = 0
    sample_count = 0
    last: dict[str, Any] = {}

    while True:
        state = robot.get_state()
        positions = np.asarray(state.position, dtype=float)
        velocities = np.asarray(state.velocity, dtype=float)
        if positions.shape != velocities.shape or positions.size <= int(np.max(indices)):
            raise RuntimeError(
                f"live joint state cannot represent left arm: position={positions.shape}, "
                f"velocity={velocities.shape}, max_index={int(np.max(indices))}"
            )
        position_errors = np.abs(positions[indices] - expected)
        speeds = np.abs(velocities[indices])
        position_local_index = int(np.argmax(position_errors))
        velocity_local_index = int(np.argmax(speeds))
        max_position_error = float(position_errors[position_local_index])
        max_velocity = float(speeds[velocity_local_index])
        position_ok = max_position_error <= float(position_tolerance_rad)
        velocity_ok = max_velocity <= float(velocity_tolerance_rad_s)
        settled_count = settled_count + 1 if position_ok and velocity_ok else 0
        sample_count += 1
        last = {
            "max_position_error_rad": max_position_error,
            "max_position_error_deg": float(np.rad2deg(max_position_error)),
            "max_position_error_joint": joint_names[int(indices[position_local_index])],
            "max_velocity_rad_s": max_velocity,
            "max_velocity_joint": joint_names[int(indices[velocity_local_index])],
            "position_ok": position_ok,
            "velocity_ok": velocity_ok,
        }
        if settled_count >= needed:
            return {
                "ok": True,
                "reason": "arm_waypoint_settled",
                "waypoint": waypoint.get("name"),
                "position_tolerance_rad": float(position_tolerance_rad),
                "velocity_tolerance_rad_s": float(velocity_tolerance_rad_s),
                "required_samples": needed,
                "settled_samples": settled_count,
                "sample_count": sample_count,
                **last,
            }
        if time.monotonic() >= deadline:
            return {
                "ok": False,
                "reason": "arm_waypoint_settle_timeout",
                "waypoint": waypoint.get("name"),
                "position_tolerance_rad": float(position_tolerance_rad),
                "velocity_tolerance_rad_s": float(velocity_tolerance_rad_s),
                "required_samples": needed,
                "settled_samples": settled_count,
                "sample_count": sample_count,
                **last,
            }
        time.sleep(max(0.0, float(poll_s)))


def _command_for_waypoint(
    home: dict[str, Any],
    waypoint: dict[str, Any],
    minimum_time_s: float,
    *,
    velocity_limit_rad_s: float | None = None,
    acceleration_limit_rad_s2: float | None = None,
) -> Any:
    del home
    q_left = np.asarray(waypoint["ik_result"]["q_arm_rad"], dtype=float).reshape(LEFT_ARM_DOF)
    return command_arm_only(
        "left",
        q_left,
        minimum_time_s=float(minimum_time_s),
        velocity_limit_rad_s=velocity_limit_rad_s,
        acceleration_limit_rad_s2=acceleration_limit_rad_s2,
    )

def _connect_for_motion(address: str, model: str, power: str, servo: str) -> Any:
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
    if not robot.is_power_on(power) and not robot.power_on(power):
        raise RuntimeError(f"failed to power on {power}")
    if not robot.is_servo_on(servo) and not robot.servo_on(servo):
        raise RuntimeError(f"failed to servo on {servo}")

    state = robot.get_control_manager_state().state
    if state in (rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault):
        if not robot.reset_fault_control_manager():
            raise RuntimeError("failed to reset control manager fault")
    if not robot.enable_control_manager():
        raise RuntimeError("failed to enable control manager")
    return robot


def _gripper_open(
    base_url: str,
    *,
    arm: str,
    action: str,
    position: float,
    expected_reached_position: float,
    tolerance: float,
    current_limit: float,
    hold_s: float,
    timeout_s: float,
    post_command_wait_s: float,
    post_command_poll_s: float,
    settle_velocity: float,
    settle_samples: int,
) -> dict[str, Any]:
    before_state = _gripper_motor_state(base_url, arm=arm, timeout_s=timeout_s)
    before = None if before_state is None else before_state.get("position")
    payload = _gripper_jog(
        base_url,
        arm=arm,
        position=position,
        current_limit=current_limit,
        hold_s=hold_s,
        timeout_s=timeout_s,
    )
    completion = _wait_for_gripper_open_settled(
        base_url,
        arm=arm,
        expected_reached_position=expected_reached_position,
        tolerance=tolerance,
        timeout_s=post_command_wait_s,
        poll_s=post_command_poll_s,
        request_timeout_s=timeout_s,
        settle_velocity=settle_velocity,
        settle_samples=settle_samples,
    )
    after_state = completion.get("motor_state")
    after = None if after_state is None else after_state.get("position")
    position_error = None if after is None else abs(float(after) - float(expected_reached_position))
    event = {
        "action": action,
        "arm": arm,
        "command_position": float(position),
        "current_limit": float(current_limit),
        "hold_s": float(hold_s),
        "position_before": before,
        "position_after": after,
        "expected_reached_position": float(expected_reached_position),
        "tolerance": float(tolerance),
        "position_error": position_error,
        "completion": completion,
        "bridge_payload": payload,
        "ok": bool(completion.get("ok", False)),
    }
    if not event["ok"]:
        raise RuntimeError(f"left gripper did not reach full-open side: {event}")
    return event


def _gripper_close(
    base_url: str,
    *,
    arm: str,
    position: float,
    current_limit: float,
    hold_s: float,
    timeout_s: float,
    min_motion: float,
    post_command_wait_s: float,
    post_command_poll_s: float,
    settle_velocity: float,
    settle_samples: int,
    target_tolerance: float,
    min_progress_fraction: float,
    contact_current_threshold: float,
    strong_contact_current_threshold: float,
) -> dict[str, Any]:
    before_state = _gripper_motor_state(base_url, arm=arm, timeout_s=timeout_s)
    before = None if before_state is None else before_state.get("position")
    payload = _gripper_jog(
        base_url,
        arm=arm,
        position=position,
        current_limit=current_limit,
        hold_s=hold_s,
        timeout_s=timeout_s,
    )
    completion = _wait_for_gripper_close_settled(
        base_url,
        arm=arm,
        command_position=position,
        before_position=before,
        min_motion=min_motion,
        timeout_s=post_command_wait_s,
        poll_s=post_command_poll_s,
        request_timeout_s=timeout_s,
        settle_velocity=settle_velocity,
        settle_samples=settle_samples,
        target_tolerance=target_tolerance,
        min_progress_fraction=min_progress_fraction,
        contact_current_threshold=contact_current_threshold,
        strong_contact_current_threshold=strong_contact_current_threshold,
    )
    after_state = completion.get("motor_state")
    after = None if after_state is None else after_state.get("position")
    event = {
        "action": "close",
        "arm": arm,
        "command_position": float(position),
        "current_limit": float(current_limit),
        "hold_s": float(hold_s),
        "position_before": before,
        "position_after": after,
        "min_motion": float(min_motion),
        "completion": completion,
        "bridge_payload": payload,
        "ok": bool(completion.get("ok", False)),
    }
    if not event["ok"]:
        raise RuntimeError(f"left gripper did not settle after close command: {event}")
    return event


def _require_gripper_bridge(base_url: str, *, timeout_s: float) -> dict[str, Any]:
    try:
        payload = _gripper_status(base_url, timeout_s=timeout_s)
    except Exception as exc:
        raise RuntimeError(
            "gripper HTTP bridge is not reachable before motion. "
            f"URL={base_url.rstrip('/')}/status. "
            "Start robot/server/rby1_gripper_http_bridge.py or rerun with --no-gripper for arm-only testing."
        ) from exc
    return {
        "ok": True,
        "url": base_url.rstrip("/"),
        "status": payload,
    }


def _gripper_jog(
    base_url: str,
    *,
    arm: str,
    position: float,
    current_limit: float,
    hold_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "arm": arm,
            "position": float(position),
            "current_limit": float(current_limit),
            "hold_s": float(hold_s),
        }
    )
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/jog?{query}", timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok", False):
        raise RuntimeError(f"left gripper command failed: {payload}")
    return payload


def _gripper_status(base_url: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/status", timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok", False):
        raise RuntimeError(f"left gripper status failed: {payload}")
    return payload


def _gripper_motor_state(base_url: str, *, arm: str, timeout_s: float) -> dict[str, float] | None:
    return _gripper_motor_state_from_payload(_gripper_status(base_url, timeout_s=timeout_s), arm=arm)


def _gripper_motor_state_from_payload(payload: dict[str, Any], *, arm: str) -> dict[str, float] | None:
    target_id = 1 if arm == "left" else 0
    for state in payload.get("motor_states", []):
        if int(state.get("id", -1)) == target_id and state.get("position") is not None:
            result = {"position": float(state["position"])}
            for key in ("velocity", "current"):
                if state.get(key) is not None:
                    result[key] = float(state[key])
            return result
    return None


def _require_gripper_at_position(
    base_url: str,
    *,
    arm: str,
    expected_position: float,
    tolerance: float,
    timeout_s: float,
    reason: str,
) -> dict[str, Any]:
    state = _gripper_motor_state(base_url, arm=arm, timeout_s=timeout_s)
    if state is None:
        raise RuntimeError(f"{reason}: missing {arm} gripper motor state")
    position = float(state["position"])
    error = abs(position - float(expected_position))
    report = {
        "ok": error <= float(tolerance),
        "reason": reason,
        "position": position,
        "expected_position": float(expected_position),
        "position_error": error,
        "tolerance": float(tolerance),
    }
    if not report["ok"]:
        raise RuntimeError(f"{reason}: {report}")
    return report


def _wait_for_gripper_open_settled(
    base_url: str,
    *,
    arm: str,
    expected_reached_position: float,
    tolerance: float,
    timeout_s: float,
    poll_s: float,
    request_timeout_s: float,
    settle_velocity: float,
    settle_samples: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    required_samples = max(1, int(settle_samples))
    settled_count = 0
    sample_count = 0
    last_state: dict[str, float] | None = None
    while time.monotonic() <= deadline:
        last_state = _gripper_motor_state(base_url, arm=arm, timeout_s=request_timeout_s)
        sample_count += 1
        position = None if last_state is None else last_state.get("position")
        velocity = None if last_state is None else last_state.get("velocity")
        position_error = (
            None if position is None else abs(float(position) - float(expected_reached_position))
        )
        at_target = position_error is not None and position_error <= float(tolerance)
        settled = velocity is not None and abs(float(velocity)) <= float(settle_velocity)
        settled_count = settled_count + 1 if at_target and settled else 0
        if settled_count >= required_samples:
            return {
                "ok": True,
                "reason": "open_target_settled",
                "motor_state": last_state,
                "position_error": position_error,
                "settled_samples": settled_count,
                "sample_count": sample_count,
            }
        time.sleep(max(0.05, float(poll_s)))
    return {
        "ok": False,
        "reason": "open_settle_timeout",
        "motor_state": last_state,
        "settled_samples": settled_count,
        "sample_count": sample_count,
    }


def _wait_for_gripper_close_settled(
    base_url: str,
    *,
    arm: str,
    command_position: float,
    before_position: float | None,
    min_motion: float,
    timeout_s: float,
    poll_s: float,
    request_timeout_s: float,
    settle_velocity: float,
    settle_samples: int,
    target_tolerance: float,
    min_progress_fraction: float,
    contact_current_threshold: float,
    strong_contact_current_threshold: float = DEFAULT_GRIPPER_STRONG_CONTACT_CURRENT,
) -> dict[str, Any]:
    if before_position is None:
        return {"ok": False, "reason": "missing_before_position", "motor_state": None}
    command_delta = float(command_position) - float(before_position)
    if abs(command_delta) <= 1e-9:
        return {"ok": False, "reason": "degenerate_close_command", "motor_state": None}
    required_motion = max(
        float(min_motion),
        abs(command_delta) * max(0.0, float(min_progress_fraction)),
    )
    strong_witness_required_motion = max(
        float(min_motion),
        abs(command_delta) * DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_FLOOR,
    )
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    required_samples = max(1, int(settle_samples))
    settled_count = 0
    sample_count = 0
    last_kind: str | None = None
    last_state: dict[str, float] | None = None
    last_signed_motion = 0.0
    last_target_error: float | None = None
    last_contact_evidence: str | None = None
    while time.monotonic() <= deadline:
        last_state = _gripper_motor_state(base_url, arm=arm, timeout_s=request_timeout_s)
        sample_count += 1
        position = None if last_state is None else last_state.get("position")
        velocity = None if last_state is None else last_state.get("velocity")
        current = None if last_state is None else last_state.get("current")
        if position is None:
            settled_count = 0
            last_kind = None
            time.sleep(max(0.05, float(poll_s)))
            continue
        last_signed_motion = float(np.sign(command_delta) * (float(position) - float(before_position)))
        last_target_error = abs(float(position) - float(command_position))
        settled = velocity is not None and abs(float(velocity)) <= float(settle_velocity)
        target_reached = last_target_error <= float(target_tolerance)
        geometry_consistent_contact = (
            last_signed_motion >= required_motion
            and not target_reached
            and current is not None
            and abs(float(current)) >= float(contact_current_threshold)
        )
        strong_current_contact = (
            last_signed_motion >= strong_witness_required_motion
            and not target_reached
            and current is not None
            and abs(float(current)) >= float(strong_contact_current_threshold)
        )
        contact_stopped = geometry_consistent_contact or strong_current_contact
        last_contact_evidence = (
            "geometry_consistent_progress"
            if geometry_consistent_contact
            else "strong_current_after_minimum_travel"
            if strong_current_contact
            else None
        )
        kind = "close_target_settled" if target_reached else "contact_settled" if contact_stopped else None
        if settled and kind is not None:
            settled_count = settled_count + 1 if kind == last_kind else 1
            last_kind = kind
        else:
            settled_count = 0
            last_kind = None
        if settled_count >= required_samples:
            return {
                "ok": True,
                "reason": kind,
                "motor_state": last_state,
                "signed_motion": last_signed_motion,
                "required_motion": required_motion,
                "strong_witness_required_motion": strong_witness_required_motion,
                "target_error": last_target_error,
                "settled_samples": settled_count,
                "sample_count": sample_count,
                "contact_detected": kind == "contact_settled",
                "contact_evidence": last_contact_evidence,
            }
        time.sleep(max(0.05, float(poll_s)))
    return {
        "ok": False,
        "reason": "close_settle_timeout",
        "motor_state": last_state,
        "signed_motion": last_signed_motion,
        "required_motion": required_motion,
        "strong_witness_required_motion": strong_witness_required_motion,
        "target_error": last_target_error,
        "settled_samples": settled_count,
        "sample_count": sample_count,
        "contact_detected": False,
        "contact_evidence": last_contact_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a validated staged harvest IK plan artifact.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--power", default=".*")
    parser.add_argument("--servo", default=".*")
    parser.add_argument("--minimum-time-s", type=float, default=10.0)
    parser.add_argument(
        "--velocity-limit-rad-s",
        type=float,
        default=None,
        help="Optional per-joint velocity limit for every left-arm waypoint.",
    )
    parser.add_argument(
        "--acceleration-limit-rad-s2",
        type=float,
        default=None,
        help="Optional per-joint acceleration limit for every left-arm waypoint.",
    )
    parser.add_argument(
        "--arm-settle-position-tolerance-rad",
        type=float,
        default=DEFAULT_ARM_SETTLE_POSITION_TOLERANCE_RAD,
        help="Maximum left-arm joint error before a waypoint may trigger its next event.",
    )
    parser.add_argument(
        "--arm-settle-velocity-rad-s",
        type=float,
        default=DEFAULT_ARM_SETTLE_VELOCITY_RAD_S,
        help="Maximum left-arm joint speed before a waypoint may trigger its next event.",
    )
    parser.add_argument(
        "--arm-settle-samples",
        type=int,
        default=DEFAULT_ARM_SETTLE_SAMPLES,
        help="Consecutive settled arm-state samples required at every waypoint.",
    )
    parser.add_argument(
        "--arm-settle-timeout-s",
        type=float,
        default=DEFAULT_ARM_SETTLE_TIMEOUT_S,
    )
    parser.add_argument(
        "--arm-settle-poll-s",
        type=float,
        default=DEFAULT_ARM_SETTLE_POLL_S,
    )
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--home-tolerance-deg", type=float, default=DEFAULT_HOME_TOLERANCE_DEG)
    parser.add_argument("--skip-home-check", action="store_true")
    parser.add_argument(
        "--skip-resume-check",
        action="store_true",
        help=(
            "When --start-after is used, do not require the live robot to match the skipped waypoint. "
            "Use only for deliberately splicing a fixed tail after a separately executed grasp."
        ),
    )
    parser.add_argument("--check-home-only", action="store_true", help="Validate live state against home without motion.")
    parser.add_argument(
        "--start-after",
        choices=("approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"),
        default=None,
        help="Resume from a validated waypoint by first checking the live joint state, then executing following waypoints.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"),
        default=None,
        help="Execute only through the named waypoint. Useful for staged live validation.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually command the robot. Dry-run by default.")
    parser.add_argument(
        "--allow-prefix-diagnostic",
        action="store_true",
        help="Allow executing a passing prefix of a failed full plan when --stop-after is set.",
    )
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument("--no-release-at-basket", action="store_true", help="Do not open the gripper at basket_place.")
    parser.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument(
        "--gripper-target-source",
        choices=("live_status", "absolute"),
        default="live_status",
        help="Derive open/closed encoder targets from live /status bounds, or use explicit absolute CLI values.",
    )
    parser.add_argument("--left-open-position", type=float, default=DEFAULT_LEFT_OPEN_POSITION)
    parser.add_argument("--left-open-reached-position", type=float, default=DEFAULT_LEFT_OPEN_REACHED_POSITION)
    parser.add_argument("--left-open-tolerance", type=float, default=0.25)
    parser.add_argument("--left-close-position", type=float, default=DEFAULT_LEFT_CLOSE_POSITION)
    parser.add_argument("--gripper-open-current-limit", type=float, default=3.0)
    parser.add_argument("--gripper-close-current-limit", type=float, default=2.0)
    parser.add_argument("--gripper-hold-s", type=float, default=1.0)
    parser.add_argument("--gripper-post-command-wait-s", type=float, default=3.0)
    parser.add_argument("--gripper-post-command-poll-s", type=float, default=0.2)
    parser.add_argument("--gripper-close-min-motion", type=float, default=0.25)
    parser.add_argument(
        "--gripper-settle-velocity",
        type=float,
        default=DEFAULT_GRIPPER_SETTLE_VELOCITY,
        help="Maximum absolute encoder velocity for a gripper command to be considered settled.",
    )
    parser.add_argument(
        "--gripper-settle-samples",
        type=int,
        default=DEFAULT_GRIPPER_SETTLE_SAMPLES,
        help="Consecutive settled status samples required before the next arm phase.",
    )
    parser.add_argument(
        "--gripper-close-target-tolerance",
        type=float,
        default=DEFAULT_GRIPPER_CLOSE_TARGET_TOLERANCE,
    )
    parser.add_argument(
        "--gripper-close-min-progress-fraction",
        type=float,
        default=DEFAULT_GRIPPER_CLOSE_MIN_PROGRESS_FRACTION,
        help="Minimum fraction of full close travel before an early stop can count as fruit contact.",
    )
    parser.add_argument(
        "--gripper-contact-current",
        type=float,
        default=DEFAULT_GRIPPER_CONTACT_CURRENT,
        help="Minimum absolute motor current for a settled pre-target stop to count as fruit contact.",
    )
    parser.add_argument(
        "--gripper-strong-contact-current",
        type=float,
        default=DEFAULT_GRIPPER_STRONG_CONTACT_CURRENT,
        help=(
            "High-confidence contact current that may override fruit-size progress uncertainty "
            "after the minimum safe travel fraction."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to persist the execution summary JSON.")
    args = parser.parse_args()

    plan = load_plan(args.plan, args.candidate_name, allow_failed_candidate=args.allow_prefix_diagnostic)
    summary = execute_plan(
        plan=plan,
        home_path=args.home,
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo,
        minimum_time_s=args.minimum_time_s,
        velocity_limit_rad_s=args.velocity_limit_rad_s,
        acceleration_limit_rad_s2=args.acceleration_limit_rad_s2,
        timeout_s=args.timeout_s,
        home_tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)),
        skip_home_check=args.skip_home_check,
        skip_resume_check=args.skip_resume_check,
        check_home_only=args.check_home_only,
        start_after=args.start_after,
        stop_after=args.stop_after,
        release_at_basket=not args.no_release_at_basket,
        execute=args.execute,
        allow_prefix_diagnostic=args.allow_prefix_diagnostic,
        gripper_url=args.gripper_url,
        use_gripper=args.use_gripper,
        gripper_target_source=args.gripper_target_source,
        left_open_position=args.left_open_position,
        left_open_reached_position=args.left_open_reached_position,
        left_open_tolerance=args.left_open_tolerance,
        left_close_position=args.left_close_position,
        gripper_open_current_limit=args.gripper_open_current_limit,
        gripper_close_current_limit=args.gripper_close_current_limit,
        gripper_hold_s=args.gripper_hold_s,
        gripper_post_command_wait_s=args.gripper_post_command_wait_s,
        gripper_post_command_poll_s=args.gripper_post_command_poll_s,
        gripper_close_min_motion=args.gripper_close_min_motion,
        gripper_settle_velocity=args.gripper_settle_velocity,
        gripper_settle_samples=args.gripper_settle_samples,
        gripper_close_target_tolerance=args.gripper_close_target_tolerance,
        gripper_close_min_progress_fraction=args.gripper_close_min_progress_fraction,
        gripper_contact_current=args.gripper_contact_current,
        gripper_strong_contact_current=args.gripper_strong_contact_current,
        arm_settle_position_tolerance_rad=args.arm_settle_position_tolerance_rad,
        arm_settle_velocity_rad_s=args.arm_settle_velocity_rad_s,
        arm_settle_samples=args.arm_settle_samples,
        arm_settle_timeout_s=args.arm_settle_timeout_s,
        arm_settle_poll_s=args.arm_settle_poll_s,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.execute:
        print("dry_run: pass --execute to command the robot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
