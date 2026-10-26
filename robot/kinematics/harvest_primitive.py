from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.geometry import orientation_residual
from robot.kinematics.ik_solver import solve_tcp_pose_ik
from robot.kinematics.types import IKSolverConfig


ARM_DOF = 7
HARVEST_PHASES = (
    "approach",
    "orient_soft_03",
    "orient_soft_07",
    "pre_grasp",
    "grasp",
    "harvest",
    "retract",
    "basket_place",
)
HARVEST_GRIPPER_EVENTS = (
    ("pre_grasp", "open"),
    ("grasp", "close"),
    ("basket_place", "release"),
)


def load_template_candidate(plan_path: Path, candidate_name: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    candidate = payload.get("best_candidate")
    if candidate_name is not None:
        matches = [
            item
            for item in payload.get("candidates", [])
            if isinstance(item, dict) and item.get("name") == candidate_name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one template candidate {candidate_name!r}, found {len(matches)}")
        candidate = matches[0]
    if not isinstance(candidate, dict):
        raise RuntimeError(f"template has no selected candidate: {plan_path}")
    names = tuple(str(item.get("name", "")) for item in candidate.get("waypoints", []))
    if names != HARVEST_PHASES:
        raise RuntimeError(f"template phase sequence mismatch: expected={HARVEST_PHASES}, actual={names}")
    return copy.deepcopy(candidate)


def template_surface_and_insertion_axis(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    waypoints = candidate["waypoints"]
    grasp = next(item for item in waypoints if item["name"] == "grasp")
    pre_grasp = next(item for item in waypoints if item["name"] == "pre_grasp")
    surface = grasp.get("surface_target_robot_m")
    if surface is None:
        raise RuntimeError("template grasp waypoint has no physical surface target")
    grasp_pose = np.asarray(grasp["final_tcp_pose"], dtype=float).reshape(4, 4)
    pre_pose = np.asarray(pre_grasp["final_tcp_pose"], dtype=float).reshape(4, 4)
    axis = grasp_pose[:3, 3] - pre_pose[:3, 3]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        raise RuntimeError("template pre_grasp->grasp insertion axis is degenerate")
    return np.asarray(surface, dtype=float).reshape(3), axis / norm


def size_normalized_insertion_profile(
    candidate: dict[str, Any],
    *,
    template_insertion_m: float,
    live_nominal_insertion_m: float | None,
) -> dict[str, Any]:
    template_insertion = float(template_insertion_m)
    template_nominal = float(
        candidate.get("manipulation_quality", {}).get(
            "nominal_grasp_insertion_m",
            template_insertion,
        )
    )
    if template_nominal <= 0.0 or template_insertion <= 0.0:
        raise RuntimeError(
            "template must provide positive nominal and demonstrated grasp insertion depths"
        )
    demonstrated_ratio = template_insertion / template_nominal
    if live_nominal_insertion_m is None:
        mode = "template_metric_depth"
        live_nominal = template_nominal
    else:
        live_nominal = float(live_nominal_insertion_m)
        if not 0.01 <= live_nominal <= 0.08:
            raise RuntimeError(
                f"selected fruit radius {live_nominal:.4f}m is outside the "
                "validated RGB-D harvest range [0.01, 0.08]m"
            )
        mode = "demonstration_ratio_scaled_by_live_rgbd_radius"
    scaled = demonstrated_ratio * live_nominal
    return {
        "mode": mode,
        "template_nominal_m": template_nominal,
        "live_nominal_m": live_nominal,
        "demonstrated_ratio": demonstrated_ratio,
        "scaled_insertion_m": scaled,
        "adjustment_m": scaled - template_insertion,
    }


def waypoint_shift(
    name: str,
    *,
    target_delta_m: np.ndarray,
    insertion_axis: np.ndarray,
    extra_insertion_m: float,
    height_offset_m: float,
    size_scaled_insertion_adjustment_m: float = 0.0,
) -> np.ndarray:
    shift = np.asarray(target_delta_m, dtype=float).reshape(3).copy()
    if name in ("grasp", "harvest"):
        shift += (
            float(extra_insertion_m) + float(size_scaled_insertion_adjustment_m)
        ) * np.asarray(insertion_axis, dtype=float).reshape(3)
    if name in ("pre_grasp", "grasp", "harvest"):
        shift += np.asarray([0.0, 0.0, float(height_offset_m)], dtype=float)
    return shift


def shift_waypoint_geometry(
    waypoint: dict[str, Any],
    *,
    pose_shift_m: np.ndarray,
    surface_shift_m: np.ndarray,
) -> None:
    pose_shift = np.asarray(pose_shift_m, dtype=float).reshape(3)
    surface_shift = np.asarray(surface_shift_m, dtype=float).reshape(3)
    for key in ("target_contact_position_robot_m", "grasp_anchor_robot_m"):
        if waypoint.get(key) is not None:
            waypoint[key] = (np.asarray(waypoint[key], dtype=float).reshape(3) + pose_shift).tolist()
    if waypoint.get("surface_target_robot_m") is not None:
        waypoint["surface_target_robot_m"] = (
            np.asarray(waypoint["surface_target_robot_m"], dtype=float).reshape(3) + surface_shift
        ).tolist()


def retarget_harvest_candidate(
    candidate: dict[str, Any],
    *,
    kinematics: Any,
    locked_state: Any,
    live_goal_robot_m: np.ndarray,
    extra_insertion_m: float,
    height_offset_m: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    joint_limit_margin_rad: float,
    live_nominal_grasp_insertion_m: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    template_surface, insertion_axis = template_surface_and_insertion_axis(candidate)
    live_surface = np.asarray(live_goal_robot_m, dtype=float).reshape(3)
    surface_delta = live_surface - template_surface
    template_grasp = next(item for item in candidate["waypoints"] if item["name"] == "grasp")
    template_contact = np.asarray(
        template_grasp.get("target_contact_position_robot_m") or template_grasp.get("grasp_anchor_robot_m"),
        dtype=float,
    ).reshape(3)
    template_insertion = float(np.dot(template_contact - template_surface, insertion_axis))
    insertion_profile = size_normalized_insertion_profile(
        candidate,
        template_insertion_m=template_insertion,
        live_nominal_insertion_m=live_nominal_grasp_insertion_m,
    )
    template_nominal_insertion = float(insertion_profile["template_nominal_m"])
    live_nominal_insertion = float(insertion_profile["live_nominal_m"])
    demonstrated_insertion_ratio = float(insertion_profile["demonstrated_ratio"])
    insertion_mode = str(insertion_profile["mode"])
    size_scaled_insertion = float(insertion_profile["scaled_insertion_m"])
    size_scaled_insertion_adjustment = float(insertion_profile["adjustment_m"])
    solved_waypoints: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    previous_q_arm: np.ndarray | None = None

    for source in candidate["waypoints"]:
        waypoint = copy.deepcopy(source)
        name = str(waypoint["name"])
        fallback = np.asarray(waypoint.get("ik_result", {}).get("q_arm_rad"), dtype=float)
        if fallback.shape != (ARM_DOF,):
            raise RuntimeError(f"template waypoint {name!r} has no seven-joint IK solution")

        if name == "basket_place":
            q_arm = fallback
            desired_pose = np.asarray(waypoint["final_tcp_pose"], dtype=float).reshape(4, 4)
            shift = np.zeros(3, dtype=float)
            retargeted = False
            solver_ok = True
            message = "fixed basket waypoint"
            iterations = 0
            residual_norm = 0.0
        else:
            base_pose = waypoint.get("final_tcp_pose") or waypoint.get("target_tcp_pose")
            if base_pose is None:
                raise RuntimeError(f"template waypoint {name!r} has no TCP pose")
            desired_pose = np.asarray(base_pose, dtype=float).reshape(4, 4).copy()
            shift = waypoint_shift(
                name,
                target_delta_m=surface_delta,
                insertion_axis=insertion_axis,
                extra_insertion_m=extra_insertion_m,
                height_offset_m=height_offset_m,
                size_scaled_insertion_adjustment_m=size_scaled_insertion_adjustment,
            )
            desired_pose[:3, 3] += shift
            seed_q_arm = fallback if previous_q_arm is None else previous_q_arm
            result = solve_tcp_pose_ik(
                kinematics,
                locked_state,
                desired_pose,
                seed_q_arm_rad=seed_q_arm,
                reference_q_arm_rad=seed_q_arm,
                config=IKSolverConfig(joint_limit_margin_rad=float(joint_limit_margin_rad)),
                enforce_joint_limit_margin=True,
            )
            q_arm = np.asarray(result.q_arm_rad, dtype=float).reshape(ARM_DOF)
            solver_ok = bool(result.ok)
            message = str(result.message)
            iterations = int(result.iterations)
            residual_norm = float(result.residual_norm)
            retargeted = True

        actual_pose = kinematics.tcp_pose(kinematics.q_full_from_arm(locked_state, q_arm))
        position_error = float(np.linalg.norm(actual_pose[:3, 3] - desired_pose[:3, 3]))
        orientation_error = float(
            np.linalg.norm(orientation_residual(actual_pose[:3, :3], desired_pose[:3, :3]))
        )
        endpoint_ok = bool(
            solver_ok
            and position_error <= float(position_tolerance_m)
            and orientation_error <= float(orientation_tolerance_rad)
        )
        report = {
            "name": name,
            "retargeted": retargeted,
            "shift_m": shift.tolist(),
            "ik_ok": solver_ok,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "position_tolerance_m": float(position_tolerance_m),
            "orientation_tolerance_rad": float(orientation_tolerance_rad),
            "seed_source": (
                "fixed_basket"
                if name == "basket_place"
                else "template" if previous_q_arm is None else "previous_waypoint"
            ),
            "ok": endpoint_ok,
        }
        reports.append(report)
        waypoint["target_tcp_pose"] = desired_pose.tolist()
        waypoint["final_tcp_pose"] = actual_pose.tolist()
        waypoint["ik_result"] = {
            "ok": solver_ok,
            "q_arm_rad": q_arm.tolist(),
            "residual_norm": residual_norm,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "message": message,
            "iterations": iterations,
        }
        waypoint["endpoint_gate"] = {
            "ok": endpoint_ok,
            "effective_ok": endpoint_ok,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "position_tolerance_m": float(position_tolerance_m),
            "orientation_tolerance_rad": float(orientation_tolerance_rad),
        }
        waypoint["target_conditioning"] = report
        if name != "basket_place":
            shift_waypoint_geometry(
                waypoint,
                pose_shift_m=shift,
                surface_shift_m=surface_delta,
            )
        solved_waypoints.append(waypoint)
        previous_q_arm = q_arm

    result_candidate = copy.deepcopy(candidate)
    result_candidate["name"] = f"{candidate['name']}_target_conditioned"
    result_candidate["waypoints"] = solved_waypoints
    continuity_transitions: list[dict[str, Any]] = []
    for source_start, source_end, solved_start, solved_end in zip(
        candidate["waypoints"],
        candidate["waypoints"][1:],
        solved_waypoints,
        solved_waypoints[1:],
    ):
        template_delta = np.asarray(source_end["ik_result"]["q_arm_rad"], dtype=float) - np.asarray(
            source_start["ik_result"]["q_arm_rad"], dtype=float
        )
        solved_delta = np.asarray(solved_end["ik_result"]["q_arm_rad"], dtype=float) - np.asarray(
            solved_start["ik_result"]["q_arm_rad"], dtype=float
        )
        template_max_abs = float(np.max(np.abs(template_delta)))
        template_l2 = float(np.linalg.norm(template_delta))
        solved_max_abs = float(np.max(np.abs(solved_delta)))
        solved_l2 = float(np.linalg.norm(solved_delta))
        max_abs_limit = max(0.35, 1.5 * template_max_abs + 0.15)
        l2_limit = max(0.50, 1.5 * template_l2 + 0.20)
        continuity_transitions.append(
            {
                "start": str(source_start["name"]),
                "end": str(source_end["name"]),
                "template_max_abs_delta_rad": template_max_abs,
                "template_l2_delta_rad": template_l2,
                "solved_max_abs_delta_rad": solved_max_abs,
                "solved_l2_delta_rad": solved_l2,
                "max_abs_limit_rad": max_abs_limit,
                "l2_limit_rad": l2_limit,
                "ok": bool(solved_max_abs <= max_abs_limit and solved_l2 <= l2_limit),
            }
        )
    result_candidate["joint_continuity_gate"] = {
        "method": "demonstration_relative_transition_envelope",
        "max_abs_scale": 1.5,
        "max_abs_additive_rad": 0.15,
        "max_abs_floor_rad": 0.35,
        "l2_scale": 1.5,
        "l2_additive_rad": 0.20,
        "l2_floor_rad": 0.50,
        "transitions": continuity_transitions,
        "ok": all(item["ok"] for item in continuity_transitions),
    }
    template_pregrasp = next(item for item in candidate["waypoints"] if item["name"] == "pre_grasp")
    result_grasp = next(item for item in solved_waypoints if item["name"] == "grasp")
    result_pregrasp = next(item for item in solved_waypoints if item["name"] == "pre_grasp")
    result_contact = np.asarray(
        result_grasp.get("target_contact_position_robot_m") or result_grasp.get("grasp_anchor_robot_m"),
        dtype=float,
    ).reshape(3)
    result_surface = np.asarray(result_grasp["surface_target_robot_m"], dtype=float).reshape(3)
    template_pregrasp_contact = np.asarray(
        template_pregrasp.get("target_contact_position_robot_m") or template_pregrasp.get("grasp_anchor_robot_m"),
        dtype=float,
    ).reshape(3)
    result_pregrasp_contact = np.asarray(
        result_pregrasp.get("target_contact_position_robot_m") or result_pregrasp.get("grasp_anchor_robot_m"),
        dtype=float,
    ).reshape(3)
    height_shift = np.asarray([0.0, 0.0, float(height_offset_m)], dtype=float)
    expected_pregrasp_offset = template_pregrasp_contact - template_surface + height_shift
    actual_pregrasp_offset = result_pregrasp_contact - live_surface
    total_insertion_adjustment = float(extra_insertion_m) + size_scaled_insertion_adjustment
    expected_grasp_offset = (
        template_contact - template_surface + total_insertion_adjustment * insertion_axis + height_shift
    )
    actual_grasp_offset = result_contact - live_surface
    pregrasp_offset_error = float(np.linalg.norm(actual_pregrasp_offset - expected_pregrasp_offset))
    grasp_offset_error = float(np.linalg.norm(actual_grasp_offset - expected_grasp_offset))
    actual_insertion = float(np.dot(result_contact - live_surface, insertion_axis))
    expected_insertion = size_scaled_insertion + float(extra_insertion_m)
    insertion_error = abs(actual_insertion - expected_insertion)
    surface_error = float(np.linalg.norm(result_surface - live_surface))
    geometry_gate = {
        "ok": bool(
            insertion_error <= 1e-6
            and surface_error <= 1e-6
            and pregrasp_offset_error <= 1e-6
            and grasp_offset_error <= 1e-6
        ),
        "point_semantics": "visible_fruit_surface",
        "template_surface_robot_m": template_surface.tolist(),
        "live_surface_robot_m": live_surface.tolist(),
        "result_surface_robot_m": result_surface.tolist(),
        "template_insertion_m": template_insertion,
        "template_nominal_grasp_insertion_m": template_nominal_insertion,
        "live_nominal_grasp_insertion_m": live_nominal_insertion,
        "demonstrated_insertion_ratio": demonstrated_insertion_ratio,
        "insertion_mode": insertion_mode,
        "size_scaled_insertion_m": size_scaled_insertion,
        "size_scaled_insertion_adjustment_m": size_scaled_insertion_adjustment,
        "extra_insertion_m": float(extra_insertion_m),
        "expected_insertion_m": expected_insertion,
        "actual_insertion_m": actual_insertion,
        "insertion_error_m": insertion_error,
        "surface_error_m": surface_error,
        "template_pregrasp_offset_m": (template_pregrasp_contact - template_surface).tolist(),
        "expected_pregrasp_offset_m": expected_pregrasp_offset.tolist(),
        "actual_pregrasp_offset_m": actual_pregrasp_offset.tolist(),
        "pregrasp_offset_error_m": pregrasp_offset_error,
        "expected_grasp_offset_m": expected_grasp_offset.tolist(),
        "actual_grasp_offset_m": actual_grasp_offset.tolist(),
        "grasp_offset_error_m": grasp_offset_error,
        "tolerance_m": 1e-6,
    }
    result_candidate["retarget_geometry_gate"] = geometry_gate
    result_candidate["gripper_events"] = [
        {"after_waypoint": waypoint, "action": action, "arm": "left"}
        for waypoint, action in HARVEST_GRIPPER_EVENTS
    ]
    result_candidate.setdefault("manipulation_quality", {}).update(
        {
            "nominal_grasp_insertion_m": live_nominal_insertion,
            "grasp_insertion_m": expected_insertion,
            "demonstrated_insertion_ratio": demonstrated_insertion_ratio,
            "insertion_mode": insertion_mode,
        }
    )
    retarget = {
        "point_semantics": "visible_fruit_surface",
        "template_surface_robot_m": template_surface.tolist(),
        "live_surface_robot_m": live_surface.tolist(),
        "surface_delta_m": surface_delta.tolist(),
        "insertion_axis_robot": insertion_axis.tolist(),
        "insertion_mode": insertion_mode,
        "template_nominal_grasp_insertion_m": template_nominal_insertion,
        "live_nominal_grasp_insertion_m": live_nominal_insertion,
        "demonstrated_insertion_ratio": demonstrated_insertion_ratio,
        "size_scaled_insertion_m": size_scaled_insertion,
        "size_scaled_insertion_adjustment_m": size_scaled_insertion_adjustment,
        "extra_insertion_m": float(extra_insertion_m),
        "height_offset_m": float(height_offset_m),
        "geometry_gate": geometry_gate,
        "reports": reports,
        "all_endpoints_ok": bool(geometry_gate["ok"] and all(item["ok"] for item in reports)),
    }
    return result_candidate, retarget


def _workspace_check(point: np.ndarray, bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    values = np.asarray(point, dtype=float).reshape(3)
    checks = {
        axis: float(bounds[axis][0]) <= float(value) <= float(bounds[axis][1])
        for axis, value in zip(("x", "y", "z"), values)
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "point_robot_m": values.tolist(),
        "bounds_m": {axis: [float(lo), float(hi)] for axis, (lo, hi) in bounds.items()},
    }


def validate_harvest_candidate(
    candidate: dict[str, Any],
    *,
    kinematics: Any,
    locked_state: Any,
    start_q_arm_rad: np.ndarray,
    joint_limit_margin_rad: float,
    minimum_clearance_m: float,
    clearance_pair_count: int,
    samples_per_segment: int,
    manipulation_bounds: dict[str, tuple[float, float]],
    transit_bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    waypoints = candidate.get("waypoints", [])
    names = tuple(str(item.get("name", "")) for item in waypoints)
    phase_order_ok = names == HARVEST_PHASES
    endpoint_ok = phase_order_ok and all(
        bool(item.get("endpoint_gate", {}).get("effective_ok", False)) for item in waypoints
    )
    q_rows = [np.asarray(item["ik_result"]["q_arm_rad"], dtype=float).reshape(ARM_DOF) for item in waypoints]
    lower, upper = kinematics.active_joint_limits(locked_state.full_q_rad)
    joint_violations: list[dict[str, Any]] = []
    for name, q_arm in zip(names, q_rows):
        for index in np.flatnonzero(
            (q_arm < lower + float(joint_limit_margin_rad))
            | (q_arm > upper - float(joint_limit_margin_rad))
        ):
            joint_violations.append(
                {
                    "waypoint": name,
                    "joint_index": int(index),
                    "q_rad": float(q_arm[index]),
                    "lower_rad": float(lower[index]),
                    "upper_rad": float(upper[index]),
                }
            )

    workspace_violations: list[dict[str, Any]] = []
    clearance_violations: list[dict[str, Any]] = []
    minimum_distance: float | None = None
    previous = np.asarray(start_q_arm_rad, dtype=float).reshape(ARM_DOF)
    sample_count = max(2, int(samples_per_segment))
    for segment_index, (name, target) in enumerate(zip(names, q_rows)):
        bounds = transit_bounds if segment_index == 0 or name == "basket_place" else manipulation_bounds
        for sample_index in range(1, sample_count + 1):
            alpha = float(sample_index) / float(sample_count)
            q_arm = (1.0 - alpha) * previous + alpha * target
            q_full = kinematics.q_full_from_arm(locked_state, q_arm)
            tcp = kinematics.tcp_pose(q_full)[:3, 3]
            workspace = _workspace_check(tcp, bounds)
            if not workspace["ok"]:
                workspace_violations.append(
                    {
                        "segment_end": name,
                        "sample_index": sample_index,
                        "alpha": alpha,
                        **workspace,
                    }
                )
            clearance = nearest_self_clearances(
                kinematics,
                q_full,
                pair_count=int(clearance_pair_count),
                margin_m=float(minimum_clearance_m),
            )
            distance = clearance.minimum_distance_m
            if distance is not None and (minimum_distance is None or distance < minimum_distance):
                minimum_distance = float(distance)
            if not clearance.ok:
                clearance_violations.append(
                    {
                        "segment_end": name,
                        "sample_index": sample_index,
                        "alpha": alpha,
                        "minimum_distance_m": distance,
                    }
                )
        previous = target

    gripper_events = candidate.get("gripper_events")
    if not isinstance(gripper_events, list):
        gripper_events = [
            {"after_waypoint": waypoint, "action": action, "arm": "left"}
            for waypoint, action in HARVEST_GRIPPER_EVENTS
        ]
    actual_gripper_events = tuple(
        (str(item.get("after_waypoint")), str(item.get("action")))
        for item in gripper_events
        if isinstance(item, dict)
    )
    gripper_schedule_ok = actual_gripper_events == HARVEST_GRIPPER_EVENTS
    gripper_schedule = [
        f"{waypoint}_{action}"
        for waypoint, action in actual_gripper_events
    ]
    retarget_geometry = candidate.get("retarget_geometry_gate", {"ok": True, "not_applicable": True})
    joint_continuity = candidate.get("joint_continuity_gate", {"ok": True, "not_applicable": True})
    hard_gate = {
        "phase_order_ok": phase_order_ok,
        "endpoint_ok": endpoint_ok,
        "joint_limits_ok": not joint_violations,
        "workspace_ok": not workspace_violations,
        "self_clearance_ok": not clearance_violations,
        "gripper_schedule_ok": bool(phase_order_ok and gripper_schedule_ok),
        "retarget_geometry_ok": bool(retarget_geometry.get("ok", False)),
        "retarget_geometry": retarget_geometry,
        "joint_continuity_ok": bool(joint_continuity.get("ok", False)),
        "joint_continuity": joint_continuity,
        "left_arm_only": True,
        "joint_limit_margin_rad": float(joint_limit_margin_rad),
        "minimum_clearance_m": float(minimum_clearance_m),
        "minimum_observed_clearance_m": minimum_distance,
        "joint_limit_violations": joint_violations,
        "workspace_violations": workspace_violations,
        "self_clearance_violations": clearance_violations,
        "gripper_schedule": gripper_schedule,
        "gripper_events": gripper_events,
    }
    hard_gate["ok"] = all(
        bool(hard_gate[key])
        for key in (
            "phase_order_ok",
            "endpoint_ok",
            "joint_limits_ok",
            "workspace_ok",
            "self_clearance_ok",
            "gripper_schedule_ok",
            "retarget_geometry_ok",
            "joint_continuity_ok",
            "left_arm_only",
        )
    )
    return hard_gate
