from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.geometry import normalize, transform_from_rotation_translation
from robot.kinematics.ik_solver import solve_tcp_pose_ik
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.types import IKSolverConfig
from robot.kinematics.utils.capture_live_realsense import (
    build_point_cloud,
    capture_realsense,
    default_transform_json,
    pick_pixel,
    point_from_pixel,
    point_to_robot,
    write_capture_artifacts,
)


REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_DATASET = "a270c038"
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_SURFEL_PYTHON = Path("C:/path/to/user/miniconda3/envs/surfel_splatting/python.exe")
DEFAULT_POWER = ".*"
DEFAULT_SERVO = ".*"
ARM_DOF = 7


def default_toolfit_json(dataset_name: str) -> Path:
    return (
        REPO_ROOT
        / "robot"
        / "calibration"
        / "session"
        / dataset_name
        / "T_robot_from_aria_selected_current_best_target_toolfit.json"
    )


def select_point(
    *,
    capture: dict[str, Any],
    point_cloud: dict[str, np.ndarray],
    pick_mode: str,
    pixel_xy: tuple[int, int] | None,
    pick_radius_px: int,
    min_depth_m: float,
    max_depth_m: float,
    axis_size_m: float,
) -> dict[str, Any]:
    if pixel_xy is not None:
        selected = point_from_pixel(
            pixel_xy=pixel_xy,
            depth_m=capture["depth_m"],
            intrinsics=capture["intrinsics"],
            radius_px=pick_radius_px,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        selected["selection_source"] = "cli_pixel"
        return selected
    if pick_mode == "rgb":
        selected = point_from_pixel(
            pixel_xy=pick_pixel(capture["color_bgr"]),
            depth_m=capture["depth_m"],
            intrinsics=capture["intrinsics"],
            radius_px=pick_radius_px,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        selected["selection_source"] = "opencv_rgb"
        return selected
    raise RuntimeError("Open3D selection is handled by the external surfel_splatting picker")


def aim_pose_from_point(
    *,
    point_robot_m: np.ndarray,
    home_tcp_position_robot_m: np.ndarray,
    stand_off_m: float,
    max_step_m: float,
    target_mode: str,
    contact_offset_tcp_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    point = np.asarray(point_robot_m, dtype=float).reshape(3)
    home_tcp = np.asarray(home_tcp_position_robot_m, dtype=float).reshape(3)
    contact_offset = np.asarray(contact_offset_tcp_m, dtype=float).reshape(3)
    backoff_axis = normalize(home_tcp - point)
    contact_center = point + float(stand_off_m) * backoff_axis
    rotation = aim_rotation_from_backoff(backoff_axis)
    if target_mode == "hover":
        target_position = contact_center
    elif target_mode == "contact_pregrasp":
        target_position = contact_center - rotation @ contact_offset
    else:
        raise ValueError(f"unsupported target mode: {target_mode}")
    uncapped_target_position = target_position.copy()
    step = target_position - home_tcp
    step_norm = float(np.linalg.norm(step))
    capped = False
    if max_step_m > 0.0 and step_norm > float(max_step_m):
        target_position = home_tcp + step * (float(max_step_m) / step_norm)
        capped = True
    actual_contact_center = target_position + rotation @ contact_offset
    diagnostics = {
        "target_mode": target_mode,
        "selected_point_robot_m": point.tolist(),
        "home_tcp_position_robot_m": home_tcp.tolist(),
        "stand_off_m": float(stand_off_m),
        "max_step_m": float(max_step_m),
        "contact_offset_tcp_m": contact_offset.tolist(),
        "uncapped_contact_center_robot_m": contact_center.tolist(),
        "uncapped_target_tcp_position_robot_m": uncapped_target_position.tolist(),
        "uncapped_step_from_home_m": step_norm,
        "step_capped": capped,
        "backoff_axis_robot": backoff_axis.tolist(),
        "target_tcp_position_robot_m": target_position.tolist(),
        "actual_contact_center_robot_m": actual_contact_center.tolist(),
        "target_step_from_home_m": float(np.linalg.norm(target_position - home_tcp)),
        "contact_center_distance_to_selected_point_m": float(np.linalg.norm(actual_contact_center - point)),
        "tcp_distance_to_selected_point_m": float(np.linalg.norm(target_position - point)),
    }
    return transform_from_rotation_translation(rotation, target_position), diagnostics


def aim_rotation_from_backoff(backoff_axis_robot: np.ndarray) -> np.ndarray:
    z_axis = normalize(backoff_axis_robot)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, up))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def workspace_check(point_robot_m: np.ndarray, bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    point = np.asarray(point_robot_m, dtype=float).reshape(3)
    checks = {
        axis: float(bounds[axis][0]) <= float(value) <= float(bounds[axis][1])
        for axis, value in zip(("x", "y", "z"), point)
    }
    return {
        "ok": bool(all(checks.values())),
        "checks": checks,
        "bounds_m": {axis: [float(lo), float(hi)] for axis, (lo, hi) in bounds.items()},
    }


def validate_home_state(robot: Any, home: dict[str, Any], *, tolerance_rad: float) -> dict[str, Any]:
    live = np.asarray(robot.get_state().position, dtype=float)
    expected = np.asarray(home["all_joints"]["positions_rad"], dtype=float)
    indices = commanded_joint_indices(home)
    delta = live[indices] - expected[indices]
    abs_delta = np.abs(delta)
    local_max_index = int(np.argmax(abs_delta))
    max_index = int(indices[local_max_index])
    max_abs_delta = float(abs_delta[local_max_index])
    joint_names = list(home["joint_names"])
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
        "checked_joint_count": int(indices.size),
    }


def commanded_joint_indices(home: dict[str, Any]) -> np.ndarray:
    joint_names = list(home["joint_names"])
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    selected: list[int] = []
    selected.extend(indices_for_names(name_to_index, home["torso"]["joint_names"]))
    selected.extend(indices_for_names(name_to_index, home["arms"]["right"]["joint_names"]))
    selected.extend(indices_for_names(name_to_index, home["arms"]["left"]["joint_names"]))
    selected.extend(indices_for_names(name_to_index, home["head"]["joint_names"]))
    return np.asarray(selected, dtype=int)


def indices_for_names(name_to_index: dict[str, int], names: list[str]) -> list[int]:
    return [int(name_to_index[name]) for name in names]


def command_positions(home: dict[str, Any], group: str, side: str | None = None) -> np.ndarray:
    values = home[group]["positions_rad"] if side is None else home[group][side]["positions_rad"]
    return np.asarray(values, dtype=float)


def command_arm(
    home: dict[str, Any],
    arm: str,
    q_arm: np.ndarray,
    *,
    minimum_time_s: float,
    velocity_limit_rad_s: float | None = None,
    acceleration_limit_rad_s2: float | None = None,
    control_hold_time_s: float | None = None,
) -> Any:
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    q = np.asarray(q_arm, dtype=float).reshape(ARM_DOF)
    left_position = q if arm == "left" else command_positions(home, "arms", "left")
    right_position = q if arm == "right" else command_positions(home, "arms", "right")

    def _joint_cmd(position: Any) -> Any:
        pos = np.asarray(position, dtype=float).reshape(-1)
        builder = rby.JointPositionCommandBuilder().set_minimum_time(float(minimum_time_s)).set_position(position)
        if velocity_limit_rad_s is not None:
            builder = builder.set_velocity_limit(np.full(pos.shape[0], float(velocity_limit_rad_s), dtype=float))
        if acceleration_limit_rad_s2 is not None:
            builder = builder.set_acceleration_limit(np.full(pos.shape[0], float(acceleration_limit_rad_s2), dtype=float))
        if control_hold_time_s is not None:  # streaming: keep tracking the last target between paced sends
            builder = builder.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(float(control_hold_time_s)))
        return builder

    body = (
        rby.BodyComponentBasedCommandBuilder()
        .set_torso_command(_joint_cmd(command_positions(home, "torso")))
        .set_right_arm_command(_joint_cmd(right_position))
        .set_left_arm_command(_joint_cmd(left_position))
    )
    head = _joint_cmd(command_positions(home, "head"))
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body).set_head_command(head)
    )


def command_arm_only(
    arm: str,
    q_arm: np.ndarray,
    *,
    minimum_time_s: float,
    velocity_limit_rad_s: float | None = None,
    acceleration_limit_rad_s2: float | None = None,
    control_hold_time_s: float | None = None,
) -> Any:
    """Build a command for one arm without commanding any fixed joint group."""
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    position = np.asarray(q_arm, dtype=float).reshape(ARM_DOF)
    joint = rby.JointPositionCommandBuilder().set_minimum_time(float(minimum_time_s)).set_position(position)
    if velocity_limit_rad_s is not None:
        joint = joint.set_velocity_limit(np.full(ARM_DOF, float(velocity_limit_rad_s), dtype=float))
    if acceleration_limit_rad_s2 is not None:
        joint = joint.set_acceleration_limit(np.full(ARM_DOF, float(acceleration_limit_rad_s2), dtype=float))
    if control_hold_time_s is not None:
        joint = joint.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(float(control_hold_time_s))
        )
    body = rby.BodyComponentBasedCommandBuilder()
    if arm == "left":
        body = body.set_left_arm_command(joint)
    else:
        body = body.set_right_arm_command(joint)
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body)
    )


def command_left_arm(home: dict[str, Any], q_left: np.ndarray, *, minimum_time_s: float) -> Any:
    return command_arm(home, "left", q_left, minimum_time_s=minimum_time_s)


def connect_motion_robot(*, address: str, model: str, power: str, servo: str) -> Any:
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


def path_clearance_report(
    kinematics: RBY1Kinematics,
    locked_state: Any,
    start_q_arm: np.ndarray,
    end_q_arm: np.ndarray,
    *,
    samples: int,
    margin_m: float,
    pair_count: int,
) -> dict[str, Any]:
    start = np.asarray(start_q_arm, dtype=float).reshape(ARM_DOF)
    end = np.asarray(end_q_arm, dtype=float).reshape(ARM_DOF)
    sample_count = max(1, int(samples))
    minimum_distance = None
    worst_index = 0
    for index in range(sample_count + 1):
        alpha = float(index) / float(sample_count)
        q_arm = (1.0 - alpha) * start + alpha * end
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        clearance = nearest_self_clearances(
            kinematics,
            q_full,
            pair_count=pair_count,
            margin_m=margin_m,
        )
        distance = clearance.minimum_distance_m
        if distance is not None and (minimum_distance is None or distance < minimum_distance):
            minimum_distance = float(distance)
            worst_index = index
    return {
        "ok": minimum_distance is None or minimum_distance >= float(margin_m),
        "minimum_distance_m": minimum_distance,
        "margin_m": float(margin_m),
        "path_samples": sample_count,
        "worst_sample_index": int(worst_index),
        "worst_alpha": float(worst_index) / float(sample_count),
    }


def bounds_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return {
        "x": (float(args.min_x_m), float(args.max_x_m)),
        "y": (float(args.min_y_m), float(args.max_y_m)),
        "z": (float(args.min_z_m), float(args.max_z_m)),
    }


def parse_vector3(value: str | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    return np.asarray([float(part) for part in parts], dtype=float)


def contact_offset_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.contact_offset_tcp_m is not None and args.contact_frame_json is not None:
        raise ValueError("use either --contact-offset-tcp-m or --contact-frame-json, not both")
    if args.contact_frame_json is not None:
        data = json.loads(args.contact_frame_json.read_text(encoding="utf-8"))
        return np.asarray(data["contact_offset_tcp_m"], dtype=float).reshape(3)
    parsed = parse_vector3(args.contact_offset_tcp_m, name="contact offset")
    if parsed is None:
        return np.zeros(3, dtype=float)
    return parsed


def metadata_artifact_path(*, runs_root: Path, dataset_name: str, run_label: str) -> Path:
    return artifact_path("realsense_click_move_arm", f"{dataset_name}_{run_label}", runs_root=runs_root)


def write_metadata_artifact(metadata: dict[str, Any]) -> None:
    artifact_json = metadata.get("artifacts", {}).get("json")
    if artifact_json:
        Path(artifact_json).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_external_open3d_picker(
    *,
    surfel_python: Path,
    capture_npz: Path,
    transform_json: Path,
    dataset_name: str,
    axis_size_m: float,
    output_json: Path,
    confirm_execute: bool,
) -> dict[str, Any]:
    if not surfel_python.exists():
        raise FileNotFoundError(f"surfel_splatting python not found: {surfel_python}")
    command = [
        str(surfel_python),
        "-B",
        str(REPO_ROOT / "robot" / "kinematics" / "utils" / "pick_realsense_capture_point.py"),
        str(capture_npz),
        "--dataset-name",
        dataset_name,
        "--transform-json",
        str(transform_json),
        "--axis-size-m",
        f"{float(axis_size_m):.6f}",
        "--output-json",
        str(output_json),
    ]
    if confirm_execute:
        command.append("--confirm-execute-key")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Open3D point picker failed with exit code {completed.returncode}")
    if not output_json.exists():
        raise RuntimeError(f"Open3D point picker did not write expected selection JSON: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if "selected_point" not in payload:
        raise RuntimeError(f"Open3D point picker output is missing selected_point: {output_json}")
    return payload


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    transform_json = args.transform_json or default_transform_json(args.dataset_name)
    run_label = args.run_label or "click_move_" + time.strftime("%H%M%S")
    selected_robot = parse_vector3(args.selected_point_robot_m, name="selected robot point")
    capture = None
    point_cloud = None
    capture_paths = None
    picker_payload = None
    execute_confirmation_required = False
    execute_confirmed = None
    if selected_robot is None:
        capture = capture_realsense(
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            serial=args.serial,
        )
        point_cloud = build_point_cloud(
            color_bgr=capture["color_bgr"],
            depth_m=capture["depth_m"],
            intrinsics=capture["intrinsics"],
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            stride=args.pointcloud_stride,
        )
        if args.pick_mode == "open3d" and args.pixel_x is None:
            capture_metadata = {
                "method": "realsense_click_move_arm_capture_for_open3d_picker",
                "dataset_name": args.dataset_name,
                "run_label": run_label,
                "transform_json": str(transform_json),
            }
            capture_paths = write_capture_artifacts(
                dataset_name=args.dataset_name,
                run_label=run_label,
                runs_root=args.runs_root,
                capture=capture,
                point_cloud=point_cloud,
                metadata=capture_metadata,
            )
            picker_json = metadata_artifact_path(
                runs_root=args.runs_root,
                dataset_name=args.dataset_name,
                run_label=run_label + "_selection",
            )
            execute_confirmation_required = bool(args.execute and not args.no_execute_key_confirm)
            picker_payload = run_external_open3d_picker(
                surfel_python=args.surfel_python,
                capture_npz=Path(capture_paths["npz"]),
                transform_json=transform_json,
                dataset_name=args.dataset_name,
                axis_size_m=args.axis_size_m,
                output_json=picker_json,
                confirm_execute=execute_confirmation_required,
            )
            selected = dict(picker_payload["selected_point"])
            selected["selection_source"] = "external_open3d_capture"
            point_robot = selected["point_robot_m"]
            execute_confirmed = (
                bool(picker_payload.get("execute_confirmed", False))
                if execute_confirmation_required
                else bool(args.execute)
            )
        else:
            selected = select_point(
                capture=capture,
                point_cloud=point_cloud,
                pick_mode=args.pick_mode,
                pixel_xy=None if args.pixel_x is None else (int(args.pixel_x), int(args.pixel_y)),
                pick_radius_px=args.pick_radius_px,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                axis_size_m=args.axis_size_m,
            )
            point_robot = point_to_robot(np.asarray(selected["point_realsense_m"], dtype=float), transform_json)
            execute_confirmed = bool(args.execute)
        if point_robot is None:
            raise RuntimeError("robot transform is required for motion")
        selected["point_robot_m"] = point_robot
    else:
        point_robot = selected_robot.tolist()
        selected = {
            "selection_source": "cli_robot_point",
            "point_robot_m": point_robot,
        }
        execute_confirmed = bool(args.execute)

    home = json.loads(args.home.read_text(encoding="utf-8"))
    toolfit = args.toolfit or default_toolfit_json(args.dataset_name)
    kinematics = RBY1Kinematics.from_live_robot(
        address=args.address,
        model_name=args.model,
        arm=args.arm,
        toolfit_path=toolfit,
    )
    locked_state = kinematics.locked_state_from_home(args.home)
    home_tcp = kinematics.tcp_pose(locked_state.full_q_rad)[:3, 3]
    target_pose, pose_diagnostics = aim_pose_from_point(
        point_robot_m=np.asarray(point_robot, dtype=float),
        home_tcp_position_robot_m=home_tcp,
        stand_off_m=args.stand_off_m,
        max_step_m=args.max_step_m,
        target_mode=args.target_mode,
        contact_offset_tcp_m=contact_offset_from_args(args),
    )
    workspace = workspace_check(target_pose[:3, 3], bounds_from_args(args))
    if not workspace["ok"]:
        raise RuntimeError(f"target stand-off pose outside configured workspace: {workspace}")

    config = IKSolverConfig(
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        posture_weight=args.posture_weight,
        smoothness_weight=args.smoothness_weight,
        clearance_weight=args.clearance_weight,
        clearance_margin_m=args.clearance_margin_m,
        clearance_pair_count=args.clearance_pair_count,
        max_iterations=args.max_iterations,
    )
    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    result = solve_tcp_pose_ik(
        kinematics,
        locked_state,
        target_pose,
        seed_q_arm_rad=home_q_arm,
        reference_q_arm_rad=home_q_arm,
        config=config,
    )
    q_full = kinematics.q_full_from_arm(locked_state, result.q_arm_rad)
    clearance = nearest_self_clearances(
        kinematics,
        q_full,
        pair_count=args.clearance_pair_count,
        margin_m=args.min_clearance_m,
    )
    path_clearance = path_clearance_report(
        kinematics,
        locked_state,
        home_q_arm,
        result.q_arm_rad,
        samples=args.path_samples,
        margin_m=args.min_clearance_m,
        pair_count=args.clearance_pair_count,
    )
    joint_delta = np.asarray(result.q_arm_rad, dtype=float) - np.asarray(home_q_arm, dtype=float)
    max_joint_delta_rad = float(np.max(np.abs(joint_delta)))
    joint_delta_ok = max_joint_delta_rad <= float(np.deg2rad(args.max_joint_delta_deg))
    solve_ok = (
        bool(result.ok)
        and result.position_error_m <= float(args.position_tolerance_m)
        and clearance.ok
        and path_clearance["ok"]
        and joint_delta_ok
    )

    should_execute = bool(args.execute and (not execute_confirmation_required or execute_confirmed))
    metadata = {
        "method": "realsense_click_move_arm_calibration_probe",
        "dataset_name": args.dataset_name,
        "run_label": run_label,
        "frame": "realsense_color_optical_frame",
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "warmup_frames": int(args.warmup_frames),
        "min_depth_m": float(args.min_depth_m),
        "max_depth_m": float(args.max_depth_m),
        "pointcloud_stride": int(args.pointcloud_stride),
        "point_count": None if point_cloud is None else int(point_cloud["points"].shape[0]),
        "intrinsics": None if capture is None else capture["intrinsics"],
        "depth_scale": None if capture is None else float(capture["depth_scale"]),
        "frame_metadata": None if capture is None else capture["frame_metadata"],
        "transform_json": str(transform_json),
        "selected_point": selected,
        "target_pose": target_pose.tolist(),
        "target_pose_diagnostics": pose_diagnostics,
        "workspace_check": workspace,
        "ik_result": {
            "ok": bool(result.ok),
            "hard_gate_ok": bool(solve_ok),
            "q_arm_rad": result.q_arm_rad.tolist(),
            "position_error_m": float(result.position_error_m),
            "orientation_error_rad": float(result.orientation_error_rad),
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "message": result.message,
        },
        "clearance": {
            "ok": clearance.ok,
            "minimum_distance_m": clearance.minimum_distance_m,
            "margin_m": float(args.min_clearance_m),
        },
        "path_clearance": path_clearance,
        "joint_delta": {
            "ok": bool(joint_delta_ok),
            "max_abs_delta_rad": max_joint_delta_rad,
            "max_abs_delta_deg": float(np.rad2deg(max_joint_delta_rad)),
            "limit_deg": float(args.max_joint_delta_deg),
            "delta_rad": joint_delta.tolist(),
        },
        "robot": {
            "address": args.address,
            "model": args.model,
            "arm": args.arm,
            "home": str(args.home),
            "toolfit": str(toolfit),
        },
        "execute_requested": bool(args.execute),
        "execute_confirmation_required": bool(execute_confirmation_required),
        "execute_confirmed": execute_confirmed,
        "execute": bool(should_execute),
    }
    if picker_payload is not None:
        metadata["picker_artifact_json"] = str(
            metadata_artifact_path(
                runs_root=args.runs_root,
                dataset_name=args.dataset_name,
                run_label=run_label + "_selection",
            )
        )
    if capture is None or point_cloud is None:
        artifact = metadata_artifact_path(
            runs_root=args.runs_root,
            dataset_name=args.dataset_name,
            run_label=run_label,
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        metadata["artifacts"] = {"json": str(artifact)}
        write_metadata_artifact(metadata)
    elif capture_paths is not None:
        metadata["artifacts"] = capture_paths
        write_metadata_artifact(metadata)
    else:
        metadata["artifacts"] = write_capture_artifacts(
            dataset_name=args.dataset_name,
            run_label=run_label,
            runs_root=args.runs_root,
            capture=capture,
            point_cloud=point_cloud,
            metadata=metadata,
        )
    if not solve_ok:
        metadata["motion"] = {"executed": False, "reason": "ik_or_clearance_gate_failed"}
        write_metadata_artifact(metadata)
        return metadata
    if not should_execute:
        reason = "dry_run" if not args.execute else "execute_not_confirmed"
        metadata["motion"] = {"executed": False, "reason": reason}
        write_metadata_artifact(metadata)
        return metadata

    robot = connect_motion_robot(
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo,
    )
    metadata["home_check"] = validate_home_state(
        robot,
        home,
        tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)),
    )
    feedback = robot.send_command(
        command_arm(home, args.arm, result.q_arm_rad, minimum_time_s=args.minimum_time_s),
        int(max(float(args.timeout_s), float(args.minimum_time_s) + 5.0)),
    ).get()
    finish_ok = feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
    if not finish_ok:
        raise RuntimeError(f"robot command failed: {feedback.finish_code}")
    metadata["motion"] = {"executed": True, "finish_code": str(feedback.finish_code)}
    write_metadata_artifact(metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Click a live RealSense RGB-D point and optionally move the RB-Y1 left arm toward it."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--toolfit", type=Path, default=None)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--pick-mode", choices=("open3d", "rgb"), default="open3d")
    parser.add_argument("--pick-radius-px", type=int, default=3)
    parser.add_argument("--pixel-x", type=int, default=None)
    parser.add_argument("--pixel-y", type=int, default=None)
    parser.add_argument("--selected-point-robot-m", default=None)
    parser.add_argument("--axis-size-m", type=float, default=0.1)
    parser.add_argument("--surfel-python", type=Path, default=DEFAULT_SURFEL_PYTHON)
    parser.add_argument("--no-execute-key-confirm", action="store_true")
    parser.add_argument("--target-mode", choices=("hover", "contact_pregrasp"), default="hover")
    parser.add_argument("--contact-offset-tcp-m", default=None)
    parser.add_argument("--contact-frame-json", type=Path, default=None)
    parser.add_argument("--stand-off-m", type=float, default=None)
    parser.add_argument("--max-step-m", type=float, default=None)
    parser.add_argument("--position-tolerance-m", type=float, default=0.015)
    parser.add_argument("--min-clearance-m", type=float, default=0.03)
    parser.add_argument("--path-samples", type=int, default=20)
    parser.add_argument("--max-joint-delta-deg", type=float, default=120.0)
    parser.add_argument("--orientation-weight", type=float, default=0.0)
    parser.add_argument("--position-weight", type=float, default=20.0)
    parser.add_argument("--posture-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--clearance-weight", type=float, default=2.0)
    parser.add_argument("--clearance-margin-m", type=float, default=0.03)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--min-x-m", type=float, default=0.25)
    parser.add_argument("--max-x-m", type=float, default=0.85)
    parser.add_argument("--min-y-m", type=float, default=-0.45)
    parser.add_argument("--max-y-m", type=float, default=0.45)
    parser.add_argument("--min-z-m", type=float, default=0.35)
    parser.add_argument("--max-z-m", type=float, default=1.20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--power", default=DEFAULT_POWER)
    parser.add_argument("--servo", default=DEFAULT_SERVO)
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--minimum-time-s", type=float, default=8.0)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    args = parser.parse_args()
    if (args.pixel_x is None) != (args.pixel_y is None):
        parser.error("--pixel-x and --pixel-y must be provided together")
    if args.selected_point_robot_m is not None and args.pixel_x is not None:
        parser.error("--selected-point-robot-m cannot be combined with --pixel-x/--pixel-y")
    if args.contact_offset_tcp_m is not None and args.contact_frame_json is not None:
        parser.error("--contact-offset-tcp-m cannot be combined with --contact-frame-json")
    if args.stand_off_m is None:
        args.stand_off_m = 0.03 if args.target_mode == "contact_pregrasp" else 0.08
    if args.max_step_m is None:
        args.max_step_m = 0.0 if args.target_mode == "contact_pregrasp" else 0.12
    if args.stand_off_m < 0.03:
        parser.error("--stand-off-m must be at least 0.03 for this safety probe")
    if args.path_samples < 1:
        parser.error("--path-samples must be at least 1")
    if args.max_joint_delta_deg <= 0.0:
        parser.error("--max-joint-delta-deg must be positive")

    payload = run_probe(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.execute:
        print("dry_run: pass --execute to command the robot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
