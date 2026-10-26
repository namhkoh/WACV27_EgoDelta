from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rby1_sdk as rby

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.geometry import normalize, transform_from_rotation_translation
from robot.kinematics.hand_prior import (
    HandTrackingSample,
    PhaseWindow,
    RobotFrameMapper,
    TimeWindow,
    build_robot_frame_mapper,
    load_frame_timestamps,
    load_hand_tracking,
    load_slam_trajectory,
    phase_time_windows,
)
from robot.kinematics.ik_solver import solve_tcp_task_axis_ik
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.types import IKSolverConfig
from robot.kinematics.utils.capture_live_realsense import (
    build_point_cloud,
    capture_realsense,
    default_transform_json,
    point_to_robot,
    write_capture_artifacts,
)
from robot.kinematics.utils.realsense_click_move_arm import (
    DEFAULT_ADDRESS,
    DEFAULT_DATASET,
    DEFAULT_HOME,
    DEFAULT_POWER,
    DEFAULT_SERVO,
    DEFAULT_SURFEL_PYTHON,
    command_arm_only,
    connect_motion_robot,
    parse_vector3,
    path_clearance_report,
    run_external_open3d_picker,
    validate_home_state,
    workspace_check,
)


REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_MANIFEST = REPO_ROOT / "data" / "policy_training" / "a270c038" / "selections_deleafing" / "phase_manifest.json"
DEFAULT_RIGHT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "core" / "T_robot_from_aria_featuredepth_hanger_trusted_right_toolfit.json"
DEFAULT_SLAM_TRAJECTORY = (
    Path("C:/path/to/user/aria/policy_training")
    / "mps_a270c038-8f08-4adb-98cc-18541bfb2165_vrs"
    / "slam"
    / "closed_loop_trajectory.csv"
)
DEFAULT_DELEAF_CONFIG = REPO_ROOT / "robot" / "kinematics" / "configs" / "deleaf_right_knife_default.json"
WORLD_DOWN = np.asarray([0.0, 0.0, -1.0], dtype=float)

HandSide = Literal["left", "right"]
HandPoint = Literal["wrist", "pinch"]


@dataclass(frozen=True)
class DeleafMpsGuidance:
    action_name: str
    hand_side: HandSide
    approach_axis_robot: np.ndarray
    cut_axis_robot: np.ndarray
    retract_axis_robot: np.ndarray
    confidence: float
    diagnostics: dict[str, Any]
    cut_anchor_robot_m: np.ndarray | None = None
    approach_path_robot_m: np.ndarray | None = None


@dataclass(frozen=True)
class DeleafWaypoint:
    name: str
    target_pose: np.ndarray
    orientation_required: bool


def load_deleaf_tool_axes(toolfit_path: Path) -> dict[str, Any]:
    data = json.loads(Path(toolfit_path).read_text(encoding="utf-8"))
    edge_axis = data.get("edge_axis_eef_unit")
    if edge_axis is None:
        return {
            "mode": "legacy_gripper_axes",
            "source": str(toolfit_path),
            "primary_axis_local": np.asarray([0.0, -1.0, 0.0], dtype=float),
            "secondary_axis_local": np.asarray([0.0, 0.0, 1.0], dtype=float),
        }
    primary = normalize(np.asarray(edge_axis, dtype=float).reshape(3))
    seed = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(seed, primary))) > 0.9:
        seed = np.asarray([0.0, 1.0, 0.0], dtype=float)
    secondary = normalize(seed - float(np.dot(seed, primary)) * primary)
    return {
        "mode": "calibrated_knife_edge_axis",
        "source": str(toolfit_path),
        "primary_axis_local": primary,
        "secondary_axis_local": secondary,
        "edge_fit": {
            "line_concentration": data.get("line_concentration"),
            "angle_error_deg": data.get("angle_error_deg"),
        },
    }


def axis_targets_from_rotation(rotation_robot_from_tcp: np.ndarray, tool_axes: dict[str, Any]) -> dict[str, list[float]]:
    rotation = np.asarray(rotation_robot_from_tcp, dtype=float).reshape(3, 3)
    primary = np.asarray(tool_axes["primary_axis_local"], dtype=float).reshape(3)
    secondary = np.asarray(tool_axes["secondary_axis_local"], dtype=float).reshape(3)
    return {
        "primary_axis_robot": (rotation @ primary).astype(float).tolist(),
        "secondary_axis_robot": (rotation @ secondary).astype(float).tolist(),
    }


def serializable_tool_axes(tool_axes: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": tool_axes["mode"],
        "source": tool_axes["source"],
        "primary_axis_local": np.asarray(tool_axes["primary_axis_local"], dtype=float).reshape(3).tolist(),
        "secondary_axis_local": np.asarray(tool_axes["secondary_axis_local"], dtype=float).reshape(3).tolist(),
        "edge_fit": tool_axes.get("edge_fit"),
    }


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_optional_float_list(text: str | None) -> tuple[float, ...] | None:
    if text is None or not str(text).strip():
        return None
    return parse_float_list(str(text))


def selected_target_anchor_contract(args: argparse.Namespace) -> dict[str, Any]:
    values = parse_optional_float_list(getattr(args, "target_extension_values_m", None))
    if values is None:
        values = (float(getattr(args, "target_extension_m", 0.0)),)
    maximum_abs_extension_m = max(abs(float(value)) for value in values)
    required = bool(getattr(args, "require_selected_point_cut_anchor", False))
    ok = maximum_abs_extension_m <= 1e-9
    diagnostic = {
        "required": required,
        "ok": bool(ok or not required),
        "target_extension_values_m": [float(value) for value in values],
        "maximum_abs_extension_m": maximum_abs_extension_m,
        "invariant": "selected_3d_point_is_nominal_cut_anchor",
    }
    if required and not ok:
        raise ValueError(
            "selected-point cut-anchor contract violated: target extension would move the cut "
            f"by up to {maximum_abs_extension_m:.6f} m; use zero extension for interactive any-target trials"
        )
    return diagnostic


def deleaf_phase_requires_orientation(name: str) -> bool:
    return name in {"pre_cut", "cut"}


def load_deleaf_config_defaults(config_path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", payload)
    if not isinstance(defaults, dict):
        raise ValueError(f"deleaf config defaults must be an object: {path}")
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    loaded: dict[str, Any] = {}
    for key, value in defaults.items():
        if key not in actions:
            raise ValueError(f"unknown deleaf config key {key!r} in {path}")
        action = actions[key]
        if action.type is Path and value is not None:
            path_value = Path(value)
            loaded[key] = path_value if path_value.is_absolute() else REPO_ROOT / path_value
        else:
            loaded[key] = value
    return loaded


def horizontal_unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=float).reshape(3).copy()
    value[2] = 0.0
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        return None
    return value / norm


def constrain_axis_to_downward(axis: np.ndarray, max_tilt_deg: float) -> np.ndarray:
    value = normalize(np.asarray(axis, dtype=float).reshape(3))
    if float(np.dot(value, WORLD_DOWN)) < 0.0:
        value = -value
    max_tilt_rad = float(np.deg2rad(max(0.0, max_tilt_deg)))
    if max_tilt_rad >= np.pi / 2.0:
        return value
    lateral = value.copy()
    lateral[2] = 0.0
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-9:
        return WORLD_DOWN.copy()
    lateral_unit = lateral / lateral_norm
    current_tilt = float(np.arctan2(lateral_norm, abs(float(value[2]))))
    if current_tilt <= max_tilt_rad:
        return value
    return normalize(WORLD_DOWN + lateral_unit * np.tan(max_tilt_rad))


def select_stroke_axis(
    *,
    guidance_axis: np.ndarray,
    prior: dict[str, Any] | None,
    policy: str,
    max_tilt_deg: float,
) -> tuple[np.ndarray, str]:
    if policy == "mps":
        return normalize(guidance_axis), "mps"
    if policy == "world_down":
        return WORLD_DOWN.copy(), "world_down_task_prior"
    if policy == "teleop_prior":
        if prior is None:
            return WORLD_DOWN.copy(), "world_down_no_teleop_prior"
        return normalize(np.asarray(prior["cut_axis_robot"], dtype=float).reshape(3)), "teleop_cut_axis_task_prior"
    if policy == "downward_constrained":
        return constrain_axis_to_downward(guidance_axis, max_tilt_deg), "mps_downward_constrained"
    raise ValueError(f"unknown stroke axis policy: {policy}")


def transform_points_to_robot(points_realsense_m: np.ndarray, transform_json: Path) -> np.ndarray:
    data = json.loads(Path(transform_json).read_text(encoding="utf-8"))
    transform = np.asarray(data["T_robot_from_realsense"], dtype=float).reshape(4, 4)
    points = np.asarray(points_realsense_m, dtype=float).reshape(-1, 3)
    homogeneous = np.ones((points.shape[0], 4), dtype=float)
    homogeneous[:, :3] = points
    return (homogeneous @ transform.T)[:, :3]


def load_capture_points_robot(capture_paths: dict[str, str] | None, transform_json: Path) -> np.ndarray | None:
    if not capture_paths or "npz" not in capture_paths:
        return None
    with np.load(capture_paths["npz"]) as data:
        points_realsense = np.asarray(data["points"], dtype=float).reshape(-1, 3)
    return transform_points_to_robot(points_realsense, transform_json)


def infer_pointcloud_safe_axis(
    *,
    points_robot_m: np.ndarray,
    cut_point_robot_m: np.ndarray,
    radius_m: float,
    z_window_m: float,
    exclusion_m: float,
) -> dict[str, Any] | None:
    points = np.asarray(points_robot_m, dtype=float).reshape(-1, 3)
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    delta = points - cut_point
    horizontal_distance = np.linalg.norm(delta[:, :2], axis=1)
    selected = (
        (horizontal_distance <= float(radius_m))
        & (horizontal_distance >= float(exclusion_m))
        & (np.abs(delta[:, 2]) <= float(z_window_m))
    )
    local = points[selected]
    if local.shape[0] < 20:
        return None
    occupied_center = np.median(local, axis=0)
    safe_axis = horizontal_unit(cut_point - occupied_center)
    if safe_axis is None:
        return None
    return {
        "safe_axis_robot": safe_axis.tolist(),
        "occupied_center_robot_m": occupied_center.astype(float).tolist(),
        "local_point_count": int(local.shape[0]),
        "radius_m": float(radius_m),
        "z_window_m": float(z_window_m),
        "exclusion_m": float(exclusion_m),
    }


def rotate_horizontal(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = horizontal_unit(axis)
    if unit is None:
        raise ValueError("cannot rotate a near-vertical horizontal axis")
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([c * unit[0] - s * unit[1], s * unit[0] + c * unit[1], 0.0], dtype=float)


def _shaft_at_q(
    kinematics: RBY1Kinematics,
    locked_state: Any,
    q_arm_rad: np.ndarray,
    tool_off: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Real knife shaft (ee_right origin, calibrated tip) from a SOLVED arm config via FK.

    Same shaft model as before (origin = tip - R @ tool_offset, valid because the tool offset is
    translation-only so the TCP frame shares ee_right's rotation), but evaluated on the FK of the
    joints the robot will actually command — not the nominal IK TARGET pose. This closes the gap
    where a converged-but-slightly-off solve (within orientation_tolerance) made the gate certify a
    shaft the arm never holds.
    """
    q_full = kinematics.q_full_from_arm(locked_state, np.asarray(q_arm_rad, dtype=float))
    tcp = np.asarray(kinematics.tcp_pose(q_full), dtype=float)
    tip = tcp[:3, 3]
    origin = tip - tcp[:3, :3] @ np.asarray(tool_off, dtype=float).reshape(3)
    return origin, tip


def _swept_shaft_samples(
    kinematics: RBY1Kinematics,
    locked_state: Any,
    solved: list[dict[str, Any]],
    tool_off: np.ndarray,
    samples_per_segment: int,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(origin, tip, label) shaft snapshots along the JOINT-SPACE interpolation between consecutive
    solved waypoints — the curve the robot actually executes (a JointPositionCommand interpolates
    each joint linearly), not a task-space lerp of target poses. Each snapshot is FK of an
    intermediate joint config, so the swept shaft the gate measures is the swept shaft that moves.
    """
    qs = [np.asarray(w["q_arm_rad"], dtype=float) for w in solved if w.get("q_arm_rad") is not None]
    names = [str(w.get("name", "")) for w in solved if w.get("q_arm_rad") is not None]
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    if not qs:
        return out
    if len(qs) == 1:
        origin, tip = _shaft_at_q(kinematics, locked_state, qs[0], tool_off)
        out.append((origin, tip, names[0]))
        return out
    base_steps = max(2, int(samples_per_segment))
    for i in range(len(qs) - 1):
        label = f"{names[i]}->{names[i + 1]}"
        # adaptive density: joint-space lerp does NOT trace a straight Cartesian line, so a large
        # per-phase reorientation (follow-mode lays the blade back 24-85 deg between phases) can bow
        # the shaft out mid-segment past an obstacle that would fall between sparse samples. Scale
        # the sample count with the joint excursion (~one sample per 0.03 rad = 1.7 deg), capped, so
        # the swept shaft is never under-read (the over-optimistic / unsafe direction for a gate).
        max_dq = float(np.max(np.abs(qs[i + 1] - qs[i])))
        steps = int(min(48, max(base_steps, int(np.ceil(max_dq / 0.03)))))
        for k in range(steps + 1):
            alpha = float(k) / float(steps)
            q = (1.0 - alpha) * qs[i] + alpha * qs[i + 1]
            origin, tip = _shaft_at_q(kinematics, locked_state, q, tool_off)
            out.append((origin, tip, label))
    return out


def pointcloud_path_clearance(
    *,
    solved: list[dict[str, Any]],
    kinematics: RBY1Kinematics,
    locked_state: Any,
    points_robot_m: np.ndarray | None,
    cut_point_robot_m: np.ndarray,
    tool_offset_local_m: np.ndarray,
    samples_per_segment: int,
    exclusion_m: float,
) -> dict[str, Any] | None:
    """Min distance from the cloud to the knife TIP along the FK joint-space path of the solved q."""
    if points_robot_m is None:
        return None
    points = np.asarray(points_robot_m, dtype=float).reshape(-1, 3)
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    tool_off = np.asarray(tool_offset_local_m, dtype=float).reshape(3)
    obstacle_points = points[np.linalg.norm(points - cut_point, axis=1) >= float(exclusion_m)]
    samples = _swept_shaft_samples(kinematics, locked_state, solved, tool_off, samples_per_segment)
    if obstacle_points.size == 0 or not samples:
        return None
    min_distance = float("inf")
    worst_segment = None
    for _origin, tip, label in samples:
        distance = float(np.min(np.linalg.norm(obstacle_points - tip, axis=1)))
        if distance < min_distance:
            min_distance = distance
            worst_segment = label
    return {
        "minimum_distance_m": min_distance,
        "worst_segment": worst_segment,
        "exclusion_m": float(exclusion_m),
        "samples_per_segment": int(samples_per_segment),
        "obstacle_point_count": int(obstacle_points.shape[0]),
        "model": "tip_path_fk_jointspace",
    }


def pointcloud_blade_clearance(
    *,
    solved: list[dict[str, Any]],
    kinematics: RBY1Kinematics,
    locked_state: Any,
    points_robot_m: np.ndarray | None,
    cut_point_robot_m: np.ndarray,
    tool_offset_local_m: np.ndarray,
    samples_per_segment: int,
    exclusion_m: float,
) -> dict[str, Any] | None:
    """Min distance from the point cloud to the swept knife SHAFT (ee_right->tip), not just the tip.

    Evaluated on the FK of the SOLVED joints, sampled along the JOINT-SPACE interpolation between
    consecutive waypoints — i.e. the real executed swept shaft, so a rod brushing the blade body or
    wrist along the actual motion is caught (the prior version lerped TARGET poses in task space,
    which neither matched the executed joint path nor the solved orientation).
    """
    if points_robot_m is None:
        return None
    points = np.asarray(points_robot_m, dtype=float).reshape(-1, 3)
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    tool_off = np.asarray(tool_offset_local_m, dtype=float).reshape(3)
    obstacle_points = points[np.linalg.norm(points - cut_point, axis=1) >= float(exclusion_m)]
    samples = _swept_shaft_samples(kinematics, locked_state, solved, tool_off, samples_per_segment)
    if obstacle_points.size == 0 or not samples:
        return None

    def segment_min(origin: np.ndarray, tip: np.ndarray) -> tuple[float, np.ndarray]:
        ab = tip - origin
        length_sq = float(ab @ ab)
        if length_sq < 1e-12:
            distances = np.linalg.norm(obstacle_points - origin, axis=1)
        else:
            alpha = np.clip((obstacle_points - origin) @ ab / length_sq, 0.0, 1.0)
            proj = origin[None, :] + alpha[:, None] * ab[None, :]
            distances = np.linalg.norm(obstacle_points - proj, axis=1)
        index = int(np.argmin(distances))
        return float(distances[index]), obstacle_points[index]

    min_distance = float("inf")
    worst_segment = None
    nearest_obstacle: np.ndarray | None = None
    for origin, tip, label in samples:
        distance, nearest = segment_min(origin, tip)
        if distance < min_distance:
            min_distance = distance
            worst_segment = label
            nearest_obstacle = nearest
    return {
        "minimum_distance_m": min_distance,
        "worst_segment": worst_segment,
        "nearest_obstacle_robot_m": None if nearest_obstacle is None else [float(v) for v in nearest_obstacle],
        "exclusion_m": float(exclusion_m),
        "samples_per_segment": int(samples_per_segment),
        "obstacle_point_count": int(obstacle_points.shape[0]),
        "model": "shaft_segment_ee_right_to_tip_fk_jointspace",
    }


def load_stakes(path: Path | None) -> list[dict[str, Any]]:
    """Load fixed support-stake obstacles (vertical cylinders) in robot frame from JSON."""
    if path is None:
        return []
    resolved = Path(path)
    if not resolved.exists():
        return []
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return list(data.get("stakes", []))


def _segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Closest Euclidean distance between segment p1-q1 and segment p2-q2 (clamped)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    closest1 = p1 + d1 * s
    closest2 = p2 + d2 * t
    return float(np.linalg.norm(closest1 - closest2))


def stake_clearance(
    *,
    solved: list[dict[str, Any]],
    kinematics: RBY1Kinematics,
    locked_state: Any,
    stakes: list[dict[str, Any]],
    tool_offset_local_m: np.ndarray,
    samples_per_segment: int,
    blade_clearance_radius_m: float = 0.0,
) -> dict[str, Any] | None:
    """Deterministic min clearance from the swept knife SHAFT to fixed stake cylinders.

    Independent of the point cloud (stakes are known fixed geometry), so it reliably catches
    the thin support stakes the cloud-based gates sample unreliably. Each stake is a vertical
    segment [x,y,z_min]->[x,y,z_max] with a radius; clearance subtracts the radius. The shaft is
    the FK of the SOLVED joints sampled along the JOINT-SPACE interpolation (the executed sweep).
    """
    if not stakes:
        return None
    tool_off = np.asarray(tool_offset_local_m, dtype=float).reshape(3)
    samples = _swept_shaft_samples(kinematics, locked_state, solved, tool_off, samples_per_segment)
    if not samples:
        return None
    stake_segments = []
    for stake in stakes:
        if stake.get("base") is not None and stake.get("top") is not None:
            # explicit (possibly tilted) segment, e.g. from touch-measurement
            base = np.asarray(stake["base"], dtype=float).reshape(3)
            top = np.asarray(stake["top"], dtype=float).reshape(3)
        else:
            base = np.asarray([float(stake["x"]), float(stake["y"]), float(stake["z_min"])], dtype=float)
            top = np.asarray([float(stake["x"]), float(stake["y"]), float(stake["z_max"])], dtype=float)
        stake_segments.append((base, top, float(stake.get("radius_m", 0.02)), str(stake.get("name", "stake"))))

    min_distance = float("inf")
    worst_segment = None
    worst_stake = None

    blade_radius = float(blade_clearance_radius_m)

    for origin, tip, label in samples:
        for base, top, radius, name in stake_segments:
            # subtract the stake radius AND the blade body half-width so the reported clearance
            # is BLADE-SURFACE to STAKE-SURFACE (the knife is ~5cm wide, not a centerline)
            distance = _segment_segment_distance(origin, tip, base, top) - radius - blade_radius
            if distance < min_distance:
                min_distance = distance
                worst_segment = label
                worst_stake = name
    return {
        "minimum_distance_m": min_distance,
        "worst_segment": worst_segment,
        "worst_stake": worst_stake,
        "stake_count": len(stake_segments),
        "blade_clearance_radius_m": blade_radius,
        "model": "blade_shaft_vs_stake_cylinders_fk_jointspace",
    }


def min_solved_path_clearance(solved: list[dict[str, Any]]) -> float | None:
    values = [
        waypoint.get("path_clearance", {}).get("minimum_distance_m")
        for waypoint in solved
        if waypoint.get("path_clearance", {}).get("minimum_distance_m") is not None
    ]
    if not values:
        return None
    return float(min(values))


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    pointcloud_clearance = candidate.get("pointcloud_path_clearance") or {}
    pointcloud_min = pointcloud_clearance.get("minimum_distance_m")
    blade_clearance = candidate.get("blade_shaft_clearance") or {}
    blade_min = blade_clearance.get("minimum_distance_m")
    stake_clear = candidate.get("stake_clearance") or {}
    stake_min = stake_clear.get("minimum_distance_m")
    self_min = candidate.get("minimum_self_path_clearance_m")
    final_position_error = candidate.get("final_position_error_m")
    extension_delta = candidate.get("target_extension_delta_m")
    contact_advance = candidate.get("contact_advance_m")
    return (
        1.0 if candidate["ok"] else 0.0,
        -1.0 if stake_min is None else float(stake_min),
        -1.0 if blade_min is None else float(blade_min),
        -1.0 if extension_delta is None else -float(extension_delta),
        -1.0 if pointcloud_min is None else float(pointcloud_min),
        -1.0 if self_min is None else float(self_min),
        -1.0 if contact_advance is None else -float(contact_advance),
        -1.0 if final_position_error is None else -float(final_position_error),
    )


def deleaf_candidate_specs(
    *,
    args: argparse.Namespace,
    cut_point_robot_m: np.ndarray,
    capture_paths: dict[str, str] | None,
    transform_json: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, np.ndarray | None]:
    points_robot = load_capture_points_robot(capture_paths, transform_json)
    inference = None
    target_extensions = parse_optional_float_list(getattr(args, "target_extension_values_m", None))
    if target_extensions is None:
        target_extensions = (float(args.target_extension_m),)
    if any(value < 0.0 for value in target_extensions):
        raise ValueError("--target-extension-values-m entries must be non-negative")
    base_axis = parse_vector3(args.contact_advance_axis_robot, name="contact advance axis")
    explicit_base_axis = base_axis is not None
    if args.use_pointcloud_candidates and points_robot is not None:
        inference = infer_pointcloud_safe_axis(
            points_robot_m=points_robot,
            cut_point_robot_m=cut_point_robot_m,
            radius_m=float(args.pointcloud_side_radius_m),
            z_window_m=float(args.pointcloud_side_z_window_m),
            exclusion_m=float(args.pointcloud_cut_exclusion_m),
        )
        if inference is not None and not explicit_base_axis:
            base_axis = np.asarray(inference["safe_axis_robot"], dtype=float)
    if base_axis is None:
        base_axis = parse_vector3(args.contact_advance_axis_robot, name="contact advance axis")
    if base_axis is None:
        return [
            {
                "name": f"single_ext_{float(target_extension_m):.3f}",
                "contact_advance_m": float(args.contact_advance_m),
                "contact_advance_axis_robot": None,
                "target_extension_m": float(target_extension_m),
            }
            for target_extension_m in target_extensions
        ], inference, points_robot
    unit = horizontal_unit(base_axis)
    if unit is None:
        return [
            {
                "name": f"single_ext_{float(target_extension_m):.3f}",
                "contact_advance_m": float(args.contact_advance_m),
                "contact_advance_axis_robot": None,
                "target_extension_m": float(target_extension_m),
            }
            for target_extension_m in target_extensions
        ], inference, points_robot
    if not args.use_pointcloud_candidates:
        return [
            {
                "name": f"axis_override_ext_{float(target_extension_m):.3f}",
                "contact_advance_m": float(args.contact_advance_m),
                "contact_advance_axis_robot": unit.tolist(),
                "target_extension_m": float(target_extension_m),
            }
            for target_extension_m in target_extensions
        ], inference, points_robot
    angle_values = parse_float_list(args.pointcloud_candidate_angles_deg)
    advance_values = parse_float_list(args.pointcloud_contact_advance_values_m)
    candidates: list[dict[str, Any]] = []
    for angle_deg in angle_values:
        axis = rotate_horizontal(unit, np.deg2rad(float(angle_deg)))
        for advance_m in advance_values:
            for target_extension_m in target_extensions:
                candidates.append(
                    {
                        "name": (
                            f"pc_axis_{float(angle_deg):+.0f}"
                            f"_advance_{float(advance_m):.3f}"
                            f"_ext_{float(target_extension_m):.3f}"
                        ),
                        "contact_advance_m": float(advance_m),
                        "contact_advance_axis_robot": axis.tolist(),
                        "target_extension_m": float(target_extension_m),
                    }
                )
    return candidates, inference, points_robot


def load_teleop_prior(
    path: Path | None,
    action_name: str,
    source_match: str | None,
    *,
    library_dir: Path | None = None,
    cut_point_robot_m: np.ndarray | None = None,
) -> dict[str, Any] | None:
    prior_records: list[tuple[dict[str, Any], str]] = []

    def _append_file(prior_path: Path) -> None:
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
        entries = list(payload.get("priors", []))
        if not entries and "waypoints" in payload:
            entries = [payload]
        for prior in entries:
            prior_records.append((prior, str(prior_path)))

    if path is not None:
        _append_file(Path(path))
    if library_dir is not None:
        root = Path(library_dir)
        if not root.is_dir():
            raise ValueError(f"teleop prior library does not exist: {root}")
        for prior_path in sorted(root.glob("*_teleop_prior.json")):
            if "_UN" in prior_path.stem:
                continue
            if path is not None and prior_path.resolve() == Path(path).resolve():
                continue
            _append_file(prior_path)
    if not prior_records:
        if path is None and library_dir is None:
            return None
        raise ValueError("teleop prior sources contain no priors")

    deduplicated: list[tuple[dict[str, Any], str]] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for prior, prior_path in prior_records:
        anchor = tuple(float(value) for value in prior.get("cut_anchor_robot_m", []))
        key = (str(prior.get("source", prior_path)), anchor)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((prior, prior_path))
    priors = [record[0] for record in deduplicated]
    if source_match is not None:
        matches = [
            (prior, prior_path)
            for prior, prior_path in deduplicated
            if source_match in Path(str(prior.get("source", prior_path))).name
            or source_match in Path(prior_path).name
        ]
        if not matches:
            raise ValueError(f"no teleop prior source contains {source_match!r}")
        chosen, chosen_path = matches[0]
        result = dict(chosen)
        result["selection"] = {
            "policy": "explicit_source_match",
            "source_match": source_match,
            "prior_file": chosen_path,
            "source": chosen.get("source"),
        }
        return result
    if cut_point_robot_m is not None:
        target = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
        ranked: list[tuple[float, dict[str, Any], str]] = []
        for prior, prior_path in deduplicated:
            anchor = prior.get("cut_anchor_robot_m")
            if anchor is None:
                continue
            distance = float(np.linalg.norm(np.asarray(anchor, dtype=float).reshape(3) - target))
            ranked.append((distance, prior, prior_path))
        if ranked:
            ranked.sort(key=lambda item: (item[0], str(item[1].get("source", item[2]))))
            distance, chosen, chosen_path = ranked[0]
            result = dict(chosen)
            result["selection"] = {
                "policy": "nearest_target_conditioned_successful_prior",
                "clicked_point_robot_m": target.tolist(),
                "distance_m": distance,
                "prior_file": chosen_path,
                "source": chosen.get("source"),
                "candidate_count": len(ranked),
            }
            return result
    action_digits = "".join(ch for ch in action_name if ch.isdigit())
    if action_digits:
        token = f"deleaf{int(action_digits):02d}"
        matches = [prior for prior in priors if token in Path(prior["source"]).name]
        if matches:
            return matches[0]
    if len(priors) == 1:
        return priors[0]
    raise ValueError(
        "multiple teleop priors available; provide a cut point for target-conditioned selection "
        "or pass --teleop-prior-source-match"
    )


def load_successful_demo_joint_seed_library(
    root: Path | None,
    *,
    cut_point_robot_m: np.ndarray,
    dataset_name: str,
    max_count: int,
) -> dict[str, Any]:
    """Load successful robot postures as IK seeds, never as executable trajectories."""
    if root is None or max_count <= 0:
        return {
            "enabled": False,
            "role": "ik_multistart_seed_only",
            "uses_joint_replay": False,
            "candidates": [],
        }
    library_root = Path(root)
    if not library_root.is_dir():
        raise ValueError(f"successful demonstration library does not exist: {library_root}")
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    records: list[dict[str, Any]] = []
    pattern = "*/success_current/actions/*/action/referenced/deleaf_click_cut/*.json"
    for plan_path in sorted(library_root.glob(pattern)):
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        if str(payload.get("dataset_name")) != str(dataset_name):
            continue
        if not bool((payload.get("hard_gate") or {}).get("ok")):
            continue
        if not bool((payload.get("motion") or {}).get("executed")):
            continue
        selected_point = (payload.get("selected_point") or {}).get("point_robot_m")
        if selected_point is None:
            continue
        anchor = np.asarray(selected_point, dtype=float).reshape(3)
        q_seed_by_name = {
            str(waypoint["name"]): [float(value) for value in waypoint["q_arm_rad"]]
            for waypoint in payload.get("waypoints", [])
            if bool(waypoint.get("ok")) and len(waypoint.get("q_arm_rad", [])) == 7
        }
        if not q_seed_by_name:
            continue
        label = plan_path.relative_to(library_root).parts[0]
        records.append(
            {
                "label": label,
                "source_plan": str(plan_path),
                "anchor_robot_m": anchor.tolist(),
                "anchor_distance_m": float(np.linalg.norm(anchor - cut_point)),
                "q_seed_by_name": q_seed_by_name,
            }
        )
    records.sort(key=lambda item: (float(item["anchor_distance_m"]), str(item["label"])))
    selected = records[: int(max_count)]
    return {
        "enabled": bool(selected),
        "root": str(library_root),
        "role": "target_conditioned_ik_multistart_seed_only",
        "selection_policy": "nearest_successful_postures_then_fresh_ik",
        "uses_joint_replay": False,
        "available_count": len(records),
        "max_count": int(max_count),
        "candidates": selected,
    }


def load_action_phase_windows(manifest_path: Path, action_name: str) -> tuple[dict[str, Any], tuple[PhaseWindow, ...]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actions = {str(action["name"]): action for action in manifest["actions"]}
    if action_name not in actions:
        raise ValueError(f"action {action_name!r} is not present in {manifest_path}")
    action = actions[action_name]
    windows = tuple(
        PhaseWindow(name=name, start_frame_idx=int(action[name][0]), end_frame_idx=int(action[name][1]))
        for name in ("approach", "cut", "retract")
    )
    return manifest, windows


def extract_deleaf_mps_guidance(
    *,
    manifest_path: Path,
    action_name: str,
    frame_index_path: Path | None,
    hand_tracking_path: Path | None,
    slam_trajectory_path: Path,
    alignment_path: Path,
    hand_side: str,
    hand_point: HandPoint,
    min_confidence: float,
    prefer_downward_cut_axis: bool,
    approach_path_points: int = 4,
) -> DeleafMpsGuidance:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    units = str(manifest.get("window_units", "frames"))
    hand_tracking = hand_tracking_path or Path(manifest["mps_hand_tracking_csv"])
    # The robot-frame mapper still needs frame timestamps to resolve the alignment
    # anchor frame, regardless of how the phase windows are keyed.
    frame_index = frame_index_path or (
        Path(manifest["timestamps_csv"]).with_name("frame_index.csv") if "timestamps_csv" in manifest else None
    )
    frame_timestamps_us = load_frame_timestamps(frame_index) if frame_index is not None else None
    if units == "timestamps_us":
        action = {str(a["name"]): a for a in manifest["actions"]}[action_name]
        time_windows = tuple(
            TimeWindow(name=name, start_us=float(action[name][0]), end_us=float(action[name][1]))
            for name in ("approach", "cut", "retract")
        )
        phase_windows = time_windows
    else:
        _, phase_windows = load_action_phase_windows(manifest_path, action_name)
        time_windows = phase_time_windows(phase_windows, frame_timestamps_us)
    samples = load_hand_tracking(hand_tracking)
    mapper = build_robot_frame_mapper(alignment_path, frame_timestamps_us, load_slam_trajectory(slam_trajectory_path))
    selected_side = choose_hand_side(samples, time_windows, hand_side, min_confidence)
    approach_axis, approach_confidence, approach_motion_m, approach_count = phase_motion_axis(
        samples, time_windows, "approach", selected_side, hand_point, mapper, min_confidence
    )
    cut_axis, cut_confidence, cut_motion_m, cut_count = phase_motion_axis(
        samples, time_windows, "cut", selected_side, hand_point, mapper, min_confidence
    )
    retract_axis, retract_confidence, retract_motion_m, retract_count = phase_motion_axis(
        samples, time_windows, "retract", selected_side, hand_point, mapper, min_confidence
    )
    if prefer_downward_cut_axis and float(np.dot(cut_axis, WORLD_DOWN)) < 0.0:
        cut_axis = -cut_axis
    confidence = float(np.mean([approach_confidence, cut_confidence, retract_confidence]))
    # Demonstrated cut location in robot frame: median hand position during the cut
    # window (falls back to the approach window if the cut window is too short/low
    # confidence). Used only to MATCH a clicked petiole to its demonstrated cut --
    # never as the actual target (the live click stays the cut location).
    cut_anchor, cut_anchor_count = phase_mean_position(
        samples, time_windows, "cut", selected_side, hand_point, mapper, min_confidence
    )
    cut_anchor_source = "cut_window_median"
    if cut_anchor is None:
        cut_anchor, cut_anchor_count = phase_mean_position(
            samples, time_windows, "approach", selected_side, hand_point, mapper, min_confidence
        )
        cut_anchor_source = "approach_window_median_fallback"
    # SEED-ONLY demonstrated approach path: the human's robot-frame hand polyline over
    # the approach window, evenly resampled. Used to warm-start IK so the solver lands in
    # a human-like joint basin. NEVER an executed waypoint -- gates/timing are unchanged.
    approach_path = phase_polyline(
        samples, time_windows, "approach", selected_side, hand_point, mapper, min_confidence, int(approach_path_points)
    )
    return DeleafMpsGuidance(
        action_name=action_name,
        hand_side=selected_side,
        approach_axis_robot=approach_axis,
        cut_axis_robot=cut_axis,
        retract_axis_robot=retract_axis,
        confidence=confidence,
        cut_anchor_robot_m=cut_anchor,
        approach_path_robot_m=approach_path,
        diagnostics={
            "cut_anchor_robot_m": None if cut_anchor is None else np.asarray(cut_anchor, dtype=float).reshape(3).tolist(),
            "cut_anchor_source": cut_anchor_source,
            "cut_anchor_samples": int(cut_anchor_count),
            "approach_path_robot_m": None if approach_path is None else np.asarray(approach_path, dtype=float).reshape(-1, 3).tolist(),
            "approach_path_points": int(approach_path_points),
            "approach_path_count": 0 if approach_path is None else int(np.asarray(approach_path).reshape(-1, 3).shape[0]),
            "manifest": str(manifest_path),
            "frame_index": str(frame_index),
            "hand_tracking": str(hand_tracking),
            "slam_trajectory": str(slam_trajectory_path),
            "alignment": str(alignment_path),
            "hand_side_requested": hand_side,
            "hand_side_selected": selected_side,
            "hand_point": hand_point,
            "min_confidence": float(min_confidence),
            "phase_windows": [window.__dict__ for window in phase_windows],
            "time_windows": [window.__dict__ for window in time_windows],
            "approach_samples": approach_count,
            "cut_samples": cut_count,
            "retract_samples": retract_count,
            "approach_motion_m": approach_motion_m,
            "cut_motion_m": cut_motion_m,
            "retract_motion_m": retract_motion_m,
            "approach_confidence": approach_confidence,
            "cut_confidence": cut_confidence,
            "retract_confidence": retract_confidence,
        },
    )


def list_manifest_actions(manifest_path: Path) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [str(action["name"]) for action in manifest["actions"]]


def build_deleaf_guidance_library(
    *,
    manifest_path: Path,
    frame_index_path: Path | None,
    hand_tracking_path: Path | None,
    slam_trajectory_path: Path,
    alignment_path: Path,
    hand_side: str,
    hand_point: HandPoint,
    min_confidence: float,
    prefer_downward_cut_axis: bool,
    approach_path_points: int = 4,
) -> list[DeleafMpsGuidance]:
    """Extract per-cut MPS guidance for EVERY action in the manifest.

    Each action keeps its own auto-selected hand side and its demonstrated cut anchor,
    so a clicked petiole can inherit the axes of the human cut nearest to it.
    """
    return [
        extract_deleaf_mps_guidance(
            manifest_path=manifest_path,
            action_name=action_name,
            frame_index_path=frame_index_path,
            hand_tracking_path=hand_tracking_path,
            slam_trajectory_path=slam_trajectory_path,
            alignment_path=alignment_path,
            hand_side=hand_side,
            hand_point=hand_point,
            min_confidence=min_confidence,
            prefer_downward_cut_axis=prefer_downward_cut_axis,
            approach_path_points=approach_path_points,
        )
        for action_name in list_manifest_actions(manifest_path)
    ]


def select_guidance_for_point(
    library: list[DeleafMpsGuidance],
    cut_point_robot_m: np.ndarray,
) -> tuple[DeleafMpsGuidance, dict[str, Any]]:
    """Pick the guidance whose demonstrated cut anchor is nearest the clicked point."""
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    ranked: list[tuple[float, DeleafMpsGuidance]] = []
    for guidance in library:
        if guidance.cut_anchor_robot_m is None:
            continue
        anchor = np.asarray(guidance.cut_anchor_robot_m, dtype=float).reshape(3)
        ranked.append((float(np.linalg.norm(anchor - cut_point)), guidance))
    if not ranked:
        raise ValueError("no manifest action has a usable cut anchor for nearest-cut selection")
    ranked.sort(key=lambda item: item[0])
    chosen_distance, chosen = ranked[0]
    selection = {
        "policy": "nearest_demonstrated_cut",
        "selected_action": chosen.action_name,
        "selected_distance_m": chosen_distance,
        "clicked_point_robot_m": cut_point.tolist(),
        "ranking": [
            {
                "action": guidance.action_name,
                "distance_m": distance,
                "hand_side": guidance.hand_side,
                "cut_anchor_robot_m": np.asarray(guidance.cut_anchor_robot_m, dtype=float).reshape(3).tolist(),
            }
            for distance, guidance in ranked
        ],
    }
    return chosen, selection


def choose_hand_side(
    samples: list[HandTrackingSample],
    time_windows: tuple[TimeWindow, ...],
    requested: str,
    min_confidence: float,
) -> HandSide:
    if requested in ("left", "right"):
        return requested
    scores = {
        side: sum(
            1
            for sample in samples
            if confidence(sample, side) >= float(min_confidence)
            and any(window.start_us <= sample.timestamp_us <= window.end_us for window in time_windows)
        )
        for side in ("left", "right")
    }
    return "left" if scores["left"] >= scores["right"] else "right"


def phase_motion_axis(
    samples: list[HandTrackingSample],
    time_windows: tuple[TimeWindow, ...],
    phase_name: str,
    side: HandSide,
    hand_point: HandPoint,
    mapper: RobotFrameMapper,
    min_confidence: float,
) -> tuple[np.ndarray, float, float, int]:
    window = next(window for window in time_windows if window.name == phase_name)
    selected = [
        sample
        for sample in samples
        if window.start_us <= sample.timestamp_us <= window.end_us and confidence(sample, side) >= float(min_confidence)
    ]
    if len(selected) < 2:
        return np.asarray([1.0, 0.0, 0.0], dtype=float), 0.0, 0.0, len(selected)
    ordered = sorted(selected, key=lambda sample: sample.timestamp_us)
    points = np.vstack(
        [
            mapper.point_to_robot(sample.timestamp_us, hand_point_device(sample.row, side, hand_point))
            for sample in ordered
        ]
    )
    split = max(1, len(points) // 3)
    start = np.median(points[:split], axis=0)
    end = np.median(points[-split:], axis=0)
    motion = end - start
    motion_m = float(np.linalg.norm(motion))
    if motion_m <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=float), 0.0, 0.0, len(selected)
    confidence_value = float(np.clip(min(1.0, motion_m / 0.03) * min(1.0, len(selected) / 6.0), 0.0, 1.0))
    return motion / motion_m, confidence_value, motion_m, len(selected)


def phase_mean_position(
    samples: list[HandTrackingSample],
    time_windows: tuple[TimeWindow, ...],
    phase_name: str,
    side: HandSide,
    hand_point: HandPoint,
    mapper: RobotFrameMapper,
    min_confidence: float,
) -> tuple[np.ndarray | None, int]:
    """Median robot-frame hand position over a phase window (the demonstrated phase anchor).

    Returns (point_robot_m, sample_count); point is None when no confident samples fall
    in the window. Mirrors phase_motion_axis's selection so the anchor and the axes are
    derived from the same hand-tracking samples and alignment.
    """
    window = next(window for window in time_windows if window.name == phase_name)
    selected = [
        sample
        for sample in samples
        if window.start_us <= sample.timestamp_us <= window.end_us and confidence(sample, side) >= float(min_confidence)
    ]
    if not selected:
        return None, 0
    points = np.vstack(
        [
            mapper.point_to_robot(sample.timestamp_us, hand_point_device(sample.row, side, hand_point))
            for sample in selected
        ]
    )
    return np.median(points, axis=0), len(selected)


def phase_polyline(
    samples: list[HandTrackingSample],
    time_windows: tuple[TimeWindow, ...],
    phase_name: str,
    side: HandSide,
    hand_point: HandPoint,
    mapper: RobotFrameMapper,
    min_confidence: float,
    num_points: int,
) -> np.ndarray | None:
    """Robot-frame polyline of the demonstrated hand path over a phase window.

    Selects the SAME confident in-window samples as phase_motion_axis (same window,
    same confidence gate, ordered by timestamp), maps each to robot frame via
    mapper.point_to_robot(hand_point_device(...)), then evenly resamples num_points
    positions along the ordered path by index. Returns None when there are too few
    samples (< 2) or num_points < 1. Used SEED-ONLY to warm-start IK along the human
    approach -- never as executed waypoints.
    """
    if int(num_points) < 1:
        return None
    window = next(window for window in time_windows if window.name == phase_name)
    selected = [
        sample
        for sample in samples
        if window.start_us <= sample.timestamp_us <= window.end_us and confidence(sample, side) >= float(min_confidence)
    ]
    if len(selected) < 2:
        return None
    ordered = sorted(selected, key=lambda sample: sample.timestamp_us)
    points = np.vstack(
        [
            mapper.point_to_robot(sample.timestamp_us, hand_point_device(sample.row, side, hand_point))
            for sample in ordered
        ]
    )
    # Even resample by index along the ordered path. With one requested point we keep
    # the final (closest-to-cut) position; otherwise we span first..last inclusive.
    if int(num_points) == 1:
        indices = np.asarray([points.shape[0] - 1], dtype=float)
    else:
        indices = np.linspace(0.0, float(points.shape[0] - 1), int(num_points))
    resampled = np.vstack(
        [
            (1.0 - (index - np.floor(index))) * points[int(np.floor(index))]
            + (index - np.floor(index)) * points[int(np.ceil(index))]
            for index in indices
        ]
    )
    return resampled


def hand_point_device(row: dict[str, str], side: HandSide, hand_point: HandPoint) -> np.ndarray:
    if hand_point == "wrist":
        return np.asarray(
            [
                float(row[f"tx_{side}_device_wrist"]),
                float(row[f"ty_{side}_device_wrist"]),
                float(row[f"tz_{side}_device_wrist"]),
            ],
            dtype=float,
        )
    thumb = landmark_device(row, side, 4)
    index = landmark_device(row, side, 8)
    return 0.5 * (thumb + index)


def landmark_device(row: dict[str, str], side: HandSide, index: int) -> np.ndarray:
    return np.asarray(
        [
            float(row[f"tx_{side}_landmark_{index}_device"]),
            float(row[f"ty_{side}_landmark_{index}_device"]),
            float(row[f"tz_{side}_landmark_{index}_device"]),
        ],
        dtype=float,
    )


def confidence(sample: HandTrackingSample, side: HandSide) -> float:
    return sample.left_confidence if side == "left" else sample.right_confidence


def post_selection_home_command(args: argparse.Namespace) -> list[str] | None:
    home_path = getattr(args, "post_selection_home", None)
    if home_path is None:
        return None
    home_path = Path(home_path)
    minimum_time_s = float(getattr(args, "post_selection_home_minimum_time_s", 10.0))
    return [
        sys.executable,
        str(REPO_ROOT / "robot" / "utils" / "restore_home_state.py"),
        "--home",
        str(home_path),
        "--execute",
        "--scope",
        "right_arm",
        "--minimum-time",
        str(minimum_time_s),
    ]


def restore_right_arm_after_selection(args: argparse.Namespace) -> dict[str, Any] | None:
    command = post_selection_home_command(args)
    if command is None:
        return None
    home_path = Path(command[command.index("--home") + 1])
    minimum_time_s = float(command[command.index("--minimum-time") + 1])
    result: dict[str, Any] = {
        "home": str(home_path),
        "scope": "right_arm",
        "minimum_time_s": minimum_time_s,
        "executed": False,
    }
    if not bool(args.execute):
        result["reason"] = "dry_run"
        return result
    print("$ " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to restore the right arm to its execution posture after target selection: "
            + subprocess.list2cmdline(command)
        )
    result["executed"] = True
    return result


def select_cut_point(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str] | None, bool | None]:
    transform_json = args.transform_json or default_transform_json(args.dataset_name)
    selected_robot = parse_vector3(args.selected_point_robot_m, name="selected robot point")
    if selected_robot is not None:
        selected = {"selection_source": "cli_robot_point", "point_robot_m": selected_robot.tolist()}
        transition = restore_right_arm_after_selection(args)
        if transition is not None:
            selected["post_selection_right_arm_restore"] = transition
        return selected, None, bool(args.execute)
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
    capture_paths = write_capture_artifacts(
        dataset_name=args.dataset_name,
        run_label=args.run_label,
        runs_root=args.runs_root,
        capture=capture,
        point_cloud=point_cloud,
        metadata={
            "method": "deleaf_click_cut_realsense_capture",
            "dataset_name": args.dataset_name,
            "run_label": args.run_label,
            "transform_json": str(transform_json),
        },
    )
    picker_payload = run_external_open3d_picker(
        surfel_python=args.surfel_python,
        capture_npz=Path(capture_paths["npz"]),
        transform_json=transform_json,
        dataset_name=args.dataset_name,
        axis_size_m=args.axis_size_m,
        output_json=artifact_path(
            "deleaf_click_cut",
            f"{args.dataset_name}_{args.run_label}_selection",
            runs_root=args.runs_root,
        ),
        confirm_execute=bool(args.execute and not args.no_execute_key_confirm),
    )
    selected = dict(picker_payload["selected_point"])
    selected["selection_source"] = "external_open3d_capture"
    if "point_robot_m" not in selected:
        selected["point_robot_m"] = point_to_robot(np.asarray(selected["point_realsense_m"], dtype=float), transform_json)
    transition = restore_right_arm_after_selection(args)
    if transition is not None:
        selected["post_selection_right_arm_restore"] = transition
    return selected, capture_paths, picker_payload.get("execute_confirmed")


def deleaf_rotation(*, approach_axis_robot: np.ndarray, cut_axis_robot: np.ndarray) -> np.ndarray:
    z_axis = -normalize(approach_axis_robot)
    y_seed = -normalize(cut_axis_robot)
    y_axis = y_seed - float(np.dot(y_seed, z_axis)) * z_axis
    if float(np.linalg.norm(y_axis)) < 1e-6:
        y_seed = np.asarray([0.0, 0.0, 1.0], dtype=float)
        y_axis = y_seed - float(np.dot(y_seed, z_axis)) * z_axis
    y_axis = normalize(y_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def yaw_about_world_z(degrees: float) -> np.ndarray:
    """Rotation about world +Z by `degrees`. Left-multiplied onto the blade orientation to
    swing the shaft's horizontal heading (e.g. off a stake) while keeping the near-vertical
    descent and the tip target unchanged."""
    theta = float(np.deg2rad(float(degrees)))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return np.asarray([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def limit_approach_standoff_to_workspace(
    *,
    pre_cut_position_robot_m: np.ndarray,
    stroke_axis_robot: np.ndarray,
    requested_standoff_m: float,
    max_z_m: float,
    ceiling_margin_m: float = 0.005,
    minimum_standoff_m: float = 0.005,
) -> tuple[float, dict[str, Any]]:
    """Shorten only the extra approach gap when a downward stroke would exceed max Z."""
    pre_cut = np.asarray(pre_cut_position_robot_m, dtype=float).reshape(3)
    stroke_axis = normalize(np.asarray(stroke_axis_robot, dtype=float).reshape(3))
    requested = float(requested_standoff_m)
    ceiling = float(max_z_m) - float(ceiling_margin_m)
    requested_approach_z = float(pre_cut[2] - stroke_axis[2] * requested)
    diagnostic: dict[str, Any] = {
        "requested_standoff_m": requested,
        "effective_standoff_m": requested,
        "workspace_ceiling_m": ceiling,
        "requested_approach_z_m": requested_approach_z,
        "effective_approach_z_m": requested_approach_z,
        "applied": False,
        "reason": "within_workspace",
    }
    if requested_approach_z <= ceiling + 1e-9:
        return requested, diagnostic

    upward_z_per_standoff = float(-stroke_axis[2])
    available_z = float(ceiling - pre_cut[2])
    if upward_z_per_standoff <= 1e-9 or available_z <= 0.0:
        diagnostic["reason"] = "cannot_limit_without_moving_pre_cut"
        return requested, diagnostic

    maximum_standoff = available_z / upward_z_per_standoff
    if maximum_standoff < float(minimum_standoff_m):
        diagnostic["reason"] = "insufficient_workspace_for_minimum_standoff"
        return requested, diagnostic

    effective = min(requested, maximum_standoff)
    effective_approach_z = float(pre_cut[2] - stroke_axis[2] * effective)
    diagnostic.update(
        {
            "effective_standoff_m": effective,
            "effective_approach_z_m": effective_approach_z,
            "applied": effective < requested - 1e-9,
            "reason": "workspace_ceiling",
        }
    )
    return effective, diagnostic


def build_deleaf_waypoints(
    *,
    cut_point_robot_m: np.ndarray,
    home_tcp_robot_m: np.ndarray,
    guidance: DeleafMpsGuidance,
    teleop_prior: dict[str, Any] | None,
    tool_axes: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[DeleafWaypoint], dict[str, Any]]:
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    home_tcp = np.asarray(home_tcp_robot_m, dtype=float).reshape(3)
    if teleop_prior is not None and args.waypoint_policy == "teleop_offset":
        return build_teleop_prior_waypoints(
            cut_point_robot_m=cut_point,
            home_tcp_robot_m=home_tcp,
            prior=teleop_prior,
            tool_axes=tool_axes,
            args=args,
        )
    if args.waypoint_policy == "mps_template":
        return build_mps_template_waypoints(
            cut_point_robot_m=cut_point,
            home_tcp_robot_m=home_tcp,
            guidance=guidance,
            prior=teleop_prior,
            tool_axes=tool_axes,
            args=args,
        )
    if args.waypoint_policy != "mps_raw":
        raise ValueError(f"unknown waypoint policy: {args.waypoint_policy}")
    fallback_approach = normalize(cut_point - home_tcp)
    approach_axis = np.asarray(guidance.approach_axis_robot, dtype=float).reshape(3)
    if guidance.diagnostics["approach_confidence"] < float(args.min_axis_confidence):
        approach_axis = fallback_approach
    if float(np.dot(approach_axis, fallback_approach)) < 0.0:
        approach_axis = -approach_axis
    cut_axis = np.asarray(guidance.cut_axis_robot, dtype=float).reshape(3)
    if guidance.diagnostics["cut_confidence"] < float(args.min_axis_confidence):
        cut_axis = WORLD_DOWN.copy()
    if args.prefer_downward_cut_axis and float(np.dot(cut_axis, WORLD_DOWN)) < 0.0:
        cut_axis = -cut_axis
    rotation = deleaf_rotation(approach_axis_robot=approach_axis, cut_axis_robot=cut_axis)
    positions = {
        "approach": cut_point - approach_axis * float(args.approach_standoff_m) - cut_axis * float(args.pre_cut_lift_m),
        "pre_cut": cut_point - cut_axis * float(args.pre_cut_lift_m),
        "cut": cut_point + cut_axis * float(args.cut_depth_m),
        "retract": cut_point - approach_axis * float(args.retract_standoff_m) - cut_axis * float(args.retract_lift_m),
    }
    post_cut_lift_m = float(args.post_cut_lift_m)
    if post_cut_lift_m < 0.0:
        raise ValueError("--post-cut-lift-m must be non-negative")
    if post_cut_lift_m > 0.0:
        stroke_axis = normalize(positions["cut"] - positions["pre_cut"])
        positions = {
            "approach": positions["approach"],
            "pre_cut": positions["pre_cut"],
            "cut": positions["cut"],
            "post_cut_lift": positions["cut"] - stroke_axis * post_cut_lift_m,
            "retract": positions["retract"],
        }
    waypoints = [
        DeleafWaypoint(
            name=name,
            target_pose=transform_from_rotation_translation(rotation, position),
            orientation_required=name in {"pre_cut", "cut"},
        )
        for name, position in positions.items()
    ]
    return waypoints, {
        "cut_point_robot_m": cut_point.tolist(),
        "home_tcp_robot_m": home_tcp.tolist(),
        "approach_axis_robot": normalize(approach_axis).tolist(),
        "cut_axis_robot": normalize(cut_axis).tolist(),
        "constrained_cut_axis_robot": (-rotation[:, 1]).tolist(),
        "rotation_robot_from_tcp": rotation.tolist(),
        "tool_axis_local": serializable_tool_axes(tool_axes),
        "axis_targets_by_name": {
            name: axis_targets_from_rotation(rotation, tool_axes)
            for name in positions
        },
        "post_cut_lift_m": post_cut_lift_m,
        "positions_robot_m": {name: value.tolist() for name, value in positions.items()},
    }


def prior_scalar_distance(prior: dict[str, Any] | None, start: str, end: str, fallback: float) -> float:
    if prior is None:
        return float(fallback)
    start_offset = np.asarray(prior["waypoints"][start]["offset_from_cut_anchor_m"], dtype=float).reshape(3)
    end_offset = np.asarray(prior["waypoints"][end]["offset_from_cut_anchor_m"], dtype=float).reshape(3)
    distance = float(np.linalg.norm(end_offset - start_offset))
    return distance if distance > 1e-6 else float(fallback)


def add_safe_entry_and_transits(
    *,
    positions: dict[str, np.ndarray],
    names: tuple[str, ...],
    home_tcp_robot_m: np.ndarray,
    contact_advance: np.ndarray,
    stroke_axis: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    entry_anchor = positions["cut"] + contact_advance
    safe_entry_axis = parse_vector3(getattr(args, "safe_entry_axis_robot", None), name="safe entry axis")
    if safe_entry_axis is not None:
        entry_vector = np.asarray(safe_entry_axis, dtype=float).reshape(3)
        entry_vector[2] = 0.0
        if float(np.linalg.norm(entry_vector)) < 1e-6:
            raise ValueError("--safe-entry-axis-robot must have a nonzero horizontal component")
    else:
        entry_vector = np.asarray(home_tcp_robot_m, dtype=float).reshape(3) - entry_anchor
        entry_vector[2] = 0.0
        if float(np.linalg.norm(entry_vector)) < 1e-6:
            entry_vector = np.asarray([-1.0, 0.0, 0.0], dtype=float)
    entry_direction = normalize(entry_vector)
    safe_entry = entry_anchor + entry_direction * float(args.safe_entry_standoff_m)
    safe_entry[2] = max(safe_entry[2], entry_anchor[2] + float(args.safe_entry_lift_m), positions["approach"][2])
    if float(args.safe_entry_standoff_m) > 0.0:
        positions = {"safe_entry": safe_entry, **positions}
        names = ("safe_entry", *names)
    transit_lift_m = float(args.transit_lift_m)
    if transit_lift_m < 0.0:
        raise ValueError("--transit-lift-m must be non-negative")
    if transit_lift_m > 0.0:
        transit_lift = np.asarray(home_tcp_robot_m, dtype=float).reshape(3).copy()
        transit_lift[2] = max(transit_lift[2] + transit_lift_m, positions[names[0]][2])
        transit_lift[2] = min(transit_lift[2], float(args.max_z_m) - 0.005)
        positions = {"transit_lift": transit_lift, **positions}
        names = ("transit_lift", *names)
    post_cut_lift_m = float(args.post_cut_lift_m)
    if post_cut_lift_m < 0.0:
        raise ValueError("--post-cut-lift-m must be non-negative")
    if post_cut_lift_m > 0.0:
        if str(getattr(args, "task_orientation_policy", "")) == "side_perpendicular":
            post_cut_lift = positions["cut"].copy()
            post_cut_lift[2] = max(post_cut_lift[2] + post_cut_lift_m, post_cut_lift[2])
            post_cut_lift[2] = min(post_cut_lift[2], float(args.max_z_m) - 0.005)
            positions["post_cut_lift"] = post_cut_lift
        else:
            positions["post_cut_lift"] = positions["cut"] - stroke_axis * post_cut_lift_m
        positions["retract_lift"] = safe_retract_lift_position(
            exit_anchor=positions["post_cut_lift"],
            args=args,
        )
        positions["retract"] = safe_post_cut_retract_position(
            exit_anchor=positions["retract_lift"],
            home_tcp_robot_m=home_tcp_robot_m,
            args=args,
        )
        prefix = tuple(name for name in ("transit_lift", "safe_entry") if name in positions)
        names = (*prefix, "approach", "pre_cut", "cut", "post_cut_lift", "retract_lift", "retract")
    return_transit_lift_m = float(args.return_transit_lift_m)
    if return_transit_lift_m < 0.0:
        raise ValueError("--return-transit-lift-m must be non-negative")
    if return_transit_lift_m > 0.0:
        exit_anchor = positions.get("post_cut_lift", positions["cut"])
        return_transit = np.asarray(home_tcp_robot_m, dtype=float).reshape(3).copy()
        return_transit[2] = max(return_transit[2] + return_transit_lift_m, exit_anchor[2])
        return_transit[2] = min(return_transit[2], float(args.max_z_m) - 0.005)
        positions["return_transit"] = return_transit
        names = tuple(
            item
            for name in names
            for item in ((name, "return_transit") if name == "retract" else (name,))
        )
    return positions, names


def safe_post_cut_retract_position(
    *,
    exit_anchor: np.ndarray,
    home_tcp_robot_m: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    anchor = np.asarray(exit_anchor, dtype=float).reshape(3)
    home_tcp = np.asarray(home_tcp_robot_m, dtype=float).reshape(3)
    exit_vector = home_tcp - anchor
    exit_vector[2] = 0.0
    if float(np.linalg.norm(exit_vector)) < 1e-6:
        exit_vector = np.asarray([-1.0, 0.0, 0.0], dtype=float)
    retract = anchor + normalize(exit_vector) * float(args.retract_standoff_m)
    retract[2] = max(anchor[2] + float(args.retract_lift_m), anchor[2])
    retract[2] = min(retract[2], float(args.max_z_m) - 0.005)
    return retract


def safe_retract_lift_position(
    *,
    exit_anchor: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    anchor = np.asarray(exit_anchor, dtype=float).reshape(3)
    lift = anchor.copy()
    lift[2] = max(anchor[2] + float(args.retract_lift_m), anchor[2])
    lift[2] = min(lift[2], float(args.max_z_m) - 0.005)
    return lift


def build_mps_template_waypoints(
    *,
    cut_point_robot_m: np.ndarray,
    home_tcp_robot_m: np.ndarray,
    guidance: DeleafMpsGuidance,
    prior: dict[str, Any] | None,
    tool_axes: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[DeleafWaypoint], dict[str, Any]]:
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    home_tcp = np.asarray(home_tcp_robot_m, dtype=float).reshape(3)
    fallback_approach = normalize(cut_point - home_tcp)
    approach_axis = np.asarray(guidance.approach_axis_robot, dtype=float).reshape(3)
    approach_source = "mps"
    if guidance.diagnostics["approach_confidence"] < float(args.min_axis_confidence):
        approach_axis = fallback_approach
        approach_source = "home_to_target_fallback"
    if float(np.dot(approach_axis, fallback_approach)) < 0.0:
        approach_axis = -approach_axis
    stroke_axis = np.asarray(guidance.cut_axis_robot, dtype=float).reshape(3)
    raw_mps_stroke_axis = stroke_axis.copy()
    stroke_source = "mps"
    if guidance.diagnostics["cut_confidence"] < float(args.min_axis_confidence):
        if prior is not None:
            stroke_axis = np.asarray(prior["cut_axis_robot"], dtype=float).reshape(3)
            stroke_source = "teleop_scalar_prior_fallback"
        else:
            stroke_axis = WORLD_DOWN.copy()
            stroke_source = "world_down_fallback"
    if args.prefer_downward_cut_axis and float(np.dot(stroke_axis, WORLD_DOWN)) < 0.0:
        stroke_axis = -stroke_axis
    if stroke_source == "mps":
        stroke_axis, stroke_source = select_stroke_axis(
            guidance_axis=stroke_axis,
            prior=prior,
            policy=args.stroke_axis_policy,
            max_tilt_deg=float(args.max_stroke_tilt_deg),
        )
    else:
        stroke_axis = normalize(stroke_axis)
    requested_axis = parse_vector3(args.contact_advance_axis_robot, name="contact advance axis")
    contact_advance_axis = np.asarray([0.0, 0.0, 0.0], dtype=float)
    if requested_axis is not None:
        contact_advance_axis = np.asarray(requested_axis, dtype=float).reshape(3)
        contact_advance_axis[2] = 0.0
    if float(np.linalg.norm(contact_advance_axis)) >= 1e-6:
        contact_advance_axis = normalize(contact_advance_axis)

    vertical_cut = bool(getattr(args, "vertical_cut", False))
    vertical_cut_rotation = None
    if vertical_cut:
        # Canonical straight-down petiole cut: the blade points straight DOWN (approach axis = world
        # -Z), the descent stroke is also world -Z, and a horizontal heading orients the edge. The
        # wrist stays untwisted and there is no lateral swing toward the rods/bed. Path/point-cloud/
        # self clearance gates still validate the descent.
        stroke_axis = WORLD_DOWN.copy()
        stroke_source = "vertical_cut_world_down"
        forced_heading = parse_vector3(getattr(args, "force_approach_axis_robot", None), name="force approach axis")
        heading = (
            np.asarray(forced_heading, dtype=float).reshape(3)
            if forced_heading is not None
            else np.asarray(approach_axis, dtype=float).reshape(3).copy()
        )
        heading[2] = 0.0
        if float(np.linalg.norm(heading)) < 1e-6:
            heading = np.asarray([1.0, 0.0, 0.0], dtype=float)
        heading = normalize(heading)
        vertical_cut_rotation = deleaf_rotation(approach_axis_robot=WORLD_DOWN.copy(), cut_axis_robot=heading)
        approach_axis = WORLD_DOWN.copy()
        approach_source = "vertical_cut_blade_down"

    orientation_policy = str(args.task_orientation_policy)
    rotation_source = "mps_axes"
    if vertical_cut:
        rotation = vertical_cut_rotation
        rotation_source = "vertical_straight_down"
    elif orientation_policy == "teleop_pre_cut" and prior is not None:
        rotation = np.asarray(prior["waypoints"]["pre_cut"]["rotation_robot_from_tcp"], dtype=float).reshape(3, 3)
        rotation_source = "teleop_pre_cut_orientation_regularizer"
    elif orientation_policy == "side_perpendicular":
        if float(np.linalg.norm(contact_advance_axis)) < 1e-6:
            raise ValueError("--task-orientation-policy side_perpendicular requires --contact-advance-axis-robot")
        approach_axis = contact_advance_axis.copy()
        approach_source = "contact_advance_axis_perpendicular"
        rotation = deleaf_rotation(approach_axis_robot=approach_axis, cut_axis_robot=stroke_axis)
        rotation_source = "contact_advance_axis_perpendicular"
    elif orientation_policy in {"teleop_pre_cut", "mps_axes"}:
        rotation = deleaf_rotation(approach_axis_robot=approach_axis, cut_axis_robot=stroke_axis)
    else:
        raise ValueError(f"unknown task orientation policy: {orientation_policy}")
    orientation_yaw_deg = float(getattr(args, "orientation_yaw_deg", 0.0))
    if abs(orientation_yaw_deg) > 1e-6 and not vertical_cut:
        # Swing the blade's horizontal heading about world +Z so the shaft can clear a stake,
        # keeping the near-vertical descent and the tip target unchanged. The IK re-aligns the
        # edge to the rotated axes; reachability/clearance gates validate each yaw.
        rotation = yaw_about_world_z(orientation_yaw_deg) @ rotation
        rotation_source = f"{rotation_source}+yaw_{orientation_yaw_deg:+.0f}deg"
    contact_advance_m = float(args.contact_advance_m)
    if contact_advance_m < 0.0:
        raise ValueError("--contact-advance-m must be non-negative")
    if float(np.linalg.norm(contact_advance_axis)) >= 1e-6:
        contact_advance = contact_advance_axis * contact_advance_m
    else:
        contact_advance = np.asarray([0.0, 0.0, 0.0], dtype=float)
    pre_cut_standoff_m = None if args.pre_cut_standoff_m is None else float(args.pre_cut_standoff_m)
    if pre_cut_standoff_m is None:
        pre_cut_standoff_m = prior_scalar_distance(prior, "pre_cut", "cut", float(args.pre_cut_lift_m))
    if pre_cut_standoff_m <= 0.0:
        raise ValueError("--pre-cut-standoff-m must be positive")
    approach_stroke_standoff_m = None if args.approach_stroke_standoff_m is None else float(args.approach_stroke_standoff_m)
    if approach_stroke_standoff_m is None:
        approach_stroke_standoff_m = max(pre_cut_standoff_m, float(args.safe_entry_lift_m))
    if approach_stroke_standoff_m <= 0.0:
        raise ValueError("--approach-stroke-standoff-m must be positive")
    target_extension_m = 0.0 if vertical_cut else float(args.target_extension_m)
    if target_extension_m < 0.0:
        raise ValueError("--target-extension-m must be non-negative")
    extension_axis = -rotation[:, 2]
    extension_axis = np.asarray(extension_axis, dtype=float).reshape(3)
    extension_axis[2] = 0.0
    if target_extension_m > 0.0:
        if float(np.linalg.norm(extension_axis)) < 1e-6:
            raise ValueError("tool approach axis has no horizontal component for target extension")
        extension = normalize(extension_axis) * target_extension_m
    else:
        extension = np.asarray([0.0, 0.0, 0.0], dtype=float)
    cut_through_m = float(args.cut_through_m)
    if cut_through_m < 0.0:
        raise ValueError("--cut-through-m must be non-negative")
    cut_target = cut_point + extension
    pre_cut_position = cut_target - stroke_axis * pre_cut_standoff_m
    requested_approach_stroke_standoff_m = approach_stroke_standoff_m
    approach_stroke_standoff_m, approach_workspace_limit = limit_approach_standoff_to_workspace(
        pre_cut_position_robot_m=pre_cut_position,
        stroke_axis_robot=stroke_axis,
        requested_standoff_m=approach_stroke_standoff_m,
        max_z_m=float(args.max_z_m),
    )
    positions = {
        "approach": pre_cut_position - stroke_axis * approach_stroke_standoff_m,
        "pre_cut": pre_cut_position,
        "cut": cut_target + stroke_axis * cut_through_m,
        "retract": cut_target - stroke_axis * pre_cut_standoff_m - normalize(fallback_approach) * float(args.retract_standoff_m),
    }
    if args.contact_advance_mode == "target":
        positions = {name: position + contact_advance for name, position in positions.items()}
    names = ("approach", "pre_cut", "cut", "retract")
    positions, names = add_safe_entry_and_transits(
        positions=positions,
        names=names,
        home_tcp_robot_m=home_tcp,
        contact_advance=contact_advance,
        stroke_axis=stroke_axis,
        args=args,
    )
    waypoints = [
        DeleafWaypoint(
            name=name,
            target_pose=transform_from_rotation_translation(rotation, positions[name]),
            orientation_required=deleaf_phase_requires_orientation(name),
        )
        for name in names
    ]
    axis_targets_by_name = {
        name: axis_targets_from_rotation(rotation, tool_axes)
        for name in names
    }
    return waypoints, {
        "waypoint_policy": "mps_template",
        "cut_point_robot_m": cut_point.tolist(),
        "home_tcp_robot_m": home_tcp.tolist(),
        "approach_axis_robot": normalize(approach_axis).tolist(),
        "approach_axis_source": approach_source,
        "cut_axis_robot": stroke_axis.astype(float).tolist(),
        "raw_mps_cut_axis_robot": normalize(raw_mps_stroke_axis).tolist(),
        "stroke_axis_robot": stroke_axis.astype(float).tolist(),
        "stroke_axis_source": stroke_source,
        "stroke_axis_policy": args.stroke_axis_policy,
        "max_stroke_tilt_deg": float(args.max_stroke_tilt_deg),
        "constrained_cut_axis_robot": (-rotation[:, 1]).tolist(),
        "task_orientation_policy": orientation_policy,
        "task_orientation_source": rotation_source,
        "orientation_yaw_deg": float(getattr(args, "orientation_yaw_deg", 0.0)),
        "orientation_required_phases": [
            waypoint.name for waypoint in waypoints if waypoint.orientation_required
        ],
        "rotation_robot_from_tcp": rotation.tolist(),
        "tool_axis_local": serializable_tool_axes(tool_axes),
        "axis_targets_by_name": axis_targets_by_name,
        "contact_advance_m": contact_advance_m,
        "contact_advance_mode": args.contact_advance_mode,
        "contact_advance_axis_robot": contact_advance_axis.astype(float).tolist(),
        "contact_advance_robot_m": contact_advance.astype(float).tolist(),
        "target_extension_m": target_extension_m,
        "target_extension_robot_m": extension.astype(float).tolist(),
        "pre_cut_standoff_m": pre_cut_standoff_m,
        "requested_approach_stroke_standoff_m": requested_approach_stroke_standoff_m,
        "approach_stroke_standoff_m": approach_stroke_standoff_m,
        "approach_workspace_limit": approach_workspace_limit,
        "transit_lift_m": float(args.transit_lift_m),
        "return_transit_lift_m": float(args.return_transit_lift_m),
        "cut_through_m": cut_through_m,
        "post_cut_lift_m": float(args.post_cut_lift_m),
        "safe_entry_standoff_m": float(args.safe_entry_standoff_m),
        "safe_entry_lift_m": float(args.safe_entry_lift_m),
        "positions_robot_m": {name: value.tolist() for name, value in positions.items()},
        "mps_role": "phase_axes_and_action_frame",
        "teleop_prior": None if prior is None else {
            "source": prior["source"],
            "indices": prior["indices"],
            "cut_anchor_robot_m": prior["cut_anchor_robot_m"],
            "role": "scalar_template_only",
            "uses_robot_frame_offsets": False,
            "uses_joint_replay": False,
            "uses_teleop_joint_seeds": False,
            "joint_seed_policy": "chained_previous_solution",
            "q_seed_by_name": {},
        },
    }


def build_teleop_prior_waypoints(
    *,
    cut_point_robot_m: np.ndarray,
    home_tcp_robot_m: np.ndarray,
    prior: dict[str, Any],
    tool_axes: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[DeleafWaypoint], dict[str, Any]]:
    cut_point = np.asarray(cut_point_robot_m, dtype=float).reshape(3)
    teleop_follow = getattr(args, "teleop_follow", False)
    names = ("approach", "pre_cut", "cut", "retract")
    rotations = {
        name: np.asarray(prior["waypoints"][name]["rotation_robot_from_tcp"], dtype=float).reshape(3, 3)
        for name in names
    }
    orientation_yaw_deg = float(getattr(args, "orientation_yaw_deg", 0.0))
    orientation_yaw_rotation = yaw_about_world_z(orientation_yaw_deg)
    if abs(orientation_yaw_deg) > 1e-6:
        # The teleop-offset path returns before the generic MPS-template orientation code. Apply
        # the same world-frame yaw here so azimuth-sweep candidates change the physical knife
        # heading while retaining the demonstrated target-relative Cartesian path.
        rotations = {
            name: orientation_yaw_rotation @ rotation
            for name, rotation in rotations.items()
        }
    locked_rotation = rotations["pre_cut"]
    # The prior's offsets are referenced to the DEEPEST through-cut sample (extract indices["cut"]=
    # cut_step+1). The operator clicks the PETIOLE SURFACE, not the bottom of the human's through-cut,
    # so anchoring the click on the deepest sample shifts the whole demo UP by (click - deepest) ~7cm,
    # which pushes the approach into the z-ceiling clamp and toward the stake's wider upper portion.
    # In demo-follow, re-reference the offsets to the pre_cut ENGAGEMENT pose (blade poised at the
    # petiole) so the click maps to where the blade meets the petiole: approach/pre_cut sit at the
    # click level (no clamp) and the cut descends BELOW by the human's demonstrated through-cut depth.
    anchor_offset = (
        np.asarray(prior["waypoints"]["pre_cut"]["offset_from_cut_anchor_m"], dtype=float).reshape(3)
        if teleop_follow
        else np.zeros(3)
    )
    positions = {
        name: cut_point
        + np.asarray(prior["waypoints"][name]["offset_from_cut_anchor_m"], dtype=float).reshape(3)
        - anchor_offset
        for name in names
    }
    # demo-follow cut-depth cap: the demo cut waypoint sits at the bottom of the human's full
    # downward draw (can be 20+cm of follow-through), which would plunge the blade deep through the
    # canopy. Cap the descent below the engagement to --teleop-cut-depth-m while keeping the human's
    # draw DIRECTION, so the blade cuts a few cm through the petiole instead of the whole draw.
    teleop_cut_depth_m = getattr(args, "teleop_cut_depth_m", None)
    if teleop_follow and teleop_cut_depth_m is not None:
        descent = positions["cut"] - positions["pre_cut"]
        length = float(np.linalg.norm(descent))
        cap = float(teleop_cut_depth_m)
        if cap > 0.0 and length > cap:
            positions["cut"] = positions["pre_cut"] + descent * (cap / length)
    stroke_axis = normalize(positions["cut"] - positions["pre_cut"])
    contact_advance_m = float(args.contact_advance_m)
    if contact_advance_m < 0.0:
        raise ValueError("--contact-advance-m must be non-negative")
    requested_axis = parse_vector3(args.contact_advance_axis_robot, name="contact advance axis")
    if requested_axis is None:
        contact_advance_axis = np.asarray(stroke_axis, dtype=float).reshape(3)
        contact_advance_axis[2] = 0.0
    else:
        contact_advance_axis = np.asarray(requested_axis, dtype=float).reshape(3)
        contact_advance_axis[2] = 0.0
    if float(np.linalg.norm(contact_advance_axis)) < 1e-6:
        contact_advance_axis = np.asarray(prior["waypoints"]["cut"]["offset_from_cut_anchor_m"], dtype=float).reshape(3)
        contact_advance_axis[2] = 0.0
    if float(np.linalg.norm(contact_advance_axis)) >= 1e-6:
        contact_advance_axis = normalize(contact_advance_axis)
        contact_advance = contact_advance_axis * contact_advance_m
        # demo-follow: the human poses already encode the contact geometry; a target-mode advance
        # would push the cut off the clicked petiole, so skip it when following the demo.
        if args.contact_advance_mode == "target" and not teleop_follow:
            positions = {name: position + contact_advance for name, position in positions.items()}
    else:
        contact_advance_axis = np.asarray([0.0, 0.0, 0.0], dtype=float)
        contact_advance = np.asarray([0.0, 0.0, 0.0], dtype=float)
    stroke_axis = normalize(positions["cut"] - positions["pre_cut"])
    pre_cut_standoff_m = None if args.pre_cut_standoff_m is None else float(args.pre_cut_standoff_m)
    # demo-follow keeps the human's pre_cut position as the IK target; the standoff overwrite would
    # discard it and re-impose a straight near-vertical chord (the thing that grazed the rod).
    if pre_cut_standoff_m is not None and not getattr(args, "teleop_follow", False):
        if pre_cut_standoff_m <= 0.0:
            raise ValueError("--pre-cut-standoff-m must be positive")
        positions["pre_cut"] = positions["cut"] - stroke_axis * pre_cut_standoff_m
    # demo-follow: the demo cut anchor already encodes the human standoff/penetration, so re-adding
    # target_extension (config default 0.07) would shift the cut off the clicked petiole. Zero it.
    target_extension_m = 0.0 if teleop_follow else float(args.target_extension_m)
    if target_extension_m < 0.0:
        raise ValueError("--target-extension-m must be non-negative")
    if target_extension_m > 0.0:
        extension_axis = -locked_rotation[:, 2]
        extension_axis = np.asarray(extension_axis, dtype=float).reshape(3)
        extension_axis[2] = 0.0
        if float(np.linalg.norm(extension_axis)) < 1e-6:
            raise ValueError("tool approach axis has no horizontal component for target extension")
        extension = normalize(extension_axis) * target_extension_m
        for name in ("approach", "pre_cut", "cut", "retract"):
            positions[name] = positions[name] + extension
    else:
        extension = np.asarray([0.0, 0.0, 0.0], dtype=float)
    approach_stroke_standoff_m = None if args.approach_stroke_standoff_m is None else float(args.approach_stroke_standoff_m)
    if approach_stroke_standoff_m is not None and approach_stroke_standoff_m <= 0.0:
        raise ValueError("--approach-stroke-standoff-m must be positive")
    approach_policy = "teleop_offset"
    if approach_stroke_standoff_m is not None and not getattr(args, "teleop_follow", False):
        positions["approach"] = positions["pre_cut"] - stroke_axis * approach_stroke_standoff_m
        approach_policy = "task_relative_requested"
    elif pre_cut_standoff_m is not None and positions["approach"][2] > float(args.max_z_m):
        approach_stroke_standoff_m = max(pre_cut_standoff_m, float(args.safe_entry_lift_m))
        positions["approach"] = positions["pre_cut"] - stroke_axis * approach_stroke_standoff_m
        approach_policy = "task_relative_workspace_fallback"
    entry_anchor = positions["cut"] + contact_advance
    requested_safe_entry_axis = parse_vector3(args.safe_entry_axis_robot, name="safe entry axis")
    if requested_safe_entry_axis is None:
        entry_vector = np.asarray(home_tcp_robot_m, dtype=float).reshape(3) - entry_anchor
    else:
        entry_vector = np.asarray(requested_safe_entry_axis, dtype=float).reshape(3)
    entry_vector[2] = 0.0
    if float(np.linalg.norm(entry_vector)) < 1e-6:
        entry_vector = np.asarray([-1.0, 0.0, 0.0], dtype=float)
    entry_direction = normalize(entry_vector)
    safe_entry = entry_anchor + entry_direction * float(args.safe_entry_standoff_m)
    safe_entry[2] = max(safe_entry[2], entry_anchor[2] + float(args.safe_entry_lift_m), positions["approach"][2])
    if float(args.safe_entry_standoff_m) > 0.0:
        positions = {"safe_entry": safe_entry, **positions}
        names = ("safe_entry", "approach", "pre_cut", "cut", "retract")
    transit_lift_m = float(args.transit_lift_m)
    if transit_lift_m < 0.0:
        raise ValueError("--transit-lift-m must be non-negative")
    if transit_lift_m > 0.0:
        transit_lift = np.asarray(home_tcp_robot_m, dtype=float).reshape(3).copy()
        transit_lift[2] = max(transit_lift[2] + transit_lift_m, positions[names[0]][2])
        transit_lift[2] = min(transit_lift[2], float(args.max_z_m) - 0.005)
        positions = {"transit_lift": transit_lift, **positions}
        names = ("transit_lift", *names)
    # demo-follow: the demo cut anchor IS the human through-cut bottom (extract sets
    # indices["cut"] = cut_step+1), so re-adding cut_through would over-plunge past the petiole
    # into the gate-blind exclusion sphere. Zero it when following the demo.
    cut_through_m = 0.0 if getattr(args, "teleop_follow", False) else float(args.cut_through_m)
    if cut_through_m < 0.0:
        raise ValueError("--cut-through-m must be non-negative")
    positions["cut"] = positions["cut"] + stroke_axis * cut_through_m
    post_cut_lift_m = float(args.post_cut_lift_m)
    if post_cut_lift_m < 0.0:
        raise ValueError("--post-cut-lift-m must be non-negative")
    if post_cut_lift_m > 0.0:
        positions["post_cut_lift"] = positions["cut"] - stroke_axis * post_cut_lift_m
        positions["retract_lift"] = safe_retract_lift_position(
            exit_anchor=positions["post_cut_lift"],
            args=args,
        )
        positions["retract"] = safe_post_cut_retract_position(
            exit_anchor=positions["retract_lift"],
            home_tcp_robot_m=home_tcp_robot_m,
            args=args,
        )
        prefix = ("transit_lift",) if "transit_lift" in positions else ()
        if "safe_entry" in positions:
            names = (*prefix, "safe_entry", "approach", "pre_cut", "cut", "post_cut_lift", "retract_lift", "retract")
        else:
            names = (*prefix, "approach", "pre_cut", "cut", "post_cut_lift", "retract_lift", "retract")
    return_transit_lift_m = float(args.return_transit_lift_m)
    if return_transit_lift_m < 0.0:
        raise ValueError("--return-transit-lift-m must be non-negative")
    if return_transit_lift_m > 0.0:
        exit_anchor = positions.get("post_cut_lift", positions["cut"])
        return_transit = np.asarray(home_tcp_robot_m, dtype=float).reshape(3).copy()
        return_transit[2] = max(return_transit[2] + return_transit_lift_m, exit_anchor[2])
        return_transit[2] = min(return_transit[2], float(args.max_z_m) - 0.005)
        positions["return_transit"] = return_transit
        names = tuple(
            item
            for name in names
            for item in ((name, "return_transit") if name == "retract" else (name,))
        )
    def _wp_rotation(name: str) -> np.ndarray:
        # demo-follow uses each phase's OWN demonstrated orientation (the human laid the blade back
        # 24-85 deg across phases); otherwise the single locked pre_cut pose as before. Synthetic
        # transit/lift/safe_entry names are not in `rotations`, so they keep the locked pose.
        if teleop_follow and name in rotations:
            return rotations[name]
        return locked_rotation

    waypoints = [
        DeleafWaypoint(
            name=name,
            target_pose=transform_from_rotation_translation(_wp_rotation(name), positions[name]),
            orientation_required=deleaf_phase_requires_orientation(name),
        )
        for name in names
    ]
    q_seed_by_name = {
        name: prior["waypoints"][name]["q_arm_rad"]
        for name in ("approach", "pre_cut", "cut", "retract")
    } if (args.use_teleop_joint_seeds or teleop_follow) else {}
    axis_targets_by_name = {
        name: axis_targets_from_rotation(_wp_rotation(name), tool_axes)
        for name in names
    }
    # HYBRID (operator's design): IK plans the approach (transit_lift->approach->pre_cut, kept above),
    # then the solver FOLLOWS the human's recorded pre_cut->cut->retract draw instead of the synthetic
    # cut + post_cut_lift + retract_lift + safe-retract. The dense path is re-anchored exactly like the
    # 4-pose poses (cut_point + offset - anchor_offset), per-sample orientation + human q seed, and was
    # joint-distance resampled by the extractor so hops are bounded. return_transit is recomputed from
    # the last (retract) dense pose. Only the cut ACTION is densified -> no high approach arc -> no z-ceiling.
    cut_path_follow_meta: dict[str, Any] | None = None
    if teleop_follow and getattr(args, "teleop_follow_cut", False) and prior.get("cut_path"):
        cut_path = prior["cut_path"]
        head = [w for w in waypoints if w.name in ("transit_lift", "safe_entry", "approach", "pre_cut")]
        # Re-anchor every recorded pose exactly like the 4-pose follow.
        poses = [cut_point + np.asarray(cp["offset_from_cut_anchor_m"], dtype=float).reshape(3) - anchor_offset for cp in cut_path]
        # Clip the FOLLOWED draw to the configured safe workspace AABB. The human's retract pulls the
        # blade all the way back toward the body (x below min_x); that disengage tail is exactly what
        # the synthetic return_transit lift already does, so we follow the cut + draw-through only as
        # far as the envelope allows, then hand off to return_transit. Never clip at/before the cut.
        ws = bounds_from_args(args)
        def _in_ws(p: np.ndarray) -> bool:
            return (ws["x"][0] <= p[0] <= ws["x"][1] and ws["y"][0] <= p[1] <= ws["y"][1] and ws["z"][0] <= p[2] <= ws["z"][1])
        cut_dense_i = next((i for i, cp in enumerate(cut_path) if cp.get("is_cut")), 0)
        keep_n = len(cut_path)
        for i in range(cut_dense_i + 1, len(cut_path)):
            if not _in_ws(poses[i]):
                keep_n = i
                break
        dropped = cut_path[keep_n:]
        cut_path = cut_path[:keep_n]
        poses = poses[:keep_n]
        if dropped:
            d0 = cut_point + np.asarray(dropped[0]["offset_from_cut_anchor_m"], dtype=float).reshape(3) - anchor_offset
            print(
                f"[teleop-follow-cut] clipped {len(dropped)} trailing retract sample(s) outside workspace "
                f"AABB x{ws['x']} y{ws['y']} z{ws['z']}; first dropped pose={np.round(d0, 3).tolist()} "
                f"(disengage tail handed to return_transit). Following {len(cut_path)} dense poses through the cut+draw.",
                flush=True,
            )
        dense_wps: list[DeleafWaypoint] = []
        for i, cp in enumerate(cut_path):
            nm = f"cut_path_{i:03d}"
            pos = poses[i]
            rot = np.asarray(cp["rotation_robot_from_tcp"], dtype=float).reshape(3, 3)
            if abs(orientation_yaw_deg) > 1e-6:
                rot = orientation_yaw_rotation @ rot
            dense_wps.append(DeleafWaypoint(name=nm, target_pose=transform_from_rotation_translation(rot, pos), orientation_required=True))
            q_seed_by_name[nm] = list(cp["q_arm_rad"])
            axis_targets_by_name[nm] = axis_targets_from_rotation(rot, tool_axes)
            positions[nm] = pos
        last_pos = poses[-1]
        rt = np.asarray(home_tcp_robot_m, dtype=float).reshape(3).copy()
        rt[2] = min(max(rt[2] + return_transit_lift_m, float(last_pos[2])), float(args.max_z_m) - 0.005)
        positions["return_transit"] = rt
        axis_targets_by_name["return_transit"] = axis_targets_from_rotation(locked_rotation, tool_axes)
        waypoints = head + dense_wps + [
            DeleafWaypoint(name="return_transit", target_pose=transform_from_rotation_translation(locked_rotation, rt), orientation_required=False)
        ]
        cut_path_follow_meta = {
            "total_cut_path_points": len(prior["cut_path"]),
            "followed_points": len(cut_path),
            "dropped_tail_points": len(dropped),
            "drop_reason": "outside_workspace_aabb" if dropped else None,
            "pinned_cut_dense_index": int(cut_dense_i),
            "last_followed_pos_robot_m": last_pos.astype(float).tolist(),
            "workspace_aabb": {k: list(v) for k, v in ws.items()},
        }
    return waypoints, {
        "cut_path_follow": cut_path_follow_meta,
        "cut_point_robot_m": cut_point.tolist(),
        "home_tcp_robot_m": np.asarray(home_tcp_robot_m, dtype=float).reshape(3).tolist(),
        "approach_axis_robot": (-locked_rotation[:, 2]).astype(float).tolist(),
        "cut_axis_robot": (-locked_rotation[:, 1]).astype(float).tolist(),
        "constrained_cut_axis_robot": (-locked_rotation[:, 1]).astype(float).tolist(),
        "rotation_robot_from_tcp": locked_rotation.astype(float).tolist(),
        "orientation_yaw_deg": orientation_yaw_deg,
        "task_orientation_source": (
            "teleop_phase_orientations"
            if teleop_follow
            else "teleop_pre_cut_orientation_regularizer"
        ) + (f"+yaw_{orientation_yaw_deg:+.0f}deg" if abs(orientation_yaw_deg) > 1e-6 else ""),
        "orientation_required_phases": [
            waypoint.name for waypoint in waypoints if waypoint.orientation_required
        ],
        "tool_axis_local": serializable_tool_axes(tool_axes),
        "axis_targets_by_name": axis_targets_by_name,
        "stroke_axis_robot": stroke_axis.astype(float).tolist(),
        "contact_advance_m": contact_advance_m,
        "contact_advance_mode": args.contact_advance_mode,
        "contact_advance_axis_robot": contact_advance_axis.astype(float).tolist(),
        "contact_advance_robot_m": contact_advance.astype(float).tolist(),
        "target_extension_m": target_extension_m,
        "target_extension_robot_m": extension.astype(float).tolist(),
        "pre_cut_standoff_m": pre_cut_standoff_m,
        "approach_stroke_standoff_m": approach_stroke_standoff_m,
        "approach_policy": approach_policy,
        "transit_lift_m": transit_lift_m,
        "return_transit_lift_m": return_transit_lift_m,
        "cut_through_m": cut_through_m,
        "post_cut_lift_m": post_cut_lift_m,
        "safe_entry_standoff_m": float(args.safe_entry_standoff_m),
        "safe_entry_lift_m": float(args.safe_entry_lift_m),
        "positions_robot_m": {name: value.tolist() for name, value in positions.items()},
        "teleop_prior": {
            "source": prior["source"],
            "indices": prior["indices"],
            "cut_anchor_robot_m": prior["cut_anchor_robot_m"],
            "orientation_policy": "lock_pre_cut_rotation",
            "uses_joint_replay": False,
            "uses_teleop_joint_seeds": bool(args.use_teleop_joint_seeds),
            "joint_seed_policy": "teleop_diagnostic_seed" if args.use_teleop_joint_seeds else "chained_previous_solution",
            "q_seed_by_name": q_seed_by_name,
        },
    }


def approach_path_seed_q(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    guidance: DeleafMpsGuidance,
    waypoint_diagnostics: dict[str, Any],
    config: IKSolverConfig,
    start_q_arm_rad: np.ndarray,
    primary_axis_local: np.ndarray,
    secondary_axis_local: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """SEED-ONLY warm-start along the demonstrated approach polyline.

    Re-anchors the human approach path to the clicked cut point, then chains best-effort
    IK solves (orientation NOT required) along it using the SAME solver the waypoints use,
    advancing the running seed only on success. Returns the FINAL chained q to seed the
    real 'approach' waypoint, plus a small diagnostic. These solves are SEED-ONLY: they are
    never added to the waypoint list, never gated, and never executed. Returns None when the
    guidance has no usable approach path / anchor (caller then keeps the default seed).
    """
    if guidance.approach_path_robot_m is None or guidance.cut_anchor_robot_m is None:
        return None
    path = np.asarray(guidance.approach_path_robot_m, dtype=float).reshape(-1, 3)
    if path.shape[0] < 1:
        return None
    anchor = np.asarray(guidance.cut_anchor_robot_m, dtype=float).reshape(3)
    cut_point = np.asarray(waypoint_diagnostics["cut_point_robot_m"], dtype=float).reshape(3)
    # Re-anchor the demonstrated polyline so it ends at the live clicked cut point.
    reanchored = (path - anchor) + cut_point
    rotation = np.asarray(waypoint_diagnostics["rotation_robot_from_tcp"], dtype=float).reshape(3, 3)
    axis_targets = axis_targets_from_rotation(rotation, {
        "primary_axis_local": primary_axis_local,
        "secondary_axis_local": secondary_axis_local,
    })
    jaw_axis_robot = np.asarray(axis_targets["primary_axis_robot"], dtype=float)
    backoff_axis_robot = np.asarray(axis_targets["secondary_axis_robot"], dtype=float)
    # Orientation NOT required for the seed chain: drive position toward each guide point,
    # keep the cut rotation as a soft target so the basin stays human-like.
    seed_config = replace(config, orientation_weight=0.0)
    running_q = np.asarray(start_q_arm_rad, dtype=float)
    solved_count = 0
    for point in reanchored:
        target_pose = transform_from_rotation_translation(rotation, np.asarray(point, dtype=float).reshape(3))
        result = solve_tcp_task_axis_ik(
            kinematics,
            locked_state,
            target_pose,
            jaw_axis_robot=jaw_axis_robot,
            backoff_axis_robot=backoff_axis_robot,
            gripper_jaw_axis_local=primary_axis_local,
            gripper_backoff_axis_local=secondary_axis_local,
            seed_q_arm_rad=running_q,
            reference_q_arm_rad=running_q,
            config=seed_config,
        )
        if result.ok:
            running_q = np.asarray(result.q_arm_rad, dtype=float)
            solved_count += 1
    diagnostics = {
        "enabled": True,
        "guide_points": int(reanchored.shape[0]),
        "guide_points_solved": int(solved_count),
        "cut_anchor_robot_m": anchor.tolist(),
        "reanchored_path_robot_m": reanchored.astype(float).tolist(),
    }
    return running_q, diagnostics


def solve_waypoints(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    waypoints: list[DeleafWaypoint],
    waypoint_diagnostics: dict[str, Any],
    config: IKSolverConfig,
    args: argparse.Namespace,
    guidance: DeleafMpsGuidance | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    previous_q = np.asarray(home_q_arm, dtype=float)
    solved: list[dict[str, Any]] = []
    hard_ok = True
    local_axes = waypoint_diagnostics.get("tool_axis_local", {})
    primary_axis_local = np.asarray(local_axes.get("primary_axis_local", [0.0, -1.0, 0.0]), dtype=float)
    secondary_axis_local = np.asarray(local_axes.get("secondary_axis_local", [0.0, 0.0, 1.0]), dtype=float)
    # SEED-ONLY: precompute a human-like warm-start for the 'approach' waypoint by chaining
    # IK along the re-anchored demonstrated approach path. This only changes the SEED handed
    # to the existing 'approach' solve; it adds no waypoints, no gates, and no motion.
    approach_seed_q: np.ndarray | None = None
    if (
        getattr(args, "mps_approach_path_guidance", False)
        and guidance is not None
        and guidance.approach_path_robot_m is not None
    ):
        seed_result = approach_path_seed_q(
            kinematics=kinematics,
            locked_state=locked_state,
            guidance=guidance,
            waypoint_diagnostics=waypoint_diagnostics,
            config=config,
            start_q_arm_rad=previous_q,
            primary_axis_local=primary_axis_local,
            secondary_axis_local=secondary_axis_local,
        )
        if seed_result is not None:
            approach_seed_q, waypoint_diagnostics["approach_path_seed"] = seed_result
    for waypoint in waypoints:
        orientation_required = waypoint.orientation_required
        waypoint_config = config if orientation_required else replace(config, orientation_weight=0.0)
        workspace = workspace_check(waypoint.target_pose[:3, 3], bounds_from_args(args))
        axis_targets = waypoint_diagnostics.get("axis_targets_by_name", {}).get(waypoint.name, {})
        teleop_seed_map = (waypoint_diagnostics.get("teleop_prior") or {}).get("q_seed_by_name", {})
        seed_q = np.asarray(teleop_seed_map.get(waypoint.name, previous_q), dtype=float)
        seed_source = "teleop_joint_seed" if waypoint.name in teleop_seed_map else "previous_solution"
        # SEED-ONLY approach warm-start: override only the 'approach' seed (never when an
        # explicit teleop joint seed is supplied). Gates/timing/waypoints are untouched.
        if approach_seed_q is not None and waypoint.name == "approach" and waypoint.name not in teleop_seed_map:
            seed_q = np.asarray(approach_seed_q, dtype=float)
            seed_source = "mps_approach_path_warm_start"
        # demo-follow: regularize the smoothness term toward the human seed (NOT the chained previous
        # solution) for teleop-seeded waypoints, so IK both starts AND stays in the human joint
        # branch instead of silently relaxing back into the failing branch with a tiny position error.
        reference_q = seed_q if waypoint.name in teleop_seed_map else previous_q
        seed_options: list[tuple[str, np.ndarray, np.ndarray]] = [(seed_source, seed_q, reference_q)]
        demo_library = waypoint_diagnostics.get("successful_demo_joint_seed_library") or {}
        for demo in demo_library.get("candidates", []):
            demo_seed = (demo.get("q_seed_by_name") or {}).get(waypoint.name)
            if demo_seed is None:
                continue
            demo_q = np.asarray(demo_seed, dtype=float).reshape(-1)
            if demo_q.shape != previous_q.shape:
                continue
            if any(np.allclose(demo_q, existing_seed, atol=1e-9) for _, existing_seed, _ in seed_options):
                continue
            seed_options.append((f"successful_demo_seed:{demo['label']}", demo_q, previous_q))

        attempts: list[dict[str, Any]] = []
        selected_attempt: dict[str, Any] | None = None
        for attempt_index, (attempt_source, attempt_seed, attempt_reference) in enumerate(seed_options):
            # Successful demonstrations are multi-start hypotheses only. The target pose,
            # smoothness reference, IK objective, and all collision gates remain live/current.
            result = solve_tcp_task_axis_ik(
                kinematics,
                locked_state,
                waypoint.target_pose,
                jaw_axis_robot=np.asarray(
                    axis_targets.get("primary_axis_robot", waypoint_diagnostics["constrained_cut_axis_robot"]),
                    dtype=float,
                ),
                backoff_axis_robot=np.asarray(
                    axis_targets.get("secondary_axis_robot", -np.asarray(waypoint_diagnostics["approach_axis_robot"], dtype=float)),
                    dtype=float,
                ),
                gripper_jaw_axis_local=primary_axis_local,
                gripper_backoff_axis_local=secondary_axis_local,
                seed_q_arm_rad=attempt_seed,
                reference_q_arm_rad=attempt_reference,
                config=waypoint_config,
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
                previous_q,
                result.q_arm_rad,
                samples=args.path_samples,
                margin_m=args.min_clearance_m,
                pair_count=args.clearance_pair_count,
            )
            waypoint_ok = (
                workspace["ok"]
                and result.ok
                and result.position_error_m <= float(args.position_tolerance_m)
                and (not orientation_required or result.orientation_error_rad <= float(args.orientation_tolerance_rad))
                and clearance.ok
                and path_clearance["ok"]
            )
            attempt = {
                "source": attempt_source,
                "result": result,
                "clearance": clearance,
                "path_clearance": path_clearance,
                "waypoint_ok": bool(waypoint_ok),
            }
            attempts.append(attempt)
            if selected_attempt is None:
                selected_attempt = attempt
            else:
                def _attempt_key(item: dict[str, Any]) -> tuple[Any, ...]:
                    item_result = item["result"]
                    return (
                        bool(item["waypoint_ok"]),
                        bool(item_result.ok),
                        item_result.position_error_m <= float(args.position_tolerance_m),
                        not orientation_required
                        or item_result.orientation_error_rad <= float(args.orientation_tolerance_rad),
                        bool(item["clearance"].ok),
                        bool(item["path_clearance"]["ok"]),
                        -float(item_result.position_error_m),
                        -float(item_result.orientation_error_rad) if orientation_required else 0.0,
                        -float(item_result.residual_norm),
                    )
                if _attempt_key(attempt) > _attempt_key(selected_attempt):
                    selected_attempt = attempt
            if waypoint_ok:
                break
            # Keep successful-demo lookup off the fast path: only expand after the normal solve fails.
            if attempt_index == 0 and len(seed_options) == 1:
                break
        if selected_attempt is None:
            raise RuntimeError(f"no IK seed attempt generated for waypoint {waypoint.name}")
        result = selected_attempt["result"]
        clearance = selected_attempt["clearance"]
        path_clearance = selected_attempt["path_clearance"]
        waypoint_ok = bool(selected_attempt["waypoint_ok"])
        seed_source = str(selected_attempt["source"])
        solved.append(
            {
                "name": waypoint.name,
                "orientation_required": bool(orientation_required),
                "target_pose": waypoint.target_pose.tolist(),
                "q_arm_rad": result.q_arm_rad.tolist(),
                "seed_source": seed_source,
                "ik_seed_attempts": [
                    {
                        "source": str(attempt["source"]),
                        "ok": bool(attempt["waypoint_ok"]),
                        "position_error_m": float(attempt["result"].position_error_m),
                        "orientation_error_rad": float(attempt["result"].orientation_error_rad),
                        "self_clearance_ok": bool(attempt["clearance"].ok),
                        "path_clearance_ok": bool(attempt["path_clearance"]["ok"]),
                    }
                    for attempt in attempts
                ],
                "ok": bool(waypoint_ok),
                "workspace": workspace,
                "ik": {
                    "ok": bool(result.ok),
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
            }
        )
        hard_ok = hard_ok and bool(waypoint_ok)
        if not waypoint_ok:
            break
        previous_q = np.asarray(result.q_arm_rad, dtype=float)
    return solved, hard_ok and len(solved) == len(waypoints)


def bounds_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return {
        "x": (float(args.min_x_m), float(args.max_x_m)),
        "y": (float(args.min_y_m), float(args.max_y_m)),
        "z": (float(args.min_z_m), float(args.max_z_m)),
    }


def start_trial_video(args: argparse.Namespace) -> dict[str, Any] | None:
    output = getattr(args, "trial_video_output", None)
    if output is None:
        return None
    video_path = Path(output).resolve()
    metadata_path = video_path.with_name(video_path.stem + "_metadata.json")
    ready_path = video_path.with_name(video_path.stem + ".ready.json")
    stop_path = video_path.with_name(video_path.stem + ".stop")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.unlink(missing_ok=True)
    stop_path.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "robot" / "evaluation" / "record_realsense_video.py"),
            "--output",
            str(video_path),
            "--metadata",
            str(metadata_path),
            "--ready-file",
            str(ready_path),
            "--stop-file",
            str(stop_path),
            "--max-duration-s",
            str(float(args.trial_video_max_duration_s)),
        ],
        cwd=REPO_ROOT,
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if ready_path.is_file():
            return {
                "process": process,
                "video": video_path,
                "metadata": metadata_path,
                "ready": ready_path,
                "stop": stop_path,
            }
        if process.poll() is not None:
            raise RuntimeError(f"trial video recorder exited before ready: {process.returncode}")
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5.0)
    raise RuntimeError("timed out waiting for trial video recorder")


def finish_trial_video(handle: dict[str, Any] | None) -> dict[str, Any] | None:
    if handle is None:
        return None
    handle["stop"].touch()
    process = handle["process"]
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)
    video_path = Path(handle["video"])
    metadata_path = Path(handle["metadata"])
    return {
        "ok": process.returncode == 0 and video_path.is_file(),
        "return_code": process.returncode,
        "video": str(video_path),
        "metadata": str(metadata_path),
    }


def start_demo_capture(args: argparse.Namespace, toolfit: Path) -> dict[str, Any] | None:
    """Begin a background 10Hz robot-state capture during a deleaf cut so a SUCCESSFUL --execute
    yields a training trajectory in the existing capture_robot_tcp_trajectory `_traj.json` format
    (reused verbatim). Uses its own read-only robot client; a capture failure never aborts the cut."""
    import threading

    from robot.policy.capture_robot_tcp_trajectory import capture as capture_trajectory

    cap_args = argparse.Namespace(
        address=args.address,
        model=args.model,
        arm="right",
        dataset_name=args.dataset_name,
        toolfit=Path(toolfit),
        duration_s=float(getattr(args, "demo_duration_s", 600.0)),
        hz=float(getattr(args, "demo_hz", 10.0)),
        manual_start=False,
        manual_stop=False,
        run_label=f"{args.run_label}_traj",
        runs_root=args.runs_root,
        plan=None,
        candidate_name=None,
        waypoint="cut",
    )
    stop_event = threading.Event()
    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            holder["payload"] = capture_trajectory(cap_args, stop_event)
        except Exception as exc:  # noqa: BLE001 - capture must never abort the cut
            holder["error"] = repr(exc)

    thread = threading.Thread(target=_worker, name="deleaf_demo_capture", daemon=True)
    thread.start()
    time.sleep(1.0)  # let the read-only capture client connect + start sampling before motion
    return {"thread": thread, "stop_event": stop_event, "holder": holder, "cap_args": cap_args}


def finish_demo_capture(
    handle: dict[str, Any] | None,
    args: argparse.Namespace,
    payload: dict[str, Any],
    cut_point: np.ndarray,
) -> dict[str, Any] | None:
    """Stop the background capture, write the `_traj.json`, and emit a ready record-action command."""
    if handle is None:
        return None
    from robot.policy.capture_robot_tcp_trajectory import write_capture as write_trajectory

    handle["stop_event"].set()
    handle["thread"].join(timeout=15.0)
    holder = handle["holder"]
    if "payload" not in holder:
        return {"recorded": False, "reason": holder.get("error", "no samples captured")}
    traj_path = write_trajectory(holder["payload"], handle["cap_args"])
    payload["artifacts"]["trajectory_json"] = str(traj_path)
    goal = np.asarray(cut_point, dtype=float).reshape(3).tolist()
    label = getattr(args, "demo_label", None) or "<LABEL>"
    record_command = (
        "python robot/policy/collect_policy_episode.py record-action "
        "--episode-json <EPISODE.json> --primitive deleaf "
        f"--target-label {label} "
        f"--action-artifact \"{payload['artifacts']['json']}\" "
        f"--trajectory-artifact \"{traj_path}\" --outcome success"
    )
    return {
        "recorded": True,
        "trajectory_json": str(traj_path),
        "sample_count": int(holder["payload"].get("sample_count", 0)),
        "goal_point_robot_m": goal,
        "demo_label": getattr(args, "demo_label", None),
        "record_action_command": record_command,
    }


def deleaf_phase_motion_profile(
    waypoint_name: str,
    args: argparse.Namespace,
    *,
    time_by_name: dict[str, float],
    path_hop_time_s: float,
) -> dict[str, float | None]:
    is_cut = waypoint_name == "cut"
    velocity_limit = (
        getattr(args, "cut_velocity_limit_rad_s", None)
        if is_cut
        else getattr(args, "right_velocity_limit_rad_s", None)
    )
    acceleration_limit = (
        getattr(args, "cut_acceleration_limit_rad_s2", None)
        if is_cut
        else getattr(args, "right_acceleration_limit_rad_s2", None)
    )
    # A missing cut override inherits the conservative global right-arm limit.
    if velocity_limit is None:
        velocity_limit = getattr(args, "right_velocity_limit_rad_s", None)
    if acceleration_limit is None:
        acceleration_limit = getattr(args, "right_acceleration_limit_rad_s2", None)
    return {
        "minimum_time_s": float(time_by_name.get(waypoint_name, path_hop_time_s)),
        "velocity_limit_rad_s": None if velocity_limit is None else float(velocity_limit),
        "acceleration_limit_rad_s2": (
            None if acceleration_limit is None else float(acceleration_limit)
        ),
    }


def execute_waypoints(robot: Any, home: dict[str, Any], solved: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    time_by_name = {
        "transit_lift": float(args.transit_lift_time_s),
        "safe_entry": float(args.safe_entry_time_s),
        "approach": float(args.approach_time_s),
        "pre_cut": float(args.pre_cut_time_s),
        "cut": float(args.cut_time_s),
        "post_cut_lift": float(args.post_cut_lift_time_s),
        "retract_lift": float(args.post_cut_lift_time_s),
        "return_transit": float(args.return_transit_time_s),
        "retract": float(args.retract_time_s),
    }
    executed: list[dict[str, Any]] = []
    # dense follow-cut waypoints (cut_path_NNN) get a uniform slow per-hop time; the joint-distance
    # resampling already bounds each hop, and the velocity/accel limits cap speed within it.
    path_hop_time = float(getattr(args, "path_waypoint_time_s", 0.8))
    for waypoint in solved:
        profile = deleaf_phase_motion_profile(
            waypoint["name"],
            args,
            time_by_name=time_by_name,
            path_hop_time_s=path_hop_time,
        )
        minimum_time = float(profile["minimum_time_s"])
        feedback = robot.send_command(
            command_arm_only(
                "right",
                np.asarray(waypoint["q_arm_rad"], dtype=float),
                minimum_time_s=minimum_time,
                velocity_limit_rad_s=profile["velocity_limit_rad_s"],
                acceleration_limit_rad_s2=profile["acceleration_limit_rad_s2"],
            ),
            int(max(float(args.timeout_s), minimum_time + 5.0)),
        ).get()
        finish_ok = feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
        executed.append(
            {
                "name": waypoint["name"],
                **profile,
                "finish_code": str(feedback.finish_code),
                "ok": bool(finish_ok),
            }
        )
        if not finish_ok:
            raise RuntimeError(f"robot command failed at {waypoint['name']}: {feedback.finish_code}")
        if waypoint["name"] == args.execute_until:
            break
    return executed


def waypoints_until(waypoints: list[DeleafWaypoint], stop_name: str) -> list[DeleafWaypoint]:
    selected: list[DeleafWaypoint] = []
    for waypoint in waypoints:
        selected.append(waypoint)
        if waypoint.name == stop_name:
            return selected
    raise ValueError(f"execute-until waypoint is not in plan: {stop_name}")


def solved_until(solved: list[dict[str, Any]], stop_name: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for waypoint in solved:
        selected.append(waypoint)
        if waypoint["name"] == stop_name:
            return selected
    return selected


def solved_gate_until(solved: list[dict[str, Any]], stop_name: str) -> bool:
    for waypoint in solved:
        if not waypoint["ok"]:
            return False
        if waypoint["name"] == stop_name:
            return True
    return False


def final_position_error_until(solved: list[dict[str, Any]], stop_name: str) -> float | None:
    selected = solved_until(solved, stop_name)
    if not selected:
        return None
    return selected[-1].get("ik", {}).get("position_error_m")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.run_label = args.run_label or "deleaf_click_" + time.strftime("%H%M%S")
    selected, capture_paths, execute_confirmed = select_cut_point(args)
    cut_point = np.asarray(selected["point_robot_m"], dtype=float).reshape(3)
    target_anchor_contract = selected_target_anchor_contract(args)
    transform_json = args.transform_json or default_transform_json(args.dataset_name)
    toolfit = args.toolfit or DEFAULT_RIGHT_TOOLFIT
    tool_axes = load_deleaf_tool_axes(toolfit)
    tool_offset_local = np.asarray(
        json.loads(Path(toolfit).read_text(encoding="utf-8")).get("estimated_tool_offset_m", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    stakes_path = args.stakes_json or (
        REPO_ROOT / "robot" / "calibration" / "session" / args.dataset_name / "stakes_robot.json"
    )
    stakes = load_stakes(stakes_path)
    guidance_selection: dict[str, Any] | None = None
    if args.mps_auto_select_action:
        # Per-cut guidance: each clicked petiole inherits the axes of the nearest
        # demonstrated human cut in the manifest (override: drop the flag and pass
        # --action-name to force one).
        guidance_library = build_deleaf_guidance_library(
            manifest_path=args.manifest,
            frame_index_path=args.frame_index,
            hand_tracking_path=args.hand_tracking,
            slam_trajectory_path=args.slam_trajectory,
            alignment_path=args.alignment or transform_json,
            hand_side=args.hand_side,
            hand_point=args.hand_point,
            min_confidence=args.mps_min_confidence,
            prefer_downward_cut_axis=args.prefer_downward_cut_axis,
            approach_path_points=int(args.mps_approach_path_points),
        )
        guidance, guidance_selection = select_guidance_for_point(guidance_library, cut_point)
    else:
        guidance = extract_deleaf_mps_guidance(
            manifest_path=args.manifest,
            action_name=args.action_name,
            frame_index_path=args.frame_index,
            hand_tracking_path=args.hand_tracking,
            slam_trajectory_path=args.slam_trajectory,
            alignment_path=args.alignment or transform_json,
            hand_side=args.hand_side,
            hand_point=args.hand_point,
            min_confidence=args.mps_min_confidence,
            prefer_downward_cut_axis=args.prefer_downward_cut_axis,
            approach_path_points=int(args.mps_approach_path_points),
        )
    teleop_prior_action_name = guidance.action_name if args.mps_auto_select_action else args.action_name
    teleop_prior = load_teleop_prior(
        args.teleop_prior_json,
        teleop_prior_action_name,
        args.teleop_prior_source_match,
        library_dir=args.teleop_prior_library_dir,
        cut_point_robot_m=cut_point,
    )
    successful_demo_seed_library = load_successful_demo_joint_seed_library(
        args.successful_demo_library,
        cut_point_robot_m=cut_point,
        dataset_name=args.dataset_name,
        max_count=int(args.successful_demo_seed_count),
    )
    if getattr(args, "urdf", None):
        # OFFLINE validation mode: load kinematics from a reference URDF (RB-Y1-a fw v1.1 =
        # submodules/rby1-sdk/models/rby1a/urdf/model_v1.1.urdf) instead of the live robot, so
        # dry-runs (no --execute) need no robot connection and can be fanned out in parallel.
        # The reference URDF may differ slightly from the firmware's; the LIVE plan+gate before
        # any motion remains the authoritative safety check. Refuses to execute offline.
        if args.execute:
            raise SystemExit("--urdf is offline validation only; remove --execute (or --urdf) to command the robot.")
        kinematics = RBY1Kinematics.from_urdf(
            urdf_path=Path(args.urdf),
            model_name=args.model,
            arm="right",
            toolfit_path=toolfit,
        )
    else:
        kinematics = RBY1Kinematics.from_live_robot(
            address=args.address,
            model_name=args.model,
            arm="right",
            toolfit_path=toolfit,
        )
    locked_state = kinematics.locked_state_from_home(args.home)
    home_tcp = kinematics.tcp_pose(locked_state.full_q_rad)[:3, 3]
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
    candidate_specs, pointcloud_inference, points_robot = deleaf_candidate_specs(
        args=args,
        cut_point_robot_m=cut_point,
        capture_paths=capture_paths,
        transform_json=transform_json,
    )
    azimuth_yaws = parse_float_list(getattr(args, "approach_azimuth_sweep_deg", "0")) or [0.0]
    adaptive_azimuth_top_k = max(0, int(getattr(args, "adaptive_approach_azimuth_top_k", 0)))
    nonzero_azimuth_yaws = [float(yaw) for yaw in azimuth_yaws if abs(float(yaw)) > 1e-6]
    base_candidate_count = len(candidate_specs)
    adaptive_azimuth_added = False
    adaptive_azimuth_finalized = False
    adaptive_azimuth_skipped_reason: str | None = None
    if nonzero_azimuth_yaws and adaptive_azimuth_top_k == 0:
        expanded_specs: list[dict[str, Any]] = []
        for spec in candidate_specs:
            for yaw in azimuth_yaws:
                spec_yaw = dict(spec)
                spec_yaw["orientation_yaw_deg"] = float(yaw)
                if abs(float(yaw)) > 1e-6:
                    spec_yaw["name"] = f"{spec['name']}_yaw{float(yaw):+.0f}"
                expanded_specs.append(spec_yaw)
        candidate_specs = expanded_specs
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    spec_index = 0
    while spec_index < len(candidate_specs):
        spec = candidate_specs[spec_index]
        spec_index += 1
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.orientation_yaw_deg = float(spec.get("orientation_yaw_deg", 0.0))
        candidate_args.contact_advance_m = float(spec["contact_advance_m"])
        candidate_args.target_extension_m = float(spec["target_extension_m"])
        axis = spec.get("contact_advance_axis_robot")
        candidate_args.contact_advance_axis_robot = None if axis is None else ",".join(f"{value:.9f}" for value in axis)
        waypoints, waypoint_diagnostics = build_deleaf_waypoints(
            cut_point_robot_m=cut_point,
            home_tcp_robot_m=home_tcp,
            guidance=guidance,
            teleop_prior=teleop_prior,
            tool_axes=tool_axes,
            args=candidate_args,
        )
        waypoint_diagnostics["successful_demo_joint_seed_library"] = successful_demo_seed_library
        solved, hard_ok = solve_waypoints(
            kinematics=kinematics,
            locked_state=locked_state,
            waypoints=waypoints,
            waypoint_diagnostics=waypoint_diagnostics,
            config=config,
            args=candidate_args,
            guidance=guidance,
        )
        execute_solved_only = solved_until(solved, args.execute_until)
        execute_hard_ok = solved_gate_until(solved, args.execute_until)
        # Foliage (pointcloud) gates: optionally restrict to the APPROACH sweep when FOLLOWING a dense
        # teleop cut (cut_path_NNN present). The followed draw deliberately moves the blade through the
        # canopy it is pruning (verified soft foliage in BD3: planar leaf sheet, not rigid), so a 20mm
        # blade-vs-cloud demand can never pass on the cut itself. The whole-path STAKE gate still covers
        # the known rigid stakes over EVERY segment, and the followed poses hit the human-demo targets to
        # ~1e-5 m, so we are provably reproducing the safe teleop draw. Inert unless a dense follow is
        # actually present -> a pure-IK cut is never loosened.
        following_dense = any(str(w.get("name", "")).startswith("cut_path_") for w in execute_solved_only)
        approach_only = bool(getattr(args, "pointcloud_gate_approach_only", False) and following_dense)
        if approach_only:
            approach_names = {"transit_lift", "safe_entry", "approach", "pre_cut"}
            pc_gate_solved = [w for w in execute_solved_only if str(w.get("name")) in approach_names]
            print(
                f"[pointcloud-gate-approach-only] foliage gates scoped to the approach sweep "
                f"({len(pc_gate_solved)} of {len(execute_solved_only)} waypoints); stake gate stays whole-path.",
                flush=True,
            )
        else:
            pc_gate_solved = execute_solved_only
        pc_clearance = pointcloud_path_clearance(
            solved=pc_gate_solved,
            kinematics=kinematics,
            locked_state=locked_state,
            points_robot_m=points_robot,
            cut_point_robot_m=cut_point,
            tool_offset_local_m=tool_offset_local,
            samples_per_segment=int(args.pointcloud_path_samples),
            exclusion_m=float(args.pointcloud_cut_exclusion_m),
        )
        pc_ok = pc_clearance is None or pc_clearance["minimum_distance_m"] >= float(args.pointcloud_min_clearance_m)
        blade_clearance = pointcloud_blade_clearance(
            solved=pc_gate_solved,
            kinematics=kinematics,
            locked_state=locked_state,
            points_robot_m=points_robot,
            cut_point_robot_m=cut_point,
            tool_offset_local_m=tool_offset_local,
            samples_per_segment=int(args.pointcloud_path_samples),
            exclusion_m=float(args.pointcloud_cut_exclusion_m),
        )
        blade_ok = blade_clearance is None or blade_clearance["minimum_distance_m"] >= float(args.pointcloud_blade_min_clearance_m)
        stake_clear = stake_clearance(
            solved=execute_solved_only,
            kinematics=kinematics,
            locked_state=locked_state,
            stakes=stakes,
            tool_offset_local_m=tool_offset_local,
            samples_per_segment=int(args.pointcloud_path_samples),
            blade_clearance_radius_m=float(getattr(args, "blade_clearance_radius_m", 0.0)),
        )
        stake_ok = stake_clear is None or stake_clear["minimum_distance_m"] >= float(args.stake_min_clearance_m)
        final_error = final_position_error_until(solved, args.execute_until)
        candidate = {
            "name": spec["name"],
            "ok": bool(execute_hard_ok and pc_ok and blade_ok and stake_ok),
            "hard_gate_ok": bool(execute_hard_ok),
            "full_hard_gate_ok": bool(hard_ok),
            "pointcloud_gate_ok": bool(pc_ok),
            "pointcloud_gate_scope": "approach_only" if approach_only else "full_path",
            "blade_gate_ok": bool(blade_ok),
            "blade_shaft_clearance": blade_clearance,
            "stake_gate_ok": bool(stake_ok),
            "stake_clearance": stake_clear,
            "contact_advance_m": float(spec["contact_advance_m"]),
            "contact_advance_axis_robot": spec.get("contact_advance_axis_robot"),
            "target_extension_m": float(spec["target_extension_m"]),
            "target_extension_delta_m": abs(float(spec["target_extension_m"]) - float(args.target_extension_m)),
            "minimum_self_path_clearance_m": min_solved_path_clearance(execute_solved_only),
            "pointcloud_path_clearance": pc_clearance,
            "final_position_error_m": None if final_error is None else float(final_error),
            "waypoint_diagnostics": waypoint_diagnostics,
            "waypoints": solved,
        }
        candidates.append(candidate)
        if best is None or candidate_sort_key(candidate) > candidate_sort_key(best):
            best = candidate
        if (
            spec_index == len(candidate_specs)
            and not adaptive_azimuth_finalized
            and nonzero_azimuth_yaws
            and adaptive_azimuth_top_k > 0
            and not any(item["ok"] for item in candidates)
        ):
            base_failures = candidates[:base_candidate_count]
            yaw_invariant_position_failure = bool(base_failures) and all(
                bool(candidate.get("waypoints"))
                and not bool(candidate["waypoints"][-1].get("orientation_required"))
                and float((candidate["waypoints"][-1].get("ik") or {}).get("position_error_m", 0.0))
                > float(args.position_tolerance_m)
                and abs(float(candidate.get("target_extension_m", 0.0))) <= 1e-9
                for candidate in base_failures
            )
            if yaw_invariant_position_failure:
                adaptive_azimuth_finalized = True
                adaptive_azimuth_skipped_reason = "yaw_invariant_position_unreachable"
                print(
                    "[adaptive-azimuth] skipped: every base candidate failed a position-only "
                    "waypoint, so wrist yaw cannot change reachability.",
                    flush=True,
                )
                continue
            ranked_base_indices = sorted(
                range(base_candidate_count),
                key=lambda index: candidate_sort_key(candidates[index]),
                reverse=True,
            )[: min(adaptive_azimuth_top_k, base_candidate_count)]
            for base_index in ranked_base_indices:
                base_spec = candidate_specs[base_index]
                for yaw in nonzero_azimuth_yaws:
                    spec_yaw = dict(base_spec)
                    spec_yaw["orientation_yaw_deg"] = yaw
                    spec_yaw["name"] = f"{base_spec['name']}_yaw{yaw:+.0f}"
                    candidate_specs.append(spec_yaw)
            adaptive_azimuth_added = True
            adaptive_azimuth_finalized = True
            print(
                f"[adaptive-azimuth] no base candidate passed; evaluating "
                f"{len(candidate_specs) - base_candidate_count} orientation fallbacks from "
                f"the safest {len(ranked_base_indices)} base path(s).",
                flush=True,
            )
    if best is None:
        raise RuntimeError("no deleaf candidates were generated")
    waypoint_diagnostics = best["waypoint_diagnostics"]
    solved = best["waypoints"]
    hard_ok = bool(best["ok"])
    should_execute = bool(args.execute and (args.no_execute_key_confirm or execute_confirmed))
    payload: dict[str, Any] = {
        "method": "deleaf_click_cut_mps_guided_ik",
        "dataset_name": args.dataset_name,
        "run_label": args.run_label,
        "selected_point": selected,
        "selected_target_anchor_contract": target_anchor_contract,
        "capture_artifacts": capture_paths,
        "transform_json": str(transform_json),
        "robot": {
            "address": args.address,
            "model": args.model,
            "arm": "right",
            "home": str(args.home),
            "toolfit": str(toolfit),
        },
        "mps_guidance": {
            "action_name": guidance.action_name,
            "hand_side": guidance.hand_side,
            "approach_axis_robot": guidance.approach_axis_robot.tolist(),
            "cut_axis_robot": guidance.cut_axis_robot.tolist(),
            "retract_axis_robot": guidance.retract_axis_robot.tolist(),
            "confidence": float(guidance.confidence),
            "cut_anchor_robot_m": None if guidance.cut_anchor_robot_m is None else np.asarray(guidance.cut_anchor_robot_m, dtype=float).reshape(3).tolist(),
            "approach_path_robot_m": None if guidance.approach_path_robot_m is None else np.asarray(guidance.approach_path_robot_m, dtype=float).reshape(-1, 3).tolist(),
            "diagnostics": guidance.diagnostics,
        },
        "mps_guidance_selection": guidance_selection,
        "waypoint_diagnostics": waypoint_diagnostics,
        "waypoints": solved,
        "selected_candidate": {
            key: value
            for key, value in best.items()
            if key not in {"waypoints", "waypoint_diagnostics"}
        },
        "candidate_summaries": [
            {key: value for key, value in candidate.items() if key not in {"waypoints", "waypoint_diagnostics"}}
            for candidate in candidates
        ],
        "pointcloud_model": {
            "enabled": bool(args.use_pointcloud_candidates),
            "inference": pointcloud_inference,
            "point_count": None if points_robot is None else int(points_robot.shape[0]),
            "min_clearance_m": float(args.pointcloud_min_clearance_m),
            "blade_min_clearance_m": float(args.pointcloud_blade_min_clearance_m),
            "blade_clearance_role": (
                "hard_gate" if float(args.pointcloud_blade_min_clearance_m) > 0.0 else "report_only_soft_foliage"
            ),
            "rigid_support_gate": "calibrated_stake_cylinders",
        },
        "planner_config": {
            "deleaf_config": None if args.deleaf_config is None else str(args.deleaf_config),
            "target_extension_values_m": args.target_extension_values_m,
            "teleop_prior_action_name": teleop_prior_action_name,
            "teleop_prior_selection": None if teleop_prior is None else teleop_prior.get("selection"),
            "successful_demo_seed_library": successful_demo_seed_library,
            "candidate_count": len(candidates),
            "orientation_search": {
                "azimuth_yaws_deg": [float(value) for value in azimuth_yaws],
                "adaptive_top_k": adaptive_azimuth_top_k,
                "adaptive_fallback_used": adaptive_azimuth_added,
                "adaptive_fallback_skipped_reason": adaptive_azimuth_skipped_reason,
                "base_candidate_count": base_candidate_count,
            },
        },
        "hard_gate": {"ok": bool(hard_ok)},
        "execute_requested": bool(args.execute),
        "execute_confirmed": execute_confirmed,
        "execute": bool(should_execute),
    }
    artifact = (
        Path(args.output_json)
        if args.output_json is not None
        else artifact_path("deleaf_click_cut", f"{args.dataset_name}_{args.run_label}", runs_root=args.runs_root)
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload["artifacts"] = {"json": str(artifact)}
    if not hard_ok:
        payload["motion"] = {"executed": False, "reason": "ik_or_clearance_gate_failed"}
        artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    if not should_execute:
        payload["motion"] = {"executed": False, "reason": "dry_run_or_execute_not_confirmed"}
        artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    home = json.loads(args.home.read_text(encoding="utf-8"))
    robot = connect_motion_robot(address=args.address, model=args.model, power=args.power, servo=args.servo)
    payload["home_check"] = validate_home_state(robot, home, tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)))
    video_handle: dict[str, Any] | None = None
    demo_handle: dict[str, Any] | None = None
    execution_error: Exception | None = None
    try:
        video_handle = start_trial_video(args)
        demo_handle = start_demo_capture(args, toolfit) if getattr(args, "collect_demo", False) else None
        payload["motion"] = {"executed": True, "waypoints": execute_waypoints(robot, home, solved, args)}
    except Exception as exc:
        execution_error = exc
        payload["motion"] = {"executed": False, "error": repr(exc)}
    finally:
        demo_info = finish_demo_capture(demo_handle, args, payload, cut_point)
        if demo_info is not None:
            payload["demo"] = demo_info
        video_info = finish_trial_video(video_handle)
        if video_info is not None:
            payload["trial_video"] = video_info
            payload["artifacts"]["video"] = video_info["video"]
            payload["artifacts"]["video_metadata"] = video_info["metadata"]
        artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if execution_error is not None:
        raise execution_error
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Click a RealSense cut point and run an MPS-guided right-arm deleafing IK action.")
    parser.add_argument("--deleaf-config", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--action-name", default="deleafing_01")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument(
        "--trial-video-output",
        type=Path,
        default=None,
        help="Record the physical execution from RealSense after planning and before motion.",
    )
    parser.add_argument("--trial-video-max-duration-s", type=float, default=180.0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--frame-index", type=Path, default=None)
    parser.add_argument("--hand-tracking", type=Path, default=None)
    parser.add_argument("--slam-trajectory", type=Path, default=DEFAULT_SLAM_TRAJECTORY)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--hand-side", choices=("auto", "left", "right"), default="right")
    parser.add_argument("--hand-point", choices=("wrist", "pinch"), default="wrist")
    parser.add_argument("--mps-min-confidence", type=float, default=0.2)
    parser.add_argument("--min-axis-confidence", type=float, default=0.15)
    parser.add_argument("--prefer-downward-cut-axis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mps-auto-select-action",
        action="store_true",
        help="Per-cut guidance: pick the manifest action whose demonstrated cut anchor is "
        "nearest the clicked petiole (instead of the fixed --action-name). Override by "
        "omitting this flag and passing --action-name.",
    )
    parser.add_argument(
        "--mps-approach-path-guidance",
        action="store_true",
        help="SEED-ONLY: warm-start the 'approach' waypoint IK by chaining solves along the "
        "human's demonstrated approach polyline (re-anchored to the clicked cut point). "
        "Lands the solver in a human-like joint basin WITHOUT changing executed waypoints, "
        "gates, or timing. Default off; when off the plan is byte-for-byte identical.",
    )
    parser.add_argument(
        "--mps-approach-path-points",
        type=int,
        default=4,
        help="Number of points to evenly resample along the demonstrated approach polyline "
        "for SEED-ONLY warm-starting (only used with --mps-approach-path-guidance).",
    )
    parser.add_argument("--waypoint-policy", choices=("teleop_offset", "mps_template", "mps_raw"), default="teleop_offset")
    parser.add_argument(
        "--task-orientation-policy",
        choices=("teleop_pre_cut", "mps_axes", "side_perpendicular"),
        default="teleop_pre_cut",
    )
    parser.add_argument("--stroke-axis-policy", choices=("mps", "downward_constrained", "world_down", "teleop_prior"), default="downward_constrained")
    parser.add_argument(
        "--vertical-cut",
        action="store_true",
        help="Canonical straight-down petiole cut: force the cut stroke to world -Z (vertical descent) and the blade vertical with a horizontal heading, keeping the wrist untwisted and avoiding lateral swing toward the rods/bed. Overrides stroke/orientation policy and sets target-extension to 0. Path/point-cloud/self clearance gates still validate it.",
    )
    parser.add_argument(
        "--force-approach-axis-robot",
        default=None,
        help="Horizontal heading (robot frame, e.g. '1,0,0') for the vertical-cut blade orientation. Only used with --vertical-cut; defaults to the horizontal component of the MPS approach axis.",
    )
    parser.add_argument("--max-stroke-tilt-deg", type=float, default=8.0)
    parser.add_argument(
        "--approach-azimuth-sweep-deg",
        default="0",
        help="Comma list of yaw angles (deg about world +Z) to post-rotate the blade orientation, "
        "swinging the shaft's horizontal heading to clear a stake while keeping the near-vertical "
        "descent and tip target fixed. Each yaw becomes an extra candidate; the stake-clearance-first "
        "ranking picks the clearing one (or fails closed if none clears). Default '0' = no sweep "
        "(unchanged behavior). Example for a stake-adjacent petiole: '0,-30,-60,-90,30,60,90'.",
    )
    parser.add_argument(
        "--adaptive-approach-azimuth-top-k",
        type=int,
        default=0,
        help="When positive, solve the zero-yaw position candidates first. If none passes all gates, "
        "apply the nonzero --approach-azimuth-sweep-deg values only to the safest K base paths. "
        "This preserves the cut point and stroke while avoiding the full position-by-orientation product.",
    )
    parser.add_argument("--teleop-prior-json", type=Path, default=None)
    parser.add_argument(
        "--teleop-prior-library-dir",
        type=Path,
        default=None,
        help="Directory of canonical successful task-relative teleop priors. The nearest cut-anchor "
        "prior supplies orientation/style only; the live selected point remains the IK target.",
    )
    parser.add_argument("--teleop-prior-source-match", default=None)
    parser.add_argument(
        "--successful-demo-library",
        type=Path,
        default=None,
        help="Curated policy_action_library root. Successful right-arm postures are used only as "
        "multi-start IK seeds for the newly selected target; recorded trajectories are never executed.",
    )
    parser.add_argument(
        "--successful-demo-seed-count",
        type=int,
        default=0,
        help="Maximum nearest successful postures to try after the normal chained IK seed fails.",
    )
    parser.add_argument("--use-teleop-joint-seeds", action="store_true")
    parser.add_argument(
        "--teleop-cut-depth-m",
        type=float,
        default=None,
        help="With --teleop-follow: cap the cut descent below the petiole engagement to this many "
        "meters (keeps the human draw DIRECTION but limits depth, so the blade cuts a few cm through "
        "instead of plunging the human's full follow-through). Default None = reproduce the full draw.",
    )
    parser.add_argument(
        "--teleop-follow-cut",
        action="store_true",
        help="With --teleop-follow + a prior carrying cut_path: IK-plan the approach (transit->approach->pre_cut) "
        "but FOLLOW the human's recorded dense pre_cut->cut->retract draw (re-anchored, IK per recorded pose, "
        "joint-distance spaced) instead of the synthetic cut/lifts/retract. The operator's hybrid; needs the prior "
        "rebuilt with --use-explicit-phases. Leaves the 4-pose follow (without this flag) untouched.",
    )
    parser.add_argument(
        "--path-waypoint-time-s",
        type=float,
        default=0.8,
        help="Per-hop minimum_time for the dense follow-cut waypoints (slow+safe; hops are joint-distance bounded).",
    )
    parser.add_argument(
        "--teleop-follow",
        action="store_true",
        help="Follow the teleop demo as IK TARGETS (with --waypoint-policy teleop_offset): keep the "
        "human per-phase positions (skip standoff overwrites), use each phase's OWN demonstrated "
        "orientation, seed AND regularize IK toward the human joint branch, and zero cut_through "
        "(the demo anchor is already the cut bottom). Fixes the fixed-orientation vertical-plunge failures.",
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument(
        "--post-selection-home",
        type=Path,
        default=None,
        help="After target selection, restore only the right arm to this execution home before IK and motion.",
    )
    parser.add_argument("--post-selection-home-minimum-time-s", type=float, default=10.0)
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
    parser.add_argument("--selected-point-robot-m", default=None)
    parser.add_argument("--axis-size-m", type=float, default=0.1)
    parser.add_argument("--surfel-python", type=Path, default=DEFAULT_SURFEL_PYTHON)
    parser.add_argument("--no-execute-key-confirm", action="store_true")
    parser.add_argument("--approach-standoff-m", type=float, default=0.10)
    parser.add_argument("--safe-entry-standoff-m", type=float, default=0.12)
    parser.add_argument("--safe-entry-axis-robot", default=None)
    parser.add_argument("--safe-entry-lift-m", type=float, default=0.14)
    parser.add_argument("--transit-lift-m", type=float, default=0.0)
    parser.add_argument("--pre-cut-lift-m", type=float, default=0.025)
    parser.add_argument("--pre-cut-standoff-m", type=float, default=None)
    parser.add_argument("--approach-stroke-standoff-m", type=float, default=None)
    parser.add_argument("--cut-depth-m", type=float, default=0.025)
    parser.add_argument("--target-extension-m", type=float, default=0.0)
    parser.add_argument("--target-extension-values-m", default=None)
    parser.add_argument(
        "--require-selected-point-cut-anchor",
        action="store_true",
        help="Fail closed unless every target-extension candidate is zero, guaranteeing that the "
        "selected 3D point remains the nominal cut anchor. Intended for interactive any-target trials.",
    )
    parser.add_argument("--contact-advance-m", type=float, default=0.0)
    parser.add_argument("--contact-advance-mode", choices=("target", "entry"), default="target")
    parser.add_argument("--contact-advance-axis-robot", default=None)
    parser.add_argument("--cut-through-m", type=float, default=0.0)
    parser.add_argument("--post-cut-lift-m", type=float, default=0.0)
    parser.add_argument("--return-transit-lift-m", type=float, default=0.0)
    parser.add_argument("--retract-standoff-m", type=float, default=0.12)
    parser.add_argument("--retract-lift-m", type=float, default=0.06)
    parser.add_argument("--use-pointcloud-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pointcloud-contact-advance-values-m", default="0.03,0.04,0.05")
    parser.add_argument("--pointcloud-candidate-angles-deg", default="0,-25,25")
    parser.add_argument("--pointcloud-side-radius-m", type=float, default=0.16)
    parser.add_argument("--pointcloud-side-z-window-m", type=float, default=0.18)
    parser.add_argument("--pointcloud-cut-exclusion-m", type=float, default=0.035)
    parser.add_argument("--pointcloud-min-clearance-m", type=float, default=0.0)
    parser.add_argument(
        "--pointcloud-blade-min-clearance-m",
        type=float,
        default=0.02,
        help="Minimum clearance (m) from the point cloud to the swept knife SHAFT (ee_right->tip), "
        "not just the tip. Catches a rod/structure brushing the blade body that the tip-only gate "
        "misses. Fail-closed: candidates below this are rejected and the candidate ranking prefers "
        "larger shaft clearance. Set 0 to report-only (no shaft gating).",
    )
    parser.add_argument(
        "--pointcloud-gate-approach-only",
        action="store_true",
        help="Restrict the pointcloud foliage gates (tip-path + blade-shaft) to the APPROACH sweep "
        "(transit_lift->approach->pre_cut) instead of the whole path. Use with --teleop-follow-cut: the "
        "followed dense draw deliberately moves the blade through the leaf canopy it is pruning, so a "
        "blade-vs-cloud clearance demand can never pass on the cut. The whole-path STAKE gate still guards "
        "the known rigid stakes over every segment. INERT unless a dense cut_path is actually being "
        "followed (a pure-IK cut is never loosened).",
    )
    parser.add_argument(
        "--stakes-json",
        type=Path,
        default=None,
        help="JSON of fixed support-stake cylinders in robot frame (default: "
        "robot/calibration/session/<dataset>/stakes_robot.json). Deterministic stake clearance "
        "independent of the noisy point cloud; the swept knife shaft is checked against each stake.",
    )
    parser.add_argument(
        "--stake-min-clearance-m",
        type=float,
        default=0.03,
        help="Minimum clearance (m) from the swept knife shaft to any fixed stake cylinder. "
        "Fail-closed: candidates below this are rejected and the candidate ranking prefers the "
        "stake-clearing approach angle. Set 0 to disable stake gating.",
    )
    parser.add_argument(
        "--blade-clearance-radius-m",
        type=float,
        default=0.0,
        help="Half-width (m) of the knife blade body perpendicular to its shaft, subtracted from "
        "the stake clearance so the gate measures BLADE-SURFACE to STAKE-SURFACE (the knife is ~5cm "
        "wide per its STL, not a centerline). Use ~0.02 with --stake-min-clearance-m ~0.015 for a real "
        "~1.5cm blade-to-stake gap. Default 0 = centerline (optimistic).",
    )
    parser.add_argument("--pointcloud-path-samples", type=int, default=8)
    parser.add_argument("--position-tolerance-m", type=float, default=0.012)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.20)
    parser.add_argument("--min-clearance-m", type=float, default=0.03)
    parser.add_argument("--path-samples", type=int, default=20)
    parser.add_argument("--orientation-weight", type=float, default=0.8)
    parser.add_argument("--position-weight", type=float, default=20.0)
    parser.add_argument("--posture-weight", type=float, default=0.03)
    parser.add_argument("--smoothness-weight", type=float, default=0.03)
    parser.add_argument("--clearance-weight", type=float, default=2.0)
    parser.add_argument("--clearance-margin-m", type=float, default=0.03)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=140)
    parser.add_argument("--min-x-m", type=float, default=0.25)
    parser.add_argument("--max-x-m", type=float, default=0.85)
    parser.add_argument("--min-y-m", type=float, default=-0.55)
    parser.add_argument("--max-y-m", type=float, default=0.55)
    parser.add_argument("--min-z-m", type=float, default=0.35)
    parser.add_argument("--max-z-m", type=float, default=1.25)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--collect-demo",
        action="store_true",
        help="On a SUCCESSFUL --execute cut, capture a robot-state trajectory in a background thread "
        "(reuses capture_robot_tcp_trajectory's 10Hz `_traj.json` format) and write it + a ready "
        "record-action command, so the cut becomes a training demo via the existing "
        "collect_policy_episode -> organize_policy_library -> train_action_chunk_policy flow. Opt-in; "
        "no effect on dry-runs or non-collect cuts.",
    )
    parser.add_argument("--demo-hz", type=float, default=10.0, help="Demo trajectory sample rate (10Hz matches the deleaf BC training).")
    parser.add_argument("--demo-label", default=None, help="Action label for the collected demo (e.g. AD2); used in the emitted record-action command.")
    parser.add_argument("--demo-duration-s", type=float, default=600.0, help="Safety cap on demo capture duration; the cut stops the capture early on completion.")
    parser.add_argument(
        "--execute-until",
        choices=(
            "transit_lift",
            "safe_entry",
            "approach",
            "pre_cut",
            "cut",
            "post_cut_lift",
            "retract_lift",
            "retract",
            "return_transit",
        ),
        default="retract",
    )
    parser.add_argument("--power", default=DEFAULT_POWER)
    parser.add_argument("--servo", default=DEFAULT_SERVO)
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--transit-lift-time-s", type=float, default=6.0)
    parser.add_argument("--safe-entry-time-s", type=float, default=6.0)
    parser.add_argument("--approach-time-s", type=float, default=6.0)
    parser.add_argument("--pre-cut-time-s", type=float, default=4.0)
    parser.add_argument("--cut-time-s", type=float, default=3.0)
    parser.add_argument("--post-cut-lift-time-s", type=float, default=3.0)
    parser.add_argument("--return-transit-time-s", type=float, default=6.0)
    parser.add_argument("--retract-time-s", type=float, default=6.0)
    parser.add_argument(
        "--right-velocity-limit-rad-s",
        type=float,
        default=None,
        help="Per-joint velocity cap (rad/s) SDK-enforced on every deterministic deleaf move (gentle, smooth, regardless of move size). ~0.25 matches the proven harvest gentle speed.",
    )
    parser.add_argument(
        "--right-acceleration-limit-rad-s2",
        type=float,
        default=None,
        help="Per-joint acceleration cap (rad/s^2) SDK-enforced on every deterministic deleaf move (softer ramps). ~0.1 is gentle.",
    )
    parser.add_argument(
        "--cut-velocity-limit-rad-s",
        type=float,
        default=None,
        help="Optional per-joint velocity cap used only for the pre_cut-to-cut stroke. Defaults to the global right-arm cap.",
    )
    parser.add_argument(
        "--cut-acceleration-limit-rad-s2",
        type=float,
        default=None,
        help="Optional per-joint acceleration cap used only for the pre_cut-to-cut stroke. Defaults to the global right-arm cap.",
    )
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="OFFLINE validation: load kinematics from this reference URDF instead of the live robot "
        "(RB-Y1-a fw v1.1 = submodules/rby1-sdk/models/rby1a/urdf/model_v1.1.urdf). Dry-run only (refuses "
        "--execute); lets dry-runs run with no robot. The live plan+gate before motion stays authoritative.",
    )
    probe_args, _ = parser.parse_known_args()
    if probe_args.deleaf_config is not None:
        parser.set_defaults(**load_deleaf_config_defaults(probe_args.deleaf_config, parser))
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.execute:
        print("dry_run: pass --execute and press e in the picker to command the robot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
