from __future__ import annotations

"""Offline ablation: IK+MPS (ours) vs Pure-MPS direct retarget, per demonstrated cut.

Two arms turn a demonstrated human cut into a robot deleaf action. Both use the SAME
robot alignment and the SAME IK solver; the ONLY difference is the TARGET fed to IK.

- Arm A (IK+MPS, ours): per-cut guidance from extract_deleaf_mps_guidance gives the
  axes + a demonstrated cut anchor (cut_anchor_robot_m). The anchor is used as the
  "click". The planner waypoints are built with build_deleaf_waypoints and solved with
  solve_waypoints (both reused verbatim from deleaf_click_cut). Metrics: hard-gate ok,
  per-waypoint position/orientation error, min self-clearance.

- Arm B (Pure MPS, direct retarget): the human RIGHT-hand pose trajectory over the
  action's approach+cut windows is copied directly. Position = hand_point_device mapped
  via mapper.point_to_robot. Orientation = human WRIST orientation (qx/qy/qz/qw_<side>_
  device_wrist) mapped to robot via the mapper rotation when present; otherwise a
  position-only fallback using the MPS cut axis (orientation_source='fallback_cut_axis').
  Each downsampled pose is solved with solve_tcp_pose_ik (chained seed) and metrics are
  aggregated per cut: fraction with valid IK, self-clearance violations, joint-limit +
  workspace violations, and the realized cut-point error vs the demonstrated anchor.

This module is OFFLINE only. It never commands the robot. RBY1Kinematics.from_live_robot
is used read-only for FK. The robot-frame alignment for the new recording is NOT
available yet, so the real run is gated; --self-test exercises the pose-extraction,
re-frame, and metric-aggregation math with synthetic inputs and never touches the robot
or the alignment.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.apps.deleaf_click_cut import (  # noqa: E402
    build_deleaf_waypoints,
    extract_deleaf_mps_guidance,
    hand_point_device,
    load_deleaf_tool_axes,
    solve_waypoints,
)
from robot.kinematics.clearance import nearest_self_clearances  # noqa: E402
from robot.kinematics.geometry import normalize, transform_from_rotation_translation  # noqa: E402
from robot.kinematics.hand_prior import (  # noqa: E402
    HandTrackingSample,
    RobotFrameMapper,
    TimeWindow,
    build_robot_frame_mapper,
    load_frame_timestamps,
    load_hand_tracking,
    load_slam_trajectory,
)
from robot.kinematics.ik_solver import solve_tcp_pose_ik  # noqa: E402
from robot.kinematics.rby1_model import RBY1Kinematics  # noqa: E402
from robot.kinematics.types import IKSolverConfig  # noqa: E402
from robot.kinematics.utils.realsense_click_move_arm import (  # noqa: E402
    DEFAULT_ADDRESS,
    DEFAULT_HOME,
    workspace_check,
)

REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_MANIFEST = REPO_ROOT / "robot" / "kinematics" / "runs" / "proposal_0d058720_manifest.json"
DEFAULT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_right_knife_tip_edge_deleaf_01.json"


# --------------------------------------------------------------------------------------
# Pure (robot/alignment-free) math -- unit tested by --self-test with synthetic inputs.
# --------------------------------------------------------------------------------------


def rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """3x3 rotation from an (x, y, z, w) quaternion (Aria/MPS wrist convention).

    Matches load_slam_trajectory's xyzw ordering. Returns identity-safe orthonormal R.
    """
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def wrist_quaternion_columns_present(fieldnames: list[str] | None, side: str) -> bool:
    """True iff the hand-tracking header carries a usable wrist quaternion for `side`."""
    if not fieldnames:
        return False
    required = {
        f"qx_{side}_device_wrist",
        f"qy_{side}_device_wrist",
        f"qz_{side}_device_wrist",
        f"qw_{side}_device_wrist",
    }
    return required.issubset(set(fieldnames))


def wrist_quaternion_device(row: dict[str, str], side: str) -> np.ndarray | None:
    """Read the (x, y, z, w) wrist quaternion from a hand-tracking row, or None."""
    keys = (
        f"qx_{side}_device_wrist",
        f"qy_{side}_device_wrist",
        f"qz_{side}_device_wrist",
        f"qw_{side}_device_wrist",
    )
    try:
        values = [float(row[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None
    quaternion = np.asarray(values, dtype=float)
    if float(np.linalg.norm(quaternion)) <= 1e-9:
        return None
    return quaternion


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (SVD) to a possibly non-orthonormal 3x3 matrix.

    The mapper rotation composes anchor/world/device rotations that are individually
    orthonormal, but accumulated float error can break assert_transform's tolerance;
    we project back to SO(3) so the IK target pose is a valid transform.
    """
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(matrix)
    proper = u @ vt
    if float(np.linalg.det(proper)) < 0.0:
        u[:, -1] *= -1.0
        proper = u @ vt
    return proper


def wrist_rotation_to_robot(mapper: RobotFrameMapper, timestamp_us: float, rotation_device: np.ndarray) -> np.ndarray:
    """Map a device-frame wrist rotation to the robot frame, column by column.

    Reuses mapper.vector_to_robot (which applies the full robot<-anchor<-world<-device
    rotation) on each basis column, then re-orthonormalizes to a proper rotation.
    """
    rotation = np.asarray(rotation_device, dtype=float).reshape(3, 3)
    columns = [mapper.vector_to_robot(timestamp_us, rotation[:, axis]) for axis in range(3)]
    return orthonormalize_rotation(np.column_stack(columns))


def downsample_indices(count: int, max_frames: int) -> list[int]:
    """Evenly spaced indices into [0, count) capped at max_frames (inclusive endpoints)."""
    if count <= 0:
        return []
    if max_frames <= 0 or count <= max_frames:
        return list(range(count))
    return sorted({int(round(value)) for value in np.linspace(0.0, float(count - 1), int(max_frames))})


def select_window_samples(
    samples: list[HandTrackingSample],
    windows: tuple[TimeWindow, ...],
    side: str,
    min_confidence: float,
) -> list[HandTrackingSample]:
    """Confident in-window samples for `side`, ordered by timestamp (approach+cut etc.).

    `windows` are the TimeWindows whose union defines the copy span (e.g. approach+cut).
    """
    def in_any_window(timestamp_us: float) -> bool:
        return any(window.start_us <= timestamp_us <= window.end_us for window in windows)

    confidence_of = (lambda sample: sample.left_confidence) if side == "left" else (lambda sample: sample.right_confidence)
    selected = [
        sample
        for sample in samples
        if in_any_window(sample.timestamp_us) and confidence_of(sample) >= float(min_confidence)
    ]
    return sorted(selected, key=lambda sample: sample.timestamp_us)


def aggregate_pose_frame_metrics(
    frames: list[dict[str, Any]],
    *,
    orientation_used: bool,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    cut_anchor_robot_m: np.ndarray | None,
) -> dict[str, Any]:
    """Aggregate per-frame Arm-B IK metrics into per-cut summary numbers.

    A frame is "valid IK" when position_error <= tol AND (when orientation is used)
    orientation_error <= tol. Also counts self-clearance, joint-limit and workspace
    violations, and reports the realized cut-point error: the position error of the
    frame whose TARGET position is closest to the demonstrated cut anchor.
    """
    total = len(frames)
    if total == 0:
        return {
            "frame_count": 0,
            "valid_ik_fraction": 0.0,
            "valid_ik_count": 0,
            "self_clearance_violations": 0,
            "joint_limit_violations": 0,
            "workspace_violations": 0,
            "mean_position_error_m": None,
            "mean_orientation_error_rad": None,
            "realized_cut_point_error_m": None,
        }

    def is_valid(frame: dict[str, Any]) -> bool:
        position_ok = float(frame["position_error_m"]) <= float(position_tolerance_m)
        orientation_ok = (not orientation_used) or float(frame["orientation_error_rad"]) <= float(orientation_tolerance_rad)
        return bool(position_ok and orientation_ok)

    valid_count = sum(1 for frame in frames if is_valid(frame))
    self_clearance_violations = sum(1 for frame in frames if not bool(frame["clearance_ok"]))
    joint_limit_violations = sum(1 for frame in frames if bool(frame["joint_limit_violation"]))
    workspace_violations = sum(1 for frame in frames if not bool(frame["workspace_ok"]))
    position_errors = [float(frame["position_error_m"]) for frame in frames]
    orientation_errors = [float(frame["orientation_error_rad"]) for frame in frames]

    realized_cut_point_error_m: float | None = None
    if cut_anchor_robot_m is not None:
        anchor = np.asarray(cut_anchor_robot_m, dtype=float).reshape(3)
        nearest_frame = min(
            frames,
            key=lambda frame: float(np.linalg.norm(np.asarray(frame["target_position_robot_m"], dtype=float) - anchor)),
        )
        realized_cut_point_error_m = float(nearest_frame["position_error_m"])

    return {
        "frame_count": int(total),
        "valid_ik_count": int(valid_count),
        "valid_ik_fraction": float(valid_count) / float(total),
        "self_clearance_violations": int(self_clearance_violations),
        "joint_limit_violations": int(joint_limit_violations),
        "workspace_violations": int(workspace_violations),
        "mean_position_error_m": float(np.mean(position_errors)),
        "mean_orientation_error_rad": float(np.mean(orientation_errors)),
        "realized_cut_point_error_m": realized_cut_point_error_m,
    }


def min_self_clearance_from_solved(solved: list[dict[str, Any]]) -> float | None:
    """Smallest per-waypoint self-clearance across a solved Arm-A waypoint list."""
    values = [
        float(waypoint["clearance"]["minimum_distance_m"])
        for waypoint in solved
        if waypoint.get("clearance", {}).get("minimum_distance_m") is not None
    ]
    if not values:
        return None
    return float(min(values))


def summarize_arm_a_solved(solved: list[dict[str, Any]], hard_ok: bool) -> dict[str, Any]:
    """Per-cut Arm-A row from a solved waypoint list (reused solve_waypoints output)."""
    position_errors = [float(waypoint["ik"]["position_error_m"]) for waypoint in solved if "ik" in waypoint]
    orientation_errors = [
        float(waypoint["ik"]["orientation_error_rad"])
        for waypoint in solved
        if waypoint.get("orientation_required") and "ik" in waypoint
    ]
    return {
        "hard_gate_ok": bool(hard_ok),
        "waypoint_count": int(len(solved)),
        "max_position_error_m": float(max(position_errors)) if position_errors else None,
        "max_orientation_error_rad": float(max(orientation_errors)) if orientation_errors else None,
        "min_self_clearance_m": min_self_clearance_from_solved(solved),
        "per_waypoint": [
            {
                "name": waypoint["name"],
                "ok": bool(waypoint["ok"]),
                "orientation_required": bool(waypoint["orientation_required"]),
                "position_error_m": float(waypoint["ik"]["position_error_m"]),
                "orientation_error_rad": float(waypoint["ik"]["orientation_error_rad"]),
                "min_self_clearance_m": waypoint.get("clearance", {}).get("minimum_distance_m"),
            }
            for waypoint in solved
            if "ik" in waypoint
        ],
    }


def aggregate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Arm A vs Arm B across all per-action rows (the honest comparison)."""
    arm_a = [row["arm_a"] for row in rows if row.get("arm_a") is not None]
    arm_b = [row["arm_b"] for row in rows if row.get("arm_b") is not None]

    def mean_or_none(values: list[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return float(np.mean(present)) if present else None

    arm_a_gate = [bool(item["hard_gate_ok"]) for item in arm_a]
    arm_a_summary = {
        "action_count": int(len(arm_a)),
        "gate_pass_rate": (float(sum(arm_a_gate)) / float(len(arm_a_gate))) if arm_a_gate else None,
        "mean_min_self_clearance_m": mean_or_none([item.get("min_self_clearance_m") for item in arm_a]),
        "mean_max_position_error_m": mean_or_none([item.get("max_position_error_m") for item in arm_a]),
        "mean_max_orientation_error_rad": mean_or_none([item.get("max_orientation_error_rad") for item in arm_a]),
    }
    arm_b_summary = {
        "action_count": int(len(arm_b)),
        "mean_valid_ik_fraction": mean_or_none([item.get("valid_ik_fraction") for item in arm_b]),
        "feasible_action_rate": (
            float(sum(1 for item in arm_b if float(item.get("valid_ik_fraction", 0.0)) >= 0.5)) / float(len(arm_b))
            if arm_b
            else None
        ),
        "mean_self_clearance_violations": mean_or_none([item.get("self_clearance_violations") for item in arm_b]),
        "mean_joint_limit_violations": mean_or_none([item.get("joint_limit_violations") for item in arm_b]),
        "mean_workspace_violations": mean_or_none([item.get("workspace_violations") for item in arm_b]),
        "mean_realized_cut_point_error_m": mean_or_none([item.get("realized_cut_point_error_m") for item in arm_b]),
    }
    return {"arm_a_ik_mps": arm_a_summary, "arm_b_pure_mps": arm_b_summary}


# --------------------------------------------------------------------------------------
# Planner argument bridge (reuse deleaf_click_cut's planner without re-declaring args).
# --------------------------------------------------------------------------------------


def build_planner_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the Namespace expected by build_deleaf_waypoints / solve_waypoints.

    The reused planner functions read ~50 fields off an argparse.Namespace. We declare
    the planner-facing subset with the SAME defaults as deleaf_click_cut.main() so the
    planner behaves identically, then override only the few fields the ablation pins
    (tolerances, address, model, home, toolfit, manifest, hand-point). Capture, picker,
    motion-timing and execute flags are intentionally omitted -- they are never reached
    on the offline planning path.
    """
    planner_args = _planner_parser().parse_args([])
    overrides = {
        "manifest": args.manifest,
        "hand_point": args.hand_point,
        "address": args.address,
        "model": args.model,
        "home": args.home,
        "toolfit": args.toolfit,
        "position_tolerance_m": args.position_tolerance_m,
        "orientation_tolerance_rad": args.orientation_tolerance_rad,
        "frame_index": args.frame_index,
        "hand_tracking": args.hand_tracking,
        "slam_trajectory": args.slam_trajectory,
        "alignment": args.alignment,
    }
    for key, value in overrides.items():
        if hasattr(planner_args, key) and value is not None:
            setattr(planner_args, key, value)
    return planner_args


def _planner_parser() -> argparse.ArgumentParser:
    """Declare the planner/IK/clearance/bounds/waypoint-geometry args the planner reads.

    Defaults mirror deleaf_click_cut.main() so build_deleaf_waypoints / solve_waypoints
    behave identically. The waypoint policy is fixed to 'mps_template' (the MPS+IK arm)
    and orientation policy to 'mps_axes' so no teleop prior is needed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--action-name", default="deleafing_01")
    parser.add_argument("--frame-index", type=Path, default=None)
    parser.add_argument("--hand-tracking", type=Path, default=None)
    parser.add_argument("--slam-trajectory", type=Path, default=None)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--hand-side", choices=("auto", "left", "right"), default="right")
    parser.add_argument("--hand-point", choices=("wrist", "pinch"), default="wrist")
    parser.add_argument("--mps-min-confidence", type=float, default=0.2)
    parser.add_argument("--min-axis-confidence", type=float, default=0.15)
    parser.add_argument("--prefer-downward-cut-axis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mps-approach-path-guidance", action="store_true")
    parser.add_argument("--mps-approach-path-points", type=int, default=4)
    parser.add_argument("--waypoint-policy", choices=("teleop_offset", "mps_template", "mps_raw"), default="mps_template")
    parser.add_argument(
        "--task-orientation-policy",
        choices=("teleop_pre_cut", "mps_axes", "side_perpendicular"),
        default="mps_axes",
    )
    parser.add_argument(
        "--stroke-axis-policy",
        choices=("mps", "downward_constrained", "world_down", "teleop_prior"),
        default="downward_constrained",
    )
    parser.add_argument("--vertical-cut", action="store_true")
    parser.add_argument("--force-approach-axis-robot", default=None)
    parser.add_argument("--max-stroke-tilt-deg", type=float, default=8.0)
    parser.add_argument("--teleop-prior-json", type=Path, default=None)
    parser.add_argument("--teleop-prior-source-match", default=None)
    parser.add_argument("--use-teleop-joint-seeds", action="store_true")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
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
    parser.add_argument("--contact-advance-m", type=float, default=0.0)
    parser.add_argument("--contact-advance-mode", choices=("target", "entry"), default="target")
    parser.add_argument("--contact-advance-axis-robot", default=None)
    parser.add_argument("--cut-through-m", type=float, default=0.0)
    parser.add_argument("--post-cut-lift-m", type=float, default=0.0)
    parser.add_argument("--return-transit-lift-m", type=float, default=0.0)
    parser.add_argument("--retract-standoff-m", type=float, default=0.12)
    parser.add_argument("--retract-lift-m", type=float, default=0.06)
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
    parser.add_argument("--execute-until", default="retract")
    return parser


def bounds_from_planner_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return {
        "x": (float(args.min_x_m), float(args.max_x_m)),
        "y": (float(args.min_y_m), float(args.max_y_m)),
        "z": (float(args.min_z_m), float(args.max_z_m)),
    }


# --------------------------------------------------------------------------------------
# Arm B: pure-MPS direct retarget (per-frame solve_tcp_pose_ik, chained seed).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseTarget:
    timestamp_us: float
    position_robot_m: np.ndarray
    rotation_robot: np.ndarray | None  # None => position-only target


def build_pure_mps_pose_targets(
    *,
    samples: list[HandTrackingSample],
    windows: tuple[TimeWindow, ...],
    side: str,
    hand_point: str,
    mapper: RobotFrameMapper,
    min_confidence: float,
    use_wrist_orientation: bool,
    cut_axis_robot: np.ndarray,
    max_frames: int,
) -> tuple[list[PoseTarget], str]:
    """Turn the human hand trajectory (approach+cut) into robot-frame IK pose targets.

    Position is always the re-framed hand point. Orientation is the re-framed wrist
    rotation when use_wrist_orientation is True and the row carries a usable quaternion;
    otherwise it is built from the MPS cut axis (orientation_source='fallback_cut_axis'),
    which only constrains the blade descent direction.
    """
    window_samples = select_window_samples(samples, windows, side, min_confidence)
    indices = downsample_indices(len(window_samples), max_frames)
    selected = [window_samples[index] for index in indices]

    orientation_source = "wrist_quaternion" if use_wrist_orientation else "fallback_cut_axis"
    fallback_rotation = cut_axis_orientation(cut_axis_robot)

    targets: list[PoseTarget] = []
    used_wrist_any = False
    for sample in selected:
        position = np.asarray(
            mapper.point_to_robot(sample.timestamp_us, hand_point_device(sample.row, side, hand_point)),
            dtype=float,
        ).reshape(3)
        rotation: np.ndarray | None
        if use_wrist_orientation:
            quaternion = wrist_quaternion_device(sample.row, side)
            if quaternion is None:
                rotation = fallback_rotation
            else:
                rotation = wrist_rotation_to_robot(
                    mapper, sample.timestamp_us, rotation_from_quaternion_xyzw(quaternion)
                )
                used_wrist_any = True
        else:
            rotation = fallback_rotation
        targets.append(PoseTarget(timestamp_us=float(sample.timestamp_us), position_robot_m=position, rotation_robot=rotation))

    if use_wrist_orientation and not used_wrist_any:
        orientation_source = "fallback_cut_axis"
    return targets, orientation_source


def cut_axis_orientation(cut_axis_robot: np.ndarray) -> np.ndarray:
    """A proper rotation whose tool z-axis follows the (downward) cut stroke axis.

    Used as the orientation seed for the position-only fallback. Mirrors deleaf_rotation's
    convention loosely: z-axis = stroke direction; the lateral axes are an arbitrary but
    consistent orthonormal completion.
    """
    z_axis = normalize(np.asarray(cut_axis_robot, dtype=float).reshape(3))
    seed = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(seed, z_axis))) > 0.9:
        seed = np.asarray([0.0, 1.0, 0.0], dtype=float)
    x_axis = normalize(seed - float(np.dot(seed, z_axis)) * z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def solve_pure_mps_frames(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    targets: list[PoseTarget],
    orientation_used: bool,
    fallback_rotation: np.ndarray,
    config: IKSolverConfig,
    planner_args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Solve each pose target with solve_tcp_pose_ik (chained seed) and collect metrics."""
    home_q_arm = np.asarray(locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)], dtype=float)
    previous_q = home_q_arm.copy()
    lower, upper = kinematics.active_joint_limits(locked_state.full_q_rad)
    bounds = bounds_from_planner_args(planner_args)
    pose_config = config if orientation_used else _drop_orientation_weight(config)

    frames: list[dict[str, Any]] = []
    for target in targets:
        rotation = target.rotation_robot if target.rotation_robot is not None else fallback_rotation
        target_pose = transform_from_rotation_translation(orthonormalize_rotation(rotation), target.position_robot_m)
        result = solve_tcp_pose_ik(
            kinematics,
            locked_state,
            target_pose,
            seed_q_arm_rad=previous_q,
            reference_q_arm_rad=previous_q,
            config=pose_config,
        )
        q_full = kinematics.q_full_from_arm(locked_state, result.q_arm_rad)
        clearance = nearest_self_clearances(
            kinematics,
            q_full,
            pair_count=int(planner_args.clearance_pair_count),
            margin_m=float(planner_args.min_clearance_m),
        )
        workspace = workspace_check(target.position_robot_m, bounds)
        joint_limit_violation = bool(
            np.any(result.q_arm_rad < lower - 1e-6) or np.any(result.q_arm_rad > upper + 1e-6)
        )
        frames.append(
            {
                "timestamp_us": float(target.timestamp_us),
                "target_position_robot_m": np.asarray(target.position_robot_m, dtype=float).reshape(3).tolist(),
                "position_error_m": float(result.position_error_m),
                "orientation_error_rad": float(result.orientation_error_rad),
                "ik_ok": bool(result.ok),
                "clearance_ok": bool(clearance.ok),
                "min_self_clearance_m": clearance.minimum_distance_m,
                "workspace_ok": bool(workspace["ok"]),
                "joint_limit_violation": joint_limit_violation,
            }
        )
        previous_q = np.asarray(result.q_arm_rad, dtype=float)
    return frames


def _drop_orientation_weight(config: IKSolverConfig) -> IKSolverConfig:
    from dataclasses import replace

    return replace(config, orientation_weight=0.0)


def windows_for_action(manifest: dict[str, Any], action_name: str, names: tuple[str, ...]) -> tuple[TimeWindow, ...]:
    """Build TimeWindows (timestamps_us) for the named phases of one manifest action."""
    units = str(manifest.get("window_units", "frames"))
    if units != "timestamps_us":
        raise ValueError(f"ablation requires window_units='timestamps_us', got {units!r}")
    action = {str(item["name"]): item for item in manifest["actions"]}[action_name]
    return tuple(
        TimeWindow(name=name, start_us=float(action[name][0]), end_us=float(action[name][1]))
        for name in names
    )


# --------------------------------------------------------------------------------------
# Real run (gated on the alignment being available).
# --------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.alignment is None:
        raise SystemExit(
            "alignment (T_robot_from_aria) is required for the real run and is not available "
            "yet for this recording. Use --self-test to validate the math offline."
        )
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    hand_tracking_path = args.hand_tracking or Path(manifest["mps_hand_tracking_csv"])
    slam_trajectory_path = args.slam_trajectory or Path(manifest["slam_trajectory_csv"])
    side = "right"

    samples = load_hand_tracking(hand_tracking_path)
    fieldnames = list(samples[0].row.keys()) if samples else None
    use_wrist_orientation = wrist_quaternion_columns_present(fieldnames, side)

    frame_index = args.frame_index
    frame_timestamps_us = load_frame_timestamps(frame_index) if frame_index is not None else None
    mapper = build_robot_frame_mapper(args.alignment, frame_timestamps_us, load_slam_trajectory(slam_trajectory_path))

    toolfit = args.toolfit or DEFAULT_TOOLFIT
    tool_axes = load_deleaf_tool_axes(toolfit)
    kinematics = RBY1Kinematics.from_live_robot(address=args.address, model_name=args.model, arm=side, toolfit_path=toolfit)
    locked_state = kinematics.locked_state_from_home(args.home)
    home_tcp = kinematics.tcp_pose(locked_state.full_q_rad)[:3, 3]

    planner_args = build_planner_args(args)
    config = IKSolverConfig(
        position_weight=float(planner_args.position_weight),
        orientation_weight=float(planner_args.orientation_weight),
        posture_weight=float(planner_args.posture_weight),
        smoothness_weight=float(planner_args.smoothness_weight),
        clearance_weight=float(planner_args.clearance_weight),
        clearance_margin_m=float(planner_args.clearance_margin_m),
        clearance_pair_count=int(planner_args.clearance_pair_count),
        max_iterations=int(planner_args.max_iterations),
    )

    rows: list[dict[str, Any]] = []
    for action in manifest["actions"]:
        action_name = str(action["name"])
        row = evaluate_action(
            action_name=action_name,
            manifest=manifest,
            manifest_path=args.manifest,
            samples=samples,
            mapper=mapper,
            kinematics=kinematics,
            locked_state=locked_state,
            home_tcp=home_tcp,
            tool_axes=tool_axes,
            config=config,
            planner_args=planner_args,
            args=args,
            side=side,
            use_wrist_orientation=use_wrist_orientation,
            hand_tracking_path=hand_tracking_path,
            slam_trajectory_path=slam_trajectory_path,
        )
        rows.append(row)

    payload = {
        "method": "ablate_deleaf_ik_vs_mps",
        "manifest": str(args.manifest),
        "alignment": str(args.alignment),
        "hand_tracking": str(hand_tracking_path),
        "slam_trajectory": str(slam_trajectory_path),
        "toolfit": str(toolfit),
        "hand_point": args.hand_point,
        "side": side,
        "orientation_source": "wrist_quaternion" if use_wrist_orientation else "fallback_cut_axis",
        "position_tolerance_m": float(args.position_tolerance_m),
        "orientation_tolerance_rad": float(args.orientation_tolerance_rad),
        "rows": rows,
        "comparison": aggregate_comparison(rows),
    }
    if args.output_json is not None:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["output_json"] = str(args.output_json)
    print_comparison_table(payload)
    return payload


def evaluate_action(
    *,
    action_name: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    samples: list[HandTrackingSample],
    mapper: RobotFrameMapper,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    home_tcp: np.ndarray,
    tool_axes: dict[str, Any],
    config: IKSolverConfig,
    planner_args: argparse.Namespace,
    args: argparse.Namespace,
    side: str,
    use_wrist_orientation: bool,
    hand_tracking_path: Path,
    slam_trajectory_path: Path,
) -> dict[str, Any]:
    """Evaluate one cut on both arms (same alignment, same IK solver, different target)."""
    # Arm A: IK + MPS guidance, cut_anchor as the click, reused planner + solver.
    guidance = extract_deleaf_mps_guidance(
        manifest_path=manifest_path,
        action_name=action_name,
        frame_index_path=args.frame_index,
        hand_tracking_path=hand_tracking_path,
        slam_trajectory_path=slam_trajectory_path,
        alignment_path=args.alignment,
        hand_side=side,
        hand_point=args.hand_point,
        min_confidence=float(planner_args.mps_min_confidence),
        prefer_downward_cut_axis=bool(planner_args.prefer_downward_cut_axis),
        approach_path_points=int(planner_args.mps_approach_path_points),
    )
    arm_a: dict[str, Any] | None = None
    arm_a_error: str | None = None
    if guidance.cut_anchor_robot_m is None:
        arm_a_error = "no_cut_anchor"
    else:
        cut_point = np.asarray(guidance.cut_anchor_robot_m, dtype=float).reshape(3)
        waypoints, waypoint_diagnostics = build_deleaf_waypoints(
            cut_point_robot_m=cut_point,
            home_tcp_robot_m=home_tcp,
            guidance=guidance,
            teleop_prior=None,
            tool_axes=tool_axes,
            args=planner_args,
        )
        solved, hard_ok = solve_waypoints(
            kinematics=kinematics,
            locked_state=locked_state,
            waypoints=waypoints,
            waypoint_diagnostics=waypoint_diagnostics,
            config=config,
            args=planner_args,
            guidance=guidance,
        )
        arm_a = summarize_arm_a_solved(solved, hard_ok)

    # Arm B: pure-MPS direct retarget over approach+cut, per-frame solve_tcp_pose_ik.
    windows = windows_for_action(manifest, action_name, ("approach", "cut"))
    fallback_rotation = cut_axis_orientation(guidance.cut_axis_robot)
    targets, orientation_source = build_pure_mps_pose_targets(
        samples=samples,
        windows=windows,
        side=side,
        hand_point=args.hand_point,
        mapper=mapper,
        min_confidence=float(planner_args.mps_min_confidence),
        use_wrist_orientation=use_wrist_orientation,
        cut_axis_robot=guidance.cut_axis_robot,
        max_frames=int(args.max_frames),
    )
    orientation_used = orientation_source == "wrist_quaternion"
    frames = solve_pure_mps_frames(
        kinematics=kinematics,
        locked_state=locked_state,
        targets=targets,
        orientation_used=orientation_used,
        fallback_rotation=fallback_rotation,
        config=config,
        planner_args=planner_args,
    )
    arm_b = aggregate_pose_frame_metrics(
        frames,
        orientation_used=orientation_used,
        position_tolerance_m=float(args.position_tolerance_m),
        orientation_tolerance_rad=float(args.orientation_tolerance_rad),
        cut_anchor_robot_m=guidance.cut_anchor_robot_m,
    )
    arm_b["orientation_source"] = orientation_source

    return {
        "action_name": action_name,
        "arm_a": arm_a,
        "arm_a_error": arm_a_error,
        "arm_b": arm_b,
    }


def print_comparison_table(payload: dict[str, Any]) -> None:
    """Compact per-action + aggregate table (Arm A vs Arm B)."""
    print(f"# ablate_deleaf_ik_vs_mps  manifest={payload['manifest']}")
    print(f"# orientation_source={payload['orientation_source']}  "
          f"pos_tol={payload['position_tolerance_m']}m  ori_tol={payload['orientation_tolerance_rad']}rad")
    header = f"{'action':<14} {'A:gate':>7} {'A:maxPos':>9} {'A:minClr':>9} {'B:validIK':>9} {'B:clrViol':>9} {'B:cutErr':>9}"
    print(header)
    print("-" * len(header))
    for row in payload["rows"]:
        arm_a = row.get("arm_a") or {}
        arm_b = row.get("arm_b") or {}
        gate = "PASS" if arm_a.get("hard_gate_ok") else ("-" if not arm_a else "FAIL")
        max_pos = arm_a.get("max_position_error_m")
        min_clr = arm_a.get("min_self_clearance_m")
        valid_ik = arm_b.get("valid_ik_fraction")
        clr_viol = arm_b.get("self_clearance_violations")
        cut_err = arm_b.get("realized_cut_point_error_m")
        print(
            f"{row['action_name']:<14} {gate:>7} "
            f"{_fmt(max_pos):>9} {_fmt(min_clr):>9} "
            f"{_fmt(valid_ik):>9} {clr_viol if clr_viol is not None else '-':>9} {_fmt(cut_err):>9}"
        )
    comparison = payload["comparison"]
    arm_a = comparison["arm_a_ik_mps"]
    arm_b = comparison["arm_b_pure_mps"]
    print("-" * len(header))
    print("AGGREGATE")
    print(f"  Arm A (IK+MPS):  gate_pass_rate={_fmt(arm_a['gate_pass_rate'])}  "
          f"mean_min_clearance={_fmt(arm_a['mean_min_self_clearance_m'])}  "
          f"mean_max_pos_err={_fmt(arm_a['mean_max_position_error_m'])}")
    print(f"  Arm B (pure MPS): feasible_rate={_fmt(arm_b['feasible_action_rate'])}  "
          f"mean_valid_ik={_fmt(arm_b['mean_valid_ik_fraction'])}  "
          f"mean_clr_viol={_fmt(arm_b['mean_self_clearance_violations'])}  "
          f"mean_cut_err={_fmt(arm_b['mean_realized_cut_point_error_m'])}")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


# --------------------------------------------------------------------------------------
# Self-test (no robot, no alignment): synthetic pose-extraction + re-frame + aggregation.
# --------------------------------------------------------------------------------------


def _identity_mapper() -> RobotFrameMapper:
    """A RobotFrameMapper whose transforms are identity (robot frame == device frame).

    Lets the self-test exercise point_to_robot / vector_to_robot wiring with a single
    pose sample, without any alignment file or SLAM trajectory on disk.
    """
    from robot.kinematics.hand_prior import DevicePoseSample, PoseLookup

    pose = DevicePoseSample(timestamp_us=0.0, T_world_from_device=np.eye(4, dtype=float))
    return RobotFrameMapper(
        T_robot_from_anchor=np.eye(4, dtype=float),
        T_anchor_from_world=np.eye(4, dtype=float),
        pose_lookup=PoseLookup([pose]),
    )


def _synthetic_row(side: str, position: np.ndarray, quaternion: np.ndarray | None) -> dict[str, str]:
    row = {
        f"tx_{side}_device_wrist": str(float(position[0])),
        f"ty_{side}_device_wrist": str(float(position[1])),
        f"tz_{side}_device_wrist": str(float(position[2])),
    }
    if quaternion is not None:
        row.update(
            {
                f"qx_{side}_device_wrist": str(float(quaternion[0])),
                f"qy_{side}_device_wrist": str(float(quaternion[1])),
                f"qz_{side}_device_wrist": str(float(quaternion[2])),
                f"qw_{side}_device_wrist": str(float(quaternion[3])),
            }
        )
    return row


def self_test() -> bool:
    """Unit-test the pure math with synthetic inputs. Prints PASS/FAIL per check."""
    checks: list[tuple[str, bool]] = []

    # 1. Quaternion -> rotation: identity quaternion -> identity rotation.
    identity = rotation_from_quaternion_xyzw(np.asarray([0.0, 0.0, 0.0, 1.0]))
    checks.append(("quat_identity", bool(np.allclose(identity, np.eye(3), atol=1e-9))))

    # 2. Quaternion -> rotation: 90deg about z maps +x -> +y.
    half = np.sqrt(0.5)
    rot_z90 = rotation_from_quaternion_xyzw(np.asarray([0.0, 0.0, half, half]))
    mapped_x = rot_z90 @ np.asarray([1.0, 0.0, 0.0])
    checks.append(("quat_z90_maps_x_to_y", bool(np.allclose(mapped_x, [0.0, 1.0, 0.0], atol=1e-6))))
    checks.append(("quat_proper_rotation", bool(abs(float(np.linalg.det(rot_z90)) - 1.0) < 1e-9)))

    # 3. orthonormalize_rotation projects a perturbed matrix back to SO(3).
    perturbed = rot_z90 + 1e-3 * np.ones((3, 3))
    fixed = orthonormalize_rotation(perturbed)
    checks.append(("orthonormalize_so3", bool(np.allclose(fixed.T @ fixed, np.eye(3), atol=1e-9))))
    checks.append(("orthonormalize_det", bool(abs(float(np.linalg.det(fixed)) - 1.0) < 1e-9)))

    # 4. wrist quaternion column presence.
    header = [
        "qx_right_device_wrist", "qy_right_device_wrist",
        "qz_right_device_wrist", "qw_right_device_wrist", "tx_right_device_wrist",
    ]
    checks.append(("wrist_present_true", wrist_quaternion_columns_present(header, "right")))
    checks.append(("wrist_present_false_left", not wrist_quaternion_columns_present(header, "left")))
    checks.append(("wrist_present_none", not wrist_quaternion_columns_present(None, "right")))

    # 5. downsample_indices: cap and endpoint inclusion.
    checks.append(("downsample_under_cap", downsample_indices(3, 8) == [0, 1, 2]))
    capped = downsample_indices(100, 5)
    checks.append(("downsample_cap_len", len(capped) == 5))
    checks.append(("downsample_endpoints", capped[0] == 0 and capped[-1] == 99))
    checks.append(("downsample_empty", downsample_indices(0, 5) == []))

    # 6. select_window_samples: confidence gate + window membership + ordering.
    mapper = _identity_mapper()
    side = "right"
    samples = [
        HandTrackingSample(timestamp_us=50.0, left_confidence=0.0, right_confidence=0.9,
                           row=_synthetic_row(side, np.asarray([0.0, 0.0, 0.0]), None)),  # out of window
        HandTrackingSample(timestamp_us=150.0, left_confidence=0.0, right_confidence=0.05,
                           row=_synthetic_row(side, np.asarray([0.1, 0.0, 0.0]), None)),  # low confidence
        HandTrackingSample(timestamp_us=120.0, left_confidence=0.0, right_confidence=0.9,
                           row=_synthetic_row(side, np.asarray([0.2, 0.0, 0.0]), None)),  # keep
        HandTrackingSample(timestamp_us=110.0, left_confidence=0.0, right_confidence=0.9,
                           row=_synthetic_row(side, np.asarray([0.3, 0.0, 0.0]), None)),  # keep
    ]
    windows = (TimeWindow(name="approach", start_us=100.0, end_us=130.0),)
    selected = select_window_samples(samples, windows, side, 0.2)
    checks.append(("window_filter_count", len(selected) == 2))
    checks.append(("window_filter_order", [s.timestamp_us for s in selected] == [110.0, 120.0]))

    # 7. build_pure_mps_pose_targets: position re-frame correctness + orientation source.
    pos_samples = [
        HandTrackingSample(timestamp_us=105.0, left_confidence=0.0, right_confidence=0.9,
                           row=_synthetic_row(side, np.asarray([0.4, 0.5, 0.6]),
                                              np.asarray([0.0, 0.0, half, half]))),
        HandTrackingSample(timestamp_us=125.0, left_confidence=0.0, right_confidence=0.9,
                           row=_synthetic_row(side, np.asarray([0.7, 0.8, 0.9]),
                                              np.asarray([0.0, 0.0, half, half]))),
    ]
    targets_wrist, src_wrist = build_pure_mps_pose_targets(
        samples=pos_samples, windows=windows, side=side, hand_point="wrist", mapper=mapper,
        min_confidence=0.2, use_wrist_orientation=True,
        cut_axis_robot=np.asarray([0.0, 0.0, -1.0]), max_frames=10,
    )
    checks.append(("targets_count", len(targets_wrist) == 2))
    # identity mapper => robot position == device wrist position.
    checks.append(("target_position_reframe",
                   bool(np.allclose(targets_wrist[0].position_robot_m, [0.4, 0.5, 0.6], atol=1e-9))))
    checks.append(("orientation_source_wrist", src_wrist == "wrist_quaternion"))
    # wrist rotation present => not None, and proper rotation.
    checks.append(("target_rotation_present", targets_wrist[0].rotation_robot is not None))
    checks.append(("target_rotation_proper",
                   bool(abs(float(np.linalg.det(targets_wrist[0].rotation_robot)) - 1.0) < 1e-6)))
    # identity mapper => robot wrist rotation == device wrist rotation (z90).
    checks.append(("target_rotation_reframe",
                   bool(np.allclose(targets_wrist[0].rotation_robot, rot_z90, atol=1e-6))))

    # 7b. fallback when wrist orientation disabled.
    targets_fb, src_fb = build_pure_mps_pose_targets(
        samples=pos_samples, windows=windows, side=side, hand_point="wrist", mapper=mapper,
        min_confidence=0.2, use_wrist_orientation=False,
        cut_axis_robot=np.asarray([0.0, 0.0, -1.0]), max_frames=10,
    )
    checks.append(("fallback_source", src_fb == "fallback_cut_axis"))
    # fallback rotation z-axis follows the cut stroke axis.
    checks.append(("fallback_z_axis_is_cut_axis",
                   bool(np.allclose(targets_fb[0].rotation_robot[:, 2], [0.0, 0.0, -1.0], atol=1e-9))))

    # 7c. fallback when wrist requested but quaternion absent in rows.
    no_quat_samples = [
        HandTrackingSample(timestamp_us=105.0, left_confidence=0.0, right_confidence=0.9,
                           row=_synthetic_row(side, np.asarray([0.4, 0.5, 0.6]), None)),
    ]
    _, src_missing = build_pure_mps_pose_targets(
        samples=no_quat_samples, windows=windows, side=side, hand_point="wrist", mapper=mapper,
        min_confidence=0.2, use_wrist_orientation=True,
        cut_axis_robot=np.asarray([0.0, 0.0, -1.0]), max_frames=10,
    )
    checks.append(("missing_quat_falls_back", src_missing == "fallback_cut_axis"))

    # 8. cut_axis_orientation: proper rotation with z == axis.
    ori = cut_axis_orientation(np.asarray([0.0, 0.0, -1.0]))
    checks.append(("cut_axis_ori_proper", bool(abs(float(np.linalg.det(ori)) - 1.0) < 1e-9)))
    checks.append(("cut_axis_ori_z", bool(np.allclose(ori[:, 2], [0.0, 0.0, -1.0], atol=1e-9))))

    # 9. aggregate_pose_frame_metrics: known synthetic frames -> exact fractions.
    frames = [
        {"target_position_robot_m": [0.5, 0.0, 0.6], "position_error_m": 0.005, "orientation_error_rad": 0.1,
         "ik_ok": True, "clearance_ok": True, "min_self_clearance_m": 0.05, "workspace_ok": True,
         "joint_limit_violation": False},
        {"target_position_robot_m": [0.5, 0.1, 0.6], "position_error_m": 0.05, "orientation_error_rad": 0.5,
         "ik_ok": True, "clearance_ok": False, "min_self_clearance_m": 0.01, "workspace_ok": False,
         "joint_limit_violation": True},
        {"target_position_robot_m": [0.5, 0.2, 0.6], "position_error_m": 0.008, "orientation_error_rad": 0.15,
         "ik_ok": True, "clearance_ok": True, "min_self_clearance_m": 0.04, "workspace_ok": True,
         "joint_limit_violation": False},
    ]
    agg = aggregate_pose_frame_metrics(
        frames, orientation_used=True, position_tolerance_m=0.012, orientation_tolerance_rad=0.2,
        cut_anchor_robot_m=np.asarray([0.5, 0.0, 0.6]),
    )
    # frames 0 and 2 are valid (pos<=0.012 AND ori<=0.2); frame 1 fails both.
    checks.append(("agg_valid_count", agg["valid_ik_count"] == 2))
    checks.append(("agg_valid_fraction", abs(agg["valid_ik_fraction"] - 2.0 / 3.0) < 1e-9))
    checks.append(("agg_clearance_violations", agg["self_clearance_violations"] == 1))
    checks.append(("agg_joint_violations", agg["joint_limit_violations"] == 1))
    checks.append(("agg_workspace_violations", agg["workspace_violations"] == 1))
    # nearest target to anchor [0.5,0,0.6] is frame 0 (error 0.005).
    checks.append(("agg_realized_cut_err", abs(agg["realized_cut_point_error_m"] - 0.005) < 1e-9))

    # 9b. position-only aggregation ignores orientation for validity.
    agg_pos_only = aggregate_pose_frame_metrics(
        frames, orientation_used=False, position_tolerance_m=0.012, orientation_tolerance_rad=0.2,
        cut_anchor_robot_m=None,
    )
    checks.append(("agg_pos_only_valid", agg_pos_only["valid_ik_count"] == 2))
    checks.append(("agg_pos_only_no_cut_err", agg_pos_only["realized_cut_point_error_m"] is None))

    # 9c. empty frames aggregate safely.
    agg_empty = aggregate_pose_frame_metrics(
        [], orientation_used=True, position_tolerance_m=0.012, orientation_tolerance_rad=0.2,
        cut_anchor_robot_m=None,
    )
    checks.append(("agg_empty_zero", agg_empty["frame_count"] == 0 and agg_empty["valid_ik_fraction"] == 0.0))

    # 10. summarize_arm_a_solved on a synthetic solved waypoint list.
    solved = [
        {"name": "approach", "ok": True, "orientation_required": False,
         "ik": {"position_error_m": 0.004, "orientation_error_rad": 0.3},
         "clearance": {"minimum_distance_m": 0.06}},
        {"name": "cut", "ok": True, "orientation_required": True,
         "ik": {"position_error_m": 0.006, "orientation_error_rad": 0.12},
         "clearance": {"minimum_distance_m": 0.04}},
    ]
    summary = summarize_arm_a_solved(solved, hard_ok=True)
    checks.append(("arm_a_max_pos", abs(summary["max_position_error_m"] - 0.006) < 1e-9))
    # orientation only counted for orientation_required waypoints (cut -> 0.12).
    checks.append(("arm_a_max_ori", abs(summary["max_orientation_error_rad"] - 0.12) < 1e-9))
    checks.append(("arm_a_min_clr", abs(summary["min_self_clearance_m"] - 0.04) < 1e-9))
    checks.append(("arm_a_gate", summary["hard_gate_ok"] is True))

    # 11. aggregate_comparison across two rows (one Arm-A pass, one fail).
    rows = [
        {"arm_a": {"hard_gate_ok": True, "min_self_clearance_m": 0.05, "max_position_error_m": 0.006,
                   "max_orientation_error_rad": 0.1},
         "arm_b": {"valid_ik_fraction": 0.8, "self_clearance_violations": 0, "joint_limit_violations": 0,
                   "workspace_violations": 0, "realized_cut_point_error_m": 0.007}},
        {"arm_a": {"hard_gate_ok": False, "min_self_clearance_m": 0.02, "max_position_error_m": 0.03,
                   "max_orientation_error_rad": 0.4},
         "arm_b": {"valid_ik_fraction": 0.2, "self_clearance_violations": 3, "joint_limit_violations": 1,
                   "workspace_violations": 2, "realized_cut_point_error_m": 0.05}},
    ]
    comparison = aggregate_comparison(rows)
    checks.append(("cmp_arm_a_gate_rate", abs(comparison["arm_a_ik_mps"]["gate_pass_rate"] - 0.5) < 1e-9))
    checks.append(("cmp_arm_b_mean_valid", abs(comparison["arm_b_pure_mps"]["mean_valid_ik_fraction"] - 0.5) < 1e-9))
    # one of two Arm-B actions has valid_ik_fraction >= 0.5 -> feasible_action_rate = 0.5.
    checks.append(("cmp_arm_b_feasible", abs(comparison["arm_b_pure_mps"]["feasible_action_rate"] - 0.5) < 1e-9))

    # Report.
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"[{status}] {name}")
    print(f"\nSELF-TEST {'PASS' if all_ok else 'FAIL'} ({sum(1 for _, ok in checks if ok)}/{len(checks)} checks)")
    return all_ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline ablation: IK+MPS (ours) vs pure-MPS direct retarget, per demonstrated cut."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--hand-tracking", type=Path, default=None)
    parser.add_argument("--slam-trajectory", type=Path, default=None)
    parser.add_argument("--frame-index", type=Path, default=None)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--hand-point", choices=("wrist", "pinch"), default="wrist")
    parser.add_argument("--position-tolerance-m", type=float, default=0.012)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=12, help="Max downsampled Arm-B poses per cut.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true",
                        help="Run the offline math unit tests with synthetic data (no robot, no alignment).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
