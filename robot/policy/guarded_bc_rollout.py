from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.policy.eef_action_utils import decode_eef_chunk_to_joints, decode_summary
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
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
    commanded_joint_indices,
    connect_motion_robot,
    run_external_open3d_picker,
    select_point,
    validate_home_state,
    workspace_check,
)
from robot.server.gripper_status import command_targets
from robot.policy.predict_action_chunk_policy import (
    feature_vector,
    policy_uses_deleaf_side_features,
    policy_uses_gcsl_features,
    policy_uses_phase_features,
    predict_bc,
    resolve_deleaf_facing,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = Path(__file__).resolve().parent
DEFAULT_ROLLOUT_ROOT = POLICY_ROOT / "rollout_logs"
DEFAULT_RIGHT_TOOLFIT = (
    REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_right_knife_tip_edge_deleaf_01.json"
)
DEFAULT_LEFT_TOOLFIT = Path("D:/tmp/egodelta_isaac_left_jaw_center_toolfit.json")
DEFAULT_GRIPPER_URL = "http://192.0.2.21:8765"
DEFAULT_LEFT_OPEN_POSITION = -8.0
DEFAULT_LEFT_OPEN_REACHED_POSITION = 1.784
DEFAULT_LEFT_CLOSE_POSITION = 10.945
DEFAULT_HARVEST_FIXED_TAIL_PLANS = {
    "AT1T1": POLICY_ROOT
    / "policy_action_library"
    / "AT1T1"
    / "success_current"
    / "actions"
    / "AT1T1_instance_001_step_01_AT1T1"
    / "planner"
    / "a270c038_AT1T1_success_current.json",
    "AT1T2": POLICY_ROOT
    / "policy_action_library"
    / "AT1T2"
    / "success_current"
    / "actions"
    / "AT1T2_instance_001_step_01_AT1T2"
    / "planner"
    / "a270c038_AT1T2_success_current.json",
}


def _deleaf_cut_plan(leaf: str, instance: str) -> Path:
    return (
        POLICY_ROOT
        / "policy_action_library"
        / leaf
        / "success_current"
        / "actions"
        / instance
        / "action"
        / "referenced"
        / "deleaf_click_cut"
        / f"a270c038_{instance}.json"
    )


# Deterministic deleaf cut plans (solved waypoints) per demonstrated leaf, mirroring the harvest
# fixed-tail plans. The learned deleaf policy does the approach; this plan does the deterministic cut.
DEFAULT_DELEAF_FIXED_TAIL_PLANS = {
    "AD1": _deleaf_cut_plan("AD1", "AD1_instance_001_step_01_AD1"),
    "BD1": _deleaf_cut_plan("BD1", "BD1_instance_005_step_01_BD1"),
    "BD2": _deleaf_cut_plan("BD2", "BD2_instance_002_step_01_BD2"),
    "BD3": _deleaf_cut_plan("BD3", "BD3_instance_003_step_01_BD3"),
    "BD4": _deleaf_cut_plan("BD4", "BD4_instance_002_step_01_BD4"),
}
ARM_DOF = 7


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parse_vector(value: str | None, *, length: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} must have {length} comma-separated values")
    try:
        return np.asarray([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must contain numeric values, for example 0.692,0.138,1.243; "
            f"replace placeholders like X,Y,Z with an actual robot-frame point"
        ) from exc


def bounds_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return {
        "x": (float(args.min_x_m), float(args.max_x_m)),
        "y": (float(args.min_y_m), float(args.max_y_m)),
        "z": (float(args.min_z_m), float(args.max_z_m)),
    }


def default_arm_for_policy(metadata: dict[str, Any]) -> str:
    primitive = str(metadata.get("primitive", "")).strip().lower()
    if primitive == "deleaf":
        return "right"
    if primitive == "harvest":
        return "left"
    raise ValueError(f"cannot infer arm from policy primitive {primitive!r}; pass --arm")


def default_toolfit_for_arm(arm: str) -> Path:
    if arm == "right":
        return DEFAULT_RIGHT_TOOLFIT
    if arm == "left":
        return DEFAULT_LEFT_TOOLFIT
    raise ValueError(f"unsupported arm: {arm}")


def connect_state_robot(*, address: str, model: str) -> Any:
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
    return robot


def load_policy(policy_dir: Path) -> tuple[dict[str, Any], Any]:
    metadata = read_json(policy_dir / "policy_metadata.json")
    arrays = np.load(policy_dir / "policy_arrays.npz", allow_pickle=False)
    return metadata, arrays


def live_state(
    *,
    robot: Any,
    kinematics: RBY1Kinematics,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = robot.get_state()
    q_full = np.asarray(state.position, dtype=float).reshape(-1)
    indices = list(kinematics.active_arm.joint_indices)
    q_arm = q_full[indices].astype(float)

    dq_full = getattr(state, "velocity", None)
    if dq_full is None:
        dq_arm = np.zeros(ARM_DOF, dtype=float)
    else:
        dq_values = np.asarray(dq_full, dtype=float).reshape(-1)
        dq_arm = dq_values[indices].astype(float) if dq_values.size > max(indices) else np.zeros(ARM_DOF, dtype=float)

    override_q = parse_vector(args.q_arm_rad, length=ARM_DOF, name="--q-arm-rad")
    override_dq = parse_vector(args.dq_arm_rad_s, length=ARM_DOF, name="--dq-arm-rad-s")
    override_tcp = parse_vector(args.tcp_position_robot_m, length=3, name="--tcp-position-robot-m")
    if override_q is not None:
        q_arm = override_q
        q_full = kinematics.q_full_from_arm(kinematics.locked_state_from_home(args.home), q_arm)
    if override_dq is not None:
        dq_arm = override_dq
    if override_tcp is None:
        tcp_position = kinematics.tcp_pose(q_full)[:3, 3].astype(float)
    else:
        tcp_position = override_tcp
    return q_full, q_arm, dq_arm, tcp_position


def offline_state(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_arm = parse_vector(args.q_arm_rad, length=ARM_DOF, name="--q-arm-rad")
    tcp_position = parse_vector(args.tcp_position_robot_m, length=3, name="--tcp-position-robot-m")
    if q_arm is None or tcp_position is None:
        raise ValueError("--skip-fk-guards requires --q-arm-rad and --tcp-position-robot-m")
    dq_arm = parse_vector(args.dq_arm_rad_s, length=ARM_DOF, name="--dq-arm-rad-s")
    if dq_arm is None:
        dq_arm = np.zeros(ARM_DOF, dtype=float)
    return q_arm, dq_arm, tcp_position


def predict_chunk(
    *,
    metadata: dict[str, Any],
    arrays: Any,
    goal_robot_m: np.ndarray,
    q_arm: np.ndarray,
    dq_arm: np.ndarray,
    tcp_position: np.ndarray,
    phase_s: float,
    phase_fraction: float,
    horizon_s: float,
    horizon_fraction: float,
    deleaf_facing: str | None,
    kinematics: RBY1Kinematics | None = None,
    locked_state: Any | None = None,
    decode_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None, np.ndarray | None]:
    x = feature_vector(
        goal_robot_m=goal_robot_m,
        q_arm_rad=q_arm,
        dq_arm_rad_s=dq_arm,
        tcp_position_robot_m=tcp_position,
        phase_s=phase_s,
        phase_fraction=phase_fraction,
        horizon_s=horizon_s,
        horizon_fraction=horizon_fraction,
        deleaf_facing=deleaf_facing,
        include_phase_features=policy_uses_phase_features(metadata),
        include_gcsl_features=policy_uses_gcsl_features(metadata),
        include_deleaf_side_features=policy_uses_deleaf_side_features(metadata),
    )
    chunk_size = int(metadata["chunk_size"])
    action_space = str(metadata.get("output", {}).get("action_space", "joint"))
    prediction = predict_bc(arrays, metadata, x)
    if action_space == "eef":
        if kinematics is None or locked_state is None:
            raise RuntimeError(
                "eef action-space policy requires live FK/IK decode: pass kinematics and locked_state "
                "(offline q/tcp overrides are not supported for eef rollouts)."
            )
        eef_chunk = np.asarray(prediction, dtype=float).reshape(chunk_size, 6)
        q_chunk, decode_reports = decode_eef_chunk_to_joints(
            kinematics=kinematics,
            locked_state=locked_state,
            q_arm_now=q_arm,
            eef_chunk=eef_chunk,
            max_steps=decode_steps,
        )
        q_chunk = np.asarray(q_chunk, dtype=float).reshape(chunk_size, ARM_DOF)
        delta = q_chunk - q_arm.reshape(1, ARM_DOF)
        return delta.astype(float), q_chunk.astype(float), decode_summary(decode_reports), eef_chunk.astype(float)
    delta = np.asarray(prediction, dtype=float).reshape(chunk_size, ARM_DOF)
    return delta.astype(float), (q_arm.reshape(1, ARM_DOF) + delta).astype(float), None, None


def gcsl_reference_duration(metadata: dict[str, Any], goal_robot_m: np.ndarray) -> dict[str, Any]:
    demos = [demo for demo in metadata.get("demos", []) if isinstance(demo, dict)]
    candidates: list[dict[str, Any]] = []
    for demo in demos:
        duration = demo.get("duration_s")
        goal = demo.get("goal_robot_m")
        if duration is None:
            continue
        row = {
            "label": demo.get("label"),
            "duration_s": float(duration),
            "goal_robot_m": goal,
            "goal_distance_m": None,
        }
        if goal is not None:
            goal_arr = np.asarray(goal, dtype=float).reshape(3)
            row["goal_distance_m"] = float(np.linalg.norm(goal_arr - goal_robot_m.reshape(3)))
        candidates.append(row)
    if not candidates:
        fallback = float(metadata.get("chunk_horizon_s", 0.0))
        return {
            "duration_s": fallback,
            "source": "chunk_horizon_fallback",
            "selected_demo": None,
            "candidates": [],
        }
    with_goal = [row for row in candidates if row["goal_distance_m"] is not None]
    if with_goal:
        selected = min(with_goal, key=lambda row: float(row["goal_distance_m"]))
        source = "nearest_demo_goal"
    else:
        durations = np.asarray([row["duration_s"] for row in candidates], dtype=float)
        median_duration = float(np.median(durations))
        selected = min(candidates, key=lambda row: abs(float(row["duration_s"]) - median_duration))
        source = "median_demo_duration"
    return {
        "duration_s": float(selected["duration_s"]),
        "source": source,
        "selected_demo": selected,
        "candidates": candidates,
    }


def retimed_phase_from_training_state(
    *,
    metadata: dict[str, Any],
    arrays: Any,
    q_arm: np.ndarray,
    tcp_position: np.ndarray,
    phase_source: str,
    counter_phase_s: float,
) -> tuple[float, dict[str, Any]]:
    if phase_source == "counter":
        return counter_phase_s, {"source": "counter", "phase_s": float(counter_phase_s)}
    if "x" not in arrays.files or "source_times_s" not in arrays.files:
        return counter_phase_s, {
            "source": "counter",
            "requested_source": phase_source,
            "reason": "policy_arrays_missing_training_rows",
            "phase_s": float(counter_phase_s),
        }
    feature_names = list(metadata.get("input", {}).get("feature_names", []))
    x_rows = np.asarray(arrays["x"], dtype=float)
    source_times = np.asarray(arrays["source_times_s"], dtype=float).reshape(-1)
    if len(x_rows) != len(source_times):
        return counter_phase_s, {
            "source": "counter",
            "requested_source": phase_source,
            "reason": "training_rows_source_times_length_mismatch",
            "phase_s": float(counter_phase_s),
        }
    if phase_source == "nearest_training_q":
        cols = [feature_names.index(f"q_arm_rad.{index}") for index in range(ARM_DOF)]
        distances = np.linalg.norm(x_rows[:, cols] - np.asarray(q_arm, dtype=float).reshape(1, ARM_DOF), axis=1)
        unit = "rad"
    elif phase_source == "nearest_training_tcp":
        cols = [feature_names.index(f"tcp_position_robot_m.{axis}") for axis in "xyz"]
        distances = np.linalg.norm(x_rows[:, cols] - np.asarray(tcp_position, dtype=float).reshape(1, 3), axis=1)
        unit = "m"
    else:
        raise ValueError(f"unsupported phase source: {phase_source}")
    index = int(np.argmin(distances))
    distance = float(distances[index])
    phase_s = float(source_times[index])
    return phase_s, {
        "source": phase_source,
        "phase_s": phase_s,
        "counter_phase_s": float(counter_phase_s),
        "phase_delta_s": float(phase_s - counter_phase_s),
        "nearest_training_index": index,
        "nearest_training_distance": distance,
        "nearest_training_distance_unit": unit,
        "nearest_training_distance_deg": float(math.degrees(distance)) if unit == "rad" else None,
    }


def harvest_label_base(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    return text.split("__", 1)[0]


def default_harvest_tail_plan_for_label(label: Any) -> Path | None:
    base = harvest_label_base(label)
    if base is None:
        return None
    return DEFAULT_HARVEST_FIXED_TAIL_PLANS.get(base)


def harvest_plan_handoff_threshold(
    plan_path: Path,
    *,
    candidate_name: str | None,
    goal_robot_m: np.ndarray,
    margin_m: float,
) -> tuple[float, dict[str, Any]]:
    payload = read_json(plan_path)
    candidate = payload.get("best_candidate")
    if candidate_name is not None:
        matches = [
            item
            for item in payload.get("candidates", [])
            if isinstance(item, dict) and item.get("name") == candidate_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one phase-plan candidate {candidate_name!r}, found {len(matches)}"
            )
        candidate = matches[0]
    if not isinstance(candidate, dict):
        raise RuntimeError(f"phase plan has no selected candidate: {plan_path}")
    approach = next(
        (
            item
            for item in candidate.get("waypoints", [])
            if isinstance(item, dict) and item.get("name") == "approach"
        ),
        None,
    )
    if approach is None or approach.get("final_tcp_pose") is None:
        raise RuntimeError(f"phase plan has no solved approach TCP pose: {plan_path}")
    approach_tcp = np.asarray(approach["final_tcp_pose"], dtype=float).reshape(4, 4)[:3, 3]
    goal = np.asarray(goal_robot_m, dtype=float).reshape(3)
    approach_goal_distance = float(np.linalg.norm(approach_tcp - goal))
    threshold = approach_goal_distance + float(margin_m)
    return threshold, {
        "source": "phase_plan_approach",
        "plan": str(plan_path),
        "approach_tcp_robot_m": approach_tcp.tolist(),
        "goal_robot_m": goal.tolist(),
        "approach_goal_distance_m": approach_goal_distance,
        "margin_m": float(margin_m),
        "threshold_m": threshold,
    }


def default_deleaf_tail_plan_for_label(label: Any) -> Path | None:
    base = harvest_label_base(label)
    if base is None:
        return None
    return DEFAULT_DELEAF_FIXED_TAIL_PLANS.get(base)


def step_delta_guard(
    *,
    q_arm: np.ndarray,
    q_chunk: np.ndarray,
    horizon_steps: int,
    max_step_delta_rad: float,
    sample_hz: float,
) -> dict[str, Any]:
    horizon = max(1, min(int(horizon_steps), len(q_chunk)))
    q_sequence = np.vstack([q_arm.reshape(1, ARM_DOF), q_chunk[:horizon]])
    increments = np.diff(q_sequence, axis=0)
    abs_increments = np.abs(increments)
    flat_index = int(np.argmax(abs_increments))
    step_index, joint_index = np.unravel_index(flat_index, abs_increments.shape)
    max_delta = float(abs_increments[step_index, joint_index])
    tolerance = max(1e-12, 1e-9 * abs(float(max_step_delta_rad)))
    return {
        "ok": bool(max_delta <= float(max_step_delta_rad) + tolerance),
        "name": "step_delta",
        "horizon_steps": int(horizon),
        "max_step_delta_rad": max_delta,
        "max_step_delta_deg": float(math.degrees(max_delta)),
        "limit_rad": float(max_step_delta_rad),
        "limit_deg": float(math.degrees(float(max_step_delta_rad))),
        "max_velocity_deg_s": float(math.degrees(max_delta) * float(sample_hz)),
        "step_index": int(step_index + 1),
        "joint_index": int(joint_index),
        "increments_rad": increments.astype(float).tolist(),
    }


def clip_q_chunk_step_delta(
    *,
    q_arm: np.ndarray,
    q_chunk: np.ndarray,
    max_step_delta_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    limit = max(0.0, float(max_step_delta_rad))
    if limit <= 0.0:
        raise ValueError("--max-step-delta-rad must be positive when --clip-step-delta is used")
    clipped_rows: list[np.ndarray] = []
    previous = np.asarray(q_arm, dtype=float).reshape(ARM_DOF)
    max_clip = 0.0
    clipped_step_count = 0
    clipped_joint_count = 0
    for target in np.asarray(q_chunk, dtype=float).reshape(-1, ARM_DOF):
        requested_delta = target - previous
        clipped_delta = np.clip(requested_delta, -limit, limit)
        delta_clip = np.abs(requested_delta - clipped_delta)
        clipped_joints = int(np.sum(delta_clip > 1e-12))
        if clipped_joints:
            clipped_step_count += 1
            clipped_joint_count += clipped_joints
            max_clip = max(max_clip, float(np.max(delta_clip)))
        clipped = previous + clipped_delta
        clipped_rows.append(clipped)
        previous = clipped
    clipped_chunk = np.vstack(clipped_rows).astype(float)
    return clipped_chunk, {
        "enabled": True,
        "limit_rad": float(limit),
        "limit_deg": float(math.degrees(limit)),
        "clipped_step_count": int(clipped_step_count),
        "clipped_joint_count": int(clipped_joint_count),
        "max_clip_rad": float(max_clip),
        "max_clip_deg": float(math.degrees(max_clip)),
    }


def clip_q_chunk_joint_limits(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    q_arm: np.ndarray,
    q_chunk: np.ndarray,
    margin_rad: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    lower, upper = kinematics.active_joint_limits(kinematics.q_full_from_arm(locked_state, q_arm))
    lower_safe = lower + float(margin_rad)
    upper_safe = upper - float(margin_rad)
    q_array = np.asarray(q_chunk, dtype=float).reshape(-1, ARM_DOF)
    clipped = np.minimum(np.maximum(q_array, lower_safe.reshape(1, -1)), upper_safe.reshape(1, -1))
    delta_clip = np.abs(q_array - clipped)
    clipped_mask = delta_clip > 1e-12
    return clipped.astype(float), {
        "enabled": True,
        "margin_rad": float(margin_rad),
        "margin_deg": float(math.degrees(float(margin_rad))),
        "clipped_step_count": int(np.sum(np.any(clipped_mask, axis=1))),
        "clipped_joint_count": int(np.sum(clipped_mask)),
        "max_clip_rad": float(np.max(delta_clip)) if delta_clip.size else 0.0,
        "max_clip_deg": float(math.degrees(float(np.max(delta_clip)))) if delta_clip.size else 0.0,
    }


def safe_prefix_before_target_distance(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    q_chunk: np.ndarray,
    goal_robot_m: np.ndarray,
    threshold_m: float,
    prefix_limit: int,
) -> dict[str, Any]:
    count = max(1, min(int(prefix_limit), len(q_chunk)))
    distances: list[float] = []
    first_crossing: int | None = None
    for index, q_arm in enumerate(q_chunk[:count]):
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        tcp = kinematics.tcp_pose(q_full)[:3, 3]
        distance = float(
            np.linalg.norm(
                np.asarray(tcp, dtype=float).reshape(3)
                - np.asarray(goal_robot_m, dtype=float).reshape(3)
            )
        )
        distances.append(distance)
        if first_crossing is None and distance <= float(threshold_m):
            first_crossing = index
    return {
        "crosses": first_crossing is not None,
        "safe_prefix_count": count if first_crossing is None else int(first_crossing),
        "first_crossing_step": None if first_crossing is None else int(first_crossing + 1),
        "threshold_m": float(threshold_m),
        "distances_m": distances,
    }


def joint_limit_guard(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    q_chunk: np.ndarray,
    horizon_steps: int,
    margin_rad: float,
) -> dict[str, Any]:
    horizon = max(1, min(int(horizon_steps), len(q_chunk)))
    lower, upper = kinematics.active_joint_limits(kinematics.q_full_from_arm(locked_state, q_chunk[0]))
    violations: list[dict[str, Any]] = []
    for step, q_arm in enumerate(q_chunk[:horizon], start=1):
        below = q_arm < (lower + float(margin_rad))
        above = q_arm > (upper - float(margin_rad))
        for joint_index in np.flatnonzero(below | above):
            violations.append(
                {
                    "step_index": int(step),
                    "joint_index": int(joint_index),
                    "q_rad": float(q_arm[joint_index]),
                    "lower_rad": float(lower[joint_index]),
                    "upper_rad": float(upper[joint_index]),
                    "margin_rad": float(margin_rad),
                }
            )
    return {
        "ok": not violations,
        "name": "joint_limits",
        "horizon_steps": int(horizon),
        "margin_rad": float(margin_rad),
        "violations": violations,
    }


def interpolated_arm_path(q_start: np.ndarray, q_chunk: np.ndarray, *, horizon_steps: int, samples_per_step: int) -> list[tuple[int, float, np.ndarray]]:
    horizon = max(1, min(int(horizon_steps), len(q_chunk)))
    q_sequence = [np.asarray(q_start, dtype=float).reshape(ARM_DOF)] + [
        np.asarray(q, dtype=float).reshape(ARM_DOF) for q in q_chunk[:horizon]
    ]
    rows: list[tuple[int, float, np.ndarray]] = []
    samples = max(1, int(samples_per_step))
    for step in range(1, len(q_sequence)):
        q0 = q_sequence[step - 1]
        q1 = q_sequence[step]
        for sample in range(1, samples + 1):
            alpha = float(sample) / float(samples)
            rows.append((step, alpha, (1.0 - alpha) * q0 + alpha * q1))
    return rows


def fk_guards(
    *,
    kinematics: RBY1Kinematics,
    locked_state: Any,
    q_arm: np.ndarray,
    q_chunk: np.ndarray,
    goal_robot_m: np.ndarray,
    tcp_start: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested_guard_horizon = args.guard_horizon_steps if args.guard_horizon_steps is not None else args.steps_to_execute
    guard_horizon = max(1, min(int(requested_guard_horizon), len(q_chunk)))
    progress_horizon = max(1, min(int(args.steps_to_execute), len(q_chunk)))
    bounds = bounds_from_args(args)
    path_rows = interpolated_arm_path(
        q_arm,
        q_chunk,
        horizon_steps=guard_horizon,
        samples_per_step=int(args.path_samples_per_step),
    )

    workspace_rows: list[dict[str, Any]] = []
    clearance_rows: list[dict[str, Any]] = []
    min_clearance: float | None = None
    tcp_by_step: dict[int, np.ndarray] = {}
    for step_index, alpha, q_sample in path_rows:
        q_full = kinematics.q_full_from_arm(locked_state, q_sample)
        tcp_position = kinematics.tcp_pose(q_full)[:3, 3].astype(float)
        tcp_by_step[step_index] = tcp_position
        workspace = workspace_check(tcp_position, bounds)
        if not workspace["ok"]:
            workspace_rows.append(
                {
                    "step_index": int(step_index),
                    "alpha": float(alpha),
                    "tcp_position_robot_m": tcp_position.tolist(),
                    "workspace": workspace,
                }
            )
        clearance = nearest_self_clearances(
            kinematics,
            q_full,
            pair_count=int(args.clearance_pair_count),
            margin_m=float(args.min_clearance_m),
        )
        distance = clearance.minimum_distance_m
        if distance is not None and (min_clearance is None or distance < min_clearance):
            min_clearance = float(distance)
        if not clearance.ok:
            clearance_rows.append(
                {
                    "step_index": int(step_index),
                    "alpha": float(alpha),
                    "minimum_distance_m": distance,
                    "margin_m": float(args.min_clearance_m),
                }
            )

    tcp_progress = tcp_by_step.get(progress_horizon)
    if tcp_progress is None:
        tcp_progress = kinematics.tcp_pose(kinematics.q_full_from_arm(locked_state, q_chunk[progress_horizon - 1]))[:3, 3]
    start_distance = float(np.linalg.norm(tcp_start - goal_robot_m))
    end_distance = float(np.linalg.norm(np.asarray(tcp_progress, dtype=float) - goal_robot_m))
    progress_ok = bool(args.disable_progress_guard or end_distance <= start_distance + float(args.progress_tolerance_m))

    return {
        "workspace": {
            "ok": not workspace_rows,
            "name": "workspace",
            "horizon_steps": int(guard_horizon),
            "violations": workspace_rows,
            "bounds_m": {axis: [float(lo), float(hi)] for axis, (lo, hi) in bounds.items()},
        },
        "self_clearance": {
            "ok": not clearance_rows,
            "name": "self_clearance",
            "horizon_steps": int(guard_horizon),
            "minimum_distance_m": min_clearance,
            "margin_m": float(args.min_clearance_m),
            "violations": clearance_rows,
        },
        "target_progress": {
            "ok": progress_ok,
            "name": "target_progress",
            "disabled": bool(args.disable_progress_guard),
            "horizon_steps": int(progress_horizon),
            "start_distance_m": start_distance,
            "end_distance_m": end_distance,
            "progress_m": float(start_distance - end_distance),
            "tolerance_m": float(args.progress_tolerance_m),
            "tcp_start_robot_m": tcp_start.astype(float).tolist(),
            "tcp_progress_robot_m": np.asarray(tcp_progress, dtype=float).tolist(),
        },
    }


def evaluate_guards(
    *,
    metadata: dict[str, Any],
    kinematics: RBY1Kinematics | None,
    locked_state: Any | None,
    q_arm: np.ndarray,
    q_chunk: np.ndarray,
    goal_robot_m: np.ndarray,
    tcp_position: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    guard_horizon = int(args.guard_horizon_steps if args.guard_horizon_steps is not None else args.steps_to_execute)
    sample_hz = float(metadata["sample_hz"])
    reports: dict[str, Any] = {
        "step_delta": step_delta_guard(
            q_arm=q_arm,
            q_chunk=q_chunk,
            horizon_steps=guard_horizon,
            max_step_delta_rad=float(args.max_step_delta_rad),
            sample_hz=sample_hz,
        )
    }
    if args.skip_fk_guards:
        reports["fk"] = {
            "skipped": True,
            "reason": "--skip-fk-guards",
        }
    else:
        if kinematics is None or locked_state is None:
            raise RuntimeError("FK guards require live kinematics; remove --skip-fk-guards only when the robot is reachable")
        reports["joint_limits"] = joint_limit_guard(
            kinematics=kinematics,
            locked_state=locked_state,
            q_chunk=q_chunk,
            horizon_steps=guard_horizon,
            margin_rad=float(args.joint_limit_margin_rad),
        )
        reports.update(
            fk_guards(
                kinematics=kinematics,
                locked_state=locked_state,
                q_arm=q_arm,
                q_chunk=q_chunk,
                goal_robot_m=goal_robot_m,
                tcp_start=tcp_position,
                args=args,
            )
        )

    failed = [
        name
        for name, report in reports.items()
        if isinstance(report, dict) and "ok" in report and not bool(report.get("ok"))
    ]
    return {
        "ok": not failed,
        "failed_guards": failed,
        "failure_details": guard_failure_details(failed, reports),
        "fallback_recommended": bool(failed),
        "reports": reports,
    }


def guard_failure_details(failed: list[str], reports: dict[str, Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for name in failed:
        report = reports.get(name)
        if not isinstance(report, dict):
            details.append({"guard": name, "reason": "missing_report"})
            continue
        if name == "step_delta":
            details.append(
                {
                    "guard": name,
                    "step_index": report.get("step_index"),
                    "joint_index": report.get("joint_index"),
                    "max_step_delta_rad": report.get("max_step_delta_rad"),
                    "max_step_delta_deg": report.get("max_step_delta_deg"),
                    "limit_rad": report.get("limit_rad"),
                    "limit_deg": report.get("limit_deg"),
                    "max_velocity_deg_s": report.get("max_velocity_deg_s"),
                }
            )
        elif name == "target_progress":
            details.append(
                {
                    "guard": name,
                    "disabled": report.get("disabled"),
                    "horizon_steps": report.get("horizon_steps"),
                    "start_distance_m": report.get("start_distance_m"),
                    "end_distance_m": report.get("end_distance_m"),
                    "progress_m": report.get("progress_m"),
                    "tolerance_m": report.get("tolerance_m"),
                    "tcp_start_robot_m": report.get("tcp_start_robot_m"),
                    "tcp_progress_robot_m": report.get("tcp_progress_robot_m"),
                }
            )
        elif name == "workspace":
            violations = report.get("violations") or []
            details.append(
                {
                    "guard": name,
                    "violation_count": len(violations),
                    "first_violation": violations[0] if violations else None,
                    "bounds_m": report.get("bounds_m"),
                }
            )
        elif name == "self_clearance":
            violations = report.get("violations") or []
            details.append(
                {
                    "guard": name,
                    "minimum_distance_m": report.get("minimum_distance_m"),
                    "margin_m": report.get("margin_m"),
                    "violation_count": len(violations),
                    "first_violation": violations[0] if violations else None,
                }
            )
        elif name == "joint_limits":
            violations = report.get("violations") or []
            details.append(
                {
                    "guard": name,
                    "margin_rad": report.get("margin_rad"),
                    "violation_count": len(violations),
                    "first_violation": violations[0] if violations else None,
                }
            )
        else:
            details.append({"guard": name, "report": report})
    return details


def _send_arm(
    robot: Any,
    home: dict[str, Any],
    arm: str,
    q_arm: np.ndarray,
    *,
    minimum_time_s: float,
    velocity_limit_rad_s: float | None,
    acceleration_limit_rad_s2: float | None,
    timeout_s: float,
    label: str | None = None,
) -> Any:
    """Push one arm joint target, returning a feedback with ``.finish_code``.

    Real backend: ``robot.send_command(command_arm_only(...)).get()``. Sim backend
    (``robot.is_sim_backend``): buffer the commanded pose for replay into the twin.
    """
    if getattr(robot, "is_sim_backend", False):
        return robot.apply_q_arm(home, arm, q_arm, minimum_time_s=float(minimum_time_s), label=label)
    return robot.send_command(
        command_arm_only(
            arm,
            q_arm,
            minimum_time_s=float(minimum_time_s),
            velocity_limit_rad_s=velocity_limit_rad_s,
            acceleration_limit_rad_s2=acceleration_limit_rad_s2,
        ),
        int(max(float(timeout_s), float(minimum_time_s) + 5.0)),
    ).get()


def execute_prefix(
    *,
    robot: Any,
    home: dict[str, Any],
    arm: str,
    q_chunk: np.ndarray,
    steps_to_execute: int,
    minimum_time_s: float,
    timeout_s: float,
    inter_step_sleep_s: float,
    execute_prefix_mode: str,
    velocity_limit_rad_s: float | None = None,
    acceleration_limit_rad_s2: float | None = None,
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    prefix_count = max(1, min(int(steps_to_execute), len(q_chunk)))
    if execute_prefix_mode == "endpoint":
        q_arm = np.asarray(q_chunk[prefix_count - 1], dtype=float)
        feedback = _send_arm(
            robot,
            home,
            arm,
            q_arm,
            minimum_time_s=float(minimum_time_s),
            velocity_limit_rad_s=velocity_limit_rad_s,
            acceleration_limit_rad_s2=acceleration_limit_rad_s2,
            timeout_s=timeout_s,
            label="policy_endpoint",
        )
        finish_code = str(feedback.finish_code)
        ok = feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
        for step_index, q_row in enumerate(q_chunk[:prefix_count], start=1):
            commanded = step_index == prefix_count
            executed.append(
                {
                    "step_index": int(step_index),
                    "q_arm_rad": np.asarray(q_row, dtype=float).tolist(),
                    "finish_code": finish_code if commanded else None,
                    "ok": bool(ok),
                    "commanded": bool(commanded),
                    "execute_prefix_mode": "endpoint",
                }
            )
        return executed
    for step_index, q_arm in enumerate(q_chunk[:prefix_count], start=1):
        feedback = _send_arm(
            robot,
            home,
            arm,
            q_arm,
            minimum_time_s=float(minimum_time_s),
            velocity_limit_rad_s=velocity_limit_rad_s,
            acceleration_limit_rad_s2=acceleration_limit_rad_s2,
            timeout_s=timeout_s,
            label="policy_step",
        )
        finish_code = str(feedback.finish_code)
        executed.append(
            {
                "step_index": int(step_index),
                "q_arm_rad": np.asarray(q_arm, dtype=float).tolist(),
                "finish_code": finish_code,
                "ok": feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok,
                "commanded": True,
                "execute_prefix_mode": "all_steps",
            }
        )
        if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            break
        if float(inter_step_sleep_s) > 0.0:
            time.sleep(float(inter_step_sleep_s))
    return executed


def _http_get_json(url: str, *, timeout_s: float, retries: int = 4, backoff_s: float = 0.15) -> dict[str, Any]:
    """GET url -> JSON, retrying transient errors. The gripper bridge drives a single serial servo
    bus and returns HTTP 500 (or drops the read) when a request races a concurrent one -- e.g. the
    live digital-twin gripper mirror polling :8765 during a harvest. Retry so a stray collision
    does not abort the grasp (the /jog command is idempotent -- re-commanding the same target is safe)."""
    last_exc: Exception | None = None
    for _i in range(max(1, int(retries))):
        try:
            with urllib.request.urlopen(url, timeout=float(timeout_s)) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # HTTPError 500 / URLError / timeout / partial read
            last_exc = exc
            time.sleep(float(backoff_s) * (_i + 1))
    raise last_exc if last_exc is not None else RuntimeError("gripper bridge GET failed")


def _gripper_status(base_url: str, *, timeout_s: float) -> dict[str, Any]:
    payload = _http_get_json(f"{base_url.rstrip('/')}/status", timeout_s=timeout_s)
    if not payload.get("ok", False):
        raise RuntimeError(f"gripper status failed: {payload}")
    return payload


def _gripper_position_from_payload(payload: dict[str, Any], *, arm: str) -> float | None:
    target_id = 1 if arm == "left" else 0
    for state in payload.get("motor_states", []):
        if int(state.get("id", -1)) == target_id and state.get("position") is not None:
            return float(state["position"])
    return None


def _gripper_position(base_url: str, *, arm: str, timeout_s: float) -> float | None:
    return _gripper_position_from_payload(_gripper_status(base_url, timeout_s=timeout_s), arm=arm)


def _require_gripper_bridge(base_url: str, *, timeout_s: float) -> dict[str, Any]:
    try:
        payload = _gripper_status(base_url, timeout_s=timeout_s)
    except Exception as exc:
        raise RuntimeError(
            "gripper HTTP bridge is not reachable before motion. "
            f"URL={base_url.rstrip('/')}/status. "
            "Start robot/server/rby1_gripper_http_bridge.py or rerun without --harvest-gripper-schedule."
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
    payload = _http_get_json(f"{base_url.rstrip('/')}/jog?{query}", timeout_s=timeout_s)
    if not payload.get("ok", False):
        raise RuntimeError(f"gripper command failed: {payload}")
    return payload


def _wait_for_gripper_open(
    base_url: str,
    *,
    arm: str,
    expected_reached_position: float,
    tolerance: float,
    timeout_s: float,
    poll_s: float,
    request_timeout_s: float,
) -> float | None:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last_position = _gripper_position(base_url, arm=arm, timeout_s=request_timeout_s)
    while time.monotonic() <= deadline:
        if (
            last_position is not None
            and abs(float(last_position) - float(expected_reached_position)) <= float(tolerance)
        ):
            return last_position
        time.sleep(max(0.05, float(poll_s)))
        last_position = _gripper_position(base_url, arm=arm, timeout_s=request_timeout_s)
    return last_position


def _gripper_close_motion_ok(
    *,
    before_position: float | None,
    after_position: float | None,
    command_position: float,
    min_motion: float,
) -> bool:
    if before_position is None or after_position is None:
        return False
    command_delta = float(command_position) - float(before_position)
    if abs(command_delta) <= 1e-9:
        return False
    signed_motion = np.sign(command_delta) * (float(after_position) - float(before_position))
    return bool(signed_motion >= float(min_motion))


def _wait_for_gripper_close_motion(
    base_url: str,
    *,
    arm: str,
    command_position: float,
    before_position: float | None,
    min_motion: float,
    timeout_s: float,
    poll_s: float,
    request_timeout_s: float,
) -> float | None:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last_position = _gripper_position(base_url, arm=arm, timeout_s=request_timeout_s)
    while time.monotonic() <= deadline:
        if _gripper_close_motion_ok(
            before_position=before_position,
            after_position=last_position,
            command_position=command_position,
            min_motion=min_motion,
        ):
            return last_position
        time.sleep(max(0.05, float(poll_s)))
        last_position = _gripper_position(base_url, arm=arm, timeout_s=request_timeout_s)
    return last_position


def _command_gripper_open(
    args: argparse.Namespace,
    *,
    arm: str,
    action: str,
    reason: str,
    phase_s: float,
    phase_fraction: float,
    tcp_goal_distance_m: float | None,
) -> dict[str, Any]:
    _sim = getattr(args, "_sim_robot", None)
    if _sim is not None:
        return _sim.gripper(
            "open", arm=arm, action=action, reason=reason, phase_s=phase_s,
            phase_fraction=phase_fraction, tcp_goal_distance_m=tcp_goal_distance_m,
        )
    status = _gripper_status(args.gripper_url, timeout_s=float(args.gripper_timeout_s))
    before = _gripper_position_from_payload(status, arm=arm)
    if args.gripper_target_source == "live_status":
        targets = command_targets(status, arm=arm)
        open_position = targets.open_position
        open_reached_position = targets.open_position
        target_source = "live_status"
    else:
        open_position = float(args.left_open_position)
        open_reached_position = float(args.left_open_reached_position)
        target_source = "absolute_cli"
    bridge_payload = _gripper_jog(
        args.gripper_url,
        arm=arm,
        position=open_position,
        current_limit=float(args.gripper_open_current_limit),
        hold_s=float(args.gripper_hold_s),
        timeout_s=float(args.gripper_timeout_s),
    )
    after = _wait_for_gripper_open(
        args.gripper_url,
        arm=arm,
        expected_reached_position=open_reached_position,
        tolerance=float(args.left_open_tolerance),
        timeout_s=float(args.gripper_post_command_wait_s),
        poll_s=float(args.gripper_post_command_poll_s),
        request_timeout_s=float(args.gripper_timeout_s),
    )
    position_error = None if after is None else abs(float(after) - open_reached_position)
    event = {
        "action": action,
        "reason": reason,
        "arm": arm,
        "phase_s": float(phase_s),
        "phase_fraction": float(phase_fraction),
        "tcp_goal_distance_m": tcp_goal_distance_m,
        "command_position": open_position,
        "position_before": before,
        "position_after": after,
        "expected_reached_position": open_reached_position,
        "tolerance": float(args.left_open_tolerance),
        "current_limit": float(args.gripper_open_current_limit),
        "hold_s": float(args.gripper_hold_s),
        "bridge_payload": bridge_payload,
        "target_source": target_source,
        "position_error": position_error,
        "ok": position_error is not None and position_error <= float(args.left_open_tolerance),
    }
    if not event["ok"]:
        raise RuntimeError(f"gripper open did not reach expected side: {event}")
    return event


def _command_gripper_close(
    args: argparse.Namespace,
    *,
    arm: str,
    reason: str,
    phase_s: float,
    phase_fraction: float,
    tcp_goal_distance_m: float | None,
) -> dict[str, Any]:
    _sim = getattr(args, "_sim_robot", None)
    if _sim is not None:
        return _sim.gripper(
            "close", arm=arm, action="grasp_close", reason=reason, phase_s=phase_s,
            phase_fraction=phase_fraction, tcp_goal_distance_m=tcp_goal_distance_m,
        )
    status = _gripper_status(args.gripper_url, timeout_s=float(args.gripper_timeout_s))
    before = _gripper_position_from_payload(status, arm=arm)
    if args.gripper_target_source == "live_status":
        close_position = command_targets(status, arm=arm).close_position
        target_source = "live_status"
    else:
        close_position = float(args.left_close_position)
        target_source = "absolute_cli"
    bridge_payload = _gripper_jog(
        args.gripper_url,
        arm=arm,
        position=close_position,
        current_limit=float(args.gripper_close_current_limit),
        hold_s=float(args.gripper_hold_s),
        timeout_s=float(args.gripper_timeout_s),
    )
    if before is None and isinstance(bridge_payload, dict):
        # Bus-contention fallback: the separate pre-read can come back empty when something else
        # polls the :8765 bridge concurrently (e.g. the live digital-twin gripper mirror). The jog
        # response carries its own atomic pre-command snapshot -- use it so a transient None read
        # never aborts an otherwise-good grasp.
        _tid = 1 if arm == "left" else 0
        for _s in (bridge_payload.get("before") or []):
            if int(_s.get("id", -1)) == _tid and _s.get("position") is not None:
                before = float(_s["position"])
                break
    after = _wait_for_gripper_close_motion(
        args.gripper_url,
        arm=arm,
        command_position=close_position,
        before_position=before,
        min_motion=float(args.gripper_close_min_motion),
        timeout_s=float(args.gripper_post_command_wait_s),
        poll_s=float(args.gripper_post_command_poll_s),
        request_timeout_s=float(args.gripper_timeout_s),
    )
    event = {
        "action": "grasp_close",
        "reason": reason,
        "arm": arm,
        "phase_s": float(phase_s),
        "phase_fraction": float(phase_fraction),
        "tcp_goal_distance_m": tcp_goal_distance_m,
        "command_position": close_position,
        "position_before": before,
        "position_after": after,
        "min_motion": float(args.gripper_close_min_motion),
        "current_limit": float(args.gripper_close_current_limit),
        "hold_s": float(args.gripper_hold_s),
        "bridge_payload": bridge_payload,
        "target_source": target_source,
        "ok": _gripper_close_motion_ok(
            before_position=before,
            after_position=after,
            command_position=close_position,
            min_motion=float(args.gripper_close_min_motion),
        ),
    }
    if not event["ok"]:
        raise RuntimeError(f"gripper close did not produce verified motion: {event}")
    return event


def policy_owns_harvest_gripper_schedule(args: argparse.Namespace) -> bool:
    if args.harvest_fixed_basket_tail_plan is None:
        return True
    return str(args.harvest_fixed_basket_tail_start_after) not in ("begin", "approach")


def maybe_apply_harvest_gripper_schedule(
    *,
    args: argparse.Namespace,
    arm: str,
    state: dict[str, bool],
    goal_robot_m: np.ndarray,
    tcp_position: np.ndarray,
    phase_s: float,
    phase_fraction: float,
) -> list[dict[str, Any]]:
    if not bool(args.harvest_gripper_schedule):
        return []
    distance = float(np.linalg.norm(np.asarray(tcp_position, dtype=float).reshape(3) - goal_robot_m.reshape(3)))
    events: list[dict[str, Any]] = []
    if not state["pre_grasp_opened"]:
        events.append(
            _command_gripper_open(
                args,
                arm=arm,
                action="pre_grasp_open",
                reason="schedule_start",
                phase_s=phase_s,
                phase_fraction=phase_fraction,
                tcp_goal_distance_m=distance,
            )
        )
        state["pre_grasp_opened"] = True
        return events

    close_by_distance = distance <= float(args.harvest_gripper_close_distance_m)
    close_by_phase = (
        phase_fraction >= float(args.harvest_gripper_close_phase)
        and distance <= float(args.harvest_gripper_phase_close_max_distance_m)
    )
    if not state["grasp_closed"] and (close_by_distance or close_by_phase):
        reason = "target_distance" if close_by_distance else "phase"
        events.append(
            _command_gripper_close(
                args,
                arm=arm,
                reason=reason,
                phase_s=phase_s,
                phase_fraction=phase_fraction,
                tcp_goal_distance_m=distance,
            )
        )
        state["grasp_closed"] = True
        return events

    if (
        args.harvest_fixed_basket_tail_plan is None
        and state["grasp_closed"]
        and not state["basket_released"]
        and phase_fraction >= float(args.harvest_gripper_release_phase)
    ):
        events.append(
            _command_gripper_open(
                args,
                arm=arm,
                action="basket_release_open",
                reason="phase",
                phase_s=phase_s,
                phase_fraction=phase_fraction,
                tcp_goal_distance_m=distance,
            )
        )
        state["basket_released"] = True
    return events


def load_harvest_tail_waypoints(
    plan_path: Path,
    *,
    candidate_name: str | None,
    start_after: str,
) -> list[dict[str, Any]]:
    payload = read_json(plan_path)
    candidate = payload.get("best_candidate")
    if candidate_name is not None:
        matches = [
            item
            for item in payload.get("candidates", [])
            if isinstance(item, dict) and item.get("name") == candidate_name
        ]
        if not matches:
            raise RuntimeError(f"candidate not found in fixed tail plan: {candidate_name}")
        candidate = matches[0]
    if not isinstance(candidate, dict):
        raise RuntimeError(f"fixed tail plan has no selected candidate: {plan_path}")
    waypoints = [item for item in candidate.get("waypoints", []) if isinstance(item, dict)]
    names = [str(item.get("name", "")) for item in waypoints]
    if str(start_after).lower() in ("begin", "start", "none", "all"):
        if not waypoints:
            raise RuntimeError(f"fixed tail plan has no waypoints: {plan_path}")
        return waypoints
    if start_after not in names:
        raise RuntimeError(f"fixed tail plan has no start-after waypoint {start_after!r}; names={names}")
    start_index = names.index(start_after)
    tail = waypoints[start_index + 1 :]
    if not tail:
        raise RuntimeError(f"fixed tail plan has no waypoints after {start_after!r}")
    return tail


def _retarget_tail_waypoint_q(
    *,
    waypoint: dict[str, Any],
    kinematics: Any,
    locked_state: Any,
    delta_m: np.ndarray,
    insertion_dir: Any,
    extra_insertion_m: float,
    height_offset_m: float,
    fallback_q: np.ndarray,
    position_tolerance_m: float = 0.01,
    orientation_tolerance_rad: float = 0.15,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-solve a deterministic tail waypoint shifted to the live target (and optionally deeper).

    The waypoint's stored TCP pose is translated by delta_m (live tomato - demo tomato), and for
    grasp/harvest waypoints an extra insertion along the demonstrated pre_grasp->grasp axis is added.
    The shifted pose is re-solved with the validated IK. If IK does not converge within tolerance the
    original (plan) joints are returned unchanged, so a bad pose is never commanded.
    """
    from robot.kinematics.ik_solver import solve_tcp_pose_ik
    from robot.kinematics.types import IKSolverConfig

    name = str(waypoint.get("name", ""))
    base = waypoint.get("final_tcp_pose") or waypoint.get("target_tcp_pose") or waypoint.get("target_pose")
    if base is None:
        return np.asarray(fallback_q, dtype=float), {"name": name, "retargeted": False, "reason": "no_tcp_pose"}
    pose = np.asarray(base, dtype=float).reshape(4, 4).copy()
    shift = np.asarray(delta_m, dtype=float).reshape(3).copy()
    if name in ("grasp", "harvest") and insertion_dir is not None and float(extra_insertion_m) != 0.0:
        shift = shift + float(extra_insertion_m) * np.asarray(insertion_dir, dtype=float).reshape(3)
    if name in ("pre_grasp", "grasp", "harvest") and float(height_offset_m) != 0.0:
        shift = shift + np.array([0.0, 0.0, float(height_offset_m)], dtype=float)
    pose[:3, 3] = pose[:3, 3] + shift
    result = solve_tcp_pose_ik(
        kinematics,
        locked_state,
        pose,
        seed_q_arm_rad=fallback_q,
        reference_q_arm_rad=fallback_q,
        config=IKSolverConfig(),
    )
    within = bool(
        result.ok
        and result.position_error_m <= float(position_tolerance_m)
        and result.orientation_error_rad <= float(orientation_tolerance_rad)
    )
    report = {
        "name": name,
        "retargeted": within,
        "shift_m": shift.astype(float).tolist(),
        "ik_ok": bool(result.ok),
        "position_error_m": float(result.position_error_m),
        "orientation_error_rad": float(result.orientation_error_rad),
    }
    if not within:
        report["reason"] = "ik_out_of_tolerance"
        return np.asarray(fallback_q, dtype=float), report
    return np.asarray(result.q_arm_rad, dtype=float).reshape(ARM_DOF), report


def execute_fixed_harvest_tail(
    *,
    args: argparse.Namespace,
    robot: Any,
    home: dict[str, Any],
    arm: str,
    phase_s: float,
    phase_fraction: float,
    kinematics: Any = None,
    locked_state: Any = None,
    live_goal_robot_m: Any = None,
) -> dict[str, Any]:
    if args.harvest_fixed_basket_tail_plan is None:
        return {"enabled": False, "executed": False}
    plan_path = Path(args.harvest_fixed_basket_tail_plan)
    full_payload = read_json(plan_path)
    candidate = full_payload.get("best_candidate")
    if args.harvest_fixed_basket_tail_candidate_name is not None:
        for item in full_payload.get("candidates", []):
            if isinstance(item, dict) and item.get("name") == args.harvest_fixed_basket_tail_candidate_name:
                candidate = item
                break
    if not isinstance(candidate, dict):
        raise RuntimeError(f"fixed tail plan has no selected candidate: {plan_path}")
    tail_waypoints = load_harvest_tail_waypoints(
        plan_path,
        candidate_name=args.harvest_fixed_basket_tail_candidate_name,
        start_after=str(args.harvest_fixed_basket_tail_start_after),
    )
    retarget_enabled = bool(args.harvest_grasp_retarget) and kinematics is not None and locked_state is not None
    extra_insertion_m = float(args.harvest_grasp_extra_insertion_m)
    height_offset_m = float(args.harvest_grasp_height_offset_m)
    delta_m = np.zeros(3, dtype=float)
    insertion_dir = None
    retarget_info: dict[str, Any] = {
        "enabled": bool(retarget_enabled),
        "extra_insertion_m": extra_insertion_m,
        "height_offset_m": height_offset_m,
    }
    if retarget_enabled or extra_insertion_m != 0.0 or height_offset_m != 0.0:
        full_wps = [w for w in candidate.get("waypoints", []) if isinstance(w, dict)]
        grasp_wp = next((w for w in full_wps if str(w.get("name")) == "grasp"), None)
        pre_wp = next((w for w in full_wps if str(w.get("name")) == "pre_grasp"), None)
        if grasp_wp is not None:
            plan_goal = grasp_wp.get("target_contact_position_robot_m") or grasp_wp.get("grasp_anchor_robot_m")
            if retarget_enabled and plan_goal is not None and live_goal_robot_m is not None:
                delta_m = (
                    np.asarray(live_goal_robot_m, dtype=float).reshape(3)
                    - np.asarray(plan_goal, dtype=float).reshape(3)
                )
                retarget_info["plan_goal_robot_m"] = np.asarray(plan_goal, dtype=float).tolist()
                retarget_info["live_goal_robot_m"] = np.asarray(live_goal_robot_m, dtype=float).tolist()
                retarget_info["delta_m"] = delta_m.tolist()
            if (
                extra_insertion_m != 0.0
                and grasp_wp.get("final_tcp_pose") is not None
                and pre_wp is not None
                and pre_wp.get("final_tcp_pose") is not None
            ):
                g_pos = np.asarray(grasp_wp["final_tcp_pose"], dtype=float).reshape(4, 4)[:3, 3]
                p_pos = np.asarray(pre_wp["final_tcp_pose"], dtype=float).reshape(4, 4)[:3, 3]
                axis = g_pos - p_pos
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm > 1e-6:
                    insertion_dir = axis / axis_norm
                    retarget_info["insertion_dir"] = insertion_dir.tolist()
    live_transition_hard_gate = None
    if (
        kinematics is not None
        and locked_state is not None
        and not retarget_enabled
        and extra_insertion_m == 0.0
        and height_offset_m == 0.0
    ):
        from robot.kinematics.harvest_primitive import validate_harvest_candidate

        state = robot.get_state()
        q_live = np.asarray(state.position, dtype=float).reshape(-1)
        q_arm_live = q_live[list(kinematics.active_arm.joint_indices)]
        live_transition_hard_gate = validate_harvest_candidate(
            candidate,
            kinematics=kinematics,
            locked_state=locked_state,
            start_q_arm_rad=q_arm_live,
            joint_limit_margin_rad=float(args.joint_limit_margin_rad),
            minimum_clearance_m=float(args.min_clearance_m),
            clearance_pair_count=int(args.clearance_pair_count),
            samples_per_segment=max(20, int(args.path_samples_per_step)),
            manipulation_bounds={
                "x": (0.20, 1.10),
                "y": (-0.75, 0.75),
                "z": (0.40, 1.45),
            },
            transit_bounds={
                "x": (-0.25, 1.10),
                "y": (-0.80, 0.80),
                "z": (0.35, 1.45),
            },
        )
        if not live_transition_hard_gate.get("ok", False):
            return {
                "enabled": True,
                "executed": False,
                "plan": str(plan_path),
                "start_after": str(args.harvest_fixed_basket_tail_start_after),
                "reason": "live_transition_hard_gate_failed",
                "live_transition_hard_gate": live_transition_hard_gate,
                "ok": False,
            }
    executed: list[dict[str, Any]] = []
    gripper_events: list[dict[str, Any]] = []
    retarget_reports: list[dict[str, Any]] = []
    for local_index, waypoint in enumerate(tail_waypoints, start=1):
        q = waypoint.get("ik_result", {}).get("q_arm_rad")
        if q is None:
            raise RuntimeError(f"fixed tail waypoint lacks ik_result.q_arm_rad: {waypoint.get('name')}")
        q_arm = np.asarray(q, dtype=float)
        if q_arm.shape != (ARM_DOF,):
            raise RuntimeError(f"fixed tail waypoint has invalid q_arm shape: {waypoint.get('name')}")
        waypoint_name = str(waypoint.get("name", f"tail_{local_index}"))
        if (retarget_enabled or extra_insertion_m != 0.0 or height_offset_m != 0.0) and waypoint_name != "basket_place":
            q_arm, retarget_report = _retarget_tail_waypoint_q(
                waypoint=waypoint,
                kinematics=kinematics,
                locked_state=locked_state,
                delta_m=delta_m,
                insertion_dir=insertion_dir,
                extra_insertion_m=extra_insertion_m,
                height_offset_m=height_offset_m,
                fallback_q=q_arm,
            )
            retarget_reports.append(retarget_report)
        feedback = _send_arm(
            robot,
            home,
            arm,
            q_arm,
            minimum_time_s=float(args.harvest_fixed_basket_tail_minimum_time_s),
            velocity_limit_rad_s=(
                float(args.harvest_tail_velocity_limit_rad_s)
                if args.harvest_tail_velocity_limit_rad_s is not None
                else None
            ),
            acceleration_limit_rad_s2=(
                float(args.harvest_tail_acceleration_limit_rad_s2)
                if args.harvest_tail_acceleration_limit_rad_s2 is not None
                else None
            ),
            timeout_s=float(args.timeout_s),
            label="tail_" + waypoint_name,
        )
        finish_code = str(feedback.finish_code)
        ok = feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
        executed.append(
            {
                "step_index": int(local_index),
                "name": str(waypoint.get("name", f"tail_{local_index}")),
                "q_arm_rad": q_arm.astype(float).tolist(),
                "finish_code": finish_code,
                "ok": bool(ok),
            }
        )
        if not ok:
            break
        if bool(args.harvest_gripper_schedule) and waypoint_name == "pre_grasp":
            gripper_events.append(
                _command_gripper_open(
                    args,
                    arm=arm,
                    action="pre_grasp_open",
                    reason="fixed_tail_pre_grasp",
                    phase_s=phase_s,
                    phase_fraction=phase_fraction,
                    tcp_goal_distance_m=None,
                )
            )
        if bool(args.harvest_gripper_schedule) and str(waypoint.get("name")) == "grasp":
            gripper_events.append(
                _command_gripper_close(
                    args,
                    arm=arm,
                    reason="fixed_tail_grasp",
                    phase_s=phase_s,
                    phase_fraction=phase_fraction,
                    tcp_goal_distance_m=None,
                )
            )
        if bool(args.harvest_gripper_schedule) and str(waypoint.get("name")) == "basket_place":
            gripper_events.append(
                _command_gripper_open(
                    args,
                    arm=arm,
                    action="basket_release_open",
                    reason="fixed_tail_basket_place",
                    phase_s=phase_s,
                    phase_fraction=phase_fraction,
                    tcp_goal_distance_m=None,
                )
            )
    return {
        "enabled": True,
        "executed": bool(executed),
        "plan": str(args.harvest_fixed_basket_tail_plan),
        "start_after": str(args.harvest_fixed_basket_tail_start_after),
        "minimum_time_s": float(args.harvest_fixed_basket_tail_minimum_time_s),
        "retarget": retarget_info,
        "live_transition_hard_gate": live_transition_hard_gate,
        "retarget_reports": retarget_reports,
        "waypoints": executed,
        "gripper_events": gripper_events,
        "ok": bool(executed) and all(bool(row.get("ok")) for row in executed),
    }


def load_deleaf_tail_waypoints(plan_path: Path, *, start_after: str) -> tuple[list[dict[str, Any]], Any]:
    """Load solved deleaf cut waypoints from a deleaf_click_cut plan JSON.

    Returns (tail_waypoints, plan_cut_point_robot_m). Waypoints live at .waypoints[]; the leaf cut
    point the plan was built for is selected_point.point_robot_m (the retarget anchor). The
    start_after sentinel begin/start/none/all returns the whole cut; otherwise the tail after the
    named phase is returned (e.g. start_after=approach -> pre_cut..return_transit).
    """
    payload = read_json(plan_path)
    waypoints = [item for item in payload.get("waypoints", []) if isinstance(item, dict)]
    if not waypoints:
        raise RuntimeError(f"deleaf cut plan has no waypoints: {plan_path}")
    anchor = None
    selected = payload.get("selected_point")
    if isinstance(selected, dict) and selected.get("point_robot_m") is not None:
        anchor = np.asarray(selected["point_robot_m"], dtype=float).reshape(3)
    if str(start_after).lower() in ("begin", "start", "none", "all"):
        return waypoints, anchor
    names = [str(item.get("name", "")) for item in waypoints]
    if start_after not in names:
        raise RuntimeError(f"deleaf cut plan has no start-after waypoint {start_after!r}; names={names}")
    start_index = names.index(start_after)
    tail = waypoints[start_index + 1 :]
    if not tail:
        raise RuntimeError(f"deleaf cut plan has no waypoints after {start_after!r}")
    return tail, anchor


def execute_fixed_deleaf_tail(
    *,
    args: argparse.Namespace,
    robot: Any,
    home: dict[str, Any],
    arm: str,
    kinematics: Any = None,
    locked_state: Any = None,
    live_goal_robot_m: Any = None,
) -> dict[str, Any]:
    """Deterministic deleaf cut tail (mirror of execute_fixed_harvest_tail; right arm, knife, NO gripper).

    Loads the matched leaf's solved cut waypoints, optionally retargets them to the live cut point by
    shifting every waypoint TCP by (live - plan_cut_point) and re-solving IK (tolerance fallback), and
    executes them with the gentle SDK velocity/acceleration caps.
    """
    if args.deleaf_fixed_tail_plan is None:
        return {"enabled": False, "executed": False}
    plan_path = Path(args.deleaf_fixed_tail_plan)
    tail_waypoints, plan_anchor = load_deleaf_tail_waypoints(
        plan_path, start_after=str(args.deleaf_fixed_tail_start_after)
    )
    retarget_enabled = (
        bool(args.deleaf_cut_retarget)
        and kinematics is not None
        and locked_state is not None
        and live_goal_robot_m is not None
        and plan_anchor is not None
    )
    delta_m = np.zeros(3, dtype=float)
    retarget_info: dict[str, Any] = {"enabled": bool(retarget_enabled)}
    if retarget_enabled:
        delta_m = np.asarray(live_goal_robot_m, dtype=float).reshape(3) - np.asarray(plan_anchor, dtype=float).reshape(3)
        retarget_info["plan_cut_point_robot_m"] = np.asarray(plan_anchor, dtype=float).tolist()
        retarget_info["live_goal_robot_m"] = np.asarray(live_goal_robot_m, dtype=float).reshape(3).tolist()
        retarget_info["delta_m"] = delta_m.tolist()
    vel = float(args.deleaf_tail_velocity_limit_rad_s) if args.deleaf_tail_velocity_limit_rad_s is not None else None
    acc = (
        float(args.deleaf_tail_acceleration_limit_rad_s2)
        if args.deleaf_tail_acceleration_limit_rad_s2 is not None
        else None
    )
    executed: list[dict[str, Any]] = []
    retarget_reports: list[dict[str, Any]] = []
    for local_index, waypoint in enumerate(tail_waypoints, start=1):
        q = waypoint.get("q_arm_rad")
        if q is None:
            raise RuntimeError(f"deleaf cut waypoint lacks q_arm_rad: {waypoint.get('name')}")
        q_arm = np.asarray(q, dtype=float).reshape(ARM_DOF)
        waypoint_name = str(waypoint.get("name", f"cut_{local_index}"))
        if retarget_enabled:
            q_arm, retarget_report = _retarget_tail_waypoint_q(
                waypoint=waypoint,
                kinematics=kinematics,
                locked_state=locked_state,
                delta_m=delta_m,
                insertion_dir=None,
                extra_insertion_m=0.0,
                height_offset_m=0.0,
                fallback_q=q_arm,
                orientation_tolerance_rad=float(args.deleaf_retarget_orientation_tol_rad),
            )
            retarget_reports.append(retarget_report)
        feedback = robot.send_command(
            command_arm_only(
                arm,
                q_arm,
                minimum_time_s=float(args.deleaf_fixed_tail_minimum_time_s),
                velocity_limit_rad_s=vel,
                acceleration_limit_rad_s2=acc,
            ),
            int(max(float(args.timeout_s), float(args.deleaf_fixed_tail_minimum_time_s) + 30.0)),
        ).get()
        ok = feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
        executed.append(
            {
                "step_index": int(local_index),
                "name": waypoint_name,
                "q_arm_rad": q_arm.astype(float).tolist(),
                "finish_code": str(feedback.finish_code),
                "ok": bool(ok),
            }
        )
        if not ok:
            break
    return {
        "enabled": True,
        "executed": bool(executed),
        "plan": str(plan_path),
        "start_after": str(args.deleaf_fixed_tail_start_after),
        "minimum_time_s": float(args.deleaf_fixed_tail_minimum_time_s),
        "retarget": retarget_info,
        "retarget_reports": retarget_reports,
        "waypoints": executed,
        "ok": bool(executed) and all(bool(row.get("ok")) for row in executed),
    }


def validate_locked_home_state(
    robot: Any,
    home: dict[str, Any],
    active_joint_indices: tuple[int, ...],
    *,
    tolerance_rad: float,
) -> dict[str, Any]:
    live = np.asarray(robot.get_state().position, dtype=float)
    expected = np.asarray(home["all_joints"]["positions_rad"], dtype=float)
    active = set(int(index) for index in active_joint_indices)
    indices = np.asarray([int(index) for index in commanded_joint_indices(home) if int(index) not in active], dtype=int)
    delta = live[indices] - expected[indices]
    abs_delta = np.abs(delta)
    local_max_index = int(np.argmax(abs_delta))
    max_index = int(indices[local_max_index])
    max_abs_delta = float(abs_delta[local_max_index])
    joint_names = list(home["joint_names"])
    if max_abs_delta > float(tolerance_rad):
        raise RuntimeError(
            "locked joints are not at home: "
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


def default_output_path(args: argparse.Namespace, metadata: dict[str, Any]) -> Path:
    if args.output_json is not None:
        return args.output_json
    run_label = args.run_label
    if run_label is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_label = f"{metadata.get('primitive', 'policy')}_{stamp}"
    return args.rollout_root / str(metadata.get("primitive", "policy")) / f"{run_label}.json"


def target_capture_run_label(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    if args.run_label:
        return f"{args.run_label}_target"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{metadata.get('primitive', 'policy')}_bc_target_{stamp}"


def capture_realsense_goal(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    transform_json = args.transform_json or default_transform_json(args.dataset_name)
    run_label = target_capture_run_label(args, metadata)
    capture = capture_realsense(
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        warmup_frames=int(args.warmup_frames),
        serial=args.serial,
    )
    point_cloud = build_point_cloud(
        color_bgr=capture["color_bgr"],
        depth_m=capture["depth_m"],
        intrinsics=capture["intrinsics"],
        min_depth_m=float(args.min_depth_m),
        max_depth_m=float(args.max_depth_m),
        stride=int(args.pointcloud_stride),
    )
    capture_metadata = {
        "method": "guarded_bc_rollout_realsense_target_capture",
        "dataset_name": args.dataset_name,
        "run_label": run_label,
        "target_source": args.target_source,
        "transform_json": str(transform_json),
        "point_count": int(point_cloud["points"].shape[0]),
        "intrinsics": capture["intrinsics"],
        "frame_metadata": capture["frame_metadata"],
    }
    capture_paths = write_capture_artifacts(
        dataset_name=args.dataset_name,
        run_label=run_label,
        runs_root=args.runs_root,
        capture=capture,
        point_cloud=point_cloud,
        metadata=capture_metadata,
    )

    picker_payload: dict[str, Any] | None = None
    if args.target_source == "realsense-open3d":
        picker_json = artifact_path(
            "live_realsense_capture",
            f"{args.dataset_name}_{run_label}_selection",
            runs_root=args.runs_root,
        )
        picker_payload = run_external_open3d_picker(
            surfel_python=args.surfel_python,
            capture_npz=Path(capture_paths["npz"]),
            transform_json=transform_json,
            dataset_name=args.dataset_name,
            axis_size_m=float(args.axis_size_m),
            output_json=picker_json,
            confirm_execute=False,
        )
        selected = dict(picker_payload["selected_point"])
        selected["selection_source"] = "external_open3d_capture"
        point_robot = selected.get("point_robot_m")
    elif args.target_source == "realsense-rgb":
        pixel_xy = None
        if args.pixel_x is not None:
            pixel_xy = (int(args.pixel_x), int(args.pixel_y))
        selected = select_point(
            capture=capture,
            point_cloud=point_cloud,
            pick_mode="rgb",
            pixel_xy=pixel_xy,
            pick_radius_px=int(args.pick_radius_px),
            min_depth_m=float(args.min_depth_m),
            max_depth_m=float(args.max_depth_m),
            axis_size_m=float(args.axis_size_m),
        )
        point_robot = point_to_robot(np.asarray(selected["point_realsense_m"], dtype=float), transform_json)
    else:
        raise ValueError(f"unsupported RealSense target source: {args.target_source}")

    if point_robot is None:
        raise RuntimeError("RealSense target selection did not produce a robot-frame point")
    selected["point_robot_m"] = [float(value) for value in np.asarray(point_robot, dtype=float).reshape(3)]

    return np.asarray(point_robot, dtype=float).reshape(3), {
        "target_source": args.target_source,
        "dataset_name": args.dataset_name,
        "run_label": run_label,
        "transform_json": str(transform_json),
        "capture_artifacts": capture_paths,
        "picker_payload": picker_payload,
        "selected_point": selected,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata, arrays = load_policy(args.policy_dir)
    primitive = str(metadata.get("primitive", "")).strip().lower()
    arm = default_arm_for_policy(metadata) if args.arm == "auto" else str(args.arm)
    toolfit = Path(args.toolfit) if args.toolfit is not None else default_toolfit_for_arm(arm)
    target_report: dict[str, Any]
    if args.target_source == "goal":
        goal_robot_m = parse_vector(args.goal_robot_m, length=3, name="--goal-robot-m")
        if goal_robot_m is None:
            raise ValueError("--goal-robot-m is required unless --target-source is realsense-open3d or realsense-rgb")
        target_report = {
            "target_source": "goal",
            "selected_point": {
                "selection_source": "cli_goal_robot_m",
                "point_robot_m": goal_robot_m.astype(float).tolist(),
            },
        }
    else:
        goal_robot_m, target_report = capture_realsense_goal(args, metadata)

    deleaf_facing = resolve_deleaf_facing(args.deleaf_label, args.deleaf_facing)
    if policy_uses_deleaf_side_features(metadata) and deleaf_facing is None:
        raise ValueError("policy was trained with deleaf side features; pass --deleaf-label or --deleaf-facing")

    goal_snapping: dict[str, Any] = {
        "enabled": bool(args.snap_goal_to_nearest_demo),
        "snapped": False,
    }
    if args.snap_goal_to_nearest_demo:
        snap_reference = gcsl_reference_duration(metadata, goal_robot_m)
        selected = snap_reference.get("selected_demo")
        distance = None if selected is None else selected.get("goal_distance_m")
        if selected is not None and distance is not None and float(distance) <= float(args.snap_goal_max_distance_m):
            raw_goal = goal_robot_m.astype(float).copy()
            goal_robot_m = np.asarray(selected["goal_robot_m"], dtype=float).reshape(3)
            goal_snapping.update(
                {
                    "snapped": True,
                    "raw_goal_robot_m": raw_goal.tolist(),
                    "policy_goal_robot_m": goal_robot_m.astype(float).tolist(),
                    "selected_demo_label": selected.get("label"),
                    "distance_m": float(distance),
                    "max_distance_m": float(args.snap_goal_max_distance_m),
                }
            )
        else:
            goal_snapping.update(
                {
                    "reason": "no_demo_goal_within_threshold",
                    "nearest_demo": selected,
                    "max_distance_m": float(args.snap_goal_max_distance_m),
                }
            )
    target_report["policy_goal_snapping"] = goal_snapping

    if (
        args.full_action
        or policy_uses_gcsl_features(metadata)
        or args.harvest_auto_fixed_basket_tail
        or args.deleaf_auto_fixed_tail
    ):
        reference_duration = gcsl_reference_duration(metadata, goal_robot_m)
    else:
        reference_duration = {
            "duration_s": float(metadata.get("chunk_horizon_s", 0.0)),
            "source": "chunk_horizon",
            "selected_demo": None,
            "candidates": [],
        }
    if args.harvest_auto_fixed_basket_tail and args.harvest_fixed_basket_tail_plan is None:
        selected_label = (reference_duration.get("selected_demo") or {}).get("label")
        tail_plan = default_harvest_tail_plan_for_label(selected_label)
        if tail_plan is not None:
            args.harvest_fixed_basket_tail_plan = tail_plan
    if args.deleaf_auto_fixed_tail and args.deleaf_fixed_tail_plan is None:
        selected_label = (reference_duration.get("selected_demo") or {}).get("label")
        deleaf_tail_plan = default_deleaf_tail_plan_for_label(selected_label)
        if deleaf_tail_plan is not None:
            args.deleaf_fixed_tail_plan = deleaf_tail_plan
    phase_duration_s_for_rollout = max(
        float(reference_duration["duration_s"]) * (float(args.full_action_duration_scale) if args.full_action else 1.0),
        1e-6,
    )
    if args.full_action:
        if args.recede_iterations is None:
            sample_hz = float(metadata["sample_hz"])
            predicted_steps = int(math.ceil(phase_duration_s_for_rollout * sample_hz))
            args.recede_iterations = max(1, int(math.ceil(predicted_steps / max(1, int(args.steps_to_execute)))))
            args.recede_iterations += int(args.extra_recede_iterations)
    elif args.recede_iterations is None:
        args.recede_iterations = 1

    home = read_json(args.home)
    robot = None
    kinematics = None
    locked_state = None
    q_full = None
    home_check = None
    locked_home_check = None
    gripper_preflight = None
    gripper_events: list[dict[str, Any]] = []
    gripper_schedule_state = {
        "pre_grasp_opened": False,
        "grasp_closed": False,
        "basket_released": False,
    }
    if args.execute and args.skip_fk_guards:
        raise RuntimeError("--execute cannot be used with --skip-fk-guards")

    if args.execute and (args.q_arm_rad is not None or args.tcp_position_robot_m is not None):
        raise RuntimeError("--execute must use live robot state; remove --q-arm-rad/--tcp-position-robot-m overrides")

    if args.harvest_gripper_schedule:
        if primitive != "harvest":
            raise RuntimeError("--harvest-gripper-schedule is only valid for harvest policies")
        if arm != "left":
            raise RuntimeError("--harvest-gripper-schedule currently expects the left harvest gripper")

    if not args.skip_fk_guards:
        if args.backend == "sim":
            from robot.policy.isaac_sim_backend import SimRobot

            robot = SimRobot(home_path=Path(args.home), arm=arm, model_name=args.model, toolfit_path=toolfit)
            args._sim_robot = robot
        else:
            robot = (
                connect_motion_robot(address=args.address, model=args.model, power=args.power, servo=args.servo)
                if args.execute
                else connect_state_robot(address=args.address, model=args.model)
            )
        kinematics = RBY1Kinematics.from_sdk_model(robot.get_dynamics(), robot.model(), arm=arm, toolfit_path=toolfit)
        locked_state = kinematics.locked_state_from_home(args.home)
        if args.require_home:
            home_check = validate_home_state(robot, home, tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)))
        elif args.execute:
            locked_home_check = validate_locked_home_state(
                robot,
                home,
                tuple(kinematics.active_arm.joint_indices),
                tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)),
            )
        if args.execute and args.harvest_gripper_schedule and args.backend != "sim":
            gripper_preflight = _require_gripper_bridge(args.gripper_url, timeout_s=float(args.gripper_timeout_s))

    iteration_count = max(1, int(args.recede_iterations if args.execute else 1))
    rollout_iterations: list[dict[str, Any]] = []
    executed_prefix: list[dict[str, Any]] = []
    execution_error = None
    policy_handoff_distance_arg = (
        args.harvest_policy_handoff_distance_m
        if primitive == "harvest"
        else (args.deleaf_policy_handoff_distance_m if primitive == "deleaf" else None)
    )
    policy_handoff_geometry: dict[str, Any] = {
        "source": "explicit_target_distance",
        "threshold_m": policy_handoff_distance_arg,
    }
    if primitive == "harvest" and args.harvest_policy_handoff_plan_margin_m is not None:
        policy_handoff_geometry = {
            "source": "phase_plan_approach",
            "margin_m": float(args.harvest_policy_handoff_plan_margin_m),
            "deferred_until_execution": not bool(args.execute),
        }
        if args.execute:
            if args.harvest_fixed_basket_tail_plan is None or goal_robot_m is None:
                raise RuntimeError(
                    "plan-derived harvest handoff requires a fixed phase plan and robot-frame goal"
                )
            policy_handoff_distance_arg, policy_handoff_geometry = harvest_plan_handoff_threshold(
                Path(args.harvest_fixed_basket_tail_plan),
                candidate_name=args.harvest_fixed_basket_tail_candidate_name,
                goal_robot_m=goal_robot_m,
                margin_m=float(args.harvest_policy_handoff_plan_margin_m),
            )
    policy_handoff = {
        "enabled": bool(
            policy_handoff_distance_arg is not None
            or args.harvest_policy_handoff_plan_margin_m is not None
        ),
        "triggered": False,
        **policy_handoff_geometry,
    }
    for iteration_index in range(iteration_count):
        if args.skip_fk_guards:
            q_full = None
            q_arm, dq_arm, tcp_position = offline_state(args)
        else:
            if robot is None or kinematics is None:
                raise RuntimeError("live rollout requires robot kinematics")
            q_full, q_arm, dq_arm, tcp_position = live_state(robot=robot, kinematics=kinematics, args=args)

        if (
            args.execute
            and policy_handoff_distance_arg is not None
            and goal_robot_m is not None
        ):
            handoff_distance_m = float(
                np.linalg.norm(
                    np.asarray(tcp_position, dtype=float).reshape(3)
                    - np.asarray(goal_robot_m, dtype=float).reshape(3)
                )
            )
            if handoff_distance_m <= float(policy_handoff_distance_arg):
                policy_handoff = {
                    "enabled": True,
                    "triggered": True,
                    "iteration_index": int(iteration_index + 1),
                    "tcp_goal_distance_m": handoff_distance_m,
                    "threshold_m": float(policy_handoff_distance_arg),
                    **policy_handoff_geometry,
                }
                break

        sample_hz = float(metadata["sample_hz"])
        counter_phase_s = float(len(executed_prefix)) / max(sample_hz, 1e-6)
        phase_s, phase_report = retimed_phase_from_training_state(
            metadata=metadata,
            arrays=arrays,
            q_arm=q_arm,
            tcp_position=tcp_position,
            phase_source=str(args.phase_source),
            counter_phase_s=counter_phase_s,
        )
        phase_duration_s = phase_duration_s_for_rollout
        phase_fraction = min(max(phase_s / phase_duration_s, 0.0), 1.0)
        horizon_s = max(phase_duration_s - phase_s, 0.0)
        horizon_fraction = min(max(horizon_s / phase_duration_s, 0.0), 1.0)
        delta_chunk, q_chunk, eef_decode_summary, pred_eef_chunk = predict_chunk(
            metadata=metadata,
            arrays=arrays,
            goal_robot_m=goal_robot_m,
            q_arm=q_arm,
            dq_arm=dq_arm,
            tcp_position=tcp_position,
            phase_s=phase_s,
            phase_fraction=phase_fraction,
            horizon_s=horizon_s,
            horizon_fraction=horizon_fraction,
            deleaf_facing=deleaf_facing,
            kinematics=kinematics,
            locked_state=locked_state,
            decode_steps=max(
                int(args.steps_to_execute),
                int(args.guard_horizon_steps if args.guard_horizon_steps is not None else args.steps_to_execute),
            ),
        )
        raw_delta_chunk = np.array(delta_chunk, copy=True)
        raw_q_chunk = np.array(q_chunk, copy=True)
        clip_report = {"enabled": False}
        joint_limit_clip_report = {"enabled": False}
        if args.clip_step_delta:
            q_chunk, clip_report = clip_q_chunk_step_delta(
                q_arm=q_arm,
                q_chunk=q_chunk,
                max_step_delta_rad=float(args.max_step_delta_rad),
            )
            delta_chunk = q_chunk - q_arm.reshape(1, ARM_DOF)
        if args.clip_joint_limits:
            if kinematics is None or locked_state is None:
                raise RuntimeError("--clip-joint-limits requires live FK guards")
            q_chunk, joint_limit_clip_report = clip_q_chunk_joint_limits(
                kinematics=kinematics,
                locked_state=locked_state,
                q_arm=q_arm,
                q_chunk=q_chunk,
                margin_rad=float(args.joint_limit_margin_rad),
            )
            if args.clip_step_delta:
                q_chunk, clip_report = clip_q_chunk_step_delta(
                    q_arm=q_arm,
                    q_chunk=q_chunk,
                    max_step_delta_rad=float(args.max_step_delta_rad),
                )
                q_chunk, joint_limit_clip_report = clip_q_chunk_joint_limits(
                    kinematics=kinematics,
                    locked_state=locked_state,
                    q_arm=q_arm,
                    q_chunk=q_chunk,
                    margin_rad=float(args.joint_limit_margin_rad),
                )
            delta_chunk = q_chunk - q_arm.reshape(1, ARM_DOF)
        original_disable_progress_guard = bool(args.disable_progress_guard)
        if args.full_action and int(iteration_index) >= int(args.progress_guard_iterations):
            args.disable_progress_guard = True
        guards = evaluate_guards(
            metadata=metadata,
            kinematics=kinematics,
            locked_state=locked_state,
            q_arm=q_arm,
            q_chunk=q_chunk,
            goal_robot_m=goal_robot_m,
            tcp_position=tcp_position,
            args=args,
        )
        args.disable_progress_guard = original_disable_progress_guard
        eef_ik_guard: dict[str, Any] = {"enabled": eef_decode_summary is not None, "ok": True}
        if eef_decode_summary is not None:
            ik_max_pos = float(eef_decode_summary.get("max_position_error_m", 0.0))
            ik_max_rot = float(eef_decode_summary.get("max_orientation_error_rad", 0.0))
            eef_ik_guard.update(
                {
                    "summary": eef_decode_summary,
                    "max_position_error_m": ik_max_pos,
                    "max_orientation_error_rad": ik_max_rot,
                    "all_ik_ok": bool(eef_decode_summary.get("all_ik_ok", False)),
                    "all_within_tolerance": bool(eef_decode_summary.get("all_within_tolerance", False)),
                    "max_position_error_limit_m": float(args.eef_ik_max_position_error_m),
                    "ok": bool(
                        eef_decode_summary.get("all_ik_ok", False)
                        and ik_max_pos <= float(args.eef_ik_max_position_error_m)
                    ),
                }
            )
        handoff_after_prefix = False
        handoff_prefix_report: dict[str, Any] | None = None
        steps_to_execute_this_iteration = int(args.steps_to_execute)
        if (
            args.execute
            and policy_handoff_distance_arg is not None
            and goal_robot_m is not None
            and kinematics is not None
            and locked_state is not None
        ):
            handoff_prefix_report = safe_prefix_before_target_distance(
                kinematics=kinematics,
                locked_state=locked_state,
                q_chunk=q_chunk,
                goal_robot_m=goal_robot_m,
                threshold_m=float(policy_handoff_distance_arg),
                prefix_limit=int(args.steps_to_execute),
            )
            if handoff_prefix_report["crosses"]:
                handoff_after_prefix = True
                steps_to_execute_this_iteration = int(
                    handoff_prefix_report["safe_prefix_count"]
                )
                policy_handoff = {
                    "enabled": True,
                    "triggered": True,
                    "trigger": "predicted_prefix_crossing",
                    "iteration_index": int(iteration_index + 1),
                    "tcp_goal_distance_m": float(
                        np.linalg.norm(
                            np.asarray(tcp_position, dtype=float).reshape(3)
                            - np.asarray(goal_robot_m, dtype=float).reshape(3)
                        )
                    ),
                    "safe_prefix_count": steps_to_execute_this_iteration,
                    "prediction": handoff_prefix_report,
                    **policy_handoff_geometry,
                }
        iteration_executed: list[dict[str, Any]] = []
        iteration_gripper_events: list[dict[str, Any]] = []
        iteration_error = None
        if args.execute:
            if not guards["ok"]:
                execution_error = "guard_failed"
                iteration_error = "guard_failed"
            elif not eef_ik_guard["ok"]:
                execution_error = "eef_ik_decode_failed"
                iteration_error = "eef_ik_decode_failed"
            elif robot is None:
                raise RuntimeError("--execute requires live robot connection; remove --skip-fk-guards")
            else:
                if (
                    args.harvest_gripper_schedule
                    and policy_owns_harvest_gripper_schedule(args)
                ):
                    try:
                        iteration_gripper_events = maybe_apply_harvest_gripper_schedule(
                            args=args,
                            arm=arm,
                            state=gripper_schedule_state,
                            goal_robot_m=goal_robot_m,
                            tcp_position=tcp_position,
                            phase_s=phase_s,
                            phase_fraction=phase_fraction,
                        )
                        for event in iteration_gripper_events:
                            event["iteration_index"] = int(iteration_index + 1)
                            gripper_events.append(event)
                    except Exception as exc:
                        execution_error = "gripper_failed"
                        iteration_error = f"gripper_failed: {exc}"
                if iteration_error is None and steps_to_execute_this_iteration > 0:
                    iteration_executed = execute_prefix(
                        robot=robot,
                        home=home,
                        arm=arm,
                        q_chunk=q_chunk,
                        steps_to_execute=steps_to_execute_this_iteration,
                        minimum_time_s=float(args.minimum_time_s),
                        timeout_s=float(args.timeout_s),
                        inter_step_sleep_s=float(args.inter_step_sleep_s),
                        execute_prefix_mode=str(args.execute_prefix_mode),
                        velocity_limit_rad_s=(
                            float(args.policy_velocity_limit_rad_s)
                            if args.policy_velocity_limit_rad_s is not None
                            else None
                        ),
                        acceleration_limit_rad_s2=(
                            float(args.policy_acceleration_limit_rad_s2)
                            if args.policy_acceleration_limit_rad_s2 is not None
                            else None
                        ),
                    )
                for row in iteration_executed:
                    row["iteration_index"] = int(iteration_index + 1)
                    row["global_step_index"] = int(len(executed_prefix) + 1)
                    executed_prefix.append(row)
                if any(not bool(row.get("ok", False)) for row in iteration_executed):
                    execution_error = "command_failed"
                    iteration_error = "command_failed"
        rollout_iterations.append(
            {
                "iteration_index": int(iteration_index + 1),
                "input": {
                    "goal_robot_m": goal_robot_m,
                    "target": target_report,
                    "q_arm_rad": q_arm,
                    "dq_arm_rad_s": dq_arm,
                    "tcp_position_robot_m": tcp_position,
                    "q_full_live_rad": q_full,
                    "counter_phase_s": float(counter_phase_s),
                    "phase_s": float(phase_s),
                    "phase_fraction": float(phase_fraction),
                    "phase_report": phase_report,
                    "gcsl_horizon_s": float(horizon_s),
                    "gcsl_horizon_fraction": float(horizon_fraction),
                    "deleaf_facing": deleaf_facing,
                },
                "prediction": {
                    "delta_q_arm_chunk_rad": delta_chunk,
                    "q_arm_chunk_rad": q_chunk,
                    "raw_delta_q_arm_chunk_rad": raw_delta_chunk if args.clip_step_delta else None,
                    "raw_q_arm_chunk_rad": raw_q_chunk if args.clip_step_delta or args.clip_joint_limits else None,
                    "step_delta_clip": clip_report,
                    "joint_limit_clip": joint_limit_clip_report,
                    "pred_eef_chunk": pred_eef_chunk,
                    "handoff_safe_prefix": handoff_prefix_report,
                },
                "guards": guards,
                "eef_ik_guard": eef_ik_guard,
                "gripper_events": iteration_gripper_events,
                "executed_prefix": iteration_executed,
                "execution_error": iteration_error,
            }
        )
        stop_after_grasp = bool(
            args.execute
            and args.harvest_gripper_schedule
            and args.harvest_stop_policy_on_grasp
            and gripper_schedule_state.get("grasp_closed")
        )
        if (
            iteration_error is not None
            or not args.execute
            or stop_after_grasp
            or handoff_after_prefix
        ):
            break

    fixed_basket_tail = {"enabled": False, "executed": False}
    if (
        args.execute
        and execution_error is None
        and args.harvest_fixed_basket_tail_plan is not None
        and policy_handoff.get("enabled", False)
        and not policy_handoff.get("triggered", False)
    ):
        execution_error = "fixed_tail_skipped_handoff_not_triggered"
        fixed_basket_tail = {
            "enabled": True,
            "executed": False,
            "reason": "policy_handoff_not_triggered",
        }
    if (
        args.execute
        and execution_error is None
        and args.harvest_fixed_basket_tail_plan is not None
        and robot is not None
    ):
        if args.harvest_fixed_basket_tail_require_closed and not gripper_schedule_state["grasp_closed"]:
            execution_error = "fixed_tail_skipped_gripper_not_closed"
            fixed_basket_tail = {
                "enabled": True,
                "executed": False,
                "reason": "gripper_not_closed",
            }
        else:
            try:
                fixed_basket_tail = execute_fixed_harvest_tail(
                    args=args,
                    robot=robot,
                    home=home,
                    arm=arm,
                    phase_s=float(len(executed_prefix)) / max(float(metadata["sample_hz"]), 1e-6),
                    phase_fraction=1.0,
                    kinematics=kinematics,
                    locked_state=locked_state,
                    live_goal_robot_m=(
                        goal_snapping.get("raw_goal_robot_m")
                        if isinstance(goal_snapping, dict) and goal_snapping.get("raw_goal_robot_m") is not None
                        else goal_robot_m
                    ),
                )
                for event in fixed_basket_tail.get("gripper_events", []):
                    event["iteration_index"] = "fixed_tail"
                    gripper_events.append(event)
                    action = str(event.get("action", ""))
                    if action == "pre_grasp_open":
                        gripper_schedule_state["pre_grasp_opened"] = True
                    elif action == "grasp_close":
                        gripper_schedule_state["grasp_closed"] = True
                    elif action == "basket_release_open":
                        gripper_schedule_state["basket_released"] = True
                if not bool(fixed_basket_tail.get("ok", False)):
                    execution_error = "fixed_tail_failed"
            except Exception as exc:
                execution_error = "fixed_tail_failed"
                fixed_basket_tail = {
                    "enabled": True,
                    "executed": False,
                    "reason": str(exc),
                }

    fixed_deleaf_tail = {"enabled": False, "executed": False}
    if (
        args.execute
        and execution_error is None
        and args.deleaf_fixed_tail_plan is not None
        and robot is not None
    ):
        try:
            fixed_deleaf_tail = execute_fixed_deleaf_tail(
                args=args,
                robot=robot,
                home=home,
                arm=arm,
                kinematics=kinematics,
                locked_state=locked_state,
                live_goal_robot_m=(
                    goal_snapping.get("raw_goal_robot_m")
                    if isinstance(goal_snapping, dict) and goal_snapping.get("raw_goal_robot_m") is not None
                    else goal_robot_m
                ),
            )
            if not bool(fixed_deleaf_tail.get("ok", False)):
                execution_error = "deleaf_tail_failed"
        except Exception as exc:
            execution_error = "deleaf_tail_failed"
            fixed_deleaf_tail = {"enabled": True, "executed": False, "reason": str(exc)}

    post_state = None
    if robot is not None and args.execute and executed_prefix:
        state = robot.get_state()
        q_live = np.asarray(state.position, dtype=float).reshape(-1)
        indices = list(kinematics.active_arm.joint_indices) if kinematics is not None else []
        post_state = {
            "q_arm_rad": q_live[indices].astype(float).tolist() if indices else None,
        }
    if rollout_iterations:
        last_iteration = rollout_iterations[-1]
        first_iteration = rollout_iterations[0]
    else:
        # immediate policy handoff (e.g. --harvest-policy-handoff-distance-m large):
        # the harvest is performed entirely by the deterministic fixed tail, so no
        # policy iteration was recorded. Synthesize a minimal stand-in.
        last_iteration = {
            "iteration_index": 0,
            "input": {"target": target_report},
            "prediction": {},
            "guards": {"ok": True, "fallback_recommended": False, "failed": []},
            "execution_error": execution_error,
        }
        first_iteration = last_iteration

    policy_motion_executed = bool(executed_prefix)
    fixed_tail_motion_executed = bool(
        fixed_basket_tail.get("executed", False) or fixed_deleaf_tail.get("executed", False)
    )
    if policy_motion_executed and fixed_tail_motion_executed:
        controller_execution_mode = "hybrid_policy_and_fixed_tail"
    elif policy_motion_executed:
        controller_execution_mode = "policy_only"
    elif fixed_tail_motion_executed:
        controller_execution_mode = "deterministic_tail_only"
    else:
        controller_execution_mode = "no_motion"
    payload = {
        "schema": "egodelta_guarded_bc_rollout_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_dir": args.policy_dir,
        "primitive": metadata.get("primitive"),
        "arm": arm,
        "toolfit": toolfit,
        "mode": "bc",
        "sample_hz": float(metadata["sample_hz"]),
        "dt_s": float(1.0 / float(metadata["sample_hz"])),
        "chunk_size": int(metadata["chunk_size"]),
        "gcsl_reference_duration": reference_duration,
        "phase_duration_s_for_rollout": float(phase_duration_s_for_rollout),
        "phase_source": str(args.phase_source),
        "policy_goal_snapping": goal_snapping,
        "policy_handoff": policy_handoff,
        "steps_to_execute": int(args.steps_to_execute),
        "execute_prefix_mode": str(args.execute_prefix_mode),
        "full_action": bool(args.full_action),
        "recede_iterations_requested": int(iteration_count),
        "recede_iterations_completed": int(len(rollout_iterations)),
        "guard_horizon_steps": int(args.guard_horizon_steps if args.guard_horizon_steps is not None else args.steps_to_execute),
        "runtime_config": {
            "max_step_delta_rad": float(args.max_step_delta_rad),
            "max_step_delta_deg": float(math.degrees(float(args.max_step_delta_rad))),
            "clip_step_delta": bool(args.clip_step_delta),
            "clip_joint_limits": bool(args.clip_joint_limits),
            "minimum_time_s": float(args.minimum_time_s),
            "policy_velocity_limit_rad_s": (
                None
                if args.policy_velocity_limit_rad_s is None
                else float(args.policy_velocity_limit_rad_s)
            ),
            "policy_acceleration_limit_rad_s2": (
                None
                if args.policy_acceleration_limit_rad_s2 is None
                else float(args.policy_acceleration_limit_rad_s2)
            ),
            "inter_step_sleep_s": float(args.inter_step_sleep_s),
            "execute_prefix_mode": str(args.execute_prefix_mode),
            "phase_source": str(args.phase_source),
            "progress_tolerance_m": float(args.progress_tolerance_m),
            "progress_guard_iterations": int(args.progress_guard_iterations),
            "disable_progress_guard": bool(args.disable_progress_guard),
            "path_samples_per_step": int(args.path_samples_per_step),
            "min_clearance_m": float(args.min_clearance_m),
        },
        "harvest_gripper_schedule": {
            "enabled": bool(args.harvest_gripper_schedule),
            "policy_owns_schedule": bool(policy_owns_harvest_gripper_schedule(args)),
            "gripper_url": args.gripper_url,
            "close_distance_m": float(args.harvest_gripper_close_distance_m),
            "close_phase": float(args.harvest_gripper_close_phase),
            "phase_close_max_distance_m": float(args.harvest_gripper_phase_close_max_distance_m),
            "release_phase": float(args.harvest_gripper_release_phase),
            "state": gripper_schedule_state,
        },
        "gripper_preflight": gripper_preflight,
        "gripper_events": gripper_events,
        "fixed_basket_tail": fixed_basket_tail,
        "fixed_deleaf_tail": fixed_deleaf_tail,
        "dry_run_scope": "first_prefix_only" if not args.execute else None,
        "execute_requested": bool(args.execute),
        "executed": bool(policy_motion_executed or fixed_tail_motion_executed),
        "policy_motion_executed": policy_motion_executed,
        "fixed_tail_motion_executed": fixed_tail_motion_executed,
        "controller_execution_mode": controller_execution_mode,
        "joint_group_scope": f"{arm}_arm_only",
        "execution_error": execution_error,
        "fallback_recommended": bool(last_iteration["guards"]["fallback_recommended"] or execution_error),
        "home_check": home_check,
        "locked_home_check": locked_home_check,
        "input": last_iteration["input"],
        "prediction": last_iteration["prediction"],
        "guards": last_iteration["guards"],
        "first_iteration": first_iteration,
        "last_iteration": last_iteration,
        "rollout_iterations": rollout_iterations,
        "executed_prefix": executed_prefix,
        "post_state": post_state,
        "note": (
            "Prediction rows are cumulative future q_arm states relative to q_now; "
            "step_delta guard is computed from consecutive q_arm states."
        ),
    }
    output_path = default_output_path(args, metadata)
    if not args.no_write_log:
        payload["output_json"] = output_path
        write_json(output_path, payload)
    if args.backend == "sim" and getattr(robot, "is_sim_backend", False):
        payload["sim_replay"] = robot.replay_into_sim(
            segment_s=float(args.sim_segment_s), interp_steps=int(args.sim_interp_steps),
            joint_vel_deg_s=float(args.sim_joint_vel_deg_s),
        )
    return jsonable(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded receding-horizon rollout wrapper for trained EgoDelta BC action-chunk policies."
    )
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--target-source",
        choices=("goal", "realsense-open3d", "realsense-rgb"),
        default="goal",
        help="Use a supplied robot-frame goal or pick one from a live RealSense capture.",
    )
    parser.add_argument("--goal-robot-m", default=None, help="Comma-separated x,y,z target in robot frame.")
    parser.add_argument(
        "--snap-goal-to-nearest-demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the nearest stored demo goal as the policy input when the clicked goal is within --snap-goal-max-distance-m. Enabled by default for harvest and deleaf; disable with --no-snap-goal-to-nearest-demo.",
    )
    parser.add_argument("--snap-goal-max-distance-m", type=float, default=0.02)
    parser.add_argument("--arm", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--toolfit", type=Path, default=None)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--power", default=DEFAULT_POWER)
    parser.add_argument("--servo", default=DEFAULT_SERVO)
    parser.add_argument("--q-arm-rad", default=None, help="Optional 7-DoF current arm state override.")
    parser.add_argument("--dq-arm-rad-s", default=None, help="Optional 7-DoF arm velocity override; zeros if omitted.")
    parser.add_argument("--tcp-position-robot-m", default=None, help="Optional current TCP x,y,z override.")
    parser.add_argument("--deleaf-label", default=None, help="Known deleaf target label such as BD1 or BD3; used by side-conditioned policies.")
    parser.add_argument("--deleaf-facing", choices=("left", "right"), default=None, help="Stem-facing side for side-conditioned deleaf policies.")
    parser.add_argument("--skip-fk-guards", action="store_true", help="Prediction/delta guard only; no robot connection.")
    parser.add_argument("--require-home", action="store_true", help="Validate home state before guard/execution.")
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--phase-source",
        choices=("counter", "nearest_training_q", "nearest_training_tcp"),
        default="counter",
        help=(
            "Phase feature source for phase-conditioned policies. counter uses executed step count; "
            "nearest_training_q/tcp retime phase from the closest stored training state."
        ),
    )
    parser.add_argument("--steps-to-execute", type=int, default=5, help="Receding-horizon prefix length to command.")
    parser.add_argument(
        "--recede-iterations",
        type=int,
        default=None,
        help="Number of guarded receding-horizon prefixes to execute after one target pick.",
    )
    parser.add_argument(
        "--full-action",
        action="store_true",
        help=(
            "Run a full action budget after one target pick. Keeps the receding prefix length and "
            "defaults to the nearest training-demo duration."
        ),
    )
    parser.add_argument(
        "--full-action-duration-scale",
        type=float,
        default=1.1,
        help="Multiplier on the selected training-demo duration when auto-sizing --full-action iterations.",
    )
    parser.add_argument(
        "--extra-recede-iterations",
        type=int,
        default=5,
        help="Extra guarded prefixes appended when auto-sizing --full-action iterations.",
    )
    parser.add_argument(
        "--guard-horizon-steps",
        type=int,
        default=None,
        help="Predicted steps checked by blocking guards; defaults to --steps-to-execute.",
    )
    parser.add_argument(
        "--eef-ik-max-position-error-m",
        type=float,
        default=0.03,
        help=(
            "EEF action-space only: halt before motion if any IK-decoded chunk step has TCP position "
            "error above this (m) or if IK did not converge. Guards against commanding unreachable poses."
        ),
    )
    parser.add_argument("--max-step-delta-rad", type=float, default=0.017)
    parser.add_argument(
        "--clip-step-delta",
        action="store_true",
        help="Clip the commanded BC chunk to --max-step-delta-rad before guard checks and execution.",
    )
    parser.add_argument("--joint-limit-margin-rad", type=float, default=0.01)
    parser.add_argument(
        "--clip-joint-limits",
        action="store_true",
        help="Clamp the commanded BC chunk inside active joint limits before guard checks and execution.",
    )
    parser.add_argument("--minimum-time-s", type=float, default=0.10)
    parser.add_argument(
        "--execute-prefix-mode",
        choices=("all_steps", "endpoint"),
        default="all_steps",
        help=(
            "all_steps commands each predicted row; endpoint commands only the final row of each "
            "receding prefix, which is faster and usually smoother but less faithful to the chunk path."
        ),
    )
    parser.add_argument("--inter-step-sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--policy-velocity-limit-rad-s",
        type=float,
        default=None,
        help="Per-joint velocity cap (rad/s) SDK-enforced on the learned-policy execute_prefix moves (smooths deleaf/harvest policy motion). ~0.25 is gentle.",
    )
    parser.add_argument(
        "--policy-acceleration-limit-rad-s2",
        type=float,
        default=None,
        help="Per-joint acceleration cap (rad/s^2) SDK-enforced on the learned-policy execute_prefix moves (softer ramps). ~0.1 is gentle.",
    )
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--harvest-gripper-schedule",
        action="store_true",
        help="For harvest rollouts, deterministically open at pre-grasp, close at grasp, and open at basket release.",
    )
    parser.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument(
        "--gripper-target-source",
        choices=("live_status", "absolute"),
        default="live_status",
        help="Derive open/closed encoder targets from live /status bounds, or use explicit absolute CLI values.",
    )
    parser.add_argument("--gripper-timeout-s", type=float, default=2.0)
    parser.add_argument("--left-open-position", type=float, default=DEFAULT_LEFT_OPEN_POSITION)
    parser.add_argument("--left-open-reached-position", type=float, default=DEFAULT_LEFT_OPEN_REACHED_POSITION)
    parser.add_argument("--left-open-tolerance", type=float, default=0.15)
    parser.add_argument("--left-close-position", type=float, default=DEFAULT_LEFT_CLOSE_POSITION)
    parser.add_argument("--gripper-open-current-limit", type=float, default=3.0)
    parser.add_argument("--gripper-close-current-limit", type=float, default=2.0)
    parser.add_argument("--gripper-hold-s", type=float, default=1.0)
    parser.add_argument("--gripper-post-command-wait-s", type=float, default=3.0)
    parser.add_argument("--gripper-post-command-poll-s", type=float, default=0.2)
    parser.add_argument("--gripper-close-min-motion", type=float, default=0.25)
    parser.add_argument("--harvest-gripper-close-distance-m", type=float, default=0.10)
    parser.add_argument("--harvest-gripper-close-phase", type=float, default=0.30)
    parser.add_argument("--harvest-gripper-phase-close-max-distance-m", type=float, default=0.12)
    parser.add_argument("--harvest-gripper-release-phase", type=float, default=0.82)
    parser.add_argument(
        "--harvest-stop-policy-on-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the learned harvest policy as soon as grasp_close fires and hand off to the fixed basket tail (avoids the no-phase policy replaying into retract).",
    )
    parser.add_argument(
        "--harvest-policy-handoff-distance-m",
        type=float,
        default=None,
        help="Route A (phase-aware): stop the learned policy and hand off to the deterministic staged tail once the live TCP is within this distance (m) of the goal. The tail then runs pre_grasp->grasp(close)->harvest->retract->basket deterministically. Use with --harvest-fixed-basket-tail-start-after approach and --no-harvest-fixed-basket-tail-require-closed.",
    )
    parser.add_argument(
        "--harvest-policy-handoff-plan-margin-m",
        type=float,
        default=None,
        help=(
            "Derive the target-distance handoff threshold from the solved phase plan's "
            "approach TCP distance plus this nonnegative margin. This avoids a "
            "target-specific fixed distance."
        ),
    )
    parser.add_argument(
        "--harvest-grasp-retarget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route A: shift the deterministic pre_grasp/grasp/harvest/retract waypoints by (live tomato - demo tomato) and re-solve IK, so the grasp lands on the actual clicked fruit instead of the demo location.",
    )
    parser.add_argument(
        "--harvest-grasp-extra-insertion-m",
        type=float,
        default=0.0,
        help="Route A: extra insertion depth (m) added to the grasp/harvest waypoints along the demonstrated pre_grasp->grasp axis, re-solved via IK. Increase if grasps are shallow.",
    )
    parser.add_argument(
        "--harvest-grasp-height-offset-m",
        type=float,
        default=0.0,
        help="Route A: vertical (robot +z) offset (m) added to the pre_grasp/grasp/harvest waypoints, re-solved via IK. Positive raises the grasp. Increase if the gripper closes too low on the fruit.",
    )
    parser.add_argument(
        "--harvest-fixed-basket-tail-plan",
        type=Path,
        default=None,
        help="Known-good harvest plan whose waypoints after --harvest-fixed-basket-tail-start-after are executed after learned grasp/retract.",
    )
    parser.add_argument(
        "--harvest-auto-fixed-basket-tail",
        action="store_true",
        help="Choose AT1T1/AT1T2 fixed basket tail from the nearest stored demo label.",
    )
    parser.add_argument("--harvest-fixed-basket-tail-candidate-name", default=None)
    parser.add_argument(
        "--harvest-fixed-basket-tail-start-after",
        choices=("begin", "approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"),
        default="retract",
        help="Which plan waypoint the deterministic tail starts after. Use 'begin' to run the whole staged solve (approach->...->basket) deterministically.",
    )
    parser.add_argument("--harvest-fixed-basket-tail-minimum-time-s", type=float, default=2.0)
    parser.add_argument(
        "--harvest-tail-velocity-limit-rad-s",
        type=float,
        default=None,
        help="Per-joint velocity cap (rad/s) SDK-enforced on every deterministic tail move (gentle approach regardless of move size). ~0.25 matches the proven frozen harvest's effective approach speed.",
    )
    parser.add_argument(
        "--harvest-tail-acceleration-limit-rad-s2",
        type=float,
        default=None,
        help="Per-joint acceleration cap (rad/s^2) SDK-enforced on every deterministic tail move (softer ramp). ~0.2 is gentle.",
    )
    parser.add_argument("--harvest-fixed-basket-tail-require-closed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deleaf-policy-handoff-distance-m",
        type=float,
        default=None,
        help="Deleaf hybrid (mirror of harvest Route A): stop the learned deleaf policy approach once the live TCP is within this distance (m) of the cut point, then hand off to the deterministic deleaf cut tail. Use ~0.10 for hybrid, or a large value (e.g. 1.0) for fully-deterministic.",
    )
    parser.add_argument(
        "--deleaf-fixed-tail-plan",
        type=Path,
        default=None,
        help="deleaf_click_cut plan JSON whose solved cut waypoints (after --deleaf-fixed-tail-start-after) are executed deterministically after the learned approach.",
    )
    parser.add_argument(
        "--deleaf-auto-fixed-tail",
        action="store_true",
        help="Choose the deterministic deleaf cut plan from the nearest stored demo leaf (AD1/BD1-4).",
    )
    parser.add_argument(
        "--deleaf-fixed-tail-start-after",
        choices=(
            "begin",
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
        default="approach",
        help="Which cut phase the deterministic tail starts after. 'approach' = tail does pre_cut->cut->...->return (policy did the approach); 'begin' = whole deterministic cut.",
    )
    parser.add_argument("--deleaf-fixed-tail-minimum-time-s", type=float, default=4.0)
    parser.add_argument(
        "--deleaf-retarget-orientation-tol-rad",
        type=float,
        default=0.40,
        help="Orientation tolerance (rad) for accepting a retargeted deleaf cut waypoint. Deleaf cuts are planned with relaxed (task-axis) orientation, so the position-correct re-solve typically has ~15deg orientation error; the default 0.40 rad accepts it instead of falling back to the demo (off-target) pose.",
    )
    parser.add_argument(
        "--deleaf-cut-retarget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shift the deterministic deleaf cut waypoints by (live cut point - plan cut point) and re-solve IK, so the cut lands on the live leaf. For an exact demo leaf the shift is ~0 (faithful replay).",
    )
    parser.add_argument(
        "--deleaf-tail-velocity-limit-rad-s",
        type=float,
        default=None,
        help="Per-joint velocity cap (rad/s) SDK-enforced on every deterministic deleaf cut move (gentle). ~0.25 is gentle.",
    )
    parser.add_argument(
        "--deleaf-tail-acceleration-limit-rad-s2",
        type=float,
        default=None,
        help="Per-joint acceleration cap (rad/s^2) SDK-enforced on every deterministic deleaf cut move (softer ramps). ~0.1 is gentle.",
    )
    parser.add_argument("--min-clearance-m", type=float, default=0.03)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--path-samples-per-step", type=int, default=4)
    parser.add_argument("--progress-tolerance-m", type=float, default=0.05)
    parser.add_argument(
        "--progress-guard-iterations",
        type=int,
        default=3,
        help="In --full-action mode, enforce target-progress guard for this many initial receding iterations.",
    )
    parser.add_argument("--disable-progress-guard", action="store_true")
    parser.add_argument("--min-x-m", type=float, default=0.25)
    parser.add_argument("--max-x-m", type=float, default=0.85)
    parser.add_argument("--min-y-m", type=float, default=-0.55)
    parser.add_argument("--max-y-m", type=float, default=0.55)
    parser.add_argument("--min-z-m", type=float, default=0.35)
    parser.add_argument("--max-z-m", type=float, default=1.35)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--axis-size-m", type=float, default=0.1)
    parser.add_argument("--surfel-python", type=Path, default=DEFAULT_SURFEL_PYTHON)
    parser.add_argument("--pick-radius-px", type=int, default=3)
    parser.add_argument("--pixel-x", type=int, default=None)
    parser.add_argument("--pixel-y", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("real", "sim"),
        default="real",
        help="Execution backend. 'real' commands the RB-Y1 over gRPC (default); "
        "'sim' runs the rollout against the Isaac Sim digital twin via the :8226 socket "
        "(offline URDF kinematics, no hardware).",
    )
    parser.add_argument(
        "--sim-segment-s",
        type=float,
        default=10.0,
        help="(--backend sim) seconds to animate each tail waypoint segment in the twin. "
        "Larger = slower. Try 15-25 for a very slow playback.",
    )
    parser.add_argument(
        "--sim-interp-steps",
        type=int,
        default=40,
        help="(--backend sim) interpolation sub-frames per waypoint segment (smoothness).",
    )
    parser.add_argument(
        "--sim-joint-vel-deg-s",
        type=float,
        default=18.0,
        help="(--backend sim) per-joint max velocity cap (deg/s) so the arm eases instead "
        "of snapping. Lower = slower/smoother.",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-write-log", action="store_true")
    args = parser.parse_args()
    if args.target_source == "goal" and args.goal_robot_m is None:
        parser.error("--goal-robot-m is required when --target-source goal")
    if args.target_source != "realsense-rgb" and (args.pixel_x is not None or args.pixel_y is not None):
        parser.error("--pixel-x/--pixel-y are only valid with --target-source realsense-rgb")
    if (args.pixel_x is None) != (args.pixel_y is None):
        parser.error("--pixel-x and --pixel-y must be provided together")
    if args.target_source != "goal" and args.skip_fk_guards and (args.q_arm_rad is None or args.tcp_position_robot_m is None):
        parser.error("RealSense target picking with --skip-fk-guards still needs --q-arm-rad and --tcp-position-robot-m")
    if args.clip_joint_limits and args.skip_fk_guards:
        parser.error("--clip-joint-limits requires FK guards; remove --skip-fk-guards")
    if args.steps_to_execute < 1:
        parser.error("--steps-to-execute must be at least 1")
    if args.recede_iterations is not None and args.recede_iterations < 1:
        parser.error("--recede-iterations must be at least 1")
    if args.progress_guard_iterations < 0:
        parser.error("--progress-guard-iterations cannot be negative")
    if args.full_action_duration_scale <= 0.0:
        parser.error("--full-action-duration-scale must be positive")
    if args.extra_recede_iterations < 0:
        parser.error("--extra-recede-iterations cannot be negative")
    if args.inter_step_sleep_s < 0.0:
        parser.error("--inter-step-sleep-s cannot be negative")
    if args.snap_goal_max_distance_m < 0.0:
        parser.error("--snap-goal-max-distance-m cannot be negative")
    if args.gripper_timeout_s <= 0.0:
        parser.error("--gripper-timeout-s must be positive")
    if args.gripper_post_command_wait_s < 0.0:
        parser.error("--gripper-post-command-wait-s cannot be negative")
    if args.gripper_post_command_poll_s <= 0.0:
        parser.error("--gripper-post-command-poll-s must be positive")
    if args.gripper_close_min_motion < 0.0:
        parser.error("--gripper-close-min-motion cannot be negative")
    if args.harvest_gripper_close_distance_m < 0.0:
        parser.error("--harvest-gripper-close-distance-m cannot be negative")
    if args.harvest_gripper_phase_close_max_distance_m < 0.0:
        parser.error("--harvest-gripper-phase-close-max-distance-m cannot be negative")
    if not 0.0 <= args.harvest_gripper_close_phase <= 1.0:
        parser.error("--harvest-gripper-close-phase must be in [0, 1]")
    if not 0.0 <= args.harvest_gripper_release_phase <= 1.0:
        parser.error("--harvest-gripper-release-phase must be in [0, 1]")
    if args.harvest_gripper_release_phase < args.harvest_gripper_close_phase:
        parser.error("--harvest-gripper-release-phase must be >= --harvest-gripper-close-phase")
    if args.harvest_fixed_basket_tail_minimum_time_s <= 0.0:
        parser.error("--harvest-fixed-basket-tail-minimum-time-s must be positive")
    if (
        args.harvest_policy_handoff_distance_m is not None
        and args.harvest_policy_handoff_plan_margin_m is not None
    ):
        parser.error(
            "choose either --harvest-policy-handoff-distance-m or "
            "--harvest-policy-handoff-plan-margin-m"
        )
    if (
        args.harvest_policy_handoff_plan_margin_m is not None
        and args.harvest_policy_handoff_plan_margin_m < 0.0
    ):
        parser.error("--harvest-policy-handoff-plan-margin-m must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.execute:
        print("dry_run: pass --execute to command the robot after reviewing guards.ok=true")
        if payload.get("full_action"):
            print("dry_run_scope: first_prefix_only; --execute will re-predict and guard each prefix live")
    elif payload.get("execution_error") == "guard_failed":
        failed = payload.get("guards", {}).get("failed_guards", [])
        details = payload.get("guards", {}).get("failure_details", [])
        last_iteration = payload.get("last_iteration", {}).get("iteration_index")
        if payload.get("executed"):
            print(
                "guard_failed_after_partial_motion: "
                f"stopped before the next prefix at iteration={last_iteration} failed_guards={failed}"
            )
        else:
            print(
                "guard_failed_before_motion: "
                f"no robot motion executed; failed_guards={failed}; use deterministic fallback for this target"
            )
        print(f"guard_failure_details: {json.dumps(jsonable(details), sort_keys=True)}")
        print(f"runtime_config: {json.dumps(jsonable(payload.get('runtime_config', {})), sort_keys=True)}")
        return 2
    elif payload.get("execution_error") == "gripper_failed":
        last_iteration = payload.get("last_iteration", {}).get("iteration_index")
        message = payload.get("last_iteration", {}).get("execution_error")
        if payload.get("executed"):
            print(
                "gripper_failed_after_partial_motion: "
                f"stopped before the next prefix at iteration={last_iteration}; {message}"
            )
        else:
            print(f"gripper_failed_before_motion: no robot motion executed; {message}")
        return 3
    elif payload.get("execution_error") in {"fixed_tail_failed", "fixed_tail_skipped_gripper_not_closed"}:
        print(
            "fixed_tail_not_completed: "
            f"execution_error={payload.get('execution_error')} "
            f"fixed_basket_tail={payload.get('fixed_basket_tail')}"
        )
        return 4
    elif payload.get("executed"):
        print(
            "executed_prefix: commanded "
            f"{len(payload.get('executed_prefix', []))} step(s) across "
            f"{payload.get('recede_iterations_completed', 1)} receding iteration(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
