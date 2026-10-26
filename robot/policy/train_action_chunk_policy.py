from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from robot.kinematics.geometry import axis_angle_from_rotation


POLICY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POLICY_ROOT.parent.parent
DEFAULT_LIBRARY_INDEX = POLICY_ROOT / "policy_action_library" / "training_index.json"
DEFAULT_OUTPUT_ROOT = POLICY_ROOT / "trained_policies"
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_LEFT_TOOLFIT = Path("D:/tmp/egodelta_isaac_left_jaw_center_toolfit.json")
DEFAULT_RBY1A_URDF = REPO_ROOT / "submodules" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model.urdf"
DELEAF_FACING_BY_LABEL = {
    "AD1": "right",
    "AD2": "left",
    "AD3": "right",
    "BD1": "left",
    "BD2": "right",
    "BD3": "right",
    "BD4": "right",
}


@dataclass(frozen=True)
class DemoTrajectory:
    label: str
    primitive: str
    trajectory_json: Path
    goal_robot_m: np.ndarray
    goal_selection_source: str
    goal_source_artifact: str | None
    goal_pixel_xy: list[float] | None
    goal_point_realsense_m: list[float] | None
    elapsed_s: np.ndarray
    q_arm_rad: np.ndarray
    dq_arm_rad_s: np.ndarray
    tcp_position_robot_m: np.ndarray
    source_sample_count: int
    source_duration_s: float
    trim_start_source_elapsed_s: float = 0.0
    source_kind: str = "real_execution"
    deleaf_facing: str | None = None
    tcp_rotation_aa_robot_m: np.ndarray | None = None


@dataclass(frozen=True)
class ChunkDataset:
    primitive: str
    sample_hz: float
    chunk_size: int
    feature_names: list[str]
    action_names: list[str]
    x: np.ndarray
    y_delta: np.ndarray
    y_abs: np.ndarray
    source_labels: list[str]
    source_times_s: np.ndarray
    demos: list[dict[str, Any]]
    action_space: str = "joint"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(path: str | Path, *, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return base / value


def finite_difference_q(elapsed_s: np.ndarray, q_arm_rad: np.ndarray) -> np.ndarray:
    if len(q_arm_rad) < 2:
        return np.zeros_like(q_arm_rad)
    deltas = np.diff(np.asarray(elapsed_s, dtype=float))
    positive = deltas[deltas > 1e-6]
    dt = float(np.median(positive)) if positive.size else 0.1
    dq = np.gradient(q_arm_rad, axis=0, edge_order=1) / max(dt, 1e-3)
    return np.nan_to_num(np.asarray(dq, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def truncate_harvest_demo_at_waypoint(
    demo: DemoTrajectory,
    action: dict[str, Any],
    *,
    waypoint_name: str,
    max_q_error_rad: float,
) -> DemoTrajectory:
    planner_text = action.get("artifacts", {}).get("planner")
    if not planner_text:
        raise ValueError(f"{demo.label}: cannot truncate harvest demo without artifacts.planner")
    planner = read_json(Path(planner_text))
    candidate = planner.get("best_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{demo.label}: planner has no best_candidate for harvest truncation")
    waypoint = None
    for item in candidate.get("waypoints", []):
        if isinstance(item, dict) and item.get("name") == waypoint_name:
            waypoint = item
            break
    if waypoint is None:
        names = [item.get("name") for item in candidate.get("waypoints", []) if isinstance(item, dict)]
        raise ValueError(f"{demo.label}: planner has no waypoint {waypoint_name!r}; names={names}")
    q = waypoint.get("ik_result", {}).get("q_arm_rad")
    if q is None:
        raise ValueError(f"{demo.label}: waypoint {waypoint_name!r} has no ik_result.q_arm_rad")
    target_q = np.asarray(q, dtype=float)
    if target_q.shape != (7,):
        raise ValueError(f"{demo.label}: waypoint {waypoint_name!r} has invalid q_arm shape")
    distances = np.linalg.norm(demo.q_arm_rad - target_q.reshape(1, 7), axis=1)
    terminal_index = int(np.argmin(distances))
    min_error = float(distances[terminal_index])
    if min_error > float(max_q_error_rad):
        raise ValueError(
            f"{demo.label}: nearest trajectory sample to waypoint {waypoint_name!r} "
            f"is too far: {min_error:.6f} rad > {float(max_q_error_rad):.6f} rad"
        )
    terminal_index = max(2, min(terminal_index, len(demo.elapsed_s) - 1))
    elapsed = demo.elapsed_s[: terminal_index + 1] - float(demo.elapsed_s[0])
    q_arm_rad = demo.q_arm_rad[: terminal_index + 1]
    tcp_position_robot_m = demo.tcp_position_robot_m[: terminal_index + 1]
    return DemoTrajectory(
        label=demo.label,
        primitive=demo.primitive,
        trajectory_json=demo.trajectory_json,
        goal_robot_m=demo.goal_robot_m,
        goal_selection_source=demo.goal_selection_source,
        goal_source_artifact=demo.goal_source_artifact,
        goal_pixel_xy=demo.goal_pixel_xy,
        goal_point_realsense_m=demo.goal_point_realsense_m,
        elapsed_s=elapsed,
        q_arm_rad=q_arm_rad,
        dq_arm_rad_s=finite_difference_q(elapsed, q_arm_rad),
        tcp_position_robot_m=tcp_position_robot_m,
        source_sample_count=demo.source_sample_count,
        source_duration_s=demo.source_duration_s,
        trim_start_source_elapsed_s=demo.trim_start_source_elapsed_s,
        source_kind=f"{demo.source_kind}_truncated_{waypoint_name}",
        deleaf_facing=demo.deleaf_facing,
    )


def load_demo(
    entry: dict[str, Any],
    action: dict[str, Any],
    *,
    harvest_stop_after: str | None = None,
    harvest_stop_match_tolerance_rad: float = 0.05,
) -> DemoTrajectory:
    primitive = str(action["primitive"])
    label = str(entry["label"])
    goal_payload = action.get("goal", {})
    goal = goal_payload.get("point_robot_m")
    if goal is None:
        raise ValueError(f"{label}: missing goal.point_robot_m")
    selection_source = str(goal_payload.get("selection_source") or "")
    if not selection_source:
        raise ValueError(f"{label}: missing goal.selection_source; refusing to train without collected GT target provenance")
    artifacts = action.get("artifacts", {})
    trajectory_path = Path(artifacts.get("trajectory", ""))
    if not trajectory_path.exists():
        raise FileNotFoundError(f"{label}: trajectory artifact does not exist: {trajectory_path}")
    payload = read_json(trajectory_path)
    samples = [sample for sample in payload.get("samples", []) if isinstance(sample, dict)]
    if not samples:
        raise ValueError(f"{label}: trajectory has no samples")

    elapsed: list[float] = []
    q_rows: list[np.ndarray] = []
    tcp_rows: list[np.ndarray] = []
    tcp_rot_rows: list[np.ndarray] = []
    tcp_rotation_ok = True
    for sample in samples:
        if "elapsed_s" not in sample or "q_arm_rad" not in sample:
            continue
        q = np.asarray(sample["q_arm_rad"], dtype=float)
        if q.shape != (7,):
            continue
        tcp = sample.get("tcp", {}).get("position_robot_m")
        if tcp is None:
            continue
        tcp_position = np.asarray(tcp, dtype=float)
        if tcp_position.shape != (3,):
            continue
        rotation = sample.get("tcp", {}).get("rotation_robot_from_tcp")
        rotation_matrix = np.asarray(rotation, dtype=float) if rotation is not None else None
        if rotation_matrix is None or rotation_matrix.shape != (3, 3):
            tcp_rotation_ok = False
            tcp_aa = np.zeros(3, dtype=float)
        else:
            tcp_aa = np.asarray(axis_angle_from_rotation(rotation_matrix), dtype=float).reshape(3)
        elapsed.append(float(sample["elapsed_s"]))
        q_rows.append(q)
        tcp_rows.append(tcp_position)
        tcp_rot_rows.append(tcp_aa)

    if len(q_rows) < 3:
        raise ValueError(f"{label}: not enough usable trajectory samples")
    elapsed_s = np.asarray(elapsed, dtype=float)
    q_arm_rad = np.vstack(q_rows).astype(float)
    tcp_position_robot_m = np.vstack(tcp_rows).astype(float)
    tcp_rotation_aa_robot_m = np.vstack(tcp_rot_rows).astype(float) if tcp_rotation_ok else None
    # Compute velocity from the captured active-arm sequence instead of slicing q_full/dq_full.
    # The full RB-Y1 state ordering has changed across scripts; q_arm_rad is already canonical.
    dq_arm_rad_s = finite_difference_q(elapsed_s, q_arm_rad)
    demo = DemoTrajectory(
        label=label,
        primitive=primitive,
        trajectory_json=trajectory_path,
        goal_robot_m=np.asarray(goal, dtype=float).reshape(3),
        goal_selection_source=selection_source,
        goal_source_artifact=goal_payload.get("source_artifact"),
        goal_pixel_xy=goal_payload.get("pixel_xy"),
        goal_point_realsense_m=goal_payload.get("point_realsense_m"),
        elapsed_s=elapsed_s,
        q_arm_rad=q_arm_rad,
        dq_arm_rad_s=dq_arm_rad_s,
        tcp_position_robot_m=tcp_position_robot_m,
        source_sample_count=int(len(elapsed_s)),
        source_duration_s=float(elapsed_s[-1] - elapsed_s[0]),
        trim_start_source_elapsed_s=0.0,
        source_kind="real_execution",
        deleaf_facing=DELEAF_FACING_BY_LABEL.get(label) if primitive == "deleaf" else None,
        tcp_rotation_aa_robot_m=tcp_rotation_aa_robot_m,
    )
    if primitive == "harvest" and harvest_stop_after is not None:
        demo = truncate_harvest_demo_at_waypoint(
            demo,
            action,
            waypoint_name=str(harvest_stop_after),
            max_q_error_rad=float(harvest_stop_match_tolerance_rad),
        )
    return demo


def trajectory_duration_s(trajectory_path: Path) -> float | None:
    if not trajectory_path.exists():
        return None
    payload = read_json(trajectory_path)
    samples = [sample for sample in payload.get("samples", []) if isinstance(sample, dict)]
    elapsed = [float(sample["elapsed_s"]) for sample in samples if "elapsed_s" in sample]
    if len(elapsed) < 2:
        return None
    return max(0.0, float(elapsed[-1] - elapsed[0]))


def pose_translation_robot_m(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4):
        return None
    return matrix[:3, 3].astype(float)


def harvest_candidate_home_state(payload: dict[str, Any], candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    robot_payload = payload.get("robot", {}) if isinstance(payload.get("robot"), dict) else {}
    home_path = Path(robot_payload.get("home_path") or DEFAULT_HOME)
    toolfit_path = Path(robot_payload.get("toolfit_path") or DEFAULT_LEFT_TOOLFIT)
    model_name = str(robot_payload.get("model") or "a")
    urdf_path = DEFAULT_RBY1A_URDF
    try:
        from robot.kinematics.rby1_model import RBY1Kinematics

        kinematics = RBY1Kinematics.from_urdf(
            urdf_path=urdf_path,
            model_name=model_name,
            arm="left",
            toolfit_path=toolfit_path,
        )
        locked = kinematics.locked_state_from_home(home_path)
        q_home = np.asarray(locked.full_q_rad[list(kinematics.active_arm.joint_indices)], dtype=float)
        tcp_home = np.asarray(kinematics.tcp_pose(locked.full_q_rad)[:3, 3], dtype=float)
        if q_home.shape == (7,) and tcp_home.shape == (3,):
            return q_home, tcp_home
    except Exception:
        pass

    segments = candidate.get("joint_motion", {}).get("segments", [])
    waypoints = candidate.get("waypoints", [])
    if not isinstance(segments, list) or not segments or not isinstance(waypoints, list) or not waypoints:
        return None
    first_segment = next(
        (
            segment
            for segment in segments
            if isinstance(segment, dict)
            and str(segment.get("from", "")).lower() == "home"
            and "delta_rad" in segment
        ),
        None,
    )
    first_waypoint = waypoints[0] if isinstance(waypoints[0], dict) else {}
    q_first = first_waypoint.get("ik_result", {}).get("q_arm_rad")
    if first_segment is None or q_first is None:
        return None
    q_home = np.asarray(q_first, dtype=float).reshape(7) - np.asarray(first_segment["delta_rad"], dtype=float).reshape(7)
    tcp_first = pose_translation_robot_m(first_waypoint.get("final_tcp_pose"))
    if tcp_first is None:
        tcp_first = pose_translation_robot_m(first_waypoint.get("target_tcp_pose"))
    if tcp_first is None:
        return None
    return q_home.astype(float), tcp_first.astype(float)


def load_harvest_candidate_demos(
    entry: dict[str, Any],
    action: dict[str, Any],
    *,
    top_k: int,
    stop_after: str | None = None,
    waypoint_dt_s: float | None = None,
    prepend_home: bool = False,
) -> list[DemoTrajectory]:
    if action.get("primitive") != "harvest" or top_k <= 0:
        return []
    candidate_dir_text = action.get("candidate_solve_dir")
    if not candidate_dir_text:
        return []
    candidate_dir = Path(candidate_dir_text)
    if not candidate_dir.exists():
        return []
    goal_payload = action.get("goal", {})
    goal = goal_payload.get("point_robot_m")
    if goal is None:
        return []

    artifacts = action.get("artifacts", {})
    trajectory_text = artifacts.get("trajectory")
    actual_duration = trajectory_duration_s(Path(trajectory_text)) if trajectory_text else None
    candidates = sorted(candidate_dir.glob("candidate_*.json"))[: max(0, int(top_k))]
    demos: list[DemoTrajectory] = []
    for rank, candidate_path in enumerate(candidates, start=1):
        payload = read_json(candidate_path)
        candidate = payload.get("best_candidate", {})
        waypoints = candidate.get("waypoints", []) if isinstance(candidate, dict) else []
        q_rows: list[np.ndarray] = []
        tcp_rows: list[np.ndarray] = []
        waypoint_names: list[str] = []
        for waypoint in waypoints:
            if not isinstance(waypoint, dict):
                continue
            q = waypoint.get("ik_result", {}).get("q_arm_rad")
            if q is None:
                continue
            q_arm = np.asarray(q, dtype=float)
            if q_arm.shape != (7,):
                continue
            tcp = pose_translation_robot_m(waypoint.get("final_tcp_pose"))
            if tcp is None:
                tcp = pose_translation_robot_m(waypoint.get("target_tcp_pose"))
            if tcp is None or tcp.shape != (3,):
                continue
            q_rows.append(q_arm)
            tcp_rows.append(tcp)
            waypoint_name = str(waypoint.get("name", f"waypoint_{len(q_rows)}"))
            waypoint_names.append(waypoint_name)
            if stop_after is not None and waypoint_name == str(stop_after):
                break
        if len(q_rows) < 3:
            continue
        if prepend_home:
            home_state = harvest_candidate_home_state(payload, candidate)
            if home_state is not None:
                q_home, tcp_home = home_state
                q_rows.insert(0, q_home)
                tcp_rows.insert(0, tcp_home)
                waypoint_names.insert(0, "home")
        if waypoint_dt_s is not None and float(waypoint_dt_s) > 0.0:
            duration_s = float(waypoint_dt_s) * float(len(q_rows) - 1)
        else:
            duration_s = actual_duration if actual_duration is not None and actual_duration > 0.0 else float(8.0 * (len(q_rows) - 1))
        elapsed_s = np.linspace(0.0, duration_s, num=len(q_rows), dtype=float)
        q_arm_rad = np.vstack(q_rows).astype(float)
        tcp_position_robot_m = np.vstack(tcp_rows).astype(float)
        demos.append(
            DemoTrajectory(
                label=f"{entry['label']}__solver_candidate_{rank:02d}",
                primitive="harvest",
                trajectory_json=candidate_path,
                goal_robot_m=np.asarray(goal, dtype=float).reshape(3),
                goal_selection_source="harvest_solver_candidate",
                goal_source_artifact=str(candidate_path),
                goal_pixel_xy=goal_payload.get("pixel_xy"),
                goal_point_realsense_m=goal_payload.get("point_realsense_m"),
                elapsed_s=elapsed_s,
                q_arm_rad=q_arm_rad,
                dq_arm_rad_s=finite_difference_q(elapsed_s, q_arm_rad),
                tcp_position_robot_m=tcp_position_robot_m,
                source_sample_count=int(len(elapsed_s)),
                source_duration_s=float(duration_s),
                trim_start_source_elapsed_s=0.0,
                source_kind="solver_candidate_waypoints",
                deleaf_facing=None,
            )
        )
    return demos


def load_demos(
    index_path: Path,
    primitive: str,
    *,
    include_labels: set[str] | None = None,
    include_harvest_candidate_solves: bool = False,
    harvest_candidate_top_k: int = 0,
    harvest_candidate_only: bool = False,
    harvest_stop_after: str | None = None,
    harvest_stop_match_tolerance_rad: float = 0.05,
    harvest_candidate_stop_after: str | None = None,
    harvest_candidate_waypoint_dt_s: float | None = None,
    harvest_candidate_prepend_home: bool = False,
) -> list[DemoTrajectory]:
    root = index_path.parent
    index = read_json(index_path)
    demos: list[DemoTrajectory] = []
    for entry in index.get("actions", []):
        label = str(entry.get("label", ""))
        if include_labels is not None and label not in include_labels:
            continue
        for action in entry.get("actions", []):
            if action.get("primitive") != primitive:
                continue
            if not (primitive == "harvest" and harvest_candidate_only):
                demos.append(
                    load_demo(
                        entry,
                        action,
                        harvest_stop_after=harvest_stop_after if primitive == "harvest" else None,
                        harvest_stop_match_tolerance_rad=float(harvest_stop_match_tolerance_rad),
                    )
                )
            if primitive == "harvest" and include_harvest_candidate_solves:
                demos.extend(
                    load_harvest_candidate_demos(
                        entry,
                        action,
                        top_k=int(harvest_candidate_top_k),
                        stop_after=harvest_candidate_stop_after,
                        waypoint_dt_s=harvest_candidate_waypoint_dt_s,
                        prepend_home=bool(harvest_candidate_prepend_home),
                    )
                )
    if not demos:
        label_note = "" if include_labels is None else f" for labels {sorted(include_labels)}"
        raise ValueError(f"no {primitive} actions found in {index_path}{label_note}")
    return demos


def trim_idle_prefix(demo: DemoTrajectory, *, start_delta_rad: float, preroll_s: float) -> DemoTrajectory:
    threshold = max(0.0, float(start_delta_rad))
    if threshold <= 0.0:
        return demo
    q0 = demo.q_arm_rad[0]
    displacement = np.linalg.norm(demo.q_arm_rad - q0.reshape(1, -1), axis=1)
    moving = np.flatnonzero(displacement > threshold)
    if moving.size == 0:
        return demo
    onset_index = int(moving[0])
    start_time = max(float(demo.elapsed_s[0]), float(demo.elapsed_s[onset_index]) - max(0.0, float(preroll_s)))
    start_index = int(np.searchsorted(demo.elapsed_s, start_time, side="left"))
    start_index = max(0, min(start_index, len(demo.elapsed_s) - 2))
    if start_index <= 0:
        return demo
    elapsed = demo.elapsed_s[start_index:] - float(demo.elapsed_s[start_index])
    return DemoTrajectory(
        label=demo.label,
        primitive=demo.primitive,
        trajectory_json=demo.trajectory_json,
        goal_robot_m=demo.goal_robot_m,
        goal_selection_source=demo.goal_selection_source,
        goal_source_artifact=demo.goal_source_artifact,
        goal_pixel_xy=demo.goal_pixel_xy,
        goal_point_realsense_m=demo.goal_point_realsense_m,
        elapsed_s=elapsed,
        q_arm_rad=demo.q_arm_rad[start_index:],
        dq_arm_rad_s=demo.dq_arm_rad_s[start_index:],
        tcp_position_robot_m=demo.tcp_position_robot_m[start_index:],
        source_sample_count=demo.source_sample_count,
        source_duration_s=demo.source_duration_s,
        trim_start_source_elapsed_s=float(demo.elapsed_s[start_index]),
        source_kind=demo.source_kind,
        deleaf_facing=demo.deleaf_facing,
        tcp_rotation_aa_robot_m=(
            None if demo.tcp_rotation_aa_robot_m is None else demo.tcp_rotation_aa_robot_m[start_index:]
        ),
    )


def resample_demo(demo: DemoTrajectory, sample_hz: float, normalize_duration_s: float | None = None) -> DemoTrajectory:
    if sample_hz <= 0.0:
        raise ValueError("--sample-hz must be positive")
    elapsed = demo.elapsed_s - float(demo.elapsed_s[0])
    duration_s = float(elapsed[-1])
    if duration_s <= 0.0:
        raise ValueError(f"{demo.label}: invalid non-positive duration")
    # Optional speed normalization: time-compress (or expand) each demo to a common target duration so
    # the per-step motion is consistent across demos and matches a known-good rollout regime. A demo
    # collected slowly (e.g. for hardware safety) yields tiny per-step deltas at a fixed sample_hz, which
    # a receding-horizon BC policy cannot make progress on (it crawls/stalls). Rescaling elapsed time
    # preserves the spatial path while scaling the per-step deltas; arm velocities scale by the same factor.
    speedup = 1.0
    if normalize_duration_s is not None and float(normalize_duration_s) > 0.0:
        speedup = duration_s / float(normalize_duration_s)
        elapsed = elapsed / speedup
        duration_s = float(normalize_duration_s)
    step_s = 1.0 / float(sample_hz)
    target_elapsed = np.arange(0.0, duration_s + 1e-9, step_s, dtype=float)
    if target_elapsed[-1] < duration_s:
        target_elapsed = np.append(target_elapsed, duration_s)

    def interp_matrix(values: np.ndarray) -> np.ndarray:
        cols = [np.interp(target_elapsed, elapsed, values[:, col]) for col in range(values.shape[1])]
        return np.vstack(cols).T.astype(float)

    return DemoTrajectory(
        label=demo.label,
        primitive=demo.primitive,
        trajectory_json=demo.trajectory_json,
        goal_robot_m=demo.goal_robot_m,
        goal_selection_source=demo.goal_selection_source,
        goal_source_artifact=demo.goal_source_artifact,
        goal_pixel_xy=demo.goal_pixel_xy,
        goal_point_realsense_m=demo.goal_point_realsense_m,
        elapsed_s=target_elapsed,
        q_arm_rad=interp_matrix(demo.q_arm_rad),
        dq_arm_rad_s=interp_matrix(demo.dq_arm_rad_s) * float(speedup),
        tcp_position_robot_m=interp_matrix(demo.tcp_position_robot_m),
        source_sample_count=demo.source_sample_count,
        source_duration_s=demo.source_duration_s,
        trim_start_source_elapsed_s=demo.trim_start_source_elapsed_s,
        source_kind=demo.source_kind,
        deleaf_facing=demo.deleaf_facing,
        tcp_rotation_aa_robot_m=(
            None if demo.tcp_rotation_aa_robot_m is None else interp_matrix(demo.tcp_rotation_aa_robot_m)
        ),
    )


def feature_vector(
    demo: DemoTrajectory,
    index: int,
    *,
    include_phase_features: bool,
    include_gcsl_features: bool,
    include_deleaf_side_features: bool,
    goal_robot_m: np.ndarray | None = None,
) -> np.ndarray:
    tcp_position = demo.tcp_position_robot_m[index]
    goal = demo.goal_robot_m if goal_robot_m is None else np.asarray(goal_robot_m, dtype=float).reshape(3)
    values = [
        goal,
        goal - tcp_position,
        demo.q_arm_rad[index],
        demo.dq_arm_rad_s[index],
        tcp_position,
    ]
    if include_phase_features:
        duration_s = max(float(demo.elapsed_s[-1]), 1e-6)
        phase_s = float(demo.elapsed_s[index])
        phase_fraction = min(max(phase_s / duration_s, 0.0), 1.0)
        values.append(np.asarray([phase_s, phase_fraction, 1.0 - phase_fraction], dtype=float))
    if include_gcsl_features:
        duration_s = max(float(demo.elapsed_s[-1]), 1e-6)
        elapsed_s = float(demo.elapsed_s[index])
        horizon_s = max(duration_s - elapsed_s, 0.0)
        horizon_fraction = min(max(horizon_s / duration_s, 0.0), 1.0)
        values.append(np.asarray([horizon_s, horizon_fraction, 1.0 - horizon_fraction], dtype=float))
    if include_deleaf_side_features:
        facing = str(demo.deleaf_facing or "").strip().lower()
        if facing not in ("left", "right"):
            raise ValueError(f"{demo.label}: missing deleaf facing side; cannot use --include-deleaf-side-features")
        values.append(
            np.asarray(
                [
                    1.0 if facing == "left" else 0.0,
                    1.0 if facing == "right" else 0.0,
                    -1.0 if facing == "left" else 1.0,
                ],
                dtype=float,
            )
        )
    return np.concatenate(values).astype(float)


def feature_names(
    *,
    include_phase_features: bool = False,
    include_gcsl_features: bool = False,
    include_deleaf_side_features: bool = False,
) -> list[str]:
    names = (
        [f"goal_robot_m.{axis}" for axis in ("x", "y", "z")]
        + [f"goal_minus_tcp_robot_m.{axis}" for axis in ("x", "y", "z")]
        + [f"q_arm_rad.{i}" for i in range(7)]
        + [f"dq_arm_rad_s.{i}" for i in range(7)]
        + [f"tcp_position_robot_m.{axis}" for axis in ("x", "y", "z")]
    )
    if include_phase_features:
        names += ["phase_s", "phase_fraction", "remaining_fraction"]
    if include_gcsl_features:
        names += ["gcsl_horizon_s", "gcsl_horizon_fraction", "gcsl_elapsed_fraction"]
    if include_deleaf_side_features:
        names += ["deleaf_left_facing", "deleaf_right_facing", "deleaf_facing_sign"]
    return names


def action_names(chunk_size: int, action_space: str = "joint") -> list[str]:
    if action_space == "eef":
        axes = ("pos_x", "pos_y", "pos_z", "rot_rx", "rot_ry", "rot_rz")
        return [
            f"tcp_pose_robot.t+{step}.{axis}"
            for step in range(1, chunk_size + 1)
            for axis in axes
        ]
    return [f"delta_q_arm_rad.t+{step}.{joint}" for step in range(1, chunk_size + 1) for joint in range(7)]


def build_chunk_dataset(
    demos: list[DemoTrajectory],
    *,
    sample_hz: float,
    chunk_size: int,
    trim_idle_prefix_enabled: bool,
    trim_start_delta_rad: float,
    trim_preroll_s: float,
    include_phase_features: bool,
    include_gcsl_features: bool,
    include_deleaf_side_features: bool,
    pad_terminal_chunks: bool,
    goal_jitter_count: int,
    goal_jitter_std_m: float,
    goal_jitter_max_m: float,
    seed: int,
    action_space: str = "joint",
    label_oversample: dict[str, int] | None = None,
    normalize_duration_s: float | None = None,
) -> ChunkDataset:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    trimmed = [
        trim_idle_prefix(demo, start_delta_rad=trim_start_delta_rad, preroll_s=trim_preroll_s)
        if trim_idle_prefix_enabled
        else demo
        for demo in demos
    ]
    resampled = [resample_demo(demo, sample_hz, normalize_duration_s=normalize_duration_s) for demo in trimmed]
    x_rows: list[np.ndarray] = []
    y_delta_rows: list[np.ndarray] = []
    y_abs_rows: list[np.ndarray] = []
    source_labels: list[str] = []
    source_times: list[float] = []
    demo_meta: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    jitter_count = max(0, int(goal_jitter_count))
    jitter_std_m = max(0.0, float(goal_jitter_std_m))
    jitter_max_m = max(0.0, float(goal_jitter_max_m))

    def append_row(
        demo: DemoTrajectory,
        index: int,
        goal_override: np.ndarray | None,
        y_abs_row: np.ndarray,
        y_delta_row: np.ndarray,
    ) -> None:
        x_rows.append(
            feature_vector(
                demo,
                index,
                include_phase_features=include_phase_features,
                include_gcsl_features=include_gcsl_features,
                include_deleaf_side_features=include_deleaf_side_features,
                goal_robot_m=goal_override,
            )
        )
        y_abs_rows.append(np.asarray(y_abs_row, dtype=float).reshape(-1))
        y_delta_rows.append(np.asarray(y_delta_row, dtype=float).reshape(-1))
        source_labels.append(demo.label)
        source_times.append(float(demo.elapsed_s[index]))

    def chunk_future_indices(demo: DemoTrajectory, index: int) -> np.ndarray:
        if pad_terminal_chunks:
            return np.minimum(np.arange(index + 1, index + chunk_size + 1), len(demo.q_arm_rad) - 1)
        return np.arange(index + 1, index + chunk_size + 1)

    def chunk_targets(demo: DemoTrajectory, index: int, future_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if action_space == "eef":
            if demo.tcp_rotation_aa_robot_m is None:
                raise ValueError(
                    f"{demo.label}: --action-space eef requires recorded tcp.rotation_robot_from_tcp in every sample"
                )
            pose = np.concatenate(
                [demo.tcp_position_robot_m[future_indices], demo.tcp_rotation_aa_robot_m[future_indices]],
                axis=1,
            )
            # Absolute EEF pose target (position m + axis-angle rad). y_abs == y_delta by design;
            # see metadata output.contract / action_space tag. The existing fit path trains on y_delta.
            return pose, pose
        future = demo.q_arm_rad[future_indices]
        q_now = demo.q_arm_rad[index]
        return future, future - q_now.reshape(1, 7)

    for demo in resampled:
        usable = len(demo.q_arm_rad) - 1 if pad_terminal_chunks else len(demo.q_arm_rad) - chunk_size
        if usable <= 0:
            raise ValueError(f"{demo.label}: trajectory shorter than chunk size {chunk_size}")
        demo_meta.append(
            {
                "label": demo.label,
                "primitive": demo.primitive,
                "trajectory_json": str(demo.trajectory_json),
                "goal_robot_m": demo.goal_robot_m.astype(float).tolist(),
                "goal_selection_source": demo.goal_selection_source,
                "goal_source_artifact": demo.goal_source_artifact,
                "goal_pixel_xy": demo.goal_pixel_xy,
                "goal_point_realsense_m": demo.goal_point_realsense_m,
                "source_duration_s": float(demo.source_duration_s),
                "source_sample_count": int(demo.source_sample_count),
                "trim_start_source_elapsed_s": float(demo.trim_start_source_elapsed_s),
                "source_kind": demo.source_kind,
                "deleaf_facing": demo.deleaf_facing,
                "duration_s": float(demo.elapsed_s[-1]),
                "resampled_sample_count": int(len(demo.elapsed_s)),
            }
        )
        for index in range(usable):
            future_indices = chunk_future_indices(demo, index)
            y_abs_row, y_delta_row = chunk_targets(demo, index, future_indices)
            append_row(demo, index, None, y_abs_row, y_delta_row)
            if jitter_count > 0 and jitter_std_m > 0.0:
                for _ in range(jitter_count):
                    jitter = rng.normal(0.0, jitter_std_m, size=3).astype(float)
                    if jitter_max_m > 0.0:
                        norm = float(np.linalg.norm(jitter))
                        if norm > jitter_max_m:
                            jitter *= jitter_max_m / max(norm, 1e-12)
                    append_row(demo, index, demo.goal_robot_m + jitter, y_abs_row, y_delta_row)
    if label_oversample:
        base_n = len(source_labels)
        for label, factor in label_oversample.items():
            extra = int(factor) - 1
            if extra <= 0:
                continue
            idxs = [i for i in range(base_n) if source_labels[i] == label]
            if not idxs:
                raise ValueError(f"--label-oversample label {label!r} not found among demos")
            for _ in range(extra):
                for i in idxs:
                    x_rows.append(x_rows[i])
                    y_abs_rows.append(y_abs_rows[i])
                    y_delta_rows.append(y_delta_rows[i])
                    source_labels.append(source_labels[i])
                    source_times.append(source_times[i])
    primitive = resampled[0].primitive
    return ChunkDataset(
        primitive=primitive,
        sample_hz=float(sample_hz),
        chunk_size=int(chunk_size),
        feature_names=feature_names(
            include_phase_features=include_phase_features,
            include_gcsl_features=include_gcsl_features,
            include_deleaf_side_features=include_deleaf_side_features,
        ),
        action_names=action_names(chunk_size, action_space),
        x=np.vstack(x_rows).astype(float),
        y_delta=np.vstack(y_delta_rows).astype(float),
        y_abs=np.vstack(y_abs_rows).astype(float),
        source_labels=source_labels,
        source_times_s=np.asarray(source_times, dtype=float),
        demos=demo_meta,
        action_space=action_space,
    )


def normalizer(values: np.ndarray, eps: float = 1e-8, min_std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(std < eps, 1.0, std)
    if min_std is not None:
        floors = np.asarray(min_std, dtype=float).reshape(std.shape)
        std = np.maximum(std, floors)
    return mean.astype(float), std.astype(float)


def feature_min_std(feature_names: list[str], *, goal_normalizer_min_std_m: float) -> np.ndarray:
    floors = np.zeros(len(feature_names), dtype=float)
    floor_m = max(0.0, float(goal_normalizer_min_std_m))
    if floor_m <= 0.0:
        return floors
    for index, name in enumerate(feature_names):
        if name.startswith("goal_robot_m.") or name.startswith("goal_minus_tcp_robot_m."):
            floors[index] = floor_m
    return floors


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("--bc-hidden-dims must contain positive integers")
    return dims


def parse_label_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    labels = {part.strip() for part in str(value).split(",") if part.strip()}
    return labels or None


def init_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, rng: np.random.Generator) -> dict[str, Any]:
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        scale = math.sqrt(2.0 / float(in_dim + out_dim))
        weights.append(rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(float))
        biases.append(np.zeros(out_dim, dtype=float))
    return {"weights": weights, "biases": biases}


def mlp_forward(params: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    activations = x
    cache: list[tuple[np.ndarray, np.ndarray]] = []
    last_index = len(params["weights"]) - 1
    for index, (weight, bias) in enumerate(zip(params["weights"], params["biases"])):
        z = activations @ weight + bias
        cache.append((activations, z))
        activations = z if index == last_index else np.tanh(z)
    return activations, cache


def mlp_predict(params: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return mlp_forward(params, x)[0]


def copy_mlp(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "weights": [np.array(weight, copy=True) for weight in params["weights"]],
        "biases": [np.array(bias, copy=True) for bias in params["biases"]],
    }


def import_torch() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch


def resolve_bc_backend(requested: str, torch_device: str) -> tuple[str, str | None]:
    requested = str(requested).lower()
    if requested not in ("auto", "numpy", "torch"):
        raise ValueError(f"unsupported BC backend: {requested}")
    if requested == "numpy":
        return "numpy", None
    torch = import_torch()
    if torch is None:
        if requested == "torch":
            raise RuntimeError(
                "PyTorch is not installed in this environment. Install a CUDA PyTorch build, "
                "or rerun with --bc-backend numpy."
            )
        return "numpy", None
    device = str(torch_device).lower()
    if device == "auto":
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if device.startswith("cuda") and not bool(torch.cuda.is_available()):
        if requested == "torch":
            raise RuntimeError("PyTorch is installed, but CUDA is not available. Use --torch-device cpu or install CUDA PyTorch.")
        return "numpy", None
    if requested == "auto" and device == "cpu":
        return "numpy", None
    return "torch", device


def fit_mlp_bc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    x_min_std: np.ndarray | None = None,
) -> dict[str, Any]:
    x_mean, x_std = normalizer(x, min_std=x_min_std)
    y_mean, y_std = normalizer(y)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    rng = np.random.default_rng(seed)
    params = init_mlp(x_norm.shape[1], hidden_dims, y_norm.shape[1], rng)
    moment_w = [np.zeros_like(weight) for weight in params["weights"]]
    velocity_w = [np.zeros_like(weight) for weight in params["weights"]]
    moment_b = [np.zeros_like(bias) for bias in params["biases"]]
    velocity_b = [np.zeros_like(bias) for bias in params["biases"]]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    n = x_norm.shape[0]
    batch_size = max(1, min(int(batch_size), n))
    loss_history: list[dict[str, float]] = []
    best_params = copy_mlp(params)
    best_loss = float("inf")
    for epoch in range(1, int(epochs) + 1):
        permutation = rng.permutation(n)
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            xb = x_norm[indices]
            yb = y_norm[indices]
            pred, cache = mlp_forward(params, xb)
            grad = 2.0 * (pred - yb) / float(np.prod(yb.shape))
            grad_w: list[np.ndarray] = []
            grad_b: list[np.ndarray] = []
            for layer in reversed(range(len(params["weights"]))):
                activation_in, z = cache[layer]
                dw = activation_in.T @ grad + float(weight_decay) * params["weights"][layer]
                db = np.sum(grad, axis=0)
                grad_w.append(dw)
                grad_b.append(db)
                if layer > 0:
                    grad = (grad @ params["weights"][layer].T) * (1.0 - np.tanh(cache[layer - 1][1]) ** 2)
            grad_w.reverse()
            grad_b.reverse()
            step += 1
            for layer in range(len(params["weights"])):
                moment_w[layer] = beta1 * moment_w[layer] + (1.0 - beta1) * grad_w[layer]
                velocity_w[layer] = beta2 * velocity_w[layer] + (1.0 - beta2) * np.square(grad_w[layer])
                moment_b[layer] = beta1 * moment_b[layer] + (1.0 - beta1) * grad_b[layer]
                velocity_b[layer] = beta2 * velocity_b[layer] + (1.0 - beta2) * np.square(grad_b[layer])
                mw_hat = moment_w[layer] / (1.0 - beta1**step)
                vw_hat = velocity_w[layer] / (1.0 - beta2**step)
                mb_hat = moment_b[layer] / (1.0 - beta1**step)
                vb_hat = velocity_b[layer] / (1.0 - beta2**step)
                params["weights"][layer] -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
                params["biases"][layer] -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0:
            full_pred = mlp_predict(params, x_norm)
            loss = float(np.mean(np.square(full_pred - y_norm)))
            loss_history.append({"epoch": float(epoch), "normalized_mse": loss})
            if loss < best_loss:
                best_loss = loss
                best_params = copy_mlp(params)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "weights": best_params["weights"],
        "biases": best_params["biases"],
        "hidden_dims": list(hidden_dims),
        "loss_history": loss_history,
        "best_normalized_mse": float(best_loss),
    }


def fit_mlp_bc_torch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    print_progress: bool = False,
    x_min_std: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    x_mean, x_std = normalizer(x, min_std=x_min_std)
    y_mean, y_std = normalizer(y)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    dev = torch.device(device)
    dtype = torch.float32
    x_tensor = torch.as_tensor(x_norm, dtype=dtype, device=dev)
    y_tensor = torch.as_tensor(y_norm, dtype=dtype, device=dev)
    w_tensor = (
        torch.as_tensor(np.asarray(sample_weights, dtype=float).reshape(-1), dtype=dtype, device=dev)
        if sample_weights is not None
        else None
    )
    dims = [int(x_tensor.shape[1])] + list(hidden_dims) + [int(y_tensor.shape[1])]
    weights = torch.nn.ParameterList()
    biases = torch.nn.ParameterList()
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed))
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        scale = math.sqrt(2.0 / float(in_dim + out_dim))
        init = torch.randn((in_dim, out_dim), generator=generator, device=dev, dtype=dtype) * scale
        weights.append(torch.nn.Parameter(init))
        biases.append(torch.nn.Parameter(torch.zeros(out_dim, device=dev, dtype=dtype)))

    def forward(inputs: Any) -> Any:
        activations = inputs
        last_index = len(weights) - 1
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            activations = activations.matmul(weight) + bias
            if index != last_index:
                activations = torch.tanh(activations)
        return activations

    optimizer = torch.optim.Adam(list(weights) + list(biases), lr=float(learning_rate), weight_decay=float(weight_decay))
    n = int(x_tensor.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    loss_history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_weights: list[np.ndarray] = []
    best_biases: list[np.ndarray] = []
    check_every = max(1, int(epochs) // 10)
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(n, device=dev)
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            prediction = forward(x_tensor.index_select(0, indices))
            target = y_tensor.index_select(0, indices)
            if w_tensor is not None:
                per_sample = torch.mean(torch.square(prediction - target), dim=1)
                w_batch = w_tensor.index_select(0, indices)
                loss = (w_batch * per_sample).sum() / torch.clamp(w_batch.sum(), min=1e-8)
            else:
                loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch == int(epochs) or epoch % check_every == 0:
            with torch.no_grad():
                if w_tensor is not None:
                    per_sample_full = torch.mean(torch.square(forward(x_tensor) - y_tensor), dim=1)
                    full_loss = float(
                        (w_tensor * per_sample_full).sum().div(torch.clamp(w_tensor.sum(), min=1e-8)).detach().cpu().item()
                    )
                else:
                    full_loss = float(torch.mean(torch.square(forward(x_tensor) - y_tensor)).detach().cpu().item())
            loss_history.append({"epoch": float(epoch), "normalized_mse": full_loss})
            if print_progress:
                print(f"[bc-train:{device}] epoch={epoch}/{int(epochs)} normalized_mse={full_loss:.8g}", flush=True)
            if full_loss < best_loss:
                best_loss = full_loss
                best_weights = [weight.detach().cpu().numpy().astype(float, copy=True) for weight in weights]
                best_biases = [bias.detach().cpu().numpy().astype(float, copy=True) for bias in biases]
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "weights": best_weights,
        "biases": best_biases,
        "hidden_dims": list(hidden_dims),
        "loss_history": loss_history,
        "best_normalized_mse": float(best_loss),
        "backend": "torch",
        "device": str(device),
    }


def fit_mlp_bc_backend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    backend: str,
    torch_device: str | None,
    print_progress: bool = False,
    x_min_std: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    if backend == "torch":
        if torch_device is None:
            raise RuntimeError("torch backend selected without a torch device")
        return fit_mlp_bc_torch(
            x,
            y,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            device=torch_device,
            print_progress=print_progress,
            x_min_std=x_min_std,
            sample_weights=sample_weights,
        )
    model = fit_mlp_bc(
        x,
        y,
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        x_min_std=x_min_std,
    )
    model["backend"] = "numpy"
    model["device"] = "cpu"
    return model


def predict_mlp_bc(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_norm = (x - model["x_mean"]) / model["x_std"]
    y_norm = mlp_predict(model, x_norm)
    return y_norm * model["y_std"] + model["y_mean"]


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
    *,
    x_min_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    x_mean, x_std = normalizer(x, min_std=x_min_std)
    y_mean, y_std = normalizer(y)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=float)], axis=1)
    reg = float(ridge_lambda) * np.eye(x_aug.shape[1], dtype=float)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_norm)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "weights": weights.astype(float),
    }


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x_norm = (x - model["x_mean"]) / model["x_std"]
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=float)], axis=1)
    return (x_aug @ model["weights"]) * model["y_std"] + model["y_mean"]


def predict_knn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    k: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    train_norm = (train_x - x_mean) / x_std
    query_norm = (query_x - x_mean) / x_std
    k = max(1, min(int(k), len(train_x)))
    predictions: list[np.ndarray] = []
    for row in query_norm:
        distances = np.linalg.norm(train_norm - row.reshape(1, -1), axis=1)
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest_distances = distances[nearest]
        weights = 1.0 / np.maximum(nearest_distances, 1e-6)
        weights /= np.sum(weights)
        predictions.append(np.sum(train_y[nearest] * weights.reshape(-1, 1), axis=0))
    return np.vstack(predictions).astype(float)


def rmse_rad(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(pred - target))))


def first_step_rmse_rad(pred: np.ndarray, target: np.ndarray) -> float:
    return rmse_rad(pred[:, :7], target[:, :7])


def evaluate_policy(
    dataset: ChunkDataset,
    *,
    ridge_lambda: float,
    knn_k: int,
    bc_model: dict[str, Any],
    bc_hidden_dims: list[int],
    bc_epochs: int,
    bc_batch_size: int,
    bc_learning_rate: float,
    bc_weight_decay: float,
    seed: int,
    bc_backend: str,
    torch_device: str | None,
    x_min_std: np.ndarray | None,
    skip_loo_eval: bool = False,
) -> dict[str, Any]:
    labels = sorted(set(dataset.source_labels))
    bc_train_pred = predict_mlp_bc(bc_model, dataset.x)
    metrics: dict[str, Any] = {
        "train": {
            "bc_chunk_rmse_rad": rmse_rad(bc_train_pred, dataset.y_delta),
            "bc_first_step_rmse_rad": first_step_rmse_rad(bc_train_pred, dataset.y_delta),
        },
        "leave_one_label_out": [],
    }
    if not skip_loo_eval:
        all_model = fit_ridge(dataset.x, dataset.y_delta, ridge_lambda, x_min_std=x_min_std)
        ridge_train_pred = predict_ridge(all_model, dataset.x)
        x_mean, x_std = normalizer(dataset.x, min_std=x_min_std)
        knn_train_pred = predict_knn(
            dataset.x,
            dataset.y_delta,
            dataset.x,
            k=max(2, knn_k + 1),
            x_mean=x_mean,
            x_std=x_std,
        )
        metrics["train"].update(
            {
                "ridge_chunk_rmse_rad": rmse_rad(ridge_train_pred, dataset.y_delta),
                "ridge_first_step_rmse_rad": first_step_rmse_rad(ridge_train_pred, dataset.y_delta),
                "knn_chunk_rmse_rad": rmse_rad(knn_train_pred, dataset.y_delta),
                "knn_first_step_rmse_rad": first_step_rmse_rad(knn_train_pred, dataset.y_delta),
            }
        )
        for label in labels:
            test_mask = np.asarray([item == label for item in dataset.source_labels], dtype=bool)
            train_mask = ~test_mask
            if int(np.sum(train_mask)) < 2 or int(np.sum(test_mask)) < 1:
                continue
            bc_loo = fit_mlp_bc_backend(
                dataset.x[train_mask],
                dataset.y_delta[train_mask],
                hidden_dims=bc_hidden_dims,
                epochs=max(150, int(bc_epochs) // 4),
                batch_size=bc_batch_size,
                learning_rate=bc_learning_rate,
                weight_decay=bc_weight_decay,
                seed=seed + labels.index(label) + 1,
                backend=bc_backend,
                torch_device=torch_device,
                x_min_std=x_min_std,
            )
            bc_pred = predict_mlp_bc(bc_loo, dataset.x[test_mask])
            ridge = fit_ridge(dataset.x[train_mask], dataset.y_delta[train_mask], ridge_lambda, x_min_std=x_min_std)
            ridge_pred = predict_ridge(ridge, dataset.x[test_mask])
            loo_x_mean, loo_x_std = normalizer(dataset.x[train_mask], min_std=x_min_std)
            knn_pred = predict_knn(
                dataset.x[train_mask],
                dataset.y_delta[train_mask],
                dataset.x[test_mask],
                k=knn_k,
                x_mean=loo_x_mean,
                x_std=loo_x_std,
            )
            metrics["leave_one_label_out"].append(
                {
                    "label": label,
                    "test_chunks": int(np.sum(test_mask)),
                    "train_chunks": int(np.sum(train_mask)),
                    "bc_chunk_rmse_rad": rmse_rad(bc_pred, dataset.y_delta[test_mask]),
                    "bc_first_step_rmse_rad": first_step_rmse_rad(bc_pred, dataset.y_delta[test_mask]),
                    "ridge_chunk_rmse_rad": rmse_rad(ridge_pred, dataset.y_delta[test_mask]),
                    "ridge_first_step_rmse_rad": first_step_rmse_rad(ridge_pred, dataset.y_delta[test_mask]),
                    "knn_chunk_rmse_rad": rmse_rad(knn_pred, dataset.y_delta[test_mask]),
                    "knn_first_step_rmse_rad": first_step_rmse_rad(knn_pred, dataset.y_delta[test_mask]),
                }
            )
    else:
        metrics["leave_one_label_out_skipped"] = True
        metrics["extra_train_diagnostics_skipped"] = True
    for section in ("train",):
        for key, value in list(metrics[section].items()):
            if key.endswith("_rad"):
                metrics[section][key.replace("_rad", "_deg")] = float(math.degrees(value))
    for row in metrics["leave_one_label_out"]:
        for key, value in list(row.items()):
            if key.endswith("_rad"):
                row[key.replace("_rad", "_deg")] = float(math.degrees(float(value)))
    return metrics


def save_policy(
    dataset: ChunkDataset,
    *,
    output_dir: Path,
    overwrite: bool,
    ridge_lambda: float,
    knn_k: int,
    bc_hidden_dims: list[int],
    bc_epochs: int,
    bc_batch_size: int,
    bc_learning_rate: float,
    bc_weight_decay: float,
    seed: int,
    trim_idle_prefix_enabled: bool,
    trim_start_delta_rad: float,
    trim_preroll_s: float,
    include_phase_features: bool,
    include_gcsl_features: bool,
    include_deleaf_side_features: bool,
    pad_terminal_chunks: bool,
    goal_jitter_count: int,
    goal_jitter_std_m: float,
    goal_jitter_max_m: float,
    goal_normalizer_min_std_m: float,
    bc_backend: str,
    torch_device: str | None,
    print_progress: bool,
    skip_loo_eval: bool,
    store_training_arrays: bool,
    label_weight: dict[str, float] | None = None,
) -> dict[str, Any]:
    existing_outputs = [output_dir / "policy_arrays.npz", output_dir / "policy_metadata.json"]
    if not overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            f"refusing to overwrite existing policy directory: {output_dir}. "
            "Pass --overwrite intentionally, or provide a new --output-dir for this run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    x_min_std = feature_min_std(dataset.feature_names, goal_normalizer_min_std_m=goal_normalizer_min_std_m)
    sample_weights = None
    if label_weight:
        sample_weights = np.ones(len(dataset.source_labels), dtype=float)
        for _i, _lbl in enumerate(dataset.source_labels):
            if _lbl in label_weight:
                sample_weights[_i] = float(label_weight[_lbl])
    bc_model = fit_mlp_bc_backend(
        dataset.x,
        dataset.y_delta,
        hidden_dims=bc_hidden_dims,
        epochs=bc_epochs,
        batch_size=bc_batch_size,
        learning_rate=bc_learning_rate,
        weight_decay=bc_weight_decay,
        seed=seed,
        backend=bc_backend,
        torch_device=torch_device,
        print_progress=print_progress,
        x_min_std=x_min_std,
        sample_weights=sample_weights,
    )
    x_mean, x_std = normalizer(dataset.x, min_std=x_min_std)
    y_mean, y_std = normalizer(dataset.y_delta)
    if skip_loo_eval:
        ridge = {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "weights": np.zeros((dataset.x.shape[1] + 1, dataset.y_delta.shape[1]), dtype=float),
        }
    else:
        ridge = fit_ridge(dataset.x, dataset.y_delta, ridge_lambda, x_min_std=x_min_std)
    arrays_path = output_dir / "policy_arrays.npz"
    array_payload: dict[str, Any] = {
        "bc_x_mean": bc_model["x_mean"],
        "bc_x_std": bc_model["x_std"],
        "bc_y_mean": bc_model["y_mean"],
        "bc_y_std": bc_model["y_std"],
        "ridge_x_mean": ridge["x_mean"],
        "ridge_x_std": ridge["x_std"],
        "ridge_y_mean": ridge["y_mean"],
        "ridge_y_std": ridge["y_std"],
        "ridge_weights": ridge["weights"],
        "knn_x_mean": x_mean,
        "knn_x_std": x_std,
        "x_min_std": x_min_std,
        **{f"bc_w{i}": weight for i, weight in enumerate(bc_model["weights"])},
        **{f"bc_b{i}": bias for i, bias in enumerate(bc_model["biases"])},
    }
    if store_training_arrays:
        array_payload.update(
            {
                "x": dataset.x,
                "y_delta": dataset.y_delta,
                "y_abs": dataset.y_abs,
                "source_times_s": dataset.source_times_s,
            }
        )
        np.savez_compressed(arrays_path, **array_payload)
    else:
        np.savez(arrays_path, **array_payload)
    labels_path = output_dir / "source_labels.json"
    write_json(labels_path, {"source_labels": dataset.source_labels})
    metrics = evaluate_policy(
        dataset,
        ridge_lambda=ridge_lambda,
        knn_k=knn_k,
        bc_model=bc_model,
        bc_hidden_dims=bc_hidden_dims,
        bc_epochs=bc_epochs,
        bc_batch_size=bc_batch_size,
        bc_learning_rate=bc_learning_rate,
        bc_weight_decay=bc_weight_decay,
        seed=seed,
        bc_backend=bc_backend,
        torch_device=torch_device,
        x_min_std=x_min_std,
        skip_loo_eval=skip_loo_eval,
    )
    metadata: dict[str, Any] = {
        "schema": "egodelta_action_chunk_policy_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primitive": dataset.primitive,
        "recommended_runtime_mode": "gcsl_mlp" if include_gcsl_features else "bc_mlp",
        "reason": (
            "Primary policy is goal-conditioned supervised learning: a supervised MLP maps "
            "(state, GT 3D goal, horizon) tuples to future arm joint-delta chunks. "
            "KNN and ridge are stored only as diagnostics/fallbacks."
            if include_gcsl_features
            else "Primary policy is behavior cloning: a supervised MLP maps state and goal features to "
            "future arm joint-delta chunks. KNN and ridge are stored only as diagnostics/fallbacks."
        ),
        "sample_hz": dataset.sample_hz,
        "chunk_size": dataset.chunk_size,
        "chunk_horizon_s": float(dataset.chunk_size / dataset.sample_hz),
        "input": {
            "feature_names": dataset.feature_names,
            "feature_count": int(dataset.x.shape[1]),
            "contract": (
                "GT 3D target point in robot frame from action_library.goal.point_robot_m plus "
            "current arm state, arm velocity, TCP position, and GCSL horizon-to-goal features."
            if include_gcsl_features
            else "3D target point in robot frame plus current arm state, arm velocity, and TCP position."
            ),
            "goal_source": "action_library.goal.point_robot_m",
        },
        "output": {
            "action_space": dataset.action_space,
            "action_names": dataset.action_names,
            "action_count": int(dataset.y_delta.shape[1]),
            "contract": (
                "Flattened chunk of absolute future TCP poses in the robot frame; each step is "
                "[position_x, position_y, position_z (m), axis_angle_rx, axis_angle_ry, axis_angle_rz (rad)]. "
                "Decode each EEF pose to a joint command with the RB-Y1 IK layer at runtime "
                "(y_abs == y_delta for this action space)."
                if dataset.action_space == "eef"
                else "Flattened chunk of future arm joint deltas in radians; add each 7-DoF step to current q_arm."
            ),
        },
        "training": {
            "action_space": dataset.action_space,
            "demo_count": len(dataset.demos),
            "chunk_count": int(dataset.x.shape[0]),
            "source_kind_counts": {
                source_kind: sum(1 for demo in dataset.demos if demo.get("source_kind") == source_kind)
                for source_kind in sorted({str(demo.get("source_kind", "unknown")) for demo in dataset.demos})
            },
            "trim_idle_prefix": bool(trim_idle_prefix_enabled),
            "trim_start_delta_rad": float(trim_start_delta_rad),
            "trim_preroll_s": float(trim_preroll_s),
            "include_phase_features": bool(include_phase_features),
            "include_gcsl_features": bool(include_gcsl_features),
            "include_deleaf_side_features": bool(include_deleaf_side_features),
            "pad_terminal_chunks": bool(pad_terminal_chunks),
            "goal_jitter_count": int(goal_jitter_count),
            "goal_jitter_std_m": float(goal_jitter_std_m),
            "goal_jitter_max_m": float(goal_jitter_max_m),
            "goal_normalizer_min_std_m": float(goal_normalizer_min_std_m),
            "training_method": "gcsl" if include_gcsl_features else "goal_conditioned_bc",
            "bc_hidden_dims": list(bc_hidden_dims),
            "bc_epochs": int(bc_epochs),
            "bc_batch_size": int(bc_batch_size),
            "bc_learning_rate": float(bc_learning_rate),
            "bc_weight_decay": float(bc_weight_decay),
            "bc_seed": int(seed),
            "bc_backend": str(bc_model.get("backend", bc_backend)),
            "bc_device": str(bc_model.get("device", torch_device or "cpu")),
            "skip_loo_eval": bool(skip_loo_eval),
            "store_training_arrays": bool(store_training_arrays),
            "bc_best_normalized_mse": float(bc_model["best_normalized_mse"]),
            "bc_loss_history": bc_model["loss_history"],
            "ridge_lambda": float(ridge_lambda),
            "knn_k": int(knn_k),
            "arrays": str(arrays_path),
            "source_labels": str(labels_path),
        },
        "demos": dataset.demos,
        "metrics": metrics,
    }
    metadata_path = output_dir / "policy_metadata.json"
    write_json(metadata_path, metadata)
    return metadata


def train(args: argparse.Namespace) -> dict[str, Any]:
    include_labels = parse_label_set(args.include_labels)
    demos = load_demos(
        args.library_index,
        args.primitive,
        include_labels=include_labels,
        include_harvest_candidate_solves=bool(args.include_harvest_candidate_solves),
        harvest_candidate_top_k=int(args.harvest_candidate_top_k),
        harvest_candidate_only=bool(args.harvest_candidate_only),
        harvest_stop_after=args.harvest_stop_after,
        harvest_stop_match_tolerance_rad=float(args.harvest_stop_match_tolerance_rad),
        harvest_candidate_stop_after=args.harvest_candidate_stop_after,
        harvest_candidate_waypoint_dt_s=args.harvest_candidate_waypoint_dt_s,
        harvest_candidate_prepend_home=bool(args.harvest_candidate_prepend_home),
    )
    include_gcsl_features = bool(args.gcsl or args.include_gcsl_features)
    label_oversample: dict[str, int] = {}
    for item in args.label_oversample or []:
        if ":" not in item:
            raise ValueError(f"--label-oversample expects LABEL:FACTOR, got {item!r}")
        _lbl, _fac = item.rsplit(":", 1)
        label_oversample[_lbl] = int(_fac)
    dataset = build_chunk_dataset(
        demos,
        sample_hz=args.sample_hz,
        chunk_size=args.chunk_size,
        trim_idle_prefix_enabled=bool(args.trim_idle_prefix),
        trim_start_delta_rad=float(args.trim_start_delta_rad),
        trim_preroll_s=float(args.trim_preroll_s),
        include_phase_features=bool(args.include_phase_features),
        include_gcsl_features=include_gcsl_features,
        include_deleaf_side_features=bool(args.include_deleaf_side_features),
        pad_terminal_chunks=bool(args.pad_terminal_chunks),
        goal_jitter_count=int(args.goal_jitter_count),
        goal_jitter_std_m=float(args.goal_jitter_std_m),
        goal_jitter_max_m=float(args.goal_jitter_max_m),
        seed=int(args.seed),
        action_space=str(args.action_space),
        label_oversample=label_oversample,
        normalize_duration_s=(float(args.normalize_demo_duration_s) if args.normalize_demo_duration_s else None),
    )
    output_dir = args.output_dir or args.output_root / args.primitive
    knn_k = int(args.knn_k if args.knn_k is not None else (1 if args.primitive == "harvest" else 3))
    bc_backend, torch_device = resolve_bc_backend(args.bc_backend, args.torch_device)
    label_weight: dict[str, float] = {}
    for item in args.label_weight or []:
        if ":" not in item:
            raise ValueError(f"--label-weight expects LABEL:WEIGHT, got {item!r}")
        _wlbl, _wfac = item.rsplit(":", 1)
        label_weight[_wlbl] = float(_wfac)
    metadata = save_policy(
        dataset,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
        ridge_lambda=args.ridge_lambda,
        knn_k=knn_k,
        bc_hidden_dims=parse_hidden_dims(args.bc_hidden_dims),
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_weight_decay=args.bc_weight_decay,
        seed=args.seed,
        trim_idle_prefix_enabled=bool(args.trim_idle_prefix),
        trim_start_delta_rad=float(args.trim_start_delta_rad),
        trim_preroll_s=float(args.trim_preroll_s),
        include_phase_features=bool(args.include_phase_features),
        include_gcsl_features=include_gcsl_features,
        include_deleaf_side_features=bool(args.include_deleaf_side_features),
        pad_terminal_chunks=bool(args.pad_terminal_chunks),
        goal_jitter_count=int(args.goal_jitter_count),
        goal_jitter_std_m=float(args.goal_jitter_std_m),
        goal_jitter_max_m=float(args.goal_jitter_max_m),
        goal_normalizer_min_std_m=float(args.goal_normalizer_min_std_m),
        bc_backend=bc_backend,
        torch_device=torch_device,
        print_progress=bool(args.print_progress),
        skip_loo_eval=bool(args.skip_loo_eval),
        store_training_arrays=bool(args.store_training_arrays),
        label_weight=label_weight,
    )
    return {
        "metadata": str(output_dir / "policy_metadata.json"),
        "arrays": str(output_dir / "policy_arrays.npz"),
        "primitive": args.primitive,
        "include_labels": sorted(include_labels) if include_labels is not None else None,
        "demo_count": metadata["training"]["demo_count"],
        "chunk_count": metadata["training"]["chunk_count"],
        "sample_hz": metadata["sample_hz"],
        "chunk_size": metadata["chunk_size"],
        "chunk_horizon_s": metadata["chunk_horizon_s"],
        "recommended_runtime_mode": metadata["recommended_runtime_mode"],
        "training_method": metadata["training"]["training_method"],
        "bc_backend": metadata["training"]["bc_backend"],
        "bc_device": metadata["training"]["bc_device"],
        "train_first_step_rmse_deg": metadata["metrics"]["train"]["bc_first_step_rmse_deg"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate goal-conditioned action-chunk policies from the curated policy action library."
    )
    parser.add_argument("--primitive", choices=("deleaf", "harvest"), required=True)
    parser.add_argument("--library-index", type=Path, default=DEFAULT_LIBRARY_INDEX)
    parser.add_argument(
        "--include-labels",
        default=None,
        help="Comma-separated action labels to include, for example AT1T1 or AT1T1,AT1T2.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing policy_arrays.npz/policy_metadata.json.")
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--action-space",
        choices=("joint", "eef"),
        default="joint",
        help=(
            "Policy target space. 'joint' (default): future joint-delta chunks, existing behavior. "
            "'eef': absolute future TCP poses (position + axis-angle), decoded to joints by the RB-Y1 IK layer at runtime."
        ),
    )
    parser.add_argument(
        "--include-harvest-candidate-solves",
        action="store_true",
        help=(
            "For harvest policies only, add archived solver-passing candidate plans as synthetic "
            "auxiliary waypoint demonstrations. These are not hardware-confirmed successes."
        ),
    )
    parser.add_argument(
        "--harvest-candidate-top-k",
        type=int,
        default=5,
        help="Number of archived harvest candidate plans to add per confirmed harvest action when enabled.",
    )
    parser.add_argument(
        "--harvest-candidate-only",
        action="store_true",
        help="For harvest, train only on archived solver candidate plans and exclude long hardware full-tail executions.",
    )
    parser.add_argument(
        "--harvest-stop-after",
        choices=("approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"),
        default=None,
        help="For real harvest execution demos, truncate the trajectory at the matching planner waypoint.",
    )
    parser.add_argument("--harvest-stop-match-tolerance-rad", type=float, default=0.05)
    parser.add_argument(
        "--harvest-candidate-stop-after",
        choices=("approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"),
        default=None,
        help="For harvest candidate demos, truncate each candidate after this waypoint.",
    )
    parser.add_argument(
        "--harvest-candidate-waypoint-dt-s",
        type=float,
        default=None,
        help="Synthetic seconds between consecutive harvest candidate waypoints; avoids stretching candidates over full hardware duration.",
    )
    parser.add_argument(
        "--harvest-candidate-prepend-home",
        action="store_true",
        help="For harvest candidate demos, prepend the left-arm home q/TCP before the solver approach waypoint.",
    )
    parser.add_argument(
        "--normalize-demo-duration-s",
        type=float,
        default=None,
        help="Time-rescale every demo to this common duration before resampling (preserves the spatial path, "
        "scales per-step deltas + velocities). Use to normalize slow-collected demos to a known-good rollout "
        "speed (e.g. 34) so a receding-horizon BC policy makes progress instead of crawling. None = keep each "
        "demo's native duration.",
    )
    parser.add_argument("--trim-idle-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim-start-delta-rad", type=float, default=0.01)
    parser.add_argument("--trim-preroll-s", type=float, default=1.0)
    parser.add_argument("--include-phase-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--include-deleaf-side-features",
        action="store_true",
        help="For deleaf policies, append stem-facing side features: left one-hot, right one-hot, and facing sign.",
    )
    parser.add_argument(
        "--gcsl",
        action="store_true",
        help=(
            "Train a goal-conditioned supervised learner using the curated GT 3D clicked target "
            "and horizon-to-goal features."
        ),
    )
    parser.add_argument(
        "--include-gcsl-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly include GCSL horizon features without changing other training options.",
    )
    parser.add_argument(
        "--pad-terminal-chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pad future chunks by holding the terminal q_arm so full-horizon policies can train on late-phase states.",
    )
    parser.add_argument(
        "--label-oversample",
        action="append",
        default=None,
        metavar="LABEL:FACTOR",
        help="Oversample a source label's chunks by an integer factor (repeatable), e.g. --label-oversample AD1:4 to up-weight a minority-side deleaf demo so the closed-loop vector field is balanced. Improves reliability on that known task.",
    )
    parser.add_argument(
        "--label-weight",
        action="append",
        default=None,
        metavar="LABEL:WEIGHT",
        help="Per-sample LOSS weight for a source label (repeatable), e.g. --label-weight AD1:4 to balance a minority-side demo WITHOUT duplicating rows (the normalizer stays on the true data, so it does not skew the fit like --label-oversample). Preferred over --label-oversample.",
    )
    parser.add_argument(
        "--goal-jitter-count",
        type=int,
        default=0,
        help="Number of jittered GT-goal copies to add per training chunk.",
    )
    parser.add_argument(
        "--goal-jitter-std-m",
        type=float,
        default=0.0,
        help="Isotropic Gaussian std in meters for goal jitter augmentation.",
    )
    parser.add_argument(
        "--goal-jitter-max-m",
        type=float,
        default=0.0,
        help="Optional max norm in meters for each goal jitter sample; 0 disables clipping.",
    )
    parser.add_argument(
        "--goal-normalizer-min-std-m",
        type=float,
        default=0.0,
        help="Minimum std floor in meters for goal_robot_m and goal_minus_tcp_robot_m input features.",
    )
    parser.add_argument("--bc-hidden-dims", default="128,128")
    parser.add_argument("--bc-epochs", type=int, default=1200)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-weight-decay", type=float, default=1e-4)
    parser.add_argument("--bc-backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--torch-device", default="auto", help="Torch device when --bc-backend uses torch: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--print-progress", action="store_true", help="Print BC loss every 10 percent of training.")
    parser.add_argument(
        "--skip-loo-eval",
        action="store_true",
        help="Skip expensive leave-one-label-out diagnostic retraining after fitting the final policy.",
    )
    parser.add_argument(
        "--store-training-arrays",
        action="store_true",
        help="Store full x/y training arrays in policy_arrays.npz. Disabled by default for faster deploy-policy saves.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument(
        "--knn-k",
        type=int,
        default=None,
        help="Nearest-neighbor chunk count. Defaults to 3 for deleafing and 1 for harvesting.",
    )
    args = parser.parse_args()
    if args.sample_hz <= 0.0:
        parser.error("--sample-hz must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.harvest_candidate_top_k < 0:
        parser.error("--harvest-candidate-top-k cannot be negative")
    if args.include_harvest_candidate_solves and args.primitive != "harvest":
        parser.error("--include-harvest-candidate-solves is only valid for --primitive harvest")
    if args.harvest_candidate_only and args.primitive != "harvest":
        parser.error("--harvest-candidate-only is only valid for --primitive harvest")
    if args.harvest_candidate_only and not args.include_harvest_candidate_solves:
        parser.error("--harvest-candidate-only requires --include-harvest-candidate-solves")
    if args.harvest_stop_after is not None and args.primitive != "harvest":
        parser.error("--harvest-stop-after is only valid for --primitive harvest")
    if args.harvest_stop_match_tolerance_rad <= 0.0:
        parser.error("--harvest-stop-match-tolerance-rad must be positive")
    if args.harvest_candidate_stop_after is not None and args.primitive != "harvest":
        parser.error("--harvest-candidate-stop-after is only valid for --primitive harvest")
    if args.harvest_candidate_waypoint_dt_s is not None and args.harvest_candidate_waypoint_dt_s <= 0.0:
        parser.error("--harvest-candidate-waypoint-dt-s must be positive when provided")
    if args.harvest_candidate_prepend_home and args.primitive != "harvest":
        parser.error("--harvest-candidate-prepend-home is only valid for --primitive harvest")
    if args.harvest_candidate_prepend_home and not args.include_harvest_candidate_solves:
        parser.error("--harvest-candidate-prepend-home requires --include-harvest-candidate-solves")
    if args.include_deleaf_side_features and args.primitive != "deleaf":
        parser.error("--include-deleaf-side-features is only valid for --primitive deleaf")
    if args.goal_jitter_count < 0:
        parser.error("--goal-jitter-count cannot be negative")
    if args.goal_jitter_std_m < 0.0:
        parser.error("--goal-jitter-std-m cannot be negative")
    if args.goal_jitter_max_m < 0.0:
        parser.error("--goal-jitter-max-m cannot be negative")
    if args.goal_normalizer_min_std_m < 0.0:
        parser.error("--goal-normalizer-min-std-m cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    summary = train(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
