from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.kinematics.geometry import orientation_residual
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import artifact_path


DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_DATASET = "a270c038"
DEFAULT_MODEL = "a"
DEFAULT_LEFT_TOOLFIT = Path("D:/tmp/egodelta_isaac_left_jaw_center_toolfit.json")
DEFAULT_RIGHT_TOOLFIT = Path("C:/path/to/projects/EgoDelta/robot/calibration/tooltips/a270c038_right_knife_tip_edge_deleaf_01.json")
TASK_NAME = "robot_tcp_trajectory_capture"
POLICY_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def pose_json(transform: np.ndarray) -> dict[str, Any]:
    return {
        "position_robot_m": transform[:3, 3].astype(float).tolist(),
        "rotation_robot_from_tcp": transform[:3, :3].astype(float).tolist(),
        "matrix": transform.astype(float).tolist(),
    }


def load_waypoint(path: Path | None, candidate_name: str | None, waypoint_name: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("best_candidate") or payload.get("selected_candidate")
    if candidate_name is not None:
        matches = [item for item in payload.get("candidates", []) if item.get("name") == candidate_name]
        if not matches:
            matches = [item for item in payload.get("candidate_summaries", []) if item.get("name") == candidate_name]
        if not matches:
            raise RuntimeError(f"candidate not found: {candidate_name}")
        candidate = matches[0]
    if (not candidate or not candidate.get("waypoints")) and "waypoints" in payload:
        candidate = payload
    if not candidate:
        raise RuntimeError("plan artifact has no selected candidate")
    matches = [item for item in candidate.get("waypoints", []) if item.get("name") == waypoint_name]
    if not matches:
        raise RuntimeError(f"waypoint not found: {waypoint_name}")
    return matches[0]


def sample(
    robot: Any,
    kinematics: RBY1Kinematics,
    waypoint: dict[str, Any] | None,
    *,
    start_s: float,
) -> dict[str, Any]:
    state = robot.get_state()
    q_live = np.asarray(state.position, dtype=float)
    eef_pose = kinematics.eef_pose(q_live)
    tcp_pose = kinematics.tcp_pose(q_live)
    arm_indices = list(kinematics.active_arm.joint_indices)
    payload: dict[str, Any] = {
        "elapsed_s": time.monotonic() - start_s,
        "wall_time_unix_s": time.time(),
        "q_full_rad": q_live.astype(float).tolist(),
        "q_arm_rad": q_live[arm_indices].astype(float).tolist(),
        "eef": pose_json(eef_pose),
        "tcp": pose_json(tcp_pose),
    }
    if hasattr(state, "velocity"):
        payload["dq_full_rad_s"] = np.asarray(state.velocity, dtype=float).astype(float).tolist()
    if waypoint is not None and "target_tcp_pose" in waypoint:
        target = np.asarray(waypoint["target_tcp_pose"], dtype=float).reshape(4, 4)
        payload["comparison"] = {
            "waypoint": waypoint["name"],
            "target_tcp_position_robot_m": target[:3, 3].astype(float).tolist(),
            "position_error_m": (tcp_pose[:3, 3] - target[:3, 3]).astype(float).tolist(),
            "position_error_norm_m": float(np.linalg.norm(tcp_pose[:3, 3] - target[:3, 3])),
            "orientation_error_rad": float(np.linalg.norm(orientation_residual(tcp_pose[:3, :3], target[:3, :3]))),
        }
    return payload


def enter_pressed() -> bool:
    if os.name != "nt":
        return False
    import msvcrt

    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\r", "\n"):
            return True
    return False


def capture(args: argparse.Namespace, stop_event: Any | None = None) -> dict[str, Any]:
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    toolfit = args.toolfit
    if toolfit is None:
        toolfit = DEFAULT_RIGHT_TOOLFIT if args.arm == "right" else DEFAULT_LEFT_TOOLFIT
    waypoint = load_waypoint(args.plan, args.candidate_name, args.waypoint)
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {args.address}")
    kinematics = RBY1Kinematics.from_sdk_model(
        robot.get_dynamics(),
        robot.model(),
        arm=args.arm,
        toolfit_path=toolfit,
    )
    if args.manual_start:
        input("Press Enter to start robot trajectory capture.")
    period_s = 1.0 / float(args.hz)
    start_s = time.monotonic()
    next_s = start_s
    samples: list[dict[str, Any]] = []
    stopped_by = "duration"
    try:
        while time.monotonic() - start_s <= float(args.duration_s):
            if stop_event is not None and stop_event.is_set():
                stopped_by = "stop_event"
                break
            samples.append(sample(robot, kinematics, waypoint, start_s=start_s))
            if args.manual_stop and enter_pressed():
                stopped_by = "manual_enter"
                break
            next_s += period_s
            sleep_s = max(0.0, next_s - time.monotonic())
            if stop_event is not None:
                if stop_event.wait(sleep_s):
                    stopped_by = "stop_event"
                    break
            else:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        stopped_by = "keyboard_interrupt"
        if not samples:
            raise
    return {
        "method": "robot_tcp_trajectory_capture",
        "dataset_name": args.dataset_name,
        "run_label": args.run_label,
        "arm": args.arm,
        "address": args.address,
        "model": args.model,
        "toolfit_path": str(toolfit),
        "duration_s": float(args.duration_s),
        "hz": float(args.hz),
        "stopped_by": stopped_by,
        "sample_count": len(samples),
        "plan": None if args.plan is None else str(args.plan),
        "candidate_name": args.candidate_name,
        "waypoint": args.waypoint if args.plan is not None else None,
        "samples": samples,
    }


def write_capture(payload: dict[str, Any], args: argparse.Namespace) -> Path:
    output = artifact_path(
        TASK_NAME,
        f"{args.dataset_name}_{args.run_label}",
        runs_root=args.runs_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture live robot joint states and calibrated TCP poses.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--toolfit", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--manual-start", action="store_true")
    parser.add_argument("--manual-stop", action="store_true", help="Press Enter during capture to stop and save early.")
    parser.add_argument("--run-label", default="robot_tcp_capture")
    parser.add_argument("--runs-root", type=Path, default=POLICY_RUNS_ROOT)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--waypoint", default="grasp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = capture(args)
    output = write_capture(payload, args)
    print(json.dumps({"artifact": str(output), "sample_count": payload["sample_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
