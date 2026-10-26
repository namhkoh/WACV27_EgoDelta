from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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

from robot.kinematics.geometry import orientation_residual
from robot.kinematics.rby1_model import RBY1Kinematics


DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_MODEL = "a"
DEFAULT_DATASET = "a270c038"


def default_toolfit_path(dataset_name: str) -> Path:
    return (
        Path(_REPO_ROOT)
        / "robot"
        / "calibration"
        / "session"
        / dataset_name
        / "T_robot_from_aria_selected_current_best_target_toolfit.json"
    )


def load_waypoint(plan_path: Path, candidate_name: str | None, waypoint_name: str) -> dict[str, Any] | None:
    if plan_path is None:
        return None
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate = payload.get("best_candidate")
    if candidate_name is not None:
        matches = [item for item in payload.get("candidates", []) if item.get("name") == candidate_name]
        if not matches:
            raise RuntimeError(f"candidate not found: {candidate_name}")
        candidate = matches[0]
    if not candidate:
        raise RuntimeError("plan artifact has no selected candidate")
    matches = [item for item in candidate.get("waypoints", []) if item.get("name") == waypoint_name]
    if not matches:
        raise RuntimeError(f"waypoint not found: {waypoint_name}")
    return matches[0]


def connect_kinematics(address: str, model_name: str, arm: str, toolfit_path: Path) -> tuple[Any, RBY1Kinematics]:
    robot = rby.create_robot(address, model_name)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
    return robot, RBY1Kinematics.from_sdk_model(robot.get_dynamics(), robot.model(), arm=arm, toolfit_path=toolfit_path)


def pose_summary(transform: np.ndarray) -> dict[str, Any]:
    return {
        "position_robot_m": [float(value) for value in transform[:3, 3]],
        "rotation_robot_from_tcp": [[float(value) for value in row] for row in transform[:3, :3]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report calibrated live RB-Y1 TCP pose without commanding motion.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--toolfit-path", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--waypoint", default="grasp")
    args = parser.parse_args()

    toolfit_path = args.toolfit_path or default_toolfit_path(args.dataset_name)
    waypoint = load_waypoint(args.plan, args.candidate_name, args.waypoint)
    robot, kinematics = connect_kinematics(args.address, args.model, args.arm, toolfit_path)
    q_live = np.asarray(robot.get_state().position, dtype=float)
    eef_pose = kinematics.eef_pose(q_live)
    tcp_pose = kinematics.tcp_pose(q_live)
    payload: dict[str, Any] = {
        "arm": args.arm,
        "toolfit_path": str(toolfit_path),
        "eef": pose_summary(eef_pose),
        "tcp": pose_summary(tcp_pose),
    }
    if waypoint is not None:
        target = np.asarray(waypoint["target_tcp_pose"], dtype=float).reshape(4, 4)
        payload["comparison"] = {
            "plan": str(args.plan),
            "candidate_name": args.candidate_name,
            "waypoint": args.waypoint,
            "target_tcp_position_robot_m": [float(value) for value in target[:3, 3]],
            "position_error_m": [float(value) for value in tcp_pose[:3, 3] - target[:3, 3]],
            "position_error_norm_m": float(np.linalg.norm(tcp_pose[:3, 3] - target[:3, 3])),
            "orientation_error_rad": float(np.linalg.norm(orientation_residual(tcp_pose[:3, :3], target[:3, :3]))),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
