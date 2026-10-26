from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
else:
    repo_root = Path(__file__).resolve().parents[2]

from robot.kinematics.geometry import normalize, transform_from_rotation_translation
from robot.kinematics.ik_solver import solve_tcp_pose_ik
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.types import IKSolverConfig


DEFAULT_TARGET_JSON = Path("D:/tmp/egodelta_isaac_target_sphere_pose.json")
DEFAULT_URDF = repo_root / "submodules" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model.urdf"
DEFAULT_HOME = repo_root / "robot" / "utils" / "homestate.json"
DEFAULT_TOOLFIT = repo_root / "robot" / "calibration" / "session" / "a270c038" / "T_robot_from_aria_selected_current_best_target_toolfit.json"
DEFAULT_OUTPUT = Path("D:/tmp/egodelta_isaac_sphere_ik_solution.json")


def parse_vec3(text: str, *, name: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must be formatted as x,y,z")
    try:
        return np.asarray([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric values") from exc


def load_target(args: argparse.Namespace) -> np.ndarray:
    if args.target_robot_m is not None:
        return parse_vec3(args.target_robot_m, name="target robot point")
    data = json.loads(args.target_json.read_text(encoding="utf-8"))
    if not data.get("ok", False):
        raise RuntimeError(f"target JSON is not valid: {args.target_json}")
    return np.asarray(data["world_position_m"], dtype=float).reshape(3)


def aim_rotation_from_backoff(backoff_axis_robot: np.ndarray) -> np.ndarray:
    z_axis = normalize(backoff_axis_robot)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, up))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def target_pose_from_sphere(
    *,
    sphere_position_robot_m: np.ndarray,
    home_tcp_position_robot_m: np.ndarray,
    stand_off_m: float,
    contact_offset_tcp_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    sphere = np.asarray(sphere_position_robot_m, dtype=float).reshape(3)
    home_tcp = np.asarray(home_tcp_position_robot_m, dtype=float).reshape(3)
    contact_offset = np.asarray(contact_offset_tcp_m, dtype=float).reshape(3)
    backoff_axis = normalize(home_tcp - sphere)
    contact_center = sphere + float(stand_off_m) * backoff_axis
    rotation = aim_rotation_from_backoff(backoff_axis)
    target_tcp = contact_center - rotation @ contact_offset
    diagnostics = {
        "sphere_position_robot_m": sphere.tolist(),
        "home_tcp_position_robot_m": home_tcp.tolist(),
        "backoff_axis_robot": backoff_axis.tolist(),
        "stand_off_m": float(stand_off_m),
        "contact_offset_tcp_m": contact_offset.tolist(),
        "target_contact_center_robot_m": contact_center.tolist(),
        "target_tcp_position_robot_m": target_tcp.tolist(),
    }
    return transform_from_rotation_translation(rotation, target_tcp), diagnostics


def solve(args: argparse.Namespace) -> dict[str, object]:
    kinematics = RBY1Kinematics.from_urdf(
        urdf_path=args.urdf,
        model_name=args.model,
        arm=args.arm,
        toolfit_path=args.toolfit,
    )
    locked_state = kinematics.locked_state_from_home(args.home)
    home_tcp = kinematics.tcp_pose(locked_state.full_q_rad)[:3, 3]
    target_pose, diagnostics = target_pose_from_sphere(
        sphere_position_robot_m=load_target(args),
        home_tcp_position_robot_m=home_tcp,
        stand_off_m=args.stand_off_m,
        contact_offset_tcp_m=parse_vec3(args.contact_offset_tcp_m, name="contact offset"),
    )
    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    result = solve_tcp_pose_ik(
        kinematics,
        locked_state,
        target_pose,
        seed_q_arm_rad=home_q_arm,
        reference_q_arm_rad=home_q_arm,
        config=IKSolverConfig(
            position_weight=args.position_weight,
            orientation_weight=args.orientation_weight,
            posture_weight=args.posture_weight,
            smoothness_weight=args.smoothness_weight,
            clearance_weight=args.clearance_weight,
            clearance_margin_m=args.clearance_margin_m,
            clearance_pair_count=args.clearance_pair_count,
            max_iterations=args.max_iterations,
        ),
    )
    return {
        "ok": bool(result.ok and result.position_error_m <= args.position_tolerance_m),
        "model": args.model,
        "arm": args.arm,
        "joint_names": list(kinematics.active_arm.joint_names),
        "q_arm_rad": result.q_arm_rad.tolist(),
        "target_tcp_pose": target_pose.tolist(),
        "diagnostics": diagnostics,
        "ik_result": {
            "ok": bool(result.ok),
            "position_error_m": float(result.position_error_m),
            "orientation_error_rad": float(result.orientation_error_rad),
            "residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "message": result.message,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve left-arm IK for an Isaac sphere target.")
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument("--target-robot-m", default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument("--model", default="a")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--stand-off-m", type=float, default=0.0)
    parser.add_argument("--contact-offset-tcp-m", default="0,0,0")
    parser.add_argument("--position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--orientation-weight", type=float, default=0.0)
    parser.add_argument("--position-weight", type=float, default=20.0)
    parser.add_argument("--posture-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--clearance-weight", type=float, default=2.0)
    parser.add_argument("--clearance-margin-m", type=float, default=0.03)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = solve(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
