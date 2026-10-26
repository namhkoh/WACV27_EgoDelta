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


from robot.kinematics.harvest_primitive import (
    load_template_candidate,
    retarget_harvest_candidate,
    validate_harvest_candidate,
)
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.harvest_target_selection import validate_target_artifact_goal
from robot.kinematics.utils.realsense_click_move_arm import validate_home_state


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_left_gripper_jaw_center_01.json"
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "robot"
    / "policy"
    / "policy_action_library"
    / "AT1T1"
    / "success_current"
    / "actions"
    / "AT1T1_instance_001_step_01_AT1T1"
    / "planner"
    / "a270c038_AT1T1_success_current.json"
)


def parse_point(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("goal must contain x,y,z in robot-frame metres")
    point = np.asarray([float(part) for part in parts], dtype=float)
    if not np.all(np.isfinite(point)):
        raise argparse.ArgumentTypeError("goal must contain finite values")
    return point


def bounds(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float) -> dict[str, tuple[float, float]]:
    return {"x": (min_x, max_x), "y": (min_y, max_y), "z": (min_z, max_z)}


def connect_state_robot(address: str, model: str) -> Any:
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
    return robot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly retarget and preflight the frozen eight-phase harvest primitive without robot motion."
    )
    parser.add_argument("--goal-robot-m", required=True, type=parse_point)
    parser.add_argument("--template-plan", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--target-artifact", type=Path, default=None)
    parser.add_argument(
        "--target-artifact-tolerance-m",
        type=float,
        default=0.002,
        help="Maximum disagreement between --goal-robot-m and a selected_point stored in --target-artifact.",
    )
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--dataset-name", default="a270c038")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--extra-insertion-m",
        type=float,
        default=0.0,
        help="Additional insertion beyond the validated template surface-to-grasp depth.",
    )
    parser.add_argument("--height-offset-m", type=float, default=0.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.15)
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--joint-limit-margin-rad", type=float, default=0.003)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.012)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--path-samples-per-segment", type=int, default=20)
    args = parser.parse_args()

    for path in (args.template_plan, args.home, args.toolfit):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.target_artifact is not None and not args.target_artifact.is_file():
        parser.error(f"--target-artifact does not exist: {args.target_artifact}")
    target_metadata = (
        {
            "artifact": None,
            "point_semantics": "visible_fruit_surface",
            "artifact_goal_checked": False,
        }
        if args.target_artifact is None
        else validate_target_artifact_goal(
            args.target_artifact,
            args.goal_robot_m,
            tolerance_m=args.target_artifact_tolerance_m,
        )
    )
    live_fruit_radius_m = target_metadata.get("estimated_fruit_radius_m")
    if (
        target_metadata.get("selection_method") == "manual_click_red_instance_center"
        and live_fruit_radius_m is None
    ):
        raise RuntimeError(
            "centered red-fruit target artifact has no metric fruit radius; "
            "recapture it with the current RGB-D picker before planning"
        )

    home = json.loads(args.home.read_text(encoding="utf-8"))
    robot = connect_state_robot(args.address, args.model)
    home_check = validate_home_state(
        robot,
        home,
        tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)),
    )
    kinematics = RBY1Kinematics.from_sdk_model(
        robot.get_dynamics(),
        robot.model(),
        arm="left",
        toolfit_path=args.toolfit,
    )
    locked_state = kinematics.locked_state_from_home(args.home)
    q_live = np.asarray(robot.get_state().position, dtype=float).reshape(-1)
    q_left = q_live[list(kinematics.active_arm.joint_indices)]

    template = load_template_candidate(args.template_plan, args.candidate_name)
    candidate, retarget = retarget_harvest_candidate(
        template,
        kinematics=kinematics,
        locked_state=locked_state,
        live_goal_robot_m=args.goal_robot_m,
        extra_insertion_m=args.extra_insertion_m,
        height_offset_m=args.height_offset_m,
        position_tolerance_m=args.position_tolerance_m,
        orientation_tolerance_rad=args.orientation_tolerance_rad,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
        live_nominal_grasp_insertion_m=live_fruit_radius_m,
    )
    hard_gate = validate_harvest_candidate(
        candidate,
        kinematics=kinematics,
        locked_state=locked_state,
        start_q_arm_rad=q_left,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
        minimum_clearance_m=args.minimum_clearance_m,
        clearance_pair_count=args.clearance_pair_count,
        samples_per_segment=args.path_samples_per_segment,
        manipulation_bounds=bounds(0.20, 1.10, -0.75, 0.75, 0.40, 1.45),
        transit_bounds=bounds(-0.25, 1.10, -0.80, 0.80, 0.35, 1.45),
    )
    candidate["hard_gate"] = hard_gate
    payload = {
        "schema": "egodelta_target_conditioned_harvest_plan_v2",
        "method": "size_normalized_demonstration_conditioned_harvest_primitive",
        "dataset_name": args.dataset_name,
        "run_label": args.run_label,
        "robot_motion": False,
        "joint_group_scope": "left_arm_only",
        "policy_motion_executed": False,
        "target": {
            "point_robot_m": args.goal_robot_m.tolist(),
            **target_metadata,
        },
        "template": {
            "plan": str(args.template_plan.resolve()),
            "candidate_name": template["name"],
        },
        "configuration": {
            "extra_insertion_m": float(args.extra_insertion_m),
            "live_fruit_radius_m": live_fruit_radius_m,
            "height_offset_m": float(args.height_offset_m),
            "position_tolerance_m": float(args.position_tolerance_m),
            "orientation_tolerance_rad": float(args.orientation_tolerance_rad),
            "minimum_clearance_m": float(args.minimum_clearance_m),
            "path_samples_per_segment": int(args.path_samples_per_segment),
            "toolfit": str(args.toolfit.resolve()),
            "home": str(args.home.resolve()),
        },
        "home_check": home_check,
        "retarget": retarget,
        "hard_gate": hard_gate,
        "best_candidate_name": candidate["name"] if hard_gate["ok"] else None,
        "best_candidate": candidate if hard_gate["ok"] else None,
        "candidates": [candidate],
    }
    output = args.output_json or artifact_path(
        "target_conditioned_harvest_plan",
        f"{args.dataset_name}_{args.run_label}",
        runs_root=args.runs_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "hard_gate": hard_gate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
