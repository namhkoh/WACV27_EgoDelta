from __future__ import annotations

import os
import sys

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

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.ik_solver import solve_tcp_pose_ik
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.task_frame import (
    DEFAULT_TARGET_PATTERN,
    HarvestTaskFrame,
    discover_harvest_task_frame_paths,
    load_harvest_task_frame,
)
from robot.kinematics.types import IKSolverConfig, LockedRobotState


DEFAULT_ADDRESS = "192.0.2.10:50051"
REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_TARGET_DIR = REPO_ROOT / "robot" / "calibration" / "targets" / "a270c038"
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_TOOLFIT = (
    REPO_ROOT
    / "robot"
    / "calibration"
    / "session"
    / "a270c038"
    / "T_robot_from_aria_selected_current_best_target_toolfit.json"
)


def solve_task_frames(
    *,
    paths: tuple[Path, ...],
    address: str,
    model: str,
    arm: str,
    home_path: Path,
    toolfit_path: Path,
    config: IKSolverConfig,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    runs_root: Path,
    run_label: str,
    dataset_name: str,
    backoff_offsets_m: tuple[float, ...],
    chain_seed: bool,
) -> tuple[dict[str, Any], ...]:
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    kinematics = RBY1Kinematics.from_live_robot(
        address=address,
        model_name=model,
        arm=arm,
        toolfit_path=toolfit_path,
    )
    locked_state = kinematics.locked_state_from_home(home_path)
    results: list[dict[str, Any]] = []
    for path in paths:
        frame = load_harvest_task_frame(path)
        waypoints = _waypoint_targets(frame, backoff_offsets_m)
        previous_q_arm: np.ndarray | None = None
        for waypoint_name, backoff_m, target_pose in waypoints:
            payload = _solve_one_waypoint(
                kinematics=kinematics,
                locked_state=locked_state,
                frame=frame,
                target_pose=target_pose,
                waypoint_name=waypoint_name,
                backoff_m=backoff_m,
                config=config,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                min_clearance_m=min_clearance_m,
                address=address,
                model=model,
                arm=arm,
                home_path=home_path,
                toolfit_path=toolfit_path,
                dataset_name=dataset_name,
                chain_seed=chain_seed,
                seed_q_arm_rad=previous_q_arm if chain_seed else None,
                reference_q_arm_rad=previous_q_arm if chain_seed else None,
            )
            if chain_seed and payload["ik_result"]["ok"]:
                previous_q_arm = np.asarray(payload["ik_result"]["q_arm_rad"], dtype=float)
            artifact = artifact_path(
                "single_waypoint_ik_smoke",
                f"{dataset_name}_{frame.target_name}_{waypoint_name}_tcp_{run_label}",
                runs_root=runs_root,
            )
            payload["artifact_path"] = str(artifact)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results.append(payload)
    return tuple(results)


def _solve_one_waypoint(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    target_pose: np.ndarray,
    waypoint_name: str,
    backoff_m: float,
    config: IKSolverConfig,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    address: str,
    model: str,
    arm: str,
    home_path: Path,
    toolfit_path: Path,
    dataset_name: str,
    chain_seed: bool,
    seed_q_arm_rad: np.ndarray | None,
    reference_q_arm_rad: np.ndarray | None,
) -> dict[str, Any]:
        result = solve_tcp_pose_ik(
            kinematics,
            locked_state,
            target_pose,
            seed_q_arm_rad=seed_q_arm_rad,
            reference_q_arm_rad=reference_q_arm_rad,
            config=config,
        )
        q_full = kinematics.q_full_from_arm(locked_state, result.q_arm_rad)
        final_pose = kinematics.tcp_pose(q_full)
        clearance = nearest_self_clearances(
            kinematics,
            q_full,
            pair_count=config.clearance_pair_count,
            margin_m=min_clearance_m,
        )
        hard_gate_ok = bool(
            result.ok
            and result.position_error_m <= position_tolerance_m
            and result.orientation_error_rad <= orientation_tolerance_rad
            and clearance.ok
        )
        payload = {
            "schema": "rby1_single_waypoint_ik_smoke",
            "version": 1,
            "dataset_name": dataset_name,
            "waypoint": {
                "name": waypoint_name,
                "backoff_m": backoff_m,
                "chain_seed": chain_seed,
            },
            "target": frame.to_summary(),
            "robot": {
                "address": address,
                "model": model,
                "arm": arm,
                "home_path": str(home_path),
                "toolfit_path": str(toolfit_path),
            },
            "solver_config": asdict(config),
            "hard_gate": {
                "ok": hard_gate_ok,
                "position_tolerance_m": position_tolerance_m,
                "orientation_tolerance_rad": orientation_tolerance_rad,
                "min_clearance_m": min_clearance_m,
                "minimum_clearance_m": clearance.minimum_distance_m,
            },
            "ik_result": {
                "ok": result.ok,
                "q_arm_rad": result.q_arm_rad.tolist(),
                "residual_norm": result.residual_norm,
                "position_error_m": result.position_error_m,
                "orientation_error_rad": result.orientation_error_rad,
                "message": result.message,
                "iterations": result.iterations,
            },
            "target_tcp_pose": target_pose.tolist(),
            "final_tcp_pose": final_pose.tolist(),
            "nearest_clearance_pairs": [asdict(pair) for pair in clearance.nearest_pairs],
        }
        return payload


def _waypoint_targets(frame: HarvestTaskFrame, backoff_offsets_m: tuple[float, ...]) -> tuple[tuple[str, float, np.ndarray], ...]:
    targets: list[tuple[str, float, np.ndarray]] = []
    for offset_m in backoff_offsets_m:
        pose = frame.tcp_grasp_pose.T.copy()
        pose[:3, 3] = frame.target_position_robot_m + float(offset_m) * frame.backoff_axis_robot
        targets.append((f"backoff_{_distance_label(offset_m)}", float(offset_m), pose))
    targets.append(("grasp", 0.0, frame.tcp_grasp_pose.T))
    return tuple(targets)


def _distance_label(distance_m: float) -> str:
    return f"{int(round(float(distance_m) * 100.0)):02d}cm"


def _paths_from_args(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.paths:
        return tuple(args.paths)
    return discover_harvest_task_frame_paths(args.target_dir, args.pattern)


def main() -> int:
    parser = argparse.ArgumentParser(description="No-motion single-waypoint IK smoke solve for harvest task frames.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--pattern", default=DEFAULT_TARGET_PATTERN)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.15)
    parser.add_argument("--min-clearance-m", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--position-weight", type=float, default=20.0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--posture-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--clearance-weight", type=float, default=2.0)
    parser.add_argument("--clearance-margin-m", type=float, default=0.03)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--backoff-offsets-m", default="0.05,0.10,0.15")
    parser.add_argument("--chain-seed", action="store_true")
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()

    config = IKSolverConfig(
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        posture_weight=args.posture_weight,
        smoothness_weight=args.smoothness_weight,
        max_iterations=args.max_iterations,
        clearance_weight=args.clearance_weight,
        clearance_margin_m=args.clearance_margin_m,
        clearance_pair_count=args.clearance_pair_count,
    )
    run_label = args.run_label or (
        f"posw{args.position_weight:g}_oriw{args.orientation_weight:g}_"
        f"clearw{args.clearance_weight:g}_clear{args.clearance_margin_m:g}_"
        f"it{args.max_iterations}_{datetime.now().strftime('%H%M%S')}"
    )
    dataset_name = args.dataset_name or _dataset_name(args.target_dir, tuple(args.paths))
    results = solve_task_frames(
        paths=_paths_from_args(args),
        address=args.address,
        model=args.model,
        arm=args.arm,
        home_path=args.home,
        toolfit_path=args.toolfit,
        config=config,
        position_tolerance_m=args.position_tolerance_m,
        orientation_tolerance_rad=args.orientation_tolerance_rad,
        min_clearance_m=args.min_clearance_m,
        runs_root=args.runs_root,
        run_label=run_label,
        dataset_name=dataset_name,
        backoff_offsets_m=_parse_offsets(args.backoff_offsets_m),
        chain_seed=args.chain_seed,
    )
    for result in results:
        print(
            result["target"]["target_name"],
            result["waypoint"]["name"],
            "hard_gate_ok=",
            result["hard_gate"]["ok"],
            "position_error_m=",
            result["ik_result"]["position_error_m"],
            "orientation_error_rad=",
            result["ik_result"]["orientation_error_rad"],
            "artifact=",
            result["artifact_path"],
        )
    return 0


def _parse_offsets(value: str) -> tuple[float, ...]:
    offsets = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if any(offset < 0.0 for offset in offsets):
        raise ValueError("backoff offsets must be non-negative")
    return offsets


def _dataset_name(target_dir: Path, paths: tuple[Path, ...]) -> str:
    if paths:
        parents = {Path(path).parent.name for path in paths}
        if len(parents) == 1:
            return next(iter(parents))
    return Path(target_dir).name


if __name__ == "__main__":
    raise SystemExit(main())
