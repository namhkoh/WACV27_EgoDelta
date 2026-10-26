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
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.rby1_model import RBY1Kinematics


REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_DATASET = "a270c038"
DEFAULT_MODEL = "a"
DEFAULT_RIGHT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "core" / "T_robot_from_aria_featuredepth_hanger_trusted_right_toolfit.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "robot" / "calibration" / "tooltips"


def pose_json(transform: np.ndarray) -> dict[str, Any]:
    return {
        "position_robot_m": transform[:3, 3].astype(float).tolist(),
        "rotation_robot_from_eef": transform[:3, :3].astype(float).tolist(),
        "matrix": transform.astype(float).tolist(),
    }


def fit_pivot_tip(eef_poses: list[np.ndarray]) -> dict[str, Any]:
    if len(eef_poses) < 4:
        raise ValueError("pivot calibration needs at least 4 contact poses")
    rows = []
    rhs = []
    for pose in eef_poses:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        rows.append(np.column_stack((rotation, -np.eye(3))))
        rhs.append(-translation)
    solution, _, rank, singular_values = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    tip_eef = solution[:3]
    contact_robot = solution[3:6]
    residuals = []
    for pose in eef_poses:
        predicted = pose[:3, :3] @ tip_eef + pose[:3, 3]
        residuals.append(float(np.linalg.norm(predicted - contact_robot)))
    return {
        "estimated_tool_offset_link_name": "ee_right",
        "estimated_tool_offset_m": tip_eef.astype(float).tolist(),
        "fixed_contact_point_robot_m": contact_robot.astype(float).tolist(),
        "sample_count": len(eef_poses),
        "residual_rms_m": float(np.sqrt(np.mean(np.square(residuals)))),
        "residual_max_m": float(np.max(residuals)),
        "residuals_m": residuals,
        "linear_system_rank": int(rank),
        "singular_values": singular_values.astype(float).tolist(),
    }


def connect_kinematics(args: argparse.Namespace) -> tuple[Any, RBY1Kinematics]:
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {args.address}")
    kinematics = RBY1Kinematics.from_sdk_model(
        robot.get_dynamics(),
        robot.model(),
        arm=args.arm,
        toolfit_path=args.reference_toolfit,
    )
    return robot, kinematics


def capture_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    robot, kinematics = connect_kinematics(args)
    samples: list[dict[str, Any]] = []
    poses: list[np.ndarray] = []
    print("Place the right knife tip on the same fixed physical point for every sample.")
    print("Use different wrist/arm orientations without sliding the tip. Press Enter to capture; type q then Enter to finish.")
    while len(samples) < int(args.count):
        command = input(f"[{len(samples) + 1}/{args.count}] capture? ").strip().lower()
        if command in {"q", "quit", "exit"}:
            break
        q_live = np.asarray(robot.get_state().position, dtype=float)
        eef_pose = kinematics.eef_pose(q_live)
        arm_indices = list(kinematics.active_arm.joint_indices)
        poses.append(eef_pose)
        samples.append(
            {
                "index": len(samples) + 1,
                "timestamp_s": time.time(),
                "q_arm_rad": q_live[arm_indices].astype(float).tolist(),
                "q_full_rad": q_live.astype(float).tolist(),
                "eef": pose_json(eef_pose),
            }
        )
        print(f"captured sample {len(samples)}")
    if len(samples) < int(args.min_count):
        raise RuntimeError(f"only captured {len(samples)} samples; need at least {args.min_count}")
    return samples, poses


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    label = args.run_label or "right_knife_tip_pivot_" + time.strftime("%Y%m%d_%H%M%S")
    return args.output_dir / f"{args.dataset_name}_{label}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a right-arm knife/tool tip by pivoting on one fixed contact point.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("right",), default="right")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--reference-toolfit", type=Path, default=DEFAULT_RIGHT_TOOLFIT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=6)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.count < args.min_count:
        parser.error("--count must be greater than or equal to --min-count")
    samples, poses = capture_samples(args)
    fit = fit_pivot_tip(poses)
    payload = {
        "method": "right_tool_tip_pivot_calibration",
        "dataset_name": args.dataset_name,
        "address": args.address,
        "model": args.model,
        "arm": args.arm,
        "reference_toolfit": str(args.reference_toolfit),
        "instructions": "All samples should keep the same physical knife tip point fixed against one stationary point.",
        **fit,
        "samples": samples,
    }
    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(path), "sample_count": fit["sample_count"], "residual_rms_m": fit["residual_rms_m"], "estimated_tool_offset_m": fit["estimated_tool_offset_m"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
