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
DEFAULT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_right_knife_tip_edge_repivot_060226.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "robot" / "calibration" / "session" / "a270c038"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record the calibrated knife TIP position in robot frame at hand-guided contact points "
        "(camera-free, FK + tool offset). Use to touch-measure fixed stakes accurately."
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--arm", choices=("right",), default="right")
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "stake_touch_points.json")
    args = parser.parse_args()

    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {args.address}")
    kinematics = RBY1Kinematics.from_sdk_model(
        robot.get_dynamics(), robot.model(), arm=args.arm, toolfit_path=args.toolfit
    )

    print("Hand-guide the BLADE TIP to touch the stake surface. Press Enter to capture a point; type a")
    print("label first (e.g. 'pymain_low') then Enter to tag it; type q then Enter to finish.")
    points: list[dict[str, Any]] = []
    while True:
        raw = input(f"[{len(points) + 1}] label or Enter to capture (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        q_live = np.asarray(robot.get_state().position, dtype=float)
        tip = np.asarray(kinematics.tcp_pose(q_live)[:3, 3], dtype=float)
        arm_indices = list(kinematics.active_arm.joint_indices)
        points.append(
            {
                "index": len(points) + 1,
                "label": raw or f"p{len(points) + 1}",
                "timestamp_s": time.time(),
                "tip_robot_m": tip.tolist(),
                "q_arm_rad": q_live[arm_indices].astype(float).tolist(),
            }
        )
        print(f"  captured {points[-1]['label']}: tip [{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}]")

    payload = {
        "method": "tool_tip_contact_points_robot_frame",
        "address": args.address,
        "model": args.model,
        "arm": args.arm,
        "toolfit": str(args.toolfit),
        "point_count": len(points),
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "point_count": len(points)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
