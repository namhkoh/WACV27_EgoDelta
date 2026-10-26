from __future__ import annotations

import argparse
import json
import os
import sys
import threading
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

from robot.kinematics.geometry import orientation_residual
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path


DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_DATASET = "a270c038"
DEFAULT_MODEL = "a"
DEFAULT_URDF_JAW_CENTER_TOOLFIT = Path("D:/tmp/egodelta_isaac_left_jaw_center_toolfit.json")
TASK_NAME = "teleop_tcp_capture"


def _pose_json(transform: np.ndarray) -> dict[str, Any]:
    return {
        "position_robot_m": transform[:3, 3].astype(float).tolist(),
        "rotation_robot_from_tcp": transform[:3, :3].astype(float).tolist(),
        "matrix": transform.astype(float).tolist(),
    }


def _load_waypoint(path: Path | None, candidate_name: str | None, waypoint_name: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def _sample(
    robot: Any,
    kinematics: RBY1Kinematics,
    waypoint: dict[str, Any] | None,
    *,
    start_s: float,
    phase: str | None = None,
    phase_index: int | None = None,
) -> dict[str, Any]:
    q_live = np.asarray(robot.get_state().position, dtype=float)
    eef_pose = kinematics.eef_pose(q_live)
    tcp_pose = kinematics.tcp_pose(q_live)
    arm_indices = list(kinematics.active_arm.joint_indices)
    payload: dict[str, Any] = {
        "elapsed_s": time.monotonic() - start_s,
        "q_full_rad": q_live.astype(float).tolist(),
        "q_arm_rad": q_live[arm_indices].astype(float).tolist(),
        "eef": _pose_json(eef_pose),
        "tcp": _pose_json(tcp_pose),
    }
    if phase is not None:
        payload["phase"] = phase
        payload["phase_index"] = phase_index
    if waypoint is not None:
        target = np.asarray(waypoint["target_tcp_pose"], dtype=float).reshape(4, 4)
        payload["comparison"] = {
            "waypoint": waypoint["name"],
            "target_tcp_position_robot_m": target[:3, 3].astype(float).tolist(),
            "position_error_m": (tcp_pose[:3, 3] - target[:3, 3]).astype(float).tolist(),
            "position_error_norm_m": float(np.linalg.norm(tcp_pose[:3, 3] - target[:3, 3])),
            "orientation_error_rad": float(np.linalg.norm(orientation_residual(tcp_pose[:3, :3], target[:3, :3]))),
        }
    return payload


DEFAULT_PHASES = "approach,pre_cut,cut,retract"


def _capture_phase_tagged(
    robot: Any,
    kinematics: RBY1Kinematics,
    waypoint: dict[str, Any] | None,
    args: argparse.Namespace,
    phases: list[str],
) -> dict[str, Any]:
    """Tap-Enter phase-labeled teleop capture.

    A background thread logs joint+TCP state at args.hz (read-only FK, no motor
    commands). The main thread blocks on input(); each Enter advances the phase
    (approach -> pre_cut -> cut -> retract) and snapshots the latest sample as a
    boundary key pose. The final Enter (after the last phase) stops and saves.
    So N phases => N Enter taps: (N-1) advance + 1 stop.
    """
    period_s = 1.0 / float(args.hz)
    samples: list[dict[str, Any]] = []
    phase_boundaries: list[dict[str, Any]] = []
    phase_lock = threading.Lock()
    phase_box = [0]
    stop_event = threading.Event()
    worker_error: list[BaseException] = []
    start_s = time.monotonic()

    def worker() -> None:
        next_s = start_s
        try:
            while not stop_event.is_set() and (time.monotonic() - start_s) <= float(args.duration_s):
                with phase_lock:
                    idx = phase_box[0]
                samples.append(
                    _sample(robot, kinematics, waypoint, start_s=start_s, phase=phases[idx], phase_index=idx)
                )
                next_s += period_s
                stop_event.wait(max(0.0, next_s - time.monotonic()))
        except BaseException as exc:  # surface to the main thread; never leave a silent dead thread
            worker_error.append(exc)
            stop_event.set()

    thread = threading.Thread(target=worker, name="teleop-capture", daemon=True)
    print(
        f"[teleop] phase-tagging ON | {args.hz:.0f} Hz | arm={args.arm} | toolfit={Path(args.toolfit).name}\n"
        f"[teleop] Enable compliant/gravity-comp mode elsewhere; this tool only RECORDS (no motor commands).\n"
        f"[teleop] Phase 1/{len(phases)}: '{phases[0]}' — hand-guide the arm now (recording)."
    )
    thread.start()
    stopped_by = "phase_complete"
    try:
        for i in range(1, len(phases)):
            input(f"[teleop] Press Enter to end '{phases[i - 1]}' -> begin '{phases[i]}' ...")
            if worker_error:
                break
            with phase_lock:
                phase_box[0] = i
            last = samples[-1] if samples else {}
            phase_boundaries.append(
                {
                    "boundary_index": i,
                    "from_phase": phases[i - 1],
                    "to_phase": phases[i],
                    "elapsed_s": last.get("elapsed_s"),
                    "sample_index": len(samples),
                    "q_arm_rad": last.get("q_arm_rad"),
                    "q_full_rad": last.get("q_full_rad"),
                    "tcp": last.get("tcp"),
                    "eef": last.get("eef"),
                }
            )
            print(f"[teleop] Phase {i + 1}/{len(phases)}: '{phases[i]}'.")
        if not worker_error:
            input(f"[teleop] Press Enter to STOP after '{phases[-1]}' and save ...")
    except KeyboardInterrupt:
        stopped_by = "keyboard_interrupt"
    stop_event.set()
    thread.join(timeout=5.0)
    if worker_error:
        raise RuntimeError(f"teleop capture worker failed: {worker_error[0]!r}")
    if stopped_by == "phase_complete" and (time.monotonic() - start_s) > float(args.duration_s):
        stopped_by = "duration"
    return {
        "method": "teleop_tcp_capture",
        "dataset_name": args.dataset_name,
        "arm": args.arm,
        "address": args.address,
        "model": args.model,
        "toolfit_path": str(args.toolfit),
        "duration_s": float(args.duration_s),
        "hz": float(args.hz),
        "stopped_by": stopped_by,
        "sample_count": len(samples),
        "plan": None if args.plan is None else str(args.plan),
        "candidate_name": args.candidate_name,
        "waypoint": args.waypoint if args.plan is not None else None,
        "phase_tagging": True,
        "phases": phases,
        "phase_boundaries": phase_boundaries,
        "demo_label": getattr(args, "demo_label", None),
        "samples": samples,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    if getattr(args, "phase_tagging", False) and Path(args.toolfit) == DEFAULT_URDF_JAW_CENTER_TOOLFIT:
        raise SystemExit(
            "--phase-tagging: pass --toolfit <right-knife repivot toolfit>; the default is the "
            "LEFT jaw-center file, so the logged right-knife TCP would be ~157mm wrong."
        )
    waypoint = _load_waypoint(args.plan, args.candidate_name, args.waypoint)
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {args.address}")
    kinematics = RBY1Kinematics.from_sdk_model(
        robot.get_dynamics(),
        robot.model(),
        arm=args.arm,
        toolfit_path=args.toolfit,
    )
    if getattr(args, "phase_tagging", False):
        phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]
        if len(phases) < 2:
            raise ValueError("--phases needs at least 2 phase names")
        return _capture_phase_tagged(robot, kinematics, waypoint, args, phases)
    period_s = 1.0 / float(args.hz)
    if args.manual_stop:
        input("Press Enter to start teleop capture. Press Enter again while capturing to stop and save.")
    start_s = time.monotonic()
    next_s = start_s
    samples: list[dict[str, Any]] = []
    stopped_by = "duration"
    try:
        while time.monotonic() - start_s <= float(args.duration_s):
            samples.append(_sample(robot, kinematics, waypoint, start_s=start_s))
            if args.manual_stop and _enter_pressed():
                stopped_by = "manual_enter"
                break
            next_s += period_s
            time.sleep(max(0.0, next_s - time.monotonic()))
    except KeyboardInterrupt:
        stopped_by = "keyboard_interrupt"
        if not samples:
            raise
    return {
        "method": "teleop_tcp_capture",
        "dataset_name": args.dataset_name,
        "arm": args.arm,
        "address": args.address,
        "model": args.model,
        "toolfit_path": str(args.toolfit),
        "duration_s": float(args.duration_s),
        "hz": float(args.hz),
        "stopped_by": stopped_by,
        "sample_count": len(samples),
        "plan": None if args.plan is None else str(args.plan),
        "candidate_name": args.candidate_name,
        "waypoint": args.waypoint if args.plan is not None else None,
        "samples": samples,
    }


def _enter_pressed() -> bool:
    if os.name != "nt":
        return False
    import msvcrt

    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\r", "\n"):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture live teleop joint states and calibrated TCP poses.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_URDF_JAW_CENTER_TOOLFIT)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--manual-stop", action="store_true", help="Press Enter during capture to stop and save early.")
    parser.add_argument(
        "--phase-tagging",
        action="store_true",
        help="Tap Enter to delimit phases (approach->pre_cut->cut->retract): records a per-sample phase tag "
        "and a key pose at each boundary. Runs the 10Hz capture on a background thread; main thread waits on Enter.",
    )
    parser.add_argument(
        "--phases",
        default=DEFAULT_PHASES,
        help="Comma-separated ordered phase names for --phase-tagging (N phases => N Enter taps: N-1 advance + 1 stop).",
    )
    parser.add_argument("--demo-label", default=None, help="Cut-point label (e.g. AD1/BD3) written into the artifact metadata.")
    parser.add_argument("--run-label", default="teleop_capture")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--waypoint", default="grasp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = capture(args)
    output = artifact_path(
        TASK_NAME,
        f"{args.dataset_name}_{args.run_label}",
        runs_root=args.runs_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "sample_count": payload["sample_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
