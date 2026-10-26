from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from robot.kinematics.apps.execute_harvest_plan import (
    DEFAULT_GRIPPER_URL,
    DEFAULT_LEFT_CLOSE_POSITION,
    DEFAULT_LEFT_OPEN_POSITION,
    DEFAULT_LEFT_OPEN_REACHED_POSITION,
    _gripper_close,
    _gripper_open,
)
from robot.server.gripper_status import command_targets, fetch_status
from robot.kinematics.utils.realsense_click_move_arm import (
    DEFAULT_ADDRESS,
    DEFAULT_HOME,
    DEFAULT_POWER,
    DEFAULT_SERVO,
    command_arm_only,
    connect_motion_robot,
    validate_home_state,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_q_samples(payload: dict[str, Any]) -> list[np.ndarray]:
    samples = payload.get("samples", [])
    q_samples: list[np.ndarray] = []
    for sample in samples:
        if not isinstance(sample, dict) or "q_arm_rad" not in sample:
            continue
        q = np.asarray(sample["q_arm_rad"], dtype=float).reshape(7)
        q_samples.append(q)
    if not q_samples:
        raise RuntimeError("trajectory has no samples with q_arm_rad")
    return q_samples


def selected_candidate(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidate = payload.get("best_candidate") or payload.get("selected_candidate")
    if isinstance(candidate, dict):
        return candidate
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        passing = [item for item in candidates if isinstance(item, dict) and item.get("hard_gate", {}).get("ok", False)]
        if passing:
            return passing[0]
    return None


def event_indices_from_plan(q_samples: list[np.ndarray], plan_json: Path | None) -> dict[str, int]:
    if plan_json is None:
        return {}
    payload = read_json(plan_json)
    candidate = selected_candidate(payload)
    if candidate is None:
        return {}
    events: dict[str, int] = {}
    for waypoint in candidate.get("waypoints", []):
        name = waypoint.get("name")
        if name not in {"pre_grasp", "grasp", "basket_place"}:
            continue
        ik_result = waypoint.get("ik_result", {})
        if "q_arm_rad" not in ik_result:
            continue
        target_q = np.asarray(ik_result["q_arm_rad"], dtype=float).reshape(7)
        distances = [float(np.linalg.norm(q - target_q)) for q in q_samples]
        events[name] = int(np.argmin(distances))
    return events


def event_indices(q_samples: list[np.ndarray], plan_json: Path | None) -> dict[str, int]:
    indices = event_indices_from_plan(q_samples, plan_json)
    last = len(q_samples) - 1
    return {
        "pre_grasp": int(indices.get("pre_grasp", round(0.30 * last))),
        "grasp": int(indices.get("grasp", round(0.42 * last))),
        "basket_place": int(indices.get("basket_place", round(0.82 * last))),
    }


def apply_event_overrides(events: dict[str, int], args: argparse.Namespace, sample_count: int) -> dict[str, int]:
    updated = dict(events)
    overrides = {
        "pre_grasp": args.pre_grasp_index,
        "grasp": args.grasp_index,
        "basket_place": args.basket_place_index,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        index = int(value)
        if index < 0 or index >= sample_count:
            raise ValueError(f"--{name.replace('_', '-')}-index must be in [0, {sample_count - 1}]")
        updated[name] = index
    return updated


def replay_indices(sample_count: int, stride: int, required: set[int]) -> list[int]:
    stride = max(1, int(stride))
    indices = set(range(0, sample_count, stride))
    indices.add(sample_count - 1)
    indices.update(index for index in required if 0 <= index < sample_count)
    return sorted(indices)


def jittered_q_samples(
    q_samples: list[np.ndarray],
    *,
    event_indices_: dict[str, int],
    jitter_deg: float,
    seed: int | None,
) -> tuple[list[np.ndarray], dict[str, Any] | None]:
    if jitter_deg <= 0.0:
        return q_samples, None
    rng = random.Random(seed)
    amplitude = math.radians(float(jitter_deg))
    entry_offset = np.asarray([rng.uniform(-amplitude, amplitude) for _ in range(7)], dtype=float)
    exit_offset = np.asarray([rng.uniform(-amplitude, amplitude) for _ in range(7)], dtype=float)
    pre_grasp_index = max(1, int(event_indices_["pre_grasp"]))
    basket_index = min(len(q_samples) - 2, int(event_indices_["basket_place"]))
    last = len(q_samples) - 1
    varied: list[np.ndarray] = []
    for index, q in enumerate(q_samples):
        offset = np.zeros(7, dtype=float)
        if index < pre_grasp_index:
            phase = float(index) / float(pre_grasp_index)
            envelope = math.sin(math.pi * phase)
            offset += entry_offset * envelope
        elif index > basket_index and last > basket_index:
            phase = float(index - basket_index) / float(last - basket_index)
            envelope = math.sin(math.pi * phase)
            offset += exit_offset * envelope
        varied.append(np.asarray(q, dtype=float) + offset)
    return varied, {
        "entry_exit_jitter_deg": float(jitter_deg),
        "seed": seed,
        "entry_offset_rad": entry_offset.astype(float).tolist(),
        "exit_offset_rad": exit_offset.astype(float).tolist(),
        "note": "Jitter tapers to zero before pre_grasp and after basket_place; grasp samples are unchanged.",
    }


def execute_replay(args: argparse.Namespace) -> dict[str, Any]:
    trajectory_path = Path(args.trajectory_json)
    trajectory = read_json(trajectory_path)
    q_samples = trajectory_q_samples(trajectory)
    source_hz = float(trajectory.get("hz", 10.0))
    elapsed_s = [
        float(sample.get("elapsed_s", index / source_hz))
        for index, sample in enumerate(trajectory.get("samples", []))
        if isinstance(sample, dict) and "q_arm_rad" in sample
    ]
    if len(elapsed_s) != len(q_samples):
        raise RuntimeError("trajectory timing and q sample counts differ")
    events = apply_event_overrides(event_indices(q_samples, args.event_plan_json), args, len(q_samples))
    q_samples, jitter_report = jittered_q_samples(
        q_samples,
        event_indices_=events,
        jitter_deg=float(args.entry_exit_jitter_deg),
        seed=args.variation_seed,
    )
    stop_index_by_name = {
        "pre_grasp": events["pre_grasp"],
        "grasp": events["grasp"],
        "basket_place": events["basket_place"],
        "end": len(q_samples) - 1,
    }
    stop_index = int(stop_index_by_name[args.stop_after])
    q_samples = q_samples[: stop_index + 1]
    elapsed_s = elapsed_s[: stop_index + 1]
    required = {index for index in events.values() if 0 <= index <= stop_index}
    selected_indices = replay_indices(len(q_samples), args.stride, required)
    summary: dict[str, Any] = {
        "method": "arm_joint_trajectory_replay",
        "trajectory_json": str(trajectory_path),
        "source_method": trajectory.get("method"),
        "source_run_label": trajectory.get("run_label"),
        "source_sample_count": len(q_samples),
        "event_plan_json": None if args.event_plan_json is None else str(args.event_plan_json),
        "event_indices": events,
        "stop_after": args.stop_after,
        "stop_index": stop_index,
        "commanded_indices": selected_indices,
        "command_count": len(selected_indices),
        "arm": args.arm,
        "execute": bool(args.execute),
        "use_gripper": bool(args.use_gripper),
        "gripper_target_source": args.gripper_target_source,
        "minimum_time_s": float(args.minimum_time_s),
        "stride": int(args.stride),
        "source_hz": source_hz,
        "source_duration_s": float(elapsed_s[-1] - elapsed_s[0]),
        "stream_timing_source": args.stream_timing_source,
        "replay_speed": float(args.replay_speed),
        "joint_group_scope": f"{args.arm}_arm_only",
        "gripper_events": [],
        "jitter": jitter_report,
    }
    if not args.execute:
        return summary

    left_open_position = float(args.left_open_position)
    left_open_reached_position = float(args.left_open_reached_position)
    left_close_position = float(args.left_close_position)
    if args.use_gripper and args.gripper_target_source == "live_status":
        status = fetch_status(args.gripper_url, timeout_s=float(args.gripper_timeout_s))
        targets = command_targets(status, arm=args.arm)
        left_open_position = targets.open_position
        left_open_reached_position = targets.open_position
        left_close_position = targets.close_position
        summary["gripper_command_targets"] = {
            "source": "live_status",
            "calibration_schema": targets.calibration_schema,
            "open_position": targets.open_position,
            "close_position": targets.close_position,
            "encoder_span": targets.encoder_span,
        }
    else:
        summary["gripper_command_targets"] = {
            "source": "absolute_cli",
            "open_position": left_open_position,
            "open_reached_position": left_open_reached_position,
            "close_position": left_close_position,
        }

    home = read_json(args.home)
    robot = connect_motion_robot(address=args.address, model=args.model, power=args.power, servo=args.servo)
    if not args.skip_home_check:
        summary["home_check"] = validate_home_state(
            robot,
            home,
            tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)),
        )

    if args.stage_to_start:
        stage_feedback = robot.send_command(
            command_arm_only(
                args.arm,
                q_samples[0],
                minimum_time_s=float(args.stage_minimum_time_s),
                velocity_limit_rad_s=float(args.stage_velocity_limit_rad_s),
                acceleration_limit_rad_s2=float(args.stage_acceleration_limit_rad_s2),
            ),
            int(max(float(args.timeout_s), float(args.stage_minimum_time_s) + 5.0)),
        ).get()
        if stage_feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"failed to stage {args.arm} arm to recorded start: {stage_feedback.finish_code}")
        summary["stage_to_start"] = {
            "ok": True,
            "minimum_time_s": float(args.stage_minimum_time_s),
            "velocity_limit_rad_s": float(args.stage_velocity_limit_rad_s),
            "acceleration_limit_rad_s2": float(args.stage_acceleration_limit_rad_s2),
            "joint_group_scope": f"{args.arm}_arm_only",
        }
    else:
        summary["stage_to_start"] = {"ok": False, "reason": "disabled"}

    cmd_stream = None
    if args.command_stream:
        try:
            robot.set_parameter("default.acceleration_limit_scaling", "0.8")
            robot.set_parameter("joint_position_command.cutoff_frequency", str(args.command_cutoff_hz))
        except Exception as exc:  # noqa: BLE001
            print(f"[command-stream] set_parameter warning: {exc}")
        cmd_stream = robot.create_command_stream(priority=1)
        summary["command_stream"] = {
            "timing_source": args.stream_timing_source,
            "fixed_dt_s": float(args.stream_dt_s),
            "replay_speed": float(args.replay_speed),
            "hold_time_s": float(args.command_hold_time_s),
            "cutoff_hz": float(args.command_cutoff_hz),
        }
        print(
            f"[command-stream] ON: timing={args.stream_timing_source} speed={args.replay_speed} "
            f"hold={args.command_hold_time_s}s cutoff={args.command_cutoff_hz}Hz"
        )

    opened = False
    closed = False
    released = False
    previous_index = selected_indices[0]
    try:
        for selected_position, index in enumerate(selected_indices):
            if args.stage_to_start and selected_position == 0 and index == 0:
                continue
            q_arm = q_samples[index]
            if cmd_stream is not None:
                if args.stream_timing_source == "recorded":
                    command_dt_s = max(
                        0.02,
                        float(elapsed_s[index] - elapsed_s[previous_index]) / float(args.replay_speed),
                    )
                else:
                    command_dt_s = float(args.stream_dt_s)
                builder = command_arm_only(
                    args.arm,
                    q_arm,
                    minimum_time_s=command_dt_s,
                    velocity_limit_rad_s=float(args.velocity_limit_rad_s),
                    acceleration_limit_rad_s2=float(args.acceleration_limit_rad_s2),
                    control_hold_time_s=float(args.command_hold_time_s),
                )
                try:
                    cmd_stream.send_command(builder)
                except RuntimeError as exc:
                    if "expired" in str(exc).lower():
                        print("  [command-stream] expired (idle) -> recreating stream")
                        cmd_stream = robot.create_command_stream(priority=1)
                        cmd_stream.send_command(builder)
                    else:
                        raise
                time.sleep(command_dt_s * 0.95)
            else:
                feedback = robot.send_command(
                    command_arm_only(
                        args.arm,
                        q_arm,
                        minimum_time_s=float(args.minimum_time_s),
                        velocity_limit_rad_s=float(args.velocity_limit_rad_s),
                        acceleration_limit_rad_s2=float(args.acceleration_limit_rad_s2),
                    ),
                    int(max(float(args.timeout_s), float(args.minimum_time_s) + 5.0)),
                ).get()
                if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
                    raise RuntimeError(f"trajectory replay command index {index} failed: {feedback.finish_code}")

            if args.use_gripper and args.arm == "left" and (not opened) and index >= events["pre_grasp"]:
                summary["gripper_events"].append(
                    _gripper_open(
                        args.gripper_url,
                        arm="left",
                        action="open",
                        position=left_open_position,
                        expected_reached_position=left_open_reached_position,
                        tolerance=args.left_open_tolerance,
                        current_limit=args.gripper_open_current_limit,
                        hold_s=args.gripper_hold_s,
                        timeout_s=args.gripper_timeout_s,
                        post_command_wait_s=args.gripper_post_command_wait_s,
                        post_command_poll_s=args.gripper_post_command_poll_s,
                    )
                )
                opened = True
            if args.use_gripper and args.arm == "left" and (not closed) and index >= events["grasp"]:
                summary["gripper_events"].append(
                    _gripper_close(
                        args.gripper_url,
                        arm="left",
                        position=left_close_position,
                        current_limit=args.gripper_close_current_limit,
                        hold_s=args.gripper_hold_s,
                        timeout_s=args.gripper_timeout_s,
                        min_motion=args.gripper_close_min_motion,
                        post_command_wait_s=args.gripper_post_command_wait_s,
                        post_command_poll_s=args.gripper_post_command_poll_s,
                    )
                )
                closed = True
            if args.use_gripper and args.arm == "left" and (not released) and index >= events["basket_place"]:
                summary["gripper_events"].append(
                    _gripper_open(
                        args.gripper_url,
                        arm="left",
                        action="release",
                        position=left_open_position,
                        expected_reached_position=left_open_reached_position,
                        tolerance=args.left_open_tolerance,
                        current_limit=args.gripper_open_current_limit,
                        hold_s=args.gripper_hold_s,
                        timeout_s=args.gripper_timeout_s,
                        post_command_wait_s=args.gripper_post_command_wait_s,
                        post_command_poll_s=args.gripper_post_command_poll_s,
                    )
                )
                released = True
            if cmd_stream is None:
                time.sleep(0.05)
            previous_index = index
    finally:
        if cmd_stream is not None:
            try:
                cmd_stream.cancel()
            except Exception as exc:  # noqa: BLE001
                summary["command_stream_cancel_warning"] = str(exc)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a captured arm joint trajectory with optional harvest gripper events.")
    parser.add_argument("trajectory_json", type=Path)
    parser.add_argument("--event-plan-json", type=Path, default=None)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--power", default=DEFAULT_POWER)
    parser.add_argument("--servo", default=DEFAULT_SERVO)
    parser.add_argument("--minimum-time-s", type=float, default=0.35)
    parser.add_argument("--stage-to-start", action="store_true",
                        help="Move only the selected arm slowly to the trajectory's first sample before replay.")
    parser.add_argument("--stage-minimum-time-s", type=float, default=6.0)
    parser.add_argument("--stage-velocity-limit-rad-s", type=float, default=0.5)
    parser.add_argument("--stage-acceleration-limit-rad-s2", type=float, default=1.0)
    parser.add_argument("--velocity-limit-rad-s", type=float, default=1.0)
    parser.add_argument("--acceleration-limit-rad-s2", type=float, default=2.0)
    parser.add_argument("--command-stream", action="store_true",
                        help="SMOOTH replay: stream paced joint targets via robot.create_command_stream() so the "
                             "controller blends consecutive samples (vs choppy blocking point-to-point moves). "
                             "Pair with a low --stride.")
    parser.add_argument("--stream-dt-s", type=float, default=0.06,
                        help="Per-sample pacing for --command-stream when --stream-timing-source=fixed.")
    parser.add_argument("--stream-timing-source", choices=("recorded", "fixed"), default="recorded",
                        help="Use recorded elapsed_s deltas by default; fixed preserves legacy constant pacing.")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Recorded-timing speed multiplier. 1.0 preserves demonstration timing.")
    parser.add_argument("--command-hold-time-s", type=float, default=1.0,
                        help="control_hold_time on each streamed command (s) -- bridges a pacing gap so it doesn't stop.")
    parser.add_argument("--command-cutoff-hz", type=float, default=5.0,
                        help="joint_position_command.cutoff_frequency (Hz) low-pass smoothing for the stream.")
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--stop-after", choices=("pre_grasp", "grasp", "basket_place", "end"), default="end")
    parser.add_argument("--pre-grasp-index", type=int, default=None)
    parser.add_argument("--grasp-index", type=int, default=None)
    parser.add_argument("--basket-place-index", type=int, default=None)
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--skip-home-check", action="store_true")
    parser.add_argument("--entry-exit-jitter-deg", type=float, default=0.0)
    parser.add_argument("--variation-seed", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument("--gripper-target-source", choices=("live_status", "absolute_cli"), default="live_status")
    parser.add_argument("--left-open-position", type=float, default=DEFAULT_LEFT_OPEN_POSITION)
    parser.add_argument("--left-open-reached-position", type=float, default=DEFAULT_LEFT_OPEN_REACHED_POSITION)
    parser.add_argument("--left-open-tolerance", type=float, default=0.75)
    parser.add_argument("--left-close-position", type=float, default=DEFAULT_LEFT_CLOSE_POSITION)
    parser.add_argument("--gripper-open-current-limit", type=float, default=3.0)
    parser.add_argument("--gripper-close-current-limit", type=float, default=2.0)
    parser.add_argument("--gripper-hold-s", type=float, default=1.0)
    parser.add_argument("--gripper-timeout-s", type=float, default=8.0)
    parser.add_argument("--gripper-post-command-wait-s", type=float, default=3.0)
    parser.add_argument("--gripper-post-command-poll-s", type=float, default=0.2)
    parser.add_argument("--gripper-close-min-motion", type=float, default=0.25)
    args = parser.parse_args()
    if args.replay_speed <= 0.0:
        parser.error("--replay-speed must be positive")
    if args.velocity_limit_rad_s <= 0.0 or args.acceleration_limit_rad_s2 <= 0.0:
        parser.error("joint velocity and acceleration limits must be positive")
    return args


def main() -> int:
    args = parse_args()
    payload = execute_replay(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.execute:
        print("dry_run: pass --execute to command the robot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
