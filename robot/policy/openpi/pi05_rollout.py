"""Rollout client for the fine-tuned pi0.5 RB-Y1-a LEFT-arm pick-and-place policy.

Unlike ``openpi_rollout.py`` (pi05_droid -> Franka joint-velocities -> cartesian
retarget -> IK), this policy was fine-tuned on OUR embodiment and emits **native RB-Y1
left-arm joint targets**, so there is NO Franka retarget and NO IK -- we command the 7
joints directly. It is a single-arm, absolute-target cousin of ``molmobot_rollout.py``.

Pipeline per inference (dry-run by default; --execute is the master motion gate):
    capture base D455 + left-wrist D405 RGB  +  read left-arm q (gRPC) + gripper raw
      (HTTP /status id 1)  ->  obs {observation/image, observation/wrist_image,
      observation/state(8), prompt}  ->  client.infer(obs)  (openpi-client websocket)
      ->  result["actions"] : (H=10, 8) ABSOLUTE  [0:7]=left-arm joint angles,
          [7]=gripper raw  (AbsoluteActions already added state back; no decode)
      ->  per chunk step: clamp move (target - live q) to a per-step cap, joint-limit
          clamp, self-clearance gate  ->  command_body_left (drives left; HOLDS the right
          arm + torso at their frozen start pose, head untouched)  ->  gripper raw -> /jog.

CRITICAL CONTRACTS (verified against the openpi serving path):
  * The server does NOT apply the data-config repack at inference, so the client sends
    the un-repacked ``observation/*`` keys that RBY1Inputs reads directly.
  * Images are sent at NATIVE capture resolution (640x480); the server's ResizeImages
    (224,224) runs identically to training, so we must NOT pre-resize here.
  * ``--task`` MUST match the language string the dataset was converted with
    (prompt_from_task=True) -- default below is the project task. Verify with
    inspect_lerobot if unsure; a mismatched prompt is silently out-of-distribution.

Server (WSL2, from ~/openpi). NOTE: --port is a top-level arg (default 8000) and must
precede the policy:checkpoint subcommand; just omit it for 8000:
    uv run --no-sync scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi05_rby1_left_pnp \
      --policy.dir=$HOME/openpi/checkpoints/pi05_rby1_left_pnp/rby1_left_pnp_v1/5000

Client (Windows), smoke first, then dry-run, then execute:
    python -m robot.policy.openpi.pi05_rollout --no-robot --loop            # cameras+infer only
    python -m robot.policy.openpi.pi05_rollout --host <wsl-ip> --loop       # dry-run (reads robot, prints targets)
    python -m robot.policy.openpi.pi05_rollout --host <wsl-ip> --execute --gripper --loop  # MOTION (slow, per-step confirm)

SAFETY (per project memory): dry-run default; slow minimum_time; per-step joint-move
clamp; joint-limit + self-clearance gates; --confirm per waypoint (on unless --no-confirm).
Validate joint mapping in dry-run BEFORE --execute.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import rby1_sdk as rby

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.utils.realsense_click_move_arm import connect_motion_robot
from robot.policy.openpi.multicam_capture import MultiCamRealSense, list_devices
from robot.policy.openpi.openpi_rollout import (
    DEFAULT_GRIPPER_URL,
    command_gripper,
    connect_state_robot,
    read_gripper_position_raw,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_left_gripper_jaw_center_01.json"
DEFAULT_OPENPI_CLIENT_SRC = "C:/path/to/projects/openpi/packages/openpi-client/src"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "rby1_left_pnp_raw"  # recorded demos (for --goto-start nearest-start)

# Cameras: SAME devices + resolution the recorder used (record_pi05_demos.py) so the
# policy sees what it trained on. base = external D455, left_wrist = D405.
DEFAULT_BASE_SERIAL = os.environ.get("BASE_CAMERA_SERIAL")
DEFAULT_WRIST_SERIAL = os.environ.get("LEFT_WRIST_CAMERA_SERIAL")

# Default language instruction. The dataset's task string (prompt_from_task) was converted
# with an embedded newline + 2 spaces, and the model trained on THIS exact string, so the
# deploy prompt must match it byte-for-byte to stay in-distribution. Verified from the
# dataset meta/tasks.jsonl: "pick up the tomato and \n  place it in the box".
DEFAULT_TASK = "pick up the tomato and \n  place it in the box"

# Gripper raw-encoder safe band. The demos' left-gripper raw spanned ~[1.77, 4.73]
# (norm_stats q01/q99); clamp the policy's gripper target into this in-distribution band
# so an OOD output cannot drive the gripper to a hardware extreme. /jog 'position' is the
# same encoder space the recorder read from /status id 1.
DEFAULT_GRIP_MIN_RAW = 1.7
DEFAULT_GRIP_MAX_RAW = 4.8

# Mean demo-start LEFT-arm pose (rad), averaged over all 40 episodes -- the in-distribution
# pose to return the arm to before a rollout (--goto-start). The per-episode starts varied
# widely (std up to ~29 deg/joint), so this is a CENTRAL start, not a tight one; the policy
# tolerates a range, but the failed deploy run started OUTSIDE even that range.
DEFAULT_START_POSE_RAD = [-0.37877, 0.89128, -1.31277, -1.80964, -1.1381, 1.20145, 0.68764]
DEFAULT_START_GRIP = 2.157  # mean demo-start gripper raw (open)
# In-distribution OPEN gripper raw (demo open-cluster center ~1.78). Used as the gripper-read
# FALLBACK -- the old [min,max] band midpoint 3.25 is a sparse mid-transition value the recorder
# NEVER produced as a static reading (81st pct, only 6.9% of frames within +-0.25), so feeding it
# into observation/state[7] corrupts the grasp-gating proprio channel. Also the hysteresis "open".
DEFAULT_OPEN_GRIP_RAW = 1.78
# Gripper grasp HYSTERESIS (raw). The demo CLOSED cluster centers ~3.75 (median 3.56, spans
# 2.8-4.76, std 0.59); the policy regresses a drifting close value that can reach 4.76 -> squish.
# Instead: commit to a grasp once the policy crosses COMMIT, then HOLD a fixed firm-but-gentle
# value (NOT the drifting raw) so the grip force is consistent; release below RELEASE.
DEFAULT_GRASP_COMMIT_RAW = 2.8
DEFAULT_GRASP_HOLD_RAW = 3.4
DEFAULT_GRASP_RELEASE_RAW = 2.2

# Demonstrated per-joint workspace envelope (min/max LEFT-arm angles, rad) over ALL 13967
# demo frames. The demos grasped the tomato WITHOUT hitting the table, so clamping every
# command to this envelope is a table-SAFE workspace floor. This matters because the
# self-clearance guard only checks the robot against ITSELF -- it knows nothing about the
# table/environment, so without this an OOD policy can (and did) drive the arm into the table.
DEMO_Q_MIN_RAD = np.array([-1.3098, 0.1228, -1.9651, -2.2565, -2.0745, -0.1991, -0.1812])
DEMO_Q_MAX_RAD = np.array([0.8007, 1.2843, -0.0368, -0.4830, -0.2964, 1.6008, 1.4018])

ARM_DOF = 7
LEFT = "left"


def _import_openpi_client(client_src: str):
    if client_src and client_src not in sys.path:
        sys.path.insert(0, client_src)
    from openpi_client import websocket_client_policy  # noqa: E402
    return websocket_client_policy


def left_arm_indices(robot: Any) -> list[int]:
    return [int(i) for i in robot.model().left_arm_idx]


def read_state(robot: Any, left_idx: list[int], *, gripper_url: str, read_gripper: bool,
               last_grip: list[float | None], grip_fallback: float,
               read_fail: list[int] | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (q_left(7), gripper_raw, state8) matching the recorder exactly:
    state8 = concat([left-arm q (7), gripper raw encoder (1)]). A failed gripper read
    falls back to the last-good value, else to grip_fallback (the in-distribution OPEN
    value) -- never 0.0, which is far outside the trained gripper range and would push
    observation/state out of distribution. read_fail (if given) counts CONSECUTIVE read
    failures so the caller can halt when the servo has dropped off the bus."""
    full = np.asarray(robot.get_state().position, dtype=np.float32)
    q_left = full[left_idx].astype(np.float32)
    grip = 0.0
    if read_gripper:
        raw = read_gripper_position_raw(gripper_url, LEFT, 3.0)
        if raw is None:
            if read_fail is not None:
                read_fail[0] += 1
            cached = last_grip[0]
            print("  [gripper] proprio read FAILED -> using "
                  + (f"last-good {cached:.3f}" if cached is not None else f"open-fallback {grip_fallback:.3f}"))
            raw = cached if cached is not None else grip_fallback
        else:
            if read_fail is not None:
                read_fail[0] = 0
            last_grip[0] = float(raw)
        grip = float(raw)
    state8 = np.concatenate([q_left, np.array([grip], dtype=np.float32)]).astype(np.float32)
    return q_left, grip, state8


def build_obs(base_rgb: np.ndarray, wrist_rgb: np.ndarray, state8: np.ndarray, prompt: str) -> dict:
    """The server applies NO repack at inference, so we send the observation/* keys that
    RBY1Inputs reads directly. Images stay at native resolution (server ResizeImages 224)."""
    return {
        "observation/image": np.ascontiguousarray(base_rgb, dtype=np.uint8),
        "observation/wrist_image": np.ascontiguousarray(wrist_rgb, dtype=np.uint8),
        "observation/state": np.asarray(state8, dtype=np.float32).reshape(ARM_DOF + 1),
        "prompt": prompt,
    }


def _joint_cmd(position, minimum_time_s, vel, acc, control_hold_time_s=None):
    pos = np.asarray(position, dtype=float).reshape(-1)
    builder = rby.JointPositionCommandBuilder().set_minimum_time(float(minimum_time_s)).set_position(pos)
    if vel is not None:
        builder = builder.set_velocity_limit(np.full(pos.shape[0], float(vel), dtype=float))
    if acc is not None:
        builder = builder.set_acceleration_limit(np.full(pos.shape[0], float(acc), dtype=float))
    if control_hold_time_s is not None:  # streaming: keep tracking the last target between sends
        builder = builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(float(control_hold_time_s)))
    return builder


def command_body_left(left_q, right_hold, torso_hold, *, minimum_time_s,
                      velocity_limit_rad_s=None, acceleration_limit_rad_s2=None,
                      control_hold_time_s=None):
    """Drive the LEFT arm to left_q while HOLDING the right arm + torso at their captured
    (frozen) poses; the HEAD is deliberately NOT commanded (the controller holds it where it
    is). This avoids command_arm's hazard of sweeping the non-driven arm + head to
    homestate.json -- molmobot verified that sweep on hardware (driving left moved the right
    arm to home). control_hold_time_s != None adds a streaming command header so the controller
    keeps tracking between paced sends (use with --command-stream)."""
    body = (
        rby.BodyComponentBasedCommandBuilder()
        .set_torso_command(_joint_cmd(torso_hold, minimum_time_s, velocity_limit_rad_s, acceleration_limit_rad_s2, control_hold_time_s))
        .set_right_arm_command(_joint_cmd(right_hold, minimum_time_s, velocity_limit_rad_s, acceleration_limit_rad_s2, control_hold_time_s))
        .set_left_arm_command(_joint_cmd(left_q, minimum_time_s, velocity_limit_rad_s, acceleration_limit_rad_s2, control_hold_time_s))
    )
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body)
    )


def load_demo_starts(raw_dir):
    """Each demo episode's first-frame left-arm pose (rad), from the raw dataset. The demos
    started from a RANGE of poses (std ~25 deg), so --goto-start moves to the NEAREST real
    start -- in-distribution (with the matching wrist view) AND a short move. None if absent."""
    import glob
    rows = []
    for f in sorted(glob.glob(str(Path(raw_dir) / "episode_*" / "frames.npz"))):
        try:
            rows.append(np.load(f)["state"][0])
        except Exception:
            continue
    return np.asarray(rows, dtype=float)[:, :ARM_DOF] if rows else None


def compute_table_floor_z(kin, robot, left_idx, raw_dir, subsample):
    """Lowest gripper-TCP height (base-frame Z, m) over all demo frames, using the LIVE
    non-arm joints (torso) for FK. The lowest the gripper reached in the demos == the grasp
    height == a table-SAFE Cartesian floor (the per-joint envelope is a loose box that can
    still dip the gripper under the table; this is the precise plane). None if demos absent."""
    import glob
    rows = []
    for f in sorted(glob.glob(str(Path(raw_dir) / "episode_*" / "frames.npz"))):
        try:
            rows.append(np.load(f)["state"][:: max(1, int(subsample)), :ARM_DOF])
        except Exception:
            continue
    if not rows:
        return None
    qs = np.concatenate(rows, axis=0)
    full = np.asarray(robot.get_state().position, dtype=float)
    zmin = float("inf")
    for q in qs:
        qf = full.copy()
        qf[left_idx] = q
        zmin = min(zmin, float(kin.tcp_pose(qf)[2, 3]))
    return zmin


def goto_start_pose(robot, kin, left_idx, right_hold, torso_hold, target7, *, args) -> bool:
    """Slowly move the LEFT arm to target7 (rad) in clamped, clearance-checked increments
    (holding right arm + torso frozen), to put it at an in-distribution demo-start pose
    before the rollout. Returns True once within ~1 deg of the target."""
    print(f"[goto-start] moving left arm to start (deg)={np.round(np.rad2deg(target7), 1)}")
    if not args.no_confirm and input("  GOTO this start pose now? [y/N] ").strip().lower() != "y":
        print("  [goto-start] skipped by user.")
        return False
    cap = args.max_step_delta_rad
    for _ in range(5000):
        full = np.asarray(robot.get_state().position, dtype=float)
        q_live = full[left_idx]
        err = np.asarray(target7, dtype=float) - q_live
        if float(np.max(np.abs(err))) < np.deg2rad(1.0):
            print("  [goto-start] reached start pose.")
            return True
        cmd = q_live + np.clip(err, -cap, cap)
        lo, hi = kin.active_joint_limits(full)
        cmd = np.clip(cmd, lo + args.joint_limit_margin_rad, hi - args.joint_limit_margin_rad)
        if not args.no_envelope:  # same table-safe demonstrated-workspace floor
            em = np.deg2rad(args.envelope_margin_deg)
            cmd = np.clip(cmd, DEMO_Q_MIN_RAD - em, DEMO_Q_MAX_RAD + em)
        q_full_t = full.copy()
        q_full_t[left_idx] = cmd
        rep = nearest_self_clearances(kin, q_full_t, pair_count=args.clearance_pair_count,
                                      margin_m=args.min_clearance_m)
        if not rep.ok:
            print(f"  [goto-start] BLOCKED by self-clearance (min={rep.minimum_distance_m}) -- aborting.")
            return False
        builder = command_body_left(cmd, right_hold, torso_hold, minimum_time_s=args.minimum_time_s,
                                    velocity_limit_rad_s=args.velocity_limit_rad_s,
                                    acceleration_limit_rad_s2=args.acceleration_limit_rad_s2)
        fb = robot.send_command(builder, int(max(args.timeout_s, args.minimum_time_s + 5.0))).get()
        if fb.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            print(f"  [goto-start] command failed ({fb.finish_code}) -- aborting.")
            return False
    print("  [goto-start] step budget exhausted before reaching target.")
    return False


def reposition_to_start(robot, kin, left_idx, right_frozen, torso_frozen, gripper_enabled, last_grip_cmd, args):
    """Move the left arm to the nearest recorded demo start (or --start-pose-deg) and open the
    gripper -- used both before the rollout and (with --reset-on-release) after each pick, to
    retreat from the goal area so the next tomato is approached from a clean start."""
    if args.start_pose_deg is not None:
        target = np.deg2rad(np.asarray(args.start_pose_deg, dtype=float))
        print("[goto-start] target = --start-pose-deg override")
    else:
        q_now = np.asarray(robot.get_state().position, dtype=float)[left_idx]
        starts = load_demo_starts(args.raw_dir)
        if starts is not None:
            d = np.linalg.norm(starts - q_now, axis=1)
            j = int(np.argmin(d))
            target = starts[j]
            print(f"[goto-start] nearest demo start = episode_{j:03d} (~{np.degrees(d[j]):.0f} deg from current)")
        else:
            target = np.asarray(DEFAULT_START_POSE_RAD, dtype=float)
            print(f"[goto-start] raw demos not found at {args.raw_dir} -> using mean start pose")
    if goto_start_pose(robot, kin, left_idx, right_frozen, torso_frozen, target, args=args) and gripper_enabled:
        gc = float(np.clip(args.start_grip, args.grip_min_raw, args.grip_max_raw))
        try:
            command_gripper(args.gripper_url, LEFT, gc, current_limit=args.gripper_current_limit,
                            hold_s=args.gripper_hold_s, timeout_s=args.timeout_s)
            last_grip_cmd[0] = gc
            print(f"  [goto-start] gripper set to {gc:.2f} (open)")
        except Exception as exc:
            print(f"  [goto-start] gripper set failed: {exc}")


def run(args: argparse.Namespace) -> None:
    websocket_client_policy = _import_openpi_client(args.openpi_client_src)

    cam_serials = {"base": args.base_serial, "left_wrist": args.wrist_serial}
    print(f"[cameras] {cam_serials}")
    for d in list_devices():
        print(f"          attached: {d['serial']} {d['name']}")

    print(f"[policy] connecting to openpi server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        print(f"[policy] server metadata: {client.get_server_metadata()}")
    except Exception as exc:
        print(f"[policy] (no metadata: {exc})")

    kin = robot = left_idx = None
    right_frozen = torso_frozen = table_floor_z = None
    cmd_stream = None  # rby1 real-time command stream (--command-stream); created once, reused
    if not args.no_robot:
        print("[robot] connecting + building kinematics ...")
        kin = RBY1Kinematics.from_live_robot(
            address=args.address, model_name=args.model, arm=LEFT, toolfit_path=args.toolfit
        )
        if args.execute:
            robot = connect_motion_robot(address=args.address, model=args.model, power=args.power, servo=args.servo)
            print("[robot] connected (MOTION).")
        else:
            robot = connect_state_robot(address=args.address, model=args.model)
            print("[robot] connected (STATE-ONLY, dry-run).")
        left_idx = left_arm_indices(robot)
        right_idx = [int(i) for i in robot.model().right_arm_idx]
        torso_idx = [int(i) for i in robot.model().torso_idx]
        print(f"[robot] left_arm_idx={left_idx}  (driving LEFT; right arm + torso HELD at current pose, head untouched)")
        if args.execute:
            # Capture right-arm + torso ONCE and re-command them to this frozen pose every
            # step, so driving the left arm never sweeps them (the command_arm-to-home hazard).
            _full0 = np.asarray(robot.get_state().position, dtype=float)
            right_frozen = _full0[right_idx].copy()
            torso_frozen = _full0[torso_idx].copy()
            print(f"[robot] frozen-hold captured: right_arm(deg)={np.round(np.rad2deg(right_frozen), 1)}")
            if not args.no_table_floor:
                if args.table_floor_z is not None:
                    table_floor_z = float(args.table_floor_z)
                    print(f"[table-floor] using FIXED floor = {table_floor_z:.3f} m (--table-floor-z)")
                else:
                    table_floor_z = compute_table_floor_z(kin, robot, left_idx, args.raw_dir, args.table_floor_subsample)
                    if table_floor_z is not None:
                        table_floor_z += args.table_z_margin_m
                        print(f"[table-floor] gripper Z floor = {table_floor_z:.3f} m "
                              f"(lowest demo reach + {args.table_z_margin_m:.3f} m margin)")
                    else:
                        print(f"[table-floor] demos not found at {args.raw_dir} -> Cartesian table floor DISABLED")
            if args.command_stream:
                try:
                    robot.set_parameter("default.acceleration_limit_scaling", "0.8")
                    robot.set_parameter("joint_position_command.cutoff_frequency", str(args.command_cutoff_hz))
                except Exception as exc:  # noqa: BLE001
                    print(f"[command-stream] set_parameter warning: {exc}")
                cmd_stream = robot.create_command_stream(priority=1)
                print(f"[command-stream] ON: paced JointPosition stream (minimum_time={args.substep_time_s:.3f}s, "
                      f"hold={args.command_hold_time_s:.1f}s, cutoff={args.command_cutoff_hz:.1f}Hz) -- smooth/continuous")

    cams = MultiCamRealSense(cam_serials, width=args.cam_width, height=args.cam_height,
                             fps=args.cam_fps, warmup_frames=args.warmup_frames)

    mode = "EXECUTE" if args.execute else ("NO-ROBOT SMOKE" if args.no_robot else "DRY-RUN")
    print(f"\n===== pi0.5 LEFT-arm rollout  mode={mode}  task={args.task!r} =====\n")

    last_grip: list[float | None] = [None]   # last-good gripper raw (read fallback)
    last_grip_cmd: list[float | None] = [None]  # last commanded gripper raw (deadband)
    gripper_enabled = bool(args.gripper)
    grip_fail = [0]
    read_fail = [0]  # consecutive gripper /status read failures (servo dropped off the bus)
    if args.execute and gripper_enabled:
        _gtest = read_gripper_position_raw(args.gripper_url, LEFT, 3.0)
        if _gtest is None:
            print(f"\n  *** GRIPPER BRIDGE UNREACHABLE at {args.gripper_url} ***")
            print( "  *** The gripper will NOT actuate and NO grasp is possible. Start the teleop's :8765")
            print( "  *** HTTP server (or pass the correct --gripper-url). Reaching works; grasping won't. ***\n")
        else:
            print(f"[gripper] bridge OK at {args.gripper_url}  (left raw = {_gtest:.2f})")

    log_fh = None
    if not args.no_robot:
        _log_dir = REPO_ROOT / "robot" / "policy" / "openpi" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = _log_dir / time.strftime("pi05_run_%Y%m%d_%H%M%S.jsonl")
        log_fh = open(_log_path, "a", encoding="utf-8")
        print(f"[log] per-step JSONL -> {_log_path}")

    def _wlog(**rec) -> None:
        if log_fh is not None:
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()

    # Optional: reposition the left arm to an in-distribution demo-start pose before rolling out.
    if args.execute and args.goto_start:
        reposition_to_start(robot, kin, left_idx, right_frozen, torso_frozen, gripper_enabled, last_grip_cmd, args)

    stream_handles: list = []  # keep recent non-blocking command handles alive (--stream)
    grasp_count = [0]  # consecutive steps the policy has commanded the gripper closed (episodic reset)
    grasp_committed = [False]  # gripper hysteresis latch: True from grasp-commit until release
    settle_count = [0]            # consecutive settled-at-grasp-depth cycles (--force-grasp band-aid)
    prev_q_settle: list = [None]  # previous cycle-start arm pose, for settle detection
    force_hold_remaining = [0]    # cycles left to HOLD a forced grasp closed (carry time)
    release_settle_count = [0]    # consecutive settled-at-goal cycles after a forced grasp (force-release)
    prev_q_release: list = [None]
    force_open = [False]          # one-shot: force the gripper fully open this cycle (force-release)
    regrasp_cooldown = [0]        # cycles to suppress a new force-grasp right after a force-release
    dynamic_floor = [table_floor_z]  # adaptive table floor (--auto-floor lowers it, bounded, when stuck)
    stuck_count = [0]  # consecutive steps stuck at the floor with the gripper still open
    try:
        iter_source = itertools.count() if (args.loop or args.infinite) else range(args.iterations)
        for it in iter_source:
            frames = cams.read()
            base_rgb = frames["base"]["color_rgb"]
            wrist_rgb = frames["left_wrist"]["color_rgb"]

            if args.no_robot:
                state8 = np.zeros(ARM_DOF + 1, dtype=np.float32)
            else:
                _q, _g, state8 = read_state(robot, left_idx, gripper_url=args.gripper_url,
                                            read_gripper=args.read_gripper_state, last_grip=last_grip,
                                            grip_fallback=DEFAULT_OPEN_GRIP_RAW, read_fail=read_fail)

            # Gripper-dropout guard: repeated /status read failures = the servo fell off the bus
            # (motor_states empty). The policy is then BLIND to its gripper and cannot grasp, so HALT
            # loudly instead of silently feeding a frozen gripper into observation/state.
            if (args.execute and args.read_gripper_state and args.gripper_fail_halt
                    and read_fail[0] >= args.gripper_fail_halt):
                print("\n" + "=" * 72)
                print(f"  *** GRIPPER LOST: {read_fail[0]} consecutive /status read failures ***")
                print( "  The servo has dropped off the bus -- the policy is blind to its gripper and")
                print( "  cannot grasp. Power-cycle the gripper + restart the :8765 bridge, confirm")
                print(f"  `curl {args.gripper_url}/status` lists id 1, then press Enter to resume (Ctrl-C to stop).")
                print("=" * 72)
                try:
                    input("  [gripper-halt] press Enter to resume... ")
                except (EOFError, KeyboardInterrupt):
                    break
                read_fail[0] = 0
                continue

            obs = build_obs(base_rgb, wrist_rgb, state8, args.task)
            if args.save_obs and it == 0:
                from PIL import Image as _Image
                d = REPO_ROOT / "robot" / "policy" / "openpi" / "obs_dump"
                d.mkdir(parents=True, exist_ok=True)
                _Image.fromarray(obs["observation/image"]).save(str(d / "base.png"))
                _Image.fromarray(obs["observation/wrist_image"]).save(str(d / "left_wrist.png"))
                print(f"  [save-obs] wrote {d}\\(base|left_wrist).png  base={obs['observation/image'].shape}")

            t0 = time.perf_counter()
            result = client.infer(obs)
            infer_ms = (time.perf_counter() - t0) * 1e3
            actions = np.asarray(result["actions"], dtype=float)  # (H, 8) ABSOLUTE
            if actions.ndim != 2 or actions.shape[1] != ARM_DOF + 1:
                raise RuntimeError(f"unexpected action shape {actions.shape}; expected (H, 8)")
            print(f"--- iter {it}  infer {infer_ms:.0f} ms  chunk={actions.shape} ---")
            print(f"  action[0] joints(deg)={np.round(np.rad2deg(actions[0, :ARM_DOF]), 1)}  gripper_raw={actions[0, ARM_DOF]:+.2f}")

            if args.no_robot:
                if not (args.loop or args.infinite):
                    break
                continue

            # ---- RECEDING HORIZON: step through the first `--exec-stride` chunk poses ----
            # action[k] = state[t+1+k] is a smooth 20Hz/50ms trajectory. The OLD path executed only a
            # single far-lookahead pose (index 7), committing ~7x more blind open-loop motion than the
            # policy was trained for (p95 jump 0.36 vs 0.05 rad) and distorting the learned shape into
            # large straight chords. Here we execute CONSECUTIVE chunk poses, streamed (non-blocking)
            # so they blend into continuous motion, then re-infer -- in-distribution per-command motion
            # at the cadence the demos were recorded. Guards run per sub-step; closing a guard-blocked
            # cycle still lets the gripper follow below (decoupled).
            mgn = args.joint_limit_margin_rad
            em = np.deg2rad(args.envelope_margin_deg)
            stride = min(max(1, int(args.exec_stride)), actions.shape[0])
            full = np.asarray(robot.get_state().position, dtype=float)
            q_live = full[left_idx]            # cycle-start live pose (and the planning-chain base)
            q_base = q_live.copy()

            if args.execute and not args.no_confirm:
                if input(f"  EXECUTE iter {it}  stride={stride} (left arm)? [y/N] ").strip().lower() != "y":
                    print("  skipped by user.")
                    break

            sent = False
            steps_sent = 0
            cmd = q_base.copy()
            report = None
            verdict = "n/a"
            eef_z = None
            grip_target = float(actions[0, ARM_DOF])
            for k in range(stride):
                target7 = actions[k, :ARM_DOF]
                grip_target = float(actions[k, ARM_DOF])
                move = target7 - q_base
                cap = args.max_move_rad
                move_c = move if args.no_guards else np.clip(move, -cap, cap)
                cmd = q_base + move_c

                lo, hi = kin.active_joint_limits(full)
                cmd = np.clip(cmd, lo + mgn, hi - mgn)
                if not args.no_envelope:  # TABLE-SAFE demonstrated-workspace clamp
                    cmd = np.clip(cmd, DEMO_Q_MIN_RAD - em, DEMO_Q_MAX_RAD + em)

                # Guards (swept self-clearance + Cartesian table floor) -- SKIPPED under --no-guards so
                # the stream paces at the target rate (the per-step clearance FK is the expensive part).
                eef_z = None
                table_block = False
                # TABLE FLOOR = collision safety (one cheap FK) -- ALWAYS checked when enabled, even
                # under --no-guards, so the arm can never be driven down through the table plane.
                if table_floor_z is not None:
                    q_full_cmd = full.copy()
                    q_full_cmd[left_idx] = cmd
                    eef_z = float(kin.tcp_pose(q_full_cmd)[2, 3])
                    table_block = eef_z < dynamic_floor[0]
                if args.no_guards:
                    report = None
                    ok = not table_block  # only the table floor remains as a hard backstop
                else:
                    n_chk = max(1, int(args.clearance_path_samples))
                    reps = []
                    for f in np.linspace(1.0 / n_chk, 1.0, n_chk):
                        q_full_chk = full.copy()
                        q_full_chk[left_idx] = q_base + f * (cmd - q_base)
                        reps.append(nearest_self_clearances(kin, q_full_chk, pair_count=args.clearance_pair_count,
                                                            margin_m=args.min_clearance_m))
                    report = min(reps, key=lambda r: (r.minimum_distance_m
                                                      if r.minimum_distance_m is not None else float("inf")))
                    ok = all(r.ok for r in reps) and not table_block
                verdict = ("PASS (GUARDS OFF)" if args.no_guards else "PASS") if ok else (
                    "BLOCK[table_floor]" if table_block else "BLOCK[self_clearance]")
                mc = "n/a" if report is None else f"{report.minimum_distance_m}"
                zstr = f"{eef_z:.3f}" if eef_z is not None else "n/a"
                print(f"  step {k}: target(deg)={np.round(np.rad2deg(cmd), 1)}  "
                      f"min_clear={mc}  eef_z={zstr}  grip={grip_target:+.2f}  -> {verdict}")

                if not ok:
                    break  # blocked (floor/self-clearance): stop advancing this cycle (gripper still follows)
                if args.execute:
                    builder = command_body_left(cmd, right_frozen, torso_frozen,
                                                minimum_time_s=args.substep_time_s,
                                                velocity_limit_rad_s=args.velocity_limit_rad_s,
                                                acceleration_limit_rad_s2=args.acceleration_limit_rad_s2,
                                                control_hold_time_s=(args.command_hold_time_s if cmd_stream is not None else None))
                    if cmd_stream is not None:
                        # SMOOTH: paced real-time command stream -- the controller blends consecutive
                        # targets (minimum_time + cutoff_frequency). One stream reused across inferences.
                        # The stream EXPIRES when idle longer than control_hold_time (a --confirm pause
                        # or a stalled inference) -> recreate it and resend so the run survives pauses.
                        try:
                            cmd_stream.send_command(builder)
                        except RuntimeError as exc:
                            if "expired" in str(exc).lower():
                                print("  [command-stream] expired (idle/paused) -> recreating stream")
                                cmd_stream = robot.create_command_stream(priority=1)
                                cmd_stream.send_command(builder)
                            else:
                                raise
                        time.sleep(args.substep_time_s * 0.95)
                    else:
                        handle = robot.send_command(builder, int(max(args.timeout_s, args.substep_time_s + 5.0)))
                        stream_handles.append(handle)  # non-blocking BURST (stops per pose -> chunky)
                        del stream_handles[:-16]
                    sent = True
                    steps_sent += 1
                q_base = cmd  # advance the planning chain (previews the full path in dry-run, executes it live)

            # --auto-floor: stuck at the floor with the gripper still OPEN means it's reaching for the
            # tomato but the floor is a notch high -> lower it a little (bounded), until it can grasp.
            if args.auto_floor and table_floor_z is not None:
                if (verdict == "BLOCK[table_floor]") and grip_target < args.grasp_commit_raw:
                    stuck_count[0] += 1
                    floor_min = table_floor_z - args.floor_relax_max
                    if stuck_count[0] >= args.floor_relax_after and dynamic_floor[0] > floor_min:
                        dynamic_floor[0] = max(dynamic_floor[0] - args.floor_relax_step, floor_min)
                        stuck_count[0] = 0
                        print(f"[auto-floor] stuck -> floor lowered to {dynamic_floor[0]:.3f} m (limit {floor_min:.3f})")
                else:
                    stuck_count[0] = 0

            if regrasp_cooldown[0] > 0:
                regrasp_cooldown[0] -= 1
            # --force-grasp BAND-AID (demo/validation): the memorized policy reliably APPROACHES the
            # tomato but often won't commit the close. When the arm has SETTLED at grasp depth (near
            # the table floor, barely moving cycle-over-cycle) with the gripper still open, force the
            # close so the PLACE half can be validated. Place the tomato where the arm settles.
            if (args.force_grasp and args.execute and gripper_enabled and regrasp_cooldown[0] == 0
                    and not grasp_committed[0] and force_hold_remaining[0] == 0):
                near_floor = (eef_z is not None and dynamic_floor[0] is not None
                              and eef_z <= dynamic_floor[0] + args.force_grasp_floor_band)
                dq = (float(np.max(np.abs(q_live - prev_q_settle[0])))
                      if prev_q_settle[0] is not None else float("inf"))
                settled = (dq < np.deg2rad(args.force_grasp_settle_deg)
                           and (near_floor or dynamic_floor[0] is None)
                           and it >= args.force_grasp_warmup_iters)
                settle_count[0] = settle_count[0] + 1 if settled else 0
                if settle_count[0] >= args.force_grasp_settle_iters:
                    grasp_committed[0] = True
                    force_hold_remaining[0] = args.force_grasp_hold_iters
                    settle_count[0] = 0
                    release_settle_count[0] = 0
                    prev_q_release[0] = q_live.copy()
                    znow = f"{eef_z:.3f}" if eef_z is not None else "n/a"
                    print(f"[force-grasp] arm SETTLED at grasp depth (eef_z={znow}), gripper open -> "
                          f"FORCING close to {args.grasp_hold_raw:.2f}, holding {args.force_grasp_hold_iters} cycles")
            prev_q_settle[0] = q_live.copy()

            # --force-grasp companion (RELEASE): once the forced grasp has been CARRIED (force_hold
            # expired) and the arm SETTLES again at the destination, force the gripper fully OPEN so the
            # tomato is released -- otherwise the band-aid grasps + carries but never lets go.
            if (args.force_grasp and args.execute and gripper_enabled
                    and grasp_committed[0] and force_hold_remaining[0] == 0):
                dqr = (float(np.max(np.abs(q_live - prev_q_release[0])))
                       if prev_q_release[0] is not None else float("inf"))
                release_settle_count[0] = (release_settle_count[0] + 1
                                           if dqr < np.deg2rad(args.force_grasp_settle_deg) else 0)
                prev_q_release[0] = q_live.copy()
                if release_settle_count[0] >= args.force_grasp_settle_iters:
                    grasp_committed[0] = False
                    force_open[0] = True
                    release_settle_count[0] = 0
                    regrasp_cooldown[0] = args.force_release_cooldown_iters
                    print("[force-release] grasp carried + arm settled at goal -> RELEASING (full open)")
            elif not grasp_committed[0]:
                release_settle_count[0] = 0
                prev_q_release[0] = q_live.copy()

            # GRIPPER (decoupled from the arm guards + HYSTERESIS). Decoupled: closing/opening the
            # fingers cannot hit the table or self-collide, so a guard-BLOCKED arm step must not
            # suppress the grasp. Hysteresis: the demo 'closed' target is a smeared continuum up to
            # 4.76, and tracking the policy's drifting raw squishes the tomato -- instead, once the
            # policy crosses COMMIT, latch and command a FIXED firm-but-gentle HOLD value (ignore the
            # drift), and release (fully open) only once the policy drops below RELEASE. A forced
            # grasp (--force-grasp) is held closed for force_hold_remaining cycles before release.
            if args.execute and gripper_enabled:
                if force_open[0]:
                    grip_cmd_raw = DEFAULT_OPEN_GRIP_RAW    # force-release: fully open to drop the tomato
                    force_open[0] = False
                elif args.raw_gripper:
                    grip_cmd_raw = grip_target              # RAW: follow the model's gripper output directly
                elif force_hold_remaining[0] > 0:
                    grasp_committed[0] = True
                    grip_cmd_raw = args.grasp_hold_raw      # forced grasp: hold closed through the lift/carry
                    force_hold_remaining[0] -= 1
                elif grasp_committed[0]:
                    if grip_target < args.grasp_release_raw:
                        grasp_committed[0] = False
                        grip_cmd_raw = DEFAULT_OPEN_GRIP_RAW
                    else:
                        grip_cmd_raw = args.grasp_hold_raw  # HOLD fixed firm-gentle close (anti-squish)
                else:
                    if grip_target > args.grasp_commit_raw:
                        grasp_committed[0] = True
                        grip_cmd_raw = args.grasp_hold_raw
                    else:
                        grip_cmd_raw = grip_target          # follow the policy while open/approaching
                grip_c = float(np.clip(grip_cmd_raw, args.grip_min_raw, args.grip_max_raw))
                prev = last_grip_cmd[0]
                if prev is None or abs(grip_c - prev) > args.gripper_deadband_raw:
                    try:
                        command_gripper(args.gripper_url, LEFT, grip_c,
                                        current_limit=args.gripper_current_limit,
                                        hold_s=args.gripper_hold_s, timeout_s=args.timeout_s)
                        print(f"  [gripper] -> raw {grip_c:+.2f}  (committed={grasp_committed[0]}, policy={grip_target:+.2f})")
                        last_grip_cmd[0] = grip_c
                        grip_fail[0] = 0
                    except Exception as exc:
                        grip_fail[0] += 1
                        print(f"  [gripper] FAILED ({grip_fail[0]}/3): {exc}")
                        if grip_fail[0] >= 3:
                            gripper_enabled = False
                            print("  [gripper] 3 consecutive failures -> gripper DISABLED for this run.")

            _wlog(t=time.time(), iter=it, stride=stride, steps_sent=steps_sent, infer_ms=round(infer_ms, 1),
                  q_live=np.round(q_live, 5).tolist(), target=np.round(cmd, 5).tolist(),
                  grip_target=grip_target, grasp_committed=bool(grasp_committed[0]),
                  eef_z=(round(eef_z, 5) if eef_z is not None else None),
                  min_clearance_m=(float(report.minimum_distance_m or -1.0) if report is not None else -1.0),
                  verdict=verdict, sent=sent)

            # Episodic reset: grasp -> release == one pick-and-place done -> retreat to a clean demo
            # start so the next tomato is approached from the start, not from the goal where it lingers.
            if args.reset_on_release and args.execute:
                if grip_target > args.grasp_close_raw:
                    grasp_count[0] += 1
                elif grasp_count[0] >= args.grasp_min_steps and grip_target < args.grasp_open_raw:
                    print(f"[episode] grasp->release detected (closed {grasp_count[0]} steps) -> retreating to demo start")
                    grasp_count[0] = 0
                    reposition_to_start(robot, kin, left_idx, right_frozen, torso_frozen,
                                        gripper_enabled, last_grip_cmd, args)
                    if args.pause_after_reset:
                        try:
                            input("  [episode] place the next tomato, then press Enter (Ctrl-C to stop)...")
                        except (EOFError, KeyboardInterrupt):
                            break

            if not (args.loop or args.infinite):
                print(f"\n[done] {mode} complete.")
                break
    finally:
        cams.close()
        if log_fh is not None:
            log_fh.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pi0.5 RB-Y1-a LEFT-arm rollout client (dry-run by default).")
    p.add_argument("--host", default="127.0.0.1", help="openpi policy server host (WSL2 IP).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--openpi-client-src", default=DEFAULT_OPENPI_CLIENT_SRC)
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Language instruction. MUST match the dataset's task string (prompt_from_task).")
    # cameras (match the recorder)
    p.add_argument("--base-serial", default=DEFAULT_BASE_SERIAL)
    p.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=15)
    p.add_argument("--warmup-frames", type=int, default=30)
    # robot
    p.add_argument("--address", default="192.0.2.10:50051")
    p.add_argument("--model", default="a")
    p.add_argument("--power", default=".*")
    p.add_argument("--servo", default=".*")
    p.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    # control + guards
    p.add_argument("--lookahead-steps", type=int, default=8,
                   help="DEPRECATED / unused: superseded by --exec-stride receding-horizon consumption.")
    p.add_argument("--exec-stride", type=int, default=8,
                   help="RECEDING HORIZON: execute the first N consecutive chunk poses (of H=10) per "
                        "inference, streamed so they blend, then re-infer. Fixes the single-far-lookahead "
                        "OOD (~7x too much blind motion). Keep high (6-10) for continuity, lower for reactivity.")
    p.add_argument("--substep-time-s", type=float, default=0.08,
                   help="Per-sub-step motion duration (s) inside the stride loop. Demo cadence is 20Hz "
                        "(0.05s); ~0.05-0.10 keeps execution near training rate. Lower = faster.")
    p.add_argument("--max-move-rad", type=float, default=0.7,
                   help="Per-sub-step move cap: |target - chain| clamped to this (rad). 0.7 ~= demo q99; "
                        "with small receding-horizon steps it rarely fires (it self-resolves at stride).")
    p.add_argument("--max-step-delta-rad", type=float, default=0.10,
                   help="Per-step clamp used by --goto-start's incremental repositioning.")
    p.add_argument("--joint-limit-margin-rad", type=float, default=0.05)
    p.add_argument("--min-clearance-m", type=float, default=0.03)
    p.add_argument("--clearance-pair-count", type=int, default=10)
    p.add_argument("--clearance-path-samples", type=int, default=5,
                   help="Check self-clearance at this many interpolated points along q_live->target "
                        "(not just the endpoint), so a mid-sweep collision is caught.")
    p.add_argument("--envelope-margin-deg", type=float, default=0.0,
                   help="Expand the demonstrated-workspace clamp (the TABLE-SAFE floor) by this many deg. "
                        "0 = clamp exactly to what the demos did (safest vs the table).")
    p.add_argument("--no-envelope", action="store_true",
                   help="DISABLE the demonstrated-workspace clamp (joint-space). The Cartesian table floor "
                        "below is the precise table guard.")
    p.add_argument("--table-z-margin-m", type=float, default=0.0,
                   help="Stop the gripper this far ABOVE the lowest demonstrated reach (m). 0 = allow the full "
                        "demonstrated grasp depth; raise (e.g. 0.01) for extra table clearance.")
    p.add_argument("--no-table-floor", action="store_true",
                   help="Disable the Cartesian gripper-height table floor (NOT recommended).")
    p.add_argument("--table-floor-z", type=float, default=None,
                   help="Set the gripper-height table floor (m, base frame) DIRECTLY instead of deriving it "
                        "from the demos. Read a good value from the eef_z printed at a successful grasp, then "
                        "set this ~1 cm below it. Reliable + no drift (unlike --auto-floor).")
    p.add_argument("--table-floor-subsample", type=int, default=1,
                   help="Use every Nth demo frame when computing the table-floor height. 1 = the TRUE deepest "
                        "reach (subsampling skips the deepest grasp frame -> floor too high -> blocks grasps).")
    p.add_argument("--auto-floor", action="store_true",
                   help="Auto-lower the table floor (bounded) while the arm is stuck at it with the gripper "
                        "still open, until it reaches + grasps -- no manual --table-z-margin-m tuning.")
    p.add_argument("--floor-relax-step", type=float, default=0.005, help="Lower the floor this much per relax (m).")
    p.add_argument("--floor-relax-after", type=int, default=4, help="Steps stuck+open before lowering the floor.")
    p.add_argument("--floor-relax-max", type=float, default=0.02,
                   help="Max the floor may drop below the demonstrated floor (m) -- the table-safety bound.")
    p.add_argument("--minimum-time-s", type=float, default=0.6,
                   help="Motion duration (s) for --goto-start / episodic-reset repositioning ONLY (the "
                        "rollout itself uses --substep-time-s). Kept slow/safe for repositioning moves.")
    p.add_argument("--velocity-limit-rad-s", type=float, default=1.0,
                   help="Per-joint velocity cap (rad/s). Demo moving-frame p95 ~1.37 rad/s; 1.0 covers most "
                        "of the approach. Raise toward ~1.4 to fully rate-match, lower for safer.")
    p.add_argument("--acceleration-limit-rad-s2", type=float, default=2.5)
    p.add_argument("--command-stream", action="store_true",
                   help="SMOOTH execution: stream each chunk pose through robot.create_command_stream() at the "
                        "substep cadence (controller blends via minimum_time + cutoff) instead of a burst of "
                        "point-to-point send_command()s that stop at each pose -> choppy. Mirrors the SDK replay.py.")
    p.add_argument("--command-hold-time-s", type=float, default=1.0,
                   help="control_hold_time on each streamed command: the controller keeps tracking the last "
                        "target this long if the next inference is late (bridges the ~0.2s inference gap).")
    p.add_argument("--command-cutoff-hz", type=float, default=5.0,
                   help="joint_position_command.cutoff_frequency (Hz) low-pass smoothing applied to the stream.")
    p.add_argument("--timeout-s", type=float, default=12.0)
    # gripper
    p.add_argument("--gripper", action="store_true", help="Actuate the left gripper from the policy.")
    p.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    p.add_argument("--read-gripper-state", action="store_true", default=True,
                   help="Feed live gripper raw into observation/state (default ON -- matches the recorder).")
    p.add_argument("--no-read-gripper-state", dest="read_gripper_state", action="store_false")
    p.add_argument("--grip-min-raw", type=float, default=DEFAULT_GRIP_MIN_RAW)
    p.add_argument("--grip-max-raw", type=float, default=DEFAULT_GRIP_MAX_RAW)
    p.add_argument("--gripper-deadband-raw", type=float, default=0.15,
                   help="Only re-command the gripper when the raw target moves more than this.")
    p.add_argument("--gripper-current-limit", type=float, default=2.0)
    p.add_argument("--gripper-hold-s", type=float, default=0.3,
                   help="Bridge hold time per /jog (s). The servo keeps torque-holding after, so keep short "
                        "so a gripper command does not stall the streamed arm cadence.")
    p.add_argument("--gripper-fail-halt", type=int, default=5,
                   help="Halt + wait LOUDLY after this many CONSECUTIVE gripper /status read failures (the "
                        "servo dropped off the bus). 0 = disable (silently fall back to last-good/open).")
    # demo-start repositioning (the demos started from a RANGE of poses, std ~25 deg)
    p.add_argument("--goto-start", action="store_true",
                   help="Before the rollout, slowly move the left arm to the NEAREST recorded demo "
                        "start (clamped, clearance-checked) so it begins in-distribution. Needs --execute.")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Recorded demo dataset (episode_*/frames.npz) used to pick the nearest start.")
    p.add_argument("--start-pose-deg", type=float, nargs=7, default=None,
                   help="Override --goto-start with explicit 7 joint angles (deg).")
    p.add_argument("--start-grip", type=float, default=DEFAULT_START_GRIP,
                   help="Gripper raw to set after --goto-start (demos started open, ~2.16).")
    # episodic reset (retreat to a demo start after each completed pick-and-place)
    p.add_argument("--reset-on-release", action="store_true",
                   help="After the policy grasps then releases (one pick-place done), retreat the arm to a "
                        "demo start so the next tomato is approached from a clean start, not the goal area.")
    p.add_argument("--pause-after-reset", action="store_true",
                   help="After retreating, wait for Enter (place the next tomato) before the next episode.")
    p.add_argument("--grasp-close-raw", type=float, default=2.5,
                   help="Gripper raw above this counts as 'grasped' (--reset-on-release detection).")
    p.add_argument("--grasp-open-raw", type=float, default=2.0,
                   help="Gripper raw below this after a grasp counts as 'released' / task done.")
    p.add_argument("--grasp-min-steps", type=int, default=3,
                   help="Require the gripper closed this many steps before a release triggers a reset (debounce).")
    # gripper close hysteresis (anti-squish: hold a fixed firm-gentle close, not the policy's drifting raw)
    p.add_argument("--grasp-commit-raw", type=float, default=DEFAULT_GRASP_COMMIT_RAW,
                   help="HYSTERESIS commit: once the policy's raw exceeds this, latch a grasp and command a "
                        "fixed HOLD value (demo open ~1.78, closed cluster starts ~2.8).")
    p.add_argument("--grasp-hold-raw", type=float, default=DEFAULT_GRASP_HOLD_RAW,
                   help="Fixed firm-but-gentle close raw commanded while a grasp is latched (demo median "
                        "close ~3.56); NOT the policy's drifting raw (reaches 4.76 -> squish).")
    p.add_argument("--grasp-release-raw", type=float, default=DEFAULT_GRASP_RELEASE_RAW,
                   help="HYSTERESIS release: once latched, the policy raw must drop below this to release "
                        "(fully open). Keep below --grasp-commit-raw for clean open/close hysteresis.")
    # --force-grasp band-aid: force the close when the arm settles at grasp depth (memorized policy
    # approaches reliably but won't always commit). Lets the policy's PLACE half be validated.
    p.add_argument("--force-grasp", action="store_true",
                   help="DEMO/BAND-AID: when the arm SETTLES at grasp depth with the gripper still open, "
                        "force the close so the place half can run. Place the tomato where the arm settles.")
    p.add_argument("--force-grasp-settle-iters", type=int, default=6,
                   help="Consecutive settled cycles required before forcing the close.")
    p.add_argument("--force-grasp-settle-deg", type=float, default=1.5,
                   help="Max per-cycle arm motion (deg, max joint) to count a cycle as 'settled'.")
    p.add_argument("--force-grasp-floor-band", type=float, default=0.03,
                   help="eef_z must be within this of the table floor (m) to count as 'at grasp depth'.")
    p.add_argument("--force-grasp-warmup-iters", type=int, default=12,
                   help="Do not force a grasp before this many iterations (let it approach first).")
    p.add_argument("--force-grasp-hold-iters", type=int, default=15,
                   help="After forcing, hold the grasp closed this many cycles (carry time) before the "
                        "force-release / normal hysteresis can open it.")
    p.add_argument("--force-release-cooldown-iters", type=int, default=25,
                   help="After a force-release (open at the goal), suppress a new force-grasp for this many "
                        "cycles so it doesn't immediately re-grab at the goal before moving to the next pick.")
    # loop / modes
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--infinite", action="store_true", help="Run forever (implies loop); Ctrl+C to stop.")
    p.add_argument("--stream", action="store_true",
                   help="Smooth mode: send each command non-blocking so the next inference overlaps the "
                        "motion and commands blend into continuous (un-chunky) motion. Default is blocking "
                        "(move, brief pause, repeat). Experimental -- watch the arm; no finish-code check.")
    p.add_argument("--execute", action="store_true", help="MASTER MOTION GATE. Off = dry-run (print only).")
    p.add_argument("--no-confirm", action="store_true")
    p.add_argument("--no-robot", action="store_true", help="Smoke: cameras + infer + print action only (no robot).")
    p.add_argument("--save-obs", action="store_true", help="Save iter-0 obs images to obs_dump/ for inspection.")
    p.add_argument("--no-guards", action="store_true",
                   help="Disable the per-step move clamp + self-clearance block (raw policy targets). The "
                        "per-step confirm prompt stays ON unless you ALSO pass --no-confirm. Joint-limit clamp "
                        "+ slow minimum_time/velocity caps are KEPT.")
    p.add_argument("--raw-gripper", action="store_true",
                   help="Command the model's RAW gripper output directly (clipped to the safe band), "
                        "bypassing the commit/hold hysteresis.")
    p.add_argument("--raw-policy", action="store_true",
                   help="DEPLOY THE BARE MODEL: strip ALL action-filtering gates -- self-clearance, the "
                        "demonstrated-workspace envelope, the per-step move clamp, the Cartesian table floor, "
                        "and the gripper hysteresis/force-grasp. KEEPS only joint-limit + velocity/accel caps "
                        "(actuator safety). Executes the model's action chunk as-is. WATCH THE ARM / e-stop ready.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.base_serial or not args.wrist_serial:
        parser.error("Set --base-serial and --wrist-serial, or BASE_CAMERA_SERIAL and LEFT_WRIST_CAMERA_SERIAL.")
    if args.raw_policy:
        args.no_guards = True
        args.no_envelope = True
        args.raw_gripper = True
        args.force_grasp = False
        print("[raw-policy] policy gates OFF (self-clearance, envelope, gripper hysteresis, force-grasp). "
              "KEEPING the Cartesian table floor as a COLLISION backstop + joint-limit/velocity caps. "
              "(Pass --no-table-floor to remove the floor too -- but then the arm WILL drop into the table.)")
    if args.infinite:
        args.loop = True
    if args.gripper and not args.execute:
        print("[warn] --gripper has no effect without --execute (dry-run prints targets only).")
    if args.no_guards:
        print("[warn] --no-guards: per-step clamp + self-clearance OFF (raw policy targets). "
              "Joint-limit clamp + slow minimum_time/velocity caps are KEPT.")
    run(args)


if __name__ == "__main__":
    main()
