"""MolmoBot RB-Y1 manipulation rollout client (dual-arm, joint-space), dry-run by default.

MolmoBot ships NATIVE RB-Y1 checkpoints (allenai/MolmoBot-RBY1Multitask, pick_pnp /
door_plus_open) trained in MolmoSpaces sim. Unlike the openpi/Franka path, the action
is **RB-Y1 joint space** -- per-arm joint-position DELTAS + absolute gripper + absolute
torso -- so there is NO Franka retarget and NO IK: we command the joints directly.

Per-tick pipeline (mirrors the MolmoBot server's expected obs/action):
    capture head + both wrist RGB  ->  build obs dict {head_camera, wrist_camera_l,
      wrist_camera_r, qpos{base,left_arm,left_gripper,right_arm,right_gripper,torso},
      task, reset?}  ->  client.infer(obs)  (websocket, openpi-client transport)
      ->  action dict {base(3 delta), left_arm(7 delta), right_arm(7 delta),
                       left_gripper(1 abs ±100), right_gripper(1 abs ±100), torso(1 abs)}
      ->  clamp per-joint delta -> q_target = q_now + delta -> joint-limit + self-clearance
      ->  command both arms (and optionally torso) -> gripper via HTTP bridge.

Server (run separately, WSL2): see molmobot_deploy_README.md ->
    python launch_scripts/serve_molmobot_rby1_multitask.py --task_type pick_pnp \
        --hf_repo allenai/MolmoBot-RBY1Multitask --port 8000

SAFETY: dry-run by default (prints proposed joint targets + guard verdicts, no motion).
--execute is the master motion gate. The base action is IGNORED (we never drive the
mobile base). The torso is pinned to home unless --drive-torso. Per project memory:
slow minimum_time, low velocity/accel caps, per-waypoint --confirm, hard per-step delta
clamp. The #1 correctness risk is JOINT ORDERING: validate joint-by-joint in dry-run
(send a tiny delta on one joint, confirm the right joint moves) BEFORE trusting --execute.
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
from robot.kinematics.utils.realsense_click_move_arm import (
    DEFAULT_HOME,
    command_positions,
    connect_motion_robot,
)
from robot.policy.openpi.multicam_capture import MultiCamRealSense, list_devices
from robot.policy.openpi.multicam_view import CameraStreamer
from robot.policy.openpi.openpi_rollout import (
    DEFAULT_GRIPPER_URL,
    DEFAULT_LEFT_CLOSE_POSITION,
    DEFAULT_RIGHT_CLOSE_POSITION,
    GRIPPER_OPEN_POSITION,
    command_gripper,
    connect_state_robot,
    read_gripper_position_raw,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "config" / "realsense_cameras.json"
# Toolfit is only needed to build RBY1Kinematics for FK/self-clearance (uses the full
# model, not the TCP); any valid ee_right toolfit works.
DEFAULT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_right_gripper_jaw_center_01.json"
DEFAULT_OPENPI_CLIENT_SRC = "C:/path/to/projects/openpi/packages/openpi-client/src"

# MolmoBot RBY1 multitask obs/action move-group dims.
ACTION_SPEC = {"base": 3, "left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1, "torso": 1}
ARM_DOF = 7
# MolmoBot gripper STATE (proprio) range from the checkpoint config (synthmanip
# observation.state dims 10/18). Our gripper encoder must be mapped INTO this range, NOT
# fed as [0,1] -- [0,1] is ~15x out of the trained range and corrupts the policy state
# (verified: that was why the policy emitted ~0 actions).
GRIPPER_STATE_RANGE = {"left": (-0.113, 0.0215), "right": (-0.120, 0.045)}


class MolmoWsClient:
    """Minimal websocket client matching MolmoBot's server transport.

    MolmoBot's server uses the pip ``msgpack_numpy`` (ndarrays tagged ``b'nd'``), while
    openpi-client's vendored ``msgpack_numpy`` tags them ``b'__ndarray__'`` -- incompatible,
    so images would reach the server as undecoded dicts. We therefore speak the server's
    own encoding directly. Protocol: on connect the server sends packed metadata; per
    request we send the packed obs and receive the packed action dict (or a string
    traceback on server error).
    """

    def __init__(self, host: str, port: int) -> None:
        import msgpack_numpy as mn
        import websockets.sync.client as wsc

        self._mn = mn
        self._ws = wsc.connect(f"ws://{host}:{port}", compression=None, max_size=None)
        self._metadata = mn.unpackb(self._ws.recv(), raw=False)

    def get_server_metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._mn.packb(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Error in inference server:\n{resp}")
        return self._mn.unpackb(resp, raw=False)


def load_camera_serials(path: Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def joint_index_groups(model: Any) -> dict[str, list[int]]:
    return {
        "left_arm": [int(i) for i in model.left_arm_idx],
        "right_arm": [int(i) for i in model.right_arm_idx],
        "torso": [int(i) for i in model.torso_idx],
    }


def _gripper_proprio(raw: float | None, close_pos: float, model_range: tuple[float, float]) -> float:
    """Map a raw gripper encoder into the model's gripper-state range (in-distribution).

    encoder -> normalized [0,1] (0=open, 1=closed) -> lerp into [min,max] from the checkpoint.
    Returns the range midpoint (still in-range) if the bridge read fails.
    """
    lo, hi = model_range
    if raw is None:
        return 0.5 * (lo + hi)
    span = close_pos - GRIPPER_OPEN_POSITION
    norm = float(np.clip((raw - GRIPPER_OPEN_POSITION) / span, 0.0, 1.0)) if abs(span) > 1e-9 else 0.0
    return lo + norm * (hi - lo)


def read_qpos(robot: Any, idx: dict[str, list[int]], *, gripper_url: str, read_gripper: bool,
              raw_cache: dict[str, float] | None = None) -> tuple[dict, np.ndarray]:
    """Read the full-body qpos dict MolmoBot expects, plus the raw full joint vector.

    A failed gripper-bridge read falls back to the LAST GOOD value (``raw_cache``) with a
    loud message -- the old silent midpoint substitution made the policy observe a
    half-open gripper mid-grasp with zero diagnostics (review finding).
    """
    full = np.asarray(robot.get_state().position, dtype=float)
    qpos: dict[str, np.ndarray] = {
        "base": np.zeros(3, dtype=np.float32),  # stationary task: base proprio = 0, base action ignored
        "left_arm": full[idx["left_arm"]].astype(np.float32),
        "right_arm": full[idx["right_arm"]].astype(np.float32),
        "torso": full[idx["torso"]].astype(np.float32),  # 6-D raw; server extracts [1,2,3]
    }
    grips = {"left": 0.0, "right": 0.0}  # 0.0 is within both trained ranges (safe default)
    if read_gripper:
        for side, close_pos in (("left", DEFAULT_LEFT_CLOSE_POSITION), ("right", DEFAULT_RIGHT_CLOSE_POSITION)):
            raw = read_gripper_position_raw(gripper_url, side, 3.0)
            if raw is None:
                cached = (raw_cache or {}).get(side)
                print(f"  [gripper] {side} proprio read FAILED -> "
                      + (f"using last-good {cached:.3f}" if cached is not None else "midpoint fallback (no last-good yet)"))
                raw = cached
            elif raw_cache is not None:
                raw_cache[side] = float(raw)
            grips[side] = _gripper_proprio(raw, close_pos, GRIPPER_STATE_RANGE[side])
    qpos["left_gripper"] = np.array([grips["left"]], dtype=np.float32)
    qpos["right_gripper"] = np.array([grips["right"]], dtype=np.float32)
    return qpos, full


def build_obs(frames: dict, qpos: dict, task: str, reset: bool) -> dict:
    """MolmoBot RBY1 obs dict. Images are native RGB uint8; the server warps the head cam itself."""
    obs = {
        "head_camera": frames["head"]["color_rgb"],
        "wrist_camera_l": frames["wrist_l"]["color_rgb"],
        "wrist_camera_r": frames["wrist_r"]["color_rgb"],
        "qpos": qpos,
        "task": task,
    }
    if reset:
        obs["reset"] = True
    return obs


def _joint_cmd(position: np.ndarray, minimum_time_s: float, vel: float | None, acc: float | None):
    pos = np.asarray(position, dtype=float).reshape(-1)
    builder = rby.JointPositionCommandBuilder().set_minimum_time(float(minimum_time_s)).set_position(pos)
    if vel is not None:
        builder = builder.set_velocity_limit(np.full(pos.shape[0], float(vel), dtype=float))
    if acc is not None:
        builder = builder.set_acceleration_limit(np.full(pos.shape[0], float(acc), dtype=float))
    return builder


def command_body(home: dict, *, left_q=None, right_q=None, torso_q=None,
                 minimum_time_s: float, velocity_limit_rad_s=None, acceleration_limit_rad_s2=None):
    """Whole-body command for BOTH arms (+ optional torso). Torso pins to home; the HEAD
    pins to the MolmoBot READY tilt (0.6 rad down -- the training viewpoint), NOT home.
    NEVER route single-arm execution through realsense_click_move_arm.command_arm here:
    it commands the OTHER arm (and head) to HOME, which from READY is a large unwanted
    sweep (verified on hardware: driving left visibly moved the right arm home)."""
    from robot.policy.molmobot.molmobot_ready_pose import READY_ARM_RAD
    lq = left_q if left_q is not None else READY_ARM_RAD
    rq = right_q if right_q is not None else READY_ARM_RAD
    tq = torso_q if torso_q is not None else command_positions(home, "torso")
    body = (
        rby.BodyComponentBasedCommandBuilder()
        .set_torso_command(_joint_cmd(tq, minimum_time_s, velocity_limit_rad_s, acceleration_limit_rad_s2))
        .set_right_arm_command(_joint_cmd(rq, minimum_time_s, velocity_limit_rad_s, acceleration_limit_rad_s2))
        .set_left_arm_command(_joint_cmd(lq, minimum_time_s, velocity_limit_rad_s, acceleration_limit_rad_s2))
    )
    # HEAD is deliberately NOT commanded: molmobot_ready_pose sets it once (pan 0,
    # tilt 0.6 down -- the training viewpoint) and the rollout must leave it static.
    # Per-step head pinning made the neck crawl whenever its start pose was stale.
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(body)
    )


def joint_limits_for(kin: RBY1Kinematics, q_full: np.ndarray, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    state = kin.make_state(np.asarray(q_full, dtype=float))
    lower = np.asarray(kin.dyn_robot.get_limit_q_lower(state), dtype=float).reshape(-1)
    upper = np.asarray(kin.dyn_robot.get_limit_q_upper(state), dtype=float).reshape(-1)
    return lower[indices], upper[indices]


def run(args: argparse.Namespace) -> None:
    serials = load_camera_serials(args.camera_config)
    cam_serials = {
        "head": args.head_serial or serials["base"],
        "wrist_l": args.left_wrist_serial or serials["left_wrist"],
        "wrist_r": args.right_wrist_serial or serials["right_wrist"],
    }
    print(f"[cameras] {cam_serials}")
    for d in list_devices():
        print(f"          attached: {d['serial']} {d['name']}")

    print(f"[policy] connecting to MolmoBot server at {args.host}:{args.port} ...")
    client = MolmoWsClient(args.host, args.port)
    try:
        print(f"[policy] server metadata: {client.get_server_metadata()}")
    except Exception as exc:
        print(f"[policy] (no metadata: {exc})")

    kin = robot = home = idx = None
    if not args.no_robot:
        print("[robot] connecting + building kinematics ...")
        kin = RBY1Kinematics.from_live_robot(
            address=args.address, model_name=args.model, arm="right", toolfit_path=args.toolfit
        )
        home = json.loads(Path(args.home).read_text(encoding="utf-8"))
        if args.execute:
            robot = connect_motion_robot(address=args.address, model=args.model, power=args.power, servo=args.servo)
            # MolmoBot's precondition is the SIM EPISODE-START pose (molmo_spaces
            # RBY1Config.init_qpos), NOT our tucked home -- the policy emits ~zero
            # actions from home (verified in dry-run; wrist pitch alone is ~99 deg off
            # the training start distribution). Validate arms against READY instead.
            from robot.policy.molmobot.molmobot_ready_pose import READY_ARM_RAD, load_ready_state
            _saved = load_ready_state()  # user-saved canonical READY (ready_state.json), if any
            _live = np.asarray(robot.get_state().position, dtype=float)
            _m = robot.model()
            for _arm_name, _idx in (("left", _m.left_arm_idx), ("right", _m.right_arm_idx)):
                _ready = _saved[f"{_arm_name}_arm"] if _saved else READY_ARM_RAD
                _d = np.abs(_live[[int(i) for i in _idx]] - _ready)
                _j = int(np.argmax(_d))
                if float(_d[_j]) > np.deg2rad(args.home_tolerance_deg):
                    raise RuntimeError(
                        f"robot is not at the MolmoBot READY pose: {_arm_name}_arm_{_j} off by "
                        f"{np.degrees(float(_d[_j])):.2f} deg (tolerance {args.home_tolerance_deg} deg). "
                        "Run: python -m robot.policy.molmobot.molmobot_ready_pose --execute"
                    )
            # TORSO matters as much as the arms (policy trained standing; a crouched torso
            # shifts the whole reach frame) -- validate it against the saved READY when one
            # exists; otherwise warn loudly that posture is unvalidated.
            if _saved is not None and "torso" in _saved:
                _dt = np.abs(_live[[int(i) for i in _m.torso_idx]] - np.asarray(_saved["torso"], dtype=float))
                _j = int(np.argmax(_dt))
                if float(_dt[_j]) > np.deg2rad(args.home_tolerance_deg):
                    raise RuntimeError(
                        f"robot is not at the MolmoBot READY pose: torso_{_j} off by "
                        f"{np.degrees(float(_dt[_j])):.2f} deg vs saved ready_state.json. "
                        "Run: python -m robot.policy.molmobot.molmobot_ready_pose --execute"
                    )
            else:
                print("[robot] WARNING: no saved ready_state.json -> torso/head NOT validated. "
                      "The policy was trained STANDING UPRIGHT; a crouched torso is silently OOD.")
            print("[robot] connected (MOTION), validated at MolmoBot READY pose.")
        else:
            robot = connect_state_robot(address=args.address, model=args.model)
            print("[robot] connected (STATE-ONLY, dry-run).")
        idx = joint_index_groups(robot.model())
        print(f"[robot] joint index groups: left_arm={idx['left_arm']} right_arm={idx['right_arm']} torso={idx['torso']}")

    cams = MultiCamRealSense(cam_serials, width=args.cam_width, height=args.cam_height,
                             fps=args.cam_fps, warmup_frames=args.warmup_frames)
    streamer = CameraStreamer(cams, window="MolmoBot cameras (head|wrist_l|wrist_r)") if args.show_cameras else None

    drive_left = args.arms in ("left", "both")
    drive_right = args.arms in ("right", "both")
    mode = "EXECUTE" if args.execute else ("NO-ROBOT SMOKE" if args.no_robot else "DRY-RUN")
    print(f"\n===== MolmoBot rollout  mode={mode}  arms={args.arms}  task={args.task!r} =====\n")

    grip_state = {"left": None, "right": None}
    grip_pending = {"left": ("", 0), "right": ("", 0)}  # gripper debounce (desired, count)
    grip_fail = {"left": 0, "right": 0}  # consecutive HTTP failures -> 3-strike disable
    gripper_enabled = bool(args.gripper)
    grip_raw_cache: dict[str, float] = {}  # last-good encoder reads (bridge-glitch fallback)
    pending_steps = 0  # accumulated-but-unsent policy steps (counter-based batch flush)

    # per-step JSONL log: enough to reconstruct a failed grasp post-hoc (what the policy
    # commanded vs where the robot was, clearance verdicts, gripper desires/state).
    log_fh = None
    if not args.no_robot:
        _log_dir = REPO_ROOT / "robot" / "policy" / "molmobot" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = _log_dir / time.strftime("run_%Y%m%d_%H%M%S.jsonl")
        log_fh = open(_log_path, "a", encoding="utf-8")
        print(f"[log] per-step JSONL -> {_log_path}")

    def _wlog(**rec) -> None:
        if log_fh is not None:
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
    cmd_left = None  # accumulated commanded joint reference (seeded from first live read)
    cmd_right = None
    try:
        iter_source = itertools.count() if args.infinite else range(args.iterations)
        for it in iter_source:
            frames = streamer.latest() if streamer is not None else cams.read()

            if args.no_robot:
                qpos = {k: np.zeros(v, np.float32) for k, v in
                        {"base": 3, "left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1, "torso": 6}.items()}
                full = None
            else:
                qpos, full = read_qpos(robot, idx, gripper_url=args.gripper_url,
                                       read_gripper=args.read_gripper_state, raw_cache=grip_raw_cache)

            obs = build_obs(frames, qpos, args.task, reset=(it == 0))
            if args.save_obs and it == 0:
                from PIL import Image as _Image

                d = REPO_ROOT / "robot" / "policy" / "molmobot" / "obs_dump"
                d.mkdir(parents=True, exist_ok=True)
                for _k in ("head_camera", "wrist_camera_l", "wrist_camera_r"):
                    _Image.fromarray(obs[_k]).save(str(d / f"{_k}.png"))
                print(f"  [save-obs] wrote {d}\\(head_camera|wrist_camera_l|wrist_camera_r).png "
                      f"head_shape={obs['head_camera'].shape}  (NOTE: server fisheye-warps head_camera)")
            t0 = time.perf_counter()
            action = client.infer(obs)
            infer_ms = (time.perf_counter() - t0) * 1e3
            la = np.asarray(action["left_arm"], float).reshape(-1)
            ra = np.asarray(action["right_arm"], float).reshape(-1)
            lg = float(np.asarray(action["left_gripper"]).reshape(-1)[0])
            rg = float(np.asarray(action["right_gripper"]).reshape(-1)[0])
            tor = float(np.asarray(action["torso"]).reshape(-1)[0]) if "torso" in action else float("nan")
            print(f"--- iter {it}  infer {infer_ms:.0f} ms ---")
            print(f"  action.left_arm  Δ(deg)={np.round(np.rad2deg(la),1)}  grip={'CLOSE' if lg>0 else 'OPEN'}({lg:+.0f})")
            print(f"  action.right_arm Δ(deg)={np.round(np.rad2deg(ra),1)}  grip={'CLOSE' if rg>0 else 'OPEN'}({rg:+.0f})  torso(abs)={tor:.3f}")

            if args.no_robot:
                if not args.loop and not args.infinite:
                    break
                continue

            # --- clamp per-joint deltas (primary safety; --no-guards passes raw deltas) ---
            cap = args.max_step_delta_rad
            la_c = la if args.no_guards else np.clip(la, -cap, cap)
            ra_c = ra if args.no_guards else np.clip(ra, -cap, cap)
            # deadband: sub-threshold deltas -> 0, so noise/bias doesn't integrate into drift.
            db = args.delta_deadband_rad
            la_c = np.where(np.abs(la_c) < db, 0.0, la_c)
            ra_c = np.where(np.abs(ra_c) < db, 0.0, ra_c)
            q_left_now = full[idx["left_arm"]]
            q_right_now = full[idx["right_arm"]]
            if cmd_left is None:
                cmd_left = q_left_now.copy()
                cmd_right = q_right_now.copy()
                # hold the torso WHERE THE SESSION STARTED (e.g. upright sim-init after
                # ready_pose --torso ready) -- pinning to homestate's crouch would undo it
                hold_torso = full[idx["torso"]].copy()
            # snapshot the commanded reference: a clearance BLOCK must truly DISCARD this
            # step's deltas (without the rollback, "skipped" deltas stayed baked into the
            # next send, and sustained blocks wound the reference toward the colliding pose)
            prev_cmd_left = cmd_left.copy()
            prev_cmd_right = cmd_right.copy()
            if args.target_from_live:
                # relative to the LIVE pose -> ratchets in gravity sag (the down-drift bug). Opt-in only.
                left_target = q_left_now + la_c
                right_target = q_right_now + ra_c
            else:
                # accumulate a commanded reference (matches sim's relative-to-commanded semantics;
                # holds against gravity instead of re-baselining to the sagged pose each tick).
                # Non-driven arm NEVER accumulates -- it holds its initial reference exactly.
                cmd_left = cmd_left + (la_c if drive_left else 0.0)
                cmd_right = cmd_right + (ra_c if drive_right else 0.0)
                left_target = cmd_left.copy()
                right_target = cmd_right.copy()

            # --- joint-limit clamp ---
            ll, lu = joint_limits_for(kin, full, idx["left_arm"])
            rl, ru = joint_limits_for(kin, full, idx["right_arm"])
            m = args.joint_limit_margin_rad
            left_target = np.clip(left_target, ll + m, lu - m)
            right_target = np.clip(right_target, rl + m, ru - m)
            if not args.target_from_live:  # anti-windup: keep the accumulated reference within limits
                cmd_left = left_target.copy()
                cmd_right = right_target.copy()

            # --- self-clearance on the proposed full config ---
            q_full_target = full.copy()
            if drive_left:
                q_full_target[idx["left_arm"]] = left_target
            if drive_right:
                q_full_target[idx["right_arm"]] = right_target
            clearance = nearest_self_clearances(kin, q_full_target, pair_count=args.clearance_pair_count,
                                                margin_m=args.min_clearance_m)
            ok = True if args.no_guards else clearance.ok
            clamp_hit = (np.abs(la) > cap).any() or (np.abs(ra) > cap).any()
            verdict = ("PASS (GUARDS OFF)" if args.no_guards else "PASS") if ok else "BLOCK[self_clearance]"
            print(f"  -> left_target(deg)={np.round(np.rad2deg(left_target),1)}")
            print(f"  -> right_target(deg)={np.round(np.rad2deg(right_target),1)}")
            print(f"  clamp_hit={clamp_hit}  min_clearance={clearance.minimum_distance_m}  drive={args.arms}  -> {verdict}")

            if not ok:
                if not args.target_from_live:
                    # ROLLBACK: discard the blocked step's deltas entirely (mirrors the
                    # joint-limit anti-windup; verified necessary -- the old code executed
                    # "skipped" steps on the next passing send)
                    cmd_left = prev_cmd_left
                    cmd_right = prev_cmd_right
                if args.execute:
                    print("  [execute] clearance blocked -> step DISCARDED (not deferred).")
                _wlog(t=time.time(), iter=it, infer_ms=round(infer_ms, 1), verdict=verdict,
                      min_clearance_m=float(clearance.minimum_distance_m), sent=False, discarded=True)
                if not (args.loop or args.infinite):
                    break
                continue

            # gripper debounce COUNTING runs every iteration (stays responsive to the
            # policy); physical actuation happens only after a confirmed, successful send.
            if args.execute and gripper_enabled:
                for side, val in (("left", lg), ("right", rg)) if args.arms == "both" else \
                        ((args.arms, lg if args.arms == "left" else rg),):
                    desired = "close" if val > 0 else "open"
                    _nm, _cnt = grip_pending[side]
                    grip_pending[side] = (desired, _cnt + 1 if _nm == desired else 1)

            sent = False
            if args.execute:
                # BATCHING: the server streams per-control-step deltas from a precomputed
                # action chunk (training executed ~8 of them per inference at 100ms each).
                # Accumulate --steps-per-command deltas (each individually clamped +
                # guard-checked above) into the commanded reference, then send ONE motion.
                # Counter-based (NOT iteration-modulo): clearance-discarded iterations do
                # not defer the flush, so one motion can never silently grow beyond the cap.
                pending_steps += 1
                _n = max(1, int(args.steps_per_command))
                _last = (not args.infinite) and (it == args.iterations - 1)
                if pending_steps >= _n or _last:
                    # per-MOTION amplitude cap: one physical motion never exceeds
                    # n x per-step cap from the LIVE pose, regardless of history
                    motion_cap = _n * cap
                    left_target = np.clip(left_target, q_left_now - motion_cap, q_left_now + motion_cap)
                    right_target = np.clip(right_target, q_right_now - motion_cap, q_right_now + motion_cap)
                    if not args.target_from_live:
                        cmd_left = left_target.copy()
                        cmd_right = right_target.copy()
                    if not args.no_confirm:
                        ans = input(f"  EXECUTE iter {it} (arms={args.arms}, {pending_steps} steps batched)? [y/N] ").strip().lower()
                        if ans != "y":
                            print("  skipped by user.")
                            break
                    # ALWAYS command both arms explicitly via command_body: the driven arm
                    # gets its policy target, the other arm HOLDS its frozen reference.
                    # command_arm is home-centric and must not be used here (it swept the
                    # non-driven arm + head to home -- hardware-verified).
                    _lq = left_target if drive_left else cmd_left
                    _rq = right_target if drive_right else cmd_right
                    builder = command_body(home, left_q=_lq, right_q=_rq,
                                           torso_q=hold_torso, minimum_time_s=args.minimum_time_s,
                                           velocity_limit_rad_s=args.velocity_limit_rad_s,
                                           acceleration_limit_rad_s2=args.acceleration_limit_rad_s2)
                    rv = robot.send_command(builder, int(args.timeout_s)).get()
                    cmd_ok = rv.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
                    print(f"  [execute] sent ({pending_steps} steps batched). finish_ok={cmd_ok}")
                    if not cmd_ok:
                        break
                    pending_steps = 0
                    sent = True

                    # --- grippers: actuate ONLY after a confirmed, successful arm send,
                    # so actuation is always covered by the confirm prompt and happens at
                    # the arm pose the policy chunk paired it with (review finding: the old
                    # placement fired the gripper on accumulate-only iterations, before any
                    # confirm and before the arm had moved) ---
                    if gripper_enabled:
                        for side in (("left", "right") if args.arms == "both" else (args.arms,)):
                            desired, _cnt = grip_pending[side]
                            if not desired or _cnt < max(1, int(args.gripper_debounce_steps)):
                                continue
                            if desired != grip_state[side]:
                                pos = (GRIPPER_OPEN_POSITION if desired == "open"
                                       else (DEFAULT_LEFT_CLOSE_POSITION if side == "left" else DEFAULT_RIGHT_CLOSE_POSITION))
                                try:
                                    command_gripper(args.gripper_url, side, pos,
                                                    current_limit=args.gripper_current_limit,
                                                    hold_s=args.gripper_hold_s, timeout_s=args.timeout_s)
                                    print(f"  [gripper] {side} {desired} -> {pos}")
                                    grip_state[side] = desired
                                    grip_fail[side] = 0
                                except Exception as exc:
                                    grip_fail[side] += 1
                                    print(f"  [gripper] {side} FAILED ({grip_fail[side]}/3): {exc}")
                                    if grip_fail[side] >= 3:
                                        gripper_enabled = False
                                        print("  [gripper] 3 consecutive failures -> gripper DISABLED "
                                              "for the rest of this run (bridge down?)")

            _wlog(t=time.time(), iter=it, infer_ms=round(infer_ms, 1),
                  q_left=np.round(q_left_now, 5).tolist(), q_right=np.round(q_right_now, 5).tolist(),
                  delta_left_raw=np.round(la, 5).tolist(), delta_right_raw=np.round(ra, 5).tolist(),
                  target_left=np.round(left_target, 5).tolist(), target_right=np.round(right_target, 5).tolist(),
                  grip_left=lg, grip_right=rg, grip_state=dict(grip_state),
                  min_clearance_m=float(clearance.minimum_distance_m), verdict=verdict,
                  sent=sent, pending_steps=pending_steps)

            if not (args.loop or args.infinite):
                print(f"\n[done] {mode} complete.")
                break
    finally:
        if streamer is not None:
            streamer.stop()
        cams.close()
        if log_fh is not None:
            log_fh.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MolmoBot RB-Y1 dual-arm rollout client (dry-run by default).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--openpi-client-src", default=DEFAULT_OPENPI_CLIENT_SRC)
    p.add_argument("--task", required=True, help="Language instruction, e.g. 'pick up the red mug'.")
    # cameras
    p.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    p.add_argument("--head-serial", default=None)
    p.add_argument("--left-wrist-serial", default=None)
    p.add_argument("--right-wrist-serial", default=None)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=15)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--show-cameras", action="store_true")
    # robot
    p.add_argument("--arms", choices=["left", "right", "both"], default="right",
                   help="Which arm(s) to drive. Start single-arm to de-risk joint mapping.")
    p.add_argument("--address", default="192.0.2.10:50051")
    p.add_argument("--model", default="a")
    p.add_argument("--power", default=".*")
    p.add_argument("--servo", default=".*")
    p.add_argument("--home", type=Path, default=DEFAULT_HOME)
    p.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    p.add_argument("--home-tolerance-deg", type=float, default=5.0)
    # control + guards
    p.add_argument("--max-step-delta-rad", type=float, default=0.10, help="Hard per-joint per-step delta clamp.")
    p.add_argument("--gripper-debounce-steps", type=int, default=3,
                   help="Require N consecutive identical gripper desires before actuating "
                        "(stops OOD-policy flicker from physically cycling the gripper).")
    p.add_argument("--steps-per-command", type=int, default=1,
                   help="Accumulate N policy steps (each clamped + guard-checked) into ONE motion "
                        "command. Training executed ~8 chunk steps per inference; 1 motion per "
                        "step wastes the chunk on settle time. Try 4.")
    p.add_argument("--delta-deadband-rad", type=float, default=0.002,
                   help="Per-joint deltas below this are zeroed (stops sub-0.1deg noise/bias integrating into drift).")
    p.add_argument("--target-from-live", action="store_true",
                   help="target = live_joints + delta (re-baselines to the sagging pose each tick -> downward drift). "
                        "Default OFF: accumulate a commanded reference, which holds against gravity (matches sim).")
    p.add_argument("--joint-limit-margin-rad", type=float, default=0.05)
    p.add_argument("--min-clearance-m", type=float, default=0.03)
    p.add_argument("--clearance-pair-count", type=int, default=10)
    p.add_argument("--minimum-time-s", type=float, default=2.0)
    p.add_argument("--velocity-limit-rad-s", type=float, default=0.20)
    p.add_argument("--acceleration-limit-rad-s2", type=float, default=0.30)
    p.add_argument("--timeout-s", type=float, default=10.0)
    # gripper
    p.add_argument("--gripper", action="store_true", help="Actuate grippers from the policy (open/close).")
    p.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    p.add_argument("--read-gripper-state", action="store_true", help="Feed live gripper proprio into qpos.")
    p.add_argument("--gripper-current-limit", type=float, default=2.0)
    p.add_argument("--gripper-hold-s", type=float, default=1.0)
    # loop / modes
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--infinite", action="store_true", help="Run forever (implies loop); Ctrl+C to stop.")
    p.add_argument("--execute", action="store_true", help="MASTER MOTION GATE. Off = dry-run (print only).")
    p.add_argument("--no-confirm", action="store_true")
    p.add_argument("--no-robot", action="store_true", help="Smoke: cameras + infer + print action only (no robot).")
    p.add_argument("--save-obs", action="store_true",
                   help="Save the 3 camera images sent on iter 0 to robot/policy/molmobot/obs_dump/ for inspection.")
    p.add_argument("--no-guards", action="store_true",
                   help="Pass the policy's RAW joint deltas (no per-step clamp, no self-clearance block); implies "
                        "--no-confirm. Joint-limit clamp + slow minimum_time/velocity caps are KEPT (hardware safety).")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.infinite:
        args.loop = True
    if args.target_from_live and args.steps_per_command > 1:
        # batching accumulates into the commanded reference; --target-from-live recomputes
        # from live each tick and accumulates nothing, so N-1 of every N policy steps would
        # be silently discarded (review finding). Refuse the combination.
        parser.error("--target-from-live is incompatible with --steps-per-command > 1 "
                     "(accumulate-only steps would be silently discarded)")
    if args.no_guards and not args.no_confirm:
        # NOTE: --no-guards used to silently imply --no-confirm (guards off AND no prompt).
        # Confirm prompts now stay on unless --no-confirm is passed explicitly.
        print("[warn] --no-guards: clamps/clearance OFF, raw policy deltas. "
              "Per-step confirm remains ON (pass --no-confirm explicitly to remove it).")
    run(args)


if __name__ == "__main__":
    main()
