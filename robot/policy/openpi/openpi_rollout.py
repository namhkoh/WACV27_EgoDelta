"""Zero-shot openpi (pi05_droid) rollout client for RB-Y1-a, dry-run by default.

Pipeline per tick (mirrors the DROID runtime, but routes the output through
RB-Y1's own IK instead of commanding Franka joints):

    capture base + wrist RGB  ->  build DROID observation  ->  policy.infer()
      ->  (10, 8) action chunk: 7 Franka joint velocities + 1 gripper
      ->  Franka Jacobian retarget to an end-effector twist            [franka_retarget]
      ->  absolute target TCP pose (clamped)  ->  solve_tcp_pose_ik     [reused]
      ->  safety guards (workspace / IK error / step-delta / joint-limit / self-clearance)
      ->  PRINT everything.  Motion only with --execute (off by default).

The openpi policy server runs separately (WSL2, PyTorch path) on :8000; this
client talks to it over websocket. See openpi_deploy_README.md.

Safety (per project memory): dry-run is the default and prints proposed motion
without commanding. --execute requires every guard to pass, uses slow
minimum_time + low velocity/accel caps, and confirms each waypoint unless
--no-confirm. Zero-shot pi05_droid output on a non-Franka arm is expected to be
erratic; clamps and guards are the safety net, not the policy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
from robot.kinematics.ik_solver import solve_tcp_pose_ik
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.types import IKSolverConfig
from robot.kinematics.utils.realsense_click_move_arm import (
    DEFAULT_HOME,
    command_arm,
    connect_motion_robot,
    validate_home_state,
    workspace_check,
)
from robot.policy.openpi.franka_retarget import retarget_step
from robot.policy.openpi.multicam_capture import MultiCamRealSense, list_devices
from robot.policy.openpi.multicam_view import CameraStreamer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "config" / "realsense_cameras.json"
DEFAULT_RIGHT_TOOLFIT = REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_right_knife_tip_edge_deleaf_01.json"
# Mirrors guarded_bc_rollout.DEFAULT_LEFT_TOOLFIT (ee_left jaw center).
DEFAULT_LEFT_TOOLFIT = Path("D:/tmp/egodelta_isaac_left_jaw_center_toolfit.json")
DEFAULT_OPENPI_CLIENT_SRC = "C:/path/to/projects/openpi/packages/openpi-client/src"
# Gripper bridge (position-mode /jog). Open is the same for both arms; close differs
# (mirrors restore_home_state.DEFAULT_{LEFT,RIGHT}_CLOSE_POSITION).
DEFAULT_GRIPPER_URL = "http://192.0.2.20:8765"
GRIPPER_OPEN_POSITION = -8.0
DEFAULT_RIGHT_CLOSE_POSITION = 4.8
DEFAULT_LEFT_CLOSE_POSITION = 10.945

# DROID was collected at 15 Hz; one chunk step is 1/15 s.
DROID_DT_S = 1.0 / 15.0


def _import_openpi_client(client_src: str):
    if client_src and client_src not in sys.path:
        sys.path.insert(0, client_src)
    from openpi_client import image_tools, websocket_client_policy  # noqa: E402

    return image_tools, websocket_client_policy


def connect_state_robot(*, address: str, model: str) -> Any:
    """State-only connection (no power/servo) for reading joints in dry-run."""
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
    return robot


def command_gripper(url: str, arm: str, position: float, *, current_limit: float,
                    hold_s: float, timeout_s: float) -> dict:
    """Drive one gripper to an absolute encoder position via the HTTP bridge /jog
    (position mode, which re-enables torque). Mirrors restore_home_state._restore_gripper."""
    query = urllib.parse.urlencode(
        {"arm": arm, "position": float(position), "current_limit": float(current_limit), "hold_s": float(hold_s)}
    )
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/jog?{query}", timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")  # surface the bridge's error message
        raise RuntimeError(f"bridge HTTP {exc.code}: {body}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(f"gripper command failed: {payload}")
    return payload


def read_gripper_position_raw(url: str, arm: str, timeout_s: float) -> float | None:
    """Read the active arm's raw gripper encoder position from the bridge /status
    (id 0 = right, id 1 = left). Returns None if the bridge/motor is unavailable."""
    target_id = 1 if arm == "left" else 0
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/status", timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    for motor in payload.get("motor_states", []):
        if int(motor.get("id", -1)) == target_id and motor.get("position") is not None:
            return float(motor["position"])
    return None


def load_camera_serials(path: Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_observation(image_tools, *, base_rgb, wrist_rgb, q_arm, gripper, prompt) -> dict:
    """DROID observation schema expected by pi05_droid (see openpi droid_policy.py)."""
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(base_rgb, 224, 224),
        "observation/wrist_image_left": image_tools.resize_with_pad(wrist_rgb, 224, 224),
        "observation/joint_position": np.asarray(q_arm, dtype=np.float32).reshape(7),
        "observation/gripper_position": np.asarray([gripper], dtype=np.float32),
        "prompt": prompt,
    }


def evaluate_guards(
    *,
    kin: RBY1Kinematics,
    locked_state,
    q_prev: np.ndarray,
    ik_result,
    target_pose: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    args,
) -> dict[str, Any]:
    q_target = np.asarray(ik_result.q_arm_rad, dtype=float)
    q_full_target = kin.q_full_from_arm(locked_state, q_target)

    ws = workspace_check(target_pose[:3, 3], bounds)
    ik_ok = bool(ik_result.ok) and ik_result.position_error_m <= args.max_ik_position_error_m \
        and ik_result.orientation_error_rad <= args.max_ik_orientation_error_rad

    step_delta = float(np.max(np.abs(q_target - np.asarray(q_prev, dtype=float))))
    step_ok = step_delta <= args.max_step_delta_rad

    lower, upper = kin.active_joint_limits(locked_state.full_q_rad)
    margin = args.joint_limit_margin_rad
    within_limits = bool(np.all(q_target >= lower + margin) and np.all(q_target <= upper - margin))

    clearance = nearest_self_clearances(
        kin, q_full_target, pair_count=args.clearance_pair_count, margin_m=args.min_clearance_m
    )

    failed = []
    if not ws["ok"]:
        failed.append("workspace")
    if not ik_ok:
        failed.append("ik")
    if not step_ok:
        failed.append("step_delta")
    if not within_limits:
        failed.append("joint_limits")
    if not clearance.ok:
        failed.append("self_clearance")

    return {
        "ok": len(failed) == 0,
        "failed": failed,
        "workspace_ok": ws["ok"],
        "ik_ok": ik_ok,
        "ik_position_error_m": float(ik_result.position_error_m),
        "ik_orientation_error_rad": float(ik_result.orientation_error_rad),
        "step_delta_rad": step_delta,
        "within_joint_limits": within_limits,
        "min_clearance_m": clearance.minimum_distance_m,
    }


def run(args: argparse.Namespace) -> None:
    image_tools, websocket_client_policy = _import_openpi_client(args.openpi_client_src)

    serials = load_camera_serials(args.camera_config)
    base_serial = args.base_serial or serials["base"]
    # Pick the ACTIVE arm's wrist camera (left_wrist / right_wrist in the config).
    wrist_serial = args.wrist_serial or serials.get(f"{args.arm}_wrist") or serials.get("wrist_a")
    if not wrist_serial:
        raise SystemExit(f"no wrist serial for arm={args.arm}: pass --wrist-serial or set {args.arm}_wrist in the camera config")
    print(f"[cameras] base={base_serial}  wrist({args.arm})={wrist_serial}")
    for d in list_devices():
        print(f"          attached: serial={d['serial']} {d['name']}")

    print(f"[policy] connecting to openpi server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        print(f"[policy] server metadata: {client.get_server_metadata()}")
    except Exception as exc:  # metadata is best-effort
        print(f"[policy] (no metadata: {exc})")

    # --- Robot / kinematics (skipped in --no-ik smoke mode) ---
    kin = locked_state = robot = home = None
    if not args.no_ik:
        print("[robot] building kinematics from firmware URDF ...")
        toolfit = args.toolfit or (DEFAULT_RIGHT_TOOLFIT if args.arm == "right" else DEFAULT_LEFT_TOOLFIT)
        print(f"[robot] arm={args.arm} toolfit={toolfit}")
        kin = RBY1Kinematics.from_live_robot(
            address=args.address, model_name=args.model, arm=args.arm, toolfit_path=toolfit
        )
        locked_state = kin.locked_state_from_home(args.home)
        home = json.loads(Path(args.home).read_text(encoding="utf-8"))
        if args.execute:
            robot = connect_motion_robot(address=args.address, model=args.model, power=args.power, servo=args.servo)
            validate_home_state(robot, home, tolerance_rad=np.deg2rad(args.home_tolerance_deg))
            print("[robot] connected (MOTION) and validated at home.")
        else:
            robot = connect_state_robot(address=args.address, model=args.model)
            print("[robot] connected (STATE-ONLY, dry-run).")

    bounds = {
        "x": (args.min_x_m, args.max_x_m),
        "y": (args.min_y_m, args.max_y_m),
        "z": (args.min_z_m, args.max_z_m),
    }
    ik_cfg = IKSolverConfig(clearance_margin_m=args.min_clearance_m, clearance_pair_count=args.clearance_pair_count)

    mode = "EXECUTE" if args.execute else ("NO-IK SMOKE" if args.no_ik else "DRY-RUN")
    print(f"\n===== openpi rollout  mode={mode}  arm={args.arm}  prompt={args.prompt!r} =====\n")

    cams = MultiCamRealSense(
        {"base": base_serial, "wrist": wrist_serial},
        width=args.cam_width, height=args.cam_height, fps=args.cam_fps,
        warmup_frames=args.warmup_frames,
    )
    # When showing cameras, a background thread becomes the SOLE reader + display so the
    # window streams smoothly even while the control thread blocks on a motion command.
    streamer = CameraStreamer(cams) if args.show_cameras else None
    try:
        # Live state seed (or fallback for --no-ik).
        if robot is not None:
            q_arm = np.asarray(robot.get_state().position, dtype=float)[list(kin.active_arm.joint_indices)]
        else:
            q_arm = np.asarray(args.fallback_q_arm or [0.0] * 7, dtype=float)
        sim_q = q_arm.copy()
        gripper_state: str | None = None
        gripper_enabled = args.gripper
        gripper_fail_count = 0
        gripper_close_position = (
            args.gripper_close_position
            if args.gripper_close_position is not None
            else (DEFAULT_LEFT_CLOSE_POSITION if args.arm == "left" else DEFAULT_RIGHT_CLOSE_POSITION)
        )

        iter_source = itertools.count() if args.infinite else range(args.iterations)
        for it in iter_source:
            frames = streamer.latest() if streamer is not None else cams.read()
            base_rgb = frames["base"]["color_rgb"]
            wrist_rgb = frames["wrist"]["color_rgb"]
            ts_spread = abs(frames["base"]["color_timestamp_ms"] - frames["wrist"]["color_timestamp_ms"])

            # In dry-run we condition on the simulated joint state; with --execute on live state.
            if args.execute and robot is not None:
                sim_q = np.asarray(robot.get_state().position, dtype=float)[list(kin.active_arm.joint_indices)]

            # Gripper proprio fed to the model: live bridge reading (normalized 0=open..1=closed,
            # matching DROID's gripper_position range) if requested, else the constant --gripper-position.
            gripper_obs = args.gripper_position
            if args.read_gripper_state:
                raw = read_gripper_position_raw(args.gripper_url, args.arm, args.timeout_s)
                if raw is not None:
                    span = gripper_close_position - GRIPPER_OPEN_POSITION
                    gripper_obs = float(np.clip((raw - GRIPPER_OPEN_POSITION) / span, 0.0, 1.0)) if span else 0.0
                    if args.gripper_state_invert:
                        gripper_obs = 1.0 - gripper_obs

            obs = build_observation(
                image_tools, base_rgb=base_rgb, wrist_rgb=wrist_rgb,
                q_arm=sim_q, gripper=gripper_obs, prompt=args.prompt,
            )
            t0 = time.perf_counter()
            result = client.infer(obs)
            infer_ms = (time.perf_counter() - t0) * 1e3
            actions = np.asarray(result["actions"], dtype=float)  # (H, 8)
            print(f"--- iter {it}  infer {infer_ms:.0f} ms  cam-skew {ts_spread:.0f} ms  "
                  f"grip_in={gripper_obs:.2f}  action_chunk {actions.shape} ---")

            if streamer is not None:
                streamer.set_status(f"iter {it}  {infer_ms:.0f}ms")

            if args.no_ik:
                print(f"  action[0] = {np.round(actions[0], 4)}  (7 joint-vel + gripper)")
                print(f"  chunk min/max per dim:\n   min {np.round(actions.min(0),3)}\n   max {np.round(actions.max(0),3)}")
                if not args.loop:
                    break
                continue

            tcp_pose = kin.tcp_pose(kin.q_full_from_arm(locked_state, sim_q))
            n_exec = min(args.steps_to_execute, actions.shape[0])
            any_fail = False
            for step in range(n_exec):
                raw = np.clip(actions[step], -1.0, 1.0)
                jvel, grip = raw[:7], float(raw[7])
                rt = retarget_step(
                    current_tcp_pose=tcp_pose, q_arm_rad=sim_q, joint_velocity=jvel,
                    dt_s=DROID_DT_S, joint_vel_scale=args.joint_vel_scale,
                    max_trans_m=args.max_trans_m, max_rot_rad=args.max_rot_rad,
                )
                ik = solve_tcp_pose_ik(
                    kin, locked_state, rt.target_tcp_pose,
                    seed_q_arm_rad=sim_q, reference_q_arm_rad=sim_q, config=ik_cfg,
                )
                guards = evaluate_guards(
                    kin=kin, locked_state=locked_state, q_prev=sim_q, ik_result=ik,
                    target_pose=rt.target_tcp_pose, bounds=bounds, args=args,
                )
                grip_cmd = "OPEN" if grip > 0.5 else "CLOSE"
                tgt = rt.target_tcp_pose[:3, 3]
                verdict = "PASS" if guards["ok"] else f"BLOCK[{','.join(guards['failed'])}]"
                clamp = ("T" if rt.trans_clamped else "-") + ("R" if rt.rot_clamped else "-")
                print(f"  step {step}: tgt_xyz=[{tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}] "
                      f"d_trans={np.linalg.norm(rt.delta_trans_m)*1e3:5.1f}mm "
                      f"d_rot={np.linalg.norm(rt.delta_rot_rad)*1e3:5.1f}mrad clamp={clamp} | "
                      f"ik_pos_err={ik.position_error_m*1e3:5.1f}mm step={guards['step_delta_rad']:.3f}rad "
                      f"clr={guards['min_clearance_m']} grip={grip_cmd} -> {verdict}")
                if not guards["ok"]:
                    any_fail = True
                    if args.execute:
                        print("  [execute] guard blocked -> stopping chunk.")
                        break
                    continue

                if args.execute:
                    if not args.no_confirm:
                        ans = input(f"  EXECUTE step {step} on {args.arm} arm? [y/N] ").strip().lower()
                        if ans != "y":
                            print("  skipped by user.")
                            break
                    builder = command_arm(
                        home, args.arm, ik.q_arm_rad, minimum_time_s=args.minimum_time_s,
                        velocity_limit_rad_s=args.velocity_limit_rad_s,
                        acceleration_limit_rad_s2=args.acceleration_limit_rad_s2,
                    )
                    rv = robot.send_command(builder, int(args.timeout_s)).get()
                    ok = rv.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
                    print(f"  [execute] sent. finish_ok={ok}")
                    if not ok:
                        break

                    if gripper_enabled:
                        # Hysteresis around 0.5 so the gripper doesn't chatter; only
                        # command the bridge on an actual open<->close state change.
                        if grip > args.gripper_open_threshold:
                            desired = "open"
                        elif grip < args.gripper_close_threshold:
                            desired = "close"
                        else:
                            desired = gripper_state
                        if desired is not None and desired != gripper_state:
                            pos = GRIPPER_OPEN_POSITION if desired == "open" else gripper_close_position
                            try:
                                command_gripper(args.gripper_url, args.arm, pos,
                                                current_limit=args.gripper_current_limit,
                                                hold_s=args.gripper_hold_s, timeout_s=args.timeout_s)
                                print(f"  [gripper] {desired} -> pos {pos} (raw={grip:.3f})")
                                gripper_state = desired
                                gripper_fail_count = 0
                            except Exception as exc:
                                gripper_fail_count += 1
                                print(f"  [gripper] FAILED: {exc}")
                                if gripper_fail_count >= 3:
                                    gripper_enabled = False
                                    print("  [gripper] DISABLED for the rest of this run after 3 failures "
                                          "(arm keeps running). Likely the bridge's Dynamixel bus is down "
                                          "(motor_states empty) -> restart the gripper bridge on the robot.")
                # advance the simulated state along the (accepted) chunk
                sim_q = np.asarray(ik.q_arm_rad, dtype=float)
                tcp_pose = kin.tcp_pose(kin.q_full_from_arm(locked_state, sim_q))

            if not args.loop:
                tag = "execute" if args.execute else "dry-run"
                if any_fail:
                    print(f"\n[done] {tag} complete (some steps blocked by guards -- expected for zero-shot).")
                else:
                    print(f"\n[done] {tag} complete.")
                break
    finally:
        if streamer is not None:
            streamer.stop()
        cams.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zero-shot openpi (pi05_droid) rollout on RB-Y1-a (dry-run by default).")
    # server
    p.add_argument("--host", default="127.0.0.1", help="openpi policy server host (WSL2 reachable via localhost).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--openpi-client-src", default=DEFAULT_OPENPI_CLIENT_SRC,
                   help="Path to openpi-client src (added to sys.path).")
    p.add_argument("--prompt", required=True, help="Language instruction, e.g. 'pick up the object'.")
    # cameras
    p.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    p.add_argument("--base-serial", default=None)
    p.add_argument("--wrist-serial", default=None, help="Active arm's D405 serial (defaults to wrist_a in config).")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    # robot
    p.add_argument("--arm", choices=["left", "right"], default="right")
    p.add_argument("--address", default="192.0.2.10:50051")
    p.add_argument("--model", default="a")
    p.add_argument("--power", default=".*")
    p.add_argument("--servo", default=".*")
    p.add_argument("--home", type=Path, default=DEFAULT_HOME)
    p.add_argument("--toolfit", type=Path, default=None,
                   help="TCP toolfit JSON. Default is arm-dependent: right=knife, left=jaw.")
    p.add_argument("--home-tolerance-deg", type=float, default=5.0)
    p.add_argument("--gripper-position", type=float, default=0.0, help="Normalized gripper state fed to the model.")
    p.add_argument("--fallback-q-arm", type=float, nargs=7, default=None, help="q_arm for --no-ik when robot absent.")
    # retarget + control
    p.add_argument("--joint-vel-scale", type=float, default=0.2,
                   help="rad/s per unit normalized DROID joint velocity (CONSERVATIVE; tune before motion).")
    p.add_argument("--max-trans-m", type=float, default=0.02, help="Per-step translation clamp (m).")
    p.add_argument("--max-rot-rad", type=float, default=0.10, help="Per-step rotation clamp (rad).")
    p.add_argument("--steps-to-execute", type=int, default=5, help="Chunk prefix length per inference (receding horizon).")
    p.add_argument("--iterations", type=int, default=1, help="Number of inference iterations (set --loop for continuous).")
    p.add_argument("--loop", action="store_true", help="Keep looping (receding horizon); else single chunk.")
    p.add_argument("--infinite", action="store_true",
                   help="Run forever (implies --loop); stop with Ctrl+C. Overrides --iterations.")
    # guards
    p.add_argument("--max-ik-position-error-m", type=float, default=0.03)
    p.add_argument("--max-ik-orientation-error-rad", type=float, default=0.20)
    p.add_argument("--max-step-delta-rad", type=float, default=0.10)
    p.add_argument("--joint-limit-margin-rad", type=float, default=0.05)
    p.add_argument("--min-clearance-m", type=float, default=0.03)
    p.add_argument("--clearance-pair-count", type=int, default=10)
    p.add_argument("--min-x-m", type=float, default=0.25)
    p.add_argument("--max-x-m", type=float, default=0.85)
    p.add_argument("--min-y-m", type=float, default=-0.45)
    p.add_argument("--max-y-m", type=float, default=0.45)
    p.add_argument("--min-z-m", type=float, default=0.35)
    p.add_argument("--max-z-m", type=float, default=1.20)
    # execution (slow + safe + confirmed per project memory)
    p.add_argument("--execute", action="store_true", help="MASTER MOTION GATE. Off = dry-run (print only).")
    p.add_argument("--no-confirm", action="store_true", help="Skip per-waypoint confirm prompt (only with --execute).")
    p.add_argument("--minimum-time-s", type=float, default=5.0)
    p.add_argument("--velocity-limit-rad-s", type=float, default=0.10)
    p.add_argument("--acceleration-limit-rad-s2", type=float, default=0.10)
    p.add_argument("--timeout-s", type=float, default=10.0)
    # gripper (driven via the HTTP bridge on open<->close state changes; opt-in)
    p.add_argument("--gripper", action="store_true",
                   help="Also drive the gripper via the HTTP bridge from the policy's gripper dim (off by default).")
    p.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    p.add_argument("--gripper-open-threshold", type=float, default=0.6, help="grip > this -> open.")
    p.add_argument("--gripper-close-threshold", type=float, default=0.4, help="grip < this -> close (between = hold).")
    p.add_argument("--gripper-close-position", type=float, default=None,
                   help="Encoder close target; default arm-dependent (right=4.8, left=10.945). Open is -8.0.")
    p.add_argument("--gripper-current-limit", type=float, default=2.0)
    p.add_argument("--gripper-hold-s", type=float, default=1.0)
    p.add_argument("--read-gripper-state", action="store_true",
                   help="Feed the LIVE gripper position (read from the bridge, normalized 0=open..1=closed) to the "
                        "model as observation/gripper_position, instead of the constant --gripper-position. "
                        "Fixes the model's gripper perception; recommended whenever the bridge is up.")
    p.add_argument("--gripper-state-invert", action="store_true",
                   help="Invert the gripper-state normalization if the model behaves as if open/closed are swapped.")
    # modes
    p.add_argument("--no-ik", action="store_true",
                   help="Smoke mode: capture + infer + print raw chunk only (no robot / IK / guards).")
    p.add_argument("--show-cameras", action="store_true",
                   help="Display the policy's live camera input (base + active wrist) in an OpenCV window during the run.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.infinite:
        args.loop = True
    if args.no_confirm and not args.execute:
        print("[warn] --no-confirm has no effect without --execute (this is a dry-run).")
    if args.gripper and not args.execute:
        print("[warn] --gripper has no effect without --execute (this is a dry-run).")
    run(args)


if __name__ == "__main__":
    main()
