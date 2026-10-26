"""Windows-side, fully READ-ONLY recorder for RB-Y1 left-arm pi0.5 demos.

Your teleop runs UNCHANGED on the UPC and already exposes everything over the network:
  * arm joints -> gRPC robot state  (192.0.2.10:50051 from Windows)
  * left gripper -> the teleop's own HTTP server  (http://192.0.2.20:8765/status)
  * cameras -> local RealSense on this Windows host

So this recorder touches nothing on the robot: it reads arm state (gRPC) + gripper
(HTTP) + cameras, and writes a RAW dataset (raw_episode_writer) that raw_to_lerobot.py
later converts to LeRobot v2.1. No fork of your teleop, no UPC-side code.

    # 1) start your teleop on the UPC (unchanged)
    # 2) on Windows:
    python -m robot.policy.openpi.record_pi05_demos \
        --task "pick up the red ball and place it in the box" --repo-id rby1_left_pnp --overwrite

Episode control (type + Enter):  s = start   e = end+save   d = discard   q = quit

state[8]  = [follower left q (7), follower gripper RAW encoder (1)]
action[8] = NEXT observed pose (state[t+1]); absolute targets -- the faithfully-tracked
            follower pose one step ahead IS what the master commanded this tick. The final
            frame's action repeats its own state (terminal hold).

To measure the gripper's true open<->close raw range first (then fully open/close it on
the master arm and watch the printed min/max):
    python -m robot.policy.openpi.record_pi05_demos --measure-gripper
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rby1_sdk as rby

from robot.policy.openpi.multicam_capture import MultiCamRealSense
from robot.policy.openpi.raw_episode_writer import RawEpisodeWriter

class _GripperPoller:
    """Background poll of the teleop's /status HTTP for the LEFT gripper RAW encoder
    position (id 1), so the capture loop never blocks on network latency.

    We store the RAW encoder, NOT a [0,1] normalization: the true open/close bounds
    are the teleop's own calibration (unknown here), so any hardcoded bounds risk
    silently compressing the signal. openpi's compute_norm_stats standardizes the
    gripper channel from the dataset's actual distribution, and deploy un-normalizes
    back to a raw target -- so as long as record + deploy share the raw encoder space
    (they do: the bridge /jog takes raw positions), no bounds guess is needed.
    """

    def __init__(self, url, hz=20.0):
        self._url = url.rstrip("/") + "/status"
        self._period = 1.0 / float(hz)
        self._lock = threading.Lock()
        self._raw = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="gripper-poll", daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            raw = None
            try:
                with urllib.request.urlopen(self._url, timeout=1.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                for motor in payload.get("motor_states", []):
                    if int(motor.get("id", -1)) == 1 and motor.get("position") is not None:
                        raw = float(motor["position"])
            except Exception:
                raw = None
            if raw is not None:
                with self._lock:
                    self._raw = raw
            self._stop.wait(self._period)

    def start(self):
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._raw

    def stop(self):
        self._stop.set()


def run(args):
    # ---- read-only robot state connection (coexists with the UPC's command stream) ----
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise SystemExit(f"failed to connect (state) to {args.address}")
    model = robot.model()
    left_idx = list(model.left_arm_idx)
    head_idx = list(model.head_idx)
    print(f"[record] gRPC state {args.address}  left_arm_idx={left_idx}  head_idx={head_idx}")

    gripper = _GripperPoller(args.gripper_url, hz=args.store_hz)
    gripper.start()
    print(f"[record] gripper poll {args.gripper_url}/status")

    cams = MultiCamRealSense(
        {"base": args.base_serial, "left_wrist": args.wrist_serial},
        width=args.cam_width, height=args.cam_height, fps=args.cam_fps, warmup_frames=30,
    )
    out_dir = args.out_dir or os.path.join(_REPO_ROOT, "data", f"{args.repo_id}_raw")
    writer = RawEpisodeWriter(out_dir=out_dir, dof=8,
                              image_hw=(args.cam_height, args.cam_width),
                              fps=int(args.store_hz), task=args.task, overwrite=args.overwrite)
    print(f"[record] RAW dataset -> {writer.root}  task='{args.task}'  store={args.store_hz}Hz")
    print(f"[record] convert later (WSL): uv run --no-sync python "
          f"robot/policy/openpi/raw_to_lerobot.py --raw-dir {writer.root} --repo-id {args.repo_id}")

    period = 1.0 / float(args.store_hz)
    head_tol = np.deg2rad(args.head_tol_deg)
    head_ref = [None]
    cmd_q = queue.Queue()
    stop = threading.Event()

    def keyboard():
        print("\n[record] commands:  s=start  e=end+save  d=discard  q=quit\n", flush=True)
        while not stop.is_set():
            try:
                line = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                cmd_q.put("quit"); return
            if line in ("s", "e", "d", "q"):
                cmd_q.put({"s": "start", "e": "save", "d": "discard", "q": "quit"}[line])
                if line == "q":
                    return

    threading.Thread(target=keyboard, name="record-keys", daemon=True).start()

    # per-episode buffers (action = next state, so we buffer then shift on save)
    buf_state, buf_base, buf_wrist = [], [], []
    armed = False
    last_warn = 0.0
    next_t = time.monotonic()

    def _flush_episode():
        n = len(buf_state)
        if n == 0:
            return 0
        for t in range(n):
            action = buf_state[t + 1] if t < n - 1 else buf_state[t]  # next-state; terminal holds
            writer.add_frame(state=buf_state[t], action=action,
                             base_rgb=buf_base[t], wrist_rgb=buf_wrist[t], task=args.task)
        saved = writer.save_episode()
        buf_state.clear(); buf_base.clear(); buf_wrist.clear()
        return saved

    try:
        while not stop.is_set():
            # 1) episode commands
            try:
                while True:
                    cmd = cmd_q.get_nowait()
                    if cmd == "start":
                        if armed:
                            print("[record] already recording; e to end first.", flush=True)
                        else:
                            armed = True; buf_state.clear(); buf_base.clear(); buf_wrist.clear()
                            print(f"[record] >>> episode {writer.episodes_saved + 1} RECORDING", flush=True)
                    elif cmd == "save":
                        if armed:
                            armed = False
                            n = _flush_episode()
                            print(f"[record] <<< saved episode {writer.episodes_saved} ({n} frames)", flush=True)
                    elif cmd == "discard":
                        if armed:
                            armed = False; buf_state.clear(); buf_base.clear(); buf_wrist.clear()
                            print("[record] xxx discarded.", flush=True)
                    elif cmd == "quit":
                        if armed:
                            buf_state.clear(); buf_base.clear(); buf_wrist.clear()
                            print("[record] quit while recording -> discarded.", flush=True)
                        stop.set()
            except queue.Empty:
                pass
            if stop.is_set():
                break

            # 2) capture one frame if armed
            if armed:
                full = np.asarray(robot.get_state().position, dtype=np.float32)
                q_left = full[left_idx]
                head = full[head_idx]
                graw = gripper.latest()  # RAW gripper encoder (openpi norm-stats scales it)
                if graw is None:
                    if time.monotonic() - last_warn > 1.0:
                        print("[record] WAITING for gripper /status (is teleop's :8765 up?)", flush=True)
                        last_warn = time.monotonic()
                else:
                    if head_ref[0] is None:
                        head_ref[0] = head.copy()
                        print(f"[record] head reference: {np.round(np.degrees(head_ref[0]),1)} deg", flush=True)
                    elif np.max(np.abs(head - head_ref[0])) > head_tol and time.monotonic() - last_warn > 1.0:
                        print(f"[record] !! HEAD MOVED {np.round(np.degrees(head-head_ref[0]),1)} deg -- "
                              "fix the head (viewpoint inconsistency).", flush=True)
                        last_warn = time.monotonic()
                    try:
                        frames = cams.read()
                        state8 = np.concatenate([q_left, np.array([graw], np.float32)]).astype(np.float32)
                        buf_state.append(state8)
                        buf_base.append(frames["base"]["color_rgb"])
                        buf_wrist.append(frames["left_wrist"]["color_rgb"])
                    except Exception as exc:
                        print(f"[record] camera read failed, skipping frame: {exc}", flush=True)

            # 3) pace
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        gripper.stop()
        cams.close()
        print(f"[record] done. {writer.episodes_saved} episodes under {writer.root}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", type=str, default="192.0.2.10:50051", help="Robot gRPC (state).")
    p.add_argument("--model", type=str, default="a")
    p.add_argument("--gripper-url", type=str, default="http://192.0.2.20:8765",
                   help="The teleop's gripper HTTP server.")
    p.add_argument("--task", type=str, default="pick up the red ball and place it in the box")
    p.add_argument("--repo-id", type=str, default="rby1_left_pnp")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--store-hz", type=float, default=20.0)
    p.add_argument("--base-serial", type=str, default=os.environ.get("BASE_CAMERA_SERIAL"), help="Base D455 serial.")
    p.add_argument("--wrist-serial", type=str, default=os.environ.get("LEFT_WRIST_CAMERA_SERIAL"), help="Left-wrist D405 serial.")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--head-tol-deg", type=float, default=3.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--measure-gripper", action="store_true",
                   help="Live-print the left gripper RAW encoder + running min/max so you can "
                        "fully open then fully close it on the master arm to read the true range.")
    args = p.parse_args()
    if not args.measure_gripper and (not args.base_serial or not args.wrist_serial):
        p.error("Set --base-serial and --wrist-serial, or BASE_CAMERA_SERIAL and LEFT_WRIST_CAMERA_SERIAL.")
    if args.measure_gripper:
        gp = _GripperPoller(args.gripper_url, hz=20.0); gp.start()
        lo, hi = float("inf"), float("-inf")
        print("[measure] fully OPEN then fully CLOSE the gripper on the master arm. Ctrl+C to stop.")
        try:
            while True:
                r = gp.latest()
                if r is not None:
                    lo, hi = min(lo, r), max(hi, r)
                    print(f"  raw={r:8.3f}   running range [{lo:.3f}, {hi:.3f}]  span={hi-lo:.3f}", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\n[measure] gripper RAW range: open/close span = {hi-lo:.3f}  (min {lo:.3f}, max {hi:.3f})")
        finally:
            gp.stop()
        return
    run(args)


if __name__ == "__main__":
    main()
