#!/usr/bin/env python
"""Offline policy<->demo REPLAY validation (NO robot, NO motion).

Feeds RECORDED training-episode observations (base+wrist JPG + state) back through the
RUNNING openpi server and compares the model's predicted absolute action chunk to the
recorded ground-truth next poses (action[t] == state[t+1], verified). This answers the
question "did we actually train the model properly?" WITHOUT touching the robot:

  * step-0 joint MAE TIGHT (< ~4 deg) on TRAINING data  -> the model fit its data well;
    training/serving is NOT broken. Deploy inconsistency is the coverage/memorization
    ceiling (40 demos) -> the fix is data, not training.
  * step-0 joint MAE POOR (> ~12 deg) on TRAINING data  -> the model can't even reproduce
    the demos it was trained on -> training/serving is BROKEN (wrong checkpoint, norm_stats
    mismatch, frozen/!loaded weights, encoder) -> fix THAT before collecting more data.

It also checks the GRIPPER-CLOSE channel specifically on grasp frames (gt raw > commit):
if the model rarely predicts a close where the demo did, the close boundary is under-fit
(the "aligns but won't grasp" symptom) even on training data.

Usage (server must be up; same --host/--port as the rollout):
  python robot/policy/openpi/validate_policy_replay.py --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the EXACT obs builder + client import + constants the live rollout uses, so the
# replay observation is byte-for-byte what inference sends.
from pi05_rollout import (  # noqa: E402
    ARM_DOF,
    DEFAULT_OPENPI_CLIENT_SRC,
    DEFAULT_RAW_DIR,
    DEFAULT_TASK,
    _import_openpi_client,
    build_obs,
)


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline replay validation of the served pi0.5 policy vs the training demos.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--openpi-client-src", default=DEFAULT_OPENPI_CLIENT_SRC)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--episodes", default="0,8,16,24,32,39",
                   help="comma-separated episode numbers to replay (default: a spread across the 40).")
    p.add_argument("--frames-per-episode", type=int, default=12,
                   help="evenly-spaced frames sampled per episode (each one inference).")
    p.add_argument("--horizon", type=int, default=10, help="chunk length H to compare against ground truth.")
    p.add_argument("--grasp-raw-threshold", type=float, default=2.8,
                   help="gt gripper raw above this = a 'closed/grasp' frame (for the close-prediction check).")
    args = p.parse_args()

    wcp = _import_openpi_client(args.openpi_client_src)
    print(f"[server] connecting {args.host}:{args.port} ...")
    client = wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        print("[server] metadata:", client.get_server_metadata())
    except Exception as exc:  # noqa: BLE001
        print("[warn] no server metadata:", exc)

    eps = [f"episode_{int(s):03d}" for s in args.episodes.split(",") if s.strip() != ""]
    H = args.horizon
    o_j0: list[float] = []
    o_jc: list[float] = []
    o_g0: list[float] = []
    o_gc: list[float] = []
    grasp_pred: list[float] = []
    grasp_gt: list[float] = []

    print(f"\n=== Replay validation: {len(eps)} episodes x {args.frames_per_episode} frames, H={H} ===")
    print(f"    task={args.task!r}\n")
    for ep in eps:
        npz = args.raw_dir / ep / "frames.npz"
        if not npz.exists():
            print(f"[skip] {ep}: no frames.npz at {npz}")
            continue
        state = np.load(npz)["state"].astype(float)  # (N,8) = [7 left-arm rad, gripper raw]
        n = state.shape[0]
        hi = max(1, n - H - 1)
        idxs = np.unique(np.linspace(0, hi - 1, args.frames_per_episode).astype(int)) if hi > 1 else np.array([0])
        ej0: list[float] = []
        ejc: list[float] = []
        eg0: list[float] = []
        egc: list[float] = []
        for i in idxs:
            base = load_rgb(args.raw_dir / ep / "base" / f"{i:06d}.jpg")
            wrist = load_rgb(args.raw_dir / ep / "wrist" / f"{i:06d}.jpg")
            obs = build_obs(base, wrist, state[i], args.task)
            pred = np.asarray(client.infer(obs)["actions"], dtype=float)  # (H,8) ABSOLUTE
            k = int(min(H, pred.shape[0], n - 1 - i))
            if k <= 0:
                continue
            gt = state[i + 1: i + 1 + k]
            pj, gj = pred[:k, :ARM_DOF], gt[:, :ARM_DOF]
            pg, gg = pred[:k, ARM_DOF], gt[:, ARM_DOF]
            ej0.append(float(np.rad2deg(np.abs(pj[0] - gj[0])).mean()))
            ejc.append(float(np.rad2deg(np.abs(pj - gj)).mean()))
            eg0.append(float(abs(pg[0] - gg[0])))
            egc.append(float(np.abs(pg - gg).mean()))
            for kk in range(k):
                if gg[kk] > args.grasp_raw_threshold:
                    grasp_pred.append(float(pg[kk]))
                    grasp_gt.append(float(gg[kk]))
        if ej0:
            print(f"{ep}: step0 jointMAE={np.mean(ej0):5.2f}deg  chunk jointMAE={np.mean(ejc):5.2f}deg  "
                  f"step0 gripMAE={np.mean(eg0):.3f}  chunk gripMAE={np.mean(egc):.3f}  ({len(ej0)} frames)")
            o_j0 += ej0
            o_jc += ejc
            o_g0 += eg0
            o_gc += egc

    if not o_j0:
        print("\n[error] no frames evaluated (check --raw-dir / episode numbers / server).")
        return

    mj0 = float(np.mean(o_j0))
    print("\n=== OVERALL (reproduction on TRAINING episodes) ===")
    print(f"  step-0 joint MAE : {mj0:5.2f} deg   (p95 {np.percentile(o_j0, 95):5.2f})")
    print(f"  chunk  joint MAE : {np.mean(o_jc):5.2f} deg   (p95 {np.percentile(o_jc, 95):5.2f})")
    print(f"  step-0 grip  MAE : {np.mean(o_g0):.3f} raw")
    print(f"  chunk  grip  MAE : {np.mean(o_gc):.3f} raw")
    if grasp_gt:
        gp, ggt = np.array(grasp_pred), np.array(grasp_gt)
        frac = float((gp > args.grasp_raw_threshold).mean()) * 100.0
        print(f"  grasp frames (gt>{args.grasp_raw_threshold}): n={len(ggt)}  pred mean={gp.mean():.2f} "
              f"(gt mean={ggt.mean():.2f})  pred crosses commit in {frac:.0f}% of them")

    print("\n=== VERDICT ===")
    if mj0 < 4.0:
        print(f"  Joint reproduction TIGHT ({mj0:.2f} deg) -> the model FIT its training data well.")
        print("  => Training/serving is NOT broken. Deploy inconsistency = coverage/memorization -> collect data.")
    elif mj0 < 12.0:
        print(f"  Joint reproduction LOOSE ({mj0:.2f} deg) -> only roughly fits even the TRAIN demos.")
        print("  => Under-training / lr / norm worth a look before collecting data.")
    else:
        print(f"  Joint reproduction POOR ({mj0:.2f} deg) on TRAINING data -> training/serving likely BROKEN.")
        print("  => Fix checkpoint/norm_stats/encoder/weight-loading BEFORE collecting more data.")
    if grasp_gt and (np.array(grasp_pred) > args.grasp_raw_threshold).mean() < 0.5:
        print("  NOTE: on gt-grasp frames the model rarely predicts a close -> the CLOSE channel is under-fit")
        print("        even on training data (the 'aligns but won't grasp' symptom is partly trained-in).")


if __name__ == "__main__":
    main()
