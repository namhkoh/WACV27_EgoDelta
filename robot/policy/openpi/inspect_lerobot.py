"""Inspect a converted LeRobot dataset before trusting it for training (RUN IN WSL).

    cd ~/openpi && uv run --no-sync python \
      /mnt/d/research/EgoDelta/robot/policy/openpi/inspect_lerobot.py --repo-id rby1_left_pnp

Checks the pre-collection gate: state/action are the LEFT arm (plausible joint ranges),
the gripper channel (dim 7) actually varies open<->close, and action ~= next state
(absolute-target sanity). Dumps a few frame images for eyeballing.
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="rby1_left_pnp")
    p.add_argument("--root", default=None)
    p.add_argument("--dump-frames", type=int, default=4, help="Save N sample frames to /tmp.")
    args = p.parse_args()

    import lerobot.common.datasets.lerobot_dataset as m
    ds = m.LeRobotDataset(args.repo_id, root=args.root)
    print(f"CODEBASE_VERSION={m.CODEBASE_VERSION}  episodes={ds.num_episodes}  frames={ds.num_frames}  fps={ds.fps}")
    print(f"features={list(ds.features)}")

    # gather state/action across all frames
    states, actions = [], []
    for i in range(len(ds)):
        it = ds[i]
        states.append(np.asarray(it["state"]).reshape(-1))
        actions.append(np.asarray(it["actions"]).reshape(-1))
    S = np.stack(states); A = np.stack(actions)
    print(f"\nstate shape={S.shape}  action shape={A.shape}")

    np.set_printoptions(precision=3, suppress=True)
    print("\n--- STATE per-dim (dims 0-6 = left arm joints rad, dim 7 = gripper [0,1]) ---")
    print("min: ", S.min(0))
    print("max: ", S.max(0))
    print("range:", S.max(0) - S.min(0))
    print("\n--- ACTION per-dim ---")
    print("min: ", A.min(0))
    print("max: ", A.max(0))

    # gripper channel (raw encoder) must vary -- compare span to your gripper's FULL
    # open<->close raw range (measure it once with --measure-gripper on the recorder).
    g = S[:, 7]
    print(f"\nGRIPPER (state dim 7, RAW encoder): min={g.min():.3f} max={g.max():.3f} span={g.max()-g.min():.3f}")
    print("  -> compare span to the gripper's full open<->close raw range. If the demo did a real")
    print("     grasp+release, this span should be ~the full range. A tiny span = no grasp learned.")

    # joint plausibility: left-arm joints should be within ~[-pi, pi]-ish, not constant
    arm_range = (S[:, :7].max(0) - S[:, :7].min(0))
    print(f"\nARM joint motion (rad): {arm_range}")
    if arm_range.max() < 0.05:
        print("  !! WARNING: arm barely moves -- wrong indices or no motion recorded.")
    else:
        print("  OK: arm joints move (this is the LEFT arm if values match your teleop).")

    # action ~= next state (absolute-target sanity): action[t] should track state[t+1]
    if len(S) > 2:
        err = np.abs(A[:-1] - S[1:]).mean()
        print(f"\nACTION vs NEXT-STATE mean abs diff: {err:.4f} rad "
              "(should be ~0 if action==next state was recorded correctly)")

    # dump a few frames
    if args.dump_frames > 0 and "image" in ds.features:
        from PIL import Image
        idxs = np.linspace(0, len(ds) - 1, args.dump_frames).astype(int)
        for k, i in enumerate(idxs):
            it = ds[int(i)]
            for key, tag in (("image", "base"), ("wrist_image", "wrist")):
                if key in it:
                    arr = np.asarray(it[key])
                    if arr.dtype != np.uint8:  # CHW float -> HWC uint8
                        arr = (np.transpose(arr, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8) if arr.ndim == 3 and arr.shape[0] == 3 else arr.astype(np.uint8)
                    Image.fromarray(arr).save(f"/tmp/inspect_{tag}_{k}.png")
        print(f"\nwrote {args.dump_frames} sample frames per cam to /tmp/inspect_*.png "
              "(open them to confirm base+wrist views look right)")


if __name__ == "__main__":
    main()
