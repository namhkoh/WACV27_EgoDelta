"""Convert raw RB-Y1 teleop episodes (written on Windows by teleop_record_lerobot.py)
into a LeRobot v2.1 dataset for openpi fine-tuning. RUN IN THE WSL openpi ENV:

    cd ~/openpi && uv run --no-sync python \
      /mnt/d/research/EgoDelta/robot/policy/openpi/raw_to_lerobot.py \
      --raw-dir /mnt/d/research/EgoDelta/data/rby1_left_pnp_raw \
      --repo-id rby1_left_pnp --overwrite

Reads the raw layout (episode_NNN/{frames.npz, base/*.jpg, wrist/*.jpg, meta.json})
and writes through the validated LeRobotEpisodeWriter (same schema the writer self-test
round-trips). After this, the dataset is ready for compute_norm_stats + training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# repo root (this file is at robot/policy/openpi/) so `import robot...` works when run
# from the openpi clone in WSL.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robot.policy.openpi.lerobot_writer import LeRobotEpisodeWriter


def _load_episode(ep_dir: Path):
    meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
    arr = np.load(ep_dir / "frames.npz")
    states, actions = arr["state"], arr["action"]
    n = states.shape[0]
    if actions.shape[0] != n:
        raise ValueError(f"{ep_dir.name}: state/action length mismatch {states.shape} vs {actions.shape}")
    base_files = sorted((ep_dir / "base").glob("*.jpg"))
    wrist_files = sorted((ep_dir / "wrist").glob("*.jpg"))
    if not (len(base_files) == len(wrist_files) == n):
        raise ValueError(f"{ep_dir.name}: frame count mismatch states={n} "
                         f"base={len(base_files)} wrist={len(wrist_files)}")
    return meta, states, actions, base_files, wrist_files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="Raw dataset dir (contains episode_NNN/).")
    p.add_argument("--repo-id", default="rby1_left_pnp", help="Output LeRobot repo_id.")
    p.add_argument("--root", default=None, help="Output root (default: HF LeRobot cache).")
    p.add_argument("--task", default=None,
                   help="Override the language instruction for ALL episodes (the task is just "
                        "the language label, independent of the recorded sensor data). e.g. "
                        "'pick up the tomato and place it in the box'. Default: keep each "
                        "episode's recorded task.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    ep_dirs = sorted(raw_dir.glob("episode_*"))
    if not ep_dirs:
        raise SystemExit(f"no episode_* dirs under {raw_dir}")

    meta0 = json.loads((ep_dirs[0] / "meta.json").read_text(encoding="utf-8"))
    dof = int(meta0["dof"])
    image_hw = tuple(meta0["image_hw"])
    fps = int(meta0["fps"])
    print(f"[convert] {len(ep_dirs)} episodes  dof={dof}  image_hw={image_hw}  fps={fps}")

    writer = LeRobotEpisodeWriter(repo_id=args.repo_id, dof=dof, image_hw=image_hw, fps=fps,
                                  root=args.root, overwrite=args.overwrite)
    total = 0
    if args.task:
        print(f"[convert] OVERRIDING task for all episodes -> '{args.task}'")
    for ep_dir in ep_dirs:
        meta, states, actions, base_files, wrist_files = _load_episode(ep_dir)
        task = args.task or meta.get("task", "")
        for i in range(states.shape[0]):
            base_rgb = np.asarray(Image.open(base_files[i]).convert("RGB"), dtype=np.uint8)
            wrist_rgb = np.asarray(Image.open(wrist_files[i]).convert("RGB"), dtype=np.uint8)
            writer.add_frame(state=states[i], action=actions[i],
                             base_rgb=base_rgb, wrist_rgb=wrist_rgb, task=task)
        n = writer.save_episode()
        total += n
        print(f"[convert] {ep_dir.name} -> {n} frames (task='{task}')")
    print(f"[convert] DONE: {writer.episodes_saved} episodes, {total} frames -> {writer.root}")
    print("[convert] next: compute_norm_stats --config-name <your config>, then train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
