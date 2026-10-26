"""Windows-side raw episode writer for RB-Y1 teleop demos (no lerobot dependency).

The recorder runs on Windows (leader-arm USB, gripper USB, RealSense USB are all
Windows-bound) where the ``lerobot`` package is NOT installed -- it lives in the WSL
openpi env. So the recorder writes a simple raw layout here, and ``raw_to_lerobot.py``
(run in WSL) converts it into the validated LeRobot v2.1 dataset. This class mirrors
``LeRobotEpisodeWriter``'s interface (add_frame / save_episode / discard_episode /
episodes_saved) so the recorder code is identical regardless of backend.

Raw layout::

    <out_dir>/
      episode_000/
        frames.npz          # state (T, dof) f32, action (T, dof) f32
        base/000000.jpg ...      # base camera RGB per frame
        wrist/000000.jpg ...     # left-wrist camera RGB per frame
        meta.json           # {task, fps, dof, image_hw, n_frames}
      episode_001/ ...
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


class RawEpisodeWriter:
    def __init__(self, *, out_dir, dof=8, image_hw=(480, 640), fps=20, task="", overwrite=False):
        self.out_dir = Path(out_dir)
        self.dof = int(dof)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.fps = int(fps)
        self.task = task
        if overwrite and self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        if self.out_dir.exists() and any(self.out_dir.glob("episode_*")):
            raise FileExistsError(
                f"raw dataset already has episodes at {self.out_dir} -- pass --overwrite to replace, "
                "or choose a fresh --repo-id (never silently append to a prior collection)."
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.root = self.out_dir
        self._episodes_saved = len(list(self.out_dir.glob("episode_*")))
        self._reset_buffer()

    def _reset_buffer(self):
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._base: list[np.ndarray] = []
        self._wrist: list[np.ndarray] = []

    def add_frame(self, *, state, action, base_rgb, wrist_rgb, task):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.dof or action.shape[0] != self.dof:
            raise ValueError(f"state/action must be {self.dof}-dim; got {state.shape} / {action.shape}")
        base_rgb = np.asarray(base_rgb)
        wrist_rgb = np.asarray(wrist_rgb)
        if base_rgb.ndim != 3 or base_rgb.shape[2] != 3 or base_rgb.dtype != np.uint8:
            raise ValueError(f"base_rgb must be HxWx3 uint8 RGB; got {base_rgb.shape} {base_rgb.dtype}")
        if wrist_rgb.ndim != 3 or wrist_rgb.shape[2] != 3 or wrist_rgb.dtype != np.uint8:
            raise ValueError(f"wrist_rgb must be HxWx3 uint8 RGB; got {wrist_rgb.shape} {wrist_rgb.dtype}")
        if task and not self.task:
            self.task = task
        self._states.append(state)
        self._actions.append(action)
        self._base.append(base_rgb.copy())
        self._wrist.append(wrist_rgb.copy())

    def save_episode(self) -> int:
        n = len(self._states)
        if n == 0:
            return 0
        ep_dir = self.out_dir / f"episode_{self._episodes_saved:03d}"
        (ep_dir / "base").mkdir(parents=True, exist_ok=True)
        (ep_dir / "wrist").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(self._base[i]).save(ep_dir / "base" / f"{i:06d}.jpg", quality=95)
            Image.fromarray(self._wrist[i]).save(ep_dir / "wrist" / f"{i:06d}.jpg", quality=95)
        np.savez(ep_dir / "frames.npz",
                 state=np.stack(self._states).astype(np.float32),
                 action=np.stack(self._actions).astype(np.float32))
        (ep_dir / "meta.json").write_text(json.dumps({
            "task": self.task, "fps": self.fps, "dof": self.dof,
            "image_hw": list(self.image_hw), "n_frames": n,
        }, indent=2), encoding="utf-8")
        self._episodes_saved += 1
        self._reset_buffer()
        return n

    def discard_episode(self) -> None:
        self._reset_buffer()

    @property
    def episodes_saved(self) -> int:
        return self._episodes_saved
