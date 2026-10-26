"""LeRobot v2.1 dataset writer for RB-Y1 single-arm imitation data (openpi-ready).

This is the ONE place that knows the on-disk dataset format. The recorder
(``teleop_record_lerobot.py``) and any future raw->lerobot converter both go
through here, so the format-correctness risk (the openpi LeRobot **v2.1** schema
that fine-tuning consumes -- v3.x is incompatible and silently wastes a whole
collection) lives in a single, offline-testable module.

Schema matches openpi's own conversion reference
(``openpi/examples/libero/convert_libero_data_to_lerobot.py``) so the training
DataConfig is a near-copy of ``LeRobotLiberoDataConfig``:
  * ``image``        -- base camera, dtype "image" (PNG), HxWx3 uint8 RGB
  * ``wrist_image``  -- active-arm wrist camera, same
  * ``state``        -- proprio, float32 (DOF,)  [7 arm joints + 1 gripper = 8]
  * ``actions``      -- absolute target, float32 (DOF,)  (plural key == openpi default
                        ``action_sequence_keys=("actions",)``, no config override needed)
  * per-frame ``task`` -- the language instruction (v2.1 stores task per frame)

openpi assumes proprio in ``state`` and actions in ``actions`` (see the libero
reference comment). The model transform pads 8->32 internally; we store raw 8.

Self-test (offline, no robot): ``python -m robot.policy.openpi.lerobot_writer --selftest``
writes a synthetic 2-episode dataset, reloads it, and asserts the v2.1 schema --
the Phase-1 gate before collecting real demos. Run it in the openpi env (WSL):
  cd ~/openpi && uv run --no-sync python /mnt/d/.../lerobot_writer.py --selftest
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np


class LeRobotEpisodeWriter:
    """Thin wrapper over ``LeRobotDataset`` for one single-arm RB-Y1 dataset.

    Usage:
        w = LeRobotEpisodeWriter(repo_id="rby1_left_pnp", dof=8, image_hw=(480, 640), fps=20)
        # per episode:
        for frame in episode_frames:   # in time order
            w.add_frame(state=..., action=..., base_rgb=..., wrist_rgb=..., task="...")
        w.save_episode()
        # (no finalize/consolidate step -- v2.1 save_episode writes everything per episode)
    """

    def __init__(
        self,
        *,
        repo_id: str,
        dof: int = 8,
        image_hw: tuple[int, int] = (480, 640),
        fps: int = 20,
        root: str | Path | None = None,
        overwrite: bool = False,
    ) -> None:
        # Imported lazily so the module imports on the Windows host (no lerobot there);
        # only the recorder/selftest -- run in the openpi env -- actually instantiate.
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

        self.dof = int(dof)
        h, w = int(image_hw[0]), int(image_hw[1])
        out_root = Path(root) if root is not None else (HF_LEROBOT_HOME / repo_id)
        if overwrite and out_root.exists():
            shutil.rmtree(out_root)
        if out_root.exists():
            raise FileExistsError(
                f"dataset already exists at {out_root} -- pass overwrite=True to replace, "
                "or choose a fresh repo_id (do NOT silently append to a prior collection)."
            )

        features = {
            "image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (self.dof,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (self.dof,), "names": ["actions"]},
        }
        self._ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(fps),
            root=str(out_root) if root is not None else None,
            robot_type="rby1a_left",
            features=features,
            use_videos=False,  # dtype "image" -> PNG frames (matches the libero reference)
            image_writer_threads=8,
            image_writer_processes=0,
        )
        self.root = self._ds.root
        self.repo_id = repo_id
        self._frames_in_episode = 0
        self._episodes_saved = 0

    def add_frame(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        task: str,
    ) -> None:
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
        self._ds.add_frame({
            "image": base_rgb,
            "wrist_image": wrist_rgb,
            "state": state,
            "actions": action,
            "task": task,
        })
        self._frames_in_episode += 1

    def save_episode(self) -> int:
        """Flush the buffered frames as one episode. Returns its frame count."""
        n = self._frames_in_episode
        if n == 0:
            return 0
        self._ds.save_episode()
        self._episodes_saved += 1
        self._frames_in_episode = 0
        return n

    def discard_episode(self) -> None:
        """Drop the in-progress episode buffer without saving (e.g. a botched demo)."""
        self._ds.clear_episode_buffer()
        self._frames_in_episode = 0

    @property
    def episodes_saved(self) -> int:
        return self._episodes_saved


def _selftest() -> int:
    """Write a synthetic 2-episode dataset, reload it, assert the v2.1 schema."""
    import tempfile

    import lerobot.common.datasets.lerobot_dataset as m

    dof, h, w, fps = 8, 64, 96, 20
    root = Path(tempfile.mkdtemp(prefix="lerobot_selftest_"))
    repo_id = "selftest/rby1_left_pnp"
    ds_root = root / repo_id
    print(f"[selftest] writing to {ds_root}")
    writer = LeRobotEpisodeWriter(repo_id=repo_id, dof=dof, image_hw=(h, w), fps=fps,
                                  root=ds_root, overwrite=True)
    rng = np.random.default_rng(0)
    ep_lens = [12, 9]
    for ei, ln in enumerate(ep_lens):
        for fi in range(ln):
            writer.add_frame(
                state=rng.standard_normal(dof).astype(np.float32),
                action=rng.standard_normal(dof).astype(np.float32),
                base_rgb=(rng.integers(0, 256, (h, w, 3))).astype(np.uint8),
                wrist_rgb=(rng.integers(0, 256, (h, w, 3))).astype(np.uint8),
                task="pick up the red ball and place it on the book",
            )
        saved = writer.save_episode()
        assert saved == ln, f"episode {ei}: saved {saved} != {ln}"

    # ---- reload + assert ----
    reloaded = m.LeRobotDataset(repo_id, root=ds_root)
    print(f"[selftest] CODEBASE_VERSION = {m.CODEBASE_VERSION}")
    assert str(m.CODEBASE_VERSION) in ("v2.1", "2.1"), f"expected v2.1, got {m.CODEBASE_VERSION}"
    assert reloaded.num_episodes == len(ep_lens), f"num_episodes {reloaded.num_episodes} != {len(ep_lens)}"
    assert reloaded.num_frames == sum(ep_lens), f"num_frames {reloaded.num_frames} != {sum(ep_lens)}"
    feats = reloaded.features
    for key, shape in (("state", (dof,)), ("actions", (dof,))):
        assert key in feats, f"missing feature {key}"
        assert tuple(feats[key]["shape"]) == shape, f"{key} shape {feats[key]['shape']} != {shape}"
    for key in ("image", "wrist_image"):
        assert key in feats, f"missing image feature {key}"
    item = reloaded[0]
    assert "task" in item and isinstance(item["task"], str), "per-frame task missing/!str"
    assert item["state"].shape[-1] == dof and item["actions"].shape[-1] == dof
    print(f"[selftest] OK: {reloaded.num_episodes} eps, {reloaded.num_frames} frames, "
          f"features={list(feats)}, task='{item['task']}'")
    shutil.rmtree(root, ignore_errors=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true", help="Offline round-trip schema test (no robot).")
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
