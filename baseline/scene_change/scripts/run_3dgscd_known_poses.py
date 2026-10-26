#!/usr/bin/env python3
"""Invoke official 3DGS-CD while preserving supplied EgoDelta query poses."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    config = args.config.resolve()
    transform = args.transform.resolve()
    output = args.output.resolve()
    checkpoint = args.ckpt.resolve()
    sys.path.insert(0, str(upstream))
    # Official EfficientSAM resolves its released weights relative to the
    # 3DGS-CD repository, matching the upstream CLI's documented invocation.
    os.chdir(upstream)
    from nerfstudio.scripts.change_det import ChangeDet

    hyperparameters = transform.parent / "configs.json"
    detector = ChangeDet(config, output)
    detector.main(
        transforms_json=transform,
        configs=hyperparameters.resolve() if hyperparameters.is_file() else None,
        checkpoint_dir=checkpoint,
        # Aria MPS poses are already known. This also prevents evaluation RGB
        # from entering pose optimization.
        refine_pose=False,
        debug=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
