#!/usr/bin/env python3
"""Read-only audit of data, upstream revisions, Python packages and GPUs."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "baseline" / "scene_change"
sys.path.insert(0, str(PACKAGE))

from egoscene.benchmark import discover_scenes  # noqa: E402
from egoscene.cli import DEFAULT_DATA, DEFAULT_MANIFEST  # noqa: E402


def main() -> int:
    errors: list[str] = []
    lock = json.loads((PACKAGE / "upstreams.lock.json").read_text(encoding="utf-8"))
    folder_names = {
        "3DGS-CD": "3DGS-CD",
        "EfficientSAM": "EfficientSAM",
        "GS-DIFF": "GS-DIFF",
        "MV3DCD": "MV3DCD",
        "PGSR": "PGSR",
    }
    print("Upstreams")
    for name, row in lock.items():
        if name not in folder_names:
            print(f"  {name:24s} {row['revision'][:12]} pip-pinned")
            continue
        folder = REPO / ".external" / folder_names[name]
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{name}: unavailable ({exc})")
            continue
        status = "ok" if actual == row["revision"] else "MISMATCH"
        print(f"  {name:12s} {actual[:12]} {status}")
        if status != "ok":
            errors.append(f"{name}: expected {row['revision']}, got {actual}")
    for patch_name, label in (
        ("3dgscd_known_eval_pose.patch", "known-pose"),
        ("3dgscd_empty_image_diff.patch", "empty-difference"),
    ):
        patch = PACKAGE / "patches" / patch_name
        try:
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(REPO / ".external" / "3DGS-CD"),
                    "apply",
                    "--reverse",
                    "--check",
                    str(patch),
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
            print(f"  3DGS-CD {label} patch ok")
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"3DGS-CD {label} patch is not applied ({exc})")

    print("\nBenchmark")
    try:
        scenes = discover_scenes(
            DEFAULT_MANIFEST,
            pairs_root=DEFAULT_DATA / "pairs_fast",
            pre_models_root=DEFAULT_DATA / "pre_models",
            post_models_root=DEFAULT_DATA / "post_models",
        )
        query_count = sum(len(scene.queries) for scene in scenes)
        print(f"  scene-interventions {len(scenes)}")
        print(f"  fixed queries       {query_count}")
        if (len(scenes), query_count) != (60, 720):
            errors.append(f"Expected 60 scenes/720 queries, got {len(scenes)}/{query_count}")
    except Exception as exc:  # audit should show every failure in one pass
        errors.append(f"benchmark discovery: {exc}")

    print("\nPackages")
    for distribution in ("torch", "gsplat", "numpy", "scipy", "Pillow", "plyfile"):
        try:
            version = importlib.metadata.version(distribution)
            print(f"  {distribution:12s} {version}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing Python distribution: {distribution}")
    try:
        import torch

        print(f"  CUDA available {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  visible GPUs   {torch.cuda.device_count()}")
    except Exception as exc:
        errors.append(f"torch import: {exc}")

    if errors:
        print("\nFAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
