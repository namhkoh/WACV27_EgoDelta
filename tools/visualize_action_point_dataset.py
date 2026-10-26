#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.manifest import load_action_manifests
from action_points.visualization import draw_label_samples, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated action-point labels.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--crop", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = load_action_manifests(args.manifest)
    out_dir = Path(args.out_dir)
    paths = []
    count = 0
    for manifest in manifests:
        for image_path, samples in group_samples_by_image(manifest["samples"]):
            if count >= args.max_samples:
                break
            path = out_dir / "samples" / f"{count:04d}_{Path(image_path).stem}.jpg"
            draw_label_samples(samples, path, crop=args.crop)
            paths.append(path)
            count += 1
        if count >= args.max_samples:
            break
    make_contact_sheet(paths, out_dir / "contact_sheet.jpg")
    print(f"Saved {len(paths)} visualizations under {out_dir}")
    return 0


def group_samples_by_image(samples: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        key = str(Path(sample["image_path"]).resolve())
        grouped.setdefault(key, []).append(sample)
    return list(grouped.items())


if __name__ == "__main__":
    raise SystemExit(main())
