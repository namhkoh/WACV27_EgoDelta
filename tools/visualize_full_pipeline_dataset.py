#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.manifest import load_action_manifests, read_json
from action_points.visualization import draw_prediction_overlay, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector+action-head predictions on generated dataset images.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_images", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ground_truth_by_image: dict[str, list[dict]] = {}
    for manifest in load_action_manifests(args.manifest):
        for sample in manifest["samples"]:
            key = str(Path(sample["image_path"]).resolve())
            ground_truth_by_image.setdefault(key, []).append(sample)

    predictions = read_json(args.predictions)["images"]
    out_dir = Path(args.out_dir)
    paths = []
    for idx, record in enumerate(predictions[: args.max_images]):
        image_path = str(Path(record["image_path"]).resolve())
        out_path = out_dir / "samples" / f"{idx:04d}_{Path(image_path).stem}.jpg"
        draw_prediction_overlay(image_path, record.get("predictions", []), out_path, ground_truth=ground_truth_by_image.get(image_path, []))
        paths.append(out_path)

    make_contact_sheet(paths, out_dir / "contact_sheet.jpg")
    print(f"Saved {len(paths)} full-pipeline dataset visualizations under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
