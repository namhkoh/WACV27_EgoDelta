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
    parser = argparse.ArgumentParser(description="Visualize action-point head predictions against pseudo labels.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--evaluation_samples", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_by_id = {}
    for manifest in load_action_manifests(args.manifest):
        for sample in manifest["samples"]:
            sample_by_id[sample["sample_id"]] = sample

    eval_samples = read_json(args.evaluation_samples)
    out_dir = Path(args.out_dir)
    paths = []
    for idx, item in enumerate(eval_samples[: args.max_samples]):
        sample = sample_by_id.get(item["sample_id"])
        if sample is None:
            continue
        pred = {
            "class_name": sample.get("class_name", "target"),
            "confidence": max(0.0, 1.0 - float(item.get("error_px", 0.0)) / 50.0),
            "bbox_xyxy": sample["bbox_xyxy"],
            "action_point_2d": item["predicted_2d"],
        }
        out_path = out_dir / "samples" / f"{idx:04d}_{sample['sample_id']}.jpg"
        draw_prediction_overlay(sample["image_path"], [pred], out_path, ground_truth=[sample])
        paths.append(out_path)

    make_contact_sheet(paths, out_dir / "contact_sheet.jpg")
    print(f"Saved {len(paths)} prediction visualizations under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
