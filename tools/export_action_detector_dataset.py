#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.manifest import load_action_manifests, load_scene_split
from action_points.yolo_export import export_yolo_detection_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export action-point pseudo bboxes to Ultralytics YOLO detect format.")
    parser.add_argument("--manifest", action="append", required=True, help="Action-point manifest JSON. Pass once per scene.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_split_json", default=None, help="JSON mapping scene_id to train/val/test.")
    parser.add_argument("--class_mode", choices=("unified", "task"), default="unified")
    parser.add_argument("--unified_class_name", default="target")
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument(
        "--allow_single_scene_debug_split",
        action="store_true",
        help="Mirror train images into val for a one-scene smoke test. Do not use for real evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = load_action_manifests(args.manifest)
    summary = export_yolo_detection_dataset(
        manifests=manifests,
        out_dir=args.out_dir,
        scene_split=load_scene_split(args.scene_split_json),
        class_mode=args.class_mode,
        unified_class_name=args.unified_class_name,
        copy_images=args.copy_images,
        allow_single_scene_debug_split=args.allow_single_scene_debug_split,
    )
    print(f"Saved YOLO dataset: {summary['out_dir']}")
    print(f"Data YAML: {summary['data_yaml']}")
    print(f"Images: {summary['num_images']} | labels: {summary['num_samples']} | classes: {summary['class_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
