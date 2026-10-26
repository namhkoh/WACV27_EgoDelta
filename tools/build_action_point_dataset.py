#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.label_generation import LabelBuildConfig, build_action_point_manifest
from action_points.manifest import write_json
from action_points.visualization import draw_label_sample, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project existing EgoDelta/2DGS component-target outputs into RGB frames to build "
            "bbox/action-point pseudo labels. This tool does not infer tomato/leaf targets itself."
        )
    )
    parser.add_argument("--dataset_dir", required=True, help="Pre-scene COLMAP dataset with images/ and sparse/0.")
    parser.add_argument(
        "--presence_npz",
        required=True,
        help="Original EgoDelta/2DGS leaf_component_targets.npz or tomato_component_targets.npz.",
    )
    parser.add_argument("--summary_json", default=None, help="Matching component target summary JSON.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--task_type", required=True, help="harvest, deleaf, clip, or another task label.")
    parser.add_argument("--class_name", default="target", help="Detector class name in unified-class mode.")
    parser.add_argument("--split", default="train", help="Initial split tag; scene-level split can override this later.")
    parser.add_argument("--images", default="images")
    parser.add_argument("--sparse", default="sparse/0")
    parser.add_argument("--mask_key", default=None)
    parser.add_argument("--cluster_key", default=None)
    parser.add_argument("--min_visible_points", type=int, default=8)
    parser.add_argument("--min_visible_fraction", type=float, default=0.01)
    parser.add_argument("--min_bbox_size_px", type=float, default=8.0)
    parser.add_argument("--bbox_percentile", type=float, default=2.0)
    parser.add_argument("--bbox_pad_px", type=float, default=8.0)
    parser.add_argument("--bbox_pad_fraction", type=float, default=0.08)
    parser.add_argument("--target_pad_px", type=float, default=6.0)
    parser.add_argument("--no_include_target_in_bbox", action="store_true")
    parser.add_argument("--min_target_confidence", type=float, default=0.0)
    parser.add_argument("--max_views_per_instance", type=int, default=0)
    parser.add_argument("--max_projected_points_record", type=int, default=64)
    parser.add_argument("--debug_vis", action="store_true")
    parser.add_argument("--debug_max_samples", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = LabelBuildConfig(
        scene_id=args.scene_id,
        task_type=args.task_type,
        class_name=args.class_name,
        split=args.split,
        min_visible_points=args.min_visible_points,
        min_visible_fraction=args.min_visible_fraction,
        min_bbox_size_px=args.min_bbox_size_px,
        bbox_percentile=args.bbox_percentile,
        bbox_pad_px=args.bbox_pad_px,
        bbox_pad_fraction=args.bbox_pad_fraction,
        target_pad_px=args.target_pad_px,
        include_target_in_bbox=not args.no_include_target_in_bbox,
        min_target_confidence=args.min_target_confidence,
        max_views_per_instance=args.max_views_per_instance,
        max_projected_points_record=args.max_projected_points_record,
    )
    manifest = build_action_point_manifest(
        dataset_dir=args.dataset_dir,
        presence_npz=args.presence_npz,
        summary_json=args.summary_json,
        config=config,
        images=args.images,
        sparse=args.sparse,
        mask_key=args.mask_key,
        cluster_key=args.cluster_key,
    )
    manifest_path = out_dir / "action_point_manifest.json"
    write_json(manifest_path, manifest)

    if args.debug_vis:
        vis_dir = out_dir / "debug_labels"
        paths = []
        for idx, sample in enumerate(manifest["samples"][: args.debug_max_samples]):
            path = vis_dir / f"{idx:04d}_{sample['sample_id']}.jpg"
            draw_label_sample(sample, path)
            paths.append(path)
        make_contact_sheet(paths, out_dir / "debug_label_contact_sheet.jpg")

    print(f"Saved manifest: {manifest_path}")
    print(f"Samples: {manifest['num_samples']} from {manifest['num_instances']} instances and {manifest['num_views']} views")
    print(f"Mask key: {manifest['source']['mask_key']} | Cluster key: {manifest['source']['cluster_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
