#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scene_id_from_manifest(path: Path) -> str:
    return path.parents[1].name


def _pipeline_argv(manifest_path: Path, out_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        "--task", "deleafing",
        "--mode", "extracted_frames",
        "--pair-manifest", str(manifest_path),
        "--rgb-stream-id", args.rgb_stream_id,
        "--frame-stride", str(args.frame_stride),
        "--sam-batch-size", str(args.sam_batch_size),
        "--sam-backend", "native_sam3",
        "--skip-pipeline-visualizations",
        "--use-aux-prompts",
        "--deleafing-sam-score-thres", str(args.deleafing_sam_score_thres),
        "--deleafing-min-mask-area-px", str(args.deleafing_min_mask_area_px),
        "--deleafing-max-mask-area-frac", str(args.deleafing_max_mask_area_frac),
        "--deleafing-max-primary-masks-per-frame", str(args.deleafing_max_primary_masks_per_frame),
        "--deleafing-pipe-like-long-side-frac", str(args.deleafing_pipe_like_long_side_frac),
        "--deleafing-pipe-like-aspect-ratio", str(args.deleafing_pipe_like_aspect_ratio),
        "--deleafing-primary-vertical-axis-thres", str(args.deleafing_primary_vertical_axis_thres),
        "--deleafing-primary-vertical-axis-max-thinness", str(args.deleafing_primary_vertical_axis_max_thinness),
        "--deleafing-primary-vertical-axis-image-direction", args.deleafing_primary_vertical_axis_image_direction,
        "--petiole-min-points-inside-mask", str(args.petiole_min_points_inside_mask),
        "--petiole-dbscan-min-samples", str(args.petiole_dbscan_min_samples),
        "--petiole-instance-min-frames", str(args.petiole_instance_min_frames),
        "--petiole-min-3d-major-extent-m", str(args.petiole_min_3d_major_extent_m),
        "--petiole-max-3d-major-extent-m", str(args.petiole_max_3d_major_extent_m),
        "--deleafing-aux-sam-score-thres", str(args.deleafing_aux_sam_score_thres),
        "--deleafing-aux-min-mask-area-px", str(args.deleafing_aux_min_mask_area_px),
        "--deleafing-main-stem-min-verticality", str(args.deleafing_main_stem_min_verticality),
        "--deleafing-main-stem-min-height-frac", str(args.deleafing_main_stem_min_height_frac),
        "--deleafing-main-stem-max-width-frac", str(args.deleafing_main_stem_max_width_frac),
        "--deleafing-main-stem-min-aspect-ratio", str(args.deleafing_main_stem_min_aspect_ratio),
        "--deleafing-aux-min-vertical-extent-m", str(args.deleafing_aux_min_vertical_extent_m),
        "--deleafing-aux-min-vertical-ratio", str(args.deleafing_aux_min_vertical_ratio),
        "--deleafing-max-aux-masks-per-frame", str(args.deleafing_max_aux_masks_per_frame),
        "--out", str(out_dir),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one in-process shard of the deleafing benchmark.")
    parser.add_argument(
        "--pairs-fast-root",
        default="/path/to/data/mps/egodelta_benchmark_data/pairs_fast",
        help="Root containing <scene>/leaf_pair/manifest.json.",
    )
    parser.add_argument("--out-root", default="outputs/deleaf_benchmark")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sam-batch-size", type=int, default=1)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument(
        "--gt-labels",
        help="Optional GT label text file. When provided, only scenes present in this file are processed.",
    )

    parser.add_argument("--deleafing-sam-score-thres", type=float, default=0.16)
    parser.add_argument("--deleafing-min-mask-area-px", type=int, default=50)
    parser.add_argument("--deleafing-max-mask-area-frac", type=float, default=0.03)
    parser.add_argument("--deleafing-max-primary-masks-per-frame", type=int, default=12)
    parser.add_argument("--deleafing-pipe-like-long-side-frac", type=float, default=0.35)
    parser.add_argument("--deleafing-pipe-like-aspect-ratio", type=float, default=6.0)
    parser.add_argument("--deleafing-primary-vertical-axis-thres", type=float, default=0.90)
    parser.add_argument("--deleafing-primary-vertical-axis-max-thinness", type=float, default=0.45)
    parser.add_argument("--deleafing-primary-vertical-axis-image-direction", default="image_horizontal")
    parser.add_argument("--petiole-min-points-inside-mask", type=int, default=4)
    parser.add_argument("--petiole-dbscan-min-samples", type=int, default=3)
    parser.add_argument("--petiole-instance-min-frames", type=int, default=1)
    parser.add_argument("--petiole-min-3d-major-extent-m", type=float, default=0.04)
    parser.add_argument("--petiole-max-3d-major-extent-m", type=float, default=0.55)
    parser.add_argument("--deleafing-aux-sam-score-thres", type=float, default=0.35)
    parser.add_argument("--deleafing-aux-min-mask-area-px", type=int, default=80)
    parser.add_argument("--deleafing-main-stem-min-verticality", type=float, default=0.82)
    parser.add_argument("--deleafing-main-stem-min-height-frac", type=float, default=0.18)
    parser.add_argument("--deleafing-main-stem-max-width-frac", type=float, default=0.22)
    parser.add_argument("--deleafing-main-stem-min-aspect-ratio", type=float, default=1.8)
    parser.add_argument("--deleafing-aux-min-vertical-extent-m", type=float, default=0.08)
    parser.add_argument("--deleafing-aux-min-vertical-ratio", type=float, default=1.25)
    parser.add_argument("--deleafing-max-aux-masks-per-frame", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards).")

    os.environ.setdefault("SAM3_NATIVE_AMP_DTYPE", "bfloat16")
    os.environ.setdefault("SAM3_DISABLE_ACT_CKPT", "1")

    manifest_paths = sorted(Path(args.pairs_fast_root).glob("*/leaf_pair/manifest.json"))
    selected = set(args.scene_id or [])
    if args.gt_labels:
        from sam_aria_change_baseline.evaluate_cut_points import parse_gt_label_file

        selected.update(parse_gt_label_file(args.gt_labels, task="deleafing").keys())
    manifest_paths = [
        path for idx, path in enumerate(manifest_paths)
        if idx % args.num_shards == args.shard_index
    ]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    from sam_aria_change_baseline.run_pipeline import main as run_pipeline_main

    for path in manifest_paths:
        scene_id = _scene_id_from_manifest(path)
        if selected and scene_id not in selected:
            continue
        out_dir = out_root / scene_id
        if out_dir.joinpath("final_predictions.json").exists() and not args.force:
            print(f"skip complete {scene_id}", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"inprocess_shard_{args.shard_index}.log"
        print(f"start {scene_id} shard={args.shard_index} device={os.environ.get('SAM3_DEVICE')}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(" ".join(_pipeline_argv(path, out_dir, args)) + "\n\n")
            log_handle.flush()
            try:
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    code = run_pipeline_main(_pipeline_argv(path, out_dir, args))
            except SystemExit as exc:
                code = int(exc.code or 0)
            except Exception:
                code = 1
                traceback.print_exc(file=log_handle)
            log_handle.flush()

        if code != 0:
            failures.append(scene_id)
            print(f"failed {scene_id} code={code} log={log_path}", flush=True)
        else:
            print(f"done {scene_id} log={log_path}", flush=True)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if failures:
        print("failed scenes: " + ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
