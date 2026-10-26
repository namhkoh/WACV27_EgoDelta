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
    argv = [
        "--task",
        "harvesting",
        "--mode",
        "extracted_frames",
        "--pair-manifest",
        str(manifest_path),
        "--rgb-stream-id",
        args.rgb_stream_id,
        "--frame-stride",
        str(args.frame_stride),
        "--sam-batch-size",
        str(args.sam_batch_size),
        "--sam-backend",
        "native_sam3",
        "--skip-pipeline-visualizations",
        "--sam-score-thres",
        str(args.sam_score_thres),
        "--min-mask-area-px",
        str(args.min_mask_area_px),
        "--max-mask-area-frac",
        str(args.max_mask_area_frac),
        "--min-visible-post-views",
        str(args.min_visible_post_views),
        "--disappeared-score-thres",
        str(args.disappeared_score_thres),
        "--change-score-mode",
        args.change_score_mode,
        "--match-radius-m",
        str(args.match_radius_m),
        "--out",
        str(out_dir),
    ]
    if args.use_aux_prompts:
        argv.append("--use-aux-prompts")
    return argv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one in-process shard of the harvesting benchmark.")
    parser.add_argument(
        "--pairs-fast-root",
        default="/path/to/data/mps/egodelta_benchmark_data/pairs_fast",
        help="Root containing <scene>/tomato_pair/manifest.json.",
    )
    parser.add_argument("--out-root", default="/tmp/harvest_benchmark")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sam-batch-size", type=int, default=16)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument(
        "--gt-labels",
        help="Optional GT label text file. When provided, only scenes present in this file are processed.",
    )

    parser.add_argument("--use-aux-prompts", action="store_true", default=True)
    parser.add_argument("--no-aux-prompts", action="store_false", dest="use_aux_prompts")
    parser.add_argument("--sam-score-thres", type=float, default=0.25)
    parser.add_argument("--min-mask-area-px", type=int, default=100)
    parser.add_argument("--max-mask-area-frac", type=float, default=0.40)
    parser.add_argument("--min-visible-post-views", type=int, default=3)
    parser.add_argument("--disappeared-score-thres", type=float, default=0.60)
    parser.add_argument("--match-radius-m", type=float, default=0.12)
    parser.add_argument(
        "--change-score-mode",
        choices=("auto", "average", "sam_absence_only", "point_disappearance_only"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards).")

    os.environ.setdefault("SAM3_NATIVE_AMP_DTYPE", "bfloat16")
    os.environ.setdefault("SAM3_DISABLE_ACT_CKPT", "1")

    manifest_paths = sorted(Path(args.pairs_fast_root).glob("*/tomato_pair/manifest.json"))
    selected = set(args.scene_id or [])
    if args.gt_labels:
        from sam_aria_change_baseline.evaluate_cut_points import parse_gt_label_file

        selected.update(parse_gt_label_file(args.gt_labels, task="harvesting").keys())
    manifest_paths = [
        path
        for idx, path in enumerate(manifest_paths)
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
