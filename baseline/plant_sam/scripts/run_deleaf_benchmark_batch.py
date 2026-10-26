#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _scene_id_from_manifest(path: Path) -> str:
    return path.parents[1].name


def _build_command(manifest_path: Path, out_dir: Path, device: str, args: argparse.Namespace) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "env",
        f"SAM3_DEVICE=cuda:{device}",
        "SAM3_NATIVE_AMP_DTYPE=bfloat16",
        "SAM3_DISABLE_ACT_CKPT=1",
        "python",
        "-m",
        "sam_aria_change_baseline.run_pipeline",
        "--task",
        "deleafing",
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
        "--use-aux-prompts",
        "--deleafing-sam-score-thres",
        str(args.deleafing_sam_score_thres),
        "--deleafing-min-mask-area-px",
        str(args.deleafing_min_mask_area_px),
        "--deleafing-max-mask-area-frac",
        str(args.deleafing_max_mask_area_frac),
        "--deleafing-max-primary-masks-per-frame",
        str(args.deleafing_max_primary_masks_per_frame),
        "--deleafing-pipe-like-long-side-frac",
        str(args.deleafing_pipe_like_long_side_frac),
        "--deleafing-pipe-like-aspect-ratio",
        str(args.deleafing_pipe_like_aspect_ratio),
        "--deleafing-primary-vertical-axis-thres",
        str(args.deleafing_primary_vertical_axis_thres),
        "--deleafing-primary-vertical-axis-max-thinness",
        str(args.deleafing_primary_vertical_axis_max_thinness),
        "--deleafing-primary-vertical-axis-image-direction",
        args.deleafing_primary_vertical_axis_image_direction,
        "--petiole-min-points-inside-mask",
        str(args.petiole_min_points_inside_mask),
        "--petiole-dbscan-min-samples",
        str(args.petiole_dbscan_min_samples),
        "--petiole-instance-min-frames",
        str(args.petiole_instance_min_frames),
        "--petiole-min-3d-major-extent-m",
        str(args.petiole_min_3d_major_extent_m),
        "--petiole-max-3d-major-extent-m",
        str(args.petiole_max_3d_major_extent_m),
        "--deleafing-aux-sam-score-thres",
        str(args.deleafing_aux_sam_score_thres),
        "--deleafing-aux-min-mask-area-px",
        str(args.deleafing_aux_min_mask_area_px),
        "--deleafing-main-stem-min-verticality",
        str(args.deleafing_main_stem_min_verticality),
        "--deleafing-main-stem-min-height-frac",
        str(args.deleafing_main_stem_min_height_frac),
        "--deleafing-main-stem-max-width-frac",
        str(args.deleafing_main_stem_max_width_frac),
        "--deleafing-main-stem-min-aspect-ratio",
        str(args.deleafing_main_stem_min_aspect_ratio),
        "--deleafing-aux-min-vertical-extent-m",
        str(args.deleafing_aux_min_vertical_extent_m),
        "--deleafing-aux-min-vertical-ratio",
        str(args.deleafing_aux_min_vertical_ratio),
        "--deleafing-max-aux-masks-per-frame",
        str(args.deleafing_max_aux_masks_per_frame),
        "--disappeared-score-thres",
        str(args.disappeared_score_thres),
        "--match-radius-m",
        str(args.match_radius_m),
        "--out",
        str(out_dir),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deleafing baseline on pairs_fast extracted-frame manifests.")
    parser.add_argument(
        "--pairs-fast-root",
        default="/path/to/data/mps/egodelta_benchmark_data/pairs_fast",
        help="Root containing <scene>/leaf_pair/manifest.json.",
    )
    parser.add_argument("--out-root", default="outputs/deleaf_benchmark")
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--conda-env", default="sam3_aria")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sam-batch-size", type=int, default=1)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scene-id", action="append", help="Run only selected recording id. May be repeated.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-s", type=float, default=30.0)

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
    parser.add_argument("--disappeared-score-thres", type=float, default=0.0)
    parser.add_argument("--match-radius-m", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest_paths = sorted(Path(args.pairs_fast_root).glob("*/leaf_pair/manifest.json"))
    selected = set(args.scene_id or [])
    scenes: list[tuple[str, Path]] = []
    for path in manifest_paths:
        scene_id = _scene_id_from_manifest(path)
        if selected and scene_id not in selected:
            continue
        scenes.append((scene_id, path))
    if args.limit is not None:
        scenes = scenes[: args.limit]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("No devices provided.")

    queue = []
    for scene_id, manifest_path in scenes:
        out_dir = out_root / scene_id
        if out_dir.joinpath("final_predictions.json").exists() and not args.force:
            print(f"skip complete {scene_id}")
            continue
        queue.append((scene_id, manifest_path, out_dir))

    running: dict[str, tuple[subprocess.Popen, Path, str, float]] = {}
    failures: list[str] = []
    while queue or running:
        for device in devices:
            if device in running or not queue:
                continue
            scene_id, manifest_path, out_dir = queue.pop(0)
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "batch_run.log"
            cmd = _build_command(manifest_path, out_dir, device, args)
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write(" ".join(cmd) + "\n\n")
            log_handle.flush()
            process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            running[device] = (process, log_path, scene_id, time.time())
            print(f"start {scene_id} on cuda:{device}")

        time.sleep(args.poll_s)
        for device, (process, log_path, scene_id, start_time) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"done {scene_id} on cuda:{device}: code={code} elapsed_min={elapsed_min:.1f}")
            if code != 0:
                failures.append(scene_id)
                print(f"  log: {log_path}")
            del running[device]

    if failures:
        print("failed scenes:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
