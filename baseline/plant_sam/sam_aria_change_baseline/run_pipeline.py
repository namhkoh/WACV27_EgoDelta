from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    PipelineConfig,
    VALID_BACKENDS,
    VALID_TASKS,
    parse_world_up,
)
from .instance_fusion import fuse_prepost
from .io_extracted import prepare_frames_from_pair_manifest
from .mask_lifting import run_mask_lifting
from .mask_postprocess import postprocess_prediction_file
from .mps_geometry import AriaMpsGeometry, ensure_same_world_frame
from .prepost_change import run_prepost_change
from .sam_backend import BackendUnavailableError, create_sam_backend
from .sam_inference import run_sam_inference
from .target_rules import run_target_rules
from .visualization import (
    save_change_debug_overlays_on_frames,
    save_summary_figure,
    save_target_overlays_on_frames,
    write_change_debug_plys,
    write_visualization_plys,
)


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r") as handle:
        return json.load(handle)


def _count_masks(predictions: list[dict[str, Any]]) -> int:
    return int(sum(len(frame.get("masks", [])) for frame in predictions))


def _per_prompt_stats(
    raw_pre: list[dict[str, Any]],
    raw_post: list[dict[str, Any]],
    filtered_pre: list[dict[str, Any]],
    filtered_post: list[dict[str, Any]],
    lifted_pre: list[dict[str, Any]],
    lifted_post: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}

    def add(prompt: str, key: str) -> None:
        if not prompt:
            return
        stats.setdefault(prompt, {"raw_masks": 0, "filtered_masks": 0, "lifted_ok": 0})
        stats[prompt][key] += 1

    for predictions, key in ((raw_pre, "raw_masks"), (raw_post, "raw_masks")):
        for frame in predictions:
            for mask in frame.get("masks", []):
                add(mask.get("prompt", ""), key)
    for predictions, key in ((filtered_pre, "filtered_masks"), (filtered_post, "filtered_masks")):
        for frame in predictions:
            for mask in frame.get("masks", []):
                for prompt in mask.get("source_prompts", [mask.get("prompt", "")]):
                    add(prompt, key)
    for lifted_records in (lifted_pre, lifted_post):
        for record in lifted_records:
            if record.get("status") not in ("ok", "dbscan_failed_kept_foreground"):
                continue
            for prompt in record.get("source_prompts", []):
                add(prompt, "lifted_ok")
    return stats


def _save_debug_stats(
    out_dir: str | Path,
    stats: dict[str, Any],
) -> Path:
    path = Path(out_dir) / "debug_stats.json"
    with path.open("w") as handle:
        json.dump(stats, handle, indent=2)
    return path


def _print_summary(stats: dict[str, Any], task: str, sam_backend: str, out_dir: str | Path) -> None:
    print("===== SAM 3 + Aria MPS Change Baseline =====")
    print(f"Task: {task}")
    print(f"SAM backend: {sam_backend}")
    print(f"Pre frames: {stats.get('num_pre_frames', 0)}")
    print(f"Post frames: {stats.get('num_post_frames', 0)}")
    print(
        "Raw SAM masks pre/post: "
        f"{stats.get('num_pre_sam_masks_raw', 0)} / {stats.get('num_post_sam_masks_raw', 0)}"
    )
    print(
        "Filtered masks pre/post: "
        f"{stats.get('num_pre_sam_masks_filtered', 0)} / "
        f"{stats.get('num_post_sam_masks_filtered', 0)}"
    )
    print(
        "Lifted 3D masks pre/post: "
        f"{stats.get('num_pre_lifted_ok', 0)} / {stats.get('num_post_lifted_ok', 0)}"
    )
    if task == "deleafing":
        print(
            "Petiole-size rejected lifted masks pre/post: "
            f"{stats.get('num_pre_lifted_size_rejected', 0)} / "
            f"{stats.get('num_post_lifted_size_rejected', 0)}"
        )
    print(f"Pre 3D instances: {stats.get('num_pre_instances', 0)}")
    print(f"Post 3D instances: {stats.get('num_post_instances', 0)}")
    print(f"Disappeared candidates: {stats.get('num_disappeared_candidates', 0)}")
    print(f"Final predictions: {stats.get('num_final_predictions', 0)}")
    print(f"Output: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM 3 concept masks plus Project Aria MPS geometry baseline."
    )
    parser.add_argument("--task", required=True, choices=VALID_TASKS)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("extracted_frames",),
        help="Use an already-extracted pre/post pair manifest such as pairs_fast/<scene>/<task>_pair/manifest.json.",
    )
    parser.add_argument("--sam-backend", default="native_sam3", choices=VALID_BACKENDS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Subsample the already-extracted frame list. Use 1 to consume every selected frame.",
    )
    parser.add_argument(
        "--sam-batch-size",
        type=int,
        default=1,
        help="Number of RGB frames to process together in SAM image inference when the backend supports batching.",
    )
    parser.add_argument(
        "--pair-manifest",
        required=True,
        help="Pre/post already-extracted pair manifest, e.g. pairs_fast/<scene>/leaf_pair/manifest.json.",
    )

    parser.add_argument("--use-video-mode", action="store_true")
    parser.add_argument("--use-aux-prompts", action="store_true")
    parser.add_argument("--exemplar-config")
    parser.add_argument("--sam-score-thres", type=float)
    parser.add_argument(
        "--deleafing-sam-score-thres",
        type=float,
        help="Primary SAM score threshold for deleafing petiole prompts.",
    )
    parser.add_argument("--aux-sam-score-thres", type=float)
    parser.add_argument(
        "--deleafing-aux-sam-score-thres",
        type=float,
        help="Auxiliary main-stem SAM score threshold for deleafing.",
    )
    parser.add_argument("--min-mask-area-px", type=int)
    parser.add_argument(
        "--deleafing-min-mask-area-px",
        type=int,
        help="Minimum primary mask area for deleafing petiole prompts.",
    )
    parser.add_argument(
        "--deleafing-max-mask-area-frac",
        type=float,
        help="Maximum primary mask area fraction for deleafing petiole prompts.",
    )
    parser.add_argument(
        "--deleafing-max-primary-masks-per-frame",
        type=int,
        help="Cap primary petiole-like masks kept per frame for deleafing.",
    )
    parser.add_argument(
        "--deleafing-pipe-like-long-side-frac",
        type=float,
        help="Reject deleafing primary masks whose long bbox side spans this image fraction.",
    )
    parser.add_argument(
        "--deleafing-pipe-like-aspect-ratio",
        type=float,
        help="Reject deleafing primary long thin pipe-like masks above this bbox aspect ratio.",
    )
    parser.add_argument(
        "--disable-deleafing-primary-axis-filter",
        action="store_true",
        help="Disable primary deleafing principal-axis rejection for main-stem-like masks.",
    )
    parser.add_argument(
        "--deleafing-primary-vertical-axis-thres",
        type=float,
        help="Reject thin primary deleafing masks aligned with this configured vertical-axis score.",
    )
    parser.add_argument(
        "--deleafing-primary-vertical-axis-max-thinness",
        type=float,
        help="Only apply primary deleafing vertical-axis rejection to masks at or below this thinness.",
    )
    parser.add_argument(
        "--deleafing-primary-vertical-axis-image-direction",
        choices=("image_vertical", "image_horizontal"),
        help=(
            "Image direction treated as real vertical for deleafing primary-axis rejection. "
            "Use image_horizontal if your exported Aria frames are rotated by 90 degrees."
        ),
    )
    parser.add_argument("--aux-min-mask-area-px", type=int)
    parser.add_argument(
        "--deleafing-aux-min-mask-area-px",
        type=int,
        help="Minimum auxiliary main-stem mask area for deleafing.",
    )
    parser.add_argument("--max-mask-area-frac", type=float)
    parser.add_argument(
        "--petiole-min-points-inside-mask",
        type=int,
        help="Minimum projected MPS points inside a deleafing mask before lifting.",
    )
    parser.add_argument(
        "--petiole-dbscan-min-samples",
        type=int,
        help="DBSCAN min_samples for sparse petiole 3D lifting.",
    )
    parser.add_argument(
        "--petiole-instance-min-frames",
        type=int,
        help="Minimum number of frames required for a fused deleafing instance.",
    )
    parser.add_argument(
        "--petiole-min-3d-major-extent-m",
        type=float,
        help="Minimum 3D PCA major-axis extent for a lifted/fused deleafing petiole.",
    )
    parser.add_argument(
        "--petiole-max-3d-major-extent-m",
        type=float,
        help="Maximum 3D PCA major-axis extent for a lifted/fused deleafing petiole.",
    )
    parser.add_argument(
        "--deleafing-main-stem-min-verticality",
        type=float,
        help="Reject deleafing auxiliary main-stem masks whose image principal axis is less vertical than this.",
    )
    parser.add_argument(
        "--deleafing-main-stem-min-height-frac",
        type=float,
        help="Reject deleafing auxiliary main-stem masks whose image bbox is too short.",
    )
    parser.add_argument(
        "--deleafing-main-stem-max-width-frac",
        type=float,
        help="Reject deleafing auxiliary main-stem masks whose image bbox is too wide.",
    )
    parser.add_argument(
        "--deleafing-main-stem-min-aspect-ratio",
        type=float,
        help="Reject deleafing auxiliary main-stem masks that are not tall enough.",
    )
    parser.add_argument(
        "--deleafing-aux-min-vertical-extent-m",
        type=float,
        help="Reject deleafing auxiliary 3D instances whose extent along world-up is too small.",
    )
    parser.add_argument(
        "--deleafing-aux-min-vertical-ratio",
        type=float,
        help="Reject deleafing auxiliary 3D instances that are not vertical-stem-like.",
    )
    parser.add_argument(
        "--deleafing-max-aux-masks-per-frame",
        type=int,
        help="Cap auxiliary main-stem masks kept per frame for deleafing.",
    )
    parser.add_argument("--depth-band-m", type=float)
    parser.add_argument("--instance-eps-m", type=float)
    parser.add_argument("--match-radius-m", type=float)
    parser.add_argument("--min-visible-post-views", type=int)
    parser.add_argument("--disappeared-score-thres", type=float)
    parser.add_argument(
        "--change-score-mode",
        choices=("auto", "average", "sam_absence_only", "point_disappearance_only"),
        default="auto",
        help=(
            "How to combine SAM absence and 3D point disappearance. In auto mode, "
            "shared extracted trajectories use SAM absence only because pre/post share one point map."
        ),
    )
    parser.add_argument("--world-up", default="0,0,1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--skip-pipeline-visualizations",
        action="store_true",
        help=(
            "Skip per-scene debug PLY/PNG/summary visualization writing. This is useful for "
            "large benchmark runs where evaluation overlays are generated separately."
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        task=args.task,
        sam_backend=args.sam_backend,
        use_video_mode=args.use_video_mode,
        rgb_stream_id=args.rgb_stream_id,
        frame_stride=args.frame_stride,
        sam_batch_size=args.sam_batch_size,
        use_undistorted_rgb=True,
        world_up=parse_world_up(args.world_up),
        change_score_mode=args.change_score_mode,
    )
    for arg_name, field_name in (
        ("sam_score_thres", "sam_score_thres"),
        ("deleafing_sam_score_thres", "deleafing_sam_score_thres"),
        ("aux_sam_score_thres", "aux_sam_score_thres"),
        ("deleafing_aux_sam_score_thres", "deleafing_aux_sam_score_thres"),
        ("min_mask_area_px", "min_mask_area_px"),
        ("deleafing_min_mask_area_px", "deleafing_min_mask_area_px"),
        ("deleafing_max_mask_area_frac", "deleafing_max_mask_area_frac"),
        ("deleafing_max_primary_masks_per_frame", "deleafing_max_primary_masks_per_frame"),
        ("deleafing_pipe_like_long_side_frac", "deleafing_pipe_like_long_side_frac"),
        ("deleafing_pipe_like_aspect_ratio", "deleafing_pipe_like_aspect_ratio"),
        ("deleafing_primary_vertical_axis_thres", "deleafing_primary_vertical_axis_thres"),
        (
            "deleafing_primary_vertical_axis_max_thinness",
            "deleafing_primary_vertical_axis_max_thinness",
        ),
        (
            "deleafing_primary_vertical_axis_image_direction",
            "deleafing_primary_vertical_axis_image_direction",
        ),
        ("aux_min_mask_area_px", "aux_min_mask_area_px"),
        ("deleafing_aux_min_mask_area_px", "deleafing_aux_min_mask_area_px"),
        ("max_mask_area_frac", "max_mask_area_frac"),
        ("petiole_min_points_inside_mask", "petiole_min_points_inside_mask"),
        ("petiole_dbscan_min_samples", "petiole_dbscan_min_samples"),
        ("petiole_instance_min_frames", "petiole_instance_min_frames"),
        ("petiole_min_3d_major_extent_m", "petiole_min_3d_major_extent_m"),
        ("petiole_max_3d_major_extent_m", "petiole_max_3d_major_extent_m"),
        ("deleafing_main_stem_min_verticality", "deleafing_main_stem_min_verticality"),
        ("deleafing_main_stem_min_height_frac", "deleafing_main_stem_min_height_frac"),
        ("deleafing_main_stem_max_width_frac", "deleafing_main_stem_max_width_frac"),
        ("deleafing_main_stem_min_aspect_ratio", "deleafing_main_stem_min_aspect_ratio"),
        ("deleafing_aux_min_vertical_extent_m", "deleafing_aux_min_vertical_extent_m"),
        ("deleafing_aux_min_vertical_ratio", "deleafing_aux_min_vertical_ratio"),
        ("deleafing_max_aux_masks_per_frame", "deleafing_max_aux_masks_per_frame"),
        ("depth_band_m", "depth_band_m"),
        ("instance_eps_m", "instance_eps_m"),
        ("match_radius_m", "match_radius_m"),
        ("min_visible_post_views", "min_visible_post_views"),
        ("disappeared_score_thres", "disappeared_score_thres"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field_name, value)
    if args.task == "deleafing":
        if args.sam_score_thres is not None and args.deleafing_sam_score_thres is None:
            config.deleafing_sam_score_thres = args.sam_score_thres
        if args.min_mask_area_px is not None and args.deleafing_min_mask_area_px is None:
            config.deleafing_min_mask_area_px = args.min_mask_area_px
        if args.aux_sam_score_thres is not None and args.deleafing_aux_sam_score_thres is None:
            config.deleafing_aux_sam_score_thres = args.aux_sam_score_thres
        if args.aux_min_mask_area_px is not None and args.deleafing_aux_min_mask_area_px is None:
            config.deleafing_aux_min_mask_area_px = args.aux_min_mask_area_px
        if args.disappeared_score_thres is not None:
            config.deleafing_disappeared_score_thres = args.disappeared_score_thres
        if args.disable_deleafing_primary_axis_filter:
            config.deleafing_primary_reject_vertical_axis = False
    config.validate()
    return config


def _validate_mode_args(args: argparse.Namespace) -> None:
    required = ("pair_manifest",)
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments for {args.mode}: {', '.join('--' + m.replace('_', '-') for m in missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_mode_args(args)
    config = _config_from_args(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.use_video_mode:
        print("Warning: video tracking mode is reserved for a backend-specific adapter; using image mode.")

    try:
        backend = create_sam_backend(config.sam_backend)
    except BackendUnavailableError as exc:
        print(str(exc))
        return 2

    warnings: list[str] = []
    pre_frames_json, post_frames_json, pre_frames, post_frames, extracted_meta, extracted_warnings = (
        prepare_frames_from_pair_manifest(
            args.pair_manifest,
            out_dir,
            frame_stride=config.frame_stride,
        )
    )
    warnings.extend(extracted_warnings)
    manifest = extracted_meta["manifest"]
    scene_frame_id = str(Path(args.pair_manifest).parents[1].name)
    pre_geometry = AriaMpsGeometry.from_pinhole_files(
        source_root=extracted_meta["pre_source_root"],
        points_source=extracted_meta["pre_points_source"],
        camera_json=extracted_meta["pre_camera_json"],
        poses_csv=extracted_meta["pre_poses_csv"],
        config=config,
        world_frame_id=scene_frame_id,
    )
    post_geometry = AriaMpsGeometry.from_pinhole_files(
        source_root=extracted_meta["post_source_root"],
        points_source=extracted_meta["post_points_source"],
        camera_json=extracted_meta["post_camera_json"],
        poses_csv=extracted_meta["post_poses_csv"],
        config=config,
        world_frame_id=scene_frame_id,
    )
    ensure_same_world_frame(pre_geometry, post_geometry)
    point_disappearance_reliable = (
        not bool(manifest.get("single_trajectory", False))
        and extracted_meta["pre_points_source"] != extracted_meta["post_points_source"]
    )
    warnings.append(
        "Using already-extracted pinhole frames from pair manifest; no VRS frame extraction was run."
    )
    if not point_disappearance_reliable and config.change_score_mode == "auto":
        warnings.append(
            "Extracted pair uses a shared trajectory/point source, so auto change scoring "
            "uses SAM absence only instead of averaging with 3D point disappearance."
        )

    warnings.extend(pre_geometry.warnings)
    if post_geometry is not pre_geometry:
        warnings.extend(post_geometry.warnings)
    raw_pre_json, raw_post_json, _ = run_sam_inference(
        pre_frames_json=pre_frames_json,
        post_frames_json=post_frames_json,
        task=config.task,
        backend=backend,
        config=config,
        out_dir=out_dir,
        use_aux_prompts=args.use_aux_prompts,
        exemplar_config_path=args.exemplar_config,
        save_visualizations=not args.skip_pipeline_visualizations,
    )
    filtered_pre_json = out_dir / "sam_predictions" / "pre_predictions_filtered.json"
    filtered_post_json = out_dir / "sam_predictions" / "post_predictions_filtered.json"
    filtered_pre, _ = postprocess_prediction_file(
        raw_pre_json,
        filtered_pre_json,
        config,
        config.task,
        save_visualizations=not args.skip_pipeline_visualizations,
    )
    filtered_post, _ = postprocess_prediction_file(
        raw_post_json,
        filtered_post_json,
        config,
        config.task,
        save_visualizations=not args.skip_pipeline_visualizations,
    )

    pre_lifted_json, post_lifted_json, _ = run_mask_lifting(
        pre_predictions_json=filtered_pre_json,
        post_predictions_json=filtered_post_json,
        pre_frames_json=pre_frames_json,
        post_frames_json=post_frames_json,
        pre_geometry=pre_geometry,
        post_geometry=post_geometry,
        config=config,
        task=config.task,
        out_dir=out_dir,
        debug=not args.skip_pipeline_visualizations,
    )
    pre_instances_json, post_instances_json, _ = fuse_prepost(
        pre_lifted_json=pre_lifted_json,
        post_lifted_json=post_lifted_json,
        task=config.task,
        config=config,
        out_dir=out_dir,
    )
    change_json, change_candidates, _ = run_prepost_change(
        pre_instances_json=pre_instances_json,
        post_instances_json=post_instances_json,
        post_frames_json=post_frames_json,
        post_geometry=post_geometry,
        post_predictions_json=filtered_post_json,
        config=config,
        task=config.task,
        out_dir=out_dir,
        point_disappearance_reliable=point_disappearance_reliable,
    )
    final_json, final_output, final_stats = run_target_rules(
        pre_instances_json=pre_instances_json,
        change_candidates_json=change_json,
        config=config,
        task=config.task,
        out_dir=out_dir,
        method_metadata={"sam_backend": config.sam_backend, "mode": args.mode},
    )

    raw_pre = _read_json(raw_pre_json)
    raw_post = _read_json(raw_post_json)
    lifted_pre = _read_json(pre_lifted_json)
    lifted_post = _read_json(post_lifted_json)
    pre_instances = _read_json(pre_instances_json)
    post_instances = _read_json(post_instances_json)

    if lifted_pre or lifted_post:
        no_support = sum(
            1
            for item in lifted_pre + lifted_post
            if str(item.get("status", "")).startswith("no_3d_support")
        )
        if no_support > 0.5 * max(1, len(lifted_pre) + len(lifted_post)):
            warnings.append(
                "Many SAM masks have zero Aria MPS support. The semi-dense point cloud may be sparse on the target object."
            )
    if _count_masks(raw_post) == 0:
        if config.task == "deleafing":
            warnings.append(
                "Post SAM produced zero masks. If petioles should still be visible, rerun with "
                "a lower --deleafing-sam-score-thres such as 0.05 and "
                "--deleafing-min-mask-area-px 10."
            )
        else:
            warnings.append(
                "Post SAM produced zero masks. If other tomatoes should still be visible, rerun with "
                "a lower --sam-score-thres such as 0.05 to avoid treating all visible pre instances "
                "as absent."
            )

    debug_stats = {
        "num_pre_frames": int(len(pre_frames)),
        "num_post_frames": int(len(post_frames)),
        "num_pre_sam_masks_raw": _count_masks(raw_pre),
        "num_post_sam_masks_raw": _count_masks(raw_post),
        "num_pre_sam_masks_filtered": _count_masks(filtered_pre),
        "num_post_sam_masks_filtered": _count_masks(filtered_post),
        "num_pre_lifted_ok": int(
            sum(1 for item in lifted_pre if item.get("status") in ("ok", "dbscan_failed_kept_foreground"))
        ),
        "num_post_lifted_ok": int(
            sum(1 for item in lifted_post if item.get("status") in ("ok", "dbscan_failed_kept_foreground"))
        ),
        "num_pre_lifted_size_rejected": int(
            sum(1 for item in lifted_pre if str(item.get("status", "")).startswith("rejected_petiole_3d"))
        ),
        "num_post_lifted_size_rejected": int(
            sum(1 for item in lifted_post if str(item.get("status", "")).startswith("rejected_petiole_3d"))
        ),
        "num_pre_instances": int(len(pre_instances)),
        "num_post_instances": int(len(post_instances)),
        "num_disappeared_candidates": int(
            sum(1 for item in change_candidates if item.get("status") == "disappeared")
        ),
        "num_final_predictions": int(final_stats["num_final_predictions"]),
        "warnings": warnings,
        "per_prompt_stats": _per_prompt_stats(
            raw_pre=raw_pre,
            raw_post=raw_post,
            filtered_pre=filtered_pre,
            filtered_post=filtered_post,
            lifted_pre=lifted_pre,
            lifted_post=lifted_post,
        ),
        "pre_mps_point_stats": pre_geometry.point_stats,
        "post_mps_point_stats": post_geometry.point_stats,
    }
    debug_path = _save_debug_stats(out_dir, debug_stats)

    if args.skip_pipeline_visualizations:
        change_debug_stats = {"skipped": True}
        target_overlay_stats = {"skipped": True}
        change_debug_overlay_stats = {"skipped": True}
    else:
        write_visualization_plys(
            pre_instances=pre_instances,
            post_instances=post_instances,
            change_candidates=change_candidates,
            predictions=final_output.get("predictions", []),
            out_dir=out_dir,
        )
        change_debug_stats = write_change_debug_plys(
            pre_instances=pre_instances,
            post_instances=post_instances,
            change_candidates=change_candidates,
            predictions=final_output.get("predictions", []),
            out_dir=out_dir,
        )
        target_overlay_stats = {
            "pre": save_target_overlays_on_frames(
                frames=pre_frames,
                geometry=pre_geometry,
                predictions=final_output.get("predictions", []),
                split="pre",
                out_dir=out_dir,
            ),
            "post": save_target_overlays_on_frames(
                frames=post_frames,
                geometry=post_geometry,
                predictions=final_output.get("predictions", []),
                split="post",
                out_dir=out_dir,
            ),
        }
        change_debug_overlay_stats = {
            "pre": save_change_debug_overlays_on_frames(
                frames=pre_frames,
                geometry=pre_geometry,
                pre_instances=pre_instances,
                post_instances=post_instances,
                change_candidates=change_candidates,
                predictions=final_output.get("predictions", []),
                split="pre",
                out_dir=out_dir,
            ),
            "post": save_change_debug_overlays_on_frames(
                frames=post_frames,
                geometry=post_geometry,
                pre_instances=pre_instances,
                post_instances=post_instances,
                change_candidates=change_candidates,
                predictions=final_output.get("predictions", []),
                split="post",
                out_dir=out_dir,
            ),
        }
    debug_stats["change_debug_visualization"] = change_debug_stats
    debug_stats["change_debug_overlay_stats"] = change_debug_overlay_stats
    debug_stats["target_overlay_stats"] = target_overlay_stats
    debug_path = _save_debug_stats(out_dir, debug_stats)
    if not args.skip_pipeline_visualizations:
        save_summary_figure(
            task=config.task,
            summary=debug_stats,
            predictions=final_output.get("predictions", []),
            out_path=out_dir / "visualizations" / "summary.png",
        )
    # Ensure the final output points at the debug stats that now exists.
    final_output["debug_stats_path"] = str(debug_path.name)
    with Path(final_json).open("w") as handle:
        json.dump(final_output, handle, indent=2)

    _print_summary(debug_stats, config.task, config.sam_backend, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
