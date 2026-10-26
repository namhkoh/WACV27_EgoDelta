"""Command-line entrypoint for the YOLO + Aria MPS harvesting PoC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig, ensure_output_dirs, parse_csv_names, parse_world_up
from .task_config import apply_task_overrides, default_task_config_path, load_task_config


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    config = config_from_args(args)
    out_dir = Path(args.out)
    ensure_output_dirs(out_dir)
    if args.task == "harvest" and args.segmentation_backend in {None, "yolo_tomato"}:
        return run_harvest_pipeline(args, config, out_dir)
    task_config = effective_task_config(args, config)
    return run_generic_task_pipeline(args, config, task_config, out_dir)


def load_scene_context(
    args: argparse.Namespace,
    config: PipelineConfig,
    out_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], Any, Any, list[str]]:
    from .io_aria import (
        extract_frames_separate_recordings,
        extract_frames_single_recording,
        load_extracted_frames_from_pair_manifest,
    )
    from .mps_geometry import MPSGeometry, assert_same_world_frame

    warnings: list[str] = []
    if args.mode == "extracted_frames":
        frames_by_split = load_extracted_frames_from_pair_manifest(
            pair_manifest=args.pair_manifest,
            out_dir=out_dir,
            config=config,
        )
        pre_geometry = MPSGeometry.from_pair_manifest(args.pair_manifest, config)
        post_geometry = pre_geometry
        if args.point_disappearance_weight is None:
            config.point_disappearance_weight = 0.0
            warnings.append(
                "extracted_frames mode uses one shared point cloud for pre/post views, so the "
                "3D point disappearance score is disabled by default."
            )
    elif args.mode == "single_recording":
        frames_by_split = extract_frames_single_recording(
            vrs_path=args.vrs,
            out_dir=out_dir,
            config=config,
            pre_start_s=args.pre_start_s,
            pre_end_s=args.pre_end_s,
            post_start_s=args.post_start_s,
            post_end_s=args.post_end_s,
        )
        pre_geometry = MPSGeometry.from_paths(args.mps, args.vrs, config)
        post_geometry = pre_geometry
        if args.point_disappearance_weight is None:
            config.point_disappearance_weight = 0.0
            warnings.append(
                "single_recording mode uses one global MPS point cloud, so the 3D point "
                "disappearance score is disabled by default. Use --point-disappearance-weight "
                "to override this if you provide time-filtered post points."
            )
    else:
        frames_by_split = extract_frames_separate_recordings(
            pre_vrs_path=args.pre_vrs,
            post_vrs_path=args.post_vrs,
            out_dir=out_dir,
            config=config,
        )
        pre_geometry = MPSGeometry.from_paths(args.pre_mps, args.pre_vrs, config)
        post_geometry = MPSGeometry.from_paths(args.post_mps, args.post_vrs, config)
        warnings.extend(assert_same_world_frame(pre_geometry.metadata, post_geometry.metadata))

    warnings.extend(pre_geometry.point_cloud_stats.warnings)
    if post_geometry is not pre_geometry:
        warnings.extend(post_geometry.point_cloud_stats.warnings)
    return frames_by_split, pre_geometry, post_geometry, warnings


def run_harvest_pipeline(
    args: argparse.Namespace,
    config: PipelineConfig,
    out_dir: Path,
) -> int:
    if not args.model:
        raise SystemExit("--model is required for --task harvest")

    from .cutpoint_peduncle import build_final_predictions, estimate_cut_points_with_mode
    from .cutpoint_proxy import estimate_proxy_cut_points
    from .instance_fusion import fuse_instances_for_split
    from .mask_lifting import lift_detections_for_split
    from .peduncle_inference import run_peduncle_for_split
    from .peduncle_lifting import lift_peduncle_detections_for_split
    from .prepost_change import detect_disappeared_tomatoes
    from .result_projection import write_projected_result_overlays
    from .visualization import write_summary_image, write_visualization_plys
    from .yolo_inference import run_yolo_for_split

    frames_by_split, pre_geometry, post_geometry, warnings = load_scene_context(args, config, out_dir)

    pre_predictions = run_yolo_for_split("pre", frames_by_split["pre"], args.model, out_dir, config)
    post_predictions = run_yolo_for_split("post", frames_by_split["post"], args.model, out_dir, config)

    if count_detections(pre_predictions) == 0:
        warnings.append("No YOLO tomato detections were found in pre frames.")
    if count_detections(post_predictions) == 0:
        warnings.append("No YOLO tomato detections were found in post frames.")

    pre_lifted = lift_detections_for_split(
        "pre", frames_by_split["pre"], pre_predictions, pre_geometry, out_dir, config
    )
    post_lifted = lift_detections_for_split(
        "post", frames_by_split["post"], post_predictions, post_geometry, out_dir, config
    )
    if count_lifted_ok(pre_lifted) == 0 and count_detections(pre_predictions) > 0:
        warnings.append(
            "YOLO detected pre tomatoes, but no detections had enough Aria MPS 3D support."
        )
    if count_lifted_ok(post_lifted) == 0 and count_detections(post_predictions) > 0:
        warnings.append(
            "YOLO detected post tomatoes, but no detections had enough Aria MPS 3D support."
        )

    pre_instances = fuse_instances_for_split("pre", pre_lifted, out_dir, config)
    post_instances = fuse_instances_for_split("post", post_lifted, out_dir, config)

    peduncle_predictions: list[dict[str, Any]] = []
    peduncle_lifted: list[dict[str, Any]] = []
    if config.cutpoint_mode != "proxy":
        if args.peduncle_model:
            peduncle_predictions = run_peduncle_for_split(
                "pre", frames_by_split["pre"], args.peduncle_model, out_dir, config
            )
            peduncle_lifted = lift_peduncle_detections_for_split(
                "pre", frames_by_split["pre"], peduncle_predictions, pre_geometry, out_dir, config
            )
            if count_detections(peduncle_predictions) == 0:
                warnings.append("No peduncle detections were found in pre frames.")
            elif count_peduncle_lifted_ok(peduncle_lifted) == 0:
                warnings.append("Peduncle detections were found, but none had enough 3D support.")
        elif config.cutpoint_mode == "peduncle_auto":
            warnings.append(
                "--cutpoint-mode peduncle_auto requested without --peduncle-model; "
                "using proxy fallback."
            )

    disappearance_results = detect_disappeared_tomatoes(
        pre_instances=pre_instances,
        post_instances=post_instances,
        post_frames=frames_by_split["post"],
        post_predictions=post_predictions,
        post_geometry=post_geometry,
        out_dir=out_dir,
        config=config,
    )
    proxy_cut_points = estimate_proxy_cut_points(
        disappearance_results=disappearance_results,
        pre_instances=pre_instances,
        out_dir=out_dir,
        config=config,
    )
    if config.cutpoint_mode == "proxy":
        cut_points = proxy_cut_points
    else:
        cut_points = estimate_cut_points_with_mode(
            disappearance_results=disappearance_results,
            pre_instances=pre_instances,
            peduncle_lifted=peduncle_lifted,
            out_dir=out_dir,
            config=config,
        )
        if any(item.get("cut_point_3d_world") is None for item in cut_points):
            warnings.append(
                "Requested peduncle cut-point mode, but at least one disappeared tomato "
                "had no associated peduncle cut point."
            )
    final_predictions = build_final_predictions(out_dir.name, cut_points)
    (out_dir / "final_predictions.json").write_text(
        json.dumps(final_predictions, indent=2),
        encoding="utf-8",
    )

    write_visualization_plys(
        pre_instances=pre_instances,
        post_instances=post_instances,
        disappearance_results=disappearance_results,
        cut_points=cut_points,
        out_dir=out_dir,
    )
    write_summary_image(
        pre_instances=pre_instances,
        post_instances=post_instances,
        disappearance_results=disappearance_results,
        final_predictions=final_predictions,
        out_dir=out_dir,
    )
    try:
        write_projected_result_overlays(
            out_dir=out_dir,
            pre_geometry=pre_geometry,
            post_geometry=post_geometry,
            visualization_stride=config.visualization_stride,
        )
    except Exception as exc:
        warnings.append(f"Could not write projected RGB result overlays: {exc}")

    debug_stats = build_debug_stats(
        frames_by_split=frames_by_split,
        pre_predictions=pre_predictions,
        post_predictions=post_predictions,
        pre_lifted=pre_lifted,
        post_lifted=post_lifted,
        pre_instances=pre_instances,
        post_instances=post_instances,
        disappearance_results=disappearance_results,
        peduncle_predictions=peduncle_predictions,
        peduncle_lifted=peduncle_lifted,
        cut_points=cut_points,
        warnings=warnings,
        pre_geometry=pre_geometry,
        post_geometry=post_geometry,
        config=config,
    )
    (out_dir / "debug_stats.json").write_text(json.dumps(debug_stats, indent=2), encoding="utf-8")
    print_summary(debug_stats, final_predictions, out_dir)
    return 0


def run_generic_task_pipeline(
    args: argparse.Namespace,
    config: PipelineConfig,
    task_config: Any,
    out_dir: Path,
) -> int:
    from .fusion.instance_fusion import fuse_task_instances_for_split
    from .lifting.mask_lifting import lift_segmentation_for_split
    from .prepost_change import detect_disappeared_instances
    from .segmentation import make_segmentation_backend, run_segmentation_for_split
    from .tasks.deleaf import (
        build_main_stem_task_config,
        build_deleaf_final_predictions,
        estimate_deleaf_cut_points,
        estimate_main_stem_axis_from_instances,
    )
    from .cutpoint_peduncle import build_final_predictions as build_harvest_final_predictions
    from .cutpoint_proxy import estimate_proxy_cut_points
    from .result_projection import write_task_projected_result_overlays
    from .visualization import write_task_summary_image, write_task_visualization_plys

    frames_by_split, pre_geometry, post_geometry, warnings = load_scene_context(args, config, out_dir)
    backend = make_segmentation_backend(
        task_config.segmentation_backend,
        model_path=args.model,
        sam3_model=args.sam3_model,
        yolo_world_model=args.yolo_world_model,
        sam_model=args.sam_model,
        yoloe_model=args.yoloe_model,
    )

    pre_predictions = run_segmentation_for_split(
        "pre", frames_by_split["pre"], backend, task_config, config, out_dir
    )
    post_predictions = run_segmentation_for_split(
        "post", frames_by_split["post"], backend, task_config, config, out_dir
    )
    if count_detections(pre_predictions) == 0:
        warnings.append("No promptable masks found. Try different prompts or yolo_world_sam backend.")
    if count_detections(post_predictions) == 0:
        warnings.append("No promptable masks found in post frames.")

    pre_lifted = lift_segmentation_for_split(
        "pre", frames_by_split["pre"], pre_predictions, pre_geometry, out_dir, task_config, config
    )
    post_lifted = lift_segmentation_for_split(
        "post", frames_by_split["post"], post_predictions, post_geometry, out_dir, task_config, config
    )
    if count_lifted_ok(pre_lifted) == 0 and count_detections(pre_predictions) > 0:
        warnings.append(
            f"{task_config.target_semantic_class} masks were found, but none had enough MPS 3D support."
        )
    if count_lifted_ok(post_lifted) == 0 and count_detections(post_predictions) > 0:
        warnings.append(
            f"Post {task_config.target_semantic_class} masks were found, but none had enough MPS 3D support."
        )

    pre_instances = fuse_task_instances_for_split("pre", pre_lifted, out_dir, task_config)
    post_instances = fuse_task_instances_for_split("post", post_lifted, out_dir, task_config)
    disappearance_results = detect_disappeared_instances(
        task_config=task_config,
        pre_instances=pre_instances,
        post_instances=post_instances,
        post_frames=frames_by_split["post"],
        post_predictions=post_predictions,
        post_geometry=post_geometry,
        out_dir=out_dir,
        point_disappearance_weight=config.point_disappearance_weight,
    )

    main_stem_axis = None
    if task_config.task_name == "harvest":
        action_points = estimate_proxy_cut_points(
            disappearance_results=disappearance_results,
            pre_instances=pre_instances,
            out_dir=out_dir,
            config=config,
        )
        final_predictions = build_harvest_final_predictions(out_dir.name, action_points)
        final_path = out_dir / "final_predictions.json"
    elif task_config.task_name == "deleaf":
        stem_config = build_main_stem_task_config(task_config)
        stem_predictions = run_segmentation_for_split(
            "pre", frames_by_split["pre"], backend, stem_config, config, out_dir
        )
        stem_lifted = lift_segmentation_for_split(
            "pre", frames_by_split["pre"], stem_predictions, pre_geometry, out_dir, stem_config, config
        )
        stem_instances = fuse_task_instances_for_split("pre", stem_lifted, out_dir, stem_config)
        main_stem_axis = estimate_main_stem_axis_from_instances(stem_instances, out_dir)
        if main_stem_axis is None:
            warnings.append(
                "No promptable main-stem 3D axis was available; using petiole geometry fallback."
            )
        action_points = estimate_deleaf_cut_points(
            disappearance_results=disappearance_results,
            pre_instances=pre_instances,
            out_dir=out_dir,
            task_config=task_config,
            main_stem_axis=main_stem_axis,
        )
        final_predictions = build_deleaf_final_predictions(out_dir.name, action_points)
        final_path = out_dir / "final_predictions_deleaf.json"
    else:
        raise ValueError(f"Unsupported generic task: {task_config.task_name}")

    final_path.write_text(json.dumps(final_predictions, indent=2), encoding="utf-8")
    write_task_visualization_plys(
        task_name=task_config.task_name,
        target_semantic_class=task_config.target_semantic_class,
        pre_instances=pre_instances,
        post_instances=post_instances,
        disappearance_results=disappearance_results,
        action_points=action_points,
        out_dir=out_dir,
        main_stem_axis=main_stem_axis,
    )
    write_task_summary_image(
        task_name=task_config.task_name,
        target_semantic_class=task_config.target_semantic_class,
        pre_instances=pre_instances,
        post_instances=post_instances,
        disappearance_results=disappearance_results,
        final_predictions=final_predictions,
        out_dir=out_dir,
    )
    try:
        write_task_projected_result_overlays(
            out_dir=out_dir,
            task_name=task_config.task_name,
            pre_geometry=pre_geometry,
            post_geometry=post_geometry,
            visualization_stride=config.visualization_stride,
        )
    except Exception as exc:
        warnings.append(f"Could not write projected task result overlays: {exc}")

    debug_stats = build_task_debug_stats(
        task_config=task_config,
        frames_by_split=frames_by_split,
        pre_predictions=pre_predictions,
        post_predictions=post_predictions,
        pre_lifted=pre_lifted,
        post_lifted=post_lifted,
        pre_instances=pre_instances,
        post_instances=post_instances,
        disappearance_results=disappearance_results,
        action_points=action_points,
        warnings=warnings,
        pre_geometry=pre_geometry,
        post_geometry=post_geometry,
        config=config,
    )
    (out_dir / "debug_stats.json").write_text(json.dumps(debug_stats, indent=2), encoding="utf-8")
    print_task_summary(debug_stats, out_dir)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aria MPS pre/post change detection for tomato harvesting and deleafing"
    )
    parser.add_argument("--task", choices=["harvest", "deleaf"], default="harvest")
    parser.add_argument(
        "--mode",
        choices=["single_recording", "separate_recordings", "extracted_frames"],
        required=True,
    )
    parser.add_argument("--model", help="Path to tomato YOLO .pt for harvest/yolo_tomato")
    parser.add_argument("--out", required=True, help="Output scene directory")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--world-up", default="0,0,1")
    parser.add_argument(
        "--image-view",
        choices=["raw", "egodelta_pinhole", "pinhole"],
        default="raw",
        help="Camera view used for extracted frames and MPS projection.",
    )
    parser.add_argument(
        "--aria-camera-label",
        default="camera-rgb",
        help="Project Aria camera calibration label used for EgoDelta pinhole rectification.",
    )
    parser.add_argument("--rect-size", type=int, default=720)
    parser.add_argument("--rect-focal", type=float, default=380.0)
    parser.add_argument("--task-config", help="Task YAML. Defaults to configs/<task>.yaml")
    parser.add_argument(
        "--segmentation-backend",
        choices=["yolo_tomato", "sam3_concept", "yolo_world_sam", "yoloe_prompt"],
        help="Override task config segmentation backend",
    )
    parser.add_argument("--sam3-model", default="sam3.pt")
    parser.add_argument("--yolo-world-model", default="yolov8l-world.pt")
    parser.add_argument("--sam-model", default="sam2_b.pt")
    parser.add_argument("--yoloe-model", default="yoloe-11s-seg.pt")

    parser.add_argument("--vrs")
    parser.add_argument("--mps")
    parser.add_argument("--pre-start-s", type=float)
    parser.add_argument("--pre-end-s", type=float)
    parser.add_argument("--post-start-s", type=float)
    parser.add_argument("--post-end-s", type=float)

    parser.add_argument("--pre-vrs")
    parser.add_argument("--post-vrs")
    parser.add_argument("--pre-mps")
    parser.add_argument("--post-mps")
    parser.add_argument(
        "--pair-manifest",
        help="Manifest for already-extracted pre/post frames and COLMAP geometry.",
    )

    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=1,
        help="Batch size for segmentation backends that support batched inference, such as YOLOE.",
    )
    parser.add_argument(
        "--visualization-stride",
        type=int,
        default=30,
        help=(
            "Write per-frame visualization images every N sampled frames. This is separate from "
            "--frame-stride and does not affect inference, lifting, fusion, or evaluation."
        ),
    )
    parser.add_argument("--no-retina-masks", action="store_true")
    parser.add_argument("--inv-dist-std-thres", type=float, default=0.005)
    parser.add_argument("--dist-std-thres", type=float, default=0.01)
    parser.add_argument("--min-points-inside-mask", type=int, default=10)
    parser.add_argument("--depth-band-m", type=float, default=0.08)
    parser.add_argument("--tomato-dbscan-eps-m", type=float, default=0.05)
    parser.add_argument("--tomato-dbscan-min-samples", type=int, default=5)
    parser.add_argument("--instance-eps-m", type=float, default=0.10)
    parser.add_argument("--instance-min-frames", type=int, default=2)
    parser.add_argument("--instance-min-points", type=int, default=20)
    parser.add_argument("--match-radius-m", type=float, default=0.12)
    parser.add_argument("--min-visible-post-views", type=int, default=3)
    parser.add_argument("--point-match-radius-m", type=float, default=0.04)
    parser.add_argument(
        "--point-disappearance-weight",
        type=float,
        default=None,
        help=(
            "Weight for the 3D point disappearance score. Defaults to 0.5 for "
            "separate_recordings and 0.0 for single_recording because a single MPS "
            "semi-dense cloud is usually global across both time windows."
        ),
    )
    parser.add_argument("--disappeared-score-thres", type=float, default=0.6)
    parser.add_argument("--top-quantile", type=float, default=0.75)
    parser.add_argument("--cut-offset-m", type=float, default=0.02)
    parser.add_argument(
        "--cutpoint-mode",
        choices=["proxy", "peduncle_keypoint", "peduncle_mask", "peduncle_auto"],
        default="proxy",
        help=(
            "How to estimate harvesting cut points. proxy keeps the original tomato-top "
            "heuristic. peduncle_keypoint uses associated 3D peduncle keypoints. "
            "peduncle_mask fits a 3D curve to associated peduncle mask points. "
            "peduncle_auto tries keypoints, then masks, then proxy fallback."
        ),
    )
    parser.add_argument("--peduncle-model", help="Optional Ultralytics peduncle pose/segmentation model")
    parser.add_argument("--peduncle-conf-thres", type=float, default=0.25)
    parser.add_argument("--peduncle-imgsz", type=int, default=640)
    parser.add_argument(
        "--peduncle-keypoint-names",
        default="fruit_attachment,peduncle_mid,stem_attachment,cut_point",
        help="Comma-separated names for peduncle pose keypoints in model output order",
    )
    parser.add_argument("--peduncle-keypoint-lift-radius-px", type=float, default=8.0)
    parser.add_argument("--peduncle-depth-band-m", type=float, default=0.05)
    parser.add_argument("--peduncle-min-points", type=int, default=5)
    parser.add_argument("--peduncle-dbscan-eps-m", type=float, default=0.03)
    parser.add_argument("--peduncle-dbscan-min-samples", type=int, default=3)
    parser.add_argument("--peduncle-association-max-distance-m", type=float, default=0.15)
    parser.add_argument("--peduncle-cut-offset-m", type=float, default=0.02)
    parser.add_argument("--debug", action="store_true", help="Reserved for verbose debugging")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "extracted_frames":
        if args.pair_manifest is None:
            parser.error("extracted_frames mode requires --pair-manifest")
    elif args.mode == "single_recording":
        missing = [
            name
            for name in ["vrs", "mps", "pre_start_s", "pre_end_s", "post_start_s", "post_end_s"]
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"single_recording mode requires: {', '.join('--' + n.replace('_', '-') for n in missing)}")
    if args.mode == "separate_recordings":
        missing = [
            name
            for name in ["pre_vrs", "post_vrs", "pre_mps", "post_mps"]
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                f"separate_recordings mode requires: {', '.join('--' + n.replace('_', '-') for n in missing)}"
            )
    if args.cutpoint_mode in {"peduncle_keypoint", "peduncle_mask"} and not args.peduncle_model:
        parser.error(f"--cutpoint-mode {args.cutpoint_mode} requires --peduncle-model")
    if args.task == "harvest" and args.segmentation_backend in {None, "yolo_tomato"} and not args.model:
        parser.error("--task harvest requires --model")
    if args.image_view == "egodelta_pinhole":
        if args.rect_size <= 0:
            parser.error("--rect-size must be positive")
        if args.rect_focal <= 0:
            parser.error("--rect-focal must be positive")
    if args.mode == "extracted_frames" and args.image_view != "pinhole":
        args.image_view = "pinhole"
    if args.mode != "extracted_frames" and args.image_view == "pinhole":
        parser.error("--image-view pinhole is only valid with --mode extracted_frames")
    if args.segmentation_batch_size < 1:
        parser.error("--segmentation-batch-size must be >= 1")
    if args.visualization_stride < 1:
        parser.error("--visualization-stride must be >= 1")


def effective_task_config(args: argparse.Namespace, config: PipelineConfig) -> Any:
    path = Path(args.task_config) if args.task_config else default_task_config_path(args.task)
    task_config = load_task_config(path)
    if task_config.task_name != args.task:
        raise ValueError(f"Task config {path} has task_name={task_config.task_name}, expected {args.task}")
    backend = args.segmentation_backend or task_config.segmentation_backend
    return apply_task_overrides(
        task_config,
        segmentation_backend=backend,
        world_up=config.world_up,
    )


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        conf_thres=args.conf_thres,
        imgsz=args.imgsz,
        retina_masks=not args.no_retina_masks,
        max_det=args.max_det,
        segmentation_batch_size=args.segmentation_batch_size,
        visualization_stride=args.visualization_stride,
        frame_stride=args.frame_stride,
        rgb_stream_id=args.rgb_stream_id,
        image_view=args.image_view,
        aria_camera_label=args.aria_camera_label,
        rect_size=args.rect_size,
        rect_focal=args.rect_focal,
        inv_dist_std_thres=args.inv_dist_std_thres,
        dist_std_thres=args.dist_std_thres,
        min_points_inside_mask=args.min_points_inside_mask,
        tomato_depth_band_m=args.depth_band_m,
        tomato_dbscan_eps_m=args.tomato_dbscan_eps_m,
        tomato_dbscan_min_samples=args.tomato_dbscan_min_samples,
        instance_cluster_eps_m=args.instance_eps_m,
        instance_min_frames=args.instance_min_frames,
        instance_min_points=args.instance_min_points,
        prepost_match_radius_m=args.match_radius_m,
        min_visible_post_views=args.min_visible_post_views,
        point_match_radius_m=args.point_match_radius_m,
        point_disappearance_weight=(
            0.5 if args.point_disappearance_weight is None else args.point_disappearance_weight
        ),
        disappeared_score_thres=args.disappeared_score_thres,
        world_up=parse_world_up(args.world_up),
        top_quantile=args.top_quantile,
        cut_offset_m=args.cut_offset_m,
        cutpoint_mode=args.cutpoint_mode,
        peduncle_conf_thres=args.peduncle_conf_thres,
        peduncle_imgsz=args.peduncle_imgsz,
        peduncle_keypoint_names=parse_csv_names(args.peduncle_keypoint_names),
        peduncle_keypoint_lift_radius_px=args.peduncle_keypoint_lift_radius_px,
        peduncle_depth_band_m=args.peduncle_depth_band_m,
        peduncle_min_points=args.peduncle_min_points,
        peduncle_dbscan_eps_m=args.peduncle_dbscan_eps_m,
        peduncle_dbscan_min_samples=args.peduncle_dbscan_min_samples,
        peduncle_association_max_distance_m=args.peduncle_association_max_distance_m,
        peduncle_cut_offset_m=args.peduncle_cut_offset_m,
    )


def build_debug_stats(
    frames_by_split: dict[str, list[dict[str, Any]]],
    pre_predictions: list[dict[str, Any]],
    post_predictions: list[dict[str, Any]],
    pre_lifted: list[dict[str, Any]],
    post_lifted: list[dict[str, Any]],
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    disappearance_results: list[dict[str, Any]],
    peduncle_predictions: list[dict[str, Any]],
    peduncle_lifted: list[dict[str, Any]],
    cut_points: list[dict[str, Any]],
    warnings: list[str],
    pre_geometry: Any,
    post_geometry: Any,
    config: PipelineConfig,
) -> dict[str, Any]:
    unique_warnings = list(dict.fromkeys(warnings))
    return {
        "num_pre_frames": len(frames_by_split["pre"]),
        "num_post_frames": len(frames_by_split["post"]),
        "num_pre_yolo_detections": count_detections(pre_predictions),
        "num_post_yolo_detections": count_detections(post_predictions),
        "num_pre_lifted_ok": count_lifted_ok(pre_lifted),
        "num_post_lifted_ok": count_lifted_ok(post_lifted),
        "num_pre_instances": len(pre_instances),
        "num_post_instances": len(post_instances),
        "num_disappeared": sum(1 for item in disappearance_results if item.get("status") == "disappeared"),
        "cutpoint_mode": config.cutpoint_mode,
        "num_pre_peduncle_detections": count_detections(peduncle_predictions),
        "num_pre_peduncle_lifted_ok": count_peduncle_lifted_ok(peduncle_lifted),
        "num_cut_points": sum(1 for item in cut_points if item.get("cut_point_3d_world") is not None),
        "warnings": unique_warnings,
        "pre_point_cloud_stats": pre_geometry.point_cloud_stats.to_dict(),
        "post_point_cloud_stats": post_geometry.point_cloud_stats.to_dict(),
        "config": config.to_dict(),
    }


def build_task_debug_stats(
    task_config: Any,
    frames_by_split: dict[str, list[dict[str, Any]]],
    pre_predictions: list[dict[str, Any]],
    post_predictions: list[dict[str, Any]],
    pre_lifted: list[dict[str, Any]],
    post_lifted: list[dict[str, Any]],
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    disappearance_results: list[dict[str, Any]],
    action_points: list[dict[str, Any]],
    warnings: list[str],
    pre_geometry: Any,
    post_geometry: Any,
    config: PipelineConfig,
) -> dict[str, Any]:
    unique_warnings = list(dict.fromkeys(warnings))
    return {
        "task": task_config.task_name,
        "num_pre_frames": len(frames_by_split["pre"]),
        "num_post_frames": len(frames_by_split["post"]),
        "segmentation_backend": task_config.segmentation_backend,
        "text_prompts": task_config.text_prompts,
        "negative_prompts": task_config.negative_prompts,
        "num_pre_2d_detections": count_detections(pre_predictions),
        "num_post_2d_detections": count_detections(post_predictions),
        "num_pre_lifted_ok": count_lifted_ok(pre_lifted),
        "num_post_lifted_ok": count_lifted_ok(post_lifted),
        "num_pre_instances": len(pre_instances),
        "num_post_instances": len(post_instances),
        "num_disappeared_candidates": sum(
            1 for item in disappearance_results if item.get("status") == "disappeared"
        ),
        "num_final_removed": len(action_points),
        "warnings": unique_warnings,
        "pre_point_cloud_stats": pre_geometry.point_cloud_stats.to_dict(),
        "post_point_cloud_stats": post_geometry.point_cloud_stats.to_dict(),
        "task_config": task_config.to_dict(),
        "pipeline_config": config.to_dict(),
    }


def count_detections(predictions: list[dict[str, Any]]) -> int:
    return int(sum(len(item.get("detections", [])) for item in predictions))


def count_lifted_ok(lifted: list[dict[str, Any]]) -> int:
    return int(sum(1 for item in lifted if item.get("status") == "ok"))


def count_peduncle_lifted_ok(lifted: list[dict[str, Any]]) -> int:
    return int(sum(1 for item in lifted if item.get("status") == "ok"))


def print_summary(debug_stats: dict[str, Any], final_predictions: dict[str, Any], out_dir: Path) -> None:
    print("===== YOLO + Aria MPS Harvesting PoC Summary =====")
    print(f"Pre frames: {debug_stats['num_pre_frames']}")
    print(f"Post frames: {debug_stats['num_post_frames']}")
    print(f"Pre YOLO tomato detections: {debug_stats['num_pre_yolo_detections']}")
    print(f"Post YOLO tomato detections: {debug_stats['num_post_yolo_detections']}")
    print(f"Pre 3D tomato instances: {debug_stats['num_pre_instances']}")
    print(f"Post 3D tomato instances: {debug_stats['num_post_instances']}")
    print(f"Disappeared tomato candidates: {debug_stats['num_disappeared']}")
    print(f"Final removed tomatoes: {len(final_predictions.get('removed_tomatoes', []))}")
    if debug_stats.get("warnings"):
        print("Warnings:")
        for warning in debug_stats["warnings"]:
            print(f"  - {warning}")
    print(f"Outputs saved to: {out_dir}")


def print_task_summary(debug_stats: dict[str, Any], out_dir: Path) -> None:
    print("===== Aria Pre/Post Change Detection Summary =====")
    print(f"Task: {debug_stats['task']}")
    print(f"Segmentation backend: {debug_stats['segmentation_backend']}")
    print(f"Pre frames: {debug_stats['num_pre_frames']}")
    print(f"Post frames: {debug_stats['num_post_frames']}")
    print(f"Pre 2D detections: {debug_stats['num_pre_2d_detections']}")
    print(f"Post 2D detections: {debug_stats['num_post_2d_detections']}")
    print(f"Pre 3D instances: {debug_stats['num_pre_instances']}")
    print(f"Post 3D instances: {debug_stats['num_post_instances']}")
    print(f"Disappeared candidates: {debug_stats['num_disappeared_candidates']}")
    print(f"Final removed targets: {debug_stats['num_final_removed']}")
    if debug_stats.get("warnings"):
        print("Warnings:")
        for warning in debug_stats["warnings"]:
            print(f"  - {warning}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
