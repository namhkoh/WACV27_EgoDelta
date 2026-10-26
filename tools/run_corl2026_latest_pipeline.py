#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = ("alignment_qc", "score", "optimize", "prune", "target", "visualize")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the current CoRL2026 EgoDelta leaf pipeline.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pre_model", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--object_type", choices=("leaf", "clip", "tomato"), default="leaf")
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--post_frame_list", default=None)
    parser.add_argument("--resolution", default=2, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--selected_clusters", default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pre_dataset_dir", default=None, help="Pre-scene COLMAP dataset for projection visualization.")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated subset of alignment_qc,score,optimize,prune,target,visualize.",
    )
    parser.add_argument("--alignment_qc_views", default=8, type=int)
    parser.add_argument("--alignment_qc_depth_point_stride", default=4, type=int)
    parser.add_argument("--alignment_qc_point_draw_stride", default=12, type=int)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def add_common(cmd, args):
    cmd.extend(["--task", args.task, "--data_root", args.data_root])
    if args.single_trajectory:
        cmd.append("--single_trajectory")
    if args.post_frame_list:
        cmd.extend(["--post_frame_list", args.post_frame_list])
    return cmd


def selected_stages(spec):
    spec = spec.strip().lower()
    if spec in ("all", ""):
        return list(STAGES)
    requested = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = sorted(set(requested) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid stages: {', '.join(STAGES)}")
    return requested


def run(cmd, dry_run):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    score_dir = out_root / f"{args.task}_scores"
    alignment_qc_dir = out_root / f"{args.task}_alignment_qc"
    opt_dir = out_root / f"{args.task}_presence_losscolor"
    prune_dir = out_root / f"{args.task}_component_pruned"
    target_dir = out_root / f"{args.task}_component_targets_validated_mask_endpoint"
    vis_dir = out_root / f"{args.task}_component_targets_validated_mask_endpoint_points_pre_color"
    stages = selected_stages(args.stages)

    if "alignment_qc" in stages:
        cmd = [args.python, "tools/visualize_post_alignment.py"]
        add_common(cmd, args)
        cmd.extend(
            [
                "--pre_model",
                args.pre_model,
                "--model_path",
                str(alignment_qc_dir),
                "--resolution",
                str(args.resolution),
                "--num_views",
                str(args.alignment_qc_views),
                "--depth_point_stride",
                str(args.alignment_qc_depth_point_stride),
                "--point_draw_stride",
                str(args.alignment_qc_point_draw_stride),
            ]
        )
        run(cmd, args.dry_run)

    if "score" in stages:
        cmd = [args.python, "tools/score_gaussian_multiview_residuals.py"]
        add_common(cmd, args)
        cmd.extend(
            [
                "--pre_model",
                args.pre_model,
                "--model_path",
                str(score_dir),
                "--resolution",
                str(args.resolution),
                "--frame_stride",
                str(args.frame_stride),
                "--score_quantile",
                "0.95",
                "--cluster_eps",
                "0.055",
                "--cluster_min_samples",
                "3",
                "--cluster_max_bbox_diag",
                "0.8",
                "--cluster_max_planarity",
                "200.0",
                "--plant_roi_mode",
                "expanded",
                "--plant_roi_max_components",
                "3",
            ]
        )
        run(cmd, args.dry_run)

    if "optimize" in stages:
        cmd = [args.python, "tools/optimize_gaussian_presence_post.py"]
        add_common(cmd, args)
        cmd.extend(
            [
                "--pre_model",
                args.pre_model,
                "--score_npz",
                str(score_dir / "gaussian_multiview_residual_scores.npz"),
                "--cluster_ids",
                args.selected_clusters,
                "--out_dir",
                str(opt_dir),
                "--resolution",
                str(args.resolution),
                "--frame_stride",
                str(args.frame_stride),
                "--iterations",
                "800",
                "--drop_sparsity_weight",
                "0.0001",
                "--loss_color_correction",
                "--candidate_expand_radius",
                "0.06",
                "--graph_expand_geodesic_max",
                "0.5",
                "--graph_expand_edge_radius",
                "0.025",
                "--graph_expand_max_nodes",
                "30000",
                "--graph_expand_green_margin",
                "-0.05",
                "--graph_expand_residual_min",
                "0.05",
                "--select_drop_min",
                "0.2",
                "--select_opacity_drop_min",
                "0.02",
                "--max_selected_bbox_diag",
                "0.8",
                "--max_selected_planarity",
                "200.0",
                "--post_cluster_eps",
                "0.06",
            ]
        )
        run(cmd, args.dry_run)

    if "prune" in stages:
        cmd = [
            args.python,
            "tools/prune_gaussian_mask_components.py",
            "--presence_npz",
            str(opt_dir / "presence_optimization_scores.npz"),
            "--score_npz",
            str(score_dir / "gaussian_multiview_residual_scores.npz"),
            "--out_dir",
            str(prune_dir),
            "--selected_clusters",
            args.selected_clusters,
            "--component_radius",
            "0.018",
            "--min_component_size",
            "120",
            "--min_component_ratio",
            "0.08",
            "--min_median_score",
            "0.25",
        ]
        run(cmd, args.dry_run)

    if args.object_type != "leaf":
        return

    if "target" in stages:
        cmd = [
            args.python,
            "tools/extract_leaf_component_targets.py",
            "--target_method",
            "validated_mask_endpoint",
            "--presence_npz",
            str(prune_dir / "component_pruned_presence_scores.npz"),
            "--score_npz",
            str(score_dir / "gaussian_multiview_residual_scores.npz"),
            "--out_dir",
            str(target_dir),
            "--selected_clusters",
            args.selected_clusters,
        ]
        run(cmd, args.dry_run)

    if "visualize" in stages:
        if args.pre_dataset_dir is not None:
            pre_dataset_dir = Path(args.pre_dataset_dir)
        else:
            task_pre = Path(args.data_root) / f"{args.task}_pre"
            pre_dataset_dir = task_pre if task_pre.exists() else Path(args.data_root)
        cmd = [args.python, "tools/visualize_component_points.py"]
        cmd.extend(
            [
                "--dataset_dir",
                str(pre_dataset_dir),
                "--presence_npz",
                str(target_dir / "leaf_component_targets.npz"),
                "--summary_json",
                str(target_dir / "leaf_component_targets_summary.json"),
                "--out_dir",
                str(vis_dir),
                "--mask_key",
                "leaf_instance_candidate",
                "--cluster_key",
                "leaf_instance_id",
                "--selected_clusters",
                args.selected_clusters,
                "--frame_stride",
                "6",
                "--max_frames",
                "18",
                "--point_color_mode",
                "cluster",
            ]
        )
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
