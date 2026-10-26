#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_STAGES = (
    "extract",
    "colmap",
    "manifest",
    "train",
    "alignment_qc",
    "score",
    "presence",
    "tomato_targets",
    "action_dataset",
    "visualize",
    "yolo",
)


@dataclass
class PipelinePaths:
    out_root: Path
    data_root: Path
    pre_extract: Path
    post_extract: Path
    pre_colmap_root: Path
    post_colmap_root: Path
    pre_dataset: Path
    post_dataset: Path
    pre_model: Path
    score_dir: Path
    presence_dir: Path
    recluster_dir: Path
    fruit_dir: Path
    split_dir: Path
    target_dir: Path
    original_projection_vis: Path
    alignment_qc_dir: Path
    action_dataset_dir: Path
    action_vis_dir: Path
    yolo_dir: Path

    @property
    def score_npz(self) -> Path:
        return self.score_dir / "gaussian_multiview_residual_scores.npz"

    @property
    def recluster_score_npz(self) -> Path:
        return self.recluster_dir / "gaussian_multiview_residual_scores.npz"

    @property
    def presence_npz(self) -> Path:
        return self.presence_dir / "presence_optimization_scores.npz"

    @property
    def fruit_npz(self) -> Path:
        return self.fruit_dir / "tomato_fruit_mask.npz"

    @property
    def split_npz(self) -> Path:
        return self.split_dir / "tomato_fruit_mask.npz"

    @property
    def target_npz(self) -> Path:
        return self.target_dir / "tomato_component_targets.npz"

    @property
    def target_summary(self) -> Path:
        return self.target_dir / "tomato_component_targets_summary.json"

    @property
    def action_manifest(self) -> Path:
        return self.action_dataset_dir / "action_point_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full tomato action-point dataset pipeline: Aria extraction, "
            "COLMAP-style data prep, original EgoDelta/2DGS tomato target finding, "
            "and bbox/action-point manifest generation."
        )
    )
    parser.add_argument("--task", default="tomato_harvest", help="Scene/task id used in manifests and outputs.")
    parser.add_argument("--object_type", choices=("tomato",), default="tomato")
    parser.add_argument("--task_type", default="harvest", help="Action label written to the action-point manifest.")
    parser.add_argument("--class_name", default="target", help="Detector class name for generated bbox labels.")
    parser.add_argument("--out_root", type=Path, required=True)

    parser.add_argument("--pre_vrs", type=Path, default=None)
    parser.add_argument("--pre_mps", type=Path, default=None)
    parser.add_argument("--post_vrs", type=Path, default=None)
    parser.add_argument("--post_mps", type=Path, default=None)
    parser.add_argument("--pre_dataset", type=Path, default=None, help="Existing pre-scene COLMAP dataset.")
    parser.add_argument("--post_dataset", type=Path, default=None, help="Existing post-scene COLMAP dataset.")
    parser.add_argument("--pre_model", type=Path, default=None, help="Existing or planned pre-scene 2DGS model dir/PLY.")

    parser.add_argument(
        "--alignment",
        choices=("identity", "icp"),
        default="identity",
        help="Use identity for pre_split data already aligned by MPS; use icp for independent MPS trajectories.",
    )
    parser.add_argument("--baseline_root", type=Path, default=None, help="Alignment root when --alignment icp is used.")

    parser.add_argument("--stages", default="all", help=f"Comma-separated subset or all. Stages: {','.join(ALL_STAGES)}")
    parser.add_argument("--python", default=sys.executable, help="Python command for subprocesses, e.g. 'python' or 'conda run -n env python'.")
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES value for 2DGS stages.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and planned writes without executing them.")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even when stage artifacts already exist.")
    parser.add_argument("--copy_images", action="store_true", help="Copy images into generated datasets instead of symlinking.")

    parser.add_argument(
        "--quality_preset",
        choices=("fast", "balanced", "high"),
        default="high",
        help=(
            "Input-density preset for new 2DGS runs. fast keeps the old sparse settings; "
            "high is recommended for the current Aria plant samples."
        ),
    )
    parser.add_argument("--extract_stride", type=int, default=None)
    parser.add_argument("--rect_size", type=int, default=720)
    parser.add_argument("--rect_focal", type=float, default=380.0)
    parser.add_argument("--preview_stride", type=int, default=30)
    parser.add_argument("--point_stride", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=300000)

    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--train_iterations", type=int, default=30000)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)

    parser.add_argument("--score_quantile", type=float, default=0.95)
    parser.add_argument("--score_min", type=float, default=0.05)
    parser.add_argument("--score_cluster_eps", type=float, default=0.06)
    parser.add_argument("--score_cluster_min_samples", type=int, default=3)
    parser.add_argument("--score_cluster_max_bbox_diag", type=float, default=0.8)
    parser.add_argument("--score_cluster_max_planarity", type=float, default=200.0)

    parser.add_argument("--presence_iterations", type=int, default=800)
    parser.add_argument("--presence_candidate_expand_radius", type=float, default=0.12)
    parser.add_argument("--presence_post_cluster_eps", type=float, default=0.04)
    parser.add_argument("--presence_post_cluster_min_samples", type=int, default=8)
    parser.add_argument("--presence_max_selected_bbox_diag", type=float, default=0.8)
    parser.add_argument("--presence_max_selected_planarity", type=float, default=200.0)
    parser.add_argument("--presence_preview_frames", type=int, default=12)

    parser.add_argument("--recluster_score_quantile", type=float, default=0.80)
    parser.add_argument("--recluster_cluster_eps", type=float, default=0.04)
    parser.add_argument("--recluster_cluster_min_samples", type=int, default=4)
    parser.add_argument("--recluster_cluster_max_bbox_diag", type=float, default=0.8)
    parser.add_argument("--recluster_cluster_max_planarity", type=float, default=200.0)

    parser.add_argument("--tomato_component_radius", type=float, default=0.018)
    parser.add_argument("--tomato_min_component_size", type=int, default=20)
    parser.add_argument("--tomato_min_instance_gaussians", type=int, default=50)
    parser.add_argument("--tomato_min_instance_fraction", type=float, default=0.08)
    parser.add_argument("--tomato_min_split_fraction", type=float, default=0.12)
    parser.add_argument("--tomato_drop_other_clusters_below", type=int, default=50)
    parser.add_argument("--target_search_radius", type=float, default=0.09)
    parser.add_argument("--target_component_radius", type=float, default=0.018)
    parser.add_argument("--target_min_component_size", type=int, default=2)
    parser.add_argument("--target_points", type=int, default=12)

    parser.add_argument("--min_visible_points", type=int, default=8)
    parser.add_argument("--min_visible_fraction", type=float, default=0.01)
    parser.add_argument("--min_bbox_size_px", type=float, default=8.0)
    parser.add_argument("--bbox_pad_px", type=float, default=8.0)
    parser.add_argument("--bbox_pad_fraction", type=float, default=0.08)
    parser.add_argument("--max_views_per_instance", type=int, default=0)
    parser.add_argument("--debug_max_samples", type=int, default=120)
    parser.add_argument("--visual_frame_stride", type=int, default=6)
    parser.add_argument("--visual_max_frames", type=int, default=18)
    parser.add_argument("--alignment_qc_views", type=int, default=8)
    parser.add_argument("--alignment_qc_depth_point_stride", type=int, default=4)
    parser.add_argument("--alignment_qc_point_draw_stride", type=int, default=12)
    parser.add_argument("--allow_single_scene_debug_split", action="store_true")
    args = parser.parse_args()
    apply_quality_preset(args)
    return args


def apply_quality_preset(args: argparse.Namespace) -> None:
    presets = {
        "fast": {"extract_stride": 5, "point_stride": 25},
        "balanced": {"extract_stride": 2, "point_stride": 10},
        "high": {"extract_stride": 1, "point_stride": 5},
    }
    preset = presets[args.quality_preset]
    if args.extract_stride is None:
        args.extract_stride = preset["extract_stride"]
    if args.point_stride is None:
        args.point_stride = preset["point_stride"]


def parse_stages(spec: str) -> set[str]:
    tokens = [part.strip() for part in spec.split(",") if part.strip()]
    if not tokens or "all" in tokens:
        return set(ALL_STAGES)
    unknown = sorted(set(tokens) - set(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid stages are {ALL_STAGES}")
    return set(tokens)


def python_prefix(args: argparse.Namespace) -> list[str]:
    return shlex.split(args.python)


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    out_root = args.out_root
    pre_dataset = args.pre_dataset or (out_root / "aria_prepost" / "pre_colmap" / f"{args.task}_pre_pre")
    post_dataset = args.post_dataset or (out_root / "aria_prepost" / "post_colmap" / f"{args.task}_post_pre")
    pre_model = args.pre_model or (out_root / "2dgs" / f"{args.task}_pre")
    targets_root = out_root / "egodelta_2dgs"
    return PipelinePaths(
        out_root=out_root,
        data_root=out_root / "data_pair",
        pre_extract=out_root / "aria_prepost" / "pre_extract",
        post_extract=out_root / "aria_prepost" / "post_extract",
        pre_colmap_root=out_root / "aria_prepost" / "pre_colmap",
        post_colmap_root=out_root / "aria_prepost" / "post_colmap",
        pre_dataset=pre_dataset,
        post_dataset=post_dataset,
        pre_model=pre_model,
        score_dir=targets_root / f"{args.task}_red_scores",
        presence_dir=targets_root / f"{args.task}_presence_opt",
        recluster_dir=targets_root / f"{args.task}_red_scores_q80_recluster",
        fruit_dir=targets_root / f"{args.task}_fruit_seeded_red_components",
        split_dir=targets_root / f"{args.task}_fruit_auto_graph_instances",
        target_dir=targets_root / f"{args.task}_fruit_shell_targets",
        original_projection_vis=targets_root / f"{args.task}_original_target_projection",
        alignment_qc_dir=out_root / "alignment_qc",
        action_dataset_dir=out_root / "action_point_dataset",
        action_vis_dir=out_root / "action_point_dataset_vis",
        yolo_dir=out_root / "yolo_detector_dataset",
    )


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    return env


def run_command(cmd: list[str], args: argparse.Namespace, env: dict[str, str] | None = None) -> None:
    print("$ " + shlex.join(str(part) for part in cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, env=env, check=True)


def stage_done(path: Path, args: argparse.Namespace) -> bool:
    return path.exists() and not args.force


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_camera_summary(dataset_dir: Path) -> dict:
    cameras_txt = dataset_dir / "sparse" / "0" / "cameras.txt"
    with cameras_txt.open() as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8 or parts[1] != "PINHOLE":
                raise RuntimeError(f"Only PINHOLE COLMAP cameras are supported: {cameras_txt}")
            return {
                "width": int(parts[2]),
                "height": int(parts[3]),
                "fx": float(parts[4]),
                "fy": float(parts[5]),
                "cx": float(parts[6]),
                "cy": float(parts[7]),
            }
    raise RuntimeError(f"No camera row found in {cameras_txt}")


def count_colmap_images(dataset_dir: Path) -> int:
    images_txt = dataset_dir / "sparse" / "0" / "images.txt"
    count = 0
    with images_txt.open() as fp:
        for line in fp:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def write_posed_frame_list(rect_dir: Path, out_path: Path, args: argparse.Namespace) -> None:
    if stage_done(out_path, args):
        return
    if args.dry_run:
        print(f"write posed frame list: {out_path}", flush=True)
        return
    index_csv = rect_dir / "frame_index.csv"
    require_path(index_csv, "frame index CSV")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with index_csv.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if str(row.get("has_pose", "0")).strip() == "1":
                rows.append(row.get("image_name") or f"frame_{int(row['frame_idx']):06d}.jpg")
    if not rows:
        raise RuntimeError(f"No posed frames found in {index_csv}")
    out_path.write_text("\n".join(rows) + "\n")


def write_identity_manifest(paths: PipelinePaths, args: argparse.Namespace) -> None:
    manifest_path = paths.data_root / "manifest.json"
    if stage_done(manifest_path, args):
        print(f"skip manifest: {manifest_path}", flush=True)
        return
    if args.dry_run:
        print(f"write identity pre/post manifest: {manifest_path}", flush=True)
        return

    require_path(paths.pre_dataset / "sparse" / "0" / "cameras.txt", "pre dataset cameras.txt")
    require_path(paths.post_dataset / "sparse" / "0" / "cameras.txt", "post dataset cameras.txt")
    paths.data_root.mkdir(parents=True, exist_ok=True)

    def entry(label: str, dataset_dir: Path, mps_dir: Path | None) -> dict:
        return {
            "name": f"{args.task}_{label}",
            "task": args.task,
            "label": label,
            "dataset_dir": str(dataset_dir.resolve()),
            "single_trajectory": False,
            "source_root": None if mps_dir is None else str(mps_dir.resolve()),
            "camera": read_camera_summary(dataset_dir),
            "num_images_with_pose": count_colmap_images(dataset_dir),
        }

    manifest = {
        "single_trajectory": False,
        "alignment": "identity",
        "alignment_method": "identity_pre_aligned_mps_split",
        "T_pre_post": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "datasets": [
            entry("pre", paths.pre_dataset, args.pre_mps),
            entry("post", paths.post_dataset, args.post_mps),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved manifest: {manifest_path}", flush=True)


def pre_model_exists(path: Path) -> bool:
    if path.is_file():
        return True
    point_cloud = path / "point_cloud"
    return point_cloud.exists() and any(point_cloud.glob("iteration_*/point_cloud.ply"))


def add_optional_max_frames(cmd: list[str], max_frames: int) -> None:
    if max_frames > 0:
        cmd.extend(["--max_frames", str(max_frames)])


def extract_stage(label: str, vrs: Path | None, mps: Path | None, out_dir: Path, args: argparse.Namespace) -> None:
    if vrs is None or mps is None:
        raise ValueError(f"--{label}_vrs and --{label}_mps are required for extract stage")
    manifest = out_dir / "extract_manifest.json"
    if stage_done(manifest, args):
        print(f"skip {label} extraction: {manifest}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/extract_aria_pinhole_views.py",
        "--vrs",
        str(vrs),
        "--mps_dir",
        str(mps),
        "--out_root",
        str(out_dir),
        "--stride",
        str(args.extract_stride),
        "--rect_size",
        str(args.rect_size),
        "--rect_focal",
        str(args.rect_focal),
        "--preview_stride",
        str(args.preview_stride),
    ]
    run_command(cmd, args)


def colmap_stage(label: str, extract_dir: Path, mps: Path | None, colmap_root: Path, args: argparse.Namespace) -> Path:
    if mps is None:
        raise ValueError(f"--{label}_mps is required for colmap stage")
    dataset_dir = colmap_root / f"{args.task}_{label}_pre"
    if stage_done(dataset_dir / "sparse" / "0" / "images.txt", args):
        print(f"skip {label} COLMAP dataset: {dataset_dir}", flush=True)
        return dataset_dir

    rect_dir = extract_dir / "frames_rect"
    frame_list = colmap_root / f"{label}_posed_frames.txt"
    write_posed_frame_list(rect_dir, frame_list, args)
    points = mps / "slam" / "semidense_points.csv.gz"
    cmd = python_prefix(args) + [
        "tools/build_single_trajectory_split.py",
        "--source_root",
        str(rect_dir),
        "--points",
        str(points),
        "--pre_frames",
        str(frame_list),
        "--post_frames",
        str(frame_list),
        "--task",
        f"{args.task}_{label}",
        "--out_root",
        str(colmap_root),
        "--point_stride",
        str(args.point_stride),
        "--max_points",
        str(args.max_points),
    ]
    if args.copy_images:
        cmd.append("--copy_images")
    if args.force:
        cmd.append("--overwrite")
    run_command(cmd, args)
    return dataset_dir


def run_manifest_stage(paths: PipelinePaths, args: argparse.Namespace) -> None:
    if args.alignment == "identity":
        write_identity_manifest(paths, args)
        return
    if args.pre_mps is None or args.post_mps is None:
        raise ValueError("--pre_mps and --post_mps are required for --alignment icp")
    cmd = python_prefix(args) + [
        "tools/build_separate_trajectory_manifest.py",
        "--task",
        args.task,
        "--data_root",
        str(paths.data_root),
        "--pre_dataset",
        str(paths.pre_dataset),
        "--post_dataset",
        str(paths.post_dataset),
        "--pre_mps",
        str(args.pre_mps),
        "--post_mps",
        str(args.post_mps),
    ]
    if args.baseline_root is not None:
        cmd.extend(["--baseline_root", str(args.baseline_root)])
    run_command(cmd, args)


def run_train_stage(paths: PipelinePaths, args: argparse.Namespace, env: dict[str, str]) -> None:
    if stage_done(paths.pre_model, args) and pre_model_exists(paths.pre_model):
        print(f"skip 2DGS train: {paths.pre_model}", flush=True)
        return
    cmd = python_prefix(args) + [
        "train.py",
        "-s",
        str(paths.pre_dataset),
        "-m",
        str(paths.pre_model),
        "-r",
        str(args.resolution),
        "--data_device",
        args.data_device,
        "--iterations",
        str(args.train_iterations),
        "--save_iterations",
        str(args.train_iterations),
        "--test_iterations",
        str(args.train_iterations),
        "--quiet",
    ]
    run_command(cmd, args, env=env)


def run_alignment_qc_stage(paths: PipelinePaths, args: argparse.Namespace, env: dict[str, str]) -> None:
    metrics_csv = paths.alignment_qc_dir / "alignment_overlay_metrics.csv"
    if stage_done(metrics_csv, args):
        print(f"skip alignment QC: {metrics_csv}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/visualize_post_alignment.py",
        "--task",
        args.task,
        "--data_root",
        str(paths.data_root),
        "--pre_model",
        str(paths.pre_model),
        "--model_path",
        str(paths.alignment_qc_dir),
        "--resolution",
        str(args.resolution),
        "--data_device",
        args.data_device,
        "--num_views",
        str(args.alignment_qc_views),
        "--depth_point_stride",
        str(args.alignment_qc_depth_point_stride),
        "--point_draw_stride",
        str(args.alignment_qc_point_draw_stride),
    ]
    if args.alignment == "identity":
        cmd.append("--single_trajectory")
    add_optional_max_frames(cmd, args.max_frames)
    run_command(cmd, args, env=env)


def run_score_stage(paths: PipelinePaths, args: argparse.Namespace, env: dict[str, str]) -> None:
    if stage_done(paths.score_npz, args):
        print(f"skip residual scoring: {paths.score_npz}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/score_gaussian_multiview_residuals.py",
        "--task",
        args.task,
        "--data_root",
        str(paths.data_root),
        "--pre_model",
        str(paths.pre_model),
        "--model_path",
        str(paths.score_dir),
        "--resolution",
        str(args.resolution),
        "--data_device",
        args.data_device,
        "--frame_stride",
        str(args.frame_stride),
        "--score_quantile",
        str(args.score_quantile),
        "--score_min",
        str(args.score_min),
        "--cluster_eps",
        str(args.score_cluster_eps),
        "--cluster_min_samples",
        str(args.score_cluster_min_samples),
        "--cluster_max_bbox_diag",
        str(args.score_cluster_max_bbox_diag),
        "--cluster_max_planarity",
        str(args.score_cluster_max_planarity),
        "--plant_roi_mode",
        "none",
        "--appearance_roi_mode",
        "red",
    ]
    add_optional_max_frames(cmd, args.max_frames)
    run_command(cmd, args, env=env)


def run_presence_stage(paths: PipelinePaths, args: argparse.Namespace, env: dict[str, str]) -> None:
    if stage_done(paths.presence_npz, args):
        print(f"skip presence optimization: {paths.presence_npz}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/optimize_gaussian_presence_post.py",
        "--task",
        args.task,
        "--data_root",
        str(paths.data_root),
        "--pre_model",
        str(paths.pre_model),
        "--score_npz",
        str(paths.score_npz),
        "--cluster_ids",
        "all",
        "--out_dir",
        str(paths.presence_dir),
        "--resolution",
        str(args.resolution),
        "--data_device",
        args.data_device,
        "--frame_stride",
        str(args.frame_stride),
        "--iterations",
        str(args.presence_iterations),
        "--candidate_expand_radius",
        str(args.presence_candidate_expand_radius),
        "--post_cluster_eps",
        str(args.presence_post_cluster_eps),
        "--post_cluster_min_samples",
        str(args.presence_post_cluster_min_samples),
        "--max_selected_bbox_diag",
        str(args.presence_max_selected_bbox_diag),
        "--max_selected_planarity",
        str(args.presence_max_selected_planarity),
        "--preview_frames",
        str(args.presence_preview_frames),
        "--loss_color_correction",
    ]
    add_optional_max_frames(cmd, args.max_frames)
    run_command(cmd, args, env=env)


def run_tomato_targets_stage(paths: PipelinePaths, args: argparse.Namespace) -> None:
    if stage_done(paths.target_npz, args):
        print(f"skip tomato target extraction: {paths.target_npz}", flush=True)
        return

    commands = [
        python_prefix(args)
        + [
            "tools/recluster_gaussian_residual_scores.py",
            "--score_npz",
            str(paths.score_npz),
            "--out_dir",
            str(paths.recluster_dir),
            "--score_quantile",
            str(args.recluster_score_quantile),
            "--score_min",
            str(args.score_min),
            "--cluster_eps",
            str(args.recluster_cluster_eps),
            "--cluster_min_samples",
            str(args.recluster_cluster_min_samples),
            "--cluster_max_bbox_diag",
            str(args.recluster_cluster_max_bbox_diag),
            "--cluster_max_planarity",
            str(args.recluster_cluster_max_planarity),
        ],
        python_prefix(args)
        + [
            "tools/filter_tomato_fruit_mask.py",
            "--presence_npz",
            str(paths.presence_npz),
            "--score_npz",
            str(paths.recluster_score_npz),
            "--out_dir",
            str(paths.fruit_dir),
            "--mask_key",
            "optimized_candidate",
            "--cluster_key",
            "post_opt_cluster_id",
            "--component_radius",
            str(args.tomato_component_radius),
            "--min_component_size",
            str(args.tomato_min_component_size),
            "--score_seeded_red_components",
        ],
        python_prefix(args)
        + [
            "tools/split_tomato_fruit_instances.py",
            "--presence_npz",
            str(paths.fruit_npz),
            "--out_dir",
            str(paths.split_dir),
            "--auto",
            "--min_instance_gaussians",
            str(args.tomato_min_instance_gaussians),
            "--min_instance_fraction",
            str(args.tomato_min_instance_fraction),
            "--min_split_fraction",
            str(args.tomato_min_split_fraction),
            "--drop_other_clusters_below",
            str(args.tomato_drop_other_clusters_below),
        ],
        python_prefix(args)
        + [
            "tools/extract_tomato_fruit_shell_targets.py",
            "--presence_npz",
            str(paths.split_npz),
            "--score_npz",
            str(paths.recluster_score_npz),
            "--out_dir",
            str(paths.target_dir),
            "--mask_key",
            "tomato_fruit_candidate",
            "--cluster_key",
            "tomato_instance_id",
            "--search_radius",
            str(args.target_search_radius),
            "--component_radius",
            str(args.target_component_radius),
            "--min_component_size",
            str(args.target_min_component_size),
            "--target_points",
            str(args.target_points),
        ],
    ]
    for cmd in commands:
        run_command(cmd, args)


def run_action_dataset_stage(paths: PipelinePaths, args: argparse.Namespace) -> None:
    if stage_done(paths.action_manifest, args):
        print(f"skip action-point dataset: {paths.action_manifest}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/build_action_point_dataset.py",
        "--dataset_dir",
        str(paths.pre_dataset),
        "--presence_npz",
        str(paths.target_npz),
        "--summary_json",
        str(paths.target_summary),
        "--out_dir",
        str(paths.action_dataset_dir),
        "--scene_id",
        args.task,
        "--task_type",
        args.task_type,
        "--class_name",
        args.class_name,
        "--mask_key",
        "tomato_instance_candidate",
        "--cluster_key",
        "tomato_instance_id",
        "--min_visible_points",
        str(args.min_visible_points),
        "--min_visible_fraction",
        str(args.min_visible_fraction),
        "--min_bbox_size_px",
        str(args.min_bbox_size_px),
        "--bbox_pad_px",
        str(args.bbox_pad_px),
        "--bbox_pad_fraction",
        str(args.bbox_pad_fraction),
        "--max_views_per_instance",
        str(args.max_views_per_instance),
        "--debug_vis",
        "--debug_max_samples",
        str(args.debug_max_samples),
    ]
    run_command(cmd, args)


def run_visualize_stage(paths: PipelinePaths, args: argparse.Namespace) -> None:
    original_sheet = paths.original_projection_vis / "contact_sheet.jpg"
    if not stage_done(original_sheet, args):
        cmd = python_prefix(args) + [
            "tools/visualize_component_points.py",
            "--dataset_dir",
            str(paths.pre_dataset),
            "--presence_npz",
            str(paths.target_npz),
            "--summary_json",
            str(paths.target_summary),
            "--out_dir",
            str(paths.original_projection_vis),
            "--mask_key",
            "tomato_instance_candidate",
            "--cluster_key",
            "tomato_instance_id",
            "--frame_stride",
            str(args.visual_frame_stride),
            "--max_frames",
            str(args.visual_max_frames),
            "--point_color_mode",
            "cluster",
        ]
        run_command(cmd, args)
    else:
        print(f"skip original target projection visualization: {original_sheet}", flush=True)

    action_sheet = paths.action_vis_dir / "contact_sheet.jpg"
    if stage_done(action_sheet, args):
        print(f"skip action-point label visualization: {action_sheet}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/visualize_action_point_dataset.py",
        "--manifest",
        str(paths.action_manifest),
        "--out_dir",
        str(paths.action_vis_dir),
        "--max_samples",
        str(args.debug_max_samples),
    ]
    run_command(cmd, args)


def run_yolo_stage(paths: PipelinePaths, args: argparse.Namespace) -> None:
    data_yaml = paths.yolo_dir / "data.yaml"
    if stage_done(data_yaml, args):
        print(f"skip YOLO export: {data_yaml}", flush=True)
        return
    cmd = python_prefix(args) + [
        "tools/export_action_detector_dataset.py",
        "--manifest",
        str(paths.action_manifest),
        "--out_dir",
        str(paths.yolo_dir),
        "--class_mode",
        "unified",
        "--unified_class_name",
        args.class_name,
    ]
    if args.copy_images:
        cmd.append("--copy_images")
    if args.allow_single_scene_debug_split:
        cmd.append("--allow_single_scene_debug_split")
    run_command(cmd, args)


def main() -> int:
    args = parse_args()
    stages = parse_stages(args.stages)
    paths = build_paths(args)
    env = command_env(args)

    print(
        json.dumps(
            {
                "task": args.task,
                "stages": [stage for stage in ALL_STAGES if stage in stages],
                "out_root": str(paths.out_root),
                "quality_preset": args.quality_preset,
                "extract_stride": args.extract_stride,
                "point_stride": args.point_stride,
                "data_root": str(paths.data_root),
                "pre_dataset": str(paths.pre_dataset),
                "post_dataset": str(paths.post_dataset),
                "pre_model": str(paths.pre_model),
                "action_manifest": str(paths.action_manifest),
            },
            indent=2,
        ),
        flush=True,
    )

    if "extract" in stages:
        extract_stage("pre", args.pre_vrs, args.pre_mps, paths.pre_extract, args)
        extract_stage("post", args.post_vrs, args.post_mps, paths.post_extract, args)

    if "colmap" in stages:
        paths.pre_dataset = colmap_stage("pre", paths.pre_extract, args.pre_mps, paths.pre_colmap_root, args)
        paths.post_dataset = colmap_stage("post", paths.post_extract, args.post_mps, paths.post_colmap_root, args)

    if "manifest" in stages:
        run_manifest_stage(paths, args)

    if "train" in stages:
        run_train_stage(paths, args, env)

    if "alignment_qc" in stages:
        run_alignment_qc_stage(paths, args, env)

    if "score" in stages:
        run_score_stage(paths, args, env)

    if "presence" in stages:
        run_presence_stage(paths, args, env)

    if "tomato_targets" in stages:
        run_tomato_targets_stage(paths, args)

    if "action_dataset" in stages:
        run_action_dataset_stage(paths, args)

    if "visualize" in stages:
        run_visualize_stage(paths, args)

    if "yolo" in stages:
        run_yolo_stage(paths, args)

    print(
        json.dumps(
            {
                "done": True,
                "target_npz": str(paths.target_npz),
                "alignment_qc": str(paths.alignment_qc_dir / "alignment_overlay_metrics.csv"),
                "action_manifest": str(paths.action_manifest),
                "action_visualization": str(paths.action_vis_dir / "contact_sheet.jpg"),
                "yolo_data_yaml": str(paths.yolo_dir / "data.yaml"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
