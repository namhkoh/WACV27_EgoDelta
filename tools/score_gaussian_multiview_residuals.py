#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import PipelineParams
from change_detection.alignment import (
    DEFAULT_BASELINE_ROOT,
    filter_camera_infos_by_frame_list,
    identity_alignment,
    load_colmap_camera_infos,
    manifest_uses_identity_alignment,
    resolve_task_alignment,
    resolve_task_datasets,
    select_camera_infos,
    transform_camera_infos_to_pre,
)
from change_detection.cluster_gaussians import dbscan_numpy
from change_detection.conflict_masks import robust_color_correct
from change_detection.export import gather_gaussian_arrays
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import fov2focal
from utils.system_utils import searchForMaxIteration


def parse_args():
    parser = argparse.ArgumentParser(description="Score pre Gaussians by multi-view post residuals")
    pipeline = PipelineParams(parser)

    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", default="data/corl2026_2dgs")
    parser.add_argument("--baseline_root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--post_frame_list", default=None)
    parser.add_argument("--pre_model", required=True)
    parser.add_argument("--pre_iteration", default=-1, type=int)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--chunk_size", default=65536, type=int)
    parser.add_argument("--alpha_min", default=0.2, type=float)
    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--depth_margin", default=0.06, type=float)
    parser.add_argument("--per_frame_residual_quantile", default=0.90, type=float)
    parser.add_argument("--min_views", default=4, type=int)

    parser.add_argument("--density_radius", default=0.035, type=float)
    parser.add_argument("--density_quantile", default=0.55, type=float)
    parser.add_argument("--score_quantile", default=0.995, type=float)
    parser.add_argument("--score_min", default=0.05, type=float)
    parser.add_argument("--cluster_eps", default=0.04, type=float)
    parser.add_argument("--cluster_min_samples", default=8, type=int)
    parser.add_argument("--cluster_max_bbox_diag", default=0.35, type=float)
    parser.add_argument("--cluster_max_planarity", default=18.0, type=float)

    parser.add_argument("--plant_roi_mode", default="none", choices=("none", "expanded"))
    parser.add_argument("--plant_seed_opacity_min", default=0.08, type=float)
    parser.add_argument("--plant_seed_green_ratio", default=1.05, type=float)
    parser.add_argument("--plant_seed_green_min", default=0.08, type=float)
    parser.add_argument("--plant_seed_exg_min", default=0.02, type=float)
    parser.add_argument("--plant_seed_exg_quantile", default=0.70, type=float)
    parser.add_argument("--plant_seed_cluster_eps", default=0.055, type=float)
    parser.add_argument("--plant_seed_min_samples", default=10, type=int)
    parser.add_argument("--plant_seed_min_component_size", default=25, type=int)
    parser.add_argument("--plant_roi_expand_radius", default=0.06, type=float)
    parser.add_argument("--plant_roi_max_components", default=2, type=int)
    parser.add_argument("--plant_roi_outside_weight", default=0.0, type=float)

    parser.add_argument("--appearance_roi_mode", default="none", choices=("none", "red", "dark"))
    parser.add_argument("--appearance_roi_outside_weight", default=0.0, type=float)
    parser.add_argument("--red_roi_opacity_min", default=0.05, type=float)
    parser.add_argument("--red_roi_min", default=0.25, type=float)
    parser.add_argument("--red_roi_rg_ratio", default=1.25, type=float)
    parser.add_argument("--red_roi_rb_ratio", default=1.25, type=float)
    parser.add_argument("--dark_roi_opacity_min", default=0.05, type=float)
    parser.add_argument("--dark_roi_luma_max", default=0.35, type=float)
    parser.add_argument("--dark_roi_value_max", default=0.45, type=float)
    parser.add_argument("--dark_roi_saturation_max", default=0.75, type=float)
    return parser.parse_args(), pipeline


def resolve_pre_ply(pre_model, iteration):
    path = Path(pre_model)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Pre model path does not exist: {path}")
    if iteration < 0:
        iteration = searchForMaxIteration(str(path / "point_cloud"))
    ply = path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Could not resolve pre-scene PLY: {ply}")
    return ply


def build_camera_args(args):
    return Namespace(resolution=args.resolution, data_device=args.data_device)


def project_points(points_world, camera):
    device = points_world.device
    r_cw = torch.as_tensor(camera.R.T, dtype=torch.float32, device=device)
    t_cw = torch.as_tensor(camera.T, dtype=torch.float32, device=device)
    points_cam = points_world @ r_cw.T + t_cw
    z = points_cam[:, 2]

    width = int(camera.image_width)
    height = int(camera.image_height)
    fx = fov2focal(camera.FoVx, width)
    fy = fov2focal(camera.FoVy, height)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5

    z_safe = z.clamp_min(1e-6)
    u = fx * (points_cam[:, 0] / z_safe) + cx
    v = fy * (points_cam[:, 1] / z_safe) + cy
    valid = (
        (z > camera.znear)
        & (z < camera.zfar)
        & (u >= 0.0)
        & (u <= width - 1.0)
        & (v >= 0.0)
        & (v <= height - 1.0)
    )
    return u, v, z, valid


def sample_image_at_uv(image, u, v, width, height):
    if u.numel() == 0:
        return torch.empty((image.shape[0], 0), dtype=image.dtype, device=image.device)
    x = (u / max(1, width - 1)) * 2.0 - 1.0
    y = (v / max(1, height - 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0]


def local_density_counts(xyz, active_mask, radius):
    active_xyz = xyz[active_mask]
    if active_xyz.shape[0] == 0:
        return np.zeros(xyz.shape[0], dtype=np.int32)
    tree = cKDTree(active_xyz)
    counts_active = tree.query_ball_point(active_xyz, radius, return_length=True)
    counts = np.zeros(xyz.shape[0], dtype=np.int32)
    counts[np.flatnonzero(active_mask)] = counts_active.astype(np.int32)
    return counts


def build_appearance_roi(rgb, opacity, args):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    opacity = opacity.reshape(-1).astype(np.float32)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    if args.appearance_roi_mode == "none":
        roi = np.ones(rgb.shape[0], dtype=bool)
    elif args.appearance_roi_mode == "red":
        roi = (
            (opacity >= args.red_roi_opacity_min)
            & (r >= args.red_roi_min)
            & (r >= g * args.red_roi_rg_ratio)
            & (r >= b * args.red_roi_rb_ratio)
        )
    elif args.appearance_roi_mode == "dark":
        value = np.max(rgb, axis=1)
        chroma = value - np.min(rgb, axis=1)
        saturation = chroma / np.maximum(value, 1e-6)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        roi = (
            (opacity >= args.dark_roi_opacity_min)
            & (luma <= args.dark_roi_luma_max)
            & (value <= args.dark_roi_value_max)
            & (saturation <= args.dark_roi_saturation_max)
        )
    else:
        raise ValueError(f"Unsupported appearance ROI mode: {args.appearance_roi_mode}")

    return {
        "appearance_roi": roi,
        "appearance_roi_summary": {
            "mode": args.appearance_roi_mode,
            "roi_gaussians": int(roi.sum()),
            "outside_weight": float(args.appearance_roi_outside_weight),
            "red_threshold": {
                "opacity_min": float(args.red_roi_opacity_min),
                "red_min": float(args.red_roi_min),
                "red_over_green": float(args.red_roi_rg_ratio),
                "red_over_blue": float(args.red_roi_rb_ratio),
            },
            "dark_threshold": {
                "opacity_min": float(args.dark_roi_opacity_min),
                "luma_max": float(args.dark_roi_luma_max),
                "value_max": float(args.dark_roi_value_max),
                "saturation_max": float(args.dark_roi_saturation_max),
            },
        },
    }


def build_expanded_plant_roi(xyz, rgb, opacity, args):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    opacity = opacity.reshape(-1).astype(np.float32)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    excess_green = 2.0 * g - r - b

    active_seed = opacity >= args.plant_seed_opacity_min
    green_pool = active_seed & (g >= args.plant_seed_green_min)
    if green_pool.any():
        robust_exg = float(np.quantile(excess_green[green_pool], args.plant_seed_exg_quantile))
    else:
        robust_exg = args.plant_seed_exg_min
    exg_thresh = max(args.plant_seed_exg_min, robust_exg)
    seed = (
        green_pool
        & (g >= r * args.plant_seed_green_ratio)
        & (g >= b * args.plant_seed_green_ratio)
        & (excess_green >= exg_thresh)
    )

    labels_seed = np.full(xyz.shape[0], -1, dtype=np.int64)
    component_seed = np.zeros(xyz.shape[0], dtype=bool)
    roi = np.zeros(xyz.shape[0], dtype=bool)
    selected_components = []

    seed_indices = np.flatnonzero(seed)
    if seed_indices.size == 0:
        return {
            "plant_roi_seed": seed,
            "plant_roi_component": component_seed,
            "plant_structure_roi": roi,
            "plant_roi_component_id": labels_seed,
            "plant_excess_green": excess_green.astype(np.float32),
            "plant_roi_summary": {
                "mode": "expanded",
                "excess_green_threshold": exg_thresh,
                "seed_gaussians": 0,
                "component_seed_gaussians": 0,
                "roi_gaussians": 0,
                "selected_components": [],
            },
        }

    labels_local = dbscan_numpy(
        xyz[seed_indices].astype(np.float32),
        args.plant_seed_cluster_eps,
        args.plant_seed_min_samples,
    )
    components = []
    for label in sorted(int(v) for v in np.unique(labels_local) if v >= 0):
        local = np.flatnonzero(labels_local == label)
        if local.size < args.plant_seed_min_component_size:
            continue
        members = seed_indices[local]
        components.append((label, members))
    components.sort(key=lambda item: item[1].size, reverse=True)
    if args.plant_roi_max_components > 0:
        components = components[: args.plant_roi_max_components]

    for new_label, (_, members) in enumerate(components):
        labels_seed[members] = new_label
        component_seed[members] = True
        c_xyz = xyz[members]
        selected_components.append(
            {
                "component_id": int(new_label),
                "seed_gaussians": int(members.size),
                "centroid_xyz": c_xyz.mean(axis=0).tolist(),
                "bbox_min": c_xyz.min(axis=0).tolist(),
                "bbox_max": c_xyz.max(axis=0).tolist(),
            }
        )

    if component_seed.any():
        active_expand = opacity > args.opacity_min
        tree = cKDTree(xyz.astype(np.float32))
        neighbor_lists = tree.query_ball_point(
            xyz[component_seed].astype(np.float32),
            args.plant_roi_expand_radius,
        )
        neighbors = [np.asarray(v, dtype=np.int64) for v in neighbor_lists if len(v) > 0]
        if neighbors:
            roi[np.unique(np.concatenate(neighbors))] = True
        roi &= active_expand

    return {
        "plant_roi_seed": seed,
        "plant_roi_component": component_seed,
        "plant_structure_roi": roi,
        "plant_roi_component_id": labels_seed,
        "plant_excess_green": excess_green.astype(np.float32),
        "plant_roi_summary": {
            "mode": "expanded",
            "excess_green_threshold": exg_thresh,
            "seed_gaussians": int(seed.sum()),
            "component_seed_gaussians": int(component_seed.sum()),
            "roi_gaussians": int(roi.sum()),
            "selected_components": selected_components,
        },
    }


def cluster_shape(xyz):
    if xyz.shape[0] < 3:
        return np.zeros(3, dtype=np.float32)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, xyz.shape[0] - 1)
    return np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)


def summarize_clusters(xyz, opacity, score, residual, visibility, candidate_indices, labels_local, args):
    summaries = []
    labels_full = np.full(xyz.shape[0], -1, dtype=np.int64)
    valid_labels = sorted(int(v) for v in np.unique(labels_local) if v >= 0)

    new_id = 0
    for label in valid_labels:
        local = np.flatnonzero(labels_local == label)
        members = candidate_indices[local]
        c_xyz = xyz[members]
        eigvals = cluster_shape(c_xyz)
        bbox_min = c_xyz.min(axis=0)
        bbox_max = c_xyz.max(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
        planarity = float(eigvals[0] / (eigvals[2] + 1e-8)) if eigvals[2] > 0 else float("inf")
        if bbox_diag > args.cluster_max_bbox_diag:
            continue
        if planarity > args.cluster_max_planarity:
            continue

        labels_full[members] = new_id
        summaries.append(
            {
                "cluster_id": new_id,
                "num_gaussians": int(members.size),
                "mean_score": float(score[members].mean()),
                "max_score": float(score[members].max()),
                "mean_residual": float(residual[members].mean()),
                "mean_visibility_count": float(visibility[members].mean()),
                "mean_opacity_pre": float(opacity[members].mean()),
                "centroid_xyz": c_xyz.mean(axis=0).tolist(),
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "bbox_diag": bbox_diag,
                "pca_eigvals": eigvals.tolist(),
                "planarity_ratio": planarity,
                "label": "dense multi-view residual candidate",
            }
        )
        new_id += 1
    return labels_full, summaries


def write_score_ply(path, arrays, score, residual, high_ratio, visibility, density, cluster_labels, candidate_mask):
    xyz = arrays["xyz"]
    rgb = np.clip(arrays["rgb"], 0.0, 1.0)
    opacity = arrays["opacity_pre"].reshape(-1)
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    colors[candidate_mask] = np.array([255, 60, 40], dtype=np.uint8)
    clustered = cluster_labels >= 0
    colors[clustered] = np.array([255, 180, 30], dtype=np.uint8)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("score", "f4"),
        ("residual_mean", "f4"),
        ("high_residual_ratio", "f4"),
        ("visibility_count", "i4"),
        ("density_count", "i4"),
        ("opacity_pre", "f4"),
        ("candidate", "u1"),
        ("cluster_id", "i4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"] = colors[:, 2]
    elements["score"] = score
    elements["residual_mean"] = residual
    elements["high_residual_ratio"] = high_ratio
    elements["visibility_count"] = visibility.astype(np.int32)
    elements["density_count"] = density.astype(np.int32)
    elements["opacity_pre"] = opacity
    elements["candidate"] = candidate_mask.astype(np.uint8)
    elements["cluster_id"] = cluster_labels.astype(np.int32)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def write_plant_roi_ply(path, arrays, plant_seed, plant_component, plant_roi, candidate_mask, cluster_labels):
    xyz = arrays["xyz"]
    rgb = np.clip(arrays["rgb"], 0.0, 1.0)
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    colors[plant_roi] = np.array([40, 210, 230], dtype=np.uint8)
    colors[plant_component] = np.array([40, 190, 70], dtype=np.uint8)
    colors[plant_seed] = np.array([20, 255, 20], dtype=np.uint8)
    colors[candidate_mask] = np.array([255, 80, 40], dtype=np.uint8)
    colors[cluster_labels >= 0] = np.array([255, 200, 30], dtype=np.uint8)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("plant_seed", "u1"),
        ("plant_component", "u1"),
        ("plant_structure_roi", "u1"),
        ("candidate", "u1"),
        ("cluster_id", "i4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"] = colors[:, 2]
    elements["plant_seed"] = plant_seed.astype(np.uint8)
    elements["plant_component"] = plant_component.astype(np.uint8)
    elements["plant_structure_roi"] = plant_roi.astype(np.uint8)
    elements["candidate"] = candidate_mask.astype(np.uint8)
    elements["cluster_id"] = cluster_labels.astype(np.int32)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


@torch.no_grad()
def main():
    args, pipeline_group = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(torch.device("cuda:0"))

    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    _, post_dataset = resolve_task_datasets(args.data_root, args.task)
    use_identity_alignment = args.single_trajectory or manifest_uses_identity_alignment(args.data_root)
    if use_identity_alignment:
        alignment_manifest, t_pre_post, alignment_path = identity_alignment("identity_single_trajectory")
    else:
        alignment_manifest, t_pre_post, alignment_path = resolve_task_alignment(args.baseline_root, args.task)

    pipe = pipeline_group.extract(args)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    pre_ply = resolve_pre_ply(args.pre_model, args.pre_iteration)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(pre_ply))
    arrays = gather_gaussian_arrays(gaussians)
    xyz_t = gaussians.get_xyz.detach()
    opacity_np = arrays["opacity_pre"].reshape(-1)
    n_gaussians = xyz_t.shape[0]

    post_camera_infos = load_colmap_camera_infos(post_dataset["dataset_dir"], images=args.images)
    post_camera_infos = filter_camera_infos_by_frame_list(post_camera_infos, args.post_frame_list)
    post_camera_infos = select_camera_infos(post_camera_infos, args.frame_stride, args.max_frames)
    post_camera_infos = transform_camera_infos_to_pre(post_camera_infos, t_pre_post)
    cameras = cameraList_from_camInfos(post_camera_infos, 1.0, build_camera_args(args))
    if not cameras:
        raise RuntimeError("No post cameras were loaded")

    residual_sum = torch.zeros(n_gaussians, dtype=torch.float32, device="cuda")
    excess_sum = torch.zeros_like(residual_sum)
    visibility_count = torch.zeros(n_gaussians, dtype=torch.int32, device="cuda")
    high_count = torch.zeros(n_gaussians, dtype=torch.int32, device="cuda")

    for camera in tqdm(cameras, desc="Scoring Gaussian residuals"):
        pre_pack = render(camera, gaussians, pipe, background)
        pre_rgb = torch.clamp(pre_pack["render"], 0.0, 1.0)
        post_rgb = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)
        pre_alpha = torch.clamp(pre_pack["rend_alpha"], 0.0, 1.0)
        pre_depth = pre_pack["surf_depth"]
        valid_color = pre_alpha > args.alpha_min
        pre_corr, _, _ = robust_color_correct(pre_rgb, post_rgb, valid_color)
        residual_map = torch.mean(torch.abs(pre_corr - post_rgb), dim=0, keepdim=True)
        radii = pre_pack["radii"].detach()

        width = int(camera.image_width)
        height = int(camera.image_height)
        frame_visible_indices = []
        frame_residual_values = []
        for start in range(0, n_gaussians, args.chunk_size):
            end = min(start + args.chunk_size, n_gaussians)
            idx = torch.arange(start, end, device="cuda")
            u, v, z, valid = project_points(xyz_t[start:end], camera)
            valid = valid & (radii[start:end] > 0)
            if not valid.any():
                continue
            local_idx = idx[valid]
            u_valid = u[valid]
            v_valid = v[valid]
            z_valid = z[valid]
            residual = sample_image_at_uv(residual_map, u_valid, v_valid, width, height)[0]
            alpha = sample_image_at_uv(pre_alpha, u_valid, v_valid, width, height)[0]
            depth = sample_image_at_uv(pre_depth, u_valid, v_valid, width, height)[0]
            visible = (alpha > args.alpha_min) & (torch.abs(z_valid - depth) < args.depth_margin)
            if not visible.any():
                continue
            frame_visible_indices.append(local_idx[visible])
            frame_residual_values.append(residual[visible])

        if not frame_visible_indices:
            continue
        visible_idx = torch.cat(frame_visible_indices)
        visible_residual = torch.cat(frame_residual_values)
        threshold = torch.quantile(visible_residual, args.per_frame_residual_quantile)
        residual_sum.index_add_(0, visible_idx, visible_residual)
        excess_sum.index_add_(0, visible_idx, torch.clamp(visible_residual - threshold, min=0.0))
        visibility_count.index_add_(0, visible_idx, torch.ones_like(visible_idx, dtype=torch.int32))
        high_count.index_add_(0, visible_idx, (visible_residual > threshold).to(torch.int32))

    visibility_np = visibility_count.detach().cpu().numpy()
    residual_mean = (
        residual_sum / visibility_count.clamp_min(1).to(torch.float32)
    ).detach().cpu().numpy()
    excess_mean = (
        excess_sum / visibility_count.clamp_min(1).to(torch.float32)
    ).detach().cpu().numpy()
    high_ratio = (
        high_count.to(torch.float32) / visibility_count.clamp_min(1).to(torch.float32)
    ).detach().cpu().numpy()

    active = opacity_np > args.opacity_min
    density = local_density_counts(arrays["xyz"], active, args.density_radius)
    active_density = density[active]
    density_min = int(np.quantile(active_density, args.density_quantile)) if active_density.size else 0
    density_scale = max(1.0, float(np.quantile(active_density, 0.95) - density_min)) if active_density.size else 1.0
    density_score = np.clip((density.astype(np.float32) - density_min) / density_scale, 0.0, 1.0)

    visible_ok = visibility_np >= args.min_views
    residual_scale = np.quantile(excess_mean[visible_ok], 0.98) if visible_ok.any() else 1.0
    residual_score = np.clip(excess_mean / max(float(residual_scale), 1e-6), 0.0, 1.0)
    consistency_score = np.clip(high_ratio / 0.5, 0.0, 1.0)
    final_score = residual_score * consistency_score * density_score
    final_score[~active] = 0.0
    final_score[~visible_ok] = 0.0
    score_before_plant_roi = final_score.copy()

    plant_roi_outputs = {
        "plant_roi_seed": np.zeros(n_gaussians, dtype=bool),
        "plant_roi_component": np.zeros(n_gaussians, dtype=bool),
        "plant_structure_roi": np.ones(n_gaussians, dtype=bool),
        "plant_roi_component_id": np.full(n_gaussians, -1, dtype=np.int64),
        "plant_excess_green": np.zeros(n_gaussians, dtype=np.float32),
        "plant_roi_summary": {
            "mode": "none",
            "seed_gaussians": 0,
            "component_seed_gaussians": 0,
            "roi_gaussians": int(n_gaussians),
            "selected_components": [],
        },
    }
    if args.plant_roi_mode == "expanded":
        plant_roi_outputs = build_expanded_plant_roi(arrays["xyz"], arrays["rgb"], opacity_np, args)
        plant_roi = plant_roi_outputs["plant_structure_roi"]
        outside_weight = float(np.clip(args.plant_roi_outside_weight, 0.0, 1.0))
        final_score = final_score * np.where(plant_roi, 1.0, outside_weight).astype(np.float32)

    appearance_roi_outputs = build_appearance_roi(arrays["rgb"], opacity_np, args)
    if args.appearance_roi_mode != "none":
        appearance_roi = appearance_roi_outputs["appearance_roi"]
        outside_weight = float(np.clip(args.appearance_roi_outside_weight, 0.0, 1.0))
        final_score = final_score * np.where(appearance_roi, 1.0, outside_weight).astype(np.float32)

    score_thresh = max(args.score_min, float(np.quantile(final_score[final_score > 0], args.score_quantile)) if (final_score > 0).any() else 1.0)
    candidate = final_score >= score_thresh
    candidate_indices = np.flatnonzero(candidate)
    if candidate_indices.size:
        labels_local = dbscan_numpy(arrays["xyz"][candidate_indices], args.cluster_eps, args.cluster_min_samples)
        cluster_labels, clusters = summarize_clusters(
            arrays["xyz"],
            opacity_np,
            final_score,
            residual_mean,
            visibility_np,
            candidate_indices,
            labels_local,
            args,
        )
    else:
        cluster_labels = np.full(n_gaussians, -1, dtype=np.int64)
        clusters = []

    np.savez(
        model_path / "gaussian_multiview_residual_scores.npz",
        gaussian_id=np.arange(n_gaussians, dtype=np.int64),
        xyz=arrays["xyz"],
        rgb=arrays["rgb"],
        opacity_pre=opacity_np,
        residual_mean=residual_mean,
        residual_excess_mean=excess_mean,
        high_residual_ratio=high_ratio,
        visibility_count=visibility_np,
        density_count=density,
        density_score=density_score,
        score_before_plant_roi=score_before_plant_roi,
        score=final_score,
        candidate=candidate,
        cluster_id=cluster_labels,
        plant_roi_seed=plant_roi_outputs["plant_roi_seed"],
        plant_roi_component=plant_roi_outputs["plant_roi_component"],
        plant_structure_roi=plant_roi_outputs["plant_structure_roi"],
        plant_roi_component_id=plant_roi_outputs["plant_roi_component_id"],
        plant_excess_green=plant_roi_outputs["plant_excess_green"],
        appearance_roi=appearance_roi_outputs["appearance_roi"],
    )
    write_score_ply(
        model_path / "gaussian_multiview_residual_scores.ply",
        arrays,
        final_score,
        residual_mean,
        high_ratio,
        visibility_np,
        density,
        cluster_labels,
        candidate,
    )
    (model_path / "multiview_residual_clusters.json").write_text(json.dumps(clusters, indent=2) + "\n")
    if args.plant_roi_mode != "none":
        write_plant_roi_ply(
            model_path / "plant_structure_roi.ply",
            arrays,
            plant_roi_outputs["plant_roi_seed"],
            plant_roi_outputs["plant_roi_component"],
            plant_roi_outputs["plant_structure_roi"],
            candidate,
            cluster_labels,
        )
        (model_path / "plant_structure_roi_summary.json").write_text(
            json.dumps(plant_roi_outputs["plant_roi_summary"], indent=2) + "\n"
        )
    config = vars(args).copy()
    config.update(
        {
            "resolved_pre_ply": str(pre_ply),
            "alignment_manifest": str(alignment_path),
            "alignment_method": alignment_manifest.get("alignment_method"),
            "num_post_cameras": len(cameras),
            "num_gaussians": int(n_gaussians),
            "density_min": density_min,
            "score_threshold": score_thresh,
            "num_candidates": int(candidate.sum()),
            "num_clusters": len(clusters),
            "plant_roi": plant_roi_outputs["plant_roi_summary"],
            "appearance_roi": appearance_roi_outputs["appearance_roi_summary"],
        }
    )
    (model_path / "score_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"Saved {model_path / 'gaussian_multiview_residual_scores.npz'}")
    print(f"Saved {model_path / 'gaussian_multiview_residual_scores.ply'}")
    if args.plant_roi_mode != "none":
        print(f"Saved {model_path / 'plant_structure_roi.ply'}")
    print(f"Saved {model_path / 'multiview_residual_clusters.json'} ({len(clusters)} clusters)")
    print(f"Candidates: {int(candidate.sum())}, score_threshold={score_thresh:.6f}")


if __name__ == "__main__":
    main()
