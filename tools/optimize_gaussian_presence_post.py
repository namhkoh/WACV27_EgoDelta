#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
from utils.loss_utils import ssim
from utils.system_utils import searchForMaxIteration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize candidate Gaussian opacity drops against post images with fixed pre geometry"
    )
    pipeline = PipelineParams(parser)

    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", default="data/corl2026_2dgs_single")
    parser.add_argument("--baseline_root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--post_frame_list", default=None)
    parser.add_argument("--pre_model", required=True)
    parser.add_argument("--pre_iteration", default=-1, type=int)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--cluster_id", default=0, type=int)
    parser.add_argument(
        "--cluster_ids",
        default=None,
        help="Comma-separated residual cluster ids, or 'all', to optimize as one multi-target candidate set.",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--model_path", default=None)

    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--iterations", default=800, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--drop_lr", default=0.4, type=float)
    parser.add_argument("--initial_drop", default=0.01, type=float)
    parser.add_argument("--lambda_dssim", default=0.0, type=float)
    parser.add_argument("--drop_sparsity_weight", default=1e-5, type=float)
    parser.add_argument("--grad_clip", default=0.0, type=float)
    parser.add_argument(
        "--loss_color_correction",
        action="store_true",
        help="Apply per-frame affine color correction to rendered RGB before computing the opacity optimization loss.",
    )
    parser.add_argument("--loss_color_alpha_min", default=0.2, type=float)

    parser.add_argument("--candidate_expand_radius", default=0.035, type=float)
    parser.add_argument("--graph_expand_geodesic_max", default=0.0, type=float)
    parser.add_argument("--graph_expand_edge_radius", default=0.018, type=float)
    parser.add_argument("--graph_expand_search_radius", default=0.55, type=float)
    parser.add_argument("--graph_expand_max_nodes", default=20000, type=int)
    parser.add_argument("--graph_expand_green_min", default=0.08, type=float)
    parser.add_argument("--graph_expand_green_margin", default=-0.03, type=float)
    parser.add_argument("--graph_expand_value_min", default=0.06, type=float)
    parser.add_argument("--graph_expand_red_ratio_max", default=1.25, type=float)
    parser.add_argument("--graph_expand_residual_min", default=0.0, type=float)
    parser.add_argument("--opacity_min", default=0.05, type=float)
    parser.add_argument("--select_drop_min", default=0.05, type=float)
    parser.add_argument("--select_opacity_drop_min", default=0.005, type=float)
    parser.add_argument("--max_selected_bbox_diag", default=0.35, type=float)
    parser.add_argument("--max_selected_planarity", default=18.0, type=float)
    parser.add_argument("--post_cluster_eps", default=0.04, type=float)
    parser.add_argument("--post_cluster_min_samples", default=8, type=int)

    parser.add_argument("--grad_probe_frames", default=0, type=int)
    parser.add_argument("--preview_frames", default=12, type=int)
    parser.add_argument("--point_radius", default=4, type=int)
    parser.add_argument("--max_projected_points", default=600, type=int)
    parser.add_argument("--eval_every", default=50, type=int)
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


def default_output_dir(args):
    data_name = Path(args.data_root).name
    if args.cluster_ids:
        cluster_label = str(args.cluster_ids).replace(",", "_").replace(" ", "")
        return Path("output") / data_name / f"presence_opt_{args.task}_clusters{cluster_label}"
    return Path("output") / data_name / f"presence_opt_{args.task}_cluster{args.cluster_id}"


def parse_cluster_selection(cluster_ids, args):
    valid_ids = sorted(int(v) for v in np.unique(cluster_ids) if int(v) >= 0)
    if args.cluster_ids is None:
        return [int(args.cluster_id)]
    spec = str(args.cluster_ids).strip().lower()
    if spec == "all":
        if not valid_ids:
            raise RuntimeError(f"No residual clusters are available in {args.score_npz}")
        return valid_ids
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        selected.append(int(part))
    if not selected:
        raise RuntimeError("--cluster_ids was provided but no ids were parsed")
    missing = sorted(set(selected) - set(valid_ids))
    if missing:
        raise RuntimeError(f"Requested cluster ids {missing} are absent from {args.score_npz}")
    return sorted(set(selected))


def appearance_graph_mask(scores, active, plant_roi, args):
    n_gaussians = active.shape[0]
    if "rgb" not in scores:
        return active & plant_roi
    rgb = np.clip(scores["rgb"].astype(np.float32), 0.0, 1.0)
    if rgb.shape[0] != n_gaussians:
        return active & plant_roi

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    value = np.max(rgb, axis=1)
    strong_red = (r > g * args.graph_expand_red_ratio_max) & (r > b * args.graph_expand_red_ratio_max)
    greenish = (
        (g >= args.graph_expand_green_min)
        & (g >= r + args.graph_expand_green_margin)
        & (g >= b + args.graph_expand_green_margin)
        & (value >= args.graph_expand_value_min)
        & ~strong_red
    )
    if "score" in scores and args.graph_expand_residual_min > 0.0:
        greenish &= scores["score"].astype(np.float32) >= args.graph_expand_residual_min
    return active & plant_roi & greenish


def graph_geodesic_expand(xyz, scores, core, active, plant_roi, args):
    n_gaussians = xyz.shape[0]
    expanded = np.zeros(n_gaussians, dtype=bool)
    meta = {
        "enabled": bool(args.graph_expand_geodesic_max > 0.0),
        "geodesic_max": float(args.graph_expand_geodesic_max),
        "edge_radius": float(args.graph_expand_edge_radius),
        "search_radius": float(args.graph_expand_search_radius),
        "graph_nodes": 0,
        "seed_nodes": 0,
        "expanded_gaussians": 0,
        "truncated_by_max_nodes": False,
    }
    if args.graph_expand_geodesic_max <= 0.0 or args.graph_expand_edge_radius <= 0.0:
        return expanded, meta

    graph_mask = appearance_graph_mask(scores, active, plant_roi, args)
    if not graph_mask.any():
        return expanded, meta

    tree_all = cKDTree(xyz)
    scoped = np.zeros(n_gaussians, dtype=bool)
    neighbor_lists = tree_all.query_ball_point(xyz[core], args.graph_expand_search_radius)
    neighbors = [np.asarray(v, dtype=np.int64) for v in neighbor_lists if len(v) > 0]
    if neighbors:
        scoped[np.unique(np.concatenate(neighbors))] = True
    graph_mask &= scoped

    graph_indices = np.flatnonzero(graph_mask).astype(np.int64)
    if graph_indices.size == 0:
        return expanded, meta
    if graph_indices.size > args.graph_expand_max_nodes:
        meta["truncated_by_max_nodes"] = True
        core_points = xyz[core]
        core_tree = cKDTree(core_points)
        dist_to_core, _ = core_tree.query(xyz[graph_indices], k=1)
        keep_local = np.argsort(dist_to_core)[: args.graph_expand_max_nodes]
        graph_indices = graph_indices[keep_local]

    local_of_global = {int(global_idx): local_idx for local_idx, global_idx in enumerate(graph_indices)}
    seed_indices = np.asarray([idx for idx in np.flatnonzero(core) if int(idx) in local_of_global], dtype=np.int64)
    if seed_indices.size == 0:
        return expanded, meta

    seed_local = np.asarray([local_of_global[int(idx)] for idx in seed_indices], dtype=np.int64)
    graph_xyz = xyz[graph_indices].astype(np.float32)
    graph_tree = cKDTree(graph_xyz)
    dist = np.full(graph_indices.shape[0], np.inf, dtype=np.float32)
    heap = []
    for local_idx in seed_local:
        dist[local_idx] = 0.0
        heapq.heappush(heap, (0.0, int(local_idx)))

    max_dist = float(args.graph_expand_geodesic_max)
    edge_radius = float(args.graph_expand_edge_radius)
    while heap:
        current_dist, local_idx = heapq.heappop(heap)
        if current_dist != float(dist[local_idx]) or current_dist > max_dist:
            continue
        for neighbor_idx in graph_tree.query_ball_point(graph_xyz[local_idx], edge_radius):
            if neighbor_idx == local_idx:
                continue
            edge = float(np.linalg.norm(graph_xyz[neighbor_idx] - graph_xyz[local_idx]))
            new_dist = current_dist + edge
            if new_dist <= max_dist and new_dist < float(dist[neighbor_idx]):
                dist[neighbor_idx] = new_dist
                heapq.heappush(heap, (new_dist, int(neighbor_idx)))

    reached = np.isfinite(dist)
    expanded[graph_indices[reached]] = True
    meta.update(
        {
            "graph_nodes": int(graph_indices.size),
            "seed_nodes": int(seed_indices.size),
            "expanded_gaussians": int(expanded.sum()),
            "max_reached_geodesic": float(dist[reached].max()) if reached.any() else 0.0,
        }
    )
    return expanded, meta


def safe_sigmoid_inverse(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def freeze_gaussians(gaussians):
    for tensor in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ):
        tensor.requires_grad_(False)


def load_post_cameras(args):
    _, post_dataset = resolve_task_datasets(args.data_root, args.task)
    use_identity = args.single_trajectory or manifest_uses_identity_alignment(args.data_root)
    if use_identity:
        alignment_manifest, t_pre_post, alignment_path = identity_alignment("identity_single_trajectory")
    else:
        alignment_manifest, t_pre_post, alignment_path = resolve_task_alignment(args.baseline_root, args.task)

    camera_infos = load_colmap_camera_infos(post_dataset["dataset_dir"], images=args.images)
    camera_infos = filter_camera_infos_by_frame_list(camera_infos, args.post_frame_list)
    camera_infos = select_camera_infos(camera_infos, args.frame_stride, args.max_frames)
    camera_infos = transform_camera_infos_to_pre(camera_infos, t_pre_post)
    cameras = cameraList_from_camInfos(camera_infos, 1.0, build_camera_args(args))
    if not cameras:
        raise RuntimeError("No post cameras were loaded")
    return cameras, alignment_manifest, alignment_path


def build_candidate_mask(arrays, scores, args):
    xyz = arrays["xyz"].astype(np.float32)
    opacity = arrays["opacity_pre"].reshape(-1).astype(np.float32)
    n_gaussians = xyz.shape[0]

    if scores["xyz"].shape[0] != n_gaussians:
        raise RuntimeError(
            f"Score NPZ has {scores['xyz'].shape[0]} Gaussians but the pre model has {n_gaussians}"
        )
    if "cluster_id" not in scores:
        raise KeyError(f"Score NPZ has no cluster_id array: {args.score_npz}")

    cluster_ids = scores["cluster_id"].astype(np.int64)
    selected_cluster_ids = parse_cluster_selection(cluster_ids, args)
    core = np.isin(cluster_ids, selected_cluster_ids)
    if not core.any():
        raise RuntimeError(f"Clusters {selected_cluster_ids} have no members in {args.score_npz}")

    expanded = core.copy()
    if args.candidate_expand_radius > 0.0:
        tree = cKDTree(xyz)
        neighbor_lists = tree.query_ball_point(xyz[core], args.candidate_expand_radius)
        if len(neighbor_lists) > 0:
            neighbors = [np.asarray(v, dtype=np.int64) for v in neighbor_lists if len(v) > 0]
            if neighbors:
                expanded[np.unique(np.concatenate(neighbors))] = True

    active = opacity > args.opacity_min
    plant_roi = scores["plant_structure_roi"].astype(bool) if "plant_structure_roi" in scores else np.ones(n_gaussians, dtype=bool)
    graph_expanded, graph_meta = graph_geodesic_expand(xyz, scores, core, active, plant_roi, args)
    expanded |= graph_expanded
    candidate = expanded & active & plant_roi
    if not candidate.any():
        raise RuntimeError(
            f"No candidate Gaussians remain after opacity_min={args.opacity_min}; "
            f"cluster members before filtering: {int(core.sum())}"
        )
    return core, candidate, plant_roi, selected_cluster_ids, graph_expanded, graph_meta


def make_effective_opacity(pre_opacity, candidate_indices_t, drop_logit):
    if drop_logit is None:
        return pre_opacity
    drop = torch.sigmoid(drop_logit).view(-1, 1)
    opacity = pre_opacity.clone()
    opacity[candidate_indices_t] = pre_opacity[candidate_indices_t] * (1.0 - drop)
    return opacity


def apply_loss_color_correction(prediction, target, alpha, alpha_min):
    valid = torch.clamp(alpha, 0.0, 1.0) > alpha_min
    with torch.no_grad():
        _, scale, offset = robust_color_correct(prediction.detach(), target, valid)
    corrected = scale.view(3, 1, 1) * prediction + offset.view(3, 1, 1)
    return torch.clamp(corrected, 0.0, 1.0)


def camera_rgb_loss(camera, gaussians, pipe, background, override_opacity, args):
    pack = render(camera, gaussians, pipe, background, override_opacity=override_opacity)
    prediction = torch.clamp(pack["render"], 0.0, 1.0)
    target = torch.clamp(camera.original_image.to(prediction.device), 0.0, 1.0)
    if args.loss_color_correction:
        prediction = apply_loss_color_correction(
            prediction,
            target,
            pack["rend_alpha"],
            args.loss_color_alpha_min,
        )
    l1 = torch.abs(prediction - target).mean()
    if args.lambda_dssim <= 0.0:
        return l1, prediction, target
    dssim = 1.0 - ssim(prediction, target)
    loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * dssim
    return loss, prediction, target


def evaluate_loss(cameras, gaussians, pipe, background, pre_opacity, candidate_indices_t, drop_logit, args):
    losses = []
    with torch.no_grad():
        override_opacity = make_effective_opacity(pre_opacity, candidate_indices_t, drop_logit)
        for camera in cameras:
            loss, _, _ = camera_rgb_loss(
                camera,
                gaussians,
                pipe,
                background,
                override_opacity,
                args,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def gradient_probe(cameras, gaussians, pipe, background, pre_opacity, candidate_indices_t, init_logit, args):
    probe_cameras = cameras
    if args.grad_probe_frames and len(cameras) > args.grad_probe_frames:
        idx = np.linspace(0, len(cameras) - 1, args.grad_probe_frames).round().astype(np.int64)
        probe_cameras = [cameras[i] for i in idx]

    drop_logit = torch.nn.Parameter(
        torch.full((candidate_indices_t.numel(), 1), init_logit, dtype=torch.float32, device="cuda")
    )
    grad_accum = torch.zeros(candidate_indices_t.numel(), dtype=torch.float32, device="cuda")
    loss_values = []
    for camera in tqdm(probe_cameras, desc="Gradient probe"):
        drop_logit.grad = None
        override_opacity = make_effective_opacity(pre_opacity, candidate_indices_t, drop_logit)
        loss, _, _ = camera_rgb_loss(
            camera,
            gaussians,
            pipe,
            background,
            override_opacity,
            args,
        )
        loss.backward()
        grad_accum += drop_logit.grad.detach().abs().view(-1)
        loss_values.append(float(loss.detach().cpu()))
    grad_accum /= max(1, len(probe_cameras))
    return grad_accum.detach().cpu().numpy(), float(np.mean(loss_values)) if loss_values else 0.0


def optimize_presence(cameras, gaussians, pipe, background, pre_opacity, candidate_indices_t, init_logit, args):
    drop_logit = torch.nn.Parameter(
        torch.full((candidate_indices_t.numel(), 1), init_logit, dtype=torch.float32, device="cuda")
    )
    optimizer = torch.optim.Adam([drop_logit], lr=args.drop_lr)
    history = []
    batch_size = max(1, min(args.batch_size, len(cameras)))

    for iteration in tqdm(range(1, args.iterations + 1), desc="Optimizing presence"):
        batch = random.sample(cameras, batch_size) if batch_size < len(cameras) else cameras
        override_opacity = make_effective_opacity(pre_opacity, candidate_indices_t, drop_logit)
        rgb_loss = 0.0
        for camera in batch:
            loss, _, _ = camera_rgb_loss(
                camera,
                gaussians,
                pipe,
                background,
                override_opacity,
                args,
            )
            rgb_loss = rgb_loss + loss
        rgb_loss = rgb_loss / len(batch)
        drop = torch.sigmoid(drop_logit)
        sparse_loss = drop.mean()
        loss = rgb_loss + args.drop_sparsity_weight * sparse_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([drop_logit], args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            drop_logit.clamp_(min=-12.0, max=12.0)

        if iteration == 1 or iteration == args.iterations or iteration % max(1, args.eval_every) == 0:
            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "rgb_loss": float(rgb_loss.detach().cpu()),
                    "sparse_loss": float(sparse_loss.detach().cpu()),
                    "mean_drop": float(drop.detach().mean().cpu()),
                    "max_drop": float(drop.detach().max().cpu()),
                }
            )
    return drop_logit.detach(), history


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
    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), valid.detach().cpu().numpy()


def tensor_to_pil(image_tensor):
    arr = image_tensor[:3].detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def residual_to_pil(prediction, target):
    residual = torch.mean(torch.abs(prediction[:3] - target[:3]), dim=0, keepdim=True).repeat(3, 1, 1)
    arr = residual.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr / max(1e-6, np.quantile(arr, 0.98)), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def draw_points(draw, us, vs, valid, color, radius):
    coords = []
    for u, v, ok in zip(us, vs, valid):
        if not ok:
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        coords.append((x, y))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    return coords


def limit_mask_by_score(mask, score, max_points):
    indices = np.flatnonzero(mask)
    if max_points <= 0 or indices.size <= max_points:
        return mask
    order = np.argsort(score[indices])[::-1][:max_points]
    limited = np.zeros_like(mask, dtype=bool)
    limited[indices[order]] = True
    return limited


def label_image(image, label):
    font = ImageFont.load_default()
    label_h = 18
    canvas = Image.new("RGB", (image.width, image.height + label_h), (24, 24, 24))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, image.height + 3), label, fill=(235, 235, 235), font=font)
    return canvas


def horizontal_panel(images, labels, thumb_w=240):
    labeled = []
    for image, label in zip(images, labels):
        image = image.convert("RGB")
        thumb_h = int(round(thumb_w * image.height / image.width))
        image = image.resize((thumb_w, thumb_h))
        labeled.append(label_image(image, label))
    width = sum(image.width for image in labeled)
    height = max(image.height for image in labeled)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    x = 0
    for image in labeled:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def make_contact_sheet(paths, out_path, cols=4, thumb_w=260):
    if not paths:
        return
    font = ImageFont.load_default()
    label_h = 18
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        thumb_h = int(round(thumb_w * img.height / img.width))
        img = img.resize((thumb_w, thumb_h))
        thumbs.append((path, img))
    thumb_h = max(img.height for _, img in thumbs)
    rows = int(math.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (path, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + thumb_h), path.stem[:42], fill=(235, 235, 235), font=font)
    canvas.save(out_path, quality=92)


def write_presence_ply(
    path,
    arrays,
    residual_score,
    cluster_id,
    post_opt_cluster_id,
    candidate,
    optimized_candidate,
    opacity_optimized,
    opacity_drop,
    presence_drop,
    grad_norm,
):
    xyz = arrays["xyz"]
    rgb = np.clip(arrays["rgb"], 0.0, 1.0)
    opacity_pre = arrays["opacity_pre"].reshape(-1)
    colors = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    colors[candidate] = np.array([255, 60, 40], dtype=np.uint8)
    colors[optimized_candidate] = np.array([255, 150, 25], dtype=np.uint8)
    colors[post_opt_cluster_id >= 0] = np.array([255, 210, 30], dtype=np.uint8)
    colors[post_opt_cluster_id == 0] = np.array([255, 245, 40], dtype=np.uint8)

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
        ("opacity_pre", "f4"),
        ("opacity_optimized", "f4"),
        ("opacity_drop", "f4"),
        ("presence_drop", "f4"),
        ("candidate", "u1"),
        ("optimized_candidate", "u1"),
        ("grad_norm", "f4"),
        ("residual_score_prior", "f4"),
        ("cluster_id", "i4"),
        ("post_opt_cluster_id", "i4"),
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
    elements["opacity_pre"] = opacity_pre
    elements["opacity_optimized"] = opacity_optimized
    elements["opacity_drop"] = opacity_drop
    elements["presence_drop"] = presence_drop
    elements["candidate"] = candidate.astype(np.uint8)
    elements["optimized_candidate"] = optimized_candidate.astype(np.uint8)
    elements["grad_norm"] = grad_norm
    elements["residual_score_prior"] = residual_score
    elements["cluster_id"] = cluster_id.astype(np.int32)
    elements["post_opt_cluster_id"] = post_opt_cluster_id.astype(np.int32)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def cluster_shape(xyz):
    if xyz.shape[0] < 3:
        return np.zeros(3, dtype=np.float32)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, xyz.shape[0] - 1)
    return np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)


def selected_geometry_summary(xyz, selected):
    if not selected.any():
        return {
            "selected_gaussians": 0,
            "centroid_xyz": None,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_diag": 0.0,
            "pca_eigvals": [0.0, 0.0, 0.0],
            "planarity_ratio": 0.0,
        }
    selected_xyz = xyz[selected]
    bbox_min = selected_xyz.min(axis=0)
    bbox_max = selected_xyz.max(axis=0)
    eigvals = cluster_shape(selected_xyz)
    planarity = float(eigvals[0] / (eigvals[2] + 1e-8)) if eigvals[2] > 0 else float("inf")
    return {
        "selected_gaussians": int(selected_xyz.shape[0]),
        "centroid_xyz": selected_xyz.mean(axis=0).tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_diag": float(np.linalg.norm(bbox_max - bbox_min)),
        "pca_eigvals": eigvals.tolist(),
        "planarity_ratio": planarity,
    }


def summarize_post_opt_clusters(
    xyz,
    candidate_core,
    optimized_candidate,
    opacity_pre,
    opacity_drop,
    presence_drop,
    grad_norm,
    residual_score,
    args,
):
    post_opt_cluster_id = np.full(xyz.shape[0], -1, dtype=np.int64)
    selected_indices = np.flatnonzero(optimized_candidate)
    if selected_indices.size == 0:
        return post_opt_cluster_id, [], 0

    labels_local = dbscan_numpy(
        xyz[selected_indices].astype(np.float32),
        args.post_cluster_eps,
        args.post_cluster_min_samples,
    )
    noise_count = int((labels_local < 0).sum())
    raw_labels = [int(v) for v in np.unique(labels_local) if v >= 0]
    raw_summaries = []
    for label in raw_labels:
        local = np.flatnonzero(labels_local == label)
        members = selected_indices[local]
        c_xyz = xyz[members]
        bbox_min = c_xyz.min(axis=0)
        bbox_max = c_xyz.max(axis=0)
        eigvals = cluster_shape(c_xyz)
        planarity = float(eigvals[0] / (eigvals[2] + 1e-8)) if eigvals[2] > 0 else float("inf")
        raw_summaries.append(
            {
                "raw_label": label,
                "members": members,
                "evidence_score": float(opacity_drop[members].sum()),
                "num_gaussians": int(members.size),
                "mean_presence_drop": float(presence_drop[members].mean()),
                "max_presence_drop": float(presence_drop[members].max()),
                "mean_opacity_pre": float(opacity_pre[members].mean()),
                "mean_opacity_drop": float(opacity_drop[members].mean()),
                "max_opacity_drop": float(opacity_drop[members].max()),
                "mean_grad_norm": float(grad_norm[members].mean()),
                "mean_residual_score_prior": float(residual_score[members].mean()),
                "residual_core_overlap": int(candidate_core[members].sum()),
                "residual_core_overlap_ratio": float(candidate_core[members].mean()),
                "centroid_xyz": c_xyz.mean(axis=0).tolist(),
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "bbox_diag": float(np.linalg.norm(bbox_max - bbox_min)),
                "pca_eigvals": eigvals.tolist(),
                "planarity_ratio": planarity,
            }
        )

    raw_summaries.sort(key=lambda item: item["evidence_score"], reverse=True)
    summaries = []
    for new_id, summary in enumerate(raw_summaries):
        members = summary.pop("members")
        post_opt_cluster_id[members] = new_id
        summary["post_opt_cluster_id"] = new_id
        summaries.append(summary)
    return post_opt_cluster_id, summaries, noise_count


def classify_result(summary, args):
    if summary["selected_gaussians"] == 0:
        return "no compact opacity drop"
    if summary["loss_improvement"] <= 0.0:
        return "appearance/alignment residual"
    primary_cluster = summary.get("primary_post_opt_cluster")
    if primary_cluster is None:
        return "scattered opacity drop"
    if (
        primary_cluster["bbox_diag"] > args.max_selected_bbox_diag
        or primary_cluster["planarity_ratio"] > args.max_selected_planarity
    ):
        return "background residual artifact"
    return "valid opacity-drop candidate"


def write_projection_images(out_dir, cameras, xyz, candidate, optimized_candidate, primary_post_opt_cluster, opacity_drop, args):
    projection_dir = out_dir / "projection_images"
    projection_dir.mkdir(parents=True, exist_ok=True)

    primary_draw = limit_mask_by_score(primary_post_opt_cluster, opacity_drop, args.max_projected_points)
    secondary_draw = limit_mask_by_score(optimized_candidate & ~primary_draw, opacity_drop, args.max_projected_points)
    candidate_draw = limit_mask_by_score(candidate & ~optimized_candidate, opacity_drop, args.max_projected_points)
    candidate_points = torch.from_numpy(xyz[candidate_draw].astype(np.float32)).cuda() if candidate_draw.any() else None
    secondary_points = torch.from_numpy(xyz[secondary_draw].astype(np.float32)).cuda() if secondary_draw.any() else None
    primary_points = torch.from_numpy(xyz[primary_draw].astype(np.float32)).cuda() if primary_draw.any() else None

    written = []
    metadata = []
    for idx, camera in enumerate(cameras[: max(0, args.preview_frames)]):
        image = tensor_to_pil(camera.original_image).convert("RGB")
        draw = ImageDraw.Draw(image)
        projected_primary = 0
        projected_secondary = 0
        if candidate_points is not None:
            u, v, valid = project_points(candidate_points, camera)
            draw_points(draw, u, v, valid, (220, 50, 40), max(1, args.point_radius - 2))
        if secondary_points is not None:
            u, v, valid = project_points(secondary_points, camera)
            coords = draw_points(draw, u, v, valid, (255, 150, 25), max(1, args.point_radius - 1))
            projected_secondary = len(coords)
        if primary_points is not None:
            u, v, valid = project_points(primary_points, camera)
            coords = draw_points(draw, u, v, valid, (255, 210, 30), args.point_radius)
            projected_primary = len(coords)
            if coords:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                pad = args.point_radius * 3
                draw.rectangle((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), outline=(255, 230, 80), width=3)

        out_path = projection_dir / f"{idx:03d}_{camera.image_name}_presence_drop.jpg"
        image.save(out_path, quality=92)
        written.append(out_path)
        metadata.append(
            {
                "view": camera.image_name,
                "projected_selected_points": projected_primary,
                "projected_primary_points": projected_primary,
                "projected_secondary_points": projected_secondary,
                "primary_points_drawn": int(primary_draw.sum()),
                "secondary_points_drawn": int(secondary_draw.sum()),
                "selected_points_drawn": int(optimized_candidate.sum()),
                "candidate_points_drawn": int(candidate_draw.sum()),
                "image": str(out_path),
            }
        )

    make_contact_sheet(written, projection_dir / "contact_sheet.jpg")
    (projection_dir / "projection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_render_previews(out_dir, cameras, gaussians, pipe, background, pre_opacity, candidate_indices_t, drop_logit, args):
    preview_dir = out_dir / "optimized_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    projection_dir = out_dir / "projection_images"
    projection_dir.mkdir(parents=True, exist_ok=True)

    written = []
    rows = []
    with torch.no_grad():
        optimized_opacity = make_effective_opacity(pre_opacity, candidate_indices_t, drop_logit)
        for idx, camera in enumerate(cameras[: max(0, args.preview_frames)]):
            pre_loss, pre_render, target = camera_rgb_loss(
                camera,
                gaussians,
                pipe,
                background,
                pre_opacity,
                args,
            )
            opt_loss, opt_render, _ = camera_rgb_loss(
                camera,
                gaussians,
                pipe,
                background,
                optimized_opacity,
                args,
            )
            post_img = tensor_to_pil(target)
            pre_img = tensor_to_pil(pre_render)
            opt_img = tensor_to_pil(opt_render)
            pre_residual = residual_to_pil(pre_render, target)
            opt_residual = residual_to_pil(opt_render, target)

            panel = horizontal_panel(
                [post_img, pre_img, opt_img],
                ["post RGB", f"pre render {float(pre_loss):.4f}", f"optimized {float(opt_loss):.4f}"],
            )
            out_path = preview_dir / f"{idx:03d}_{camera.image_name}_before_after.jpg"
            panel.save(out_path, quality=92)
            written.append(out_path)

            row = horizontal_panel(
                [post_img, pre_img, opt_img, pre_residual, opt_residual],
                ["post", "pre", "optimized", "pre residual", "opt residual"],
                thumb_w=180,
            )
            row_path = preview_dir / f"{idx:03d}_{camera.image_name}_render_row.jpg"
            row.save(row_path, quality=92)
            rows.append(row_path)

    make_contact_sheet(written, preview_dir / "contact_sheet.jpg", cols=2, thumb_w=640)
    make_contact_sheet(rows, projection_dir / "render_contact_sheet.jpg", cols=1, thumb_w=900)


def main():
    args, pipeline_group = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This optimization tool requires CUDA because the renderer runs on CUDA")
    torch.cuda.set_device(torch.device("cuda:0"))

    out_dir = Path(args.out_dir or args.model_path or default_output_dir(args))
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline_group.extract(args)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    pre_ply = resolve_pre_ply(args.pre_model, args.pre_iteration)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(pre_ply))
    freeze_gaussians(gaussians)
    arrays = gather_gaussian_arrays(gaussians)
    n_gaussians = arrays["xyz"].shape[0]

    scores = np.load(args.score_npz)
    candidate_core, candidate, plant_roi, selected_cluster_ids, graph_expanded, graph_meta = build_candidate_mask(
        arrays,
        scores,
        args,
    )
    candidate_indices = np.flatnonzero(candidate).astype(np.int64)
    candidate_indices_t = torch.from_numpy(candidate_indices).long().cuda()
    pre_opacity = gaussians.get_opacity.detach()
    residual_score = scores["score"].astype(np.float32) if "score" in scores else np.zeros(n_gaussians, dtype=np.float32)
    cluster_ids = scores["cluster_id"].astype(np.int64)
    gaussian_id = scores["gaussian_id"].astype(np.int64) if "gaussian_id" in scores else np.arange(n_gaussians, dtype=np.int64)

    cameras, alignment_manifest, alignment_path = load_post_cameras(args)
    init_logit = safe_sigmoid_inverse(args.initial_drop)

    print(f"Loaded {n_gaussians} Gaussians from {pre_ply}")
    print(f"Using {len(cameras)} post cameras")
    print(
        f"Candidate Gaussians: {candidate_indices.size} "
        f"(cluster core: {int(candidate_core.sum())}, residual clusters: {selected_cluster_ids})"
    )
    if graph_meta["enabled"]:
        print(
            f"Graph-expanded Gaussians: {graph_meta['expanded_gaussians']} "
            f"(graph nodes: {graph_meta['graph_nodes']}, seeds: {graph_meta['seed_nodes']})"
        )

    grad_candidate, probe_loss = gradient_probe(
        cameras,
        gaussians,
        pipe,
        background,
        pre_opacity,
        candidate_indices_t,
        init_logit,
        args,
    )
    grad_norm = np.zeros(n_gaussians, dtype=np.float32)
    grad_norm[candidate_indices] = grad_candidate.astype(np.float32)
    np.savez(
        out_dir / "gradient_probe_scores.npz",
        gaussian_id=gaussian_id,
        xyz=arrays["xyz"],
        candidate=candidate,
        candidate_core=candidate_core,
        plant_structure_roi=plant_roi,
        candidate_graph_expanded=graph_expanded,
        grad_norm=grad_norm,
        residual_score_prior=residual_score,
        cluster_id=cluster_ids,
        loss_probe_initial_drop=np.float32(probe_loss),
    )
    (out_dir / "gradient_probe_summary.json").write_text(
        json.dumps(
            {
                "cluster_ids": selected_cluster_ids,
                "graph_expansion": graph_meta,
                "probe_frame_count": int(min(args.grad_probe_frames, len(cameras)) if args.grad_probe_frames else len(cameras)),
                "loss_probe_initial_drop": probe_loss,
                "candidate_gaussians": int(candidate.sum()),
                "plant_roi_gaussians": int(plant_roi.sum()),
                "mean_grad_norm_candidate": float(grad_norm[candidate].mean()) if candidate.any() else 0.0,
                "max_grad_norm_candidate": float(grad_norm[candidate].max()) if candidate.any() else 0.0,
            },
            indent=2,
        )
        + "\n"
    )

    loss_before = evaluate_loss(
        cameras,
        gaussians,
        pipe,
        background,
        pre_opacity,
        candidate_indices_t,
        None,
        args,
    )
    drop_logit, loss_history = optimize_presence(
        cameras,
        gaussians,
        pipe,
        background,
        pre_opacity,
        candidate_indices_t,
        init_logit,
        args,
    )
    loss_after = evaluate_loss(
        cameras,
        gaussians,
        pipe,
        background,
        pre_opacity,
        candidate_indices_t,
        drop_logit,
        args,
    )

    opacity_pre_np = arrays["opacity_pre"].reshape(-1).astype(np.float32)
    drop_np = torch.sigmoid(drop_logit).detach().cpu().numpy().reshape(-1).astype(np.float32)
    presence_drop = np.zeros(n_gaussians, dtype=np.float32)
    presence_drop[candidate_indices] = drop_np
    opacity_optimized = opacity_pre_np.copy()
    opacity_optimized[candidate_indices] = opacity_pre_np[candidate_indices] * (1.0 - drop_np)
    opacity_drop = opacity_pre_np - opacity_optimized
    optimized_candidate = (
        candidate
        & (presence_drop >= args.select_drop_min)
        & (opacity_drop >= args.select_opacity_drop_min)
    )
    post_opt_cluster_id, post_opt_clusters, post_opt_noise_count = summarize_post_opt_clusters(
        arrays["xyz"],
        candidate_core,
        optimized_candidate,
        opacity_pre_np,
        opacity_drop,
        presence_drop,
        grad_norm,
        residual_score,
        args,
    )
    primary_post_opt_cluster = post_opt_cluster_id == 0

    np.savez(
        out_dir / "presence_optimization_scores.npz",
        gaussian_id=gaussian_id,
        xyz=arrays["xyz"],
        opacity_pre=opacity_pre_np,
        opacity_optimized=opacity_optimized,
        opacity_drop=opacity_drop,
        presence_drop=presence_drop,
        candidate=candidate,
        candidate_core=candidate_core,
        candidate_graph_expanded=graph_expanded,
        plant_structure_roi=plant_roi,
        optimized_candidate=optimized_candidate,
        post_opt_cluster_id=post_opt_cluster_id,
        primary_post_opt_cluster=primary_post_opt_cluster,
        grad_norm=grad_norm,
        residual_score_prior=residual_score,
        cluster_id=cluster_ids,
        loss_before=np.float32(loss_before),
        loss_after=np.float32(loss_after),
        loss_improvement=np.float32(loss_before - loss_after),
    )
    write_presence_ply(
        out_dir / "presence_optimization_scores.ply",
        arrays,
        residual_score,
        cluster_ids,
        post_opt_cluster_id,
        candidate,
        optimized_candidate,
        opacity_optimized,
        opacity_drop,
        presence_drop,
        grad_norm,
    )
    (out_dir / "post_optimization_clusters.json").write_text(json.dumps(post_opt_clusters, indent=2) + "\n")

    geometry = selected_geometry_summary(arrays["xyz"], optimized_candidate)
    primary_geometry = selected_geometry_summary(arrays["xyz"], primary_post_opt_cluster)
    summary = {
        "task": args.task,
        "cluster_id": int(selected_cluster_ids[0]) if len(selected_cluster_ids) == 1 else None,
        "cluster_ids": selected_cluster_ids,
        "resolved_pre_ply": str(pre_ply),
        "score_npz": str(args.score_npz),
        "alignment_manifest": str(alignment_path),
        "alignment_method": alignment_manifest.get("alignment_method"),
        "post_frame_count": len(cameras),
        "num_gaussians": int(n_gaussians),
        "candidate_expand_radius": float(args.candidate_expand_radius),
        "graph_expansion": graph_meta,
        "candidate_gaussians": int(candidate.sum()),
        "candidate_core_gaussians": int(candidate_core.sum()),
        "plant_roi_gaussians": int(plant_roi.sum()),
        "optimized_candidate_gaussians": int(optimized_candidate.sum()),
        "post_opt_num_clusters": len(post_opt_clusters),
        "post_opt_noise_gaussians": post_opt_noise_count,
        "primary_post_opt_cluster": post_opt_clusters[0] if post_opt_clusters else None,
        "loss_probe_initial_drop": probe_loss,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_improvement": loss_before - loss_after,
        "mean_presence_drop_candidate": float(presence_drop[candidate].mean()) if candidate.any() else 0.0,
        "max_presence_drop_candidate": float(presence_drop[candidate].max()) if candidate.any() else 0.0,
        "mean_opacity_drop_candidate": float(opacity_drop[candidate].mean()) if candidate.any() else 0.0,
        "max_opacity_drop_candidate": float(opacity_drop[candidate].max()) if candidate.any() else 0.0,
        "mean_grad_norm_candidate": float(grad_norm[candidate].mean()) if candidate.any() else 0.0,
        "max_grad_norm_candidate": float(grad_norm[candidate].max()) if candidate.any() else 0.0,
        "loss_history": loss_history,
    }
    summary.update(geometry)
    summary["primary_cluster_geometry"] = primary_geometry
    summary["failure_label"] = classify_result(summary, args)

    (out_dir / "presence_optimization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "presence_optimization_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    write_projection_images(
        out_dir,
        cameras,
        arrays["xyz"],
        candidate,
        optimized_candidate,
        primary_post_opt_cluster,
        opacity_drop,
        args,
    )
    write_render_previews(out_dir, cameras, gaussians, pipe, background, pre_opacity, candidate_indices_t, drop_logit, args)

    print(f"Saved {out_dir / 'presence_optimization_scores.npz'}")
    print(f"Saved {out_dir / 'presence_optimization_scores.ply'}")
    print(f"Saved {out_dir / 'presence_optimization_summary.json'}")
    print(f"Saved {out_dir / 'post_optimization_clusters.json'}")
    print(f"Loss before/after: {loss_before:.6f} -> {loss_after:.6f}")
    print(f"Selected opacity-drop Gaussians: {int(optimized_candidate.sum())}")
    print(f"Post-opt clusters: {len(post_opt_clusters)} (noise {post_opt_noise_count})")
    print(f"Label: {summary['failure_label']}")


if __name__ == "__main__":
    main()
