#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import fov2focal
from utils.system_utils import searchForMaxIteration


def parse_args():
    parser = argparse.ArgumentParser(description="Create zoomed verification views for optimized Gaussian presence drops")
    pipeline = PipelineParams(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", default="data/corl2026_2dgs_single")
    parser.add_argument("--baseline_root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--post_frame_list", default=None)
    parser.add_argument("--pre_model", required=True)
    parser.add_argument("--pre_iteration", default=-1, type=int)
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--post_opt_cluster_id",
        default=0,
        type=int,
        help="Post-optimization cluster to visualize. Use -1 to visualize all optimized candidates.",
    )

    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_frames", default=24, type=int)

    parser.add_argument("--crop_pad", default=80, type=int)
    parser.add_argument("--min_crop", default=220, type=int)
    parser.add_argument("--max_crop", default=520, type=int)
    parser.add_argument("--zoom_size", default=320, type=int)
    parser.add_argument("--point_radius", default=3, type=int)
    parser.add_argument("--heat_quantile", default=0.995, type=float)
    parser.add_argument("--mask_threshold", default=0.25, type=float)
    parser.add_argument("--support_quantile", default=0.98, type=float)
    parser.add_argument("--support_threshold", default=0.05, type=float)
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


def load_post_cameras(args):
    _, post_dataset = resolve_task_datasets(args.data_root, args.task)
    use_identity = args.single_trajectory or manifest_uses_identity_alignment(args.data_root)
    if use_identity:
        _, t_pre_post, _ = identity_alignment("identity_single_trajectory")
    else:
        _, t_pre_post, _ = resolve_task_alignment(args.baseline_root, args.task)
    camera_infos = load_colmap_camera_infos(post_dataset["dataset_dir"], images=args.images)
    camera_infos = filter_camera_infos_by_frame_list(camera_infos, args.post_frame_list)
    camera_infos = select_camera_infos(camera_infos, args.frame_stride, args.max_frames)
    camera_infos = transform_camera_infos_to_pre(camera_infos, t_pre_post)
    cameras = cameraList_from_camInfos(camera_infos, 1.0, build_camera_args(args))
    if not cameras:
        raise RuntimeError("No post cameras were loaded")
    return cameras


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


def tensor_to_rgb_array(image_tensor):
    arr = image_tensor[:3].detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def rgb_array_to_pil(arr):
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def crop_box_from_points(us, vs, valid, width, height, args):
    if not valid.any():
        return None
    xs = us[valid]
    ys = vs[valid]
    x0 = float(xs.min())
    x1 = float(xs.max())
    y0 = float(ys.min())
    y1 = float(ys.max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) + 2.0 * args.crop_pad
    side = min(max(side, float(args.min_crop)), float(args.max_crop))
    left = int(round(cx - 0.5 * side))
    top = int(round(cy - 0.5 * side))
    right = left + int(round(side))
    bottom = top + int(round(side))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left = max(0, left - (right - width))
        right = width
    if bottom > height:
        top = max(0, top - (bottom - height))
        bottom = height
    return (left, top, right, bottom)


def crop_rgb(arr, box):
    left, top, right, bottom = box
    return arr[top:bottom, left:right]


def scalar_heat_overlay(base_rgb, scalar, quantile):
    scalar = np.clip(scalar, 0.0, None)
    scale = float(np.quantile(scalar[scalar > 0.0], quantile)) if (scalar > 0.0).any() else 1.0
    norm = np.clip(scalar / max(scale, 1e-8), 0.0, 1.0)
    heat = np.zeros_like(base_rgb)
    heat[..., 0] = 1.0
    heat[..., 1] = 0.15 + 0.85 * norm
    heat[..., 2] = 0.02
    alpha = 0.75 * norm[..., None]
    return base_rgb * (1.0 - alpha) + heat * alpha


def normalize_scalar(scalar, quantile):
    scalar = np.clip(scalar, 0.0, None)
    scale = float(np.quantile(scalar[scalar > 0.0], quantile)) if (scalar > 0.0).any() else 1.0
    return np.clip(scalar / max(scale, 1e-8), 0.0, 1.0)


def mask_overlay(base_rgb, scalar, quantile, threshold):
    norm = normalize_scalar(scalar, quantile)
    mask = norm >= threshold
    overlay = base_rgb.copy()
    heat = np.zeros_like(base_rgb)
    heat[..., 0] = 1.0
    heat[..., 1] = 0.25 + 0.75 * norm
    heat[..., 2] = 0.02
    alpha = np.zeros((*norm.shape, 1), dtype=np.float32)
    alpha[mask] = (0.25 + 0.55 * norm[mask])[..., None]
    overlay = overlay * (1.0 - alpha) + heat * alpha
    return overlay, mask


def support_overlay(base_rgb, support, quantile, threshold):
    norm = normalize_scalar(support, quantile)
    mask = norm >= threshold
    overlay = base_rgb.copy()
    color = np.zeros_like(base_rgb)
    color[..., 0] = 0.05
    color[..., 1] = 1.0
    color[..., 2] = 0.25
    alpha = np.zeros((*norm.shape, 1), dtype=np.float32)
    alpha[mask] = (0.20 + 0.55 * norm[mask])[..., None]
    overlay = overlay * (1.0 - alpha) + color * alpha
    return overlay, mask


def draw_mask_box(image, mask):
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(255, 245, 60), width=2)
    return box


def scalar_to_heatmap(scalar, quantile):
    scalar = np.clip(scalar, 0.0, None)
    scale = float(np.quantile(scalar[scalar > 0.0], quantile)) if (scalar > 0.0).any() else 1.0
    norm = np.clip(scalar / max(scale, 1e-8), 0.0, 1.0)
    heat = np.zeros((*scalar.shape, 3), dtype=np.float32)
    heat[..., 0] = norm
    heat[..., 1] = np.sqrt(norm)
    heat[..., 2] = 0.08 * (1.0 - norm)
    return heat


def draw_projected_points(image, us, vs, valid, box, radius):
    left, top, _, _ = box
    draw = ImageDraw.Draw(image)
    coords = []
    for u, v, ok in zip(us, vs, valid):
        if not ok:
            continue
        x = int(round(float(u) - left))
        y = int(round(float(v) - top))
        coords.append((x, y))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 235, 40), width=2)
    if coords:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        pad = max(4, radius * 3)
        draw.rectangle((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), outline=(255, 245, 80), width=2)
    return len(coords)


def label_image(image, label):
    font = ImageFont.load_default()
    label_h = 18
    canvas = Image.new("RGB", (image.width, image.height + label_h), (24, 24, 24))
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, image.height + 3), label, fill=(235, 235, 235), font=font)
    return canvas


def resize_labeled(image, label, size):
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return label_image(resized, label)


def make_row(images, labels, size):
    labeled = [resize_labeled(image, label, size) for image, label in zip(images, labels)]
    canvas = Image.new("RGB", (len(labeled) * size, labeled[0].height), (18, 18, 18))
    for idx, image in enumerate(labeled):
        canvas.paste(image, (idx * size, 0))
    return canvas


def make_contact_sheet(paths, out_path, cols=1):
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = int(math.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * width, rows * height), (18, 18, 18))
    for idx, image in enumerate(images):
        x = (idx % cols) * width
        y = (idx // cols) * height
        canvas.paste(image, (x, y))
    canvas.save(out_path, quality=92)


def main():
    args, pipeline_group = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This verifier requires CUDA because it re-renders the scene")
    torch.cuda.set_device(torch.device("cuda:0"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    outline_dir = out_dir / "outline_frames"
    outline_dir.mkdir(parents=True, exist_ok=True)

    scores = np.load(args.presence_npz)
    xyz = scores["xyz"].astype(np.float32)
    if args.post_opt_cluster_id < 0 and "optimized_candidate" in scores:
        selected = scores["optimized_candidate"].astype(bool)
    elif "primary_post_opt_cluster" in scores and args.post_opt_cluster_id == 0:
        selected = scores["primary_post_opt_cluster"].astype(bool)
    elif "post_opt_cluster_id" in scores:
        selected = scores["post_opt_cluster_id"].astype(np.int64) == args.post_opt_cluster_id
    else:
        selected = scores["optimized_candidate"].astype(bool)
    if not selected.any():
        raise RuntimeError("No selected optimized Gaussians were found for visualization")

    pre_opacity = torch.from_numpy(scores["opacity_pre"].astype(np.float32).reshape(-1, 1)).cuda()
    optimized_opacity = torch.from_numpy(scores["opacity_optimized"].astype(np.float32).reshape(-1, 1)).cuda()
    opacity_drop_np = scores["opacity_drop"].astype(np.float32).reshape(-1, 1)
    selected_support_opacity_np = np.zeros_like(opacity_drop_np)
    selected_support_opacity_np[selected] = opacity_drop_np[selected]
    selected_support_opacity = torch.from_numpy(selected_support_opacity_np).cuda()
    selected_points = torch.from_numpy(xyz[selected]).cuda()

    pre_ply = resolve_pre_ply(args.pre_model, args.pre_iteration)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(pre_ply))
    for tensor in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ):
        tensor.requires_grad_(False)

    cameras = load_post_cameras(args)
    pipe = pipeline_group.extract(args)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    rows = []
    outline_rows = []
    metadata = []
    for idx, camera in enumerate(cameras):
        with torch.no_grad():
            pre_render = torch.clamp(render(camera, gaussians, pipe, background, override_opacity=pre_opacity)["render"], 0.0, 1.0)
            opt_render = torch.clamp(render(camera, gaussians, pipe, background, override_opacity=optimized_opacity)["render"], 0.0, 1.0)
            support_pack = render(camera, gaussians, pipe, background, override_opacity=selected_support_opacity)
            support_alpha = torch.clamp(support_pack["rend_alpha"], 0.0, 1.0)
        post = torch.clamp(camera.original_image.cuda(), 0.0, 1.0)

        post_rgb = tensor_to_rgb_array(post)
        pre_rgb = tensor_to_rgb_array(pre_render)
        opt_rgb = tensor_to_rgb_array(opt_render)
        support = support_alpha[0].detach().cpu().numpy()
        delta = np.mean(np.abs(pre_rgb - opt_rgb), axis=2)
        residual_pre = np.mean(np.abs(pre_rgb - post_rgb), axis=2)
        residual_opt = np.mean(np.abs(opt_rgb - post_rgb), axis=2)

        u, v, valid = project_points(selected_points, camera)
        box = crop_box_from_points(u, v, valid, int(camera.image_width), int(camera.image_height), args)
        if box is None:
            continue

        post_crop = crop_rgb(post_rgb, box)
        pre_crop = crop_rgb(pre_rgb, box)
        opt_crop = crop_rgb(opt_rgb, box)
        delta_crop = crop_rgb(delta, box)
        support_crop = crop_rgb(support, box)
        residual_pre_crop = crop_rgb(residual_pre, box)
        residual_opt_crop = crop_rgb(residual_opt, box)

        post_points = rgb_array_to_pil(post_crop)
        visible_count = draw_projected_points(post_points, u, v, valid, box, args.point_radius)
        overlay = rgb_array_to_pil(scalar_heat_overlay(post_crop, delta_crop, args.heat_quantile))
        draw_projected_points(overlay, u, v, valid, box, args.point_radius)
        post_mask_overlay, removed_mask = mask_overlay(post_crop, delta_crop, args.heat_quantile, args.mask_threshold)
        pre_mask_overlay, _ = mask_overlay(pre_crop, delta_crop, args.heat_quantile, args.mask_threshold)
        opt_mask_overlay, _ = mask_overlay(opt_crop, delta_crop, args.heat_quantile, args.mask_threshold)
        post_mask_image = rgb_array_to_pil(post_mask_overlay)
        pre_mask_image = rgb_array_to_pil(pre_mask_overlay)
        opt_mask_image = rgb_array_to_pil(opt_mask_overlay)
        mask_box = draw_mask_box(post_mask_image, removed_mask)
        draw_mask_box(pre_mask_image, removed_mask)
        draw_mask_box(opt_mask_image, removed_mask)
        support_mask_overlay, support_mask = support_overlay(
            post_crop,
            support_crop,
            args.support_quantile,
            args.support_threshold,
        )
        support_mask_image = rgb_array_to_pil(support_mask_overlay)
        support_mask_box = draw_mask_box(support_mask_image, support_mask)

        row = make_row(
            [
                post_points,
                overlay,
                support_mask_image,
                rgb_array_to_pil(pre_crop),
                rgb_array_to_pil(opt_crop),
                rgb_array_to_pil(scalar_to_heatmap(residual_pre_crop, args.heat_quantile)),
                rgb_array_to_pil(scalar_to_heatmap(residual_opt_crop, args.heat_quantile)),
            ],
            [
                f"{camera.image_name} post+points",
                "removed footprint",
                "selected support",
                "pre render",
                "optimized render",
                "pre residual",
                "opt residual",
            ],
            args.zoom_size,
        )
        row_path = frames_dir / f"{idx:03d}_{camera.image_name}_verification_zoom.jpg"
        row.save(row_path, quality=92)
        rows.append(row_path)
        outline_row = make_row(
            [
                rgb_array_to_pil(post_crop),
                support_mask_image,
                post_mask_image,
                pre_mask_image,
                opt_mask_image,
                rgb_array_to_pil(scalar_to_heatmap(delta_crop, args.heat_quantile)),
            ],
            [
                f"{camera.image_name} post",
                "post+support mask",
                "post+removed mask",
                "pre render+mask",
                "optimized+mask",
                "removed heat",
            ],
            args.zoom_size,
        )
        outline_path = outline_dir / f"{idx:03d}_{camera.image_name}_removed_outline.jpg"
        outline_row.save(outline_path, quality=92)
        outline_rows.append(outline_path)
        metadata.append(
            {
                "view": camera.image_name,
                "image": str(row_path),
                "outline_image": str(outline_path),
                "crop_box": list(box),
                "visible_selected_points": int(visible_count),
                "support_mask_box_in_crop": list(support_mask_box) if support_mask_box is not None else None,
                "support_mask_pixels": int(support_mask.sum()),
                "mean_selected_support": float(support_crop.mean()),
                "max_selected_support": float(support_crop.max()),
                "removed_mask_box_in_crop": list(mask_box) if mask_box is not None else None,
                "removed_mask_pixels": int(removed_mask.sum()),
                "mean_removed_footprint": float(delta_crop.mean()),
                "max_removed_footprint": float(delta_crop.max()),
                "mean_pre_residual_crop": float(residual_pre_crop.mean()),
                "mean_opt_residual_crop": float(residual_opt_crop.mean()),
                "crop_loss_improvement": float(residual_pre_crop.mean() - residual_opt_crop.mean()),
            }
        )

    make_contact_sheet(rows, out_dir / "verification_zoom_contact_sheet.jpg", cols=1)
    make_contact_sheet(outline_rows, out_dir / "verification_removed_outline_contact_sheet.jpg", cols=1)
    (out_dir / "verification_zoom_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(rows)} zoom rows to {frames_dir}")
    print(f"Saved {len(outline_rows)} outline rows to {outline_dir}")
    print(f"Saved {out_dir / 'verification_zoom_contact_sheet.jpg'}")
    print(f"Saved {out_dir / 'verification_removed_outline_contact_sheet.jpg'}")
    print(f"Saved {out_dir / 'verification_zoom_metadata.json'}")


if __name__ == "__main__":
    main()
