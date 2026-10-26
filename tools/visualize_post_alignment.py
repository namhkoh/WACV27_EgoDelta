#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

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
    transform_points,
)
from change_detection.conflict_masks import build_conflict_masks
from change_detection.depth_from_pointcloud import load_point_cloud_path, load_post_point_cloud, project_sparse_depth
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import fov2focal
from utils.system_utils import searchForMaxIteration


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize post-to-pre alignment overlays")
    pipeline = PipelineParams(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", default="data/corl2026_2dgs")
    parser.add_argument("--baseline_root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--single_trajectory", action="store_true", help="Use identity pre/post alignment")
    parser.add_argument("--post_frame_list", default=None, help="Optional text file of post frame indices or image names to visualize")
    parser.add_argument("--post_point_cloud", default=None, help="Optional explicit post point cloud path")
    parser.add_argument("--pre_model", required=True)
    parser.add_argument("--pre_iteration", default=-1, type=int)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", default=2, type=int)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--num_views", default=16, type=int)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--depth_point_stride", default=1, type=int)
    parser.add_argument("--point_draw_stride", default=8, type=int)
    parser.add_argument("--alpha_min", default=0.2, type=float)
    parser.add_argument("--depth_margin", default=0.03, type=float)
    parser.add_argument("--rgb_thresh", default=0.12, type=float)
    parser.add_argument("--rgb_quantile", default=0.90, type=float)
    return parser.parse_args(), pipeline


def resolve_pre_ply(pre_model, iteration):
    path = Path(pre_model)
    if path.is_file():
        return path
    if iteration < 0:
        iteration = searchForMaxIteration(str(path / "point_cloud"))
    return path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def tensor_image(tensor):
    arr = tensor.detach().float().cpu()
    if arr.ndim == 3:
        arr = arr[:3].permute(1, 2, 0).numpy()
    else:
        arr = arr.numpy()
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def mask_image(mask, color):
    mask = mask.detach().bool().cpu().numpy().squeeze()
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = np.asarray(color, dtype=np.uint8)
    return out


def residual_heatmap(pre_depth, post_depth, valid, clip=0.15):
    pre = pre_depth.detach().float().cpu().numpy().squeeze()
    post = post_depth.detach().float().cpu().numpy().squeeze()
    valid_np = valid.detach().bool().cpu().numpy().squeeze()
    diff = np.zeros_like(pre, dtype=np.float32)
    diff[valid_np] = np.clip((pre[valid_np] - post[valid_np]) / clip, -1.0, 1.0)
    out = np.zeros((*pre.shape, 3), dtype=np.uint8)
    pos = diff > 0
    neg = diff < 0
    out[pos, 0] = (np.abs(diff[pos]) * 255).astype(np.uint8)
    out[neg, 2] = (np.abs(diff[neg]) * 255).astype(np.uint8)
    out[..., 1] = (valid_np.astype(np.uint8) * 40)
    return out


@torch.no_grad()
def project_pixels(points_world, camera, stride):
    points = points_world[:: max(1, stride)]
    device = points.device
    r_cw = torch.as_tensor(camera.R.T, dtype=torch.float32, device=device)
    t_cw = torch.as_tensor(camera.T, dtype=torch.float32, device=device)
    points_cam = points @ r_cw.T + t_cw
    z = points_cam[:, 2]
    width = int(camera.image_width)
    height = int(camera.image_height)
    fx = fov2focal(camera.FoVx, width)
    fy = fov2focal(camera.FoVy, height)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    u = torch.round(fx * (points_cam[:, 0] / z.clamp_min(1e-6)) + cx).long()
    v = torch.round(fy * (points_cam[:, 1] / z.clamp_min(1e-6)) + cy).long()
    valid = (z > camera.znear) & (z < camera.zfar) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[valid].detach().cpu().numpy(), v[valid].detach().cpu().numpy()


def draw_points(base, u, v, color=(0, 255, 255)):
    image = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y in zip(u.tolist(), v.tolist()):
        draw.point((int(x), int(y)), fill=color)
    return np.asarray(image)


def make_sheet(panels, labels):
    pil_panels = []
    for arr, label in zip(panels, labels):
        image = Image.fromarray(arr).convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + 24), "white")
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((6, image.height + 5), label, fill=(0, 0, 0))
        pil_panels.append(canvas)
    width = pil_panels[0].width * 2
    height = pil_panels[0].height * 3
    sheet = Image.new("RGB", (width, height), "white")
    for idx, panel in enumerate(pil_panels):
        sheet.paste(panel, ((idx % 2) * panel.width, (idx // 2) * panel.height))
    return sheet


def main():
    args, pipeline_group = parse_args()
    out_dir = Path(args.model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, post_dataset = resolve_task_datasets(args.data_root, args.task)
    use_identity_alignment = args.single_trajectory or manifest_uses_identity_alignment(args.data_root)
    if use_identity_alignment:
        alignment_manifest, t_pre_post, alignment_path = identity_alignment("identity_single_trajectory")
        baseline_task_dir = None
    else:
        alignment_manifest, t_pre_post, alignment_path = resolve_task_alignment(args.baseline_root, args.task)
        baseline_task_dir = Path(args.baseline_root) / "out" / args.task

    pipe = pipeline_group.extract(args)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(resolve_pre_ply(args.pre_model, args.pre_iteration)))

    camera_infos = load_colmap_camera_infos(post_dataset["dataset_dir"], images=args.images)
    camera_infos = filter_camera_infos_by_frame_list(camera_infos, args.post_frame_list)
    camera_infos = select_camera_infos(camera_infos, args.frame_stride, args.max_frames)
    if args.num_views and len(camera_infos) > args.num_views:
        ids = np.linspace(0, len(camera_infos) - 1, args.num_views).round().astype(np.int64)
        camera_infos = [camera_infos[int(i)] for i in ids]
    camera_infos = transform_camera_infos_to_pre(camera_infos, t_pre_post)
    cameras = cameraList_from_camInfos(
        camera_infos,
        1.0,
        Namespace(resolution=args.resolution, data_device=args.data_device),
    )
    if not cameras:
        raise RuntimeError("No post cameras were loaded")

    if args.post_point_cloud:
        post_points, _, point_cloud_path = load_point_cloud_path(args.post_point_cloud)
    else:
        post_points, _, point_cloud_path = load_post_point_cloud(post_dataset["dataset_dir"], baseline_task_dir)
    post_points = torch.from_numpy(transform_points(post_points, t_pre_post).astype(np.float32)).cuda()
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    rows = []
    for idx, camera in enumerate(cameras):
        pre_pack = render(camera, gaussians, pipe, background)
        post_depth, valid_post_depth = project_sparse_depth(
            post_points,
            camera,
            point_stride=args.depth_point_stride,
        )
        masks = build_conflict_masks(
            pre_pack,
            camera.original_image.cuda(),
            post_depth=post_depth,
            valid_post_depth=valid_post_depth,
            alpha_min=args.alpha_min,
            depth_margin=args.depth_margin,
            rgb_thresh=args.rgb_thresh,
            rgb_quantile=args.rgb_quantile,
            use_rgb_conflict=True,
            use_depth_conflict=True,
        )
        u, v = project_pixels(post_points, camera, args.point_draw_stride)
        post_rgb = tensor_image(camera.original_image)
        pre_rgb = tensor_image(masks["pre_rgb"])
        post_overlay = draw_points(post_rgb, u, v)
        pre_overlay = draw_points(pre_rgb, u, v)
        heatmap = residual_heatmap(masks["pre_depth"], post_depth, valid_post_depth)
        m_free = mask_image(masks["M_free"], (255, 0, 0))
        m_new = mask_image(masks["M_new_occlusion"], (0, 0, 255))
        m_rgb = mask_image(masks["M_rgb"], (255, 255, 0))
        sheet = make_sheet(
            [post_overlay, pre_overlay, heatmap, m_free, m_new, m_rgb],
            [
                "post RGB + projected post cloud",
                "pre render + projected post cloud",
                "depth residual: red=pre closer, blue=post closer",
                "M_free strong removal mask",
                "M_new_occlusion excluded mask",
                "M_rgb weak color mask",
            ],
        )
        name = f"{idx:03d}_{camera.image_name}"
        sheet.save(out_dir / f"{name}_alignment_overlay.jpg", quality=95)

        valid = masks["valid_mask"]
        common = valid_post_depth & valid
        abs_diff = torch.abs(masks["pre_depth"] - post_depth)
        rows.append(
            {
                "view": name,
                "projected_points": int(u.shape[0]),
                "valid_pre_ratio": float(valid.float().mean().item()),
                "valid_post_depth_ratio": float(valid_post_depth.float().mean().item()),
                "common_depth_ratio": float(common.float().mean().item()),
                "median_abs_depth_diff_common": float(torch.median(abs_diff[common]).item()) if common.any() else 0.0,
                "mean_abs_depth_diff_common": float(torch.mean(abs_diff[common]).item()) if common.any() else 0.0,
                "M_free_ratio": float(masks["M_free"].float().mean().item()),
                "M_new_occlusion_ratio": float(masks["M_new_occlusion"].float().mean().item()),
                "M_rgb_ratio": float(masks["M_rgb"].float().mean().item()),
                "rgb_residual_valid_mean": float(masks["rgb_residual"][valid].mean().item()) if valid.any() else 0.0,
            }
        )

    with open(out_dir / "alignment_overlay_metrics.csv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "alignment_overlay_config.json", "w") as fp:
        json.dump(
            {
                "task": args.task,
                "alignment_manifest": str(alignment_path),
                "alignment_method": alignment_manifest.get("alignment_method"),
                "post_point_cloud": str(point_cloud_path),
                "post_frame_list": str(args.post_frame_list),
                "num_views": len(cameras),
                "resolution": args.resolution,
                "depth_point_stride": args.depth_point_stride,
                "point_draw_stride": args.point_draw_stride,
            },
            fp,
            indent=2,
        )
    print(f"Saved overlays and metrics to {out_dir}")


if __name__ == "__main__":
    main()
