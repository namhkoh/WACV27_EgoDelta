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
from change_detection.alignment import DEFAULT_BASELINE_ROOT
from change_detection.projection import load_projection_cameras, project_points
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


COLORS = {
    0: (255, 170, 30),
    1: (80, 170, 255),
    2: (255, 80, 180),
    3: (80, 220, 100),
    4: (210, 210, 80),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create side-by-side leaf mask visualizations: rendered support mask and projected Gaussian points."
    )
    pipeline = PipelineParams(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pre_model", required=True)
    parser.add_argument("--pre_iteration", default=-1, type=int)
    parser.add_argument("--connection_summary_json", default=None)
    parser.add_argument("--selected_clusters", default="all")
    parser.add_argument(
        "--mask_key",
        choices=(
            "optimized_candidate",
            "completed_candidate",
            "connection_path_candidate",
            "leaf_instance_candidate",
        ),
        default="optimized_candidate",
    )
    parser.add_argument("--cluster_key", default=None)
    parser.add_argument("--support_weight_key", choices=("opacity_drop", "opacity_pre"), default="opacity_drop")

    parser.add_argument("--baseline_root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--camera_scene", choices=("pre", "post"), default="pre")
    parser.add_argument("--post_frame_list", default=None)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", default=2, type=int)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--frame_stride", default=6, type=int)
    parser.add_argument("--max_frames", default=18, type=int)

    parser.add_argument("--support_quantile", default=0.98, type=float)
    parser.add_argument("--support_threshold", default=0.05, type=float)
    parser.add_argument("--max_points_per_cluster", default=1800, type=int)
    parser.add_argument("--point_radius", default=3, type=int)
    parser.add_argument("--point_style", choices=("blob", "pointcloud"), default="blob")
    parser.add_argument("--point_color_mode", choices=("cluster", "yellow"), default="cluster")
    parser.add_argument("--target_radius", default=8, type=int)
    parser.add_argument("--hide_target_labels", action="store_true")
    parser.add_argument("--crop_pad", default=100, type=int)
    parser.add_argument("--min_crop", default=360, type=int)
    parser.add_argument("--max_crop", default=980, type=int)
    parser.add_argument("--zoom_size", default=520, type=int)
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


def parse_cluster_ids(spec, available):
    spec = str(spec).strip().lower()
    if spec in ("all", "-1", ""):
        return sorted(int(v) for v in available if int(v) >= 0)
    requested = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    missing = sorted(set(requested) - {int(v) for v in available})
    if missing:
        raise RuntimeError(f"Requested clusters {missing} are not present")
    return requested


def tensor_to_pil(image_tensor):
    arr = image_tensor[:3].detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def normalize_scalar(scalar, quantile):
    scalar = np.clip(scalar, 0.0, None)
    scale = float(np.quantile(scalar[scalar > 0.0], quantile)) if (scalar > 0.0).any() else 1.0
    return np.clip(scalar / max(scale, 1e-8), 0.0, 1.0)


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
    return overlay * (1.0 - alpha) + color * alpha, mask


def crop_box(us, vs, valid, width, height, args):
    if not valid.any():
        return (0, 0, width, height)
    x0 = float(us[valid].min())
    x1 = float(us[valid].max())
    y0 = float(vs[valid].min())
    y1 = float(vs[valid].max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) + 2.0 * args.crop_pad
    side = min(max(side, args.min_crop), args.max_crop, max(width, height))
    left = int(round(max(0.0, min(cx - side * 0.5, width - side))))
    top = int(round(max(0.0, min(cy - side * 0.5, height - side))))
    return (left, top, int(round(left + side)), int(round(top + side)))


def draw_mask_box(image, mask):
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    ImageDraw.Draw(image).rectangle(box, outline=(255, 245, 80), width=2)
    return box


def sample_points(points, max_points, seed):
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(points.shape[0], max_points, replace=False)]


def load_target_points(path, selected_clusters):
    if path is None:
        return []
    data = json.loads(Path(path).read_text())
    targets = []
    selected = set(selected_clusters)
    for item in data.get("clusters", []):
        cluster_id = int(item["cluster_id"])
        target = item.get("target_xyz")
        if cluster_id not in selected or target is None:
            continue
        targets.append({"cluster_id": cluster_id, "point": np.asarray(target, dtype=np.float32)})
    return targets


def make_contact_sheet(paths, out_path, cols=2, thumb_w=360):
    if not paths:
        return
    font = ImageFont.load_default()
    label_h = 18
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        thumb_h = int(round(thumb_w * img.height / img.width))
        thumbs.append((path, img.resize((thumb_w, thumb_h))))
    thumb_h = max(img.height for _, img in thumbs)
    rows = int(math.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (path, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + thumb_h), path.stem[:48], fill=(235, 235, 235), font=font)
    canvas.save(out_path, quality=92)


def draw_pointcloud_point(draw, x, y, color, radius):
    if radius <= 1:
        draw.point((x, y), fill=color)
        return
    draw.point((x, y), fill=color)
    draw.point((x - 1, y), fill=color)
    draw.point((x + 1, y), fill=color)
    draw.point((x, y - 1), fill=color)
    draw.point((x, y + 1), fill=color)


def main():
    args, pipeline_group = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to render the support mask")
    torch.cuda.set_device(torch.device("cuda:0"))

    out_dir = Path(args.out_dir)
    support_dir = out_dir / "support_mask"
    points_dir = out_dir / "points"
    support_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    presence = np.load(args.presence_npz)
    xyz = presence["xyz"].astype(np.float32)
    if args.mask_key not in presence.files:
        raise RuntimeError(f"{args.mask_key} is absent from {args.presence_npz}")
    selected_all = presence[args.mask_key].astype(bool)
    cluster_key = args.cluster_key
    if cluster_key is None:
        cluster_key = "completed_cluster_id" if args.mask_key == "completed_candidate" and "completed_cluster_id" in presence.files else "post_opt_cluster_id"
    if cluster_key not in presence.files:
        raise RuntimeError(f"{cluster_key} is absent from {args.presence_npz}")
    cluster_id = presence[cluster_key].astype(np.int64)
    available_clusters = sorted(int(v) for v in np.unique(cluster_id[selected_all]) if int(v) >= 0)
    selected_clusters = parse_cluster_ids(args.selected_clusters, available_clusters)
    selected = selected_all & np.isin(cluster_id, np.asarray(selected_clusters, dtype=np.int64))
    if not selected.any():
        raise RuntimeError("No Gaussians selected for visualization")

    support_weights = presence[args.support_weight_key].astype(np.float32).reshape(-1, 1)
    selected_support_np = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    selected_support_np[selected] = np.maximum(support_weights[selected], 0.0)
    selected_support = torch.from_numpy(selected_support_np).cuda()

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

    camera_args = Namespace(
        task=args.task,
        data_root=args.data_root,
        baseline_root=args.baseline_root,
        single_trajectory=args.single_trajectory,
        camera_scene=args.camera_scene,
        images=args.images,
        resolution=args.resolution,
        data_device=args.data_device,
        post_frame_list=args.post_frame_list,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    cameras = load_projection_cameras(camera_args)
    pipe = pipeline_group.extract(args)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    cluster_payload = []
    for cid in selected_clusters:
        mask = selected & (cluster_id == cid)
        points = sample_points(xyz[mask], args.max_points_per_cluster, 1000 + cid)
        if points.shape[0]:
            cluster_payload.append((cid, torch.from_numpy(points).float().cuda()))

    selected_points = torch.from_numpy(xyz[selected]).float().cuda()
    target_items = load_target_points(args.connection_summary_json, selected_clusters)
    target_points = (
        torch.from_numpy(np.asarray([item["point"] for item in target_items], dtype=np.float32)).float().cuda()
        if target_items
        else None
    )

    support_paths = []
    point_paths = []
    metadata = []
    for frame_idx, camera in enumerate(cameras):
        base_img = tensor_to_pil(camera.original_image).convert("RGB")
        base_rgb = np.asarray(base_img).astype(np.float32) / 255.0

        su, sv, svalid = project_points(selected_points, camera)
        all_u = [su]
        all_v = [sv]
        all_valid = [svalid]
        projected = []
        for cid, points_t in cluster_payload:
            u, v, valid = project_points(points_t, camera)
            all_u.append(u)
            all_v.append(v)
            all_valid.append(valid)
            projected.append((cid, u, v, valid))
        if target_points is not None:
            tu, tv, tvalid = project_points(target_points, camera)
            all_u.append(tu)
            all_v.append(tv)
            all_valid.append(tvalid)
        else:
            tu, tv, tvalid = np.array([]), np.array([]), np.array([], dtype=bool)

        box = crop_box(
            np.concatenate(all_u),
            np.concatenate(all_v),
            np.concatenate(all_valid),
            base_img.width,
            base_img.height,
            args,
        )

        with torch.no_grad():
            support_pack = render(camera, gaussians, pipe, background, override_opacity=selected_support)
            support_alpha = torch.clamp(support_pack["rend_alpha"], 0.0, 1.0)[0].detach().cpu().numpy()

        left, top, right, bottom = box
        support_crop = support_alpha[top:bottom, left:right]
        base_crop = base_rgb[top:bottom, left:right]
        support_rgb, support_mask = support_overlay(base_crop, support_crop, args.support_quantile, args.support_threshold)
        support_img = Image.fromarray((np.clip(support_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
        draw_mask_box(support_img, support_mask)

        points_img = base_img.crop(box)
        sx = (right - left) / max(1, points_img.width)
        sy = (bottom - top) / max(1, points_img.height)
        sx = args.zoom_size / max(1, right - left)
        sy = args.zoom_size / max(1, bottom - top)
        support_img = support_img.resize((args.zoom_size, args.zoom_size), Image.Resampling.BILINEAR)
        points_img = points_img.resize((args.zoom_size, args.zoom_size), Image.Resampling.BILINEAR)
        support_draw = ImageDraw.Draw(support_img)
        points_draw = ImageDraw.Draw(points_img)

        for cid, u, v, valid in projected:
            color = (255, 180, 30) if args.point_color_mode == "yellow" else COLORS.get(cid, (255, 180, 40))
            for uu, vv, ok in zip(u, v, valid):
                if not ok:
                    continue
                x = int(round((float(uu) - left) * sx))
                y = int(round((float(vv) - top) * sy))
                if args.point_style == "pointcloud":
                    draw_pointcloud_point(points_draw, x, y, color, args.point_radius)
                else:
                    r = args.point_radius
                    points_draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)

        for item, uu, vv, ok in zip(target_items, tu, tv, tvalid):
            if not ok:
                continue
            x = int(round((float(uu) - left) * sx))
            y = int(round((float(vv) - top) * sy))
            color = COLORS.get(item["cluster_id"], (255, 255, 255))
            r = args.target_radius
            for draw in (support_draw, points_draw):
                draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=3)
                draw.line((x - r - 4, y, x + r + 4, y), fill=color, width=3)
                draw.line((x, y - r - 4, x, y + r + 4), fill=color, width=3)
                if not args.hide_target_labels:
                    draw.text((x + r + 4, y - r), f"T{item['cluster_id']}", fill=(255, 255, 255), font=ImageFont.load_default())

        support_path = support_dir / f"{frame_idx:03d}_{camera.image_name}_support_mask.jpg"
        point_path = points_dir / f"{frame_idx:03d}_{camera.image_name}_points.jpg"
        support_img.save(support_path, quality=92)
        points_img.save(point_path, quality=92)
        support_paths.append(support_path)
        point_paths.append(point_path)
        metadata.append(
            {
                "frame": camera.image_name,
                "crop_box": list(box),
                "visible_selected_points": int(svalid.sum()),
                "support_mask_pixels": int(support_mask.sum()),
                "support_image": str(support_path),
                "point_image": str(point_path),
            }
        )

    make_contact_sheet(support_paths, out_dir / "support_mask_contact_sheet.jpg")
    make_contact_sheet(point_paths, out_dir / "points_contact_sheet.jpg")
    (out_dir / "visualization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved support mask contact sheet: {out_dir / 'support_mask_contact_sheet.jpg'}")
    print(f"Saved point contact sheet: {out_dir / 'points_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
