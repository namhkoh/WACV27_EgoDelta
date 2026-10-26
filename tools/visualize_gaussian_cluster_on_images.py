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
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import fov2focal


def parse_args():
    parser = argparse.ArgumentParser(description="Project scored Gaussian clusters onto post images")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", default="data/corl2026_2dgs")
    parser.add_argument("--baseline_root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cluster_id", default=0, type=int)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--max_frames", default=24, type=int)
    parser.add_argument("--post_frame_list", default=None)
    parser.add_argument("--single_trajectory", action="store_true")
    parser.add_argument("--draw_candidates", action="store_true")
    parser.add_argument("--point_radius", default=4, type=int)
    return parser.parse_args()


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
    return u.detach().cpu().numpy(), v.detach().cpu().numpy(), valid.detach().cpu().numpy()


def tensor_to_pil(image_tensor):
    arr = image_tensor[:3].detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
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


def make_contact_sheet(paths, out_path, cols=4, thumb_w=220):
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
        draw.text((x + 4, y + thumb_h), path.stem, fill=(235, 235, 235), font=font)
    canvas.save(out_path, quality=92)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    scores = np.load(args.score_npz)
    xyz = scores["xyz"].astype(np.float32)
    cluster_ids = scores["cluster_id"]
    candidate = scores["candidate"].astype(bool)
    cluster_mask = cluster_ids == args.cluster_id
    if not cluster_mask.any():
        raise RuntimeError(f"Cluster {args.cluster_id} has no members in {args.score_npz}")

    cluster_points = torch.from_numpy(xyz[cluster_mask]).cuda()
    candidate_points = torch.from_numpy(xyz[candidate]).cuda() if args.draw_candidates and candidate.any() else None

    written = []
    metadata = []
    for idx, camera in enumerate(cameras):
        image = tensor_to_pil(camera.original_image).convert("RGB")
        draw = ImageDraw.Draw(image)
        if candidate_points is not None:
            u, v, valid = project_points(candidate_points, camera)
            draw_points(draw, u, v, valid, (230, 40, 40), max(1, args.point_radius - 2))

        u, v, valid = project_points(cluster_points, camera)
        coords = draw_points(draw, u, v, valid, (255, 180, 20), args.point_radius)
        if coords:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            pad = args.point_radius * 3
            draw.rectangle((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), outline=(255, 210, 40), width=3)
        draw.text((8, 8), f"cluster {args.cluster_id}: {len(coords)}/{cluster_points.shape[0]} projected", fill=(255, 230, 80))

        out_path = out_dir / f"{idx:03d}_{camera.image_name}_cluster_{args.cluster_id}.jpg"
        image.save(out_path, quality=92)
        written.append(out_path)
        metadata.append(
            {
                "view": camera.image_name,
                "projected_cluster_points": len(coords),
                "cluster_points": int(cluster_points.shape[0]),
                "image": str(out_path),
            }
        )

    make_contact_sheet(written, out_dir / f"cluster_{args.cluster_id}_contact_sheet.jpg")
    (out_dir / f"cluster_{args.cluster_id}_projection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(written)} overlays to {out_dir}")
    print(f"Saved {out_dir / f'cluster_{args.cluster_id}_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
