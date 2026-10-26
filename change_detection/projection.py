from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
from PIL import Image

from change_detection.alignment import (
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


def build_camera_args(args):
    return Namespace(resolution=args.resolution, data_device=args.data_device)


def load_projection_cameras(args):
    pre_dataset, post_dataset = resolve_task_datasets(args.data_root, args.task)
    dataset = pre_dataset if args.camera_scene == "pre" else post_dataset
    use_identity = args.single_trajectory or manifest_uses_identity_alignment(args.data_root)
    if use_identity:
        _, t_pre_post, _ = identity_alignment("identity_single_trajectory")
    else:
        _, t_pre_post, _ = resolve_task_alignment(args.baseline_root, args.task)
    camera_infos = load_colmap_camera_infos(dataset["dataset_dir"], images=args.images)
    camera_infos = filter_camera_infos_by_frame_list(camera_infos, args.post_frame_list)
    camera_infos = select_camera_infos(camera_infos, args.frame_stride, args.max_frames)
    if args.camera_scene == "post":
        camera_infos = transform_camera_infos_to_pre(camera_infos, t_pre_post)
    cameras = cameraList_from_camInfos(camera_infos, 1.0, build_camera_args(args))
    if not cameras:
        raise RuntimeError(f"No {args.camera_scene} cameras were loaded")
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


def tensor_to_pil(image_tensor):
    arr = image_tensor[:3].detach().cpu().permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)
