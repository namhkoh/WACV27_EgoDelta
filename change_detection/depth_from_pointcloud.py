from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from scene.colmap_loader import read_points3D_binary, read_points3D_text
from utils.graphics_utils import fov2focal


def load_ply_xyz_rgb(path):
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    names = {prop.name for prop in vertices.properties}
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    else:
        rgb = np.full_like(xyz, 0.5, dtype=np.float32)
    return xyz, rgb


def load_colmap_points(source_path):
    sparse_dir = Path(source_path) / "sparse" / "0"
    bin_path = sparse_dir / "points3D.bin"
    txt_path = sparse_dir / "points3D.txt"
    if bin_path.exists():
        xyz, rgb, _ = read_points3D_binary(str(bin_path))
        return xyz.astype(np.float32), (rgb.astype(np.float32) / 255.0), str(bin_path)
    xyz, rgb, _ = read_points3D_text(str(txt_path))
    return xyz.astype(np.float32), (rgb.astype(np.float32) / 255.0), str(txt_path)


def load_post_point_cloud(post_source, baseline_task_dir=None):
    candidates = []
    if baseline_task_dir is not None:
        baseline_task_dir = Path(baseline_task_dir)
        candidates.extend(
            [
                baseline_task_dir / "dense_post.ply",
                baseline_task_dir / "colored_post.ply",
                baseline_task_dir / "post_only_voxels.ply",
            ]
        )
    for path in candidates:
        if path.exists():
            xyz, rgb = load_ply_xyz_rgb(path)
            return xyz, rgb, str(path)
    xyz, rgb, path = load_colmap_points(post_source)
    return xyz, rgb, path


def load_point_cloud_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud does not exist: {path}")
    if path.suffix.lower() == ".ply":
        xyz, rgb = load_ply_xyz_rgb(path)
        return xyz, rgb, str(path)
    if path.name.startswith("points3D"):
        try:
            xyz, rgb, _ = read_points3D_binary(str(path))
        except Exception:
            xyz, rgb, _ = read_points3D_text(str(path))
        return xyz.astype(np.float32), (rgb.astype(np.float32) / 255.0), str(path)
    if path.name == "0" and path.parent.name == "sparse":
        sparse_dir = path
        for candidate in (sparse_dir / "points3D.bin", sparse_dir / "points3D.txt"):
            if candidate.exists():
                return load_point_cloud_path(candidate)
        raise FileNotFoundError(f"No points3D.bin/txt found in {sparse_dir}")
    return load_post_point_cloud(path)


def _scatter_min(depth_flat, linear_idx, z):
    if hasattr(depth_flat, "scatter_reduce_"):
        depth_flat.scatter_reduce_(0, linear_idx, z, reduce="amin", include_self=True)
        return depth_flat

    order = torch.argsort(linear_idx)
    linear_sorted = linear_idx[order]
    z_sorted = z[order]
    unique_idx = torch.unique_consecutive(linear_sorted)
    for idx in unique_idx:
        values = z_sorted[linear_sorted == idx]
        depth_flat[idx] = torch.min(values)
    return depth_flat


@torch.no_grad()
def project_sparse_depth(points_world, camera, point_stride=1):
    if not torch.is_tensor(points_world):
        points_world = torch.as_tensor(points_world, dtype=torch.float32, device="cuda")
    points = points_world[:: max(1, point_stride)]
    device = points.device

    r_cw = torch.as_tensor(camera.R.T, dtype=torch.float32, device=device)
    t_cw = torch.as_tensor(camera.T, dtype=torch.float32, device=device)
    points_cam = points @ r_cw.T + t_cw

    z = points_cam[:, 2]
    valid_z = (z > camera.znear) & (z < camera.zfar)

    width = int(camera.image_width)
    height = int(camera.image_height)
    fx = fov2focal(camera.FoVx, width)
    fy = fov2focal(camera.FoVy, height)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5

    u = torch.round(fx * (points_cam[:, 0] / z.clamp_min(1e-6)) + cx).long()
    v = torch.round(fy * (points_cam[:, 1] / z.clamp_min(1e-6)) + cy).long()
    valid_pix = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    depth_flat = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    if valid_pix.any():
        linear = v[valid_pix] * width + u[valid_pix]
        depth_flat = _scatter_min(depth_flat, linear, z[valid_pix])

    depth = depth_flat.reshape(1, height, width)
    valid = torch.isfinite(depth)
    depth = torch.where(valid, depth, torch.zeros_like(depth))
    return depth, valid
