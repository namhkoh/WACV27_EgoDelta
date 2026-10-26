from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import CameraInfo, readColmapCameras


DEFAULT_BASELINE_ROOT = Path(
    "/path/to/workspace/CoRL2026/semidense_geo_baseline"
)


def read_json(path):
    with open(path, "r") as fp:
        return json.load(fp)


def resolve_task_datasets(data_root, task):
    manifest = read_json(Path(data_root) / "manifest.json")
    datasets = {
        entry["name"]: entry
        for entry in manifest["datasets"]
        if entry.get("task") == task
    }
    pre = datasets.get(f"{task}_pre")
    post = datasets.get(f"{task}_post")
    if pre is None or post is None:
        raise ValueError(f"Could not find pre/post datasets for task '{task}' in {data_root}")
    return pre, post


def manifest_uses_identity_alignment(data_root):
    manifest = read_json(Path(data_root) / "manifest.json")
    if manifest.get("single_trajectory"):
        return True
    if manifest.get("alignment") == "identity":
        return True
    return manifest.get("alignment_method") == "identity_single_trajectory"


def resolve_task_alignment(baseline_root, task):
    path = Path(baseline_root) / "out" / task / f"manifest_{task}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing alignment manifest: {path}")
    manifest = read_json(path)
    if "T_pre_post" not in manifest:
        raise KeyError(f"Alignment manifest has no T_pre_post: {path}")
    return manifest, np.asarray(manifest["T_pre_post"], dtype=np.float64), path


def identity_alignment(label="identity"):
    manifest = {"alignment_method": label, "T_pre_post": np.eye(4).tolist()}
    return manifest, np.eye(4, dtype=np.float64), label


def load_colmap_camera_infos(source_path, images="images"):
    sparse_dir = Path(source_path) / "sparse" / "0"
    try:
        extrinsics = read_extrinsics_binary(str(sparse_dir / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(str(sparse_dir / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse_dir / "cameras.txt"))

    images_folder = os.path.join(source_path, "images" if images is None else images)
    cameras = readColmapCameras(extrinsics, intrinsics, images_folder)
    return sorted(cameras, key=lambda x: x.image_name)


def _normalize_frame_token(token):
    token = token.strip()
    if not token or token.startswith("#"):
        return None
    token = token.replace(",", " ").split()[0].strip()
    if not token:
        return None
    if token.lower() in {"frame_idx", "frame", "image", "image_name", "name"}:
        return None
    if token.isdigit():
        return f"frame_{int(token):06d}"
    return Path(token).stem


def load_frame_name_set(frame_list_path):
    if frame_list_path is None:
        return None
    names = set()
    with open(frame_list_path, "r") as fp:
        for line in fp:
            name = _normalize_frame_token(line)
            if name is not None:
                names.add(name)
    if not names:
        raise ValueError(f"No frame names found in {frame_list_path}")
    return names


def filter_camera_infos_by_frame_list(camera_infos, frame_list_path):
    names = load_frame_name_set(frame_list_path)
    if names is None:
        return camera_infos
    selected = [cam for cam in camera_infos if Path(cam.image_name).stem in names]
    if not selected:
        raise RuntimeError(f"No cameras matched frame list: {frame_list_path}")
    return selected


def transform_points(points, transform):
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_camera_info_to_pre(camera_info: CameraInfo, t_pre_post):
    """Transform a post-world camera so it observes pre-world coordinates.

    The alignment manifest is treated as a column-vector transform:
    X_pre = T_pre_post * X_post.
    """

    t_pre_post = np.asarray(t_pre_post, dtype=np.float64)
    r_pre_post = t_pre_post[:3, :3]
    t_pre_post_vec = t_pre_post[:3, 3]

    post_from_pre_r = np.linalg.inv(r_pre_post)
    post_from_pre_t = -post_from_pre_r @ t_pre_post_vec

    r_cw_post = np.asarray(camera_info.R, dtype=np.float64).T
    t_cw_post = np.asarray(camera_info.T, dtype=np.float64)
    r_cw_pre = r_cw_post @ post_from_pre_r
    t_cw_pre = r_cw_post @ post_from_pre_t + t_cw_post

    return camera_info._replace(R=r_cw_pre.T, T=t_cw_pre)


def transform_camera_infos_to_pre(camera_infos, t_pre_post):
    return [transform_camera_info_to_pre(cam, t_pre_post) for cam in camera_infos]


def select_camera_infos(camera_infos, frame_stride=1, max_frames=0):
    selected = camera_infos[:: max(1, frame_stride)]
    if max_frames and len(selected) > max_frames:
        idx = np.linspace(0, len(selected) - 1, max_frames).round().astype(np.int64)
        selected = [selected[i] for i in idx]
    return selected
