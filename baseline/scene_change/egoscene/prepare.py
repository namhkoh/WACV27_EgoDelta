from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .benchmark import SceneSpec, selected_frame_names, source_image
from .colmap import ColmapImage, read_images_txt, write_images_txt


def prepare_mv3dcd_scene(scene: SceneSpec, output: str | Path, *, post_views: int = 10) -> Path:
    """Build the official MV3DCD COLMAP layout without copying RGB data."""
    output = Path(output).expanduser().resolve()
    images_dir = output / "images"
    sparse_out = output / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    pre_names = selected_frame_names(scene.pre_dataset)
    post_names = selected_frame_names(scene.post_dataset, post_views)
    pre_sparse = Path(scene.pre_dataset["dataset_dir"]) / "sparse" / "0"
    post_sparse = Path(scene.post_dataset["dataset_dir"]) / "sparse" / "0"
    pre_records = _records_by_name(pre_sparse / "images.txt")
    post_records = _records_by_name(post_sparse / "images.txt")
    combined: list[ColmapImage] = []

    for name in pre_names:
        record = _require_record(pre_records, name, scene.scene_id, "pre")
        linked_name = f"pre_{name}"
        _ensure_symlink(images_dir / linked_name, source_image(scene.pre_dataset, name))
        combined.append(_renamed(record, linked_name))
    for name in post_names:
        record = _require_record(post_records, name, scene.scene_id, "post")
        # MV3DCD's official loader assigns every filename containing "test" to
        # the post-change camera set used for mask-channel learning.
        linked_name = f"test_post_{name}"
        _ensure_symlink(images_dir / linked_name, source_image(scene.post_dataset, name))
        combined.append(_renamed(record, linked_name))

    shutil.copy2(pre_sparse / "cameras.txt", sparse_out / "cameras.txt")
    write_images_txt(sparse_out / "images.txt", combined)
    _ensure_symlink(sparse_out / "points3D.ply", pre_sparse / "points3D.ply")
    metadata = {
        "schema_version": 1,
        "baseline": "mv3dcd",
        "scene_id": scene.scene_id,
        "scene_uuid": scene.scene_uuid,
        "pre_view_count": len(pre_names),
        "post_input_view_count": len(post_names),
        "pre_frames": pre_names,
        "post_input_frames": post_names,
        "evaluation_frames": [task["image_name"] for task in scene.queries],
        "evaluation_views_are_not_in_post_change_training_set": True,
        "source_pair_manifest": str(scene.pair_manifest),
    }
    (output / "egodelta_split.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def prepare_3dgscd_scene(scene: SceneSpec, output: str | Path, *, post_views: int = 10) -> Path:
    """Build 3DGS-CD's Nerfstudio layout from known EgoDelta MPS poses."""
    output = Path(output).expanduser().resolve()
    for directory in (output / "rgb", output / "rgb_new", output / "rgb_eval"):
        directory.mkdir(parents=True, exist_ok=True)

    pre_names = selected_frame_names(scene.pre_dataset)
    post_names = selected_frame_names(scene.post_dataset, post_views)
    pre_cameras = _camera_index(scene.pre_model / "cameras.json")
    post_cameras = _camera_index(scene.post_model / "cameras.json")

    pre_frames = []
    for name in pre_names:
        camera = _require_camera(pre_cameras, name, scene.scene_id, "pre")
        _ensure_symlink(output / "rgb" / name, source_image(scene.pre_dataset, name))
        pre_frames.append(_nerfstudio_frame(f"rgb/{name}", camera))

    post_frames = []
    for name in post_names:
        camera = _require_camera(post_cameras, name, scene.scene_id, "post")
        _ensure_symlink(output / "rgb_new" / name, source_image(scene.post_dataset, name))
        post_frames.append(_nerfstudio_frame(f"rgb_new/{name}", camera))

    eval_frames = []
    for task in scene.queries:
        name = task["image_name"]
        _ensure_symlink(output / "rgb_eval" / name, Path(task["source_image"]))
        eval_frames.append(_nerfstudio_frame(f"rgb_eval/{name}", _camera_from_task(task)))

    if not pre_frames:
        raise ValueError(f"No pre frames for {scene.scene_id}")
    shared = _nerfstudio_intrinsics(pre_cameras[Path(pre_names[0]).stem])
    pre_paths = [frame["file_path"] for frame in pre_frames]
    post_paths = [frame["file_path"] for frame in post_frames]
    eval_paths = [frame["file_path"] for frame in eval_frames]
    point_source = Path(scene.pre_dataset["dataset_dir"]) / "sparse" / "0" / "points3D.ply"
    _ensure_symlink(output / "sparse_pc.ply", point_source)

    pretrain = {
        **shared,
        "ply_file_path": "sparse_pc.ply",
        "frames": pre_frames,
        "train_filenames": pre_paths,
        # Validation is diagnostic only; overlap is intentional so every pre
        # image remains available for fitting the reference reconstruction.
        "val_filenames": pre_paths[:: max(1, len(pre_paths) // 12)][:12],
        "test_filenames": pre_paths[:: max(1, len(pre_paths) // 12)][:12],
    }
    combined = {
        **shared,
        "ply_file_path": "sparse_pc.ply",
        "frames": pre_frames + post_frames + eval_frames,
        "train_filenames": pre_paths + post_paths,
        "val_filenames": eval_paths,
        "test_filenames": eval_paths,
    }
    (output / "transforms_pretrain.json").write_text(json.dumps(pretrain, indent=2) + "\n", encoding="utf-8")
    (output / "transforms.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    configs = {
        "sam_threshold": 0.95,
        "mask_refine_sparse_view": 0.0,
        "area_threshold": 0.01,
        "pcd_filtering": 0.98,
        "pre_train_pred_bbox_expand": 0.05,
        "voxel_dim": 300,
        "bbox3d_expand": 1.8,
        "mask3d_dilate_uniform": 1,
        "mask3d_dilate_top": 0,
        "pose_change_break": None,
        "pose_refine_lr": 1.0e-3,
        "pose_refine_epochs": 50,
        "pose_refine_patience": 20,
        "vis_check_threshold": 0.8,
        "proj_check_cutoff": 0.95,
        "val_move_in_dilate_3d": 0.05,
        "val_move_out_dilate_3d": 0.05,
    }
    (output / "configs.json").write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    split = {
        "schema_version": 1,
        "baseline": "3dgscd",
        "scene_id": scene.scene_id,
        "scene_uuid": scene.scene_uuid,
        "pre_view_count": len(pre_names),
        "post_input_view_count": len(post_names),
        "pre_frames": pre_names,
        "post_input_frames": post_names,
        "evaluation_frames": [task["image_name"] for task in scene.queries],
        "uses_known_mps_poses": True,
        "evaluation_views_are_not_post_change_inputs": True,
        "source_pair_manifest": str(scene.pair_manifest),
    }
    (output / "egodelta_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return output


def _records_by_name(path: Path) -> dict[str, ColmapImage]:
    return {row.name: row for row in read_images_txt(path)}


def _require_record(records: dict[str, ColmapImage], name: str, scene_id: str, label: str) -> ColmapImage:
    try:
        return records[name]
    except KeyError as exc:
        raise KeyError(f"{name} is absent from {label} COLMAP images.txt for {scene_id}") from exc


def _renamed(record: ColmapImage, name: str) -> ColmapImage:
    return ColmapImage(**{**record.__dict__, "name": name})


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(f"Existing symlink points elsewhere: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)


def _camera_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["img_name"]): row for row in rows}


def _require_camera(cameras: dict[str, dict[str, Any]], image_name: str, scene_id: str, label: str) -> dict[str, Any]:
    stem = Path(image_name).stem
    try:
        return cameras[stem]
    except KeyError as exc:
        raise KeyError(f"{image_name} is absent from {label} cameras.json for {scene_id}") from exc


def _camera_from_task(task: dict[str, Any]) -> dict[str, Any]:
    camera = task["camera"]
    return {
        "position": camera["camera_center_world"],
        "rotation": camera["rotation_camera_to_world"],
        "width": camera["width"],
        "height": camera["height"],
        "fx": camera["fx"],
        "fy": camera["fy"],
        "cx": camera["cx"],
        "cy": camera["cy"],
    }


def _nerfstudio_frame(path: str, camera: dict[str, Any]) -> dict[str, Any]:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(camera["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(camera["position"], dtype=np.float64)
    # EgoDelta cameras use OpenCV axes; Nerfstudio transforms use OpenGL axes.
    c2w[:3, 1:3] *= -1.0
    return {"file_path": path, "transform_matrix": c2w.tolist()}


def _nerfstudio_intrinsics(camera: dict[str, Any]) -> dict[str, Any]:
    width = int(camera["width"])
    height = int(camera["height"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    return {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": float(camera.get("cx", width / 2.0)),
        "cy": float(camera.get("cy", height / 2.0)),
        "w": width,
        "h": height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_angle_x": 2.0 * math.atan(width / (2.0 * fx)),
        "camera_angle_y": 2.0 * math.atan(height / (2.0 * fy)),
    }
