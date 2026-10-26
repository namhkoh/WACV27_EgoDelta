from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from action_points.colmap import CameraView, project_single_point, project_world_points
from action_points.label_generation import LabelBuildConfig, build_action_point_manifest
from action_points.yolo_export import export_yolo_detection_dataset


def test_project_world_points_marks_visibility():
    view = CameraView(
        image_id=1,
        image_name="frame_000001.jpg",
        image_path=Path("frame_000001.jpg"),
        width=100,
        height=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        r_cw=np.eye(3),
        t_cw=np.zeros(3),
    )
    points = np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.0, -1.0], [2.0, 0.0, 1.0]])
    uv, depth, valid, points_cam = project_world_points(points, view)
    assert np.allclose(uv[0], [50.0, 50.0])
    assert np.allclose(uv[1], [70.0, 50.0])
    assert depth[2] < 0.0
    assert points_cam.shape == (4, 3)
    assert valid.tolist() == [True, True, False, False]

    target_uv, target_depth, target_valid, target_cam = project_single_point(np.asarray([0.0, 0.0, 1.0]), view)
    assert np.allclose(target_uv, [50.0, 50.0])
    assert target_depth == 1.0
    assert target_valid
    assert np.allclose(target_cam, [0.0, 0.0, 1.0])


def test_build_manifest_from_component_targets(tmp_path):
    dataset_dir = make_colmap_dataset(tmp_path)
    xyz = np.asarray(
        [
            [-0.10, -0.05, 1.0],
            [-0.05, -0.03, 1.0],
            [0.00, 0.00, 1.0],
            [0.05, 0.03, 1.0],
            [0.10, 0.05, 1.0],
            [0.02, -0.02, 1.0],
            [-0.02, 0.02, 1.0],
            [0.04, 0.01, 1.0],
        ],
        dtype=np.float32,
    )
    npz_path = tmp_path / "leaf_component_targets.npz"
    np.savez_compressed(
        npz_path,
        xyz=xyz,
        leaf_instance_candidate=np.ones((xyz.shape[0],), dtype=bool),
        leaf_instance_id=np.zeros((xyz.shape[0],), dtype=np.int32),
        interaction_target_instance_id=np.asarray([0], dtype=np.int32),
        interaction_target_xyz=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        interaction_target_confidence=np.asarray([0.75], dtype=np.float32),
    )
    summary_path = tmp_path / "leaf_component_targets_summary.json"
    summary_path.write_text(
        json.dumps({"clusters": [{"cluster_id": 0, "target_xyz": [0.0, 0.0, 1.0], "target_confidence": 0.75}]}),
        encoding="utf-8",
    )

    manifest = build_action_point_manifest(
        dataset_dir=dataset_dir,
        presence_npz=npz_path,
        summary_json=summary_path,
        config=LabelBuildConfig(scene_id="scene_a", task_type="deleaf", min_visible_points=4),
    )

    assert manifest["num_samples"] == 1
    sample = manifest["samples"][0]
    assert sample["scene_id"] == "scene_a"
    assert sample["task_type"] == "deleaf"
    assert sample["bbox_xyxy"][0] < sample["target_2d"][0] < sample["bbox_xyxy"][2]
    assert sample["bbox_xyxy"][1] < sample["target_2d"][1] < sample["bbox_xyxy"][3]
    assert 0.0 <= sample["target_rel_xy"][0] <= 1.0
    assert 0.0 <= sample["target_rel_xy"][1] <= 1.0


def test_export_yolo_dataset(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100), (20, 20, 20)).save(image_path)
    manifest = {
        "samples": [
            {
                "scene_id": "scene_a",
                "split": "train",
                "task_type": "deleaf",
                "class_name": "target",
                "image_path": str(image_path),
                "image_name": image_path.name,
                "bbox_xywh_norm": [0.5, 0.5, 0.2, 0.3],
            }
        ]
    }
    summary = export_yolo_detection_dataset([manifest], tmp_path / "yolo", allow_single_scene_debug_split=True)
    assert Path(summary["data_yaml"]).exists()
    assert (tmp_path / "yolo" / "labels" / "train" / "scene_a_image.txt").read_text().strip() == (
        "0 0.50000000 0.50000000 0.20000000 0.30000000"
    )


def make_colmap_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    image_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), (30, 30, 30)).save(image_dir / "frame_000001.jpg")
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 100 100 100 100 50 50\n",
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "# Image list\n1 1 0 0 0 0 0 0 1 frame_000001.jpg\n\n",
        encoding="utf-8",
    )
    return dataset_dir
