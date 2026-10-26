from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

from .benchmark import SceneSpec
from .outputs import write_prediction, write_scene_metadata


def collect_3dgscd_predictions(
    scene: SceneSpec,
    official_scene_root: str | Path,
    output_root: str | Path,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Normalize official move-out masks into EgoDelta's prediction schema."""
    official_scene_root = Path(official_scene_root)
    mask_root = official_scene_root / "masks_new"
    predictions = []
    for task in scene.queries:
        match = re.search(r"(\d+)(?!.*\d)", Path(task["image_name"]).stem)
        if match is None:
            raise ValueError(f"Image has no numeric frame ID: {task['image_name']}")
        frame_id = int(match.group(1))
        candidates = (
            mask_root / f"mask_{frame_id:05g}.png",
            mask_root / f"mask_{frame_id:05d}.png",
            mask_root / f"mask_{frame_id}.png",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"No 3DGS-CD move-out mask for {task['image_name']}: {candidates}")
        probability = np.asarray(Image.open(source).convert("L"), dtype=np.float32) / 255.0
        expected = (int(task["camera"]["height"]), int(task["camera"]["width"]))
        if probability.shape != expected:
            raise ValueError(f"3DGS-CD mask shape is {probability.shape}, expected {expected}: {source}")
        paths = write_prediction(output_root, scene.scene_id, task["image_name"], probability, threshold=threshold)
        predictions.append({"task_id": task["task_id"], "image_name": task["image_name"], "official_mask": str(source), **paths})
    metadata = {
        "baseline": "3dgscd_official",
        "official_output": str(official_scene_root.resolve()),
        "prediction_semantics": "move-out / previous-location mask_*.png",
        "uses_known_aria_mps_evaluation_poses": True,
        "evaluation_pose_refinement": False,
        "threshold": threshold,
        "predictions": predictions,
    }
    write_scene_metadata(output_root, scene.scene_id, metadata)
    return metadata


def render_mv3dcd_queries(
    scene: SceneSpec,
    model_root: str | Path,
    output_root: str | Path,
    upstream_root: str | Path,
    *,
    iteration: int = 13_000,
    alpha_iteration: int = 7_000,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Render a trained official MV3DCD change channel at EgoDelta GT poses."""
    upstream_root = Path(upstream_root).expanduser().resolve()
    model_root = Path(model_root).expanduser().resolve()
    sys.path.insert(0, str(upstream_root))
    try:
        from gaussian_renderer import gsplat_render, gsplat_render_mask
        from scene.cameras import Camera as OfficialCamera
        from scene.gaussian_model import GaussianModel
    except ImportError as exc:
        raise RuntimeError(f"Could not import official MV3DCD from {upstream_root}") from exc

    point_cloud = model_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    alpha_point_cloud = model_root / "point_cloud" / f"iteration_{alpha_iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(point_cloud)
    if not alpha_point_cloud.is_file():
        raise FileNotFoundError(alpha_point_cloud)
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(point_cloud))
    # The official final renderer gates the 13k change channel with the alpha
    # masks rendered from the frozen 7k reference reconstruction. The 10k
    # render is run with augmentation and does not overwrite those masks.
    alpha_gaussians = GaussianModel(3)
    alpha_gaussians.load_ply(str(alpha_point_cloud))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    class Pipeline:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    predictions = []
    for index, task in enumerate(scene.queries):
        row = task["camera"]
        width, height = int(row["width"]), int(row["height"])
        rotation_c2w = np.asarray(row["rotation_camera_to_world"], dtype=np.float32)
        rotation_w2c = rotation_c2w.T
        center = np.asarray(row["camera_center_world"], dtype=np.float32)
        translation_w2c = -(rotation_w2c @ center)
        blank = torch.zeros((3, height, width), dtype=torch.float32)
        camera = OfficialCamera(
            colmap_id=index,
            R=rotation_c2w,
            T=translation_w2c,
            FoVx=2.0 * math.atan(width / (2.0 * float(row["fx"]))),
            FoVy=2.0 * math.atan(height / (2.0 * float(row["fy"]))),
            image=blank,
            gt_alpha_mask=None,
            image_name=Path(task["image_name"]).stem,
            uid=index,
            change_mask=None,
            alpha_mask=None,
        )
        with torch.no_grad():
            rendered = gsplat_render_mask(camera, gaussians, Pipeline(), background)
            reference = gsplat_render(camera, alpha_gaussians, Pipeline(), background)
        probability = rendered["change_mask"][0].clamp(0, 1)
        # Match the official final renderer: suppress pixels unsupported by the
        # frozen reference reconstruction's normalized alpha map.
        alpha = reference["alpha"][..., 0]
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min()).clamp_min(1.0e-8)
        probability = probability * (alpha > 0.5)
        paths = write_prediction(
            output_root,
            scene.scene_id,
            task["image_name"],
            probability.detach().cpu().numpy(),
            threshold=threshold,
        )
        predictions.append({"task_id": task["task_id"], "image_name": task["image_name"], **paths})
    split_path = model_root / "egodelta_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else None
    metadata = {
        "baseline": "mv3dcd_official",
        "upstream_model": str(model_root),
        "input_split_manifest": str(split_path) if split is not None else None,
        "pre_view_count": split.get("pre_view_count") if split is not None else None,
        "post_input_view_count": split.get("post_input_view_count") if split is not None else None,
        "iteration": iteration,
        "alpha_iteration": alpha_iteration,
        "threshold": threshold,
        "prediction_semantics": "official learned change channel rendered at fixed EgoDelta benchmark poses",
        "predictions": predictions,
    }
    write_scene_metadata(output_root, scene.scene_id, metadata)
    return metadata
