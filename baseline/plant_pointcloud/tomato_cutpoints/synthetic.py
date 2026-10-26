from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tomato_cutpoints.config import load_config, save_config
from tomato_cutpoints.io.aria_mps import write_minimal_observations_csv_gz, write_minimal_semidense_csv_gz
from tomato_cutpoints.registration.transforms import identity_transform


def _small_transform() -> np.ndarray:
    angle = np.deg2rad(1.5)
    c, s = np.cos(angle), np.sin(angle)
    t = identity_transform()
    t[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t[:3, 3] = np.array([0.006, -0.004, 0.002])
    return t


def _background(rng: np.random.Generator) -> np.ndarray:
    plane = np.column_stack(
        [
            rng.uniform(-0.18, 0.58, 900),
            rng.normal(0.20, 0.002, 900),
            rng.uniform(0.25, 0.90, 900),
        ]
    )
    vines = []
    for x in [-0.03, 0.00, 0.31]:
        z = np.linspace(0.25, 0.90, 180)
        vines.append(
            np.column_stack(
                [
                    np.full_like(z, x) + rng.normal(0.0, 0.002, len(z)),
                    rng.normal(0.0, 0.002, len(z)),
                    z + rng.normal(0.0, 0.001, len(z)),
                ]
            )
        )
    return np.vstack([plane, *vines])


def _sphere(rng: np.random.Generator, center: np.ndarray, radius: float, n: int) -> np.ndarray:
    vec = rng.normal(size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return center.reshape(1, 3) + radius * vec + rng.normal(0.0, 0.002, (n, 3))


def _leaf(rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.0, 1.0, 420)
    width = 0.018 * np.sin(np.pi * t)
    lateral = rng.normal(0.0, width + 0.002)
    x = 0.02 + 0.22 * t + rng.normal(0.0, 0.003, len(t))
    y = lateral
    z = 0.55 + 0.03 * np.sin(np.pi * t) + rng.normal(0.0, 0.003, len(t))
    return np.column_stack([x, y, z])


def generate_synthetic_scene(task: str, seed: int = 13, perturb_post: bool = True) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    background = _background(rng)
    task = task.lower()
    if task == "harvest":
        center = np.array([0.36, 0.0, 0.55])
        removed = _sphere(rng, center, 0.045, 520)
        gt = {
            "task": "harvest",
            "targets": [
                {
                    "object_center_world": center.tolist(),
                    "cut_point_world": [0.315, 0.0, 0.55],
                    "cluster_bbox": {"min": removed.min(axis=0).tolist(), "max": removed.max(axis=0).tolist()},
                }
            ],
        }
    elif task == "deleaf":
        removed = _leaf(rng)
        gt = {
            "task": "deleaf",
            "targets": [
                {
                    "object_center_world": removed.mean(axis=0).tolist(),
                    "cut_point_world": removed[np.argmin(removed[:, 0])].tolist(),
                    "cluster_bbox": {"min": removed.min(axis=0).tolist(), "max": removed.max(axis=0).tolist()},
                }
            ],
        }
    else:
        raise ValueError(f"Unsupported synthetic task: {task}")

    pre_xyz = np.vstack([background, removed])
    post_world = background.copy()
    post_world = post_world[rng.random(len(post_world)) > 0.03]
    pre_xyz = pre_xyz + rng.normal(0.0, 0.001, pre_xyz.shape)
    post_world = post_world + rng.normal(0.0, 0.001, post_world.shape)

    t_post_to_pre = _small_transform() if perturb_post else identity_transform()
    post_raw = post_world
    if perturb_post:
        post_raw_h = np.column_stack([post_world, np.ones(len(post_world))])
        post_raw = (post_raw_h @ np.linalg.inv(t_post_to_pre).T)[:, :3]

    roi = {
        "crop_boxes": [{"name": "plant_area", "min": [-0.25, -0.12, 0.20], "max": [0.65, 0.28, 0.95]}],
        "exclude_boxes": [],
        "stable_boxes": [{"name": "trellis_plane", "min": [-0.22, 0.16, 0.22], "max": [0.62, 0.25, 0.95]}],
    }
    return {
        "pre_xyz": pre_xyz,
        "post_xyz": post_raw,
        "T_post_to_pre_gt": t_post_to_pre,
        "gt": gt,
        "roi": roi,
    }


def write_synthetic_trial(base_dir: str | Path, task: str, seed: int = 13, perturb_post: bool = True) -> dict[str, Path]:
    base_dir = Path(base_dir)
    scene = generate_synthetic_scene(task, seed=seed, perturb_post=perturb_post)
    input_dir = base_dir / "synthetic_inputs"
    pre_dir = input_dir / "pre" / "slam"
    post_dir = input_dir / "post" / "slam"
    pre_points = pre_dir / "semidense_points.csv.gz"
    post_points = post_dir / "semidense_points.csv.gz"
    pre_obs = pre_dir / "semidense_observations.csv.gz"
    post_obs = post_dir / "semidense_observations.csv.gz"
    roi_path = input_dir / "stable_trellis_boxes.json"
    gt_path = input_dir / "ground_truth.json"
    config_path = input_dir / "synthetic_config.yaml"
    transform_gt_path = input_dir / "T_post_to_pre_gt.txt"

    write_minimal_semidense_csv_gz(pre_points, scene["pre_xyz"], uid_start=0)
    write_minimal_semidense_csv_gz(post_points, scene["post_xyz"], uid_start=10_000_000)
    write_minimal_observations_csv_gz(pre_obs, len(scene["pre_xyz"]), uid_start=0, repeats=3)
    write_minimal_observations_csv_gz(post_obs, len(scene["post_xyz"]), uid_start=10_000_000, repeats=3)
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text(json.dumps(scene["roi"], indent=2) + "\n", encoding="utf-8")
    gt_path.write_text(json.dumps(scene["gt"], indent=2) + "\n", encoding="utf-8")
    np.savetxt(transform_gt_path, scene["T_post_to_pre_gt"], fmt="%.10f")

    cfg = load_config(None)
    cfg["m3c2"]["enabled"] = False
    cfg["registration"]["refine_icp"] = True
    cfg["scoring"]["min_combined_score"] = 0.15
    cfg["dbscan"]["min_cluster_points"] = 12
    cfg["dbscan"]["eps_m"] = 0.035
    save_config(cfg, config_path)

    return {
        "pre_points": pre_points,
        "post_points": post_points,
        "pre_observations": pre_obs,
        "post_observations": post_obs,
        "roi": roi_path,
        "gt": gt_path,
        "config": config_path,
        "transform_gt": transform_gt_path,
    }
