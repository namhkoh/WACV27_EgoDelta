from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "quality": {
        "max_inv_dist_std": 0.005,
        "max_dist_std": 0.01,
        "min_observations": 2,
        "max_points": 2_000_000,
        "statistical_outlier": {
            "enabled": False,
            "nb_neighbors": 20,
            "std_ratio": 2.0,
        },
    },
    "roi": {
        "crop_boxes": [],
        "exclude_boxes": [],
    },
    "registration": {
        "assume_same_world_frame": True,
        "refine_icp": False,
        "voxel_size": 0.01,
        "icp_max_correspondence_distance": 0.03,
        "icp_max_iterations": 80,
        "min_icp_fitness": 0.3,
        "max_icp_rmse": 0.03,
    },
    "nn": {
        "threshold_m": 0.025,
        "auto_threshold": True,
        "auto_threshold_k_mad": 4.0,
        "stable_percentile": 95,
    },
    "m3c2": {
        "enabled": True,
        "corepoint_stride": 1,
        "max_corepoints": 300_000,
        "cyl_radius": 0.03,
        "normal_radii": [0.02, 0.04, 0.08],
        "max_distance": 0.10,
        "threshold_m": 0.025,
        "use_lodetection": True,
        "min_samples1": 3,
        "min_samples2": 1,
    },
    "scoring": {
        "nn_weight": 0.55,
        "m3c2_weight": 0.45,
        "density_weight": 0.0,
        "fusion": "weighted_average",
        "min_combined_score": 0.5,
    },
    "density": {
        "enabled": False,
        "radius_m": 0.04,
        "min_pre_neighbors": 3,
        "min_drop_fraction": 0.35,
        "min_drop_count": 2,
        "max_post_neighbors": 2,
        "min_expected_post_ratio": 0.25,
        "candidate_threshold": 0.25,
    },
    "dbscan": {
        "eps_m": 0.03,
        "min_samples": 8,
        "min_cluster_points": 20,
        "max_cluster_points": 200_000,
    },
    "task_priors": {
        "harvest": {
            "prefer_compact": True,
            "min_diameter_m": 0.02,
            "max_diameter_m": 0.15,
        },
        "deleaf": {
            "prefer_elongated": True,
            "min_length_m": 0.04,
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return ``base``."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    deep_update(cfg, load_yaml(path))
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def config_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
