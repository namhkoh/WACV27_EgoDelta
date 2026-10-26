"""Task-level configuration for Aria pre/post change detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class TaskConfig:
    task_name: str
    target_semantic_class: str

    segmentation_backend: str
    text_prompts: list[str]
    negative_prompts: list[str]

    min_2d_confidence: float
    min_mask_area_px: int
    max_mask_area_px: int | None

    depth_band_m: float
    lifting_dbscan_eps_m: float
    lifting_dbscan_min_samples: int

    instance_cluster_eps_m: float
    instance_min_frames: int
    instance_min_points: int

    prepost_match_radius_m: float
    point_match_radius_m: float
    min_visible_post_views: int
    disappeared_score_thres: float

    action_point_type: str

    world_up: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["world_up"] = list(self.world_up)
        return data


DEFAULT_TASK_CONFIGS = {
    "harvest": "configs/harvest.yaml",
    "deleaf": "configs/deleaf.yaml",
}


def load_task_config(path: str | Path) -> TaskConfig:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {path}")
    if yaml is not None:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    else:
        data = _load_simple_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Task config must be a mapping: {path}")
    return task_config_from_dict(data)


def task_config_from_dict(data: dict[str, Any]) -> TaskConfig:
    allowed = {field.name for field in fields(TaskConfig)}
    missing = sorted(allowed.difference(data))
    if missing:
        raise ValueError(f"Task config is missing required fields: {', '.join(missing)}")
    extra = sorted(set(data).difference(allowed))
    if extra:
        raise ValueError(f"Task config contains unknown fields: {', '.join(extra)}")

    world_up_raw = data["world_up"]
    if len(world_up_raw) != 3:
        raise ValueError("TaskConfig.world_up must contain exactly 3 values")
    world_up = tuple(float(v) for v in world_up_raw)
    return TaskConfig(
        task_name=str(data["task_name"]),
        target_semantic_class=str(data["target_semantic_class"]),
        segmentation_backend=str(data["segmentation_backend"]),
        text_prompts=[str(v) for v in data["text_prompts"]],
        negative_prompts=[str(v) for v in data["negative_prompts"]],
        min_2d_confidence=float(data["min_2d_confidence"]),
        min_mask_area_px=int(data["min_mask_area_px"]),
        max_mask_area_px=(
            None if data["max_mask_area_px"] is None else int(data["max_mask_area_px"])
        ),
        depth_band_m=float(data["depth_band_m"]),
        lifting_dbscan_eps_m=float(data["lifting_dbscan_eps_m"]),
        lifting_dbscan_min_samples=int(data["lifting_dbscan_min_samples"]),
        instance_cluster_eps_m=float(data["instance_cluster_eps_m"]),
        instance_min_frames=int(data["instance_min_frames"]),
        instance_min_points=int(data["instance_min_points"]),
        prepost_match_radius_m=float(data["prepost_match_radius_m"]),
        point_match_radius_m=float(data["point_match_radius_m"]),
        min_visible_post_views=int(data["min_visible_post_views"]),
        disappeared_score_thres=float(data["disappeared_score_thres"]),
        action_point_type=str(data["action_point_type"]),
        world_up=world_up,  # type: ignore[arg-type]
    )


def default_task_config_path(task_name: str) -> Path:
    try:
        return Path(DEFAULT_TASK_CONFIGS[task_name])
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc


def apply_task_overrides(
    task_config: TaskConfig,
    *,
    segmentation_backend: str | None = None,
    world_up: tuple[float, float, float] | None = None,
) -> TaskConfig:
    data = task_config.to_dict()
    if segmentation_backend:
        data["segmentation_backend"] = segmentation_backend
    if world_up is not None:
        data["world_up"] = list(world_up)
    return task_config_from_dict(data)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Tiny YAML subset parser for this repo's flat config files.

    PyYAML is listed in requirements, but this fallback keeps CLI help and
    lightweight tests usable in minimal environments.
    """

    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError(f"List item without key in {path}: {raw_line}")
            data.setdefault(current_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
        key, value = stripped.split(":", 1)
        current_key = key.strip()
        value = value.strip()
        if value == "":
            data[current_key] = []
        else:
            data[current_key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inside = value[1:-1].strip()
        if not inside:
            return []
        return [_parse_scalar(part.strip()) for part in inside.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
