"""Configuration dataclasses for the harvesting PoC pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    conf_thres: float = 0.25
    imgsz: int = 640
    retina_masks: bool = True
    max_det: int = 300
    segmentation_batch_size: int = 1
    visualization_stride: int = 30

    frame_stride: int = 10
    rgb_stream_id: str = "214-1"
    image_view: str = "raw"
    aria_camera_label: str = "camera-rgb"
    rect_size: int = 720
    rect_focal: float = 380.0

    inv_dist_std_thres: float = 0.005
    dist_std_thres: float = 0.01

    min_points_inside_mask: int = 10
    tomato_depth_band_m: float = 0.08
    tomato_dbscan_eps_m: float = 0.05
    tomato_dbscan_min_samples: int = 5

    instance_cluster_eps_m: float = 0.10
    instance_min_frames: int = 2
    instance_min_points: int = 20

    prepost_match_radius_m: float = 0.12
    min_visible_post_views: int = 3
    point_match_radius_m: float = 0.04
    point_disappearance_weight: float = 0.5
    disappeared_score_thres: float = 0.6

    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    top_quantile: float = 0.75
    cut_offset_m: float = 0.02

    cutpoint_mode: str = "proxy"
    peduncle_conf_thres: float = 0.25
    peduncle_imgsz: int = 640
    peduncle_keypoint_names: tuple[str, ...] = (
        "fruit_attachment",
        "peduncle_mid",
        "stem_attachment",
        "cut_point",
    )
    peduncle_keypoint_lift_radius_px: float = 8.0
    peduncle_depth_band_m: float = 0.05
    peduncle_min_points: int = 5
    peduncle_dbscan_eps_m: float = 0.03
    peduncle_dbscan_min_samples: int = 3
    peduncle_association_max_distance_m: float = 0.15
    peduncle_cut_offset_m: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_world_up(value: str | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("world_up tuple must have exactly 3 values")
        return tuple(float(v) for v in value)
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--world-up must be formatted as x,y,z")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_csv_names(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return tuple(part.strip() for part in value.split(",") if part.strip())


def should_write_visualization(index: int, visualization_stride: int) -> bool:
    """Return whether a sampled-frame visualization should be written."""

    return visualization_stride > 0 and index % visualization_stride == 0


def ensure_output_dirs(out_dir: str | Path) -> None:
    out = Path(out_dir)
    dirs = [
        out,
        out / "frames" / "pre",
        out / "frames" / "post",
        out / "yolo_predictions" / "masks",
        out / "lifted_3d" / "points",
        out / "visualizations" / "yolo_pre",
        out / "visualizations" / "yolo_post",
        out / "visualizations" / "lifting_debug" / "pre",
        out / "visualizations" / "lifting_debug" / "post",
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def relpath(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
