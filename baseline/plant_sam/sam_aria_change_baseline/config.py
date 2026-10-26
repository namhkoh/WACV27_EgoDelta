from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


VALID_TASKS = ("harvesting", "deleafing")
VALID_BACKENDS = ("native_sam3", "ultralytics_sam3")
VALID_CHANGE_SCORE_MODES = (
    "auto",
    "average",
    "sam_absence_only",
    "point_disappearance_only",
)

EGODELTA_RECTIFIED_RGB_SIZE = 720
EGODELTA_RECTIFIED_RGB_FOCAL = 380.0


@dataclass
class PipelineConfig:
    task: str = "harvesting"
    sam_backend: str = "native_sam3"
    use_video_mode: bool = False

    rgb_stream_id: str = "214-1"
    frame_stride: int = 1
    sam_batch_size: int = 1
    image_size: int | None = None
    use_undistorted_rgb: bool = False
    undistort_rect_size: int = EGODELTA_RECTIFIED_RGB_SIZE
    undistort_rect_focal: float = EGODELTA_RECTIFIED_RGB_FOCAL

    sam_score_thres: float = 0.25
    deleafing_sam_score_thres: float = 0.16
    aux_sam_score_thres: float = 0.10
    deleafing_aux_sam_score_thres: float = 0.35
    min_mask_area_px: int = 100
    deleafing_min_mask_area_px: int = 50
    deleafing_max_mask_area_frac: float = 0.03
    aux_min_mask_area_px: int = 20
    deleafing_aux_min_mask_area_px: int = 80
    max_mask_area_frac: float = 0.40

    mask_nms_iou_thres: float = 0.75
    mask_dedup_iou_thres: float = 0.85

    inv_dist_std_thres: float = 0.005
    dist_std_thres: float = 0.01

    min_points_inside_mask: int = 10
    petiole_min_points_inside_mask: int = 4
    depth_quantile: float = 0.15

    tomato_depth_band_m: float = 0.08
    petiole_depth_band_m: float = 0.04
    petiole_min_3d_major_extent_m: float = 0.04
    petiole_max_3d_major_extent_m: float = 0.55

    tomato_dbscan_eps_m: float = 0.05
    petiole_dbscan_eps_m: float = 0.04
    dbscan_min_samples: int = 5
    petiole_dbscan_min_samples: int = 3

    tomato_instance_eps_m: float = 0.10
    petiole_instance_eps_m: float = 0.08

    instance_min_frames: int = 2
    petiole_instance_min_frames: int = 1
    auxiliary_instance_min_frames: int = 1
    tomato_instance_min_points: int = 20
    petiole_instance_min_points: int = 10
    auxiliary_instance_min_points: int = 5

    tomato_prepost_match_radius_m: float = 0.12
    petiole_prepost_match_radius_m: float = 0.02

    min_visible_post_views: int = 3
    point_match_radius_m: float = 0.04
    disappeared_score_thres: float = 0.60
    deleafing_disappeared_score_thres: float | None = 0.0
    change_score_mode: str = "auto"

    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    tomato_top_quantile: float = 0.75
    harvest_cut_offset_m: float = 0.02

    petiole_stem_side_quantile: float = 0.10
    deleafing_main_stem_min_verticality: float = 0.82
    deleafing_main_stem_min_height_frac: float = 0.18
    deleafing_main_stem_max_width_frac: float = 0.22
    deleafing_main_stem_min_aspect_ratio: float = 1.8
    deleafing_aux_min_vertical_extent_m: float = 0.08
    deleafing_aux_min_vertical_ratio: float = 1.25
    deleafing_primary_leaf_blob_min_area_px: int = 800
    deleafing_primary_leaf_blob_thinness: float = 0.70
    deleafing_primary_leaf_blob_extent: float = 0.45
    deleafing_primary_final_component_filter: bool = True
    deleafing_primary_reject_vertical_axis: bool = True
    deleafing_primary_vertical_axis_thres: float = 0.90
    deleafing_primary_vertical_axis_max_thinness: float = 0.45
    deleafing_primary_vertical_axis_image_direction: str = "image_horizontal"
    deleafing_max_primary_masks_per_frame: int = 12
    deleafing_max_aux_masks_per_frame: int = 8
    deleafing_long_horizontal_width_frac: float = 0.35
    deleafing_long_horizontal_aspect_ratio: float = 5.0
    deleafing_pipe_like_long_side_frac: float = 0.35
    deleafing_pipe_like_aspect_ratio: float = 6.0

    # CLI-wide overrides. If None, task-specific defaults above are used.
    depth_band_m: float | None = None
    instance_eps_m: float | None = None
    match_radius_m: float | None = None

    def validate(self) -> None:
        if self.task not in VALID_TASKS:
            raise ValueError(f"Unsupported task {self.task!r}; expected one of {VALID_TASKS}.")
        if self.sam_backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unsupported SAM backend {self.sam_backend!r}; expected one of {VALID_BACKENDS}."
            )
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")
        if self.sam_batch_size < 1:
            raise ValueError("sam_batch_size must be >= 1.")
        if self.undistort_rect_size <= 0:
            raise ValueError("undistort_rect_size must be positive.")
        if self.undistort_rect_focal <= 0:
            raise ValueError("undistort_rect_focal must be positive.")
        if self.change_score_mode not in VALID_CHANGE_SCORE_MODES:
            raise ValueError(
                f"Unsupported change_score_mode {self.change_score_mode!r}; "
                f"expected one of {VALID_CHANGE_SCORE_MODES}."
            )
        if self.deleafing_primary_vertical_axis_image_direction not in (
            "image_vertical",
            "image_horizontal",
        ):
            raise ValueError(
                "deleafing_primary_vertical_axis_image_direction must be "
                "'image_vertical' or 'image_horizontal'."
            )
        if not (0.0 < self.depth_quantile < 1.0):
            raise ValueError("depth_quantile must be in (0, 1).")

    def depth_band_for_task(self, task: str | None = None) -> float:
        if self.depth_band_m is not None:
            return self.depth_band_m
        task = task or self.task
        if task == "harvesting":
            return self.tomato_depth_band_m
        if task == "deleafing":
            return self.petiole_depth_band_m
        raise ValueError(f"Unsupported task {task!r}.")

    def dbscan_eps_for_task(self, task: str | None = None) -> float:
        task = task or self.task
        if task == "harvesting":
            return self.tomato_dbscan_eps_m
        if task == "deleafing":
            return self.petiole_dbscan_eps_m
        raise ValueError(f"Unsupported task {task!r}.")

    def instance_eps_for_task(self, task: str | None = None) -> float:
        if self.instance_eps_m is not None:
            return self.instance_eps_m
        task = task or self.task
        if task == "harvesting":
            return self.tomato_instance_eps_m
        if task == "deleafing":
            return self.petiole_instance_eps_m
        raise ValueError(f"Unsupported task {task!r}.")

    def instance_min_points_for_task(self, task: str | None = None) -> int:
        task = task or self.task
        if task == "harvesting":
            return self.tomato_instance_min_points
        if task == "deleafing":
            return self.petiole_instance_min_points
        raise ValueError(f"Unsupported task {task!r}.")

    def score_threshold_for_group(self, semantic_group: str, task: str | None = None) -> float:
        task = task or self.task
        if semantic_group == "auxiliary":
            if task == "deleafing":
                return self.deleafing_aux_sam_score_thres
            return self.aux_sam_score_thres
        if task == "deleafing":
            return self.deleafing_sam_score_thres
        return self.sam_score_thres

    def min_mask_area_for_group(self, semantic_group: str, task: str | None = None) -> int:
        task = task or self.task
        if semantic_group == "auxiliary":
            if task == "deleafing":
                return self.deleafing_aux_min_mask_area_px
            return self.aux_min_mask_area_px
        if task == "deleafing":
            return self.deleafing_min_mask_area_px
        return self.min_mask_area_px

    def min_points_inside_mask_for_task(self, task: str | None = None) -> int:
        task = task or self.task
        if task == "deleafing":
            return self.petiole_min_points_inside_mask
        return self.min_points_inside_mask

    def dbscan_min_samples_for_task(self, task: str | None = None) -> int:
        task = task or self.task
        if task == "deleafing":
            return self.petiole_dbscan_min_samples
        return self.dbscan_min_samples

    def instance_min_frames_for_group(self, semantic_group: str, task: str | None = None) -> int:
        if semantic_group == "auxiliary":
            return self.auxiliary_instance_min_frames
        if (task or self.task) == "deleafing":
            return self.petiole_instance_min_frames
        return self.instance_min_frames

    def instance_min_points_for_group(self, semantic_group: str, task: str | None = None) -> int:
        if semantic_group == "auxiliary":
            return self.auxiliary_instance_min_points
        return self.instance_min_points_for_task(task)

    def prepost_match_radius_for_task(self, task: str | None = None) -> float:
        if self.match_radius_m is not None:
            return self.match_radius_m
        task = task or self.task
        if task == "harvesting":
            return self.tomato_prepost_match_radius_m
        if task == "deleafing":
            return self.petiole_prepost_match_radius_m
        raise ValueError(f"Unsupported task {task!r}.")

    def disappeared_score_threshold_for_task(self, task: str | None = None) -> float:
        task = task or self.task
        if task == "deleafing" and self.deleafing_disappeared_score_thres is not None:
            return self.deleafing_disappeared_score_thres
        return self.disappeared_score_thres


def parse_world_up(value: str | Iterable[float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = [float(x.strip()) for x in value.split(",")]
    else:
        parts = [float(x) for x in value]
    if len(parts) != 3:
        raise ValueError("--world-up must contain three comma-separated numbers, e.g. 0,0,1.")
    return (parts[0], parts[1], parts[2])
