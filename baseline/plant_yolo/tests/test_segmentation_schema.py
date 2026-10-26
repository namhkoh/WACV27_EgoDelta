import numpy as np

from yolo_aria_harvest_poc.segmentation.base import (
    SegmentationDetection,
    mask_iou,
    mask_to_bbox_xyxy,
    refine_mask_for_task,
    rejection_reason,
)
from yolo_aria_harvest_poc.task_config import load_task_config


def test_segmentation_detection_schema_round_trip():
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="leaf stem",
        confidence=0.5,
        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=9,
        backend="sam3_concept",
        prompt_used="leaf stem",
    )

    payload = detection.to_dict()

    assert payload["semantic_class"] == "petiole"
    assert payload["mask"] == detection.mask_path
    assert payload["backend"] == "sam3_concept"


def test_mask_bbox_and_iou():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 2:5] = True

    assert mask_to_bbox_xyxy(mask) == [2.0, 1.0, 5.0, 4.0]
    assert mask_iou(mask, mask) == 1.0


def test_deleaf_leaf_blade_prompts_are_rejected_as_negative():
    task_config = load_task_config("configs/deleaf.yaml")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:65, 25:65] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="tomato leaf",
        confidence=0.8,
        bbox_xyxy=[25.0, 25.0, 65.0, 65.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="tomato leaf",
    )

    assert rejection_reason(detection, mask, 100, 100, task_config) == "negative_prompt"


def test_deleaf_petiole_prompts_still_require_elongated_masks():
    task_config = load_task_config("configs/deleaf.yaml")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:65, 25:65] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="leaf stem",
        confidence=0.8,
        bbox_xyxy=[25.0, 25.0, 65.0, 65.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="leaf stem",
    )

    assert rejection_reason(detection, mask, 100, 100, task_config) == "not_elongated"


def test_deleaf_elongated_petiole_prompt_is_kept():
    task_config = load_task_config("configs/deleaf.yaml")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[44:56, 10:90] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="tomato leaf petiole",
        confidence=0.8,
        bbox_xyxy=[10.0, 44.0, 90.0, 56.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="tomato leaf petiole",
    )

    assert rejection_reason(detection, mask, 100, 100, task_config) is None


def test_negative_prompt_detection_is_rejected():
    task_config = load_task_config("configs/deleaf.yaml")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="tomato stem",
        confidence=0.95,
        bbox_xyxy=[20.0, 20.0, 80.0, 80.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="tomato stem",
    )

    assert rejection_reason(detection, mask, 100, 100, task_config) == "negative_prompt"


def test_deleaf_dark_pipe_like_mask_is_rejected():
    task_config = load_task_config("configs/deleaf.yaml")
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    image[45:55, 10:90] = [20, 25, 20]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 10:90] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="pale green leaf stem",
        confidence=0.8,
        bbox_xyxy=[10.0, 45.0, 90.0, 55.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="pale green leaf stem",
    )

    assert (
        rejection_reason(detection, mask, 100, 100, task_config, image_bgr=image)
        == "likely_dark_pipe_or_frame"
    )


def test_deleaf_green_branch_like_mask_is_kept_by_visual_filter():
    task_config = load_task_config("configs/deleaf.yaml")
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    image[45:55, 10:90] = [45, 120, 70]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 10:90] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="pale green leaf stem",
        confidence=0.8,
        bbox_xyxy=[10.0, 45.0, 90.0, 55.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="pale green leaf stem",
    )

    assert rejection_reason(detection, mask, 100, 100, task_config, image_bgr=image) is None


def test_deleaf_leaf_cluster_prompt_can_keep_branch_with_leaflets():
    task_config = load_task_config("configs/deleaf.yaml")
    image = np.full((100, 100, 3), [80, 130, 80], dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:45, 20:85] = 1
    mask[45:85, 42:62] = 1
    detection = SegmentationDetection(
        frame_id="pre_000001",
        timestamp_ns=123,
        det_id="pre_000001_det_000",
        semantic_class="petiole",
        raw_class_name="leaf branch with leaflets",
        confidence=0.12,
        bbox_xyxy=[20.0, 20.0, 85.0, 85.0],
        mask_path="segmentation_predictions/deleaf/masks/a.npz",
        mask_area_px=int(mask.sum()),
        backend="yoloe_prompt",
        prompt_used="leaf branch with leaflets",
    )

    assert rejection_reason(detection, mask, 100, 100, task_config, image_bgr=image) is None


def test_deleaf_refinement_removes_dark_pipe_pixels_from_leaf_cluster_mask():
    task_config = load_task_config("configs/deleaf.yaml")
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 10:90] = 1
    image[25:75, 10:35] = [15, 15, 15]
    image[32:68, 40:82] = [55, 135, 85]

    refined = refine_mask_for_task(mask, image, task_config, "leaf branch with leaflets")

    assert refined[45, 55]
    assert not refined[45, 20]
    assert refined.sum() < mask.sum()
