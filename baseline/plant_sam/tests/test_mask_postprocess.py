import numpy as np

from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.mask_postprocess import filter_and_deduplicate_frame_masks


def _record(mask, prompt="tomato fruit", score=0.9):
    return {
        "mask_id": prompt.replace(" ", "_"),
        "prompt": prompt,
        "semantic_group": "primary",
        "score": score,
        "mask": mask,
    }


def test_duplicate_masks_with_high_iou_deduplicate():
    config = PipelineConfig(min_mask_area_px=1, mask_dedup_iou_thres=0.5)
    mask_a = np.zeros((64, 64), dtype=bool)
    mask_b = np.zeros((64, 64), dtype=bool)
    mask_a[10:30, 10:30] = True
    mask_b[11:31, 11:31] = True

    filtered, _ = filter_and_deduplicate_frame_masks(
        [_record(mask_a, "tomato fruit", 0.8), _record(mask_b, "ripe tomato", 0.9)],
        config=config,
        task="harvesting",
        image_shape=(64, 64),
    )

    assert len(filtered) == 1
    assert set(filtered[0]["source_prompts"]) == {"tomato fruit", "ripe tomato"}
    assert filtered[0]["prompt"] == "ripe tomato"


def test_small_masks_are_filtered():
    config = PipelineConfig(min_mask_area_px=10)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10, 10] = True

    filtered, stats = filter_and_deduplicate_frame_masks(
        [_record(mask)],
        config=config,
        task="harvesting",
        image_shape=(64, 64),
    )

    assert filtered == []
    assert stats["rejected"]["area_below_threshold"] == 1


def test_deleafing_primary_default_min_area_is_50_pixels():
    config = PipelineConfig(task="deleafing")
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:17, 10:17] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(64, 64),
    )

    assert int(mask.sum()) == 49
    assert filtered == []
    assert stats["rejected"]["area_below_threshold"] == 1


def test_deleafing_primary_allows_moderately_large_petiole_candidate():
    config = PipelineConfig(task="deleafing", deleafing_primary_reject_vertical_axis=False)
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[300:380, 200:500] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, _ = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(1000, 1000),
    )

    assert int(mask.sum()) == 24000
    assert len(filtered) == 1


def test_huge_masks_are_filtered():
    config = PipelineConfig(min_mask_area_px=1, max_mask_area_frac=0.4)
    mask = np.ones((64, 64), dtype=bool)

    filtered, stats = filter_and_deduplicate_frame_masks(
        [_record(mask)],
        config=config,
        task="harvesting",
        image_shape=(64, 64),
    )

    assert filtered == []
    assert stats["rejected"]["area_above_threshold"] == 1


def test_auxiliary_masks_use_relaxed_area_threshold():
    config = PipelineConfig(min_mask_area_px=100, aux_min_mask_area_px=5, aux_sam_score_thres=0.1)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:13, 10:13] = True
    record = _record(mask, prompt="tomato peduncle", score=0.2)
    record["semantic_group"] = "auxiliary"

    filtered, _ = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="harvesting",
        image_shape=(64, 64),
    )

    assert len(filtered) == 1
    assert filtered[0]["semantic_group"] == "auxiliary"


def test_deleafing_auxiliary_main_stem_rejects_horizontal_petioles():
    config = PipelineConfig(task="deleafing", deleafing_aux_min_mask_area_px=1)
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:33, 8:56] = True
    record = _record(mask, prompt="main stem of tomato plant, without any leaves or branches", score=0.9)
    record["semantic_group"] = "auxiliary"

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(64, 64),
    )

    assert filtered == []
    assert stats["rejected"]["main_stem_aux_too_short"] == 1


def test_deleafing_auxiliary_main_stem_rejects_old_broad_prompt():
    config = PipelineConfig(task="deleafing", deleafing_aux_min_mask_area_px=1)
    mask = np.zeros((64, 64), dtype=bool)
    mask[5:60, 30:33] = True
    record = _record(mask, prompt="main stem", score=0.9)
    record["semantic_group"] = "auxiliary"

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(64, 64),
    )

    assert filtered == []
    assert stats["rejected"]["deleafing_aux_prompt_not_active"] == 1


def test_deleafing_auxiliary_main_stem_accepts_tall_vertical_mask():
    config = PipelineConfig(task="deleafing", deleafing_aux_min_mask_area_px=1)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 48:52] = True
    record = _record(mask, prompt="main stem of tomato plant, without any leaves or branches", score=0.9)
    record["semantic_group"] = "auxiliary"

    filtered, _ = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(100, 100),
    )

    assert len(filtered) == 1


def test_deleafing_primary_rejects_long_horizontal_support_like_mask():
    config = PipelineConfig(
        task="deleafing",
        deleafing_min_mask_area_px=1,
        deleafing_max_mask_area_frac=0.5,
        deleafing_primary_reject_vertical_axis=False,
    )
    mask = np.zeros((100, 200), dtype=bool)
    mask[40:44, 20:160] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(100, 200),
    )

    assert filtered == []
    assert stats["rejected"]["deleafing_primary_long_horizontal_support_like"] == 1


def test_deleafing_primary_rejects_long_pipe_like_mask_in_any_orientation():
    config = PipelineConfig(
        task="deleafing",
        deleafing_min_mask_area_px=1,
        deleafing_max_mask_area_frac=0.5,
        deleafing_primary_reject_vertical_axis=False,
    )
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:90, 96:100] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(100, 200),
    )

    assert filtered == []
    assert stats["rejected"]["deleafing_primary_long_pipe_like"] == 1


def test_deleafing_primary_rejects_image_horizontal_axis_like_main_stem():
    config = PipelineConfig(
        task="deleafing",
        deleafing_min_mask_area_px=1,
        deleafing_max_mask_area_frac=0.5,
        deleafing_long_horizontal_width_frac=2.0,
        deleafing_pipe_like_long_side_frac=2.0,
    )
    mask = np.zeros((120, 120), dtype=bool)
    mask[58:62, 10:110] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(120, 120),
    )

    assert filtered == []
    assert stats["rejected"]["deleafing_primary_vertical_axis_like_main_stem"] == 1


def test_deleafing_primary_cleanup_removes_axis_aligned_component_but_keeps_mask_record():
    config = PipelineConfig(
        task="deleafing",
        deleafing_min_mask_area_px=1,
        deleafing_max_mask_area_frac=0.5,
        deleafing_long_horizontal_width_frac=2.0,
        deleafing_pipe_like_long_side_frac=2.0,
    )
    mask = np.zeros((160, 160), dtype=bool)
    mask[76:84, 15:95] = True
    mask[20:120, 124:130] = True
    kept_area = int(mask[20:120, 124:130].sum())
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(160, 160),
    )

    assert len(filtered) == 1
    assert "parent_mask_id" not in filtered[0]
    assert "_cc_" not in filtered[0]["mask_id"]
    assert filtered[0]["area_px"] == kept_area
    assert filtered[0]["cleanup_stats"]["num_components"] == 2
    assert filtered[0]["cleanup_stats"]["kept_components"] == 1
    assert filtered[0]["cleanup_stats"]["axis_rejected_components"] == 1
    assert stats["kept_after_dedup"] == 1


def test_deleafing_primary_keeps_horizontal_elongated_candidate():
    config = PipelineConfig(
        task="deleafing",
        deleafing_min_mask_area_px=1,
        deleafing_max_mask_area_frac=0.5,
        deleafing_long_horizontal_aspect_ratio=20.0,
        deleafing_pipe_like_aspect_ratio=20.0,
        deleafing_primary_reject_vertical_axis=False,
    )
    mask = np.zeros((120, 120), dtype=bool)
    mask[54:66, 20:105] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, _ = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(120, 120),
    )

    assert len(filtered) == 1


def test_deleafing_primary_keeps_one_record_after_component_cleanup():
    config = PipelineConfig(task="deleafing", deleafing_primary_reject_vertical_axis=False)
    mask = np.zeros((120, 120), dtype=bool)
    mask[54:62, 20:55] = True
    mask[10:15, 10:15] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(120, 120),
    )

    assert len(filtered) == 1
    assert "parent_mask_id" not in filtered[0]
    assert "_cc_" not in filtered[0]["mask_id"]
    assert filtered[0]["cleanup_stats"]["num_components"] == 2
    assert filtered[0]["cleanup_stats"]["kept_components"] == 1
    assert stats["kept_after_dedup"] == 1


def test_deleafing_primary_keeps_disconnected_valid_components_in_one_mask():
    config = PipelineConfig(
        task="deleafing",
        deleafing_max_mask_area_frac=0.10,
        deleafing_primary_reject_vertical_axis=False,
    )
    mask = np.zeros((160, 160), dtype=bool)
    mask[45:57, 20:65] = True
    mask[95:107, 75:120] = True
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(160, 160),
    )

    assert len(filtered) == 1
    assert "parent_mask_id" not in filtered[0]
    assert "_cc_" not in filtered[0]["mask_id"]
    assert filtered[0]["cleanup_stats"]["num_components"] == 2
    assert filtered[0]["cleanup_stats"]["kept_components"] == 2
    assert stats["kept_after_dedup"] == 1


def test_deleafing_primary_final_filter_removes_disconnected_leaf_component():
    config = PipelineConfig(
        task="deleafing",
        deleafing_min_mask_area_px=1,
        deleafing_max_mask_area_frac=0.50,
        deleafing_long_horizontal_aspect_ratio=20.0,
        deleafing_pipe_like_aspect_ratio=20.0,
        deleafing_primary_reject_vertical_axis=False,
    )
    mask = np.zeros((160, 160), dtype=bool)
    mask[75:83, 20:110] = True
    mask[120:132, 30:42] = True
    petiole_area = int(mask[75:83, 20:110].sum())
    record = _record(
        mask,
        prompt="side stem of tomato plant, which is not a main stem and hung horizontally",
        score=0.9,
    )

    filtered, stats = filter_and_deduplicate_frame_masks(
        [record],
        config=config,
        task="deleafing",
        image_shape=(160, 160),
    )

    assert len(filtered) == 1
    assert filtered[0]["area_px"] == petiole_area
    assert filtered[0]["cleanup_stats"]["final_num_components"] == 2
    assert filtered[0]["cleanup_stats"]["final_kept_components"] == 1
    assert filtered[0]["cleanup_stats"]["final_rejected_components"] == 1
    assert stats["rejected"]["deleafing_primary_component_compact_leaf_blob"] == 1
