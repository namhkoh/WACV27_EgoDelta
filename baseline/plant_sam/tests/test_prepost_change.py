from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.prepost_change import detect_prepost_changes


def _instance(instance_id, centroid, visible=None):
    item = {
        "instance_id": instance_id,
        "semantic_group": "primary",
        "target_class": "tomato",
        "centroid_3d": centroid,
        "radius_m": 0.03,
    }
    if visible is not None:
        item["visible_post_views_override"] = visible
    return item


def test_pre_instance_near_post_instance_is_still_present():
    config = PipelineConfig(min_visible_post_views=1)
    pre = [_instance("pre0", [0, 0, 1])]
    post = [_instance("post0", [0.02, 0, 1])]

    changes = detect_prepost_changes(pre, post, config, "harvesting")

    assert changes[0]["status"] == "still_present"


def test_pre_instance_without_post_match_but_visible_is_disappeared():
    config = PipelineConfig(min_visible_post_views=1)
    pre = [_instance("pre0", [0, 0, 1], visible=1)]

    changes = detect_prepost_changes(pre, [], config, "harvesting", post_points_world=[])

    assert changes[0]["status"] == "disappeared"
    assert changes[0]["final_change_score"] == 1.0


def test_pre_instance_without_post_visibility_is_unknown():
    config = PipelineConfig(min_visible_post_views=2)
    pre = [_instance("pre0", [0, 0, 1], visible=1)]

    changes = detect_prepost_changes(pre, [], config, "harvesting")

    assert changes[0]["status"] == "unknown_not_enough_post_visibility"

