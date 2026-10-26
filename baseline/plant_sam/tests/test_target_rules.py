import numpy as np

from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.target_rules import generate_task_predictions


def test_harvesting_proxy_target_is_above_tomato_centroid_along_world_up(tmp_path):
    points = np.array([[0, 0, 0.9], [0, 0, 1.0], [0.02, 0, 1.1], [-0.02, 0, 1.1]])
    path = tmp_path / "tomato.npy"
    np.save(path, points)
    instance = {
        "instance_id": "pre_harvesting_primary_000",
        "semantic_group": "primary",
        "target_class": "tomato",
        "centroid_3d": [0, 0, 1.0],
        "points_3d_path": str(path),
        "source_prompts": ["tomato fruit"],
    }
    candidate = {
        "pre_instance_id": instance["instance_id"],
        "status": "disappeared",
        "final_change_score": 0.8,
        "visible_post_views": 3,
    }

    predictions = generate_task_predictions([candidate], [instance], PipelineConfig(), "harvesting")

    assert predictions[0]["target_point_type"] == "proxy_cut_point_from_tomato_top_surface"
    assert predictions[0]["target_point_3d_world"][2] > instance["centroid_3d"][2]


def test_deleafing_ambiguous_endpoint_outputs_two_endpoints(tmp_path):
    points = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    path = tmp_path / "petiole.npy"
    np.save(path, points)
    instance = {
        "instance_id": "pre_deleafing_primary_000",
        "semantic_group": "primary",
        "target_class": "petiole_or_leaf_stem",
        "centroid_3d": [0.1, 0, 0],
        "points_3d_path": str(path),
        "source_prompts": ["petiole"],
    }
    candidate = {
        "pre_instance_id": instance["instance_id"],
        "status": "disappeared",
        "final_change_score": 0.9,
        "visible_post_views": 3,
    }

    predictions = generate_task_predictions(
        [candidate], [instance], PipelineConfig(task="deleafing"), "deleafing"
    )

    assert predictions[0]["target_point_type"] == "ambiguous_petiole_endpoint"
    assert len(predictions[0]["candidate_target_points_3d_world"]) == 2


def test_deleafing_ignores_short_lateral_auxiliary_as_main_stem(tmp_path):
    petiole_points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    aux_points = np.array([[0.08, 0.01, 0.0], [0.16, 0.01, 0.01], [0.24, 0.01, 0.0]])
    petiole_path = tmp_path / "petiole.npy"
    aux_path = tmp_path / "bad_aux.npy"
    np.save(petiole_path, petiole_points)
    np.save(aux_path, aux_points)
    instance = {
        "instance_id": "pre_deleafing_primary_000",
        "semantic_group": "primary",
        "target_class": "petiole_or_leaf_stem",
        "centroid_3d": [0.1, 0, 0],
        "points_3d_path": str(petiole_path),
        "source_prompts": ["tomato petiole"],
    }
    bad_aux = {
        "instance_id": "pre_deleafing_auxiliary_000",
        "semantic_group": "auxiliary",
        "target_class": "auxiliary_structure",
        "centroid_3d": [0.16, 0.01, 0.0],
        "points_3d_path": str(aux_path),
        "source_prompts": ["main stem of tomato plant, without any leaves or branches"],
    }
    candidate = {
        "pre_instance_id": instance["instance_id"],
        "status": "disappeared",
        "final_change_score": 0.9,
        "visible_post_views": 3,
    }

    predictions = generate_task_predictions(
        [candidate], [instance, bad_aux], PipelineConfig(task="deleafing"), "deleafing"
    )

    assert predictions[0]["target_point_type"] == "ambiguous_petiole_endpoint"
    assert predictions[0]["used_auxiliary_structure"] is False


def test_deleafing_uses_reliable_vertical_auxiliary_main_stem(tmp_path):
    petiole_points = np.array([[0.0, 0.0, 0.05], [0.1, 0.0, 0.05], [0.2, 0.0, 0.05]])
    stem_points = np.array([[0.0, 0.01, -0.05], [0.0, 0.01, 0.05], [0.0, 0.01, 0.15]])
    petiole_path = tmp_path / "petiole.npy"
    stem_path = tmp_path / "stem.npy"
    np.save(petiole_path, petiole_points)
    np.save(stem_path, stem_points)
    instance = {
        "instance_id": "pre_deleafing_primary_000",
        "semantic_group": "primary",
        "target_class": "petiole_or_leaf_stem",
        "centroid_3d": [0.1, 0, 0.05],
        "points_3d_path": str(petiole_path),
        "source_prompts": ["tomato petiole"],
    }
    good_aux = {
        "instance_id": "pre_deleafing_auxiliary_000",
        "semantic_group": "auxiliary",
        "target_class": "auxiliary_structure",
        "centroid_3d": [0.0, 0.01, 0.05],
        "points_3d_path": str(stem_path),
        "source_prompts": ["main stem of tomato plant, without any leaves or branches"],
    }
    candidate = {
        "pre_instance_id": instance["instance_id"],
        "status": "disappeared",
        "final_change_score": 0.9,
        "visible_post_views": 3,
    }

    predictions = generate_task_predictions(
        [candidate], [instance, good_aux], PipelineConfig(task="deleafing"), "deleafing"
    )

    assert predictions[0]["target_point_type"] == "petiole_endpoint_near_main_stem"
    assert predictions[0]["used_auxiliary_structure"] is True
