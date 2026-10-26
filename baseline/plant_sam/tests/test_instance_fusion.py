import numpy as np

from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.instance_fusion import fuse_lifted_detections


def test_synthetic_clusters_fuse_into_correct_number_of_instances(tmp_path):
    config = PipelineConfig(
        task="harvesting",
        instance_min_frames=1,
        tomato_instance_min_points=1,
        sam_score_thres=0.0,
        tomato_instance_eps_m=0.04,
    )
    records = []
    for idx, center in enumerate(([0, 0, 1], [0.01, 0, 1], [0.2, 0, 1], [0.21, 0, 1])):
        points = np.asarray(center, dtype=float).reshape((1, 3))
        path = tmp_path / f"pts_{idx}.npy"
        np.save(path, points)
        records.append(
            {
                "mask_id": f"m{idx}",
                "frame_id": f"f{idx}",
                "status": "ok",
                "semantic_group": "primary",
                "target_class": "tomato",
                "centroid_3d": list(center),
                "score_2d": 0.9,
                "source_prompts": ["tomato fruit"],
                "points_3d_path": str(path),
            }
        )

    instances = fuse_lifted_detections(records, config, "harvesting", "pre", tmp_path)

    assert len(instances) == 2


def test_deleafing_instance_size_filter_keeps_only_petiole_sized_cluster(tmp_path):
    config = PipelineConfig(
        task="deleafing",
        petiole_instance_min_frames=1,
        petiole_instance_min_points=1,
        deleafing_sam_score_thres=0.0,
        petiole_instance_eps_m=0.20,
        petiole_min_3d_major_extent_m=0.12,
        petiole_max_3d_major_extent_m=0.55,
    )
    point_sets = [
        np.array([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.20, 0.0, 0.30]]),
        np.array([[1.0, 0.0, 0.0], [1.02, 0.0, 0.0], [1.02, 0.0, 0.02]]),
        np.array([[2.0, 0.0, 0.0], [2.02, 0.0, 0.0], [2.02, 0.0, 0.80]]),
    ]
    records = []
    for idx, points in enumerate(point_sets):
        path = tmp_path / f"petiole_pts_{idx}.npy"
        np.save(path, points)
        records.append(
            {
                "mask_id": f"m{idx}",
                "frame_id": f"f{idx}",
                "status": "ok",
                "semantic_group": "primary",
                "target_class": "petiole_or_leaf_stem",
                "centroid_3d": np.median(points, axis=0).tolist(),
                "score_2d": 0.9,
                "source_prompts": ["tomato petiole"],
                "points_3d_path": str(path),
            }
        )

    instances = fuse_lifted_detections(records, config, "deleafing", "pre", tmp_path)

    assert len(instances) == 1
    assert instances[0]["source_mask_ids"] == ["m0"]
    assert instances[0]["metric_size_stats"]["major_extent_m"] >= 0.12
