import numpy as np

from yolo_aria_harvest_poc.lifting.mask_lifting import lift_single_segmentation_detection
from yolo_aria_harvest_poc.task_config import load_task_config


def test_generic_mask_lifting_returns_ok_cluster(tmp_path):
    task_config = load_task_config("configs/deleaf.yaml")
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:8, 3:8] = True
    points_world = np.array(
        [
            [0.00, 0.00, 1.00],
            [0.01, 0.00, 1.01],
            [0.00, 0.01, 1.02],
            [0.01, 0.01, 1.03],
        ],
        dtype=float,
    )
    uv = np.array([[4, 4], [5, 4], [4, 5], [5, 5]], dtype=float)
    depth = np.array([1.00, 1.01, 1.02, 1.03], dtype=float)
    valid = np.ones((4,), dtype=bool)
    points_dir = tmp_path / "points"
    points_dir.mkdir()

    lifted, _, _ = lift_single_segmentation_detection(
        split="pre",
        frame_id="pre_000001",
        detection={"det_id": "det_000", "confidence": 0.9, "semantic_class": "petiole"},
        mask=mask,
        points_world=points_world,
        uv=uv,
        depth=depth,
        valid_projection=valid,
        out_dir=tmp_path,
        points_dir=points_dir,
        task_config=task_config,
    )

    assert lifted["status"] == "ok"
    assert lifted["centroid_3d"] is not None
    assert lifted["num_points_after_cluster"] >= 3
