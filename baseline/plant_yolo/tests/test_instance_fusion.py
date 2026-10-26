import numpy as np

from yolo_aria_harvest_poc.instance_fusion import fuse_lifted_detections


def test_instance_fusion_produces_two_tomato_instances(tmp_path):
    points_dir = tmp_path / "lifted_3d" / "points"
    points_dir.mkdir(parents=True)
    lifted = []
    centers = [
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [1.0, 0.0, 0.0],
        [1.02, 0.0, 0.0],
        [1.0, 0.02, 0.0],
    ]
    for idx, center in enumerate(centers):
        center = np.asarray(center, dtype=float)
        points = center + 0.005 * np.random.default_rng(idx).normal(size=(12, 3))
        path = points_dir / f"det_{idx}.npy"
        np.save(path, points)
        lifted.append(
            {
                "det_id": f"det_{idx}",
                "frame_id": f"frame_{idx}",
                "status": "ok",
                "centroid_3d": center.tolist(),
                "confidence_2d": 0.9,
                "points_3d_path": str(path.relative_to(tmp_path)),
            }
        )

    instances = fuse_lifted_detections(
        split="pre",
        lifted_detections=lifted,
        out_dir=tmp_path,
        eps_m=0.10,
        min_cluster_samples=2,
        min_frames=2,
        min_points=20,
        min_confidence=0.25,
    )

    assert len(instances) == 2
    centroids = sorted(instance["centroid_3d"][0] for instance in instances)
    assert centroids[0] < 0.1
    assert centroids[1] > 0.9

