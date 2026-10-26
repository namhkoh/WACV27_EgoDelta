from __future__ import annotations

import json

import numpy as np

from tomato_cutpoints.pipeline import run_pipeline
from tomato_cutpoints.synthetic import write_synthetic_trial


def test_synthetic_harvest_finds_removed_tomato_and_icp(tmp_path):
    paths = write_synthetic_trial(tmp_path, "harvest", seed=21, perturb_post=True)
    run_dir = tmp_path / "run"
    targets_doc = run_pipeline(
        task="harvest",
        pre_points=paths["pre_points"],
        post_points=paths["post_points"],
        pre_observations=paths["pre_observations"],
        post_observations=paths["post_observations"],
        stable_roi_json=paths["roi"],
        out_dir=run_dir,
        config_path=paths["config"],
        refine_icp=True,
    )
    gt = json.loads(paths["gt"].read_text())
    assert len(targets_doc["targets"]) == 1
    target = targets_doc["targets"][0]
    center_err = np.linalg.norm(np.asarray(target["object_center_world"]) - np.asarray(gt["targets"][0]["object_center_world"]))
    cut_err = np.linalg.norm(np.asarray(target["cut_point_world"]) - np.asarray(gt["targets"][0]["cut_point_world"]))
    assert center_err < 0.06
    assert cut_err < 0.08
    assert (run_dir / "clusters.csv").exists()
    assert (run_dir / "removed_points.parquet").exists()
    assert (run_dir / "plys" / "cut_points.ply").exists()

    estimated_t = np.loadtxt(run_dir / "alignment_transform_post_to_pre.txt")
    gt_t = np.loadtxt(paths["transform_gt"])
    assert np.linalg.norm(estimated_t[:3, 3] - gt_t[:3, 3]) < 0.035
