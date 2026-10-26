from __future__ import annotations

import json

import numpy as np

from tomato_cutpoints.pipeline import run_pipeline
from tomato_cutpoints.synthetic import write_synthetic_trial


def test_synthetic_deleaf_finds_cut_endpoint(tmp_path):
    paths = write_synthetic_trial(tmp_path, "deleaf", seed=22, perturb_post=True)
    targets_doc = run_pipeline(
        task="deleaf",
        pre_points=paths["pre_points"],
        post_points=paths["post_points"],
        pre_observations=paths["pre_observations"],
        post_observations=paths["post_observations"],
        stable_roi_json=paths["roi"],
        out_dir=tmp_path / "run",
        config_path=paths["config"],
        refine_icp=True,
    )
    gt = json.loads(paths["gt"].read_text())
    assert len(targets_doc["targets"]) == 1
    target = targets_doc["targets"][0]
    assert target["object_type"] == "leaf"
    cut_err = np.linalg.norm(np.asarray(target["cut_point_world"]) - np.asarray(gt["targets"][0]["cut_point_world"]))
    assert cut_err < 0.08
    assert np.linalg.norm(target["removal_direction_world"]) > 0.9
    assert (tmp_path / "run" / "plys" / "cut_points.ply").exists()
