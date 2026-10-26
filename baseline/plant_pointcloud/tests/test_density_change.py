from __future__ import annotations

import numpy as np

from tomato_cutpoints.change.density import compute_density_drop_score
from tomato_cutpoints.change.scoring import combine_change_scores


def test_density_drop_scores_thin_removed_structure_with_nearby_background():
    line = np.asarray([[0.0, 0.0, z] for z in np.linspace(-0.03, 0.03, 7)], dtype=np.float64)
    background_pre = np.asarray([[0.08, 0.0, z] for z in np.linspace(-0.03, 0.03, 7)], dtype=np.float64)
    pre = np.vstack([line, background_pre])
    post = background_pre.copy()
    config = {
        "density": {
            "enabled": True,
            "radius_m": 0.035,
            "min_pre_neighbors": 3,
            "min_drop_fraction": 0.30,
            "min_drop_count": 2,
            "max_post_neighbors": 1,
            "candidate_threshold": 0.25,
        }
    }

    score, diagnostics = compute_density_drop_score(pre, post, config)

    assert diagnostics["n_density_candidates"] >= len(line)
    assert float(np.median(score[: len(line)])) > 0.8
    assert float(np.max(score[len(line) :])) < 0.25


def test_max_fusion_preserves_density_candidates_when_nn_score_is_low():
    combined, diagnostics = combine_change_scores(
        nn_dist=np.asarray([0.0, 0.0]),
        nn_threshold=0.025,
        m3c2_dist=None,
        m3c2_threshold=0.025,
        config={"scoring": {"nn_weight": 1.0, "density_weight": 1.0, "fusion": "max", "min_combined_score": 0.25}},
        density_score=np.asarray([0.8, 0.0]),
    )

    assert diagnostics["n_removed_candidates"] == 1
    assert combined.tolist() == [0.8, 0.0]
