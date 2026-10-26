import numpy as np

from egodelta_diff.evaluation import evaluate_points


def test_one_to_one_gated_matching():
    gt = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    predictions = np.asarray([[0.01, 0, 0], [0.02, 0, 0], [1.04, 0, 0]], dtype=float)
    result = evaluate_points(predictions, gt, threshold_m=0.03)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
