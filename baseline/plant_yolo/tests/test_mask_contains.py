import numpy as np

from yolo_aria_harvest_poc.mask_lifting import mask_contains_uv


def test_mask_contains_uv_inside_and_outside():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    uv = np.array(
        [
            [1.2, 1.2],
            [3.9, 3.9],
            [4.0, 4.0],
            [-1.0, 2.0],
            [2.0, np.nan],
        ]
    )

    contained = mask_contains_uv(mask, uv)

    assert contained.tolist() == [True, True, False, False, False]

