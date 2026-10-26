import numpy as np

from yolo_aria_harvest_poc.tasks.deleaf import choose_petiole_base


def test_deleaf_cutpoint_chooses_endpoint_closer_to_main_stem():
    endpoint_a = np.array([0.01, 0.0, 0.0])
    endpoint_b = np.array([0.50, 0.0, 0.0])
    main_stem_axis = {
        "point_on_axis": [0.0, 0.0, 0.0],
        "direction": [0.0, 0.0, 1.0],
    }

    base, tip, used = choose_petiole_base(
        endpoint_a,
        endpoint_b,
        main_stem_axis=main_stem_axis,
        plant_centroid=None,
    )

    assert used is True
    assert np.allclose(base, endpoint_a)
    assert np.allclose(tip, endpoint_b)
