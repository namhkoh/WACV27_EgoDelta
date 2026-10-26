from __future__ import annotations

import numpy as np
import pandas as pd

from tomato_cutpoints.io.aria_mps import load_semidense_points, write_minimal_observations_csv_gz


def test_loader_reads_minimal_semidense_points_and_observations(tmp_path):
    points_path = tmp_path / "slam" / "semidense_points.csv.gz"
    obs_path = tmp_path / "slam" / "semidense_observations.csv.gz"
    points_path.parent.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "uid": [10, 11, 12],
            "px_world": [0.0, 1.0, 2.0],
            "py_world": [0.1, 1.1, 2.1],
            "pz_world": [0.2, 1.2, 2.2],
            "inv_dist_std": [0.001, 0.002, 0.003],
            "dist_std": [0.002, 0.003, 0.004],
            "graph_uid": ["g", "g", "g"],
        }
    )
    df.to_csv(points_path, index=False, compression="gzip")
    write_minimal_observations_csv_gz(obs_path, n_points=3, uid_start=10, repeats=4)

    cloud = load_semidense_points(points_path, obs_path)

    assert cloud.xyz.shape == (3, 3)
    assert np.allclose(cloud.xyz[1], [1.0, 1.1, 1.2])
    assert cloud.frame_id == "g"
    assert cloud.points_df["obs_count"].tolist() == [4, 4, 4]
    assert cloud.metadata["loader"] in {"pandas_csv", "projectaria_tools"}
