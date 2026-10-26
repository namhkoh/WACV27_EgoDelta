from __future__ import annotations

from pathlib import Path

import numpy as np

from egodelta_diff.data import find_scene_targets, match_targets_to_points, parse_gt_markdown


def test_parse_gt_markdown_and_find_scene_targets(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.md"
    gt_path.write_text(
        "\n".join(
            [
                "1. 4d5ebec9-3ebd-4c65-b01d-8d2a0c668df2.vrs",
                "deleafing",
                "[0.1, 0.2, 0.3]",
                "harvesting",
                "[0.4, 0.5, 0.6]",
            ]
        )
    )

    entries = parse_gt_markdown(gt_path)
    found = find_scene_targets("egodelta_4d5ebec93ebd_leaf", entries)

    assert found is not None
    scene_uuid, task, points = found
    assert scene_uuid == "4d5ebec9-3ebd-4c65-b01d-8d2a0c668df2"
    assert task == "deleafing"
    np.testing.assert_allclose(points, [[0.1, 0.2, 0.3]])


def test_match_targets_to_points_uses_unique_assignments() -> None:
    gt_points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    points = np.asarray([[0.01, 0.0, 0.0], [1.02, 0.0, 0.0]], dtype=np.float32)
    scores = np.asarray([0.7, 0.8], dtype=np.float32)
    source_indices = np.asarray([10, 20], dtype=np.int64)

    matches = match_targets_to_points(gt_points, points, scores, source_indices)

    assert [match.target_index for match in matches] == [0, 1]
    assert [match.source_index for match in matches] == [10, 20]
    assert [round(match.distance_m, 2) for match in matches] == [0.01, 0.02]

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from egodelta_diff.data import load_2dgs_ply


def test_load_2dgs_ply_activates_scale_and_opacity(tmp_path: Path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    vertices = np.zeros(1, dtype=dtype)
    vertices["scale_0"] = np.log(0.25)
    vertices["scale_1"] = np.log(0.5)
    vertices["rot_0"] = 2.0
    path = tmp_path / "point_cloud.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    g = load_2dgs_ply(path, "cpu")
    assert np.allclose(g.scales.numpy(), [[0.25, 0.5]])
    assert np.allclose(g.opacity.numpy(), [0.5])

