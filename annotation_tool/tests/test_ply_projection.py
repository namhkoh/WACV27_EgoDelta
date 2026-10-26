from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from egomask.ply import read_gaussian_projection_fields, read_vertex_fields
from egomask.projection import Camera, project_points, rasterize_components, select_temporal_views


class PlyTests(unittest.TestCase):
    def test_reads_ascii_vertex_fields_and_decodes_gaussian_values(self) -> None:
        content = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float scale_0
property float scale_1
property float opacity
end_header
1 2 3 -2 -3 0
4 5 6 -1 -4 2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.ply"
            path.write_text(content, encoding="ascii")
            xyz = read_vertex_fields(path, ("x", "y", "z"))
            projection = read_gaussian_projection_fields(path)

        np.testing.assert_allclose(xyz["x"], [1, 4])
        np.testing.assert_allclose(projection["world_scale"], np.exp([-2, -1]))
        np.testing.assert_allclose(projection["alpha"], [0.5, 1.0 / (1.0 + np.exp(-2))])


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = Camera(
            image_name="frame_000001.jpg",
            width=100,
            height=80,
            fx=50.0,
            fy=50.0,
            cx=49.5,
            cy=39.5,
            center=np.zeros(3),
            rotation=np.eye(3),
        )

    def test_projects_world_points(self) -> None:
        uv, depth, valid = project_points(np.asarray([[0, 0, 1], [1, 0, -1]]), self.camera)
        np.testing.assert_allclose(uv[0], [49.5, 39.5])
        self.assertEqual(depth[0], 1.0)
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])

    def test_rasterizer_uses_instance_labels_and_occlusion(self) -> None:
        components = [
            {
                "xyz": np.asarray([[0.0, 0.0, 1.0]]),
                "world_scale": np.asarray([0.02]),
                "alpha": np.asarray([1.0]),
            },
            {
                "xyz": np.asarray([[0.2, 0.0, 2.0]]),
                "world_scale": np.asarray([0.02]),
                "alpha": np.asarray([1.0]),
            },
        ]
        scene_depth = np.full((80, 100), np.inf, dtype=np.float32)
        scene_depth[40, 54] = 1.0
        image, metadata = rasterize_components(
            components,
            self.camera,
            scene_depth=scene_depth,
            occlusion_tolerance_m=0.01,
            occlusion_relative_tolerance=0.0,
        )
        labels = np.asarray(image)
        self.assertGreater(np.count_nonzero(labels == 1), 0)
        self.assertEqual(np.count_nonzero(labels == 2), 0)
        self.assertGreater(metadata["visible_pixels"], 0)

    def test_temporal_selection_is_deterministic_and_spread(self) -> None:
        cameras = [
            Camera(
                image_name=f"frame_{index:06d}.jpg",
                width=100,
                height=80,
                fx=50.0,
                fy=50.0,
                cx=49.5,
                cy=39.5,
                center=np.zeros(3),
                rotation=np.eye(3),
            )
            for index in range(120)
        ]
        counts = [[index + 1] for index in range(120)]
        selected = select_temporal_views(cameras, counts, 25, min_visible_points=1)
        self.assertEqual(len(selected), 25)
        self.assertEqual(len(set(selected)), 25)
        self.assertLess(selected[0], 10)
        self.assertGreater(selected[-1], 110)


if __name__ == "__main__":
    unittest.main()
