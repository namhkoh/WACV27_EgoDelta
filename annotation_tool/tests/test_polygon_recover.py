from __future__ import annotations

import unittest

import cv2
import numpy as np

from egomask.polygon_recover import recover_convex_polygon_fill, recover_polygon_fill


class PolygonRecoveryTests(unittest.TestCase):
    def test_fills_a_human_outline_drawn_over_existing_labels(self) -> None:
        initial = np.zeros((160, 180), dtype=np.uint8)
        cv2.circle(initial, (70, 80), 35, 1, cv2.FILLED)
        cv2.circle(initial, (115, 80), 35, 2, cv2.FILLED)
        source = initial.copy()
        polygon = np.array([[30, 30], [150, 40], [145, 125], [40, 135]], dtype=np.int32)
        cv2.polylines(source, [polygon], True, 3, thickness=2)

        recovered, metrics = recover_polygon_fill(
            initial,
            source,
            closing_size=11,
            minimum_contour_area=200,
        )

        self.assertTrue(recovered[80, 90])
        self.assertFalse(recovered[10, 10])
        self.assertEqual(metrics["contour_count"], 1)
        self.assertGreater(int(recovered.sum()), 9_000)

    def test_fills_a_small_convex_fruit_outline(self) -> None:
        initial = np.zeros((100, 120), dtype=np.uint8)
        cv2.circle(initial, (60, 50), 25, 1, cv2.FILLED)
        source = initial.copy()
        cv2.ellipse(source, (60, 50), (20, 17), 0, 0, 360, 0, thickness=2)

        recovered, metrics = recover_convex_polygon_fill(initial, source)

        self.assertTrue(recovered[50, 60])
        self.assertFalse(recovered[10, 10])
        self.assertEqual(metrics["contour_count"], 1)
        self.assertGreater(int(recovered.sum()), 900)


if __name__ == "__main__":
    unittest.main()
