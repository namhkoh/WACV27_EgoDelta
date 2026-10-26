from __future__ import annotations

import numpy as np
import unittest

from egomask.brush_repair import repair_brush_persistence_mask


class BrushPersistenceRepairTests(unittest.TestCase):
    def test_reconnects_dabs_fills_holes_and_removes_island(self) -> None:
        mask = np.zeros((48, 64), dtype=bool)
        mask[15:29, 12:24] = True
        mask[15:29, 29:41] = True
        mask[19:22, 18:21] = False
        mask[3, 3] = True

        repaired = repair_brush_persistence_mask(
            mask, kernel_size=9, minimum_component_area=16
        )

        self.assertTrue(repaired[20, 26])
        self.assertTrue(repaired[20, 19])
        self.assertFalse(repaired[3, 3])

    def test_rejects_even_kernel(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd"):
            repair_brush_persistence_mask(np.ones((4, 4), dtype=bool), kernel_size=4)


if __name__ == "__main__":
    unittest.main()
