from __future__ import annotations

import unittest

import numpy as np

from egomask.multiview_unify import choose_footprint_sigma


class MultiviewUnifyTests(unittest.TestCase):
    def test_selects_stable_sigma_and_reports_leave_one_out(self) -> None:
        scores = np.asarray(
            [
                [0.50, 0.80, 0.60],
                [0.55, 0.85, 0.58],
                [0.52, 0.82, 0.59],
            ]
        )
        selected, leave_one_out = choose_footprint_sigma(scores, (0.5, 0.65, 0.8))
        self.assertEqual(selected, 1)
        self.assertEqual(leave_one_out, [0.8, 0.85, 0.82])

    def test_requires_multiple_views(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two views"):
            choose_footprint_sigma(np.asarray([[0.4, 0.5]]), (0.5, 0.65))


if __name__ == "__main__":
    unittest.main()
