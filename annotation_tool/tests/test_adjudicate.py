from __future__ import annotations

import unittest

from egomask.adjudicate import assign_reviewers


class HumanAdjudicationTests(unittest.TestCase):
    def test_assignments_are_independent_and_balanced(self) -> None:
        tasks = [
            {
                "task_id": f"scene__frame_{index:03d}",
                "scene_id": "scene",
                "image_name": f"frame_{index:03d}.jpg",
                "assigned_to": f"annotator_{index % 2 + 1}",
            }
            for index in range(17)
        ]
        reviewers = ("annotator_1", "annotator_2", "annotator_3", "annotator_4")
        assignments = assign_reviewers(tasks, reviewers)
        self.assertEqual(set(assignments), {task["task_id"] for task in tasks})
        for task in tasks:
            self.assertNotEqual(assignments[task["task_id"]], task["assigned_to"])
        counts = [list(assignments.values()).count(reviewer) for reviewer in reviewers]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_requires_an_independent_reviewer(self) -> None:
        task = {
            "task_id": "scene__frame",
            "scene_id": "scene",
            "image_name": "frame.jpg",
            "assigned_to": "annotator_1",
        }
        with self.assertRaisesRegex(ValueError, "No independent reviewer"):
            assign_reviewers([task], ("annotator_1",))


if __name__ == "__main__":
    unittest.main()
