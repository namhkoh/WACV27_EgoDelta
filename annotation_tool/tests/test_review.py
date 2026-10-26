from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from egomask.review import FinalExportReview


class FinalExportReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "export"
        self.root.mkdir()
        self.flags_path = Path(self.temporary.name) / "review" / "flags.json"
        self.image = Path(self.temporary.name) / "frame.jpg"
        Image.new("RGB", (12, 10), "green").save(self.image)
        mask = self.root / "binary_masks" / "scene" / "frame__component_000.png"
        mask.parent.mkdir(parents=True)
        Image.new("L", (12, 10), 255).save(mask)
        self.dataset = {
            "schema_version": 1,
            "created_utc": "2026-08-17T00:00:00+00:00",
            "component_policy": "single_foreground_union",
            "solid_binary_masks": True,
            "tasks": [
                {
                    "task_id": "scene__frame",
                    "scene_id": "scene",
                    "scene_uuid": "uuid",
                    "task_kind": "deleafing",
                    "image_name": "frame.jpg",
                    "annotator": "annotator_4",
                    "mask_source": "source_annotation",
                    "camera": {"width": 12, "height": 10},
                    "source_image": str(self.image),
                    "binary_masks": ["binary_masks/scene/frame__component_000.png"],
                }
            ],
        }
        self._write_dataset()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_dataset(self) -> None:
        (self.root / "dataset.json").write_text(
            json.dumps(self.dataset), encoding="utf-8"
        )

    def test_exposes_every_exported_task_read_only(self) -> None:
        review = FinalExportReview(self.root)

        payload = review.bootstrap()
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["tasks"][0]["annotator"], "annotator_4")
        self.assertEqual(review.image_path("scene__frame"), self.image.resolve())
        self.assertTrue(review.mask_path("scene__frame").is_file())

    def test_rejects_mask_path_outside_export(self) -> None:
        outside = Path(self.temporary.name) / "outside.png"
        Image.new("L", (12, 10), 255).save(outside)
        self.dataset["tasks"][0]["binary_masks"] = ["../outside.png"]
        self._write_dataset()

        with self.assertRaisesRegex(ValueError, "escapes export root"):
            FinalExportReview(self.root)

    def test_persists_independent_reviewer_flags(self) -> None:
        review = FinalExportReview(self.root, flags_path=self.flags_path)

        first = review.set_flag("scene__frame", "annotator_1", True)
        second = review.set_flag("scene__frame", "annotator_3", True)
        self.assertTrue(first["flagged"])
        self.assertEqual(second["flagged_by"], ["annotator_1", "annotator_3"])
        self.assertEqual(second["flagged_count"], 1)

        reloaded = FinalExportReview(self.root, flags_path=self.flags_path)
        payload = reloaded.bootstrap("annotator_1")
        self.assertEqual(payload["flagged_count"], 1)
        self.assertEqual(payload["reviewer_flagged_count"], 1)
        self.assertTrue(payload["tasks"][0]["flagged"])
        self.assertTrue(payload["tasks"][0]["needs_redo"])

        remaining = reloaded.set_flag("scene__frame", "annotator_1", False)
        self.assertFalse(remaining["flagged"])
        self.assertTrue(remaining["needs_redo"])
        self.assertEqual(remaining["flagged_by"], ["annotator_3"])


if __name__ == "__main__":
    unittest.main()
