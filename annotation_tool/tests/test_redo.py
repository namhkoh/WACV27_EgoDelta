from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from egomask.redo import finalize_flagged_corrections, prepare_flagged_corrections
from egomask.review import FinalExportReview
from egomask.storage import AnnotationStore


class FlaggedCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.final = root / "final"
        self.flags = root / "flags.json"
        self.corrections = root / "corrections"
        self.output = root / "corrected"
        (self.source / "initial_masks" / "scene").mkdir(parents=True)
        (self.final / "binary_masks" / "scene").mkdir(parents=True)
        (self.final / "label_maps" / "scene").mkdir(parents=True)
        self.image = root / "frame.jpg"
        Image.new("RGB", (12, 10), "green").save(self.image)
        projected = Image.new("L", (12, 10), 0)
        projected.putpixel((2, 2), 1)
        projected.save(self.source / "initial_masks" / "scene" / "frame.png")
        final_label = Image.new("L", (12, 10), 0)
        final_label.putpixel((5, 5), 1)
        final_label.save(self.final / "label_maps" / "scene" / "frame.png")
        final_binary = final_label.point(lambda value: 255 if value else 0, mode="L")
        final_binary.save(self.final / "binary_masks" / "scene" / "frame__component_000.png")
        task = {
            "task_id": "scene__frame",
            "scene_id": "scene",
            "scene_uuid": "uuid",
            "task_kind": "deleafing",
            "image_name": "frame.jpg",
            "source_image": str(self.image),
            "initial_mask": "initial_masks/scene/frame.png",
            "assigned_to": "annotator_1",
            "camera": {"width": 12, "height": 10},
            "projection": {},
            "components": [
                {
                    "component_id": 0,
                    "label": 1,
                    "color": "#20d6c7",
                    "adjacent_gt_indices": [0],
                    "primitive_count": 4,
                    "source_ply": "/tmp/component.ply",
                }
            ],
        }
        source_manifest = {
            "schema_version": 1,
            "annotators": [
                {"id": "annotator_1", "token": "secret", "assigned_tasks": 1}
            ],
            "scenes": [],
            "tasks": [task],
        }
        (self.source / "manifest.json").write_text(
            json.dumps(source_manifest), encoding="utf-8"
        )
        final_task = {
            **{key: task[key] for key in ("task_id", "scene_id", "scene_uuid", "task_kind", "image_name", "source_image", "camera")},
            "annotator": "annotator_1",
            "mask_source": "human_polygon_outline_fill",
            "label_map": "label_maps/scene/frame.png",
            "binary_masks": ["binary_masks/scene/frame__component_000.png"],
            "mask_sha256": "source-label-sha",
        }
        (self.final / "dataset.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "component_policy": "single_foreground_union",
                    "solid_binary_masks": True,
                    "tasks": [final_task],
                }
            ),
            encoding="utf-8",
        )
        FinalExportReview(self.final, flags_path=self.flags).set_flag(
            "scene__frame", "annotator_1", True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_and_finalize_preserve_reference_and_apply_human_edit(self) -> None:
        manifest = prepare_flagged_corrections(
            self.source,
            self.final,
            self.flags,
            self.corrections,
            reviewer="annotator_1",
        )
        self.assertEqual(len(manifest["tasks"]), 1)
        task = manifest["tasks"][0]
        with Image.open(self.corrections / task["initial_mask"]) as current:
            self.assertEqual(current.getpixel((5, 5)), 1)
            self.assertEqual(current.getpixel((2, 2)), 0)
        with Image.open(self.corrections / task["reference_mask"]) as reference:
            self.assertEqual(reference.getpixel((2, 2)), 1)

        premature = self.output.parent / "premature"
        with self.assertRaisesRegex(ValueError, "Correction is incomplete"):
            finalize_flagged_corrections(self.corrections, self.final, premature)
        self.assertFalse(premature.exists())

        corrected = Image.new("L", (12, 10), 0)
        corrected.putpixel((8, 7), 1)
        buffer = BytesIO()
        corrected.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        store = AnnotationStore(self.corrections)
        store.save(
            "annotator_1",
            {
                "task_id": "scene__frame",
                "status": "complete",
                "revision": 0,
                "note": "fixed boundary",
                "mask_data_url": data_url,
            },
        )

        result = finalize_flagged_corrections(
            self.corrections, self.final, self.output
        )
        self.assertEqual(result["corrected_task_count"], 1)
        with Image.open(self.output / "label_maps" / "scene" / "frame.png") as label:
            self.assertEqual(label.getpixel((8, 7)), 1)
            self.assertEqual(label.getpixel((5, 5)), 0)
        with Image.open(
            self.output / "binary_masks" / "scene" / "frame__component_000.png"
        ) as binary:
            self.assertEqual(binary.getpixel((8, 7)), 255)
        self.assertEqual(result["tasks"][0]["mask_source"], "human_flagged_redo")


if __name__ == "__main__":
    unittest.main()
