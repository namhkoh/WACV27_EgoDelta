from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from egomask.export import export_run, solidify_binary_mask
from egomask.storage import AnnotationStore, RevisionConflict


class StorageExportTests(unittest.TestCase):
    def test_solid_binary_mask_fills_holes_and_removes_tool_specks(self) -> None:
        mask = np.zeros((40, 50), dtype=bool)
        mask[8:32, 10:40] = True
        mask[15:23, 20:28] = False
        mask[2, 2] = True

        cleaned = solidify_binary_mask(mask)

        self.assertTrue(cleaned[18, 24])
        self.assertFalse(cleaned[2, 2])
        self.assertTrue(cleaned[10, 12])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "run"
        self.root.mkdir()
        source = Path(self.temporary.name) / "frame.jpg"
        Image.new("RGB", (12, 10), (20, 40, 60)).save(source)
        initial = self.root / "initial_masks" / "scene" / "frame.png"
        initial.parent.mkdir(parents=True)
        Image.new("L", (12, 10), 0).save(initial)
        manifest = {
            "schema_version": 1,
            "oracle_root": "/oracle",
            "oracle_manifest_sha256": "abc",
            "annotators": [{"id": "ann", "token": "secret", "assigned_tasks": 1}],
            "tasks": [
                {
                    "task_id": "scene__frame",
                    "assigned_to": "ann",
                    "scene_id": "scene",
                    "scene_uuid": "uuid",
                    "task_kind": "deleafing",
                    "image_name": "frame.jpg",
                    "source_image": str(source),
                    "initial_mask": str(initial.relative_to(self.root)),
                    "camera": {"width": 12, "height": 10},
                    "components": [
                        {
                            "component_id": 0,
                            "label": 1,
                            "color": "#20d6c7",
                            "adjacent_gt_indices": [0],
                            "primitive_count": 4,
                            "source_ply": "/component.ply",
                        },
                        {
                            "component_id": 1,
                            "label": 2,
                            "color": "#ff9f43",
                            "adjacent_gt_indices": [1],
                            "primitive_count": 3,
                            "source_ply": "/component_2.ply",
                        },
                    ],
                    "projection": {"visible_pixels": 0},
                }
            ],
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_save_is_revision_checked_and_exported(self) -> None:
        store = AnnotationStore(self.root)
        self.assertEqual(store.authenticate("secret"), "ann")
        self.assertEqual(store.mask_path("ann", "scene__frame"), store.initial_mask_path("ann", "scene__frame"))
        mask = Image.new("RGB", (12, 10), (0, 0, 0))
        mask.putpixel((4, 5), (1, 1, 1))
        mask.putpixel((5, 5), (2, 2, 2))
        buffer = BytesIO()
        mask.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "task_id": "scene__frame",
            "status": "complete",
            "revision": 0,
            "note": "checked",
            "mask_data_url": data_url,
        }
        record = store.save("ann", payload)
        self.assertEqual(record["revision"], 1)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(store.mask_path("ann", "scene__frame"), store.annotation_mask_path("ann", "scene__frame"))
        self.assertNotEqual(store.mask_path("ann", "scene__frame"), store.initial_mask_path("ann", "scene__frame"))
        with self.assertRaises(RevisionConflict):
            store.save("ann", payload)

        output = Path(self.temporary.name) / "export"
        result = export_run(self.root, output, binary_components=True, annotators=["ann"])
        self.assertEqual(result["exported_task_count"], 1)
        self.assertEqual(result["selected_annotators"], ["ann"])
        with Image.open(output / "label_maps" / "scene" / "frame.png") as exported:
            self.assertEqual(exported.mode, "L")
            self.assertEqual(exported.getpixel((4, 5)), 1)
        with Image.open(output / "binary_masks" / "scene" / "frame__component_000.png") as binary:
            self.assertEqual(binary.getpixel((4, 5)), 255)

        union_output = Path(self.temporary.name) / "union_export"
        union_result = export_run(
            self.root,
            union_output,
            binary_components=True,
            single_component=True,
            annotators=["ann"],
        )
        self.assertEqual(union_result["component_policy"], "single_foreground_union")
        self.assertEqual(len(union_result["tasks"][0]["components"]), 1)
        self.assertEqual(len(union_result["tasks"][0]["source_components"]), 2)
        with Image.open(union_output / "label_maps" / "scene" / "frame.png") as exported:
            self.assertEqual(exported.getpixel((4, 5)), 1)
            self.assertEqual(exported.getpixel((5, 5)), 1)
        with Image.open(
            union_output / "binary_masks" / "scene" / "frame__component_000.png"
        ) as binary:
            self.assertEqual(binary.getpixel((4, 5)), 255)
            self.assertEqual(binary.getpixel((5, 5)), 255)

        override_root = Path(self.temporary.name) / "overrides"
        override_mask = override_root / "masks" / "scene__frame.png"
        override_mask.parent.mkdir(parents=True)
        replacement = Image.new("L", (12, 10), 0)
        replacement.putpixel((7, 6), 255)
        replacement.save(override_mask)
        (override_root / "overrides.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": "unit_test_mask_replacement",
                    "overrides": [
                        {
                            "task_id": "scene__frame",
                            "mask": "masks/scene__frame.png",
                            "reason": "test replacement",
                            "source_annotation_sha256": hashlib.sha256(
                                store.annotation_mask_path("ann", "scene__frame").read_bytes()
                            ).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        override_output = Path(self.temporary.name) / "override_export"
        override_result = export_run(
            self.root,
            override_output,
            single_component=True,
            mask_overrides=override_root,
        )
        self.assertEqual(override_result["mask_overrides"]["applied_task_count"], 1)
        self.assertEqual(
            override_result["tasks"][0]["mask_source"],
            "unit_test_mask_replacement",
        )
        self.assertIn("mask_override", override_result["tasks"][0])
        self.assertIn("source_annotation_sha256", override_result["tasks"][0])
        with Image.open(override_output / "label_maps" / "scene" / "frame.png") as exported:
            self.assertEqual(exported.getpixel((4, 5)), 0)
            self.assertEqual(exported.getpixel((7, 6)), 1)

        with self.assertRaisesRegex(ValueError, "Unknown annotators"):
            export_run(self.root, Path(self.temporary.name) / "invalid", annotators=["missing"])


if __name__ == "__main__":
    unittest.main()
