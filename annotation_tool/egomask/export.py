from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

from .storage import AnnotationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and export completed EgoMask labels")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--annotator",
        action="append",
        dest="annotators",
        help="Export only this annotator's assignment; repeat to select several annotators.",
    )
    parser.add_argument(
        "--binary-components",
        action="store_true",
        help="Also emit one binary PNG per component instance",
    )
    parser.add_argument(
        "--single-component",
        action="store_true",
        help="Export every nonzero source label as one foreground component (label 1).",
    )
    parser.add_argument(
        "--solid-binary-masks",
        action="store_true",
        help=(
            "With --single-component, fill internal holes and remove tiny isolated "
            "tool-mark fragments from the exported foreground mask."
        ),
    )
    parser.add_argument(
        "--mask-overrides",
        type=Path,
        help=(
            "Optional non-destructive mask-replacement directory containing overrides.json; "
            "listed masks replace source annotations only in the export."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_run(
            args.run,
            args.output,
            allow_incomplete=args.allow_incomplete,
            binary_components=args.binary_components,
            single_component=args.single_component,
            solid_binary_masks=args.solid_binary_masks,
            annotators=args.annotators,
            mask_overrides=args.mask_overrides,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {result['exported_task_count']} masks to {args.output.resolve()}")
    if result["skipped_task_count"]:
        print(f"Skipped {result['skipped_task_count']} incomplete or flagged tasks")
    return 0


def export_run(
    run_root: str | Path,
    output: str | Path,
    *,
    allow_incomplete: bool = False,
    binary_components: bool = False,
    single_component: bool = False,
    solid_binary_masks: bool = False,
    annotators: list[str] | tuple[str, ...] | set[str] | None = None,
    mask_overrides: str | Path | None = None,
) -> dict:
    store = AnnotationStore(run_root)
    if solid_binary_masks and not single_component:
        raise ValueError("--solid-binary-masks requires --single-component")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected_annotators = set(annotators or ())
    known_annotators = {row["id"] for row in store.manifest["annotators"]}
    unknown_annotators = selected_annotators.difference(known_annotators)
    if unknown_annotators:
        raise ValueError(f"Unknown annotators: {sorted(unknown_annotators)}")
    selected_tasks = [
        task
        for task in store.manifest["tasks"]
        if not selected_annotators or task["assigned_to"] in selected_annotators
    ]
    overrides, override_metadata = _load_mask_overrides(mask_overrides, store.manifest["tasks"])
    complete_count = sum(
        store.load_record(task["assigned_to"], task["task_id"])["status"] == "complete"
        for task in selected_tasks
    )
    remaining = len(selected_tasks) - complete_count
    if remaining and not allow_incomplete:
        raise ValueError(
            f"Run is not export-ready: {remaining} tasks are not complete "
            "in the selected assignment. Resolve them or pass --allow-incomplete."
        )

    exported: list[dict] = []
    skipped: list[dict] = []
    csv_rows: list[dict] = []
    for task in selected_tasks:
        annotator = task["assigned_to"]
        record = store.load_record(annotator, task["task_id"])
        if record["status"] != "complete":
            skipped.append(
                {
                    "task_id": task["task_id"],
                    "assigned_to": annotator,
                    "status": record["status"],
                }
            )
            continue

        source_annotation = store.annotation_mask_path(annotator, task["task_id"])
        if not source_annotation.is_file():
            raise FileNotFoundError(f"Completed task has no saved mask: {task['task_id']}")
        override = overrides.get(task["task_id"])
        source_annotation_sha256 = _sha256(source_annotation)
        if override and override["source_annotation_sha256"] != source_annotation_sha256:
            raise ValueError(
                f"Stale override for {task['task_id']}: source annotation checksum changed"
            )
        source_mask = override["absolute_mask"] if override else source_annotation
        with Image.open(source_mask) as loaded:
            mask = loaded.convert("L")
            mask.load()
        if mask.size != (task["camera"]["width"], task["camera"]["height"]):
            raise ValueError(f"Wrong mask dimensions for {task['task_id']}: {mask.size}")
        _, maximum = mask.getextrema()
        if not single_component and maximum > len(task["components"]):
            raise ValueError(f"Invalid label {maximum} in {task['task_id']}")
        source_components = task["components"]
        if single_component:
            binary = np.asarray(mask, dtype=np.uint8) > 0
            if solid_binary_masks:
                binary = solidify_binary_mask(
                    binary,
                    # Leaf annotations contain narrow brush/eraser cuts from
                    # component editing. Tomato instances need the gentler
                    # setting so adjacent fruits are not bridged.
                    closing_size=15 if task["task_kind"] == "deleafing" else 3,
                )
            mask = Image.fromarray(binary.astype(np.uint8), mode="L")
            export_components = [
                {
                    "component_id": 0,
                    "label": 1,
                    "role": "union_of_visible_changed_components",
                    "source_component_ids": [
                        int(component["component_id"]) for component in source_components
                    ],
                }
            ]
        else:
            export_components = source_components

        relative_mask = Path("label_maps") / task["scene_id"] / f"{Path(task['image_name']).stem}.png"
        destination = output / relative_mask
        _save_image_atomic(destination, mask)

        binary_paths = []
        if binary_components:
            for component in export_components:
                label = int(component["label"])
                binary = mask.point(lambda value, wanted=label: 255 if value == wanted else 0, mode="L")
                relative_binary = (
                    Path("binary_masks")
                    / task["scene_id"]
                    / f"{Path(task['image_name']).stem}__component_{component['component_id']:03d}.png"
                )
                _save_image_atomic(output / relative_binary, binary)
                binary_paths.append(str(relative_binary))

        entry = {
            "task_id": task["task_id"],
            "scene_id": task["scene_id"],
            "scene_uuid": task["scene_uuid"],
            "task_kind": task["task_kind"],
            "image_name": task["image_name"],
            "source_image": task["source_image"],
            "label_map": str(relative_mask),
            "binary_masks": binary_paths,
            "camera": task["camera"],
            "components": export_components,
            **({"source_components": source_components} if single_component else {}),
            "annotator": annotator,
            "annotation_revision": record["revision"],
            "annotation_updated_utc": record["updated_utc"],
            "annotation_note": record["note"],
            "mask_source": (
                str(override_metadata.get("method") or "mask_override")
                if override
                else "source_annotation"
            ),
            "source_annotation_sha256": source_annotation_sha256,
            "mask_sha256": _sha256(destination),
        }
        if override:
            entry["mask_override"] = {
                key: value for key, value in override.items() if key != "absolute_mask"
            }
        exported.append(entry)
        csv_rows.append(
            {
                "task_id": task["task_id"],
                "scene_id": task["scene_id"],
                "task_kind": task["task_kind"],
                "image_name": task["image_name"],
                "label_map": str(relative_mask),
                "component_count": len(export_components),
                "annotator": annotator,
                "revision": record["revision"],
                "note": record["note"],
            }
        )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(store.run_root),
        "source_oracle_root": store.manifest["oracle_root"],
        "source_oracle_manifest_sha256": store.manifest["oracle_manifest_sha256"],
        "selected_annotators": sorted(selected_annotators) if selected_annotators else [],
        "mask_encoding": (
            "8-bit PNG; 0=background, 1=union of all visible changed components"
            if single_component
            else "8-bit PNG; 0=background, component_id+1=instance label"
        ),
        "component_policy": "single_foreground_union" if single_component else "source_instances",
        "solid_binary_masks": solid_binary_masks,
        "mask_overrides": (
            {
                **override_metadata,
                "applied_task_count": sum(task["task_id"] in overrides for task in selected_tasks),
            }
            if override_metadata
            else None
        ),
        "exported_task_count": len(exported),
        "skipped_task_count": len(skipped),
        "tasks": exported,
        "skipped_tasks": skipped,
    }
    _write_json_atomic(output / "dataset.json", payload)
    _write_csv(output / "annotations.csv", csv_rows)
    _write_checksums(output)
    return payload


def solidify_binary_mask(
    mask: np.ndarray,
    *,
    minimum_component_area: int = 16,
    closing_size: int = 3,
) -> np.ndarray:
    """Remove tiny tool marks and fill holes without joining separate large regions."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary mask must be a 2D array")
    if minimum_component_area < 1:
        raise ValueError("minimum_component_area must be positive")
    if closing_size < 1 or closing_size % 2 == 0:
        raise ValueError("closing_size must be a positive odd integer")

    def remove_small_components(source: np.ndarray) -> np.ndarray:
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            source.astype(np.uint8), connectivity=8
        )
        result = np.zeros_like(source, dtype=bool)
        for component in range(1, component_count):
            if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_component_area:
                result[components == component] = True
        return result

    cleaned = remove_small_components(binary)

    if closing_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        )
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        # Closing can erode a marginal retained fragment below the threshold.
        cleaned = remove_small_components(cleaned)

    height, width = cleaned.shape
    background_count, background, background_stats, _ = cv2.connectedComponentsWithStats(
        (~cleaned).astype(np.uint8), connectivity=8
    )
    for component in range(1, background_count):
        x, y, component_width, component_height, _ = background_stats[component]
        touches_image_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_image_border:
            cleaned[background == component] = True
    return cleaned


def _load_mask_overrides(
    root: str | Path | None,
    manifest_tasks: list[dict],
) -> tuple[dict[str, dict], dict | None]:
    if root is None:
        return {}, None
    root = Path(root).expanduser().resolve()
    manifest_path = root / "overrides.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Mask override manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported mask override schema in {manifest_path}")
    rows = payload.get("overrides")
    if not isinstance(rows, list):
        raise ValueError(f"Mask override manifest has no overrides list: {manifest_path}")

    known_tasks = {task["task_id"] for task in manifest_tasks}
    overrides: dict[str, dict] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in known_tasks:
            raise ValueError(f"Unknown override task_id: {task_id!r}")
        if task_id in overrides:
            raise ValueError(f"Duplicate override task_id: {task_id}")
        relative_mask = Path(str(row.get("mask", "")))
        if not relative_mask.as_posix() or relative_mask.is_absolute():
            raise ValueError(f"Override mask must be a relative path for {task_id}")
        absolute_mask = (root / relative_mask).resolve()
        try:
            absolute_mask.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Override mask escapes its root for {task_id}: {relative_mask}") from exc
        if not absolute_mask.is_file():
            raise FileNotFoundError(f"Missing override mask for {task_id}: {absolute_mask}")
        declared_mask_sha256 = str(row.get("mask_sha256", "")).lower()
        if declared_mask_sha256 and declared_mask_sha256 != _sha256(absolute_mask):
            raise ValueError(f"Override mask checksum mismatch for {task_id}")
        source_annotation_sha256 = str(row.get("source_annotation_sha256", "")).lower()
        if len(source_annotation_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_annotation_sha256
        ):
            raise ValueError(f"Override has no valid source annotation checksum for {task_id}")
        row = {**row, "source_annotation_sha256": source_annotation_sha256}
        overrides[task_id] = {**row, "mask": str(relative_mask), "absolute_mask": absolute_mask}

    return overrides, {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "declared_task_count": len(overrides),
        "method": payload.get("method"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "task_id",
        "scene_id",
        "task_kind",
        "image_name",
        "label_map",
        "component_count",
        "annotator",
        "revision",
        "note",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path != checksum_path)
    lines = [f"{_sha256(path)}  {path.relative_to(output)}" for path in paths]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_image_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
