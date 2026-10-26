from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image

from .review import FinalExportReview
from .storage import AnnotationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and finalize human corrections for flagged final masks."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Create an editing run from every globally flagged final mask."
    )
    prepare.add_argument("--source-run", type=Path, required=True)
    prepare.add_argument("--final-export", type=Path, required=True)
    prepare.add_argument("--flags", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--reviewer", default="annotator_1")

    finalize = commands.add_parser(
        "finalize", help="Apply completed human corrections to a new frozen export."
    )
    finalize.add_argument("--correction-run", type=Path, required=True)
    finalize.add_argument("--final-export", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_flagged_corrections(
                args.source_run,
                args.final_export,
                args.flags,
                args.output,
                reviewer=args.reviewer,
            )
            print(f"Prepared {len(result['tasks'])} flagged corrections in {args.output.resolve()}")
        else:
            result = finalize_flagged_corrections(
                args.correction_run, args.final_export, args.output
            )
            print(
                f"Applied {result['corrected_task_count']} human corrections to "
                f"{args.output.resolve()}"
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def prepare_flagged_corrections(
    source_run: str | Path,
    final_export: str | Path,
    flags_path: str | Path,
    output: str | Path,
    *,
    reviewer: str = "annotator_1",
) -> dict:
    source = AnnotationStore(source_run)
    flags_path = Path(flags_path).expanduser().resolve()
    final = FinalExportReview(final_export, flags_path=flags_path)
    if reviewer not in source.annotators:
        raise ValueError(f"Unknown reviewer: {reviewer}")
    selected_ids = [
        task["task_id"]
        for task in final.dataset["tasks"]
        if final.flag_state(task["task_id"])["needs_redo"]
    ]
    if not selected_ids:
        raise ValueError("The final-review queue contains no flagged masks")
    output = _require_empty_output(output)
    source_tasks = {task["task_id"]: task for task in source.manifest["tasks"]}
    unknown = set(selected_ids).difference(source_tasks)
    if unknown:
        raise ValueError(f"Flagged tasks are missing from the source run: {sorted(unknown)[:5]}")

    prepared_tasks: list[dict] = []
    for task_id in selected_ids:
        task = source_tasks[task_id]
        final_task = final.tasks[task_id]
        expected_size = (int(task["camera"]["width"]), int(task["camera"]["height"]))
        final_mask_path = final.mask_path(task_id)
        with Image.open(final_mask_path) as loaded:
            final_mask = loaded.convert("L")
            final_mask.load()
        if final_mask.size != expected_size:
            raise ValueError(f"Wrong final-mask dimensions for {task_id}: {final_mask.size}")
        final_mask = final_mask.point(lambda value: 1 if value else 0, mode="L")
        relative_initial = (
            Path("initial_masks") / task["scene_id"] / f"{Path(task['image_name']).stem}.png"
        )
        _save_png_atomic(output / relative_initial, final_mask)

        original_annotator = task["assigned_to"]
        reference_path = source.initial_mask_path(original_annotator, task_id)
        with Image.open(reference_path) as loaded:
            reference = loaded.convert("L")
            reference.load()
        if reference.size != expected_size:
            raise ValueError(f"Wrong original-projection dimensions for {task_id}: {reference.size}")
        relative_reference = (
            Path("reference_masks") / task["scene_id"] / f"{Path(task['image_name']).stem}.png"
        )
        _save_png_atomic(output / relative_reference, reference)

        source_components = task["components"]
        state = final.flag_state(task_id)
        prepared_tasks.append(
            {
                **task,
                "assigned_to": reviewer,
                "initial_mask": str(relative_initial),
                "reference_mask": str(relative_reference),
                "source_components": source_components,
                "components": [_union_component(source_components)],
                "correction": {
                    "workflow": "flagged_final_mask_redo",
                    "flagged_by": state["flagged_by"],
                    "flagged_updated_utc": state["flagged_updated_utc"],
                    "original_annotator": original_annotator,
                    "source_final_mask_sha256": _sha256(final_mask_path),
                    "source_final_label_map_sha256": final_task.get("mask_sha256"),
                    "original_projected_mask_sha256": _sha256(reference_path),
                    "instruction": (
                        "Correct the current cleaned final mask as one foreground component. "
                        "Use the magenta outline and standalone preview only to compare against "
                        "the untouched original 3D-component projection. Save only the visible "
                        "human-corrected RGB silhouette."
                    ),
                },
            }
        )

    annotators = []
    for row in source.manifest["annotators"]:
        annotators.append(
            {
                "id": row["id"],
                "token": row["token"],
                "assigned_tasks": len(prepared_tasks) if row["id"] == reviewer else 0,
            }
        )
    selected_scenes = {task["scene_id"] for task in prepared_tasks}
    manifest = {
        "schema_version": 1,
        "tool": "egomask",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "flagged_final_mask_correction",
        "human_only": True,
        "component_policy": "single_foreground_union",
        "reviewer": reviewer,
        "source_run": str(source.run_root),
        "source_manifest_sha256": _sha256(source.run_root / "manifest.json"),
        "source_final_export": str(final.export_root),
        "source_final_dataset_sha256": final.dataset_sha256,
        "review_flags": str(flags_path),
        "review_flags_sha256": _sha256(flags_path),
        "flagged_task_count": len(prepared_tasks),
        "review_policy": (
            "Every task was explicitly flagged against the frozen human-mask export. "
            "The editable initialization is that final mask; the original 3D projection is "
            "a non-editable visual reference."
        ),
        "annotators": annotators,
        "scenes": [
            scene
            for scene in source.manifest.get("scenes", [])
            if scene["scene_id"] in selected_scenes
        ],
        "tasks": prepared_tasks,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    _write_access_links(output, annotators)
    return manifest


def finalize_flagged_corrections(
    correction_run: str | Path,
    final_export: str | Path,
    output: str | Path,
) -> dict:
    corrections = AnnotationStore(correction_run)
    if corrections.manifest.get("workflow") != "flagged_final_mask_correction":
        raise ValueError("Run is not a flagged final-mask correction assignment")
    if not corrections.manifest.get("human_only"):
        raise ValueError("Correction run is not marked human-only")
    final = FinalExportReview(final_export)
    brush_repair = corrections.manifest.get("brush_persistence_repair")
    if corrections.manifest.get("source_final_dataset_sha256") != final.dataset_sha256:
        raise ValueError("Frozen final export changed after the correction run was prepared")
    validated: list[tuple[dict, str, dict, Image.Image]] = []
    for task in corrections.manifest["tasks"]:
        task_id = task["task_id"]
        reviewer = task["assigned_to"]
        record = corrections.load_record(reviewer, task_id)
        if record["status"] != "complete":
            raise ValueError(f"Correction is incomplete for {task_id}: {record['status']}")
        corrected_path = corrections.annotation_mask_path(reviewer, task_id)
        if not corrected_path.is_file() or _sha256(corrected_path) != record.get("mask_sha256"):
            raise ValueError(f"Corrected mask is missing or corrupt: {task_id}")
        with Image.open(corrected_path) as loaded:
            corrected = loaded.convert("L")
            corrected.load()
        expected_size = (int(task["camera"]["width"]), int(task["camera"]["height"]))
        if corrected.size != expected_size:
            raise ValueError(f"Wrong corrected-mask dimensions for {task_id}: {corrected.size}")
        if corrected.getextrema()[1] > 1:
            raise ValueError(f"Corrected mask is not a single-component label map: {task_id}")
        validated.append((task, reviewer, record, corrected))

    # Do not create a partial frozen export when even one correction is still
    # incomplete. All human records and masks are validated before copying.
    output = _require_empty_output(output)
    shutil.copytree(final.export_root, output, dirs_exist_ok=True)
    dataset_path = output / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    output_tasks = {task["task_id"]: task for task in dataset["tasks"]}

    for task, reviewer, record, corrected in validated:
        task_id = task["task_id"]
        output_task = output_tasks[task_id]
        label_path = output / output_task["label_map"]
        binary_paths = output_task.get("binary_masks") or []
        if len(binary_paths) != 1:
            raise ValueError(f"Expected one binary mask for {task_id}")
        binary_path = output / binary_paths[0]
        _save_png_atomic(label_path, corrected)
        binary = corrected.point(lambda value: 255 if value else 0, mode="L")
        _save_png_atomic(binary_path, binary)
        previous_sha256 = output_task.get("mask_sha256")
        output_task.update(
            {
                "annotator": reviewer,
                "annotation_revision": int(record["revision"]),
                "annotation_updated_utc": record["updated_utc"],
                "annotation_note": record["note"],
                "mask_source": (
                    "human_flagged_redo_brush_reconstructed"
                    if record.get("persistence_repair")
                    else "human_flagged_redo"
                ),
                "mask_sha256": _sha256(label_path),
                "correction": {
                    "workflow": "flagged_final_mask_redo",
                    "human_reviewer": reviewer,
                    "human_revision": int(record["revision"]),
                    "human_updated_utc": record["updated_utc"],
                    "human_note": record["note"],
                    "previous_final_mask_sha256": previous_sha256,
                    "original_projected_mask_sha256": task["correction"][
                        "original_projected_mask_sha256"
                    ],
                    **(
                        {"brush_persistence_repair": record["persistence_repair"]}
                        if record.get("persistence_repair")
                        else {}
                    ),
                },
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    dataset.update(
        {
            "created_utc": now,
            "source_final_export": str(final.export_root),
            "source_final_dataset_sha256": final.dataset_sha256,
            "correction_run": str(corrections.run_root),
            "correction_manifest_sha256": _sha256(corrections.run_root / "manifest.json"),
            "corrected_task_count": len(corrections.manifest["tasks"]),
            "human_only": True,
            **({"brush_persistence_repair": brush_repair} if brush_repair else {}),
        }
    )
    _write_json_atomic(dataset_path, dataset)
    repair_report = (
        f"\n\nBrush persistence repairs: {brush_repair['affected_task_count']}. "
        "These masks retain the human-edited pixels and deterministically reconnect "
        "stroke samples affected by the browser low-label rasterization bug; no model "
        "generated or replaced the semantic annotations."
        if brush_repair
        else ""
    )
    report = (
        f"# Human flagged-mask correction report\n\n"
        f"Created: {now}\n\n"
        f"Corrected masks: {len(corrections.manifest['tasks'])} / {len(dataset['tasks'])}\n\n"
        "Every replacement was completed in EgoMask by the assigned human reviewer. "
        "The original 3D-projected mask was displayed only as a non-editable reference."
        f"{repair_report}\n"
    )
    (output / "CORRECTION_REPORT.md").write_text(report, encoding="utf-8")
    _write_checksums(output)
    return dataset


def _union_component(components: list[dict]) -> dict:
    return {
        "component_id": 0,
        "label": 1,
        "color": "#20d6c7",
        "adjacent_gt_indices": sorted(
            {
                int(index)
                for component in components
                for index in component["adjacent_gt_indices"]
            }
        ),
        "primitive_count": sum(int(component["primitive_count"]) for component in components),
        "source_ply": components[0]["source_ply"],
        "role": "human_visible_change_union",
    }


def _require_empty_output(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_png_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_access_links(output: Path, annotators: list[dict]) -> None:
    path = output / "ACCESS_LINKS.txt"
    lines = [
        "# Private correction/review access paths",
        "# Append &review=final to open the global final-mask browser.",
        "",
        *(f"{row['id']}: /?token={row['token']}" for row in annotators),
    ]
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(output)}" for path in paths) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
