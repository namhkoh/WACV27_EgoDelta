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

import cv2
import numpy as np
from PIL import Image

from .storage import AnnotationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct continuous masks from human brush pixels affected by the "
            "low-label browser rasterization bug. The source run is never modified."
        )
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--minimum-component-area", type=int, default=32)
    parser.add_argument(
        "--confirm-complete",
        action="store_true",
        help="Restore copied records to complete after the human confirms the queue is finished.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = repair_correction_run(
            args.run,
            args.output,
            kernel_size=args.kernel_size,
            minimum_component_area=args.minimum_component_area,
            confirm_complete=args.confirm_complete,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Reconstructed {result['affected_task_count']} brush-affected masks; "
        f"preserved {result['preserved_task_count']} masks exactly in {Path(args.output).resolve()}"
    )
    return 0


def repair_correction_run(
    run_root: str | Path,
    output: str | Path,
    *,
    kernel_size: int = 9,
    minimum_component_area: int = 32,
    confirm_complete: bool = False,
) -> dict:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if minimum_component_area < 1:
        raise ValueError("minimum_component_area must be positive")
    source = AnnotationStore(run_root)
    if source.manifest.get("workflow") != "flagged_final_mask_correction":
        raise ValueError("Run is not a flagged final-mask correction assignment")
    if not source.manifest.get("human_only"):
        raise ValueError("Correction run is not marked human-only")

    validated: list[tuple[dict, dict, Path, np.ndarray, bool]] = []
    incomplete = []
    for task in source.manifest["tasks"]:
        task_id = task["task_id"]
        annotator = task["assigned_to"]
        record = source.load_record(annotator, task_id)
        if record["status"] != "complete":
            incomplete.append(task_id)
        mask_path = source.annotation_mask_path(annotator, task_id)
        if not mask_path.is_file() or _sha256(mask_path) != record.get("mask_sha256"):
            raise ValueError(f"Human correction is missing or corrupt: {task_id}")
        mask = _read_binary_mask(mask_path)
        expected_shape = (int(task["camera"]["height"]), int(task["camera"]["width"]))
        if mask.shape != expected_shape:
            raise ValueError(f"Wrong correction dimensions for {task_id}: {mask.shape}")
        if not mask.any():
            raise ValueError(f"Human correction is empty: {task_id}")
        affected = _has_small_components(mask, minimum_component_area)
        validated.append((task, record, mask_path, mask, affected))
    if incomplete and not confirm_complete:
        raise ValueError(
            f"Correction run contains {len(incomplete)} incomplete task(s); "
            "pass --confirm-complete only after the human reviewer confirms completion"
        )

    output = _require_empty_output(output)
    # Source and destination live on the same annotation volume. Hard-link the
    # immutable source assets so the new run is fast and byte-exact; every file
    # changed below is replaced atomically, which breaks its link without ever
    # modifying the original run.
    shutil.copytree(source.run_root, output, dirs_exist_ok=True, copy_function=os.link)
    now = datetime.now(timezone.utc).isoformat()
    repair_rows = []
    for task, source_record, source_mask_path, source_mask, affected in validated:
        if not affected:
            continue
        task_id = task["task_id"]
        annotator = task["assigned_to"]
        repaired = repair_brush_persistence_mask(
            source_mask,
            kernel_size=kernel_size,
            minimum_component_area=minimum_component_area,
        )
        if not repaired.any():
            raise ValueError(f"Brush reconstruction removed the full mask: {task_id}")
        destination = output / "annotations" / annotator / f"{task_id}.png"
        _save_binary_png_atomic(destination, repaired)
        record_path = output / "records" / annotator / f"{task_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source_digest = _sha256(source_mask_path)
        record.update(
            {
                "status": "complete",
                "mask_sha256": _sha256(destination),
                "nonzero_bbox": _bbox(repaired),
                "persistence_repair": {
                    "method": "human_brush_capsule_reconstruction_v1",
                    "reason": "browser_low_label_antialiasing_disconnected_saved_strokes",
                    "source_mask_sha256": source_digest,
                    "repaired_utc": now,
                    "kernel_size": kernel_size,
                    "minimum_component_area": minimum_component_area,
                },
            }
        )
        _write_json_atomic(record_path, record)
        repair_rows.append(
            {
                "task_id": task_id,
                "annotator": annotator,
                "source_mask_sha256": source_digest,
                "repaired_mask_sha256": _sha256(destination),
                "source_area": int(source_mask.sum()),
                "repaired_area": int(repaired.sum()),
                "iou": _iou(source_mask, repaired),
            }
        )

    # A copied record may be in_progress simply because reopening the final
    # task changed its UI state. User confirmation restores that copied record
    # without mutating the original human run.
    if confirm_complete:
        affected_ids = {row["task_id"] for row in repair_rows}
        for task, _, _, _, _ in validated:
            if task["task_id"] in affected_ids:
                continue
            record_path = output / "records" / task["assigned_to"] / f"{task['task_id']}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                record["status"] = "complete"
                record["completion_restored_utc"] = now
                _write_json_atomic(record_path, record)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = {
        "method": "human_brush_capsule_reconstruction_v1",
        "created_utc": now,
        "source_correction_run": str(source.run_root),
        "source_manifest_sha256": _sha256(source.run_root / "manifest.json"),
        "reason": "browser_low_label_antialiasing_disconnected_saved_strokes",
        "kernel_size": kernel_size,
        "minimum_component_area": minimum_component_area,
        "affected_task_count": len(repair_rows),
        "preserved_task_count": len(validated) - len(repair_rows),
        "human_completion_confirmed": bool(confirm_complete),
        "repairs": repair_rows,
    }
    manifest["brush_persistence_repair"] = metadata
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(output / "BRUSH_PERSISTENCE_REPAIR.json", metadata)

    # Re-open the copied run to verify every mask-record checksum and state.
    repaired_store = AnnotationStore(output)
    for task in repaired_store.manifest["tasks"]:
        annotator = task["assigned_to"]
        task_id = task["task_id"]
        record = repaired_store.load_record(annotator, task_id)
        if record["status"] != "complete":
            raise ValueError(f"Repaired correction remains incomplete: {task_id}")
        path = repaired_store.annotation_mask_path(annotator, task_id)
        if _sha256(path) != record.get("mask_sha256"):
            raise ValueError(f"Repaired correction checksum mismatch: {task_id}")
        mask = _read_binary_mask(path)
        if _has_small_components(mask, minimum_component_area):
            raise ValueError(f"Repaired correction retains tiny fragments: {task_id}")
        if _hole_area(mask):
            raise ValueError(f"Repaired correction retains enclosed holes: {task_id}")
    return metadata


def repair_brush_persistence_mask(
    mask: np.ndarray,
    *,
    kernel_size: int = 9,
    minimum_component_area: int = 32,
) -> np.ndarray:
    """Reconnect saved brush dabs, fill tool gaps, then discard tiny islands."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if minimum_component_area < 1:
        raise ValueError("minimum_component_area must be positive")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    repaired = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    repaired = _fill_holes(repaired)
    repaired = _remove_small_components(repaired, minimum_component_area)
    return repaired


def _read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        mask = np.asarray(loaded.convert("L"), dtype=np.uint8)
    return mask > 0


def _has_small_components(mask: np.ndarray, threshold: int) -> bool:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return any(int(stats[index, cv2.CC_STAT_AREA]) < threshold for index in range(1, count))


def _remove_small_components(mask: np.ndarray, threshold: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= threshold
    keep[0] = False
    return keep[labels]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    _, labels, _, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), 8)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    holes = np.logical_and(labels > 0, ~np.isin(labels, border_labels))
    return np.logical_or(mask, holes)


def _hole_area(mask: np.ndarray) -> int:
    return int(np.logical_and(_fill_holes(mask), ~mask).sum())


def _bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _require_empty_output(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_binary_png_atomic(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(temporary, format="PNG", optimize=True)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
