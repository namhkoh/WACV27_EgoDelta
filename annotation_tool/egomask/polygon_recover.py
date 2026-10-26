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


DEFAULT_ANNOTATORS = ("annotator_1", "annotator_2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover filled human polygons drawn over an uncleared initial mask, without "
            "modifying the source annotations."
        )
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator", action="append", dest="annotators")
    parser.add_argument("--minimum-initial-iou", type=float, default=0.80)
    parser.add_argument("--closing-size", type=int, default=81)
    parser.add_argument("--minimum-contour-area", type=float, default=500.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_polygon_overrides(
            args.run,
            args.output,
            annotators=args.annotators or DEFAULT_ANNOTATORS,
            minimum_initial_iou=args.minimum_initial_iou,
            closing_size=args.closing_size,
            minimum_contour_area=args.minimum_contour_area,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Recovered {len(payload['overrides'])} human polygon masks; "
        f"kept {payload['kept_source_count']} already-redrawn source masks"
    )
    return 0


def build_polygon_overrides(
    run_root: str | Path,
    output: str | Path,
    *,
    annotators: tuple[str, ...] | list[str] = DEFAULT_ANNOTATORS,
    minimum_initial_iou: float = 0.80,
    closing_size: int = 81,
    minimum_contour_area: float = 500.0,
) -> dict:
    store = AnnotationStore(run_root)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    annotators = tuple(dict.fromkeys(annotators))
    unknown = set(annotators).difference(store.annotators)
    if unknown:
        raise ValueError(f"Unknown annotators: {sorted(unknown)}")
    if not 0 <= minimum_initial_iou <= 1:
        raise ValueError("minimum_initial_iou must be in [0, 1]")
    if closing_size < 3 or closing_size % 2 == 0:
        raise ValueError("closing_size must be an odd integer of at least 3")
    if minimum_contour_area <= 0:
        raise ValueError("minimum_contour_area must be positive")

    overrides: list[dict] = []
    audit_rows: list[dict] = []
    selected_tasks = [
        task for task in store.manifest["tasks"] if task["assigned_to"] in annotators
    ]
    for task in selected_tasks:
        task_id = task["task_id"]
        annotator = task["assigned_to"]
        record = store.load_record(annotator, task_id)
        if record["status"] != "complete":
            raise ValueError(f"Source annotation is not complete: {task_id}")
        source_path = store.annotation_mask_path(annotator, task_id)
        initial_path = store.initial_mask_path(annotator, task_id)
        source = _load_labels(source_path, task)
        initial = _load_labels(initial_path, task)
        source_foreground = source > 0
        initial_foreground = initial > 0
        initial_iou = _iou(source_foreground, initial_foreground)
        source_area = int(source_foreground.sum())
        initial_area = int(initial_foreground.sum())
        common = {
            "task_id": task_id,
            "scene_id": task["scene_id"],
            "task_kind": task["task_kind"],
            "image_name": task["image_name"],
            "annotator": annotator,
            "source_annotation_sha256": _sha256(source_path),
            "initial_mask_sha256": _sha256(initial_path),
            "initial_iou": initial_iou,
            "source_area": source_area,
            "initial_area": initial_area,
        }
        if initial_iou < minimum_initial_iou:
            audit_rows.append(
                {
                    **common,
                    "decision": "keep_source_human_mask",
                    "reason": "saved mask already differs substantially from the initial projection",
                    "delta_pixels": int((source != initial).sum()),
                    "recovered_area": source_area,
                    "recovered_to_initial_ratio": source_area / max(initial_area, 1),
                    "contour_count": "",
                }
            )
            continue

        if task["task_kind"] == "harvesting":
            recovered, metrics = recover_convex_polygon_fill(initial, source)
            recovery_strategy = "clustered_convex_outline_fill"
        else:
            recovered, metrics = recover_polygon_fill(
                initial,
                source,
                closing_size=closing_size,
                minimum_contour_area=minimum_contour_area,
            )
            recovery_strategy = "closed_outline_with_source_interior_fill"
        recovered_area = int(recovered.sum())
        if not recovered_area:
            audit_rows.append(
                {
                    **common,
                    "decision": "needs_human_review",
                    "reason": "no recoverable polygon contour",
                    **metrics,
                    "recovered_area": 0,
                    "recovered_to_initial_ratio": 0.0,
                }
            )
            continue

        relative_mask = Path("masks") / f"{task_id}.png"
        _save_png_atomic(
            output / relative_mask,
            # Store a conventional binary PNG (0 background, 255 foreground).
            # The export loader treats any non-zero value as foreground, but 255
            # also makes the recovered human mask directly viewable in image tools.
            Image.fromarray(recovered.astype(np.uint8) * 255, mode="L"),
        )
        recovered_mask_sha256 = _sha256(output / relative_mask)
        area_ratio = recovered_area / max(initial_area, 1)
        warnings = []
        if not 0.15 <= area_ratio <= 1.25:
            warnings.append("recovered area is outside the expected range")
        override = {
            "task_id": task_id,
            "mask": str(relative_mask),
            "reason": "fill human polygon boundary drawn over the initial mask",
            "source_annotation_sha256": common["source_annotation_sha256"],
            "mask_sha256": recovered_mask_sha256,
            "source_annotator": annotator,
            "human_evidence": "label-wise saved-mask versus initial-mask boundary",
            "recovery_strategy": recovery_strategy,
            "initial_mask_sha256": common["initial_mask_sha256"],
            "initial_iou": initial_iou,
            "closing_size": closing_size,
            "minimum_contour_area": minimum_contour_area,
            "delta_pixels": metrics["delta_pixels"],
            "contour_count": metrics["contour_count"],
            "recovered_area": recovered_area,
            "recovered_to_initial_ratio": area_ratio,
            "warnings": warnings,
        }
        overrides.append(override)
        audit_rows.append(
            {
                **common,
                "decision": "recover_human_polygon" if not warnings else "recover_and_review",
                "reason": override["reason"],
                "delta_pixels": metrics["delta_pixels"],
                "recovered_area": recovered_area,
                "recovered_to_initial_ratio": area_ratio,
                "contour_count": metrics["contour_count"],
            }
        )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "human_polygon_outline_fill",
        "status": "candidate_requires_visual_verification",
        "source_run": str(store.run_root),
        "source_manifest_sha256": _sha256(store.run_root / "manifest.json"),
        "selected_annotators": list(annotators),
        "minimum_initial_iou": minimum_initial_iou,
        "closing_size": closing_size,
        "minimum_contour_area": minimum_contour_area,
        "selected_task_count": len(selected_tasks),
        "kept_source_count": sum(row["decision"] == "keep_source_human_mask" for row in audit_rows),
        "needs_human_review_count": sum(row["decision"] == "needs_human_review" for row in audit_rows),
        "warning_count": sum(row["decision"] == "recover_and_review" for row in audit_rows),
        "overrides": overrides,
    }
    _write_json_atomic(output / "overrides.json", payload)
    _write_audit_csv(output / "audit.csv", audit_rows)
    return payload


def recover_polygon_fill(
    initial: np.ndarray,
    source: np.ndarray,
    *,
    closing_size: int = 81,
    minimum_contour_area: float = 500.0,
) -> tuple[np.ndarray, dict]:
    initial = np.asarray(initial)
    source = np.asarray(source)
    if initial.shape != source.shape or initial.ndim != 2:
        raise ValueError("initial and source masks must be same-sized 2D arrays")
    delta = (source != initial).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    closed = cv2.morphologyEx(delta, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = [contour for contour in contours if cv2.contourArea(contour) >= minimum_contour_area]
    recovered = np.zeros_like(delta)
    envelopes = np.zeros_like(delta)
    for contour in accepted:
        cv2.drawContours(recovered, [contour], -1, 1, cv2.FILLED)
        cv2.drawContours(envelopes, [cv2.convexHull(contour)], -1, 1, cv2.FILLED)

    # A polygon painted over an uncleared component can leave little or no
    # label difference across portions where its selected label already
    # matched the starting label. Those missing portions appear as crescents
    # or brush-shaped gaps in the recovered mask. Inside each evidence
    # envelope, reuse only foreground that is present in the completed human
    # annotation. This fills the polygon interior without restoring unrelated
    # projected blobs elsewhere in the image.
    recovered |= ((source > 0) & (envelopes > 0)).astype(np.uint8)
    return recovered.astype(bool), {
        "delta_pixels": int(delta.sum()),
        "contour_count": len(accepted),
    }


def recover_convex_polygon_fill(
    initial: np.ndarray,
    source: np.ndarray,
    *,
    cluster_size: int = 11,
    minimum_delta_pixels: int = 10,
    minimum_hull_area: float = 50.0,
) -> tuple[np.ndarray, dict]:
    initial = np.asarray(initial)
    source = np.asarray(source)
    if initial.shape != source.shape or initial.ndim != 2:
        raise ValueError("initial and source masks must be same-sized 2D arrays")
    delta = (source != initial).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cluster_size, cluster_size))
    expanded = cv2.dilate(delta, kernel)
    component_count, components = cv2.connectedComponents(expanded, connectivity=8)
    clusters: list[tuple[int, np.ndarray]] = []
    for component in range(1, component_count):
        ys, xs = np.nonzero((components == component) & (delta > 0))
        if len(xs) >= minimum_delta_pixels:
            clusters.append((len(xs), np.column_stack((xs, ys)).astype(np.int32)))
    if not clusters:
        return np.zeros_like(delta, dtype=bool), {
            "delta_pixels": int(delta.sum()),
            "contour_count": 0,
        }
    largest = max(count for count, _ in clusters)
    recovered = np.zeros_like(delta)
    accepted = 0
    for count, points in clusters:
        if count < max(minimum_delta_pixels, int(np.ceil(0.10 * largest))):
            continue
        hull = cv2.convexHull(points)
        if cv2.contourArea(hull) < minimum_hull_area:
            continue
        cv2.drawContours(recovered, [hull], -1, 1, cv2.FILLED)
        accepted += 1
    return recovered.astype(bool), {
        "delta_pixels": int(delta.sum()),
        "contour_count": accepted,
    }


def _load_labels(path: Path, task: dict) -> np.ndarray:
    with Image.open(path) as loaded:
        image = loaded.convert("L")
        image.load()
    expected = (int(task["camera"]["width"]), int(task["camera"]["height"]))
    if image.size != expected:
        raise ValueError(f"Wrong mask dimensions for {task['task_id']}: {image.size}")
    labels = np.asarray(image)
    if int(labels.max(initial=0)) > len(task["components"]):
        raise ValueError(f"Invalid mask label for {task['task_id']}")
    return labels


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(np.logical_and(first, second).sum() / union)


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "task_id",
        "scene_id",
        "task_kind",
        "image_name",
        "annotator",
        "decision",
        "reason",
        "initial_iou",
        "source_area",
        "initial_area",
        "delta_pixels",
        "recovered_area",
        "recovered_to_initial_ratio",
        "contour_count",
        "source_annotation_sha256",
        "initial_mask_sha256",
    )
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


def _save_png_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG")
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
