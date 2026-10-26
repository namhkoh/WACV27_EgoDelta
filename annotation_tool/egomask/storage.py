from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading

from PIL import Image, ImageChops


VALID_STATUSES = {"in_progress", "complete", "needs_review"}


class AccessDenied(Exception):
    pass


class RevisionConflict(Exception):
    def __init__(self, current_revision: int):
        super().__init__(f"Stale revision; server is at revision {current_revision}")
        self.current_revision = current_revision


class InvalidAnnotation(Exception):
    pass


class AnnotationStore:
    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root).expanduser().resolve()
        manifest_path = self.run_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Annotation manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported manifest schema: {self.manifest.get('schema_version')}")
        self.tasks = {row["task_id"]: row for row in self.manifest["tasks"]}
        if len(self.tasks) != len(self.manifest["tasks"]):
            raise ValueError("Manifest contains duplicate task IDs")
        self.annotators_by_token = {row["token"]: row["id"] for row in self.manifest["annotators"]}
        self.annotators = {row["id"] for row in self.manifest["annotators"]}
        self._lock = threading.RLock()

    def authenticate(self, token: str | None) -> str:
        annotator = self.annotators_by_token.get(token or "")
        if annotator is None:
            raise AccessDenied("Invalid or missing annotator token")
        return annotator

    def require_task(self, annotator: str, task_id: str) -> dict:
        task = self.tasks.get(task_id)
        if task is None or task["assigned_to"] != annotator:
            raise AccessDenied("Task is not assigned to this annotator")
        return task

    def bootstrap(self, annotator: str) -> dict:
        task_rows = []
        for task in self.manifest["tasks"]:
            if task["assigned_to"] != annotator:
                continue
            record = self.load_record(annotator, task["task_id"])
            task_rows.append(
                {
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "task_kind": task["task_kind"],
                    "image_name": task["image_name"],
                    "status": record["status"],
                    "revision": record["revision"],
                }
            )
        return {
            "annotator": annotator,
            "tasks": task_rows,
            "progress": _progress(task_rows),
            "schema_version": self.manifest["schema_version"],
        }

    def task_payload(self, annotator: str, task_id: str) -> dict:
        task = self.require_task(annotator, task_id)
        record = self.load_record(annotator, task_id)
        return {
            "task_id": task_id,
            "scene_id": task["scene_id"],
            "scene_uuid": task["scene_uuid"],
            "task_kind": task["task_kind"],
            "image_name": task["image_name"],
            "width": task["camera"]["width"],
            "height": task["camera"]["height"],
            "components": [
                {
                    "component_id": row["component_id"],
                    "label": row["label"],
                    "color": row["color"],
                    "adjacent_gt_indices": row["adjacent_gt_indices"],
                }
                for row in task["components"]
            ],
            "projection": task["projection"],
            "adjudication": task.get("adjudication"),
            "correction": task.get("correction"),
            "record": record,
            "has_saved_mask": self.annotation_mask_path(annotator, task_id).is_file(),
            "has_reference_mask": bool(task.get("reference_mask")),
        }

    def image_path(self, annotator: str, task_id: str) -> Path:
        task = self.require_task(annotator, task_id)
        path = Path(task["source_image"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def mask_path(self, annotator: str, task_id: str) -> Path:
        self.require_task(annotator, task_id)
        saved = self.annotation_mask_path(annotator, task_id)
        if saved.is_file():
            return saved
        return self.initial_mask_path(annotator, task_id)

    def initial_mask_path(self, annotator: str, task_id: str) -> Path:
        task = self.require_task(annotator, task_id)
        initial = (self.run_root / task["initial_mask"]).resolve()
        if not initial.is_relative_to(self.run_root) or not initial.is_file():
            raise FileNotFoundError(initial)
        return initial

    def reference_mask_path(self, annotator: str, task_id: str) -> Path:
        task = self.require_task(annotator, task_id)
        relative = task.get("reference_mask")
        if not relative:
            raise FileNotFoundError(f"Task has no reference mask: {task_id}")
        reference = (self.run_root / str(relative)).resolve()
        if not reference.is_relative_to(self.run_root) or not reference.is_file():
            raise FileNotFoundError(reference)
        return reference

    def annotation_mask_path(self, annotator: str, task_id: str) -> Path:
        if annotator not in self.annotators or task_id not in self.tasks:
            raise AccessDenied("Unknown annotation target")
        return self.run_root / "annotations" / annotator / f"{task_id}.png"

    def record_path(self, annotator: str, task_id: str) -> Path:
        if annotator not in self.annotators or task_id not in self.tasks:
            raise AccessDenied("Unknown annotation target")
        return self.run_root / "records" / annotator / f"{task_id}.json"

    def load_record(self, annotator: str, task_id: str) -> dict:
        self.require_task(annotator, task_id)
        path = self.record_path(annotator, task_id)
        if not path.is_file():
            return {"status": "not_started", "revision": 0, "note": "", "updated_utc": None}
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidAnnotation(f"Could not read annotation record {path}: {exc}") from exc
        return record

    def save(self, annotator: str, payload: dict) -> dict:
        task_id = str(payload.get("task_id", ""))
        task = self.require_task(annotator, task_id)
        status = str(payload.get("status", ""))
        if status not in VALID_STATUSES:
            raise InvalidAnnotation(f"Invalid status {status!r}")
        note = str(payload.get("note", "")).strip()
        if len(note) > 2000:
            raise InvalidAnnotation("Note exceeds 2,000 characters")
        try:
            client_revision = int(payload.get("revision", -1))
        except (TypeError, ValueError) as exc:
            raise InvalidAnnotation("revision must be an integer") from exc

        mask = _decode_and_validate_mask(
            payload.get("mask_data_url"),
            width=int(task["camera"]["width"]),
            height=int(task["camera"]["height"]),
            maximum_label=len(task["components"]),
        )
        with self._lock:
            current = self.load_record(annotator, task_id)
            if client_revision != int(current["revision"]):
                raise RevisionConflict(int(current["revision"]))
            revision = client_revision + 1
            mask_path = self.annotation_mask_path(annotator, task_id)
            _save_image_atomic(mask_path, mask)
            digest = _sha256(mask_path)
            record = {
                "task_id": task_id,
                "annotator": annotator,
                "status": status,
                "revision": revision,
                "note": note,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "mask_sha256": digest,
                "nonzero_bbox": list(mask.getbbox()) if mask.getbbox() else None,
            }
            _write_json_atomic(self.record_path(annotator, task_id), record)
            return record


def summarize_run(store: AnnotationStore) -> dict:
    rows = []
    for annotator in sorted(store.annotators):
        tasks = [task for task in store.manifest["tasks"] if task["assigned_to"] == annotator]
        brief = []
        for task in tasks:
            record = store.load_record(annotator, task["task_id"])
            brief.append({"status": record["status"]})
        rows.append({"annotator": annotator, **_progress(brief)})
    total = {
        key: sum(row[key] for row in rows)
        for key in ("total", "not_started", "in_progress", "needs_review", "complete")
    }
    return {"annotators": rows, "total": total}


def _progress(rows: list[dict]) -> dict:
    statuses = [row["status"] for row in rows]
    return {
        "total": len(statuses),
        "not_started": statuses.count("not_started"),
        "in_progress": statuses.count("in_progress"),
        "needs_review": statuses.count("needs_review"),
        "complete": statuses.count("complete"),
    }


def _decode_and_validate_mask(data_url, *, width: int, height: int, maximum_label: int) -> Image.Image:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise InvalidAnnotation("mask_data_url must be a base64 PNG data URL")
    encoded = data_url.split(",", 1)[1]
    if len(encoded) > 12_000_000:
        raise InvalidAnnotation("Encoded mask is too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise InvalidAnnotation("Mask is not valid base64") from exc
    if len(raw) > 8_000_000:
        raise InvalidAnnotation("Decoded mask is too large")
    try:
        with Image.open(BytesIO(raw)) as loaded:
            loaded.load()
            if loaded.size != (width, height):
                raise InvalidAnnotation(
                    f"Mask is {loaded.size[0]}x{loaded.size[1]}, expected {width}x{height}"
                )
            rgb = loaded.convert("RGB")
    except InvalidAnnotation:
        raise
    except Exception as exc:
        raise InvalidAnnotation(f"Could not decode PNG mask: {exc}") from exc

    red, green, blue = rgb.split()
    if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
        raise InvalidAnnotation("Mask must encode identical label values in the RGB channels")
    _, observed_maximum = red.getextrema()
    if observed_maximum > maximum_label:
        raise InvalidAnnotation(
            f"Mask contains label {observed_maximum}, but this scene has only {maximum_label} components"
        )
    return red


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
