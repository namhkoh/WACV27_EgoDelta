from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading


class FinalExportReview:
    """Read-only index over a frozen EgoMask export."""

    def __init__(self, export_root: str | Path, *, flags_path: str | Path | None = None):
        self.export_root = Path(export_root).expanduser().resolve()
        dataset_path = self.export_root / "dataset.json"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Final export dataset not found: {dataset_path}")
        dataset_bytes = dataset_path.read_bytes()
        self.dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
        self.dataset = json.loads(dataset_bytes)
        if self.dataset.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported final export schema: {self.dataset.get('schema_version')}"
            )
        rows = self.dataset.get("tasks")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Final export has no tasks")
        self.tasks = {str(row.get("task_id", "")): row for row in rows}
        if "" in self.tasks or len(self.tasks) != len(rows):
            raise ValueError("Final export contains missing or duplicate task IDs")
        self.flags_path = (
            Path(flags_path).expanduser().resolve() if flags_path is not None else None
        )
        self._lock = threading.RLock()
        self._flags = self._load_flags()

        # Validate every path once at startup. Browser requests can then select
        # only a known task ID and never pass a filesystem path to the server.
        for task_id in self.tasks:
            self.image_path(task_id)
            self.mask_path(task_id)

    def summary(self, reviewer: str | None = None) -> dict:
        return {
            "enabled": True,
            "name": self.export_root.name,
            "total": len(self.tasks),
            "created_utc": self.dataset.get("created_utc"),
            "component_policy": self.dataset.get("component_policy"),
            "solid_binary_masks": bool(self.dataset.get("solid_binary_masks")),
            "flagging_enabled": self.flags_path is not None,
            "flagged_count": self.flagged_count(),
            "reviewer_flagged_count": self.reviewer_flagged_count(reviewer),
        }

    def bootstrap(self, reviewer: str | None = None) -> dict:
        return {
            **self.summary(reviewer),
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "scene_id": task["scene_id"],
                    "scene_uuid": task.get("scene_uuid"),
                    "task_kind": task["task_kind"],
                    "image_name": task["image_name"],
                    "annotator": task.get("annotator"),
                    "mask_source": task.get("mask_source"),
                    "width": int(task["camera"]["width"]),
                    "height": int(task["camera"]["height"]),
                    **self.flag_state(task["task_id"], reviewer),
                }
                for task in self.dataset["tasks"]
            ],
        }

    def flag_state(self, task_id: str, reviewer: str | None = None) -> dict:
        self.require_task(task_id)
        with self._lock:
            reviewers = self._flags["flags"].get(task_id, {})
            flagged_by = sorted(reviewers)
            return {
                "flagged": reviewer in reviewers if reviewer is not None else False,
                "needs_redo": bool(flagged_by),
                "flagged_by": flagged_by,
                "flagged_updated_utc": max(reviewers.values()) if reviewers else None,
            }

    def flagged_count(self) -> int:
        with self._lock:
            return sum(bool(reviewers) for reviewers in self._flags["flags"].values())

    def reviewer_flagged_count(self, reviewer: str | None) -> int:
        if reviewer is None:
            return 0
        with self._lock:
            return sum(
                reviewer in reviewers for reviewers in self._flags["flags"].values()
            )

    def set_flag(self, task_id: str, reviewer: str, flagged: bool) -> dict:
        self.require_task(task_id)
        if self.flags_path is None:
            raise ValueError("Final-review flagging is not configured")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            flags = self._flags["flags"]
            reviewers = flags.setdefault(task_id, {})
            if flagged:
                reviewers[reviewer] = now
            else:
                reviewers.pop(reviewer, None)
                if not reviewers:
                    flags.pop(task_id, None)
            self._flags["updated_utc"] = now
            self._write_flags()
            return {
                **self.flag_state(task_id, reviewer),
                "flagged_count": self.flagged_count(),
                "reviewer_flagged_count": self.reviewer_flagged_count(reviewer),
            }

    def require_task(self, task_id: str) -> dict:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown final-export task: {task_id}")
        return task

    def image_path(self, task_id: str) -> Path:
        task = self.require_task(task_id)
        path = Path(str(task.get("source_image", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def mask_path(self, task_id: str) -> Path:
        task = self.require_task(task_id)
        candidates = task.get("binary_masks") or [task.get("label_map")]
        relative = Path(str(candidates[0] or ""))
        if relative.is_absolute():
            raise ValueError(f"Final mask path must be relative for {task_id}")
        path = (self.export_root / relative).resolve()
        if not path.is_relative_to(self.export_root):
            raise ValueError(f"Final mask escapes export root for {task_id}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _load_flags(self) -> dict:
        empty = {
            "schema_version": 1,
            "final_export": str(self.export_root),
            "dataset_sha256": self.dataset_sha256,
            "updated_utc": None,
            "flags": {},
        }
        if self.flags_path is None or not self.flags_path.is_file():
            return empty
        payload = json.loads(self.flags_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported final-review flags schema: {self.flags_path}")
        if payload.get("dataset_sha256") != self.dataset_sha256:
            raise ValueError("Final-review flags refer to a different frozen export")
        flags = payload.get("flags")
        if not isinstance(flags, dict):
            raise ValueError("Final-review flags must contain an object named 'flags'")
        unknown = set(flags).difference(self.tasks)
        if unknown:
            raise ValueError(f"Final-review flags contain unknown tasks: {sorted(unknown)[:5]}")
        for task_id, reviewers in flags.items():
            if not isinstance(reviewers, dict) or not all(
                isinstance(reviewer, str) and isinstance(updated, str)
                for reviewer, updated in reviewers.items()
            ):
                raise ValueError(f"Invalid final-review flag record for {task_id}")
        return payload

    def _write_flags(self) -> None:
        assert self.flags_path is not None
        self.flags_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.flags_path.name}.",
            suffix=".tmp",
            dir=self.flags_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._flags, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.flags_path)
        finally:
            if temporary.exists():
                temporary.unlink()
