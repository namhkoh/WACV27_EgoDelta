from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


RUNS_ROOT = Path(__file__).resolve().parent / "runs"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def date_code(now: datetime | None = None) -> str:
    value = now or datetime.now()
    return value.strftime("%m%d%y")


def safe_name(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("artifact name cannot be empty")
    return cleaned


def run_directory(task_name: str, *, now: datetime | None = None, runs_root: Path = RUNS_ROOT) -> Path:
    return Path(runs_root) / date_code(now) / safe_name(task_name)


def artifact_path(
    task_name: str,
    artifact_name: str,
    *,
    suffix: str = ".json",
    now: datetime | None = None,
    runs_root: Path = RUNS_ROOT,
) -> Path:
    if not suffix.startswith("."):
        raise ValueError("suffix must start with '.'")
    return run_directory(task_name, now=now, runs_root=runs_root) / f"{safe_name(artifact_name)}{suffix}"


def write_json_artifact(
    task_name: str,
    artifact_name: str,
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
    runs_root: Path = RUNS_ROOT,
) -> Path:
    path = artifact_path(task_name, artifact_name, now=now, runs_root=runs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
