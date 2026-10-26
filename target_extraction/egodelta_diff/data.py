"""Ground-truth parsing used only by the evaluation script."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


TASK_BY_SUFFIX = {"leaf": "deleafing", "tomato": "harvesting"}


def parse_gt_markdown(path: Path) -> dict[str, dict[str, np.ndarray]]:
    entries: dict[str, dict[str, list[list[float]]]] = {}
    current_uuid: str | None = None
    current_task: str | None = None
    header_re = re.compile(r"^\d+\.\s+([0-9a-fA-F-]+)\.vrs$")

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = header_re.match(line)
        if header:
            current_uuid = header.group(1).lower()
            entries[current_uuid] = {"deleafing": [], "harvesting": []}
            current_task = None
            continue
        if line in {"deleafing", "harvesting"}:
            current_task = line
            continue
        if line.startswith("[") and current_uuid and current_task:
            point = json.loads(line)
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"Invalid GT point in {path}: {line}")
            entries[current_uuid][current_task].append([float(value) for value in point])

    return {
        scene_uuid: {
            # Preserve the benchmark parser's historical float32 convention so
            # reference distances remain bit-for-bit reproducible.
            task: np.asarray(points, dtype=np.float32).reshape((-1, 3))
            for task, points in tasks.items()
        }
        for scene_uuid, tasks in entries.items()
    }


def find_scene_targets(
    scene_name: str,
    entries: dict[str, dict[str, np.ndarray]],
) -> tuple[str, str, np.ndarray] | None:
    match = re.match(r"^egodelta_([0-9a-fA-F]+)_(leaf|tomato)$", scene_name)
    if not match:
        return None
    compact_prefix = match.group(1).lower()
    task = TASK_BY_SUFFIX[match.group(2)]
    matches = [
        scene_uuid
        for scene_uuid in entries
        if scene_uuid.replace("-", "").startswith(compact_prefix)
    ]
    if not matches:
        return None
    scene_uuid = matches[0]
    return scene_uuid, task, entries[scene_uuid][task]
