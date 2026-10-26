from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_action_manifests(paths: Iterable[str | Path]) -> list[dict]:
    manifests = []
    for path in paths:
        data = read_json(path)
        if "samples" not in data:
            raise RuntimeError(f"Action-point manifest has no samples: {path}")
        manifests.append(data)
    return manifests


def iter_samples(manifests: Iterable[dict]):
    for manifest in manifests:
        for sample in manifest.get("samples", []):
            yield sample


def split_csv(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {part.strip() for part in str(value).split(",") if part.strip()}


def load_scene_split(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = read_json(path)
    if "scene_to_split" in data:
        data = data["scene_to_split"]
    return {str(scene): str(split) for scene, split in data.items()}


def sample_split(sample: dict, scene_split: dict[str, str] | None = None, default: str = "train") -> str:
    scene_split = scene_split or {}
    scene_id = str(sample.get("scene_id", ""))
    if scene_id in scene_split:
        return scene_split[scene_id]
    return str(sample.get("split") or default)


def filter_samples(
    samples: Iterable[dict],
    split: str | None = None,
    scenes: set[str] | None = None,
    scene_split: dict[str, str] | None = None,
) -> list[dict]:
    selected = []
    for sample in samples:
        scene_id = str(sample.get("scene_id", ""))
        if scenes is not None and scene_id not in scenes:
            continue
        if split is not None and sample_split(sample, scene_split) != split:
            continue
        selected.append(sample)
    return selected
