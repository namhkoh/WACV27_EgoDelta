"""Read-only access to the frozen, continuous Stage-2 cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Stage2Cache:
    scene_id: str
    mu: np.ndarray
    normals: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    overlap: np.ndarray
    source_indices: np.ndarray
    metadata: dict[str, Any]


def load_stage2_cache(path: Path) -> Stage2Cache:
    """Load only the fields consumed by the selected Stage-3 method."""

    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return Stage2Cache(
            scene_id=str(metadata["scene_id"]),
            mu=data["mu"].copy(),
            normals=data["normals"].copy(),
            delta=data["delta"].copy(),
            omega=data["omega"].copy(),
            overlap=data["overlap"].copy(),
            source_indices=data["source_indices"].copy(),
            metadata=metadata,
        )


def cache_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
