"""Lossless cache boundary between frozen Stage 2 and experimental Stage 3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import EgoDeltaDiffResult


@dataclass(frozen=True)
class Stage2Cache:
    scene_id: str
    mu: np.ndarray
    normals: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    overlap: np.ndarray
    source_indices: np.ndarray
    rgb01: np.ndarray
    legacy_cluster_labels: np.ndarray
    metadata: dict[str, Any]


def save_stage2_cache(path: Path, result: EgoDeltaDiffResult) -> None:
    """Save every Stage-3 input plus legacy-only RGB/cluster compatibility data."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "scene_id": str(result.diagnostics.get("scene_name", path.stem)),
        "pre_ply": str(result.diagnostics.get("pre_ply", "")),
        "post_ply": str(result.diagnostics.get("post_ply", "")),
        "stage2_diagnostics": result.diagnostics,
    }
    np.savez_compressed(
        path,
        mu=_numpy(result.g_pre.mu, np.float32),
        normals=_numpy(result.g_pre.normal, np.float32),
        delta=_numpy(result.delta_pre, np.float32),
        omega=_numpy(result.omega_pre, np.float32),
        overlap=_numpy(result.overlap_pre, bool),
        source_indices=_numpy(result.g_pre.source_indices, np.int64),
        rgb01=_numpy(result.g_pre.rgb01, np.float32),
        legacy_cluster_labels=_numpy(result.cluster_labels, np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_stage2_cache(path: Path) -> Stage2Cache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return Stage2Cache(
            scene_id=str(metadata["scene_id"]),
            mu=data["mu"].copy(),
            normals=data["normals"].copy(),
            delta=data["delta"].copy(),
            omega=data["omega"].copy(),
            overlap=data["overlap"].copy(),
            source_indices=data["source_indices"].copy(),
            rgb01=data["rgb01"].copy(),
            legacy_cluster_labels=data["legacy_cluster_labels"].copy(),
            metadata=metadata,
        )


def cache_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value, dtype):
    return value.detach().cpu().numpy().astype(dtype, copy=False)
