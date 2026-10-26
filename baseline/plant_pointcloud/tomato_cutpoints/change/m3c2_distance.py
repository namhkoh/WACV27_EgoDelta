from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("tomato_cutpoints")


@dataclass
class M3C2Result:
    corepoint_indices: np.ndarray
    distances: np.ndarray
    full_distances: np.ndarray
    uncertainties: dict[str, np.ndarray]
    normals: np.ndarray | None


def _select_corepoint_indices(n_points: int, config: dict[str, Any], seed: int = 13) -> np.ndarray:
    stride = max(1, int(config.get("corepoint_stride", 1)))
    indices = np.arange(0, n_points, stride, dtype=np.int64)
    max_corepoints = int(config.get("max_corepoints", 300_000))
    if len(indices) > max_corepoints:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_corepoints, replace=False)).astype(np.int64)
    return indices


def _uncertainties_to_dict(raw: Any) -> dict[str, np.ndarray]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): np.asarray(v) for k, v in raw.items()}
    if hasattr(raw, "_asdict"):
        return {str(k): np.asarray(v) for k, v in raw._asdict().items()}
    result: dict[str, np.ndarray] = {}
    for name in ("lodetection", "num_samples1", "num_samples2", "spread1", "spread2"):
        if hasattr(raw, name):
            result[name] = np.asarray(getattr(raw, name))
    return result


def run_m3c2(
    pre_xyz: np.ndarray,
    post_xyz_aligned: np.ndarray,
    config: dict[str, Any],
    out_dir: str | Path | None = None,
    strict: bool = False,
) -> M3C2Result | None:
    mcfg = config.get("m3c2", config)
    if not mcfg.get("enabled", True):
        return None

    pre_xyz = np.asarray(pre_xyz, dtype=np.float64)
    post_xyz_aligned = np.asarray(post_xyz_aligned, dtype=np.float64)
    if len(pre_xyz) == 0 or len(post_xyz_aligned) == 0:
        return None

    try:
        import py4dgeo  # type: ignore

        corepoint_indices = _select_corepoint_indices(len(pre_xyz), mcfg)
        corepoints = pre_xyz[corepoint_indices].astype(np.float64)
        epoch1 = py4dgeo.Epoch(pre_xyz.astype(np.float64))
        epoch2 = py4dgeo.Epoch(post_xyz_aligned.astype(np.float64))
        m3c2 = py4dgeo.M3C2(
            epochs=(epoch1, epoch2),
            corepoints=corepoints,
            cyl_radius=float(mcfg.get("cyl_radius", 0.03)),
            normal_radii=list(mcfg.get("normal_radii", [0.02, 0.04, 0.08])),
            max_distance=float(mcfg.get("max_distance", 0.10)),
        )
        distances, uncertainties_raw = m3c2.run()
        distances = np.asarray(distances, dtype=np.float64)
        uncertainties = _uncertainties_to_dict(uncertainties_raw)
        normals = None
        if hasattr(m3c2, "directions"):
            try:
                normals = np.asarray(m3c2.directions(), dtype=np.float64)
            except Exception:
                normals = None

        full = np.full(len(pre_xyz), np.nan, dtype=np.float64)
        full[corepoint_indices[: len(distances)]] = distances
        return M3C2Result(corepoint_indices[: len(distances)], distances, full, uncertainties, normals)
    except Exception as exc:
        LOGGER.exception("M3C2 failed; continuing in NN-only mode")
        if out_dir is not None:
            path = Path(out_dir) / "m3c2_failed.txt"
            path.write_text(str(exc) + "\n", encoding="utf-8")
        if strict:
            raise
        return None


def m3c2_candidate_mask(result: M3C2Result | None, config: dict[str, Any]) -> np.ndarray | None:
    if result is None:
        return None
    mcfg = config.get("m3c2", config)
    distances = result.full_distances
    mask = np.isfinite(distances) & (np.abs(distances) > float(mcfg.get("threshold_m", 0.025)))

    if mcfg.get("use_lodetection", True) and "lodetection" in result.uncertainties:
        lod_core = np.asarray(result.uncertainties["lodetection"], dtype=np.float64)
        lod_full = np.full_like(distances, np.nan, dtype=np.float64)
        lod_full[result.corepoint_indices[: len(lod_core)]] = lod_core
        mask &= np.isnan(lod_full) | (np.abs(distances) > lod_full)

    if "num_samples1" in result.uncertainties:
        samples_core = np.asarray(result.uncertainties["num_samples1"], dtype=np.float64)
        samples_full = np.full_like(distances, np.nan, dtype=np.float64)
        samples_full[result.corepoint_indices[: len(samples_core)]] = samples_core
        mask &= np.isnan(samples_full) | (samples_full >= float(mcfg.get("min_samples1", 3)))

    return mask
