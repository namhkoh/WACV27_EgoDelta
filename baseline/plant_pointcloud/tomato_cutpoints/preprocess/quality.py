from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LOGGER = logging.getLogger("tomato_cutpoints")


def _apply_column_threshold(
    df: pd.DataFrame,
    mask: np.ndarray,
    column: str,
    threshold: float,
    op: str,
    report: dict[str, Any],
) -> np.ndarray:
    before = int(mask.sum())
    if column not in df.columns:
        LOGGER.warning("Quality column %s missing; continuing without that filter", column)
        report[column] = {"missing": True, "removed": 0}
        return mask

    values = pd.to_numeric(df[column], errors="coerce").to_numpy()
    if op == "<=":
        keep = np.isnan(values) | (values <= threshold)
    elif op == ">=":
        keep = np.isnan(values) | (values >= threshold)
    else:
        raise ValueError(f"Unsupported op: {op}")
    mask = mask & keep
    removed = before - int(mask.sum())
    report[column] = {"missing": False, "threshold": threshold, "removed": removed}
    LOGGER.info("Quality filter %s %s %s removed %d points", column, op, threshold, removed)
    return mask


def _statistical_outlier_mask(xyz: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    if len(xyz) <= nb_neighbors + 1:
        return np.ones(len(xyz), dtype=bool)
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=nb_neighbors + 1, workers=-1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return mean_dists <= threshold


def filter_quality(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    qcfg = config.get("quality", config)
    mask = np.ones(len(df), dtype=bool)
    report: dict[str, Any] = {"n_input": int(len(df)), "filters": {}}

    if qcfg.get("max_inv_dist_std") is not None:
        mask = _apply_column_threshold(
            df, mask, "inv_dist_std", float(qcfg["max_inv_dist_std"]), "<=", report["filters"]
        )
    if qcfg.get("max_dist_std") is not None:
        mask = _apply_column_threshold(df, mask, "dist_std", float(qcfg["max_dist_std"]), "<=", report["filters"])
    if qcfg.get("min_observations") is not None:
        mask = _apply_column_threshold(
            df, mask, "obs_count", float(qcfg["min_observations"]), ">=", report["filters"]
        )

    out = df.loc[mask].reset_index(drop=True)

    stat_cfg = qcfg.get("statistical_outlier", {})
    if stat_cfg.get("enabled", False) and len(out):
        xyz_tmp = out[["x", "y", "z"]].to_numpy(dtype=np.float64)
        stat_mask = _statistical_outlier_mask(
            xyz_tmp,
            int(stat_cfg.get("nb_neighbors", 20)),
            float(stat_cfg.get("std_ratio", 2.0)),
        )
        removed = int((~stat_mask).sum())
        LOGGER.info("Statistical outlier removal removed %d points", removed)
        report["filters"]["statistical_outlier"] = {"removed": removed}
        out = out.loc[stat_mask].reset_index(drop=True)

    xyz = out[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
    report["n_output"] = int(len(out))
    report["n_removed"] = int(report["n_input"] - report["n_output"])
    return out, xyz, report
