from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("tomato_cutpoints")


@dataclass
class PointCloudData:
    points_df: pd.DataFrame
    xyz: np.ndarray
    frame_id: str | None
    source_path: Path
    metadata: dict[str, Any]


def _object_to_record(point: Any, idx: int) -> dict[str, Any]:
    def first_attr(names: list[str], default: Any = None) -> Any:
        for name in names:
            if hasattr(point, name):
                value = getattr(point, name)
                return value() if callable(value) and name.startswith("get_") else value
        return default

    pos = first_attr(["position_world", "position", "point_world", "world_position", "p_world"], None)
    if pos is None and hasattr(point, "get_position_world"):
        pos = point.get_position_world()
    if pos is None:
        pos = [first_attr(["x", "px_world"], np.nan), first_attr(["y", "py_world"], np.nan), first_attr(["z", "pz_world"], np.nan)]
    pos = np.asarray(pos, dtype=float).reshape(-1)

    return {
        "uid": first_attr(["uid", "id", "track_id"], idx),
        "graph_uid": first_attr(["graph_uid", "graph_id"], None),
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "inv_dist_std": first_attr(["inv_dist_std", "inverse_distance_std"], np.nan),
        "dist_std": first_attr(["dist_std", "distance_std"], np.nan),
    }


def _load_with_projectaria(path: Path) -> pd.DataFrame | None:
    try:
        import projectaria_tools.core.mps as mps  # type: ignore
    except Exception:
        return None

    try:
        points = mps.read_global_point_cloud(str(path))
    except Exception as exc:
        LOGGER.warning("Project Aria Tools failed to read %s; falling back to pandas: %s", path, exc)
        return None

    try:
        # Keep filtering controlled by YAML quality thresholds instead of applying
        # Project Aria Tools' built-in confidence defaults implicitly.
        try:
            from projectaria_tools.core.mps.utils import filter_points_from_confidence as _unused_filter  # type: ignore # noqa: F401
        except Exception:
            _unused_filter = None
        records = [_object_to_record(point, idx) for idx, point in enumerate(points)]
        df = pd.DataFrame.from_records(records)
        if {"x", "y", "z"}.issubset(df.columns):
            return df
    except Exception as exc:
        LOGGER.warning("Could not convert Project Aria point objects; falling back to pandas: %s", exc)
    return None


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_coord_columns(columns: list[str]) -> tuple[str, str, str]:
    normalized = {_normalize_col(c): c for c in columns}
    candidates = [
        ("pxworld", "pyworld", "pzworld"),
        ("px", "py", "pz"),
        ("xworld", "yworld", "zworld"),
        ("x", "y", "z"),
    ]
    for triplet in candidates:
        if all(c in normalized for c in triplet):
            return normalized[triplet[0]], normalized[triplet[1]], normalized[triplet[2]]

    axis_matches: dict[str, str] = {}
    for col in columns:
        norm = _normalize_col(col)
        for axis in ("x", "y", "z"):
            if axis in axis_matches:
                continue
            if re.search(rf"(^|p){axis}(world)?$", norm) or (axis in norm and "world" in norm):
                axis_matches[axis] = col
    if all(axis in axis_matches for axis in ("x", "y", "z")):
        return axis_matches["x"], axis_matches["y"], axis_matches["z"]
    raise ValueError(f"Could not identify coordinate columns from: {columns}")


def _find_optional_column(columns: list[str], names: list[str]) -> str | None:
    normalized = {_normalize_col(c): c for c in columns}
    for name in names:
        norm = _normalize_col(name)
        if norm in normalized:
            return normalized[norm]
    return None


def _load_with_pandas(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    df = pd.read_csv(path, compression=compression, comment="#")
    x_col, y_col, z_col = _find_coord_columns(list(df.columns))

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "z": pd.to_numeric(df[z_col], errors="coerce"),
        }
    )
    for target, names in {
        "uid": ["uid", "point_uid", "track_id", "id"],
        "graph_uid": ["graph_uid", "graph_id"],
        "inv_dist_std": ["inv_dist_std", "inverse_distance_std"],
        "dist_std": ["dist_std", "distance_std"],
        "obs_count": ["obs_count", "observation_count", "num_observations"],
        "pre_obs_count": ["pre_obs_count", "pre_observation_count"],
        "post_obs_count": ["post_obs_count", "post_observation_count"],
    }.items():
        col = _find_optional_column(list(df.columns), names)
        if col is not None:
            out[target] = df[col]

    if "uid" not in out.columns:
        out["uid"] = np.arange(len(out), dtype=np.int64)
    _synthesize_obs_count(out)

    return out


def _load_extra_csv_columns(path: Path, uid_values: pd.Series) -> pd.DataFrame | None:
    """Read custom CSV columns that Project Aria Tools may ignore, then merge by uid."""
    compression = "gzip" if path.suffix == ".gz" else "infer"
    try:
        header = pd.read_csv(path, compression=compression, comment="#", nrows=0)
    except Exception:
        return None

    uid_col = _find_optional_column(list(header.columns), ["uid", "point_uid", "track_id", "id"])
    if uid_col is None:
        return None
    requested = {
        "obs_count": ["obs_count", "observation_count", "num_observations"],
        "pre_obs_count": ["pre_obs_count", "pre_observation_count"],
        "post_obs_count": ["post_obs_count", "post_observation_count"],
    }
    usecols = [uid_col]
    rename = {uid_col: "uid"}
    for target, names in requested.items():
        col = _find_optional_column(list(header.columns), names)
        if col is not None:
            usecols.append(col)
            rename[col] = target
    if len(usecols) == 1:
        return None

    try:
        extras = pd.read_csv(path, compression=compression, comment="#", usecols=usecols).rename(columns=rename)
    except Exception as exc:
        LOGGER.warning("Could not preserve custom CSV observation columns from %s: %s", path, exc)
        return None

    # Keep only rows that can merge to loaded Aria points; this avoids carrying all raw rows downstream.
    try:
        keep = extras["uid"].isin(uid_values)
        extras = extras.loc[keep].drop_duplicates("uid", keep="first")
    except Exception:
        extras = extras.drop_duplicates("uid", keep="first")
    return extras


def _synthesize_obs_count(df: pd.DataFrame) -> None:
    if "obs_count" in df.columns:
        df["obs_count"] = pd.to_numeric(df["obs_count"], errors="coerce")
        return
    count_cols = [c for c in ("pre_obs_count", "post_obs_count") if c in df.columns]
    if not count_cols:
        return
    counts = [pd.to_numeric(df[c], errors="coerce").fillna(0) for c in count_cols]
    df["obs_count"] = pd.concat(counts, axis=1).max(axis=1).astype(np.int64)


def _infer_frame_id(df: pd.DataFrame) -> str | None:
    if "graph_uid" not in df.columns:
        return None
    values = df["graph_uid"].dropna().unique()
    if len(values) == 1:
        return str(values[0])
    if len(values) > 1:
        return "multiple:" + ",".join(str(v) for v in values[:5])
    return None


def load_observation_counts(observations_path: str | Path, chunksize: int = 1_000_000) -> pd.Series:
    path = Path(observations_path)
    compression = "gzip" if path.suffix == ".gz" else "infer"
    counts: pd.Series | None = None

    for chunk in pd.read_csv(path, compression=compression, comment="#", chunksize=chunksize):
        uid_col = _find_optional_column(list(chunk.columns), ["uid", "point_uid", "global_point_uid", "track_id", "id"])
        if uid_col is None:
            raise ValueError(f"Could not find uid column in observations file: {path}")
        vc = chunk[uid_col].value_counts(dropna=True)
        counts = vc if counts is None else counts.add(vc, fill_value=0)

    if counts is None:
        return pd.Series(dtype=np.int64, name="obs_count")
    counts = counts.astype(np.int64)
    counts.name = "obs_count"
    return counts


def _deterministic_subsample(df: pd.DataFrame, max_points: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_points <= 0 or len(df) <= max_points:
        return df, {"subsampled": False, "n_before": len(df), "n_after": len(df)}
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(df), size=max_points, replace=False))
    return df.iloc[indices].reset_index(drop=True), {
        "subsampled": True,
        "n_before": len(df),
        "n_after": max_points,
        "seed": seed,
    }


def load_semidense_points(
    path: str | Path,
    observations_path: str | Path | None = None,
    max_points: int | None = None,
    random_seed: int = 13,
) -> PointCloudData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Semi-dense points file does not exist: {path}")

    metadata: dict[str, Any] = {"loader": "projectaria_tools"}
    df = _load_with_projectaria(path)
    if df is None:
        metadata["loader"] = "pandas_csv"
        df = _load_with_pandas(path)
    elif "uid" in df.columns:
        extras = _load_extra_csv_columns(path, df["uid"])
        if extras is not None:
            df = df.merge(extras, on="uid", how="left")
            _synthesize_obs_count(df)
            metadata["preserved_csv_columns"] = [c for c in extras.columns if c != "uid"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

    if observations_path is not None:
        counts = load_observation_counts(observations_path)
        df = df.merge(counts.rename("obs_count"), left_on="uid", right_index=True, how="left")
        df["obs_count"] = df["obs_count"].fillna(0).astype(np.int64)
        metadata["observations_path"] = str(observations_path)

    if max_points is not None:
        df, subsample_info = _deterministic_subsample(df, int(max_points), random_seed)
        metadata["max_points_subsample"] = subsample_info
        if subsample_info.get("subsampled"):
            LOGGER.info(
                "Subsampled %s from %d to %d points with seed %d",
                path,
                subsample_info["n_before"],
                subsample_info["n_after"],
                subsample_info["seed"],
            )

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
    return PointCloudData(
        points_df=df,
        xyz=xyz,
        frame_id=_infer_frame_id(df),
        source_path=path,
        metadata=metadata,
    )


def write_minimal_semidense_csv_gz(path: str | Path, xyz: np.ndarray, uid_start: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=float)
    df = pd.DataFrame(
        {
            "uid": np.arange(uid_start, uid_start + len(xyz), dtype=np.int64),
            "px_world": xyz[:, 0],
            "py_world": xyz[:, 1],
            "pz_world": xyz[:, 2],
            "inv_dist_std": np.full(len(xyz), 0.001),
            "dist_std": np.full(len(xyz), 0.002),
            "graph_uid": "synthetic",
        }
    )
    df.to_csv(path, index=False, compression="gzip")


def write_minimal_observations_csv_gz(path: str | Path, n_points: int, uid_start: int = 0, repeats: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uids = np.repeat(np.arange(uid_start, uid_start + n_points, dtype=np.int64), repeats)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("uid,frame_id\n")
        for i, uid in enumerate(uids):
            f.write(f"{uid},{i % repeats}\n")
