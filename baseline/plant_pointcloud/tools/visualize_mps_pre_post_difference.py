from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree


def normalize_col(c: str) -> str:
    return (
        str(c)
        .strip()
        .replace("#", "")
        .replace("[m]", "")
        .replace("(m)", "")
        .replace(" ", "")
        .lower()
    )


def find_col(columns, candidates):
    mapping = {normalize_col(c): c for c in columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in mapping:
            return mapping[key]
    raise ValueError(
        f"Could not find any of {candidates}. Available columns: {list(columns)}"
    )


def find_xyz_columns(df: pd.DataFrame):
    candidates = [
        ("px_world", "py_world", "pz_world"),
        ("p_x_world", "p_y_world", "p_z_world"),
        ("x_world", "y_world", "z_world"),
        ("position_world_x", "position_world_y", "position_world_z"),
        ("x", "y", "z"),
    ]

    mapping = {normalize_col(c): c for c in df.columns}

    for xs, ys, zs in candidates:
        if normalize_col(xs) in mapping and normalize_col(ys) in mapping and normalize_col(zs) in mapping:
            return mapping[normalize_col(xs)], mapping[normalize_col(ys)], mapping[normalize_col(zs)]

    raise ValueError(f"Could not find XYZ columns. Available columns: {df.columns.tolist()}")


def add_uid_counts(dst: dict[int, int], uid_series: pd.Series) -> None:
    uid_series = pd.to_numeric(uid_series, errors="coerce").dropna().astype(np.int64)
    counts = uid_series.value_counts()
    for uid, count in counts.items():
        uid = int(uid)
        dst[uid] = dst.get(uid, 0) + int(count)


def get_observation_columns(obs_path: Path):
    first = next(pd.read_csv(obs_path, compression="gzip", chunksize=10))
    uid_col = find_col(first.columns, ["uid", "point_uid"])
    ts_col = find_col(
        first.columns,
        [
            "frame_tracking_timestamp_us",
            "tracking_timestamp_us",
            "frame_capture_timestamp_us",
            "timestamp_us",
        ],
    )
    return uid_col, ts_col


def find_observation_t0_us(obs_path: Path, ts_col: str) -> int:
    t0 = None

    for chunk in pd.read_csv(obs_path, compression="gzip", chunksize=1_000_000):
        t = pd.to_numeric(chunk[ts_col], errors="coerce").dropna()
        if len(t) == 0:
            continue
        t_min = int(t.min())
        t0 = t_min if t0 is None else min(t0, t_min)

    if t0 is None:
        raise RuntimeError("Could not find any valid observation timestamps.")

    return int(t0)


def count_pre_post_observations(
    obs_path: Path,
    pre_start_us: int,
    pre_end_us: int,
    post_start_us: int,
    post_end_us: int,
):
    uid_col, ts_col = get_observation_columns(obs_path)

    pre_counts: dict[int, int] = {}
    post_counts: dict[int, int] = {}

    for chunk in pd.read_csv(obs_path, compression="gzip", chunksize=1_000_000):
        t = pd.to_numeric(chunk[ts_col], errors="coerce")

        pre_mask = (t >= pre_start_us) & (t <= pre_end_us)
        post_mask = (t >= post_start_us) & (t <= post_end_us)

        add_uid_counts(pre_counts, chunk.loc[pre_mask, uid_col])
        add_uid_counts(post_counts, chunk.loc[post_mask, uid_col])

    return pre_counts, post_counts


def write_point_cloud(path: Path, xyz: np.ndarray, colors: np.ndarray | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        colors = colors[finite]
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_point_cloud(str(path), pcd)
    if not ok:
        raise RuntimeError(f"Open3D failed to write {path}")

    print(f"Wrote {len(xyz):,} points: {path}")


def write_marker_spheres(
    path: Path,
    xyz: np.ndarray,
    color,
    radius_m: float,
    max_markers: int,
    seed: int = 13,
):
    if radius_m <= 0 or len(xyz) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    if len(xyz) > max_markers:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xyz), size=max_markers, replace=False)
        xyz = xyz[keep]

    mesh = o3d.geometry.TriangleMesh()

    for p in xyz:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius_m, resolution=6)
        sphere.translate(p)
        sphere.paint_uniform_color(color)
        mesh += sphere

    ok = o3d.io.write_triangle_mesh(str(path), mesh)
    if not ok:
        raise RuntimeError(f"Open3D failed to write {path}")

    print(f"Wrote {len(xyz):,} marker spheres: {path}")


def nn_distance(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    if len(src_xyz) == 0:
        return np.array([], dtype=np.float64)

    if len(dst_xyz) == 0:
        return np.full(len(src_xyz), np.inf, dtype=np.float64)

    tree = cKDTree(dst_xyz)
    d, _ = tree.query(src_xyz, k=1)
    return d.astype(np.float64)


def color_by_nn_distance(d: np.ndarray, threshold_m: float) -> np.ndarray:
    """
    Gray = persistent / close to post cloud.
    Yellow -> red = likely removed / far from post cloud.
    """
    colors = np.zeros((len(d), 3), dtype=np.float64)
    colors[:] = [0.55, 0.55, 0.55]

    finite = np.isfinite(d)
    candidate = finite & (d > threshold_m)

    if candidate.any():
        high = max(np.percentile(d[candidate], 95), threshold_m * 2.0)
        denom = max(high - threshold_m, 1e-9)

        t = np.clip((d[candidate] - threshold_m) / denom, 0.0, 1.0)

        # yellow to red
        colors[candidate, 0] = 1.0
        colors[candidate, 1] = 1.0 - t
        colors[candidate, 2] = 0.0

    colors[~finite] = [1.0, 0.0, 1.0]
    return colors


def stats_for_array(x: np.ndarray):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}

    return {
        "count": int(len(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def make_crop_mask(
    xyz: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    roi_center,
    roi_radius,
    roi_min,
    roi_max,
    crop_percentile: float,
):
    finite = np.isfinite(xyz).all(axis=1)
    crop = finite.copy()

    if roi_center is not None:
        center = np.array(roi_center, dtype=np.float64)
        radius = float(roi_radius)
        crop &= np.linalg.norm(xyz - center, axis=1) <= radius

    if roi_min is not None and roi_max is not None:
        mn = np.array(roi_min, dtype=np.float64)
        mx = np.array(roi_max, dtype=np.float64)
        crop &= np.all((xyz >= mn) & (xyz <= mx), axis=1)

    # Percentile outlier removal, computed from the selected pre/post points.
    # This prevents a few far outliers from making MeshLab zoom out too much.
    if crop_percentile < 100.0:
        ref = crop & (pre_mask | post_mask)
        if ref.sum() > 10:
            center = np.median(xyz[ref], axis=0)
            dist = np.linalg.norm(xyz - center, axis=1)
            threshold = np.percentile(dist[ref], crop_percentile)
            crop &= dist <= threshold

    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="semidense_points.csv.gz")
    parser.add_argument("--observations", required=True, help="semidense_observations.csv.gz")
    parser.add_argument("--pre-start-s", type=float, required=True)
    parser.add_argument("--pre-end-s", type=float, required=True)
    parser.add_argument("--post-start-s", type=float, required=True)
    parser.add_argument("--post-end-s", type=float, required=True)
    parser.add_argument("--min-obs", type=int, default=1)
    parser.add_argument("--nn-threshold-m", type=float, default=0.025)
    parser.add_argument("--crop-percentile", type=float, default=99.5)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument(
        "--roi-center",
        nargs=3,
        type=float,
        default=None,
        help="Optional tomato ROI center: x y z in MPS world coordinates.",
    )
    parser.add_argument(
        "--roi-radius",
        type=float,
        default=0.25,
        help="ROI radius in meters when --roi-center is used.",
    )
    parser.add_argument(
        "--roi-min",
        nargs=3,
        type=float,
        default=None,
        help="Optional ROI box minimum: x y z.",
    )
    parser.add_argument(
        "--roi-max",
        nargs=3,
        type=float,
        default=None,
        help="Optional ROI box maximum: x y z.",
    )
    parser.add_argument(
        "--marker-radius-m",
        type=float,
        default=0.0,
        help="If > 0, write sphere-marker meshes for removed/added candidates.",
    )
    parser.add_argument("--max-marker-spheres", type=int, default=2000)

    args = parser.parse_args()

    points_path = Path(args.points)
    obs_path = Path(args.observations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading points...")
    points = pd.read_csv(points_path, compression="gzip")

    uid_col = find_col(points.columns, ["uid", "point_uid"])
    xcol, ycol, zcol = find_xyz_columns(points)

    points_uid = pd.to_numeric(points[uid_col], errors="coerce")
    valid_uid = points_uid.notna()
    points = points.loc[valid_uid].reset_index(drop=True)
    points_uid = points_uid.loc[valid_uid].astype(np.int64).to_numpy()

    xyz = points[[xcol, ycol, zcol]].to_numpy(dtype=np.float64)
    finite_xyz = np.isfinite(xyz).all(axis=1)

    print("Finding observation timestamp origin...")
    _, ts_col = get_observation_columns(obs_path)
    t0_us = find_observation_t0_us(obs_path, ts_col)

    pre_start_us = int(t0_us + args.pre_start_s * 1_000_000)
    pre_end_us = int(t0_us + args.pre_end_s * 1_000_000)
    post_start_us = int(t0_us + args.post_start_s * 1_000_000)
    post_end_us = int(t0_us + args.post_end_s * 1_000_000)

    print("Counting pre/post observations...")
    pre_counts, post_counts = count_pre_post_observations(
        obs_path=obs_path,
        pre_start_us=pre_start_us,
        pre_end_us=pre_end_us,
        post_start_us=post_start_us,
        post_end_us=post_end_us,
    )

    pre_obs = np.array([pre_counts.get(int(uid), 0) for uid in points_uid], dtype=np.int32)
    post_obs = np.array([post_counts.get(int(uid), 0) for uid in points_uid], dtype=np.int32)

    pre_mask = finite_xyz & (pre_obs >= args.min_obs)
    post_mask = finite_xyz & (post_obs >= args.min_obs)
    both_mask = pre_mask & post_mask
    pre_only_mask = pre_mask & ~post_mask
    post_only_mask = post_mask & ~pre_mask

    crop_mask = make_crop_mask(
        xyz=xyz,
        pre_mask=pre_mask,
        post_mask=post_mask,
        roi_center=args.roi_center,
        roi_radius=args.roi_radius,
        roi_min=args.roi_min,
        roi_max=args.roi_max,
        crop_percentile=args.crop_percentile,
    )

    print("\nCounts before crop:")
    print(f"  full points: {len(xyz):,}")
    print(f"  pre:         {pre_mask.sum():,}")
    print(f"  post:        {post_mask.sum():,}")
    print(f"  both:        {both_mask.sum():,}")
    print(f"  pre only:    {pre_only_mask.sum():,}")
    print(f"  post only:   {post_only_mask.sum():,}")

    print("\nCounts after crop/outlier removal:")
    print(f"  crop points: {(crop_mask & finite_xyz).sum():,}")
    print(f"  pre:         {(crop_mask & pre_mask).sum():,}")
    print(f"  post:        {(crop_mask & post_mask).sum():,}")
    print(f"  both:        {(crop_mask & both_mask).sum():,}")
    print(f"  pre only:    {(crop_mask & pre_only_mask).sum():,}")
    print(f"  post only:   {(crop_mask & post_only_mask).sum():,}")

    # 1. Context overlay:
    # gray = full cropped MPS context
    # red = pre only
    # green = post only
    # white = both
    context_mask = crop_mask & finite_xyz
    context_xyz = xyz[context_mask]

    context_colors = np.zeros((context_mask.sum(), 3), dtype=np.float64)
    context_colors[:] = [0.22, 0.22, 0.22]

    local_pre_only = pre_only_mask[context_mask]
    local_post_only = post_only_mask[context_mask]
    local_both = both_mask[context_mask]

    context_colors[local_pre_only] = [1.0, 0.0, 0.0]
    context_colors[local_post_only] = [0.0, 1.0, 0.0]
    context_colors[local_both] = [1.0, 1.0, 1.0]

    write_point_cloud(
        out_dir / "01_overlay_context_gray_pre_red_post_green_both_white.ply",
        context_xyz,
        context_colors,
    )

    # 2. Selected pre/post only overlay.
    selected_mask = crop_mask & (pre_mask | post_mask)
    selected_xyz = xyz[selected_mask]

    selected_colors = np.zeros((selected_mask.sum(), 3), dtype=np.float64)
    selected_colors[:] = [0.5, 0.5, 0.5]
    selected_colors[pre_only_mask[selected_mask]] = [1.0, 0.0, 0.0]
    selected_colors[post_only_mask[selected_mask]] = [0.0, 1.0, 0.0]
    selected_colors[both_mask[selected_mask]] = [1.0, 1.0, 1.0]

    write_point_cloud(
        out_dir / "02_overlay_selected_only_pre_red_post_green_both_white.ply",
        selected_xyz,
        selected_colors,
    )

    # 3. NN-based pre-to-post removal visualization.
    pre_idx = np.where(crop_mask & pre_mask)[0]
    post_idx = np.where(crop_mask & post_mask)[0]

    pre_xyz = xyz[pre_idx]
    post_xyz = xyz[post_idx]

    d_pre_to_post = nn_distance(pre_xyz, post_xyz)
    removed_candidate = d_pre_to_post > args.nn_threshold_m

    pre_nn_colors = color_by_nn_distance(d_pre_to_post, args.nn_threshold_m)

    write_point_cloud(
        out_dir / "03_pre_points_colored_by_distance_to_post_gray_persistent_yellow_red_removed.ply",
        pre_xyz,
        pre_nn_colors,
    )

    write_point_cloud(
        out_dir / "04_removed_candidates_only_pre_far_from_post_red.ply",
        pre_xyz[removed_candidate],
        np.tile(np.array([[1.0, 0.0, 0.0]]), (removed_candidate.sum(), 1)),
    )

    # 4. NN-based post-to-pre addition visualization.
    d_post_to_pre = nn_distance(post_xyz, pre_xyz)
    added_candidate = d_post_to_pre > args.nn_threshold_m

        # 7. Geometry-based combined overlay.
    # This is usually more meaningful than exact UID overlap.
    # white = pre point has nearby post point
    # red   = pre point is far from post point, candidate removed
    # green = post point is far from pre point, candidate added/newly observed

    persistent_pre = d_pre_to_post <= args.nn_threshold_m
    removed_pre = d_pre_to_post > args.nn_threshold_m
    added_post = d_post_to_pre > args.nn_threshold_m

    geo_xyz_parts = []
    geo_color_parts = []

    if persistent_pre.any():
        geo_xyz_parts.append(pre_xyz[persistent_pre])
        geo_color_parts.append(
            np.tile(np.array([[1.0, 1.0, 1.0]]), (persistent_pre.sum(), 1))
        )

    if removed_pre.any():
        geo_xyz_parts.append(pre_xyz[removed_pre])
        geo_color_parts.append(
            np.tile(np.array([[1.0, 0.0, 0.0]]), (removed_pre.sum(), 1))
        )

    if added_post.any():
        geo_xyz_parts.append(post_xyz[added_post])
        geo_color_parts.append(
            np.tile(np.array([[0.0, 1.0, 0.0]]), (added_post.sum(), 1))
        )

    if geo_xyz_parts:
        geo_xyz = np.vstack(geo_xyz_parts)
        geo_colors = np.vstack(geo_color_parts)

        write_point_cloud(
            out_dir / "09_geometric_overlay_white_persistent_red_removed_green_added.ply",
            geo_xyz,
            geo_colors,
        )

    # Optional marker spheres for geometry-based removed/added points.
    if args.marker_radius_m > 0:
        write_marker_spheres(
            out_dir / "10_geometric_removed_red_spheres.ply",
            pre_xyz[removed_pre],
            color=[1.0, 0.0, 0.0],
            radius_m=args.marker_radius_m,
            max_markers=args.max_marker_spheres,
        )

        write_marker_spheres(
            out_dir / "11_geometric_added_green_spheres.ply",
            post_xyz[added_post],
            color=[0.0, 1.0, 0.0],
            radius_m=args.marker_radius_m,
            max_markers=args.max_marker_spheres,
        )

    post_nn_colors = color_by_nn_distance(d_post_to_pre, args.nn_threshold_m)

    write_point_cloud(
        out_dir / "05_post_points_colored_by_distance_to_pre_gray_persistent_yellow_green_added.ply",
        post_xyz,
        post_nn_colors,
    )

    # recolor candidates to green for post additions
    added_colors = np.tile(np.array([[0.0, 1.0, 0.0]]), (added_candidate.sum(), 1))
    write_point_cloud(
        out_dir / "06_added_candidates_only_post_far_from_pre_green.ply",
        post_xyz[added_candidate],
        added_colors,
    )

    # Optional sphere markers for MeshLab visibility.
    if args.marker_radius_m > 0:
        write_marker_spheres(
            out_dir / "07_removed_candidate_markers_red_spheres.ply",
            pre_xyz[removed_candidate],
            color=[1.0, 0.0, 0.0],
            radius_m=args.marker_radius_m,
            max_markers=args.max_marker_spheres,
        )

        write_marker_spheres(
            out_dir / "08_added_candidate_markers_green_spheres.ply",
            post_xyz[added_candidate],
            color=[0.0, 1.0, 0.0],
            radius_m=args.marker_radius_m,
            max_markers=args.max_marker_spheres,
        )

    summary = {
        "time_windows_seconds": {
            "pre": [args.pre_start_s, args.pre_end_s],
            "post": [args.post_start_s, args.post_end_s],
        },
        "min_obs": args.min_obs,
        "nn_threshold_m": args.nn_threshold_m,
        "crop_percentile": args.crop_percentile,
        "roi_center": args.roi_center,
        "roi_radius": args.roi_radius if args.roi_center is not None else None,
        "roi_min": args.roi_min,
        "roi_max": args.roi_max,
        "counts_before_crop": {
            "full_points": int(len(xyz)),
            "pre": int(pre_mask.sum()),
            "post": int(post_mask.sum()),
            "both": int(both_mask.sum()),
            "pre_only": int(pre_only_mask.sum()),
            "post_only": int(post_only_mask.sum()),
        },
        "counts_after_crop": {
            "crop_points": int((crop_mask & finite_xyz).sum()),
            "pre": int((crop_mask & pre_mask).sum()),
            "post": int((crop_mask & post_mask).sum()),
            "both": int((crop_mask & both_mask).sum()),
            "pre_only": int((crop_mask & pre_only_mask).sum()),
            "post_only": int((crop_mask & post_only_mask).sum()),
        },
        "pre_to_post_nn_distance_stats_m": stats_for_array(d_pre_to_post),
        "post_to_pre_nn_distance_stats_m": stats_for_array(d_post_to_pre),
        "removed_candidates_pre_far_from_post": int(removed_candidate.sum()),
        "added_candidates_post_far_from_pre": int(added_candidate.sum()),
        "outputs": [
            "01_overlay_context_gray_pre_red_post_green_both_white.ply",
            "02_overlay_selected_only_pre_red_post_green_both_white.ply",
            "03_pre_points_colored_by_distance_to_post_gray_persistent_yellow_red_removed.ply",
            "04_removed_candidates_only_pre_far_from_post_red.ply",
            "05_post_points_colored_by_distance_to_pre_gray_persistent_yellow_green_added.ply",
            "06_added_candidates_only_post_far_from_pre_green.ply",
            "07_removed_candidate_markers_red_spheres.ply",
            "08_added_candidate_markers_green_spheres.ply",
        ],
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()