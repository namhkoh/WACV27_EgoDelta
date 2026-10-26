from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import open3d as o3d


def normalize_col(c: str) -> str:
    return (
        c.strip()
        .replace("#", "")
        .replace("[m]", "")
        .replace(" ", "")
        .lower()
    )


def find_xyz_columns(df: pd.DataFrame):
    norm_to_orig = {normalize_col(c): c for c in df.columns}

    candidates = [
        ("px_world", "py_world", "pz_world"),
        ("p_x_world", "p_y_world", "p_z_world"),
        ("x_world", "y_world", "z_world"),
        ("position_world_x", "position_world_y", "position_world_z"),
        ("x", "y", "z"),
    ]

    for xs, ys, zs in candidates:
        if xs in norm_to_orig and ys in norm_to_orig and zs in norm_to_orig:
            return norm_to_orig[xs], norm_to_orig[ys], norm_to_orig[zs]

    print("Available columns:")
    for c in df.columns:
        print(f"  {repr(c)} -> normalized={normalize_col(c)}")

    raise ValueError("Could not find XYZ columns in the semidense point file.")


def color_by_scalar(values: np.ndarray):
    """
    Simple blue-to-red scalar coloring.
    Lower values are blue-ish; higher values are red-ish.
    """
    values = values.astype(np.float64)
    finite = np.isfinite(values)

    colors = np.zeros((len(values), 3), dtype=np.float64)
    colors[:] = np.array([0.7, 0.7, 0.7])

    if finite.sum() == 0:
        return colors

    lo = np.nanpercentile(values[finite], 5)
    hi = np.nanpercentile(values[finite], 95)
    denom = max(hi - lo, 1e-12)

    t = np.clip((values - lo) / denom, 0.0, 1.0)
    colors[:, 0] = t
    colors[:, 1] = 0.2
    colors[:, 2] = 1.0 - t
    colors[~finite] = np.array([0.5, 0.5, 0.5])

    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="Path to semidense_points.csv.gz")
    parser.add_argument("--out", required=True, help="Output PLY path")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="0 means export all points. Otherwise deterministic random subsample.",
    )
    parser.add_argument(
        "--color-by",
        default="none",
        choices=["none", "dist_std", "inv_dist_std"],
        help="Optional uncertainty coloring.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    points_path = Path(args.points)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {points_path}")
    df = pd.read_csv(points_path, compression="gzip")

    xcol, ycol, zcol = find_xyz_columns(df)
    xyz = df[[xcol, ycol, zcol]].to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.all():
        print(f"Removing non-finite XYZ rows: {(~finite_mask).sum():,}")
        df = df.loc[finite_mask].reset_index(drop=True)
        xyz = xyz[finite_mask]

    print(f"Raw valid points: {len(xyz):,}")

    if args.max_points and len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(xyz), size=args.max_points, replace=False)
        keep.sort()
        df = df.iloc[keep].reset_index(drop=True)
        xyz = xyz[keep]
        print(f"Subsampled for visualization: {len(xyz):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if args.color_by != "none":
        if args.color_by not in df.columns:
            print(f"WARNING: {args.color_by} not found. Writing uncolored PLY.")
        else:
            colors = color_by_scalar(df[args.color_by].to_numpy())
            pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Writing: {out_path}")
    ok = o3d.io.write_point_cloud(str(out_path), pcd)
    if not ok:
        raise RuntimeError(f"Open3D failed to write {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()