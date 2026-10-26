from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import open3d as o3d


def find_xyz_columns(df):
    cols = {c.strip().lower(): c for c in df.columns}
    for xs, ys, zs in [
        ("px_world", "py_world", "pz_world"),
        ("p_x_world", "p_y_world", "p_z_world"),
        ("x_world", "y_world", "z_world"),
        ("x", "y", "z"),
    ]:
        if xs in cols and ys in cols and zs in cols:
            return cols[xs], cols[ys], cols[zs]
    raise ValueError(f"Could not find XYZ columns. Columns: {df.columns.tolist()}")


def update_uid_counts(counts, uids):
    vc = uids.value_counts()
    for uid, c in vc.items():
        counts[int(uid)] = counts.get(int(uid), 0) + int(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--pre-start-s", type=float, required=True)
    parser.add_argument("--pre-end-s", type=float, required=True)
    parser.add_argument("--post-start-s", type=float, required=True)
    parser.add_argument("--post-end-s", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-obs", type=int, default=1)
    args = parser.parse_args()

    points_path = Path(args.points)
    obs_path = Path(args.observations)

    print("Reading points...")
    points = pd.read_csv(points_path, compression="gzip")
    xcol, ycol, zcol = find_xyz_columns(points)

    print("Finding observation timestamp origin...")
    t0 = None
    for chunk in pd.read_csv(obs_path, compression="gzip", chunksize=1_000_000):
        tmin = int(chunk["frame_tracking_timestamp_us"].min())
        t0 = tmin if t0 is None else min(t0, tmin)

    pre_start = int(t0 + args.pre_start_s * 1_000_000)
    pre_end = int(t0 + args.pre_end_s * 1_000_000)
    post_start = int(t0 + args.post_start_s * 1_000_000)
    post_end = int(t0 + args.post_end_s * 1_000_000)

    pre_counts = {}
    post_counts = {}

    print("Counting pre/post observations...")
    for chunk in pd.read_csv(obs_path, compression="gzip", chunksize=1_000_000):
        t = chunk["frame_tracking_timestamp_us"]

        pre_uids = chunk.loc[(t >= pre_start) & (t <= pre_end), "uid"]
        post_uids = chunk.loc[(t >= post_start) & (t <= post_end), "uid"]

        update_uid_counts(pre_counts, pre_uids)
        update_uid_counts(post_counts, post_uids)

    points["pre_obs"] = points["uid"].map(pre_counts).fillna(0).astype(int)
    points["post_obs"] = points["uid"].map(post_counts).fillna(0).astype(int)

    keep = (points["pre_obs"] >= args.min_obs) | (points["post_obs"] >= args.min_obs)
    points = points.loc[keep].reset_index(drop=True)

    xyz = points[[xcol, ycol, zcol]].to_numpy(dtype=np.float64)

    colors = np.zeros((len(points), 3), dtype=np.float64)

    pre = points["pre_obs"].to_numpy() >= args.min_obs
    post = points["post_obs"].to_numpy() >= args.min_obs

    both = pre & post
    pre_only = pre & ~post
    post_only = post & ~pre

    colors[both] = [1.0, 1.0, 1.0]
    colors[pre_only] = [1.0, 0.0, 0.0]
    colors[post_only] = [0.0, 1.0, 0.0]

    print(f"points kept: {len(points):,}")
    print(f"both:      {both.sum():,}")
    print(f"pre only:  {pre_only.sum():,}")
    print(f"post only: {post_only.sum():,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()