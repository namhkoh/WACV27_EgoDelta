#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd


def load_semidense(mps_dir: Path, inv_max: float, dist_max: float) -> np.ndarray:
    path = mps_dir / "slam" / "semidense_points.csv.gz"
    df = pd.read_csv(path)
    keep = (df["inv_dist_std"] <= inv_max) & (df["dist_std"] <= dist_max)
    return df.loc[keep, ["px_world", "py_world", "pz_world"]].to_numpy(np.float64)


def to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def prep_features(pcd: o3d.geometry.PointCloud, voxel: float):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return down, fpfh


def align_to_pre(src_pts: np.ndarray, tgt_pts: np.ndarray, coarse_voxel: float) -> tuple[np.ndarray, dict]:
    """Return T such that column-vector X_pre = T * X_src."""

    src = to_pcd(src_pts)
    tgt = to_pcd(tgt_pts)
    src_down, src_fpfh = prep_features(src, coarse_voxel)
    tgt_down, tgt_fpfh = prep_features(tgt, coarse_voxel)

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=coarse_voxel * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(coarse_voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
    )
    transform = ransac.transformation

    icp_diag = None
    for voxel in (0.02, 0.01):
        src_ref = src.voxel_down_sample(voxel)
        tgt_ref = tgt.voxel_down_sample(voxel)
        tgt_ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        icp = o3d.pipelines.registration.registration_icp(
            src_ref,
            tgt_ref,
            voxel * 2.0,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        transform = icp.transformation
        icp_diag = icp

    diag = {
        "ransac_fitness": float(ransac.fitness),
        "ransac_inlier_rmse_m": float(ransac.inlier_rmse),
        "icp_fitness": float(icp_diag.fitness),
        "icp_inlier_rmse_m": float(icp_diag.inlier_rmse),
    }
    return np.asarray(transform, dtype=np.float64), diag


def read_camera_summary(dataset_dir: Path) -> dict:
    cameras_txt = dataset_dir / "sparse" / "0" / "cameras.txt"
    with cameras_txt.open() as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[1] != "PINHOLE":
                break
            return {
                "width": int(parts[2]),
                "height": int(parts[3]),
                "fx": float(parts[4]),
                "fy": float(parts[5]),
                "cx": float(parts[6]),
                "cy": float(parts[7]),
            }
    raise RuntimeError(f"Could not read PINHOLE camera from {cameras_txt}")


def count_images(dataset_dir: Path) -> int:
    images_txt = dataset_dir / "sparse" / "0" / "images.txt"
    count = 0
    with images_txt.open() as fp:
        for line in fp:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def dataset_entry(task: str, label: str, dataset_dir: Path, mps_dir: Path) -> dict:
    return {
        "name": f"{task}_{label}",
        "task": task,
        "label": label,
        "dataset_dir": str(dataset_dir.resolve()),
        "single_trajectory": False,
        "source_root": str(mps_dir.resolve()),
        "camera": read_camera_summary(dataset_dir),
        "num_images_with_pose": count_images(dataset_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a pre/post manifest for separate Aria trajectories")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--pre_dataset", type=Path, required=True)
    parser.add_argument("--post_dataset", type=Path, required=True)
    parser.add_argument("--pre_mps", type=Path, required=True)
    parser.add_argument("--post_mps", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, default=None)
    parser.add_argument("--inv_dist_std_max", type=float, default=0.05)
    parser.add_argument("--dist_std_max", type=float, default=0.10)
    parser.add_argument("--coarse_voxel", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    baseline_root = args.baseline_root or (args.data_root / "alignment_baseline")
    alignment_dir = baseline_root / "out" / args.task
    alignment_dir.mkdir(parents=True, exist_ok=True)

    pre_pts = load_semidense(args.pre_mps, args.inv_dist_std_max, args.dist_std_max)
    post_pts = load_semidense(args.post_mps, args.inv_dist_std_max, args.dist_std_max)
    t_pre_post, diag = align_to_pre(post_pts, pre_pts, args.coarse_voxel)

    alignment_manifest = {
        "task": args.task,
        "pre_mps": str(args.pre_mps.resolve()),
        "post_mps": str(args.post_mps.resolve()),
        "alignment_method": (
            f"fpfh_ransac_fit={diag['ransac_fitness']:.3f}_rmse={diag['ransac_inlier_rmse_m']:.4f}|"
            f"icp_fit={diag['icp_fitness']:.3f}_rmse={diag['icp_inlier_rmse_m']:.4f}"
        ),
        "n_pre_semidense": int(len(pre_pts)),
        "n_post_semidense": int(len(post_pts)),
        "diagnostics": diag,
        "T_pre_post": t_pre_post.tolist(),
    }
    alignment_path = alignment_dir / f"manifest_{args.task}.json"
    alignment_path.write_text(json.dumps(alignment_manifest, indent=2) + "\n")

    manifest = {
        "single_trajectory": False,
        "alignment": "separate_trajectory",
        "alignment_method": alignment_manifest["alignment_method"],
        "baseline_root": str(baseline_root.resolve()),
        "alignment_manifest": str(alignment_path.resolve()),
        "T_pre_post": t_pre_post.tolist(),
        "datasets": [
            dataset_entry(args.task, "pre", args.pre_dataset, args.pre_mps),
            dataset_entry(args.task, "post", args.post_dataset, args.post_mps),
        ],
    }
    manifest_path = args.data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "manifest": str(manifest_path),
        "alignment_manifest": str(alignment_path),
        "alignment_method": alignment_manifest["alignment_method"],
        "T_pre_post": t_pre_post.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
