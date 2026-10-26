from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tomato_cutpoints.preprocess.downsample import voxel_downsample_indices
from tomato_cutpoints.registration.transforms import apply_transform, best_fit_transform, identity_transform

LOGGER = logging.getLogger("tomato_cutpoints")


def _open3d_icp(
    pre_xyz: np.ndarray,
    post_xyz: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return None

    reg_cfg = config.get("registration", config)
    voxel_size = float(reg_cfg.get("voxel_size", 0.01))
    max_corr = float(reg_cfg.get("icp_max_correspondence_distance", 0.03))
    max_iter = int(reg_cfg.get("icp_max_iterations", 80))

    if len(pre_xyz) < 10 or len(post_xyz) < 10:
        return None

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pre_xyz)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(post_xyz)

    if voxel_size > 0:
        target = target.voxel_down_sample(voxel_size)
        source = source.voxel_down_sample(voxel_size)

    if len(target.points) < 10 or len(source.points) < 10:
        return None

    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 3, 0.03), max_nn=30))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 3, 0.03), max_nn=30))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)

    try:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_corr, identity_transform(), estimation, criteria
        )
        method = "open3d_point_to_plane"
    except Exception:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_corr, identity_transform(), estimation, criteria
        )
        method = "open3d_point_to_point"

    diagnostics = {
        "method": method,
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "n_pre_icp": int(len(target.points)),
        "n_post_icp": int(len(source.points)),
        "warning_flags": [],
    }
    return np.asarray(result.transformation, dtype=np.float64), diagnostics


def _numpy_icp(pre_xyz: np.ndarray, post_xyz: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    reg_cfg = config.get("registration", config)
    voxel_size = float(reg_cfg.get("voxel_size", 0.01))
    max_corr = float(reg_cfg.get("icp_max_correspondence_distance", 0.03))
    max_iter = int(reg_cfg.get("icp_max_iterations", 80))

    pre_idx = voxel_downsample_indices(pre_xyz, voxel_size)
    post_idx = voxel_downsample_indices(post_xyz, voxel_size)
    target = pre_xyz[pre_idx]
    source0 = post_xyz[post_idx]

    warnings: list[str] = []
    if len(target) < 3 or len(source0) < 3:
        warnings.append("not_enough_points_for_icp")
        return identity_transform(), {
            "method": "numpy_point_to_point",
            "fitness": 0.0,
            "inlier_rmse": float("inf"),
            "n_pre_icp": int(len(target)),
            "n_post_icp": int(len(source0)),
            "warning_flags": warnings,
        }

    tree = cKDTree(target)
    transform = identity_transform()
    prev_rmse = float("inf")
    inlier_rmse = float("inf")
    fitness = 0.0

    for _ in range(max_iter):
        source = apply_transform(source0, transform)
        dists, indices = tree.query(source, k=1, workers=-1)
        inlier_mask = dists <= max_corr
        if inlier_mask.sum() < 3:
            warnings.append("too_few_icp_correspondences")
            break
        delta = best_fit_transform(source[inlier_mask], target[indices[inlier_mask]])
        transform = delta @ transform
        inlier_rmse = float(np.sqrt(np.mean(dists[inlier_mask] ** 2)))
        fitness = float(inlier_mask.sum() / len(source0))
        if abs(prev_rmse - inlier_rmse) < 1e-7:
            break
        prev_rmse = inlier_rmse

    return transform, {
        "method": "numpy_point_to_point",
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "n_pre_icp": int(len(target)),
        "n_post_icp": int(len(source0)),
        "warning_flags": warnings,
    }


def refine_post_to_pre_icp(
    pre_xyz: np.ndarray,
    post_xyz_aligned_initial: np.ndarray,
    config: dict[str, Any],
    use_open3d: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an incremental transform mapping initially aligned post points into pre coordinates."""
    pre_xyz = np.asarray(pre_xyz, dtype=np.float64)
    post_xyz_aligned_initial = np.asarray(post_xyz_aligned_initial, dtype=np.float64)

    if use_open3d:
        result = _open3d_icp(pre_xyz, post_xyz_aligned_initial, config)
        if result is not None:
            transform, diagnostics = result
        else:
            transform, diagnostics = _numpy_icp(pre_xyz, post_xyz_aligned_initial, config)
            diagnostics["warning_flags"].append("open3d_unavailable_or_failed")
    else:
        transform, diagnostics = _numpy_icp(pre_xyz, post_xyz_aligned_initial, config)

    reg_cfg = config.get("registration", config)
    warnings = diagnostics.setdefault("warning_flags", [])
    if diagnostics.get("fitness", 0.0) < float(reg_cfg.get("min_icp_fitness", 0.3)):
        warnings.append("low_icp_fitness")
    if diagnostics.get("inlier_rmse", float("inf")) > float(reg_cfg.get("max_icp_rmse", 0.03)):
        warnings.append("high_icp_rmse")

    LOGGER.info(
        "ICP %s fitness=%.3f rmse=%.4f using %d/%d points",
        diagnostics.get("method"),
        diagnostics.get("fitness", 0.0),
        diagnostics.get("inlier_rmse", float("inf")),
        diagnostics.get("n_post_icp", 0),
        diagnostics.get("n_pre_icp", 0),
    )
    return transform, diagnostics
