from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tomato_cutpoints.change.density import compute_density_drop_score
from tomato_cutpoints.change.m3c2_distance import M3C2Result, m3c2_candidate_mask, run_m3c2
from tomato_cutpoints.change.nn_distance import auto_nn_threshold, compute_pre_to_post_nn
from tomato_cutpoints.change.scoring import combine_change_scores
from tomato_cutpoints.clustering.cluster_features import serializable_cluster_feature
from tomato_cutpoints.clustering.dbscan import cluster_removed_points
from tomato_cutpoints.clustering.split import split_merged_harvest_clusters
from tomato_cutpoints.config import load_config, save_config
from tomato_cutpoints.io.aria_mps import load_semidense_points
from tomato_cutpoints.io.export import safe_write_parquet, write_json, write_matrix_txt
from tomato_cutpoints.preprocess.quality import filter_quality
from tomato_cutpoints.preprocess.roi import apply_roi_filter, load_roi_json, merge_roi_config, stable_mask
from tomato_cutpoints.registration.icp import refine_post_to_pre_icp
from tomato_cutpoints.registration.transforms import apply_transform, identity_transform, load_transform
from tomato_cutpoints.targets.rules import build_targets_document, infer_targets, rank_clusters_for_task
from tomato_cutpoints.utils.paths import ensure_dir
from tomato_cutpoints.viz.open3d_viz import write_point_cloud_ply
from tomato_cutpoints.viz.plots import write_diagnostic_plots

LOGGER = logging.getLogger("tomato_cutpoints")
PLY_OUTPUTS = {
    "pre_filtered.ply",
    "post_filtered_aligned.ply",
    "removed_candidates.ply",
    "clustered_removed_candidates.ply",
    "target_candidates.ply",
    "cut_points.ply",
}


def _write_empty_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _remove_unrequested_ply_outputs(plys_dir: Path) -> None:
    for path in plys_dir.glob("*.ply"):
        if path.name not in PLY_OUTPUTS:
            path.unlink()


def _full_uncertainty(result: M3C2Result | None, name: str, n_points: int) -> np.ndarray | None:
    if result is None or name not in result.uncertainties:
        return None
    values = np.asarray(result.uncertainties[name])
    full = np.full(n_points, np.nan, dtype=np.float64)
    full[result.corepoint_indices[: len(values)]] = values
    return full


def _save_m3c2_outputs(out_dir: Path, result: M3C2Result | None) -> None:
    if result is None:
        return
    np.save(out_dir / "m3c2_corepoint_indices.npy", result.corepoint_indices)
    np.save(out_dir / "m3c2_distances.npy", result.distances)
    np.savez(out_dir / "m3c2_uncertainties.npz", **result.uncertainties)
    if result.normals is not None:
        np.save(out_dir / "m3c2_normals.npy", result.normals)


def _clusters_dataframe(features: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        row = serializable_cluster_feature(feature)
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_points",
                "centroid",
                "min_xyz",
                "max_xyz",
                "bbox_dimensions",
                "mean_nn_distance",
                "mean_m3c2_distance",
                "mean_combined_score",
                "pca_eigenvalues",
                "pca_eigenvectors",
                "linearity",
                "planarity",
                "compactness",
                "approximate_diameter",
            ]
        )
    return pd.DataFrame(rows)


def _attach_cluster_bbox_to_targets(targets: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    by_id = {int(c["cluster_id"]): c for c in clusters}
    for target in targets:
        cluster = by_id.get(int(target.get("cluster_id", -1)))
        if cluster is None:
            continue
        target.setdefault("diagnostics", {})["cluster_bbox"] = {
            "min": np.asarray(cluster["min_xyz"], dtype=float).tolist(),
            "max": np.asarray(cluster["max_xyz"], dtype=float).tolist(),
        }
        if "scene_robust_distance" in cluster:
            target["diagnostics"]["scene_robust_distance"] = cluster["scene_robust_distance"]
        if "split_parent_cluster_id" in cluster:
            target["diagnostics"]["split_parent_cluster_id"] = cluster["split_parent_cluster_id"]


def _cluster_colors(cluster_ids: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [0.95, 0.10, 0.08],
            [0.10, 0.62, 0.95],
            [0.18, 0.72, 0.30],
            [0.94, 0.66, 0.12],
            [0.72, 0.25, 0.85],
            [0.05, 0.70, 0.65],
            [0.85, 0.32, 0.22],
            [0.45, 0.45, 0.45],
        ],
        dtype=float,
    )
    ids = np.asarray(cluster_ids, dtype=np.int64)
    return palette[np.mod(ids, len(palette))]


def _ranked_cluster_colors(cluster_ids: np.ndarray, ranked_cluster_ids: list[int]) -> np.ndarray:
    palette = np.asarray(
        [
            [0.95, 0.10, 0.08],
            [0.10, 0.62, 0.95],
            [0.18, 0.72, 0.30],
            [0.94, 0.66, 0.12],
            [0.72, 0.25, 0.85],
            [0.00, 0.78, 0.72],
            [0.98, 0.38, 0.62],
            [0.55, 0.80, 0.16],
            [0.98, 0.55, 0.10],
            [0.58, 0.58, 0.58],
        ],
        dtype=float,
    )
    color_by_id = {int(cluster_id): palette[i % len(palette)] for i, cluster_id in enumerate(ranked_cluster_ids)}
    fallback = np.asarray([0.45, 0.45, 0.45], dtype=float)
    return np.asarray([color_by_id.get(int(cluster_id), fallback) for cluster_id in cluster_ids], dtype=float)


def _annotate_scene_context(clusters: list[dict[str, Any]], pre_xyz: np.ndarray) -> None:
    if len(pre_xyz) == 0:
        return
    scene_center = np.median(pre_xyz, axis=0)
    scene_mad = np.median(np.abs(pre_xyz - scene_center), axis=0)
    robust_scale = 1.4826 * np.maximum(scene_mad, 1e-6)
    for cluster in clusters:
        centroid = np.asarray(cluster["centroid"], dtype=np.float64)
        robust_offset = (centroid - scene_center) / robust_scale
        cluster["scene_center_world"] = scene_center
        cluster["scene_robust_offset"] = robust_offset
        cluster["scene_robust_distance"] = float(np.linalg.norm(robust_offset))


def _ranking_record(rank: int, cluster: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": int(rank),
        "cluster_id": int(cluster["cluster_id"]),
        "n_points": int(cluster.get("n_points", 0)),
        "approximate_diameter": float(cluster.get("approximate_diameter") or 0.0),
        "scene_robust_distance": (
            float(cluster["scene_robust_distance"]) if "scene_robust_distance" in cluster else None
        ),
        "centroid": np.asarray(cluster["centroid"], dtype=float).tolist(),
    }
    if "target_selection_score" in cluster:
        record["target_selection_score"] = float(cluster["target_selection_score"])
    return record


def _cut_point_marker_cloud(
    targets: list[dict[str, Any]],
    marker_radius: float = 0.015,
    ray_length: float = 0.075,
) -> tuple[np.ndarray, np.ndarray]:
    marker_radius = max(float(marker_radius), 1e-6)
    ray_length = max(float(ray_length), 0.0)
    palette = np.asarray(
        [
            [1.00, 0.88, 0.05],
            [0.00, 0.95, 0.95],
            [1.00, 0.25, 0.70],
            [0.35, 1.00, 0.25],
            [1.00, 0.55, 0.05],
        ],
        dtype=np.float64,
    )
    axes = np.eye(3, dtype=np.float64)
    line_steps = np.linspace(-marker_radius, marker_radius, 11)
    ray_steps = np.linspace(0.0, ray_length, 24)

    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for rank, target in enumerate(targets):
        cut = target.get("cut_point_world")
        if cut is None:
            continue
        center = np.asarray(cut, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            continue
        color = palette[rank % len(palette)]

        points.append(center.reshape(1, 3))
        colors.append(color.reshape(1, 3))
        for axis in axes:
            line = center.reshape(1, 3) + line_steps.reshape(-1, 1) * axis.reshape(1, 3)
            points.append(line)
            colors.append(np.tile(color.reshape(1, 3), (len(line), 1)))

        removal = target.get("removal_direction_world") or target.get("approach_direction_world")
        if removal is None or ray_length <= 0.0:
            continue
        direction = np.asarray(removal, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not np.isfinite(norm) or norm <= 1e-9:
            continue
        direction = direction / norm
        ray = center.reshape(1, 3) + ray_steps.reshape(-1, 1) * direction.reshape(1, 3)
        ray_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        points.append(ray)
        colors.append(np.tile(ray_color.reshape(1, 3), (len(ray), 1)))

    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.vstack(points), np.vstack(colors)


def _warn_frame_mismatch(pre_frame: str | None, post_frame: str | None, diagnostics: dict[str, Any]) -> None:
    if pre_frame is not None and post_frame is not None and pre_frame != post_frame:
        msg = (
            "Pre and post point clouds may not be in a common world frame. "
            "Provide --post-to-pre-transform or use --refine-icp on stable regions."
        )
        LOGGER.warning(msg)
        diagnostics.setdefault("warnings", []).append(msg)


def run_pipeline(
    task: str,
    pre_points: str | Path,
    post_points: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    pre_observations: str | Path | None = None,
    post_observations: str | Path | None = None,
    stable_roi_json: str | Path | None = None,
    refine_icp: bool = False,
    post_to_pre_transform: str | Path | None = None,
    strict: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    del verbose
    out_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    plys_dir = ensure_dir(out_dir / "plys")
    _remove_unrequested_ply_outputs(plys_dir)

    cfg = load_config(config_path)
    cfg["task"] = task
    if refine_icp:
        cfg.setdefault("registration", {})["refine_icp"] = True
    save_config(cfg, out_dir / "config_resolved.yaml")

    diagnostics: dict[str, Any] = {
        "task": task,
        "inputs": {
            "pre_points": str(pre_points),
            "post_points": str(post_points),
            "pre_observations": str(pre_observations) if pre_observations else None,
            "post_observations": str(post_observations) if post_observations else None,
        },
        "warnings": [],
    }

    max_points = int(cfg.get("quality", {}).get("max_points", 2_000_000))
    pre_data = load_semidense_points(pre_points, pre_observations, max_points=max_points, random_seed=13)
    post_data = load_semidense_points(post_points, post_observations, max_points=max_points, random_seed=17)
    _warn_frame_mismatch(pre_data.frame_id, post_data.frame_id, diagnostics)
    diagnostics["loaders"] = {"pre": pre_data.metadata, "post": post_data.metadata}
    diagnostics["frames"] = {"pre": pre_data.frame_id, "post": post_data.frame_id}

    if (
        not cfg.get("registration", {}).get("assume_same_world_frame", True)
        and post_to_pre_transform is None
        and not cfg.get("registration", {}).get("refine_icp", False)
    ):
        msg = (
            "Pre and post point clouds may not be in a common world frame. "
            "Provide --post-to-pre-transform or use --refine-icp on stable regions."
        )
        LOGGER.warning(msg)
        diagnostics["warnings"].append(msg)

    pre_df, pre_xyz, pre_quality = filter_quality(pre_data.points_df, cfg)
    post_df, post_xyz_raw, post_quality = filter_quality(post_data.points_df, cfg)
    quality_report: dict[str, Any] = {"pre": pre_quality, "post": post_quality}

    t_initial = load_transform(post_to_pre_transform) if post_to_pre_transform else identity_transform()
    post_xyz_initial = apply_transform(post_xyz_raw, t_initial)

    roi = merge_roi_config(cfg.get("roi", {}), load_roi_json(stable_roi_json))
    pre_roi_mask, pre_roi_report = apply_roi_filter(pre_xyz, roi)
    post_roi_mask, post_roi_report = apply_roi_filter(post_xyz_initial, roi)
    pre_df = pre_df.loc[pre_roi_mask].reset_index(drop=True)
    post_df = post_df.loc[post_roi_mask].reset_index(drop=True)
    pre_xyz = pre_xyz[pre_roi_mask]
    post_xyz_initial = post_xyz_initial[post_roi_mask]
    post_xyz_raw = post_xyz_raw[post_roi_mask]
    quality_report["roi"] = {"pre": pre_roi_report, "post": post_roi_report, "stable_boxes": roi.get("stable_boxes", [])}
    write_json(out_dir / "quality_report.json", quality_report)

    alignment_diag: dict[str, Any]
    final_t_post_to_pre = t_initial.copy()
    post_xyz_aligned = post_xyz_initial.copy()
    if cfg.get("registration", {}).get("refine_icp", False):
        pre_stable = stable_mask(pre_xyz, roi)
        post_stable = stable_mask(post_xyz_initial, roi)
        if pre_stable.sum() < 10 or post_stable.sum() < 10:
            msg = "No stable_boxes with enough points were provided; using full crop region for ICP, which dynamic objects can bias."
            LOGGER.warning(msg)
            diagnostics["warnings"].append(msg)
            pre_icp = pre_xyz
            post_icp = post_xyz_initial
        else:
            pre_icp = pre_xyz[pre_stable]
            post_icp = post_xyz_initial[post_stable]
        t_delta, alignment_diag = refine_post_to_pre_icp(pre_icp, post_icp, cfg)
        final_t_post_to_pre = t_delta @ t_initial
        post_xyz_aligned = apply_transform(post_xyz_initial, t_delta)
    else:
        alignment_diag = {
            "method": "identity_or_explicit_transform",
            "fitness": None,
            "inlier_rmse": None,
            "n_pre_icp": 0,
            "n_post_icp": 0,
            "warning_flags": [],
        }

    alignment_diag["T_post_to_pre"] = final_t_post_to_pre.tolist()
    np.save(out_dir / "alignment_transform_post_to_pre.npy", final_t_post_to_pre)
    write_matrix_txt(out_dir / "alignment_transform_post_to_pre.txt", final_t_post_to_pre)
    write_json(out_dir / "alignment_diagnostics.json", alignment_diag)

    post_df = post_df.copy()
    if len(post_df) == len(post_xyz_aligned):
        post_df.loc[:, ["x", "y", "z"]] = post_xyz_aligned

    write_point_cloud_ply(plys_dir / "pre_filtered.ply", pre_xyz, (0.55, 0.55, 0.55))
    write_point_cloud_ply(plys_dir / "post_filtered_aligned.ply", post_xyz_aligned, (0.12, 0.35, 0.90))

    nn_pre_to_post = compute_pre_to_post_nn(pre_xyz, post_xyz_aligned)
    nn_post_to_pre = compute_pre_to_post_nn(post_xyz_aligned, pre_xyz)
    np.save(out_dir / "distances_nn_pre_to_post.npy", nn_pre_to_post)
    np.save(out_dir / "distances_nn_post_to_pre.npy", nn_post_to_pre)

    nn_cfg = cfg.get("nn", {})
    stable_for_threshold = stable_mask(pre_xyz, roi) if roi.get("stable_boxes") else None
    if nn_cfg.get("auto_threshold", True):
        nn_threshold, nn_threshold_diag = auto_nn_threshold(
            nn_pre_to_post,
            stable_for_threshold,
            float(nn_cfg.get("threshold_m", 0.025)),
            float(nn_cfg.get("auto_threshold_k_mad", 4.0)),
        )
    else:
        nn_threshold = float(nn_cfg.get("threshold_m", 0.025))
        nn_threshold_diag = {"threshold_m": nn_threshold, "auto_used": False, "reason": "disabled"}
    candidate_nn = nn_pre_to_post > nn_threshold
    diagnostics["nn"] = {
        "threshold": nn_threshold_diag,
        "n_candidate_nn": int(candidate_nn.sum()),
        "post_to_pre_addition_diagnostic_count": int((nn_post_to_pre > nn_threshold).sum()),
    }

    m3c2_result = run_m3c2(pre_xyz, post_xyz_aligned, cfg, out_dir=out_dir, strict=strict)
    _save_m3c2_outputs(out_dir, m3c2_result)
    m3c2_full = m3c2_result.full_distances if m3c2_result is not None else None
    candidate_m3c2 = m3c2_candidate_mask(m3c2_result, cfg)
    m3c2_for_scoring = None
    if m3c2_full is not None:
        m3c2_for_scoring = m3c2_full.copy()
        if candidate_m3c2 is not None:
            m3c2_for_scoring[~candidate_m3c2] = np.nan
    diagnostics["m3c2"] = {
        "available": m3c2_result is not None,
        "n_corepoints": int(len(m3c2_result.corepoint_indices)) if m3c2_result is not None else 0,
        "n_candidate_m3c2": int(candidate_m3c2.sum()) if candidate_m3c2 is not None else 0,
    }

    density_score = None
    if cfg.get("density", {}).get("enabled", False):
        density_score, density_diag = compute_density_drop_score(pre_xyz, post_xyz_aligned, cfg)
        diagnostics["density"] = density_diag

    combined_score, scoring_diag = combine_change_scores(
        nn_pre_to_post,
        nn_threshold,
        m3c2_for_scoring,
        float(cfg.get("m3c2", {}).get("threshold_m", 0.025)),
        cfg,
        density_score=density_score,
    )
    removed_mask = combined_score >= float(cfg.get("scoring", {}).get("min_combined_score", 0.5))
    diagnostics["scoring"] = scoring_diag
    diagnostics["scoring"]["n_removed_final"] = int(removed_mask.sum())

    m3c2_lod = _full_uncertainty(m3c2_result, "lodetection", len(pre_xyz))
    m3c2_num_samples1 = _full_uncertainty(m3c2_result, "num_samples1", len(pre_xyz))
    m3c2_num_samples2 = _full_uncertainty(m3c2_result, "num_samples2", len(pre_xyz))
    removed_df = pre_df.loc[removed_mask].copy()
    removed_df["original_index"] = np.where(removed_mask)[0]
    removed_df["nn_distance"] = nn_pre_to_post[removed_mask]
    if m3c2_full is not None:
        removed_df["m3c2_distance"] = m3c2_full[removed_mask]
    else:
        removed_df["m3c2_distance"] = np.nan
    if m3c2_lod is not None:
        removed_df["m3c2_lodetection"] = m3c2_lod[removed_mask]
    else:
        removed_df["m3c2_lodetection"] = np.nan
    if m3c2_num_samples1 is not None:
        removed_df["m3c2_num_samples1"] = m3c2_num_samples1[removed_mask]
    else:
        removed_df["m3c2_num_samples1"] = np.nan
    if m3c2_num_samples2 is not None:
        removed_df["m3c2_num_samples2"] = m3c2_num_samples2[removed_mask]
    else:
        removed_df["m3c2_num_samples2"] = np.nan
    if density_score is not None:
        removed_df["density_score"] = density_score[removed_mask]
    else:
        removed_df["density_score"] = np.nan
    removed_df["combined_score"] = combined_score[removed_mask]
    for col in ["uid", "obs_count", "inv_dist_std", "dist_std"]:
        if col not in removed_df.columns:
            removed_df[col] = np.nan
    requested_cols = [
        "uid",
        "x",
        "y",
        "z",
        "nn_distance",
        "m3c2_distance",
        "m3c2_lodetection",
        "m3c2_num_samples1",
        "m3c2_num_samples2",
        "density_score",
        "combined_score",
        "obs_count",
        "inv_dist_std",
        "dist_std",
        "original_index",
    ]
    safe_write_parquet(removed_df[requested_cols], out_dir / "removed_points.parquet")

    removed_xyz = pre_xyz[removed_mask]
    original_indices = np.where(removed_mask)[0]
    removed_nn = nn_pre_to_post[removed_mask]
    removed_m3c2 = m3c2_full[removed_mask] if m3c2_full is not None else None
    removed_scores = combined_score[removed_mask]
    clusters = cluster_removed_points(removed_xyz, original_indices, removed_nn, removed_m3c2, removed_scores, cfg)
    if task == "harvest":
        clusters, split_diag = split_merged_harvest_clusters(
            clusters,
            original_indices,
            removed_nn,
            removed_m3c2,
            removed_scores,
            cfg,
        )
        diagnostics["cluster_splitting"] = split_diag
    _annotate_scene_context(clusters.features, pre_xyz)
    ranked_clusters = rank_clusters_for_task(task, clusters.features, cfg, post_xyz_aligned)
    target_candidate_clusters = rank_clusters_for_task(
        task,
        clusters.features,
        cfg,
        post_xyz_aligned,
    )
    diagnostics["target_selection"] = {
        "ranking": [
            _ranking_record(rank, cluster)
            for rank, cluster in enumerate(ranked_clusters, start=1)
        ],
        "candidate_visualization_ranking": [
            _ranking_record(rank, cluster)
            for rank, cluster in enumerate(target_candidate_clusters[:10], start=1)
        ],
    }
    clusters_df = _clusters_dataframe(clusters.features)
    clusters_df.to_csv(out_dir / "clusters.csv", index=False)
    safe_write_parquet(clusters.cluster_points, out_dir / "cluster_points.parquet")
    write_json(out_dir / "cluster_features.json", [serializable_cluster_feature(f) for f in clusters.features])

    targets = infer_targets(task, clusters.features, pre_xyz, post_xyz_aligned, cfg)
    _attach_cluster_bbox_to_targets(targets, ranked_clusters + target_candidate_clusters + clusters.features)
    targets_doc = build_targets_document(task, final_t_post_to_pre, targets)
    write_json(out_dir / "targets.json", targets_doc)
    cutpoint_xyz, cutpoint_colors = _cut_point_marker_cloud(targets)
    write_point_cloud_ply(plys_dir / "cut_points.ply", cutpoint_xyz, cutpoint_colors)

    write_point_cloud_ply(plys_dir / "removed_candidates.ply", removed_xyz, (0.95, 0.10, 0.08))

    clustered_diagnostic_points = clusters.all_cluster_points
    if len(clustered_diagnostic_points):
        clustered_ids = clustered_diagnostic_points["cluster_id"].to_numpy(dtype=np.int64)
        clustered_xyz = clustered_diagnostic_points[["x", "y", "z"]].to_numpy(dtype=np.float64)
        write_point_cloud_ply(plys_dir / "clustered_removed_candidates.ply", clustered_xyz, _cluster_colors(clustered_ids))
    else:
        clustered_xyz = np.empty((0, 3), dtype=np.float64)
        write_point_cloud_ply(plys_dir / "clustered_removed_candidates.ply", clustered_xyz, (0.95, 0.45, 0.05))

    selected_target_cluster_ids = {int(target["cluster_id"]) for target in targets if target.get("cluster_id") is not None}
    top_candidate_cluster_ids = [int(cluster["cluster_id"]) for cluster in target_candidate_clusters[:10]]
    top_candidate_clusters = target_candidate_clusters[:10]
    if top_candidate_clusters:
        target_ply_xyz = np.vstack(
            [np.asarray(cluster["points"], dtype=np.float64) for cluster in top_candidate_clusters]
        )
        target_ply_ids = np.concatenate(
            [
                np.full(int(cluster.get("n_points", 0)), int(cluster["cluster_id"]), dtype=np.int64)
                for cluster in top_candidate_clusters
            ]
        )
        target_ply_colors = _ranked_cluster_colors(target_ply_ids, top_candidate_cluster_ids)
    else:
        target_ply_xyz = np.empty((0, 3), dtype=np.float64)
        target_ply_colors = np.asarray((0.20, 0.90, 0.25), dtype=float)
    write_point_cloud_ply(plys_dir / "target_candidates.ply", target_ply_xyz, target_ply_colors)
    diagnostics["removed_candidates_ply"] = {
        "mode": "all_removed_points",
        "n_points": int(len(removed_xyz)),
        "fallback_used": False,
    }
    diagnostics["clustered_removed_candidates_ply"] = {
        "mode": "clustered_removed_points",
        "n_points": int(len(clustered_xyz)),
    }
    diagnostics["target_candidates_ply"] = {
        "mode": "top_ranked_target_candidate_clusters",
        "n_points": int(len(target_ply_xyz)),
        "target_cluster_ids": sorted(selected_target_cluster_ids),
        "candidate_cluster_ids": top_candidate_cluster_ids,
        "max_candidate_clusters": 10,
    }
    diagnostics["cut_points_ply"] = {
        "mode": "cut_point_cross_and_removal_direction_ray",
        "n_points": int(len(cutpoint_xyz)),
        "target_count": int(len(targets)),
        "marker_radius_m": 0.015,
        "ray_length_m": 0.075,
    }

    write_diagnostic_plots(
        plots_dir,
        nn_pre_to_post,
        m3c2_result.distances if m3c2_result is not None else None,
        [int(f["n_points"]) for f in clusters.features],
    )

    diagnostics["counts"] = {
        "pre_filtered": int(len(pre_xyz)),
        "post_filtered_aligned": int(len(post_xyz_aligned)),
        "removed_points": int(len(removed_xyz)),
        "clusters": int(len(clusters.features)),
        "targets": int(len(targets)),
    }
    if len(removed_xyz) == 0:
        diagnostics["warnings"].append("no_removed_candidates_found")
    if len(clusters.features) == 0:
        diagnostics["warnings"].append("no_clusters_passed_filters")
    write_json(out_dir / "diagnostics.json", diagnostics)
    return targets_doc
