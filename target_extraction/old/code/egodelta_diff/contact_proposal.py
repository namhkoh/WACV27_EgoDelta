from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .targets import evaluate_proposals


@dataclass(frozen=True)
class ContactProposalConfig:
    removed_score_min: float = 0.5
    component_eps_m: float = 0.03
    component_min_points: int = 8
    support_dilation_m: float = 0.03
    min_support_points: int = 8
    dedup_radius_m: float = 0.025
    max_proposals_per_scene: int = 64
    primary_leaf_score_min: float = 0.0
    primary_leaf_min_support_count: int = 0
    primary_leaf_min_axis_length_m: float = 0.0
    primary_leaf_min_contact_count: int = 0
    primary_leaf_min_contact_green_strength: float = 0.0
    primary_leaf_require_cluster: bool = False
    primary_tomato_score_min: float = 0.0
    primary_tomato_max_count: int = 2
    static_green_ratio_min: float = 0.30
    static_green_minus_red_min: float = -0.05
    leaf_require_clustered_component: bool = False
    leaf_min_component_clustered_ratio: float = 0.0
    leaf_endpoint_fraction: float = 0.15
    leaf_min_axis_length_m: float = 0.015
    leaf_max_stem_distance_m: float = 0.08
    leaf_contact_radius_m: float = 0.03
    leaf_min_contact_points: int = 8
    leaf_min_contact_green_strength: float = 0.02
    leaf_core_snap_max_static_distance_m: float = 0.015
    leaf_static_joint_max_distance_m: float = 0.015
    leaf_static_joint_min_mean_score: float = 0.80
    leaf_endpoint_contact_count_weight: float = 0.020
    leaf_endpoint_green_strength_weight: float = 0.45
    leaf_endpoint_distance_weight: float = 2.0
    leaf_support_overlap_dedup_min: float = 0.0
    leaf_support_cluster_dedup_min: float = 0.0
    leaf_support_cluster_dedup_max_support_ratio: float = 0.75
    tomato_fruit_green_minus_red_max: float = 0.0352147
    tomato_fruit_green_ratio_max: float = 0.336774
    tomato_green_ratio_min: float = 0.35
    tomato_green_minus_red_min: float = 0.02
    tomato_lobe_min_points: int = 8
    tomato_contact_radius_m: float = 0.06
    tomato_max_static_stem_distance_m: float = 0.10
    tomato_require_cluster: bool = True
    tomato_peak_seed_score_min: float = 0.5
    tomato_peak_score_min: float = 0.9
    tomato_peak_nms_radius_m: float = 0.08
    tomato_peak_support_radius_m: float = 0.02
    tomato_peak_min_support: int = 80
    tomato_peak_max_per_support: int = 4
    tomato_peak_snap_radius_m: float = 0.075
    tomato_peak_snap_max_m: float = 0.065
    tomato_peak_snap_pair_radius_m: float = 0.045
    tomato_peak_snap_support_radius_m: float = 0.025
    tomato_peak_snap_strong_green_ratio_min: float = 0.40
    tomato_peak_snap_strong_green_minus_red_min: float = 0.08
    tomato_secondary_recovery: bool = True
    tomato_secondary_seed_score_min: float = 0.82
    tomato_secondary_min_support: int = 45
    tomato_secondary_min_separation_m: float = 0.055
    tomato_secondary_min_snap_density: float = 0.10
    tomato_primary_peak_only: bool = True


@dataclass(frozen=True)
class ContactProposal:
    proposal_id: int
    object_type: str
    method: str
    representative_point: list[float]
    source_index: int
    support_cluster_ids: list[int]
    support_count: int
    score: float
    score_terms: dict[str, float]
    stem_distance_m: float | None
    contact_distance_m: float | None
    support_id: str = ""
    rank: int = 0
    primary: bool = False
    fallback_reason: str | None = None
    primary_reason: str = ""


def contact_config_from_dict(config: dict[str, Any] | None) -> ContactProposalConfig:
    cfg = config or {}
    if "contact" in cfg and isinstance(cfg["contact"], dict):
        cfg = cfg["contact"]
    return ContactProposalConfig(
        removed_score_min=float(cfg.get("removed_score_min", 0.5)),
        component_eps_m=float(cfg.get("component_eps_m", 0.03)),
        component_min_points=int(cfg.get("component_min_points", 8)),
        support_dilation_m=float(cfg.get("support_dilation_m", 0.03)),
        min_support_points=int(cfg.get("min_support_points", 8)),
        dedup_radius_m=float(cfg.get("dedup_radius_m", 0.025)),
        max_proposals_per_scene=int(cfg.get("max_proposals_per_scene", 64)),
        primary_leaf_score_min=float(cfg.get("primary_leaf_score_min", 0.0)),
        primary_leaf_min_support_count=int(cfg.get("primary_leaf_min_support_count", 0)),
        primary_leaf_min_axis_length_m=float(cfg.get("primary_leaf_min_axis_length_m", 0.0)),
        primary_leaf_min_contact_count=int(cfg.get("primary_leaf_min_contact_count", 0)),
        primary_leaf_min_contact_green_strength=float(cfg.get("primary_leaf_min_contact_green_strength", 0.0)),
        primary_leaf_require_cluster=bool(cfg.get("primary_leaf_require_cluster", False)),
        primary_tomato_score_min=float(cfg.get("primary_tomato_score_min", 0.0)),
        primary_tomato_max_count=int(cfg.get("primary_tomato_max_count", cfg.get("primary_tomato_top_k", 2))),
        static_green_ratio_min=float(cfg.get("static_green_ratio_min", 0.30)),
        static_green_minus_red_min=float(cfg.get("static_green_minus_red_min", -0.05)),
        leaf_require_clustered_component=bool(cfg.get("leaf_require_clustered_component", False)),
        leaf_min_component_clustered_ratio=float(cfg.get("leaf_min_component_clustered_ratio", 0.0)),
        leaf_endpoint_fraction=float(cfg.get("leaf_endpoint_fraction", 0.15)),
        leaf_min_axis_length_m=float(cfg.get("leaf_min_axis_length_m", 0.015)),
        leaf_max_stem_distance_m=float(cfg.get("leaf_max_stem_distance_m", 0.08)),
        leaf_contact_radius_m=float(cfg.get("leaf_contact_radius_m", 0.03)),
        leaf_min_contact_points=int(cfg.get("leaf_min_contact_points", 8)),
        leaf_min_contact_green_strength=float(cfg.get("leaf_min_contact_green_strength", 0.02)),
        leaf_core_snap_max_static_distance_m=float(cfg.get("leaf_core_snap_max_static_distance_m", 0.015)),
        leaf_static_joint_max_distance_m=float(cfg.get("leaf_static_joint_max_distance_m", 0.015)),
        leaf_static_joint_min_mean_score=float(cfg.get("leaf_static_joint_min_mean_score", 0.80)),
        leaf_endpoint_contact_count_weight=float(cfg.get("leaf_endpoint_contact_count_weight", 0.020)),
        leaf_endpoint_green_strength_weight=float(cfg.get("leaf_endpoint_green_strength_weight", 0.45)),
        leaf_endpoint_distance_weight=float(cfg.get("leaf_endpoint_distance_weight", 2.0)),
        leaf_support_overlap_dedup_min=float(cfg.get("leaf_support_overlap_dedup_min", 0.0)),
        leaf_support_cluster_dedup_min=float(cfg.get("leaf_support_cluster_dedup_min", 0.0)),
        leaf_support_cluster_dedup_max_support_ratio=float(
            cfg.get("leaf_support_cluster_dedup_max_support_ratio", 0.75)
        ),
        tomato_fruit_green_minus_red_max=float(cfg.get("tomato_fruit_green_minus_red_max", 0.0352147)),
        tomato_fruit_green_ratio_max=float(cfg.get("tomato_fruit_green_ratio_max", 0.336774)),
        tomato_green_ratio_min=float(cfg.get("tomato_green_ratio_min", 0.35)),
        tomato_green_minus_red_min=float(cfg.get("tomato_green_minus_red_min", 0.02)),
        tomato_lobe_min_points=int(cfg.get("tomato_lobe_min_points", 8)),
        tomato_contact_radius_m=float(cfg.get("tomato_contact_radius_m", 0.06)),
        tomato_max_static_stem_distance_m=float(cfg.get("tomato_max_static_stem_distance_m", 0.10)),
        tomato_require_cluster=bool(cfg.get("tomato_require_cluster", True)),
        tomato_peak_seed_score_min=float(cfg.get("tomato_peak_seed_score_min", 0.5)),
        tomato_peak_score_min=float(cfg.get("tomato_peak_score_min", 0.9)),
        tomato_peak_nms_radius_m=float(cfg.get("tomato_peak_nms_radius_m", 0.08)),
        tomato_peak_support_radius_m=float(cfg.get("tomato_peak_support_radius_m", 0.02)),
        tomato_peak_min_support=int(cfg.get("tomato_peak_min_support", 80)),
        tomato_peak_max_per_support=int(cfg.get("tomato_peak_max_per_support", 4)),
        tomato_peak_snap_radius_m=float(cfg.get("tomato_peak_snap_radius_m", 0.075)),
        tomato_peak_snap_max_m=float(cfg.get("tomato_peak_snap_max_m", 0.065)),
        tomato_peak_snap_pair_radius_m=float(cfg.get("tomato_peak_snap_pair_radius_m", 0.045)),
        tomato_peak_snap_support_radius_m=float(cfg.get("tomato_peak_snap_support_radius_m", 0.025)),
        tomato_peak_snap_strong_green_ratio_min=float(cfg.get("tomato_peak_snap_strong_green_ratio_min", 0.40)),
        tomato_peak_snap_strong_green_minus_red_min=float(
            cfg.get("tomato_peak_snap_strong_green_minus_red_min", 0.08)
        ),
        tomato_secondary_recovery=bool(cfg.get("tomato_secondary_recovery", True)),
        tomato_secondary_seed_score_min=float(cfg.get("tomato_secondary_seed_score_min", 0.82)),
        tomato_secondary_min_support=int(cfg.get("tomato_secondary_min_support", 45)),
        tomato_secondary_min_separation_m=float(cfg.get("tomato_secondary_min_separation_m", 0.055)),
        tomato_secondary_min_snap_density=float(cfg.get("tomato_secondary_min_snap_density", 0.10)),
        tomato_primary_peak_only=bool(cfg.get("tomato_primary_peak_only", True)),
    )


def generate_contact_proposals(
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    rgb01: np.ndarray,
    removed_mask: np.ndarray,
    cluster_labels: np.ndarray | None,
    config: ContactProposalConfig,
    object_type: str,
) -> list[ContactProposal]:
    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    removed_mask = np.asarray(removed_mask, dtype=bool)
    if cluster_labels is None:
        cluster_labels = np.full(len(points), -1, dtype=np.int32)
    else:
        cluster_labels = np.asarray(cluster_labels, dtype=np.int32)

    if len(points) == 0 or len(points) != len(scores) or len(points) != len(removed_mask):
        return []

    object_type = object_type.lower()
    static_idx = _static_candidate_indices(points, rgb01, removed_mask, object_type, config)

    proposals: list[ContactProposal] = []
    if object_type == "tomato":
        for support_idx in _support_components(points, scores, removed_mask, cluster_labels, config):
            proposals.extend(
                _tomato_contact_proposals(
                    points,
                    scores,
                    source_indices,
                    rgb01,
                    cluster_labels,
                    support_idx,
                    static_idx,
                    config,
                )
            )
    else:
        for leaf_support_idx in _leaf_proposal_support_units(
            points,
            scores,
            removed_mask,
            cluster_labels,
            config,
        ):
            proposals.extend(
                _leaf_contact_proposals(
                    points,
                    scores,
                    source_indices,
                    rgb01,
                    cluster_labels,
                    leaf_support_idx,
                    static_idx,
                    config,
                    object_type=object_type if object_type else "leaf",
                )
            )

    return _deduplicate_and_reindex(
        proposals,
        config.dedup_radius_m,
        config.max_proposals_per_scene,
        config,
    )


def evaluate_contact_proposals(
    proposals: list[ContactProposal],
    gt_points: np.ndarray,
    gate_m: float = 0.03,
) -> dict[str, Any]:
    return evaluate_proposals(proposals, gt_points, gate_m=gate_m)


def evaluate_contact_proposal_sets(
    proposals: list[ContactProposal],
    gt_points: np.ndarray,
    gate_m: float = 0.03,
) -> dict[str, Any]:
    primary_proposals = select_primary_contact_proposals(proposals)
    primary_metrics = evaluate_contact_proposals(primary_proposals, gt_points, gate_m=gate_m)
    all_candidate_metrics = evaluate_contact_proposals(proposals, gt_points, gate_m=gate_m)
    payload = dict(primary_metrics)
    payload["primary_metrics"] = primary_metrics
    payload["all_candidate_metrics"] = all_candidate_metrics
    payload["primary_proposal_count"] = len(primary_proposals)
    payload["all_candidate_count"] = len(proposals)
    return payload


def select_primary_contact_proposals(proposals: list[ContactProposal]) -> list[ContactProposal]:
    return [proposal for proposal in proposals if proposal.primary]


def save_contact_proposals_json(
    path: Path,
    proposals: list[ContactProposal],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "proposals": [asdict(proposal) for proposal in proposals],
    }
    path.write_text(json.dumps(payload, indent=2))


def save_contact_proposals_ply(path: Path, proposals: list[ContactProposal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("score", "f4"),
        ("support_count", "i4"),
        ("cluster_id", "i4"),
        ("proposal_id", "i4"),
        ("rank", "i4"),
        ("primary", "u1"),
        ("source_index", "i4"),
        ("stem_distance_m", "f4"),
        ("contact_distance_m", "f4"),
    ]
    vertices = np.empty(len(proposals), dtype=dtype)
    for i, proposal in enumerate(proposals):
        point = np.asarray(proposal.representative_point, dtype=np.float32)
        vertices["x"][i], vertices["y"][i], vertices["z"][i] = point
        if proposal.object_type == "tomato":
            vertices["red"][i], vertices["green"][i], vertices["blue"][i] = 255, 128, 0
        else:
            vertices["red"][i], vertices["green"][i], vertices["blue"][i] = 0, 255, 255
        vertices["score"][i] = np.float32(proposal.score)
        vertices["support_count"][i] = np.int32(proposal.support_count)
        vertices["cluster_id"][i] = np.int32(proposal.support_cluster_ids[0] if proposal.support_cluster_ids else -1)
        vertices["proposal_id"][i] = np.int32(proposal.proposal_id)
        vertices["rank"][i] = np.int32(proposal.rank)
        vertices["primary"][i] = np.uint8(1 if proposal.primary else 0)
        vertices["source_index"][i] = np.int32(proposal.source_index)
        vertices["stem_distance_m"][i] = np.float32(_none_to_inf(proposal.stem_distance_m))
        vertices["contact_distance_m"][i] = np.float32(_none_to_inf(proposal.contact_distance_m))
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def save_contact_eval_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def _support_components(
    points: np.ndarray,
    scores: np.ndarray,
    removed_mask: np.ndarray,
    cluster_labels: np.ndarray,
    config: ContactProposalConfig,
) -> list[np.ndarray]:
    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(scores)
    removed_idx = np.nonzero(removed_mask & finite_mask & (scores >= config.removed_score_min))[0]
    if len(removed_idx) == 0:
        return []

    removed_points = points[removed_idx]
    removed_tree = cKDTree(removed_points)
    covered = np.zeros(len(removed_idx), dtype=bool)
    components: list[np.ndarray] = []

    labels = cluster_labels[removed_idx]
    for label in sorted(int(x) for x in np.unique(labels) if int(x) >= 0):
        local_seed = np.nonzero(labels == label)[0]
        if len(local_seed) < config.component_min_points:
            continue
        support_local = _dilate_local_indices(removed_tree, removed_points[local_seed], config.support_dilation_m)
        covered[support_local] = True
        support_global = removed_idx[support_local]
        if len(support_global) >= config.min_support_points:
            components.append(np.unique(support_global))

    noise_local = np.nonzero(~covered)[0]
    if len(noise_local) >= config.component_min_points:
        noise_components = _dbscan_components(
            removed_points[noise_local],
            config.component_eps_m,
            config.component_min_points,
        )
        for local_component in noise_components:
            support_global = removed_idx[noise_local[local_component]]
            if len(support_global) >= config.min_support_points:
                components.append(np.unique(support_global))

    if not components and len(removed_idx) >= config.min_support_points:
        components.append(removed_idx)
    return components


def _leaf_proposal_support_units(
    points: np.ndarray,
    scores: np.ndarray,
    removed_mask: np.ndarray,
    cluster_labels: np.ndarray,
    config: ContactProposalConfig,
) -> list[np.ndarray]:
    units: list[np.ndarray] = []
    for support_idx in _support_components(points, scores, removed_mask, cluster_labels, config):
        if config.leaf_require_clustered_component and not np.any(cluster_labels[support_idx] >= 0):
            continue
        if config.leaf_min_component_clustered_ratio > 0.0:
            clustered_ratio = float(np.mean(cluster_labels[support_idx] >= 0))
            if clustered_ratio < config.leaf_min_component_clustered_ratio:
                continue
        units.append(np.asarray(support_idx, dtype=np.int64))
    return _deduplicate_leaf_support_units(units, config.leaf_support_overlap_dedup_min)


def _deduplicate_leaf_support_units(units: list[np.ndarray], min_overlap: float) -> list[np.ndarray]:
    if not units:
        return []
    threshold = float(min_overlap)
    if threshold <= 0.0:
        return [np.asarray(unit, dtype=np.int64) for unit in units]

    normalized = [np.unique(np.asarray(unit, dtype=np.int64)) for unit in units if len(unit)]
    ordered = sorted(normalized, key=lambda unit: len(unit), reverse=True)
    kept: list[np.ndarray] = []
    for unit in ordered:
        if any(_support_subset_overlap(unit, kept_unit) >= threshold for kept_unit in kept):
            continue
        kept.append(unit)
    return kept


def _support_subset_overlap(a: np.ndarray, b: np.ndarray) -> float:
    denom = min(len(a), len(b))
    if denom <= 0:
        return 0.0
    shared = len(np.intersect1d(a, b, assume_unique=True))
    return float(shared) / float(denom)


def _leaf_contact_proposals(
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    rgb01: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    static_idx: np.ndarray,
    config: ContactProposalConfig,
    object_type: str,
) -> list[ContactProposal]:
    return _retained_green_leaf_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        cluster_labels,
        support_idx,
        static_idx,
        config,
        object_type,
    )


def _retained_green_leaf_contact_proposals(
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    rgb01: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    static_idx: np.ndarray,
    config: ContactProposalConfig,
    object_type: str,
) -> list[ContactProposal]:
    support_idx = np.asarray(support_idx, dtype=np.int64)
    if len(support_idx) < config.min_support_points:
        return []

    support_points = points[support_idx]
    centered = support_points - support_points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    axis = vh[0].astype(np.float32) if len(vh) else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    projection = centered @ axis
    axis_length = float(projection.max() - projection.min()) if len(projection) else 0.0

    q = min(max(config.leaf_endpoint_fraction, 0.01), 0.49)
    lo = np.quantile(projection, q)
    hi = np.quantile(projection, 1.0 - q)

    endpoint_sides = [
        ("low", np.nonzero(projection <= lo)[0]),
        ("high", np.nonzero(projection >= hi)[0]),
    ]
    if not any(len(endpoint_local) for _, endpoint_local in endpoint_sides):
        endpoint_sides = [("all", np.arange(len(support_idx)))]

    static_tree: cKDTree | None = None
    static_points = np.empty((0, 3), dtype=np.float32)
    if len(static_idx):
        static_points = points[static_idx]
        static_tree = cKDTree(static_points)

    endpoint_candidates: list[dict[str, Any]] = []
    for side_name, endpoint_local in endpoint_sides:
        if len(endpoint_local) == 0:
            continue
        candidate = _leaf_endpoint_candidate(
            points,
            scores,
            rgb01,
            support_idx,
            endpoint_local,
            static_idx,
            static_points,
            static_tree,
            config,
            side_name,
        )
        if candidate is not None:
            endpoint_candidates.append(candidate)

    if endpoint_candidates:
        best = max(endpoint_candidates, key=lambda item: item["endpoint_score"])
        best_local = int(best["local_index"])
        stem_distance = float(best["contact_distance_m"])
        contact_count = int(best["contact_count"])
        contact_green_strength = float(best["contact_green_strength"])
        endpoint_side = str(best["side"])
        best_local, stem_distance, snapped_to_core = _snap_leaf_target_to_core_endpoint(
            points,
            scores,
            cluster_labels,
            support_idx,
            best_local,
            next(indices for side, indices in endpoint_sides if side == endpoint_side),
            static_tree,
            config,
            stem_distance,
        )
    else:
        stem_distances = np.full(len(support_idx), np.inf, dtype=np.float32)
        static_joint = _leaf_static_joint_fallback_candidate(
            points,
            scores,
            cluster_labels,
            support_idx,
            static_tree,
            config,
        )
        if static_joint is not None:
            best_local = int(static_joint["local_index"])
            stem_distance = float(static_joint["stem_distance_m"])
            contact_count = 0
            contact_green_strength = 0.0
            endpoint_side = "static_joint"
        elif static_tree is not None:
            stem_distances, _ = static_tree.query(support_points, k=1)
            stem_distances = stem_distances.astype(np.float32)
        if static_joint is None:
            endpoint_local = np.concatenate([indices for _, indices in endpoint_sides if len(indices)])
            if len(endpoint_local) == 0:
                endpoint_local = np.arange(len(support_idx))
            endpoint_stem_distances = stem_distances[endpoint_local]
            if np.isfinite(endpoint_stem_distances).any():
                best_local = int(endpoint_local[int(np.nanargmin(endpoint_stem_distances))])
            else:
                endpoint_scores = scores[support_idx[endpoint_local]]
                best_local = int(endpoint_local[int(np.argmax(endpoint_scores))])
            stem_distance = float(stem_distances[best_local]) if np.isfinite(stem_distances[best_local]) else None
            contact_count = 0
            contact_green_strength = 0.0
            endpoint_side = "fallback"

    best_global = int(support_idx[best_local])
    fallback = None
    if axis_length < config.leaf_min_axis_length_m:
        fallback = "short_support_axis"
    if stem_distance is None:
        fallback = "no_static_stem"
    elif stem_distance > config.leaf_max_stem_distance_m:
        fallback = "stem_far"

    score_terms = {
        "mean_removed_score": float(scores[support_idx].mean()),
        "axis_length_m": axis_length,
        "stem_distance_m": _none_to_inf(stem_distance),
        "contact_count": float(contact_count),
        "contact_green_strength": contact_green_strength,
        "endpoint_side": 0.0 if endpoint_side == "low" else 1.0,
        "snapped_to_core": 1.0 if endpoint_candidates and snapped_to_core else 0.0,
        "static_joint_fallback": 1.0 if endpoint_side == "static_joint" else 0.0,
    }
    contact_bonus = 0.04 * float(np.log1p(contact_count)) + 0.35 * contact_green_strength
    distance_penalty = 0.0 if stem_distance is None else 0.5 * float(stem_distance)
    score = float(score_terms["mean_removed_score"] + axis_length + contact_bonus - distance_penalty)
    return [
        _contact_proposal(
            proposal_id=0,
            object_type=object_type,
            method="leaf_stem_endpoint",
            points=points,
            source_indices=source_indices,
            cluster_labels=cluster_labels,
            support_idx=support_idx,
            point_idx=best_global,
            score=score,
            score_terms=score_terms,
            stem_distance_m=stem_distance,
            contact_distance_m=stem_distance,
            fallback_reason=fallback,
        )
    ]


def _leaf_core_indices(
    scores: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    config: ContactProposalConfig,
) -> np.ndarray:
    support_idx = np.asarray(support_idx, dtype=np.int64)
    return support_idx[
        (cluster_labels[support_idx] >= 0)
        & np.isfinite(scores[support_idx])
        & (scores[support_idx] >= config.removed_score_min)
    ]


def _snap_leaf_target_to_core_endpoint(
    points: np.ndarray,
    scores: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    best_local: int,
    endpoint_local: np.ndarray,
    static_tree: cKDTree | None,
    config: ContactProposalConfig,
    current_stem_distance: float | None,
) -> tuple[int, float | None, bool]:
    """Move dilation/noise endpoint targets back onto the clustered removed mask."""
    if static_tree is None or len(endpoint_local) == 0:
        return best_local, current_stem_distance, False

    best_global = int(support_idx[best_local])
    if cluster_labels[best_global] >= 0:
        return best_local, current_stem_distance, False

    endpoint_global = support_idx[np.asarray(endpoint_local, dtype=np.int64)]
    core_global = _leaf_core_indices(scores, cluster_labels, endpoint_global, config)
    if len(core_global) == 0:
        return best_local, current_stem_distance, False

    core_distances, _ = static_tree.query(points[core_global], k=1)
    core_distances = np.asarray(core_distances, dtype=np.float32)
    if not np.isfinite(core_distances).any():
        return best_local, current_stem_distance, False

    core_pos = int(np.nanargmin(core_distances))
    core_distance = float(core_distances[core_pos])
    snap_radius = min(config.leaf_core_snap_max_static_distance_m, config.leaf_max_stem_distance_m)
    if core_distance > snap_radius:
        return best_local, current_stem_distance, False

    core_global_idx = int(core_global[core_pos])
    local_matches = np.nonzero(support_idx == core_global_idx)[0]
    if len(local_matches) == 0:
        return best_local, current_stem_distance, False
    return int(local_matches[0]), core_distance, True


def _leaf_static_joint_fallback_candidate(
    points: np.ndarray,
    scores: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    static_tree: cKDTree | None,
    config: ContactProposalConfig,
) -> dict[str, float | int] | None:
    if static_tree is None or len(support_idx) == 0:
        return None

    core_global = _leaf_core_indices(scores, cluster_labels, support_idx, config)
    if len(core_global) == 0:
        return None

    mean_score = float(scores[core_global].mean())
    if mean_score < config.leaf_static_joint_min_mean_score:
        return None

    core_distances, _ = static_tree.query(points[core_global], k=1)
    core_distances = np.asarray(core_distances, dtype=np.float32)
    if not np.isfinite(core_distances).any():
        return None

    best_pos = int(np.nanargmin(core_distances))
    best_distance = float(core_distances[best_pos])
    if best_distance > min(config.leaf_static_joint_max_distance_m, config.leaf_max_stem_distance_m):
        return None

    best_global = int(core_global[best_pos])
    local_matches = np.nonzero(support_idx == best_global)[0]
    if len(local_matches) == 0:
        return None
    return {
        "local_index": int(local_matches[0]),
        "stem_distance_m": best_distance,
    }


def _leaf_endpoint_candidate(
    points: np.ndarray,
    scores: np.ndarray,
    rgb01: np.ndarray,
    support_idx: np.ndarray,
    endpoint_local: np.ndarray,
    static_idx: np.ndarray,
    static_points: np.ndarray,
    static_tree: cKDTree | None,
    config: ContactProposalConfig,
    side_name: str,
) -> dict[str, Any] | None:
    if static_tree is None or len(static_idx) == 0 or len(endpoint_local) == 0:
        return None

    endpoint_points = points[support_idx[endpoint_local]]
    nearest_distances, nearest_static_local = static_tree.query(endpoint_points, k=1)
    nearest_distances = np.asarray(nearest_distances, dtype=np.float32)
    if not np.isfinite(nearest_distances).any():
        return None

    best_endpoint_pos = int(np.nanargmin(nearest_distances))
    best_local = int(endpoint_local[best_endpoint_pos])
    contact_distance = float(nearest_distances[best_endpoint_pos])
    if contact_distance > config.leaf_max_stem_distance_m:
        return None

    contact_local: set[int] = set()
    for local_neighbors in static_tree.query_ball_point(endpoint_points, r=config.leaf_contact_radius_m):
        contact_local.update(int(x) for x in local_neighbors)
    if len(contact_local) < config.leaf_min_contact_points:
        return None

    contact_local_arr = np.asarray(sorted(contact_local), dtype=np.int64)
    contact_rgb = rgb01[static_idx[contact_local_arr]]
    green_minus_red, green_ratio = _green_features(contact_rgb)
    green_strength = float(
        np.mean(np.maximum(green_ratio - config.static_green_ratio_min, 0.0))
        + 0.5 * np.mean(np.maximum(green_minus_red - config.static_green_minus_red_min, 0.0))
    )
    if green_strength < config.leaf_min_contact_green_strength:
        return None

    # Contact count and distance stabilize cases where both PCA endpoints touch
    # retained green geometry but only one endpoint is the true joint side.
    endpoint_score = (
        float(scores[support_idx].mean())
        + config.leaf_endpoint_contact_count_weight * float(np.log1p(len(contact_local_arr)))
        + config.leaf_endpoint_green_strength_weight * green_strength
        - config.leaf_endpoint_distance_weight * contact_distance
    )
    return {
        "side": side_name,
        "local_index": best_local,
        "contact_distance_m": contact_distance,
        "contact_count": int(len(contact_local_arr)),
        "contact_green_strength": green_strength,
        "endpoint_score": float(endpoint_score),
    }


def _tomato_contact_proposals(
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    rgb01: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    static_idx: np.ndarray,
    config: ContactProposalConfig,
) -> list[ContactProposal]:
    support_idx = np.asarray(support_idx, dtype=np.int64)
    if len(support_idx) < config.min_support_points:
        return []
    support_clusters = [int(x) for x in np.unique(cluster_labels[support_idx]) if int(x) >= 0]
    if config.tomato_require_cluster and not support_clusters:
        return []
    support_green_minus_red, support_green_ratio = _green_features(rgb01[support_idx].mean(axis=0, keepdims=True))
    if not (
        float(support_green_minus_red[0]) <= config.tomato_fruit_green_minus_red_max
        or float(support_green_ratio[0]) <= config.tomato_fruit_green_ratio_max
    ):
        return []

    return _tomato_peak_seed_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        cluster_labels,
        support_idx,
        static_idx,
        config,
    )


def _tomato_peak_seed_proposals(
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    rgb01: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    static_idx: np.ndarray,
    config: ContactProposalConfig,
) -> list[ContactProposal]:
    candidate_local = np.nonzero(scores[support_idx] >= config.tomato_peak_seed_score_min)[0]
    if len(candidate_local) == 0:
        return []

    candidate_global = support_idx[candidate_local]
    candidate_points = points[candidate_global]
    candidate_scores = scores[candidate_global]
    candidate_tree = cKDTree(candidate_points)
    sorted_local = np.argsort(-candidate_scores)
    suppressed = np.zeros(len(candidate_global), dtype=bool)
    proposals: list[ContactProposal] = []

    for local_idx in sorted_local:
        global_idx = int(candidate_global[local_idx])
        peak_score = float(scores[global_idx])
        support_local = candidate_tree.query_ball_point(
            candidate_points[local_idx],
            r=config.tomato_peak_support_radius_m,
        )
        strict_candidate = (
            not suppressed[local_idx]
            and peak_score >= config.tomato_peak_score_min
            and len(support_local) >= config.tomato_peak_min_support
        )
        secondary_candidate = (
            config.tomato_secondary_recovery
            and not strict_candidate
            and peak_score >= config.tomato_secondary_seed_score_min
            and len(support_local) >= config.tomato_secondary_min_support
        )
        if not strict_candidate and not secondary_candidate:
            continue
        support_global = candidate_global[np.asarray(support_local, dtype=np.int64)]
        support_scores = np.clip(scores[support_global].astype(np.float64), 1.0e-8, None)
        support_centroid = np.average(points[support_global].astype(np.float64), axis=0, weights=support_scores)
        medoid_global = int(
            support_global[int(np.argmin(np.linalg.norm(points[support_global] - support_centroid, axis=1)))]
        )
        contact_global, snap_terms = _tomato_snap_peak_to_red_green_interface(
            points,
            scores,
            rgb01,
            support_idx,
            peak_global=global_idx,
            medoid_global=medoid_global,
            static_idx=static_idx,
            config=config,
        )
        if secondary_candidate:
            if float(snap_terms.get("red_green_snap", 0.0)) < 0.5:
                continue
            if float(snap_terms.get("snap_density_score", 0.0)) < config.tomato_secondary_min_snap_density:
                continue
            contact_point = points[contact_global]
            if any(
                float(np.linalg.norm(contact_point - np.asarray(proposal.representative_point, dtype=np.float32)))
                < config.tomato_secondary_min_separation_m
                for proposal in proposals
            ):
                continue

        score_terms = {
            "peak_score": peak_score,
            "local_support_count": float(len(support_local)),
            "peak_seed_medoid": 1.0,
            "secondary_recovery": 1.0 if secondary_candidate else 0.0,
            **snap_terms,
        }
        support_bonus = min(len(support_local), 500) / 500.0
        density_bonus = 0.08 * float(snap_terms.get("snap_density_score", 0.0))
        secondary_penalty = 0.08 if secondary_candidate else 0.0
        proposals.append(
            _contact_proposal(
                proposal_id=len(proposals),
                object_type="tomato",
                method="tomato_high_score_contact_seed",
                points=points,
                source_indices=source_indices,
                cluster_labels=cluster_labels,
                support_idx=support_idx,
                point_idx=contact_global,
                score=float(peak_score + support_bonus + density_bonus - secondary_penalty),
                score_terms=score_terms,
                stem_distance_m=None,
                contact_distance_m=(
                    None
                    if snap_terms.get("snap_pair_distance_m", float("inf")) == float("inf")
                    else float(snap_terms["snap_pair_distance_m"])
                ),
                fallback_reason=None,
            )
        )
        if strict_candidate:
            suppress_local = candidate_tree.query_ball_point(
                candidate_points[local_idx],
                r=config.tomato_peak_nms_radius_m,
            )
            suppressed[np.asarray(suppress_local, dtype=np.int64)] = True
        if len(proposals) >= config.tomato_peak_max_per_support:
            break

    return proposals


def _tomato_snap_peak_to_red_green_interface(
    points: np.ndarray,
    scores: np.ndarray,
    rgb01: np.ndarray,
    support_idx: np.ndarray,
    peak_global: int,
    medoid_global: int,
    static_idx: np.ndarray,
    config: ContactProposalConfig,
) -> tuple[int, dict[str, float]]:
    """Move a high-score tomato fruit seed toward the local calyx/fruit interface.

    The removed mask already tends to contain the GT contact point. The failure is
    that the highest score density mode sits on the red fruit surface. This snap is
    deliberately local to the fruit peak, so distant green leakage cannot dominate
    the contact the way the broad red/green fallback can.
    """
    default_terms = {
        "red_green_snap": 0.0,
        "snap_distance_m": 0.0,
        "snap_pair_distance_m": float("inf"),
        "snap_density_score": 0.0,
        "snap_static_distance_m": float("inf"),
        "snap_green_ratio": 0.0,
        "snap_green_minus_red": 0.0,
    }
    if len(support_idx) == 0:
        return medoid_global, default_terms

    peak_point = points[int(peak_global)]
    support_distances = np.linalg.norm(points[support_idx] - peak_point, axis=1)
    local_support = support_idx[support_distances <= config.tomato_peak_snap_radius_m]
    if len(local_support) < config.tomato_lobe_min_points:
        return medoid_global, default_terms

    local_rgb = rgb01[local_support]
    green_minus_red, green_ratio = _green_features(local_rgb)
    fruit_local = np.nonzero(
        (green_minus_red <= config.tomato_fruit_green_minus_red_max)
        | (green_ratio <= config.tomato_fruit_green_ratio_max)
    )[0]
    green_local = np.nonzero(
        (green_ratio >= config.tomato_green_ratio_min)
        | (green_minus_red >= config.tomato_green_minus_red_min)
    )[0]
    if len(fruit_local) == 0 or len(green_local) == 0:
        return medoid_global, default_terms

    fruit_global = local_support[fruit_local]
    strong_green_local = np.nonzero(
        (green_ratio >= config.tomato_peak_snap_strong_green_ratio_min)
        | (green_minus_red >= config.tomato_peak_snap_strong_green_minus_red_min)
    )[0]
    green_global = local_support[strong_green_local if len(strong_green_local) else green_local]
    fruit_tree = cKDTree(points[fruit_global])
    green_tree = cKDTree(points[green_global])
    local_points = points[local_support]
    fruit_distances, _ = fruit_tree.query(local_points, k=1)
    green_distances, _ = green_tree.query(local_points, k=1)
    fruit_distances = np.asarray(fruit_distances, dtype=np.float32)
    green_distances = np.asarray(green_distances, dtype=np.float32)

    static_distances = np.full(len(local_support), np.inf, dtype=np.float32)
    if len(static_idx):
        static_tree = cKDTree(points[static_idx])
        static_distances, _ = static_tree.query(local_points, k=1)
        static_distances = np.asarray(static_distances, dtype=np.float32)

    valid_candidate = np.isfinite(fruit_distances) & (fruit_distances <= config.tomato_peak_snap_pair_radius_m)
    valid_candidate &= (
        (np.isfinite(green_distances) & (green_distances <= config.tomato_peak_snap_pair_radius_m))
        | (
            np.isfinite(static_distances)
            & (static_distances <= config.tomato_max_static_stem_distance_m)
        )
    )
    if np.any(valid_candidate):
        valid_positions = np.nonzero(valid_candidate)[0]
        valid_points = local_points[valid_positions]
        green_counts = np.asarray(
            [len(x) for x in green_tree.query_ball_point(valid_points, r=config.tomato_peak_snap_support_radius_m)],
            dtype=np.float32,
        )
        fruit_counts = np.asarray(
            [len(x) for x in fruit_tree.query_ball_point(valid_points, r=config.tomato_peak_snap_support_radius_m)],
            dtype=np.float32,
        )
        density_score = np.sqrt(
            np.minimum(green_counts, 25.0) / 25.0
            * np.minimum(fruit_counts, 60.0) / 60.0
        )
        valid_green_minus_red = green_minus_red[valid_positions]
        valid_green_ratio = green_ratio[valid_positions]
        green_color_score = np.maximum(valid_green_ratio - config.tomato_fruit_green_ratio_max, 0.0) + np.maximum(
            valid_green_minus_red - config.tomato_fruit_green_minus_red_max,
            0.0,
        )
        fruit_proximity = np.exp(-fruit_distances[valid_positions] / config.tomato_peak_snap_support_radius_m)
        green_proximity = np.exp(-green_distances[valid_positions] / config.tomato_peak_snap_support_radius_m)
        static_proximity = np.exp(
            -static_distances[valid_positions] / max(config.tomato_contact_radius_m, 1.0e-6)
        )
        static_proximity[~np.isfinite(static_distances[valid_positions])] = 0.0
        peak_distance = np.linalg.norm(valid_points - peak_point, axis=1)
        candidate_score = (
            1.20 * density_score
            + 0.25 * fruit_proximity
            + 0.35 * green_proximity
            + 0.35 * static_proximity
            + 0.15 * green_color_score
            + 0.05 * scores[local_support[valid_positions]]
            - 0.18 * peak_distance / max(config.tomato_peak_snap_radius_m, 1.0e-6)
        )
        best_valid_pos = int(np.argmax(candidate_score))
        contact_global = int(local_support[valid_positions[best_valid_pos]])
        snap_distance = float(np.linalg.norm(points[contact_global] - points[medoid_global]))
        if snap_distance <= config.tomato_peak_snap_max_m:
            contact_green_minus_red, contact_green_ratio = _green_features(rgb01[[contact_global]])
            terms = {
                "red_green_snap": 1.0,
                "snap_distance_m": snap_distance,
                "snap_pair_distance_m": float(
                    min(
                        green_distances[valid_positions[best_valid_pos]],
                        static_distances[valid_positions[best_valid_pos]],
                    )
                ),
                "snap_density_score": float(density_score[best_valid_pos]),
                "snap_static_distance_m": float(static_distances[valid_positions[best_valid_pos]]),
                "snap_green_ratio": float(contact_green_ratio[0]),
                "snap_green_minus_red": float(contact_green_minus_red[0]),
            }
            return contact_global, terms

    green_global = local_support[green_local]
    green_tree = cKDTree(points[green_global])
    pair_distances, nearest_green = green_tree.query(points[fruit_global], k=1)
    pair_distances = np.asarray(pair_distances, dtype=np.float32)
    valid_pair = np.isfinite(pair_distances) & (pair_distances <= config.tomato_peak_snap_pair_radius_m)
    if not np.any(valid_pair):
        return medoid_global, default_terms

    valid_fruit_pos = np.nonzero(valid_pair)[0]
    valid_fruit_global = fruit_global[valid_fruit_pos]
    valid_green_global = green_global[np.asarray(nearest_green[valid_fruit_pos], dtype=np.int64)]
    fruit_peak_distance = np.linalg.norm(points[valid_fruit_global] - peak_point, axis=1)
    valid_green_minus_red, valid_green_ratio = _green_features(rgb01[valid_green_global])
    green_bonus = np.maximum(valid_green_ratio - config.tomato_fruit_green_ratio_max, 0.0) + np.maximum(
        valid_green_minus_red - config.tomato_fruit_green_minus_red_max,
        0.0,
    )
    pair_objective = (
        pair_distances[valid_fruit_pos].astype(np.float64)
        + 0.20 * fruit_peak_distance.astype(np.float64)
        - 0.015 * green_bonus.astype(np.float64)
        - 0.010 * scores[valid_fruit_global].astype(np.float64)
    )
    best_pair_pos = int(np.argmin(pair_objective))
    best_fruit = int(valid_fruit_global[best_pair_pos])
    best_green = int(valid_green_global[best_pair_pos])
    midpoint = 0.5 * (points[best_fruit] + points[best_green])

    local_dist_to_midpoint = np.linalg.norm(points[local_support] - midpoint, axis=1)
    contact_global = int(local_support[int(np.argmin(local_dist_to_midpoint))])
    snap_distance = float(np.linalg.norm(points[contact_global] - points[medoid_global]))
    if snap_distance > config.tomato_peak_snap_max_m:
        return medoid_global, default_terms

    contact_green_minus_red, contact_green_ratio = _green_features(rgb01[[contact_global]])
    terms = {
        "red_green_snap": 1.0,
        "snap_distance_m": snap_distance,
        "snap_pair_distance_m": float(pair_distances[valid_fruit_pos][best_pair_pos]),
        "snap_green_ratio": float(contact_green_ratio[0]),
        "snap_green_minus_red": float(contact_green_minus_red[0]),
    }
    return contact_global, terms


def _contact_proposal(
    proposal_id: int,
    object_type: str,
    method: str,
    points: np.ndarray,
    source_indices: np.ndarray,
    cluster_labels: np.ndarray,
    support_idx: np.ndarray,
    point_idx: int,
    score: float,
    score_terms: dict[str, float],
    stem_distance_m: float | None,
    contact_distance_m: float | None,
    fallback_reason: str | None,
) -> ContactProposal:
    clusters = sorted(int(x) for x in np.unique(cluster_labels[support_idx]) if int(x) >= 0)
    return ContactProposal(
        proposal_id=proposal_id,
        object_type=object_type,
        method=method,
        representative_point=points[point_idx].astype(float).tolist(),
        source_index=int(source_indices[point_idx]),
        support_cluster_ids=clusters,
        support_count=int(len(support_idx)),
        score=float(score),
        score_terms=score_terms,
        stem_distance_m=None if stem_distance_m is None else float(stem_distance_m),
        contact_distance_m=None if contact_distance_m is None else float(contact_distance_m),
        support_id=_support_id(object_type, source_indices, support_idx),
        fallback_reason=fallback_reason,
    )


def _static_candidate_indices(
    points: np.ndarray,
    rgb01: np.ndarray,
    removed_mask: np.ndarray,
    object_type: str,
    config: ContactProposalConfig,
) -> np.ndarray:
    static_idx = np.nonzero((~removed_mask) & np.isfinite(points).all(axis=1))[0]
    if len(static_idx) == 0:
        return static_idx.astype(np.int64)
    if object_type not in {"leaf", "tomato"}:
        return static_idx.astype(np.int64)

    green_minus_red, green_ratio = _green_features(rgb01[static_idx])
    color_mask = (green_ratio >= config.static_green_ratio_min) | (
        green_minus_red >= config.static_green_minus_red_min
    )
    color_idx = static_idx[color_mask]
    if len(color_idx) >= max(10, min(50, len(static_idx) // 20)):
        return color_idx.astype(np.int64)
    return static_idx.astype(np.int64)


def _dbscan_components(points: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    if len(points) < min_points:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(points)
    return [
        np.nonzero(labels == label)[0].astype(np.int64)
        for label in sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    ]


def _dilate_local_indices(tree: cKDTree, seed_points: np.ndarray, radius: float) -> np.ndarray:
    if len(seed_points) == 0:
        return np.empty(0, dtype=np.int64)
    neighbors: set[int] = set()
    for local in tree.query_ball_point(seed_points, r=radius):
        neighbors.update(int(x) for x in local)
    return np.asarray(sorted(neighbors), dtype=np.int64)


def _green_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.float32)
    denom = np.clip(rgb.sum(axis=1), 1.0e-8, None)
    green_minus_red = rgb[:, 1] - rgb[:, 0]
    green_ratio = rgb[:, 1] / denom
    return green_minus_red, green_ratio


def _support_id(object_type: str, source_indices: np.ndarray, support_idx: np.ndarray) -> str:
    support_sources = np.unique(np.asarray(source_indices[support_idx], dtype=np.int64))
    if len(support_sources) == 0:
        return f"{object_type}:empty"
    checksum = int(np.sum(np.mod(support_sources, 1_000_003), dtype=np.int64) % 1_000_003)
    return (
        f"{object_type}:n{len(support_sources)}:"
        f"min{int(support_sources.min())}:max{int(support_sources.max())}:sum{checksum}"
    )


def _deduplicate_and_reindex(
    proposals: list[ContactProposal],
    radius_m: float,
    max_count: int,
    config: ContactProposalConfig,
) -> list[ContactProposal]:
    if not proposals:
        return []
    ordered = sorted(proposals, key=lambda proposal: proposal.score, reverse=True)
    kept: list[ContactProposal] = []
    kept_points: list[np.ndarray] = []
    for proposal in ordered:
        point = np.asarray(proposal.representative_point, dtype=np.float32)
        if _is_leaf_support_cluster_duplicate(proposal, kept, config):
            continue
        if any(float(np.linalg.norm(point - kept_point)) < radius_m for kept_point in kept_points):
            continue
        kept_points.append(point)
        kept.append(proposal)
        if len(kept) >= max_count:
            break
    return _reindex_and_mark_primary(kept, config)


def _is_leaf_support_cluster_duplicate(
    proposal: ContactProposal,
    kept: list[ContactProposal],
    config: ContactProposalConfig,
) -> bool:
    threshold = float(config.leaf_support_cluster_dedup_min)
    if proposal.object_type != "leaf" or threshold <= 0.0 or not proposal.support_cluster_ids:
        return False
    support_ratio_limit = float(config.leaf_support_cluster_dedup_max_support_ratio)
    proposal_clusters = set(int(x) for x in proposal.support_cluster_ids)
    for kept_proposal in kept:
        if kept_proposal.object_type != "leaf" or not kept_proposal.support_cluster_ids:
            continue
        if proposal.support_count > kept_proposal.support_count * support_ratio_limit:
            continue
        kept_clusters = set(int(x) for x in kept_proposal.support_cluster_ids)
        denom = min(len(proposal_clusters), len(kept_clusters))
        if denom <= 0:
            continue
        overlap = len(proposal_clusters & kept_clusters) / float(denom)
        if overlap >= threshold:
            return True
    return False


def _reindex_and_mark_primary(
    proposals: list[ContactProposal],
    config: ContactProposalConfig,
) -> list[ContactProposal]:
    primary_decisions = _primary_decisions(proposals, config)
    output: list[ContactProposal] = []
    for i, proposal in enumerate(proposals):
        if i < len(primary_decisions):
            is_primary, primary_reason = primary_decisions[i]
        else:
            is_primary, primary_reason = False, "missing_primary_decision"
        output.append(
            ContactProposal(
                proposal_id=i,
                object_type=proposal.object_type,
                method=proposal.method,
                representative_point=proposal.representative_point,
                source_index=proposal.source_index,
                support_cluster_ids=proposal.support_cluster_ids,
                support_count=proposal.support_count,
                score=proposal.score,
                score_terms=proposal.score_terms,
                stem_distance_m=proposal.stem_distance_m,
                contact_distance_m=proposal.contact_distance_m,
                support_id=proposal.support_id,
                rank=i + 1,
                primary=is_primary,
                fallback_reason=proposal.fallback_reason,
                primary_reason=primary_reason,
            )
        )
    return output


def _primary_decisions(
    proposals: list[ContactProposal],
    config: ContactProposalConfig,
) -> list[tuple[bool, str]]:
    tomato_kept = 0
    tomato_limit = max(0, int(config.primary_tomato_max_count))
    decisions: list[tuple[bool, str]] = []
    for proposal in proposals:
        if proposal.object_type == "tomato":
            rejection_reason = _tomato_primary_rejection_reason(proposal, config)
            if rejection_reason is not None:
                decisions.append((False, rejection_reason))
            elif tomato_kept >= tomato_limit:
                decisions.append((False, "outside_tomato_primary_limit"))
            else:
                decisions.append((True, "accepted"))
                tomato_kept += 1
            continue

        rejection_reason = _leaf_primary_rejection_reason(proposal, config)
        decisions.append((rejection_reason is None, "accepted" if rejection_reason is None else rejection_reason))
    return decisions


def _tomato_primary_rejection_reason(
    proposal: ContactProposal,
    config: ContactProposalConfig,
) -> str | None:
    if float(proposal.score) < config.primary_tomato_score_min:
        return "tomato_score_below_min"
    if config.tomato_primary_peak_only and proposal.method != "tomato_high_score_contact_seed":
        return "tomato_non_peak_fallback"
    if proposal.fallback_reason is not None:
        return f"tomato_fallback_{proposal.fallback_reason}"
    return None


def _leaf_primary_rejection_reason(
    proposal: ContactProposal,
    config: ContactProposalConfig,
) -> str | None:
    if _passes_leaf_static_joint_anchor(proposal, config):
        return None
    terms = proposal.score_terms
    if float(proposal.score) < config.primary_leaf_score_min:
        return "leaf_score_below_min"
    if proposal.support_count < config.primary_leaf_min_support_count:
        return "leaf_support_count_below_min"
    if float(terms.get("axis_length_m", 0.0)) < config.primary_leaf_min_axis_length_m:
        return "leaf_axis_length_below_min"
    if float(terms.get("contact_count", 0.0)) < config.primary_leaf_min_contact_count:
        return "leaf_contact_count_below_min"
    if float(terms.get("contact_green_strength", 0.0)) < config.primary_leaf_min_contact_green_strength:
        return "leaf_contact_green_below_min"
    if config.primary_leaf_require_cluster and not proposal.support_cluster_ids:
        return "leaf_missing_cluster"
    if proposal.fallback_reason is not None:
        return f"leaf_fallback_{proposal.fallback_reason}"
    return None


def _passes_leaf_static_joint_anchor(
    proposal: ContactProposal,
    config: ContactProposalConfig,
) -> bool:
    terms = proposal.score_terms
    if float(terms.get("static_joint_fallback", 0.0)) < 0.5:
        return False
    if proposal.support_count < config.primary_leaf_min_support_count:
        return False
    if float(terms.get("axis_length_m", 0.0)) < config.primary_leaf_min_axis_length_m:
        return False
    if float(terms.get("mean_removed_score", 0.0)) < config.leaf_static_joint_min_mean_score:
        return False
    if float(terms.get("stem_distance_m", float("inf"))) > min(
        config.leaf_static_joint_max_distance_m,
        config.leaf_max_stem_distance_m,
    ):
        return False
    if config.primary_leaf_require_cluster and not proposal.support_cluster_ids:
        return False
    return True


def _none_to_inf(value: float | None) -> float:
    if value is None:
        return float("inf")
    return float(value)

