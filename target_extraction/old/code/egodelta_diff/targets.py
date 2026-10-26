from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ProposalConfig:
    seed_score_min: float = 0.5
    nms_radius_m: float = 0.03
    support_radius_m: float = 0.02
    min_support: int = 10
    min_score_mass: float = 0.0
    min_peak_score: float = 0.5


@dataclass(frozen=True)
class TargetProposal:
    proposal_id: int
    representative_point: list[float]
    weighted_centroid: list[float]
    peak_point: list[float]
    peak_score: float
    source_index: int
    support_count: int
    score_mass: float
    mean_score: float
    local_radius_m: float


@dataclass(frozen=True)
class ProposalMatch:
    proposal_id: int
    target_index: int
    distance_m: float
    within_gate: bool


def proposal_config_from_dict(config: dict[str, Any] | None) -> ProposalConfig:
    cfg = config or {}
    if "baseline_nms" in cfg and isinstance(cfg["baseline_nms"], dict):
        cfg = cfg["baseline_nms"]
    return ProposalConfig(
        seed_score_min=float(cfg.get("seed_score_min", 0.5)),
        nms_radius_m=float(cfg.get("nms_radius_m", 0.03)),
        support_radius_m=float(cfg.get("support_radius_m", 0.02)),
        min_support=int(cfg.get("min_support", 10)),
        min_score_mass=float(cfg.get("min_score_mass", 0.0)),
        min_peak_score=float(cfg.get("min_peak_score", 0.5)),
    )


def generate_target_proposals(
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    config: ProposalConfig,
) -> list[TargetProposal]:
    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if len(points) == 0:
        return []

    candidate_mask = scores >= config.seed_score_min
    candidate_indices = np.nonzero(candidate_mask)[0]
    if len(candidate_indices) == 0:
        return []

    candidate_points = points[candidate_indices]
    candidate_scores = scores[candidate_indices]
    candidate_tree = cKDTree(candidate_points)
    support_tree = candidate_tree
    sorted_local = np.argsort(-candidate_scores)
    suppressed = np.zeros(len(candidate_indices), dtype=bool)

    proposals: list[TargetProposal] = []
    for local_idx in sorted_local:
        if suppressed[local_idx]:
            continue
        global_idx = int(candidate_indices[local_idx])
        peak_score = float(scores[global_idx])
        if peak_score < config.min_peak_score:
            continue

        support_local = support_tree.query_ball_point(
            candidate_points[local_idx],
            r=config.support_radius_m,
        )
        if len(support_local) < config.min_support:
            continue

        support_global = candidate_indices[np.asarray(support_local, dtype=np.int64)]
        support_scores = scores[support_global].astype(np.float64)
        score_mass = float(support_scores.sum())
        if score_mass < config.min_score_mass:
            continue

        support_points = points[support_global].astype(np.float64)
        weights = np.clip(support_scores, 1.0e-8, None)
        weighted_centroid = np.average(support_points, axis=0, weights=weights)
        proposal_id = len(proposals)
        proposals.append(
            TargetProposal(
                proposal_id=proposal_id,
                representative_point=points[global_idx].astype(float).tolist(),
                weighted_centroid=weighted_centroid.astype(float).tolist(),
                peak_point=points[global_idx].astype(float).tolist(),
                peak_score=peak_score,
                source_index=int(source_indices[global_idx]),
                support_count=int(len(support_local)),
                score_mass=score_mass,
                mean_score=float(support_scores.mean()),
                local_radius_m=float(config.support_radius_m),
            )
        )

        suppress_local = candidate_tree.query_ball_point(
            candidate_points[local_idx],
            r=config.nms_radius_m,
        )
        suppressed[np.asarray(suppress_local, dtype=np.int64)] = True

    return proposals


def evaluate_proposals(
    proposals: list[TargetProposal],
    gt_points: np.ndarray,
    gate_m: float = 0.03,
) -> dict[str, Any]:
    gt_points = np.asarray(gt_points, dtype=np.float32).reshape((-1, 3))
    proposal_points = np.asarray(
        [proposal.representative_point for proposal in proposals],
        dtype=np.float32,
    ).reshape((-1, 3))

    proposal_count = int(len(proposals))
    gt_count = int(len(gt_points))
    if proposal_count == 0 and gt_count == 0:
        return _proposal_metrics(proposal_count, gt_count, 0, 0, 0, [], [])
    if proposal_count == 0:
        return _proposal_metrics(proposal_count, gt_count, 0, 0, gt_count, [], [])
    if gt_count == 0:
        return _proposal_metrics(proposal_count, gt_count, 0, proposal_count, 0, [], [])

    distances = np.linalg.norm(proposal_points[:, None, :] - gt_points[None, :, :], axis=2)
    proposal_rows, target_cols = linear_sum_assignment(distances)
    matches: list[ProposalMatch] = []
    tp = 0
    assigned_proposals: set[int] = set()
    assigned_targets: set[int] = set()
    false_matched = 0
    tp_distances: list[float] = []
    all_assignment_distances: list[float] = []

    for proposal_idx, target_idx in zip(proposal_rows.tolist(), target_cols.tolist()):
        distance = float(distances[proposal_idx, target_idx])
        within_gate = distance <= gate_m
        matches.append(
            ProposalMatch(
                proposal_id=int(proposals[proposal_idx].proposal_id),
                target_index=int(target_idx),
                distance_m=distance,
                within_gate=within_gate,
            )
        )
        assigned_proposals.add(proposal_idx)
        assigned_targets.add(target_idx)
        all_assignment_distances.append(distance)
        if within_gate:
            tp += 1
            tp_distances.append(distance)
        else:
            false_matched += 1

    unmatched_proposals = proposal_count - len(assigned_proposals)
    unmatched_targets = gt_count - len(assigned_targets)
    fp = unmatched_proposals + false_matched
    fn = unmatched_targets + false_matched
    return _proposal_metrics(
        proposal_count,
        gt_count,
        tp,
        fp,
        fn,
        matches,
        tp_distances,
        all_assignment_distances,
    )


def save_target_proposals_json(
    path: Path,
    proposals: list[TargetProposal],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "proposals": [asdict(proposal) for proposal in proposals],
    }
    path.write_text(json.dumps(payload, indent=2))


def save_target_proposals_ply(path: Path, proposals: list[TargetProposal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("peak_score", "f4"),
        ("support_count", "i4"),
        ("score_mass", "f4"),
        ("proposal_id", "i4"),
        ("source_index", "i4"),
    ]
    vertices = np.empty(len(proposals), dtype=dtype)
    for i, proposal in enumerate(proposals):
        point = np.asarray(proposal.representative_point, dtype=np.float32)
        vertices["x"][i], vertices["y"][i], vertices["z"][i] = point
        vertices["red"][i] = 0
        vertices["green"][i] = 255
        vertices["blue"][i] = 255
        vertices["peak_score"][i] = np.float32(proposal.peak_score)
        vertices["support_count"][i] = np.int32(proposal.support_count)
        vertices["score_mass"][i] = np.float32(proposal.score_mass)
        vertices["proposal_id"][i] = np.int32(proposal.proposal_id)
        vertices["source_index"][i] = np.int32(proposal.source_index)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def save_proposal_eval_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def _proposal_metrics(
    proposal_count: int,
    gt_count: int,
    tp: int,
    fp: int,
    fn: int,
    matches: list[ProposalMatch],
    tp_distances: list[float],
    all_assignment_distances: list[float] | None = None,
) -> dict[str, Any]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if tp_distances:
        mean_distance_m = float(np.mean(tp_distances))
        max_distance_m = float(np.max(tp_distances))
    else:
        mean_distance_m = float("inf")
        max_distance_m = float("inf")
    if all_assignment_distances:
        assignment_mean_distance_m = float(np.mean(all_assignment_distances))
        assignment_max_distance_m = float(np.max(all_assignment_distances))
    else:
        assignment_mean_distance_m = float("inf")
        assignment_max_distance_m = float("inf")

    return {
        "proposal_count": int(proposal_count),
        "gt_count": int(gt_count),
        "count_l1": int(abs(proposal_count - gt_count)),
        "exact_count": bool(proposal_count == gt_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance_m": mean_distance_m,
        "max_distance_m": max_distance_m,
        "assignment_mean_distance_m": assignment_mean_distance_m,
        "assignment_max_distance_m": assignment_max_distance_m,
        "matches": [asdict(match) for match in matches],
    }

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_THRESHOLD_M = 0.03


@dataclass(frozen=True)
class CutPointMatch:
    pred_index: int
    gt_index: int
    distance_m: float
    within_threshold: bool


@dataclass(frozen=True)
class SceneCutPointEvaluation:
    scene: str
    num_gt: int
    num_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    matched_distances_m: tuple[float, ...]
    gt_to_nearest_distances_m: tuple[float, ...]
    matches: tuple[CutPointMatch, ...]

    @property
    def matched_mean_distance(self) -> float | None:
        return _mean_or_none(self.matched_distances_m)

    @property
    def gt_to_nearest_mean_distance(self) -> float | None:
        return _mean_or_none(self.gt_to_nearest_distances_m)

    def to_metrics_dict(self, threshold_cm: int = 3, include_matches: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "scene": self.scene,
            "num_gt": self.num_gt,
            "num_pred": self.num_pred,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            f"Precision@{threshold_cm}cm": self.precision,
            f"Recall@{threshold_cm}cm": self.recall,
            f"F1@{threshold_cm}cm": self.f1,
            "matched_mean_distance": self.matched_mean_distance,
            "gt_to_nearest_mean_distance": self.gt_to_nearest_mean_distance,
        }
        if include_matches:
            row["matches"] = [
                {
                    "pred_index": match.pred_index,
                    "gt_index": match.gt_index,
                    "distance_m": match.distance_m,
                    "within_threshold": match.within_threshold,
                }
                for match in self.matches
            ]
            row["matched_distances_m"] = list(self.matched_distances_m)
            row["gt_to_nearest_distances_m"] = list(self.gt_to_nearest_distances_m)
        return row


def evaluate_cut_points(
    gt_points: np.ndarray | Iterable[Iterable[float]],
    pred_points: np.ndarray | Iterable[Iterable[float]],
    *,
    scene: str = "",
    threshold_m: float = DEFAULT_THRESHOLD_M,
) -> SceneCutPointEvaluation:
    """Evaluate unordered 3D cut points with one-to-one Hungarian matching."""

    gt = _as_points(gt_points, "gt_points")
    pred = _as_points(pred_points, "pred_points")
    num_gt = int(len(gt))
    num_pred = int(len(pred))

    if num_gt == 0 and num_pred == 0:
        return _scene_eval(scene, num_gt, num_pred, 0, 0, 0, [], [], [])
    if num_gt == 0:
        return _scene_eval(scene, num_gt, num_pred, 0, num_pred, 0, [], [], [])
    if num_pred == 0:
        return _scene_eval(scene, num_gt, num_pred, 0, 0, num_gt, [], [], [])

    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    invalid_cost = float(np.max(distances) + threshold_m * (min(num_pred, num_gt) + 1) + 1.0)
    assignment_cost = np.where(distances <= threshold_m, distances, invalid_cost)
    pred_rows, gt_cols = linear_sum_assignment(assignment_cost)
    matches: list[CutPointMatch] = []
    matched_tp_distances: list[float] = []
    tp = 0

    for pred_idx, gt_idx in zip(pred_rows.tolist(), gt_cols.tolist()):
        distance = float(distances[pred_idx, gt_idx])
        within_threshold = distance <= threshold_m
        matches.append(
            CutPointMatch(
                pred_index=int(pred_idx),
                gt_index=int(gt_idx),
                distance_m=distance,
                within_threshold=within_threshold,
            )
        )
        if within_threshold:
            tp += 1
            matched_tp_distances.append(distance)

    fp = num_pred - tp
    fn = num_gt - tp
    gt_to_nearest = distances.min(axis=0).astype(float).tolist()
    return _scene_eval(
        scene,
        num_gt,
        num_pred,
        tp,
        fp,
        fn,
        matched_tp_distances,
        gt_to_nearest,
        matches,
    )


def aggregate_cut_point_evaluations(
    evaluations: Iterable[SceneCutPointEvaluation],
    *,
    threshold_cm: int = 3,
) -> dict[str, Any]:
    rows = list(evaluations)
    tp = int(sum(row.tp for row in rows))
    fp = int(sum(row.fp for row in rows))
    fn = int(sum(row.fn for row in rows))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    matched_distances = [
        distance
        for row in rows
        for distance in row.matched_distances_m
    ]
    gt_to_nearest_distances = [
        distance
        for row in rows
        for distance in row.gt_to_nearest_distances_m
    ]
    return {
        "scene_count": len(rows),
        "total_gt": int(sum(row.num_gt for row in rows)),
        "total_pred": int(sum(row.num_pred for row in rows)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        f"Micro_Precision@{threshold_cm}cm": precision,
        f"Micro_Recall@{threshold_cm}cm": recall,
        f"Micro_F1@{threshold_cm}cm": f1,
        "matched_localization_distance": {
            "mean": _mean_or_none(matched_distances),
            "median": _median_or_none(matched_distances),
            "count": len(matched_distances),
        },
        "gt_to_nearest_distance": {
            "mean": _mean_or_none(gt_to_nearest_distances),
            "median": _median_or_none(gt_to_nearest_distances),
            "count": len(gt_to_nearest_distances),
        },
    }


def build_cut_point_report(
    evaluations: Iterable[SceneCutPointEvaluation],
    *,
    threshold_m: float = DEFAULT_THRESHOLD_M,
    metadata: dict[str, Any] | None = None,
    include_matches: bool = False,
) -> dict[str, Any]:
    rows = list(evaluations)
    threshold_cm = int(round(threshold_m * 100.0))
    return {
        "metadata": {
            "threshold_m": float(threshold_m),
            "threshold_cm": threshold_cm,
            **(metadata or {}),
        },
        "aggregate": aggregate_cut_point_evaluations(rows, threshold_cm=threshold_cm),
        "per_scene": [
            row.to_metrics_dict(threshold_cm=threshold_cm, include_matches=include_matches)
            for row in rows
        ],
    }


def save_cut_point_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


def save_cut_point_per_scene_csv(path: Path, report: dict[str, Any]) -> None:
    rows = report.get("per_scene", [])
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in rows[0] if key != "matches"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _scene_eval(
    scene: str,
    num_gt: int,
    num_pred: int,
    tp: int,
    fp: int,
    fn: int,
    matched_distances: Iterable[float],
    gt_to_nearest_distances: Iterable[float],
    matches: Iterable[CutPointMatch],
) -> SceneCutPointEvaluation:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return SceneCutPointEvaluation(
        scene=scene,
        num_gt=int(num_gt),
        num_pred=int(num_pred),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        precision=precision,
        recall=recall,
        f1=f1,
        matched_distances_m=tuple(float(value) for value in matched_distances),
        gt_to_nearest_distances_m=tuple(float(value) for value in gt_to_nearest_distances),
        matches=tuple(matches),
    )


def _as_points(points: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(list(points) if not isinstance(points, np.ndarray) else points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    try:
        array = array.reshape((-1, 3))
    except ValueError as exc:
        raise ValueError(f"{name} must be convertible to an Nx3 array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def _mean_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()) if array.size else None


def _median_or_none(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.median(array)) if array.size else None

"""Shared evidence utilities for the contact-guarded leaf selector."""


from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuleLeafCandidate:
    scene: str
    proposal_id: int
    representative_point: list[float]
    rank: int
    score: float
    support_count: int
    support_cluster_count: int
    fallback_reason: str | None
    score_terms: dict[str, float]


EVIDENCE_NAMES = (
    "proposal_score",
    "axis_length",
    "support_count",
    "contact_count",
    "contact_green_strength",
    "stem_proximity",
    "cluster_support",
)


def rule_candidate_from_json(scene: str, payload: dict[str, Any]) -> RuleLeafCandidate:
    return RuleLeafCandidate(
        scene=scene,
        proposal_id=int(payload["proposal_id"]),
        representative_point=[float(x) for x in payload["representative_point"]],
        rank=int(payload.get("rank", int(payload["proposal_id"]) + 1)),
        score=float(payload.get("score", 0.0)),
        support_count=int(payload.get("support_count", 0)),
        support_cluster_count=len(payload.get("support_cluster_ids", []) or []),
        fallback_reason=payload.get("fallback_reason"),
        score_terms={str(k): float(v) for k, v in (payload.get("score_terms", {}) or {}).items()},
    )


def leaf_consensus_quality(candidates: list[RuleLeafCandidate]) -> np.ndarray:
    if not candidates:
        return np.empty(0, dtype=np.float64)
    evidence = leaf_evidence_matrix(candidates)
    fallback_penalty = np.asarray(
        [1.0 if candidate.fallback_reason is not None else 0.0 for candidate in candidates],
        dtype=np.float64,
    )
    static_joint_penalty = np.asarray(
        [candidate.score_terms.get("static_joint_fallback", 0.0) for candidate in candidates],
        dtype=np.float64,
    )
    return evidence.mean(axis=1) - 0.20 * fallback_penalty - 0.15 * static_joint_penalty


def leaf_evidence_matrix(candidates: list[RuleLeafCandidate]) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(EVIDENCE_NAMES)), dtype=np.float64)
    score_terms = [candidate.score_terms for candidate in candidates]
    return np.stack(
        [
            _robust_unit([candidate.score for candidate in candidates]),
            _robust_unit([terms.get("axis_length_m", 0.0) for terms in score_terms]),
            _robust_unit([np.log1p(max(candidate.support_count, 0)) for candidate in candidates]),
            _robust_unit([np.log1p(max(terms.get("contact_count", 0.0), 0.0)) for terms in score_terms]),
            _robust_unit([terms.get("contact_green_strength", 0.0) for terms in score_terms]),
            _robust_unit(
                [min(terms.get("stem_distance_m", float("inf")), 0.20) for terms in score_terms],
                higher_is_better=False,
            ),
            np.asarray([1.0 if candidate.support_cluster_count > 0 else 0.0 for candidate in candidates]),
        ],
        axis=1,
    ).astype(np.float64)


def _robust_unit(values: list[float], higher_is_better: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=np.float64)
    finite_values = array[finite]
    if len(finite_values) == 1:
        lower = float(finite_values[0])
        upper = lower + 1.0
    else:
        lower, upper = np.percentile(finite_values, [10.0, 90.0]).astype(float).tolist()
    denominator = upper - lower
    if denominator <= 1.0e-9:
        denominator = max(abs(upper), 1.0)
    scaled = np.clip((array - lower) / denominator, 0.0, 1.0)
    scaled[~finite] = 0.0
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.astype(np.float64)


__all__ = [
    "EVIDENCE_NAMES",
    "RuleLeafCandidate",
    "leaf_consensus_quality",
    "leaf_evidence_matrix",
    "rule_candidate_from_json",
]

import numpy as np



STEM_PROXIMITY_INDEX = EVIDENCE_NAMES.index("stem_proximity")


def select_contact_guarded_rule_leaf_candidates(
    candidates: list[RuleLeafCandidate],
) -> list[RuleLeafCandidate]:
    """Select leaf targets by consensus while requiring contact proximity.

    The rule accepts candidates with at least 6 of 7 robust scene-local evidence
    votes, but the ignored channel cannot be stem/contact proximity. This
    enforces the physical expectation that a cut point lies near retained plant
    structure.
    """
    if not candidates:
        return []

    evidence = leaf_evidence_matrix(candidates)
    votes = np.sum(evidence >= 0.5, axis=1)
    selected_mask = (
        (votes >= len(EVIDENCE_NAMES) - 1)
        & (evidence[:, STEM_PROXIMITY_INDEX] >= 0.5)
    )

    if not np.any(selected_mask):
        quality = leaf_consensus_quality(candidates)
        selected_mask[int(np.argmax(quality))] = True

    selected = [
        candidate
        for candidate, keep in zip(candidates, selected_mask.tolist())
        if keep
    ]
    return sorted(selected, key=lambda candidate: candidate.rank)


__all__ = [
    "RuleLeafCandidate",
    "select_contact_guarded_rule_leaf_candidates",
    "rule_candidate_from_json",
]

