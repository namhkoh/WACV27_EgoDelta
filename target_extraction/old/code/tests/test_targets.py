from __future__ import annotations

import numpy as np

from egodelta_diff.targets import ProposalConfig, evaluate_proposals, generate_target_proposals


def test_generate_target_proposals_suppresses_nearby_peak() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.105, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.85, 0.7], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)

    proposals = generate_target_proposals(
        points,
        scores,
        source_indices,
        ProposalConfig(
            seed_score_min=0.5,
            nms_radius_m=0.02,
            support_radius_m=0.02,
            min_support=1,
            min_score_mass=0.0,
            min_peak_score=0.5,
        ),
    )

    assert len(proposals) == 2
    np.testing.assert_allclose(proposals[0].representative_point, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(proposals[1].representative_point, [0.10, 0.0, 0.0])


def test_evaluate_proposals_counts_fp_fn_with_distance_gate() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    proposals = generate_target_proposals(
        points,
        scores,
        source_indices,
        ProposalConfig(min_support=1),
    )
    gt_points = np.asarray([[0.005, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)

    metrics = evaluate_proposals(proposals, gt_points, gate_m=0.03)

    assert metrics["proposal_count"] == 2
    assert metrics["gt_count"] == 2
    assert metrics["count_l1"] == 0
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5

import math

import numpy as np

from egodelta_diff.targets import (
    aggregate_cut_point_evaluations,
    evaluate_cut_points,
)


def test_cut_point_eval_matches_unordered_variable_size_points() -> None:
    gt = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    pred = np.asarray(
        [[1.02, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    result = evaluate_cut_points(gt, pred, threshold_m=0.03)

    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 0
    assert math.isclose(result.precision, 2.0 / 3.0)
    assert result.recall == 1.0
    assert math.isclose(result.f1, 0.8)
    assert math.isclose(result.matched_mean_distance or 0.0, 0.015)
    assert math.isclose(result.gt_to_nearest_mean_distance or 0.0, 0.015)


def test_cut_point_eval_counts_above_threshold_assignment_as_fp_and_fn() -> None:
    result = evaluate_cut_points(
        gt_points=[[0.0, 0.0, 0.0]],
        pred_points=[[0.05, 0.0, 0.0]],
        threshold_m=0.03,
    )

    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.matched_mean_distance is None
    assert math.isclose(result.gt_to_nearest_mean_distance or 0.0, 0.05)


def test_cut_point_eval_prioritizes_valid_threshold_matches() -> None:
    gt = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.029, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pred = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0014137931, 0.0289655172, 0.0],
        ],
        dtype=np.float64,
    )

    result = evaluate_cut_points(gt, pred, threshold_m=0.03)

    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0


def test_cut_point_aggregate_uses_accumulated_pairs_not_scene_means() -> None:
    first = evaluate_cut_points(
        gt_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        pred_points=[[0.01, 0.0, 0.0], [1.02, 0.0, 0.0]],
        threshold_m=0.03,
    )
    second = evaluate_cut_points(
        gt_points=[[0.0, 0.0, 0.0]],
        pred_points=[[0.03, 0.0, 0.0]],
        threshold_m=0.03,
    )

    aggregate = aggregate_cut_point_evaluations([first, second])

    assert aggregate["TP"] == 3
    assert aggregate["FP"] == 0
    assert aggregate["FN"] == 0
    assert math.isclose(aggregate["matched_localization_distance"]["mean"], 0.02)
    assert math.isclose(aggregate["gt_to_nearest_distance"]["median"], 0.02)

from egodelta_diff.targets import (
    RuleLeafCandidate,
    select_contact_guarded_rule_leaf_candidates,
)


def test_contact_guarded_rule_rejects_candidate_with_weak_stem_proximity() -> None:
    candidates = [
        _candidate(0, score=1.0, axis=0.40, support=500, contact=120, green=0.20, stem=0.001),
        _candidate(1, score=0.8, axis=0.30, support=400, contact=100, green=0.19, stem=0.20),
        _candidate(2, score=0.1, axis=0.01, support=10, contact=1, green=0.01, stem=0.30),
    ]

    selected = select_contact_guarded_rule_leaf_candidates(candidates)

    assert [candidate.proposal_id for candidate in selected] == [0]


def test_contact_guarded_rule_falls_back_when_no_candidate_passes_guard() -> None:
    candidates = [
        _candidate(0, score=0.2, axis=0.01, support=10, contact=1, green=0.01, stem=0.20),
        _candidate(1, score=0.3, axis=0.02, support=20, contact=2, green=0.02, stem=0.25),
    ]

    selected = select_contact_guarded_rule_leaf_candidates(candidates)

    assert len(selected) == 1


def _candidate(
    proposal_id: int,
    *,
    score: float,
    axis: float,
    support: int,
    contact: int,
    green: float,
    stem: float,
) -> RuleLeafCandidate:
    return RuleLeafCandidate(
        scene="synthetic_leaf",
        proposal_id=proposal_id,
        representative_point=[float(proposal_id), 0.0, 0.0],
        rank=proposal_id + 1,
        score=score,
        support_count=support,
        support_cluster_count=1,
        fallback_reason=None,
        score_terms={
            "axis_length_m": axis,
            "contact_count": float(contact),
            "contact_green_strength": green,
            "stem_distance_m": stem,
            "static_joint_fallback": 0.0,
        },
    )

import numpy as np

from egodelta_diff.contact_proposal import (
    ContactProposal,
    ContactProposalConfig,
    _deduplicate_and_reindex,
    _deduplicate_leaf_support_units,
    _reindex_and_mark_primary,
    evaluate_contact_proposals,
    generate_contact_proposals,
)


def test_leaf_contact_uses_stem_side_endpoint_and_noise() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.10, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.0, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray([[0.1, 0.5, 0.1]] * len(points), dtype=np.float32)
    removed_mask = np.asarray([False, True, True, True, True])
    cluster_labels = np.asarray([-1, 0, 0, -1, -1], dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.05,
            leaf_endpoint_fraction=0.25,
            leaf_min_contact_points=1,
        ),
        object_type="leaf",
    )

    assert len(proposals) == 1
    np.testing.assert_allclose(proposals[0].representative_point, [0.01, 0.0, 0.0])
    assert proposals[0].source_index == 1
    assert proposals[0].method == "leaf_stem_endpoint"
    assert proposals[0].rank == 1
    assert proposals[0].primary is True
    assert proposals[0].primary_reason == "accepted"
    assert proposals[0].support_id.startswith("leaf:n")


def test_tomato_contact_splits_two_fruit_lobes_inside_one_cluster() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.125, 0.0, 0.0],
            [0.105, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.9, 0.8, 0.9, 0.9, 0.8], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray(
        [
            [0.8, 0.2, 0.1],
            [0.8, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.2, 0.1],
            [0.8, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    removed_mask = np.ones(len(points), dtype=bool)
    cluster_labels = np.zeros(len(points), dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            tomato_lobe_min_points=2,
            tomato_peak_score_min=0.89,
            tomato_peak_min_support=2,
            tomato_peak_support_radius_m=0.025,
            dedup_radius_m=0.03,
        ),
        object_type="tomato",
    )

    assert len(proposals) == 2
    assert {proposal.method for proposal in proposals} == {"tomato_high_score_contact_seed"}
    assert [proposal.rank for proposal in proposals] == [1, 2]
    assert [proposal.primary for proposal in proposals] == [True, True]

    gt_points = np.asarray([[0.01, 0.0, 0.0], [0.11, 0.0, 0.0]], dtype=np.float32)
    metrics = evaluate_contact_proposals(proposals, gt_points, gate_m=0.03)
    assert metrics["tp"] == 2


def test_tomato_without_peak_evidence_returns_no_proposals() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.9, 0.8], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray(
        [
            [0.8, 0.2, 0.1],
            [0.8, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    removed_mask = np.ones(len(points), dtype=bool)
    cluster_labels = np.zeros(len(points), dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            tomato_lobe_min_points=2,
            dedup_radius_m=0.03,
        ),
        object_type="tomato",
    )

    assert proposals == []


def test_leaf_contact_keeps_target_on_removed_mask() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.08, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.0, 0.9, 0.9, 0.9], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray([[0.1, 0.5, 0.1]] * len(points), dtype=np.float32)
    removed_mask = np.asarray([False, True, True, True])
    cluster_labels = np.asarray([-1, 0, 0, 0], dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.05,
            leaf_min_contact_points=1,
        ),
        object_type="leaf",
    )

    assert len(proposals) == 1
    assert proposals[0].method == "leaf_stem_endpoint"
    assert proposals[0].source_index in {1, 2, 3}
    assert proposals[0].source_index != 0


def test_leaf_contact_snaps_clusterless_endpoint_to_removed_core() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.009, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.08, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.0, 0.95, 0.8, 0.8, 0.8], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray([[0.1, 0.5, 0.1]] * len(points), dtype=np.float32)
    removed_mask = np.asarray([False, True, True, True, True])
    cluster_labels = np.asarray([-1, -1, 0, 0, 0], dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.02,
            leaf_endpoint_fraction=0.49,
            leaf_contact_radius_m=0.02,
            leaf_min_contact_points=1,
        ),
        object_type="leaf",
    )

    assert len(proposals) == 1
    assert proposals[0].source_index == 2
    assert proposals[0].score_terms["snapped_to_core"] == 1.0


def test_leaf_static_joint_fallback_can_be_primary_without_green_endpoint() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]]
        + [[0.05, y, 0.0] for y in np.linspace(-0.06, 0.06, 9)],
        dtype=np.float32,
    )
    scores = np.asarray([0.0] + [0.9] * (len(points) - 1), dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray([[0.1, 0.5, 0.1]] * len(points), dtype=np.float32)
    removed_mask = np.asarray([False] + [True] * (len(points) - 1))
    cluster_labels = np.asarray([-1] + [0] * (len(points) - 1), dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.02,
            leaf_endpoint_fraction=0.20,
            leaf_contact_radius_m=0.02,
            leaf_max_stem_distance_m=0.02,
            leaf_min_contact_points=2,
            primary_leaf_min_support_count=2,
            primary_leaf_min_axis_length_m=0.05,
            primary_leaf_min_contact_count=8,
            primary_leaf_min_contact_green_strength=0.18,
            primary_leaf_require_cluster=True,
        ),
        object_type="leaf",
    )

    assert len(proposals) == 1
    assert proposals[0].source_index == 1
    assert proposals[0].score_terms["static_joint_fallback"] == 1.0
    assert proposals[0].primary is True


def test_leaf_endpoint_prefers_stronger_contact_over_greener_wrong_side() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.004, 0.0],
            [0.0, -0.004, 0.0],
            [0.15, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.035, 0.0, 0.0],
            [0.07, 0.0, 0.0],
            [0.10, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.0, 0.0, 0.0, 0.0, 0.85, 0.85, 0.85, 0.85], dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray(
        [
            [0.2, 0.45, 0.1],
            [0.2, 0.45, 0.1],
            [0.2, 0.45, 0.1],
            [0.1, 0.4, 0.1],
            [0.2, 0.45, 0.1],
            [0.2, 0.45, 0.1],
            [0.2, 0.45, 0.1],
            [0.2, 0.45, 0.1],
        ],
        dtype=np.float32,
    )
    removed_mask = np.asarray([False, False, False, False, True, True, True, True])
    cluster_labels = np.asarray([-1, -1, -1, -1, 0, 0, 0, 0], dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.02,
            leaf_endpoint_fraction=0.25,
            leaf_contact_radius_m=0.06,
            leaf_max_stem_distance_m=0.08,
            leaf_min_contact_points=1,
            leaf_min_contact_green_strength=0.02,
            leaf_endpoint_contact_count_weight=0.020,
            leaf_endpoint_green_strength_weight=0.45,
            leaf_endpoint_distance_weight=2.0,
        ),
        object_type="leaf",
    )

    assert len(proposals) == 1
    assert proposals[0].source_index == 4
    assert proposals[0].score_terms["endpoint_side"] == 0.0


def test_leaf_support_units_dedup_overlapping_stem_subsets() -> None:
    units = [
        np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        np.asarray([1, 2, 3, 4], dtype=np.int64),
        np.asarray([20, 21, 22], dtype=np.int64),
    ]

    deduped = _deduplicate_leaf_support_units(units, min_overlap=0.80)

    assert [len(unit) for unit in deduped] == [6, 3]
    assert set(deduped[0].tolist()) == {0, 1, 2, 3, 4, 5}
    assert set(deduped[1].tolist()) == {20, 21, 22}


def test_leaf_proposal_dedup_skips_small_cluster_subset_duplicate() -> None:
    proposals = [
        ContactProposal(
            proposal_id=0,
            object_type="leaf",
            method="leaf_stem_endpoint",
            representative_point=[0.0, 0.0, 0.0],
            source_index=10,
            support_cluster_ids=[2, 3, 7, 9],
            support_count=2000,
            score=1.5,
            score_terms={"axis_length_m": 0.2, "contact_count": 100.0, "contact_green_strength": 0.2},
            stem_distance_m=0.001,
            contact_distance_m=0.001,
        ),
        ContactProposal(
            proposal_id=1,
            object_type="leaf",
            method="leaf_stem_endpoint",
            representative_point=[0.12, 0.0, 0.0],
            source_index=20,
            support_cluster_ids=[2, 3, 9],
            support_count=500,
            score=1.2,
            score_terms={"axis_length_m": 0.2, "contact_count": 100.0, "contact_green_strength": 0.2},
            stem_distance_m=0.001,
            contact_distance_m=0.001,
        ),
    ]

    deduped = _deduplicate_and_reindex(
        proposals,
        radius_m=0.025,
        max_count=64,
        config=ContactProposalConfig(
            leaf_support_cluster_dedup_min=0.80,
            leaf_support_cluster_dedup_max_support_ratio=0.75,
        ),
    )

    assert len(deduped) == 1
    assert deduped[0].source_index == 10


def test_leaf_primary_gate_does_not_apply_fixed_count_window() -> None:
    points = np.asarray(
        [[base + offset, 0.0, 0.0] for base in (0.0, 0.12, 0.24, 0.36) for offset in (0.0, 0.01, 0.04, 0.08)],
        dtype=np.float32,
    )
    scores = np.asarray([0.9] * len(points), dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray([[0.1, 0.5, 0.1]] * len(points), dtype=np.float32)
    removed_mask = np.asarray([False, True, True, True] * 4)
    cluster_labels = np.asarray([-1, 0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, -1, 3, 3, 3], dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.001,
        ),
        object_type="leaf",
    )

    assert len([proposal for proposal in proposals if proposal.primary]) == 4


def test_score_threshold_primary_policy_uses_confidence_not_count() -> None:
    points = np.asarray(
        [[i * 0.04, 0.0, 0.0] for i in range(9)] + [[0.0, 0.04, 0.0]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9] * len(points), dtype=np.float32)
    source_indices = np.arange(len(points), dtype=np.int64)
    rgb01 = np.asarray([[0.1, 0.5, 0.1]] * len(points), dtype=np.float32)
    removed_mask = np.asarray([True] * 9 + [False])
    cluster_labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, -1], dtype=np.int32)

    proposals = generate_contact_proposals(
        points,
        scores,
        source_indices,
        rgb01,
        removed_mask,
        cluster_labels,
        ContactProposalConfig(
            component_min_points=2,
            min_support_points=2,
            support_dilation_m=0.001,
            leaf_min_contact_points=1,
            primary_leaf_score_min=0.92,
        ),
        object_type="leaf",
    )

    assert len(proposals) == 3
    assert len([proposal for proposal in proposals if proposal.primary]) < 3
    assert all(proposal.score >= 0.92 for proposal in proposals if proposal.primary)


def test_leaf_primary_gate_can_skip_high_score_noise() -> None:
    proposals = [
        ContactProposal(
            proposal_id=0,
            object_type="leaf",
            method="leaf_stem_endpoint",
            representative_point=[0.0, 0.0, 0.0],
            source_index=10,
            support_cluster_ids=[],
            support_count=200,
            score=2.0,
            score_terms={"axis_length_m": 0.25, "contact_count": 100.0, "contact_green_strength": 0.2},
            stem_distance_m=0.001,
            contact_distance_m=0.001,
        ),
        ContactProposal(
            proposal_id=1,
            object_type="leaf",
            method="leaf_stem_endpoint",
            representative_point=[0.1, 0.0, 0.0],
            source_index=20,
            support_cluster_ids=[0],
            support_count=200,
            score=1.6,
            score_terms={"axis_length_m": 0.25, "contact_count": 100.0, "contact_green_strength": 0.2},
            stem_distance_m=0.001,
            contact_distance_m=0.001,
        ),
    ]

    marked = _reindex_and_mark_primary(
        proposals,
        ContactProposalConfig(
            primary_leaf_score_min=1.0,
            primary_leaf_min_support_count=100,
            primary_leaf_min_axis_length_m=0.15,
            primary_leaf_min_contact_count=8,
            primary_leaf_min_contact_green_strength=0.02,
            primary_leaf_require_cluster=True,
        ),
    )

    assert [proposal.primary for proposal in marked] == [False, True]
    assert marked[0].primary_reason == "leaf_missing_cluster"
    assert marked[1].primary_reason == "accepted"

