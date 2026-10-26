from __future__ import annotations

import numpy as np
import torch

from egodelta_diff.primitives import cluster_removed
from egodelta_diff.primitives import GaussianSet2D


def _gaussian_set(points: np.ndarray) -> GaussianSet2D:
    n = len(points)
    rotation = torch.eye(3, dtype=torch.float32).repeat(n, 1, 1)
    return GaussianSet2D(
        mu=torch.from_numpy(points.astype(np.float32)),
        rotation=rotation,
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1),
        scales=torch.ones((n, 2), dtype=torch.float32) * 0.01,
        normal=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(n, 1),
        opacity=torch.ones(n, dtype=torch.float32),
        dc_color=torch.zeros((n, 3), dtype=torch.float32),
        sh_rest=torch.empty((n, 0, 3), dtype=torch.float32),
        source_indices=torch.arange(n, dtype=torch.long),
    )


def test_cluster_keeps_large_volume_component_when_axis_is_long() -> None:
    t = np.linspace(0.0, 1.0, 120, dtype=np.float32)
    points = np.stack([0.2 * t, 0.2 * t, 0.5 * t], axis=1)
    g = _gaussian_set(points)
    removed_mask = torch.ones(len(points), dtype=torch.bool)
    scores = torch.ones(len(points), dtype=torch.float32)

    rejected, rejected_labels = cluster_removed(
        g,
        removed_mask,
        scores,
        eps=0.02,
        min_samples=4,
        min_volume=1.0e-5,
        max_volume=1.0e-2,
    )
    assert rejected == []
    assert torch.all(rejected_labels == -1)

    kept, kept_labels = cluster_removed(
        g,
        removed_mask,
        scores,
        eps=0.02,
        min_samples=4,
        min_volume=1.0e-5,
        max_volume=1.0e-2,
        min_axis_length_m=0.15,
        keep_large_axis_components=True,
    )
    assert len(kept) == 1
    assert kept[0].axis_length_m >= 0.15
    assert torch.all(kept_labels == 0)

import torch

from egodelta_diff.primitives import estimate_geometric_drift, weighted_quantile
from egodelta_diff.primitives import GaussianSet2D, quaternion_to_matrix


def _drift_gaussians(points: torch.Tensor) -> GaussianSet2D:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(len(points), 1)
    rotation = quaternion_to_matrix(q)
    return GaussianSet2D(
        mu=points,
        rotation=rotation,
        quaternion=q,
        scales=torch.ones(len(points), 2) * 0.01,
        normal=rotation[:, :, 2],
        opacity=torch.ones(len(points)),
        dc_color=torch.zeros(len(points), 3),
        sh_rest=torch.empty(len(points), 0, 3),
        source_indices=torch.arange(len(points)),
    )


def test_weighted_quantile_prefers_high_weight_value():
    value = weighted_quantile(torch.tensor([0.0, 10.0]), torch.tensor([1.0, 100.0]), 0.5)
    assert value.item() == 10.0


def test_geometric_drift_separates_normal_and_tangent():
    a = _drift_gaussians(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    b = _drift_gaussians(torch.tensor([[0.0, 0.0, 0.2], [1.1, 0.0, 0.0]]))
    un2, ut2 = estimate_geometric_drift(a, b, quantile=1.0)
    assert abs(un2**0.5 - 0.2) < 1e-5
    assert abs(ut2**0.5 - 0.1) < 1e-5

import torch

from egodelta_diff.primitives import in_shared_frustum
from egodelta_diff.primitives import GaussianSet2D, quaternion_to_matrix
from egodelta_diff.data import Camera


def test_frustum_keeps_front_point_and_rejects_behind_point():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    rotation = quaternion_to_matrix(q)
    g = GaussianSet2D(
        mu=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        rotation=rotation,
        quaternion=q,
        scales=torch.ones(2, 2) * 0.01,
        normal=rotation[:, :, 2],
        opacity=torch.ones(2),
        dc_color=torch.zeros(2, 3),
        sh_rest=torch.empty(2, 0, 3),
        source_indices=torch.arange(2),
    )
    cam = Camera(
        id=0,
        img_name="",
        width=100,
        height=100,
        fx=50.0,
        fy=50.0,
        cx=50.0,
        cy=50.0,
        center=torch.zeros(3),
        rotation=torch.eye(3),
    )
    mask = in_shared_frustum(g, [cam], near=0.01, far=10.0)
    assert mask.tolist() == [True, False]

import torch

from egodelta_diff.primitives import GaussianSet2D, build_effective_covariance, quaternion_to_matrix


def _toy_gaussians() -> GaussianSet2D:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    rotation = quaternion_to_matrix(q)
    return GaussianSet2D(
        mu=torch.zeros(2, 3),
        rotation=rotation,
        quaternion=q,
        scales=torch.ones(2, 2) * 0.1,
        normal=rotation[:, :, 2],
        opacity=torch.ones(2),
        dc_color=torch.zeros(2, 3),
        sh_rest=torch.empty(2, 0, 3),
        source_indices=torch.arange(2),
    )


def test_quaternion_to_matrix_identity():
    mat = quaternion_to_matrix(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(mat[0], torch.eye(3), atol=1e-6)


def test_2dgs_covariance_is_rank_two_before_inflation():
    g = _toy_gaussians()
    eig = torch.linalg.eigvalsh(g.cov_3d)
    assert torch.allclose(eig[:, 0], torch.zeros(2), atol=1e-7)


def test_effective_covariance_is_full_rank_with_drift():
    g = _toy_gaussians()
    h = torch.eye(3).expand(2, -1, -1)
    sigma, _ = build_effective_covariance(g, h, u_n2=0.01, u_t2=0.01)
    eig = torch.linalg.eigvalsh(sigma)
    assert bool((eig > 0).all())

import torch

from egodelta_diff.primitives import (
    aggregate_score,
    compute_confidence,
    score_primitives,
    score_primitives_fixed_radius,
)
from egodelta_diff.primitives import GaussianSet2D, quaternion_to_matrix


def _scoring_gaussians(points: torch.Tensor, color: torch.Tensor) -> GaussianSet2D:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(len(points), 1)
    rotation = quaternion_to_matrix(q)
    return GaussianSet2D(
        mu=points,
        rotation=rotation,
        quaternion=q,
        scales=torch.ones(len(points), 2) * 0.01,
        normal=rotation[:, :, 2],
        opacity=torch.ones(len(points)),
        dc_color=color,
        sh_rest=torch.empty(len(points), 0, 3),
        source_indices=torch.arange(len(points)),
    )


def test_score_primitives_identical_match_has_low_change():
    g = _scoring_gaussians(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.1, 0.2, 0.3]]))
    sigma = torch.eye(3).unsqueeze(0) * 0.1
    delta_geo, delta_app = score_primitives(g, g, sigma, sigma, sigma_c2=0.1, show_progress=False)
    assert delta_geo.item() < 1e-6
    assert delta_app.item() < 1e-6


def test_aggregate_score_clamps_before_confidence():
    delta = aggregate_score(torch.tensor([0.8]), torch.tensor([0.8]), torch.tensor([0.5]))
    assert delta.item() == 0.5


def test_aggregate_score_supports_scoring_ablation_modes():
    geo = torch.tensor([0.7])
    app = torch.tensor([0.2])
    omega = torch.tensor([0.5])

    assert torch.isclose(aggregate_score(geo, app, omega, mode="full"), torch.tensor([0.45])).item()
    assert torch.isclose(aggregate_score(geo, app, omega, mode="geometry_only"), torch.tensor([0.35])).item()
    assert torch.isclose(aggregate_score(geo, app, omega, mode="appearance_only"), torch.tensor([0.1])).item()
    assert torch.isclose(
        aggregate_score(geo, app, omega, mode="full", use_confidence_weight=False),
        torch.tensor([0.9]),
    ).item()


def test_confidence_shape():
    h = torch.eye(3).expand(4, -1, -1)
    omega = compute_confidence(h)
    assert omega.shape == (4,)


def test_fixed_radius_scoring_identical_match_has_low_change():
    g = _scoring_gaussians(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.1, 0.2, 0.3]]))
    delta_geo, delta_app = score_primitives_fixed_radius(g, g, sigma_c2=0.1, radius_m=0.05, show_progress=False)
    assert delta_geo.item() < 1e-6
    assert delta_app.item() < 1e-6


def test_fixed_radius_scoring_missing_neighbor_has_high_change():
    g_a = _scoring_gaussians(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.1, 0.2, 0.3]]))
    g_b = _scoring_gaussians(torch.tensor([[0.2, 0.0, 0.0]]), torch.tensor([[0.1, 0.2, 0.3]]))
    delta_geo, delta_app = score_primitives_fixed_radius(
        g_a,
        g_b,
        sigma_c2=0.1,
        radius_m=0.05,
        show_progress=False,
    )
    assert delta_geo.item() == 1.0
    assert delta_app.item() == 1.0
