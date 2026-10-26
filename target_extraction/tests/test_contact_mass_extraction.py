from __future__ import annotations

import numpy as np
import inspect

from egodelta_diff import EndpointSide, extract_for_endpoint_side
from egodelta_diff.contact_mass_extraction import ExtractionConfig, extract_targets
from egodelta_diff.removed_surface_extraction import (
    RemovedSurfaceConfig,
    _complete_partition,
    extract_removed_surface_targets,
)


def test_confidence_partition_rejects_double_weighting_input():
    mu = np.asarray([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0]], dtype=float)
    normals = np.tile([0, 0, 1], (3, 1))
    delta = np.asarray([0.8, 0.1, 0.1])
    omega = np.asarray([0.7, 1.0, 1.0])
    overlap = np.ones(3, dtype=bool)
    try:
        extract_targets(mu, normals, delta, omega, overlap)
    except ValueError as error:
        assert "delta exceeds omega" in str(error)
    else:
        raise AssertionError("invalid confidence partition was accepted")


def test_task_agnostic_synthetic_interface_selects_retained_boundary():
    rng = np.random.default_rng(4)
    # Removed material extends along +X from a retained vertical interface.
    removed = np.column_stack(
        [rng.uniform(0.001, 0.035, 80), rng.normal(0, 0.001, 80), rng.normal(0, 0.001, 80)]
    )
    retained = np.column_stack(
        [rng.normal(0, 0.001, 80), rng.normal(0, 0.001, 80), rng.uniform(-0.02, 0.02, 80)]
    )
    mu = np.vstack([removed, retained])
    normals = np.tile([0, 1, 0], (len(mu), 1))
    delta = np.r_[np.full(len(removed), 0.9), np.full(len(retained), 0.1)]
    omega = np.ones(len(mu))
    overlap = np.ones(len(mu), dtype=bool)
    config = ExtractionConfig(
        dbscan_eps_m=0.006,
        dbscan_min_samples=4,
        normalized_removed_mass_min=0.0,
        nms_radius_m=0.03,
    )
    result = extract_targets(mu, normals, delta, omega, overlap, config=config)
    assert result.targets
    assert np.min(np.linalg.norm(result.points, axis=1)) < 0.015
    # The public call has no task or crop label.
    assert "task" not in extract_targets.__annotations__


def test_removed_surface_decoder_selects_gravity_facing_surface():
    rng = np.random.default_rng(12)
    phi = rng.uniform(0.0, 2.0 * np.pi, 300)
    costheta = rng.uniform(-1.0, 1.0, 300)
    theta = np.arccos(costheta)
    radius = 0.02
    mu = radius * np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    delta = np.full(len(mu), 0.95)
    overlap = np.ones(len(mu), dtype=bool)
    config = RemovedSurfaceConfig(
        dbscan_eps_m=0.02,
        dbscan_min_samples=4,
        component_mean_delta_min=0.8,
        component_size_min=20,
        mode_min_cluster_size=400,
    )
    result = extract_removed_surface_targets(mu, delta, overlap, config=config)
    assert len(result.targets) == 1
    assert result.targets[0].xyz[2] > 0.01


def test_dense_mode_completion_is_exhaustive_and_disjoint():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]]
    )
    units = _complete_partition(points, [np.asarray([0]), np.asarray([3])])
    assigned = np.concatenate(units)
    assert sorted(assigned.tolist()) == list(range(len(points)))
    assert len(np.unique(assigned)) == len(points)


def test_mechanics_router_exposes_endpoint_side_not_crop_or_task():
    parameters = inspect.signature(extract_for_endpoint_side).parameters
    assert "endpoint_side" in parameters
    assert "task" not in parameters
    assert "crop" not in parameters
    assert EndpointSide.RETAINED.value == "retained"
    assert EndpointSide.REMOVED.value == "removed"
