from __future__ import annotations

import inspect

import numpy as np
import pytest

from egodelta_diff.contact_mass_extraction import (
    ContactMassConfig,
    _topographic_prominence,
    extract_contact_mass_targets,
)


def _scene():
    mu = np.asarray([
        [-0.02, 0.0, 0.00], [-0.01, 0.0, 0.00],
        [0.00, 0.0, -0.02], [0.00, 0.0, -0.01],
        [0.00, 0.0, 0.00], [0.00, 0.0, 0.01], [0.00, 0.0, 0.02],
        [0.50, 0.0, 0.00], [0.51, 0.0, 0.00],
    ], dtype=np.float64)
    normals = np.tile([0.0, 1.0, 0.0], (len(mu), 1))
    delta = np.asarray([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    return mu, normals, delta, np.ones(len(mu)), np.ones(len(mu), dtype=bool)


def _config(**changes):
    values = {
        "tau_c": 0.3,
        "dbscan_eps_m": 0.021,
        "dbscan_min_samples": 1,
        "nms_radius_m": 0.03,
        "deployment_offset_m": 0.0,
    }
    values.update(changes)
    return ContactMassConfig(**values)


def test_api_has_no_task_crop_or_rgb_input():
    parameters = inspect.signature(extract_contact_mass_targets).parameters
    assert {"task", "crop", "object_type", "rgb", "rgb01", "color"}.isdisjoint(parameters)


def test_same_call_handles_synthetic_leaf_and_fruit_without_label():
    arrays = _scene()
    cfg = _config()
    leaf = extract_contact_mass_targets(*arrays, config=cfg)
    fruit = extract_contact_mass_targets(*arrays, config=cfg)
    np.testing.assert_allclose(leaf.components[0].predicted_xyz, fruit.components[0].predicted_xyz)


def test_scale_is_derived_and_rescales_with_scene():
    arrays = _scene()
    cfg = _config(dbscan_eps_m=0.084)
    first = extract_contact_mass_targets(*arrays, config=cfg)
    second = extract_contact_mass_targets(arrays[0] * 4.0, *arrays[1:], config=cfg)
    assert second.sigma == pytest.approx(4.0 * first.sigma)


def test_global_retained_mask_excludes_removed_component():
    arrays = _scene()
    result = extract_contact_mass_targets(*arrays, config=_config())
    assert set(result.components[0].rho_indices.tolist()).isdisjoint({0, 1, 2, 3})


def test_normal_sign_is_invariant():
    arrays = _scene()
    cfg = _config()
    first = extract_contact_mass_targets(*arrays, config=cfg)
    second = extract_contact_mass_targets(arrays[0], -arrays[1], *arrays[2:], config=cfg)
    np.testing.assert_allclose(first.components[0].predicted_xyz, second.components[0].predicted_xyz)
    assert first.components[0].conf_c == pytest.approx(second.components[0].conf_c)


def test_no_retained_neighbor_abstains_without_fallback():
    mu, normals, delta, omega, overlap = _scene()
    delta[:] = 0.9
    result = extract_contact_mass_targets(mu, normals, delta, omega, overlap, config=_config())
    assert not result.components
    assert result.abstained_component_count == result.formed_component_count
    assert result.groups[0].zero_mass_cause == "no_retained_neighbors"


def test_normal_suppression_is_distinguished():
    mu, normals, delta, omega, overlap = _scene()
    normals[:4] = [1.0, 0.0, 0.0]
    normals[4:] = [0.0, 1.0, 0.0]
    component = extract_contact_mass_targets(
        mu, normals, delta, omega, overlap,
        config=_config(),
    ).groups[0]
    assert component.retained_rho_values.sum() == 0.0
    assert component.no_normal_contact_mass > 0.0
    assert component.zero_mass_cause == "normal_suppression"


def test_true_mirror_lives_on_removed_side():
    arrays = _scene()
    cfg = _config()
    primary = extract_contact_mass_targets(*arrays, config=cfg, variant="primary")
    mirror = extract_contact_mass_targets(*arrays, config=cfg, variant="E3-3")
    assert primary.components[0].anchor_side == "retained"
    assert mirror.components[0].anchor_side == "removed"
    assert set(mirror.components[0].rho_indices).issubset(set(mirror.components[0].primitive_indices))


def test_result_exposes_interface_pair_without_task_branch():
    result = extract_contact_mass_targets(*_scene(), config=_config())
    assert result.retained_targets
    assert result.removed_targets
    retained = result.targets("retained")[0]
    removed = result.targets("removed")[0]
    assert retained.retained_xyz is not None and retained.removed_xyz is not None
    assert removed.retained_xyz is not None and removed.removed_xyz is not None
    assert retained.pair_gap_m is not None
    assert retained.prominence >= 0.0
    assert removed.prominence >= 0.0


def test_topographic_prominence_uses_highest_connecting_saddle():
    points = np.asarray(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=np.float64,
    )
    rho = np.asarray([5.0, 1.0, 3.0], dtype=np.float64)
    result = _topographic_prominence(
        points,
        np.arange(3),
        rho,
        0.011,
        [0, 2],
    )

    assert result[0]["prominence"] == pytest.approx(5.0)
    assert result[0]["is_global_max"] is True
    assert result[2]["saddle_height"] == pytest.approx(1.0)
    assert result[2]["prominence"] == pytest.approx(2.0)
    assert result[2]["is_global_max"] is False


def test_corrected_weights_partition_confidence():
    mu, normals, delta, omega, overlap = _scene()
    omega[:] = 0.8
    delta[:] = np.minimum(delta, omega)
    result = extract_contact_mass_targets(mu, normals, delta, omega, overlap, config=_config())
    grouped = np.concatenate([group.primitive_indices for group in result.groups])
    expected = sum(float(delta[index]) for index in grouped)
    assert sum(group.removed_mass for group in result.groups) == pytest.approx(expected)


def test_delta_above_omega_fails_partition_assertion():
    mu, normals, delta, omega, overlap = _scene()
    omega[0] = 0.5
    with pytest.raises(ValueError, match="delta exceeds omega"):
        extract_contact_mass_targets(mu, normals, delta, omega, overlap, config=_config())


def test_deployment_offset_zero_returns_prediction():
    result = extract_contact_mass_targets(*_scene(), config=_config())
    np.testing.assert_allclose(result.components[0].deployment_xyz, result.components[0].predicted_xyz)


def test_invalid_scene_scale_fails_explicitly():
    mu = np.zeros((2, 3))
    normals = np.tile([0, 0, 1.0], (2, 1))
    with pytest.raises(ValueError, match="scene scale"):
        extract_contact_mass_targets(mu, normals, np.ones(2), np.ones(2), np.ones(2, dtype=bool))
