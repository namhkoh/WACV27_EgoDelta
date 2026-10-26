"""Removed-side geometric target decoder for detach/grasp mechanics.

The removed object is present in the pre-action reconstruction, so its own
geometry and the known gravity direction define the acted-on endpoint.  Dense
HDBSCAN modes seed intervention units; a nearest-core completion assigns every
parent-component primitive to one unit before a gravity-facing surface mode is
locally refined.

This module consumes no color, crop label, hand trajectory, or class-specific
shape descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, HDBSCAN

from .contact_mass_extraction import _median_nonself_nn


@dataclass(frozen=True)
class RemovedSurfaceConfig:
    # Shared instance formation inherited by both endpoint-side decoders.
    delta_threshold: float = 0.5
    dbscan_eps_m: float = 0.02
    dbscan_min_samples: int = 20
    # Harvesting component-abstention settings, fitted in-sample on 30 scenes.
    component_mean_delta_min: float = 0.9008104936197888
    component_size_min: int = 205
    # Dense-mode separation setting, fitted in-sample on 30 scenes.
    mode_min_cluster_size: int = 120
    # Scene-derived/conventional values, not dataset-fitted thresholds.
    sigma_multiplier: float = 2.0
    density_tukey_iqr_multiplier: float = 1.5


@dataclass(frozen=True)
class RemovedSurfaceTarget:
    component_id: int
    mode_id: int
    primitive_index: int
    source_index: int
    xyz: np.ndarray
    raw_xyz: np.ndarray
    component_size: int
    component_mean_delta: float
    component_median_density: float
    density_fence: float
    primitive_indices: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class RemovedSurfaceResult:
    sigma_m: float
    median_nearest_neighbor_m: float
    targets: list[RemovedSurfaceTarget]
    formed_component_count: int
    accepted_component_count: int
    overlap_primitive_count: int
    noise_primitive_count: int

    @property
    def points(self) -> np.ndarray:
        if not self.targets:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray([target.xyz for target in self.targets], dtype=np.float64)


def extract_removed_surface_targets(
    mu: np.ndarray,
    delta: np.ndarray,
    overlap: np.ndarray,
    source_indices: np.ndarray | None = None,
    config: RemovedSurfaceConfig | None = None,
) -> RemovedSurfaceResult:
    """Extract removed-side gravity-facing surface targets.

    The endpoint side is explicit in this function's API.  It does not inspect
    a task or crop label.
    """

    cfg = config or RemovedSurfaceConfig()
    _validate_config(cfg)
    mu, delta, overlap, source_indices = _validate_inputs(
        mu, delta, overlap, source_indices
    )
    selected = np.flatnonzero(overlap)
    if selected.size < 2:
        raise ValueError("Extraction needs at least two overlap primitives")

    points = mu[selected]
    score = delta[selected]
    sources = source_indices[selected]
    median_nn = _median_nonself_nn(points)
    if not np.isfinite(median_nn) or median_nn <= 0.0:
        raise ValueError(f"Invalid scene median nearest-neighbor distance: {median_nn}")
    sigma = float(cfg.sigma_multiplier * median_nn)

    removed = np.flatnonzero(score > cfg.delta_threshold)
    labels = _dbscan_labels(
        points[removed], cfg.dbscan_eps_m, cfg.dbscan_min_samples
    )
    component_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)

    scene_density = cKDTree(points).query_ball_point(
        points, sigma, return_length=True
    )
    q1, q3 = np.percentile(scene_density, [25.0, 75.0])
    density_fence = float(q3 + cfg.density_tukey_iqr_multiplier * (q3 - q1))

    targets: list[RemovedSurfaceTarget] = []
    accepted_count = 0
    for component_id in component_ids:
        component = removed[np.flatnonzero(labels == component_id)]
        component_mean_delta = float(np.mean(score[component]))
        component_median_density = float(np.median(scene_density[component]))
        if not (
            len(component) >= cfg.component_size_min
            and component_mean_delta >= cfg.component_mean_delta_min
            and component_median_density <= density_fence
        ):
            continue
        accepted_count += 1
        component_points = points[component]
        cores = _dense_mode_cores(
            component_points,
            cfg.mode_min_cluster_size,
            cfg.dbscan_min_samples,
        )
        units = _complete_partition(component_points, cores)
        for mode_id, unit in enumerate(units):
            raw_local, raw_xyz, refined_xyz = _gravity_surface_peak(
                component_points[unit], score[component][unit], sigma
            )
            local_primitive = int(unit[raw_local])
            selected_index = int(component[local_primitive])
            targets.append(
                RemovedSurfaceTarget(
                    component_id=component_id,
                    mode_id=mode_id,
                    primitive_index=int(selected[selected_index]),
                    source_index=int(sources[selected_index]),
                    xyz=refined_xyz,
                    raw_xyz=raw_xyz,
                    component_size=int(len(component)),
                    component_mean_delta=component_mean_delta,
                    component_median_density=component_median_density,
                    density_fence=density_fence,
                    primitive_indices=selected[component[unit]].copy(),
                )
            )

    targets.sort(key=lambda target: (target.component_id, target.mode_id))
    return RemovedSurfaceResult(
        sigma_m=sigma,
        median_nearest_neighbor_m=median_nn,
        targets=targets,
        formed_component_count=len(component_ids),
        accepted_component_count=accepted_count,
        overlap_primitive_count=int(selected.size),
        noise_primitive_count=int(np.count_nonzero(labels < 0)),
    )


def _dense_mode_cores(points, min_cluster_size, min_samples):
    if len(points) < max(min_cluster_size, min_samples):
        return [np.arange(len(points), dtype=np.int64)]
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
        allow_single_cluster=True,
        n_jobs=-1,
    ).fit_predict(points)
    cores = [
        np.flatnonzero(labels == label)
        for label in sorted(int(value) for value in np.unique(labels) if value >= 0)
    ]
    return cores or [np.arange(len(points), dtype=np.int64)]


def _complete_partition(points, cores):
    """Assign every parent primitive to its nearest dense-mode primitive."""

    if len(cores) == 1:
        return [np.arange(len(points), dtype=np.int64)]
    distances = np.column_stack(
        [cKDTree(points[core]).query(points, k=1)[0] for core in cores]
    )
    owner = np.argmin(distances, axis=1)
    return [np.flatnonzero(owner == index) for index in range(len(cores))]


def _gravity_surface_peak(points, weights, sigma):
    """Return the locally refined central surface point facing world +Z."""

    center = np.average(points, axis=0, weights=weights)
    radial_to_gravity_axis = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    central = np.flatnonzero(radial_to_gravity_axis <= sigma)
    if not len(central):
        central = np.arange(len(points), dtype=np.int64)
    raw_index = int(central[np.argmax(points[central, 2])])
    raw_xyz = points[raw_index].copy()
    local = (
        (np.linalg.norm(points[:, :2] - raw_xyz[:2], axis=1) <= sigma)
        & (points[:, 2] >= raw_xyz[2] - sigma)
    )
    refined = np.average(points[local], axis=0, weights=weights[local])
    return raw_index, raw_xyz, np.asarray(refined, dtype=np.float64)


def _dbscan_labels(points, eps, min_samples):
    if not len(points):
        return np.empty(0, dtype=np.int64)
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)


def _validate_inputs(mu, delta, overlap, source_indices):
    mu = np.asarray(mu, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=bool)
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError("mu must have shape (N, 3)")
    count = len(mu)
    if delta.shape != (count,) or overlap.shape != (count,):
        raise ValueError("delta and overlap must have shape (N,)")
    if source_indices is None:
        source_indices = np.arange(count, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if source_indices.shape != (count,):
        raise ValueError("source_indices must have shape (N,)")
    if not np.isfinite(mu).all() or not np.isfinite(delta).all():
        raise ValueError("Inputs must be finite")
    if np.any(delta < 0.0):
        raise ValueError("delta must be non-negative")
    return mu, delta, overlap, source_indices


def _validate_config(config):
    if not 0.0 <= config.delta_threshold <= 1.0:
        raise ValueError("delta_threshold must be in [0, 1]")
    if config.dbscan_eps_m <= 0.0 or config.dbscan_min_samples < 1:
        raise ValueError("DBSCAN settings must be positive")
    if not 0.0 <= config.component_mean_delta_min <= 1.0:
        raise ValueError("component_mean_delta_min must be in [0, 1]")
    if config.component_size_min < 1 or config.mode_min_cluster_size < 1:
        raise ValueError("Component and mode sizes must be positive")
    if config.sigma_multiplier <= 0.0 or config.density_tukey_iqr_multiplier < 0.0:
        raise ValueError("Scale and density-fence settings must be non-negative")
