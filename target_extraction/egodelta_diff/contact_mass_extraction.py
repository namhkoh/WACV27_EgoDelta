"""Retained-side geometry-only Stage-3 endpoint decoder.

The implementation intentionally contains no legacy variants or ablation
branches.  It partitions Stage-2 confidence into removed and retained evidence,
forms standard DBSCAN instances, constructs a retained-side contact field,
detects and locally refines its modes, and emits one interface-boundary mode
from every sufficiently supported component.  The decoder applies when tool
mechanics define the target on structure that persists after the intervention.

Stage 3 reads no appearance feature.  Appearance can enter only through the
upstream continuous change score ``delta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class ExtractionConfig:
    # Shared Stage-2/instance setting.
    delta_threshold: float = 0.5
    # Three fitted Stage-3 values reported for the final method.
    dbscan_eps_m: float = 0.02
    dbscan_min_samples: int = 20
    normalized_removed_mass_min: float = 0.0037746118219953335
    # Derived/domain values, not dataset-fitted Stage-3 knobs.
    sigma_multiplier: float = 2.0
    nms_radius_m: float = 0.03


@dataclass(frozen=True)
class Target:
    component_id: int
    primitive_index: int
    source_index: int
    xyz: np.ndarray
    raw_xyz: np.ndarray
    component_removed_mass: float
    normalized_removed_mass: float
    m_peak: float
    one_sidedness: float
    boundary_distance_m: float
    boundary_score: float
    primitive_indices: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class ExtractionResult:
    sigma_m: float
    median_nearest_neighbor_m: float
    targets: list[Target]
    formed_component_count: int
    accepted_component_count: int
    overlap_primitive_count: int
    retained_primitive_count: int
    noise_primitive_count: int
    peak_count_before_nms: int
    peak_count_after_nms: int

    @property
    def points(self) -> np.ndarray:
        if not self.targets:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray([target.xyz for target in self.targets], dtype=np.float64)


def extract_targets(
    mu: np.ndarray,
    normals: np.ndarray,
    delta: np.ndarray,
    omega: np.ndarray,
    overlap: np.ndarray,
    source_indices: np.ndarray | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract one retained-side attachment endpoint per accepted component."""

    cfg = config or ExtractionConfig()
    _validate_config(cfg)
    mu, normals, delta, omega, overlap, source_indices = _validate_inputs(
        mu, normals, delta, omega, overlap, source_indices
    )
    selected = np.flatnonzero(overlap)
    if selected.size < 2:
        raise ValueError("Extraction needs at least two overlap primitives")

    points = mu[selected]
    normal = normals[selected]
    score = delta[selected]
    confidence = omega[selected]
    sources = source_indices[selected]

    violation = score - confidence
    if np.any(violation > 0.0):
        raise ValueError(
            "Cached delta exceeds omega; the confidence partition "
            f"is invalid (max violation {float(violation.max()):.9g})"
        )
    w_rem = score
    w_ret = (confidence - score) * (score <= cfg.delta_threshold)

    median_nn = _median_nonself_nn(points)
    if not np.isfinite(median_nn) or median_nn <= 0.0:
        raise ValueError(f"Invalid scene median nearest-neighbor distance: {median_nn}")
    sigma = float(cfg.sigma_multiplier * median_nn)

    removed = np.flatnonzero(score > cfg.delta_threshold)
    labels = _dbscan_labels(points[removed], cfg.dbscan_eps_m, cfg.dbscan_min_samples)
    component_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    component_indices = {
        component_id: removed[np.flatnonzero(labels == component_id)]
        for component_id in component_ids
    }
    retained = np.flatnonzero(w_ret > 0.0)
    retained_tree = cKDTree(points[retained]) if retained.size else None

    candidates: list[dict[str, object]] = []
    component_mass: dict[int, float] = {}
    for component_id in component_ids:
        component = component_indices[component_id]
        component_mass[component_id] = float(w_rem[component].sum())
        field_indices, rho = _retained_contact_field(
            points,
            normal,
            component,
            retained,
            retained_tree,
            w_rem,
            w_ret,
            sigma,
        )
        candidates.extend(
            _refined_peaks(points, field_indices, rho, sigma, component_id)
        )

    # Preserve the frozen evaluation order: modes first compete in scene-wide
    # 3 cm NMS, then the component gate and boundary selector are applied.
    kept = _global_nms(candidates, cfg.nms_radius_m)
    kept_by_component: dict[int, list[dict[str, object]]] = {}
    for candidate in kept:
        kept_by_component.setdefault(int(candidate["component_id"]), []).append(candidate)

    accepted_count = 0
    targets: list[Target] = []
    for component_id in component_ids:
        mass = component_mass[component_id]
        normalized_mass = mass / selected.size
        if normalized_mass < cfg.normalized_removed_mass_min:
            continue
        accepted_count += 1
        component_candidates = kept_by_component.get(component_id, [])
        if not component_candidates:
            continue
        component = component_indices[component_id]
        scored = [
            _boundary_score(candidate, points[component], w_rem[component], sigma)
            for candidate in component_candidates
        ]
        winner = sorted(
            scored,
            key=lambda row: (
                -float(row["boundary_score"]),
                -float(row["m_peak"]),
                int(row["primitive_index"]),
            ),
        )[0]
        primitive_index = int(winner["primitive_index"])
        targets.append(
            Target(
                component_id=component_id,
                primitive_index=int(selected[primitive_index]),
                source_index=int(sources[primitive_index]),
                xyz=np.asarray(winner["refined_xyz"], dtype=np.float64).copy(),
                raw_xyz=np.asarray(winner["raw_xyz"], dtype=np.float64).copy(),
                component_removed_mass=mass,
                normalized_removed_mass=float(normalized_mass),
                m_peak=float(winner["m_peak"]),
                one_sidedness=float(winner["one_sidedness"]),
                boundary_distance_m=float(winner["boundary_distance_m"]),
                boundary_score=float(winner["boundary_score"]),
                primitive_indices=selected[component].copy(),
            )
        )

    targets.sort(key=lambda target: target.component_id)
    return ExtractionResult(
        sigma_m=sigma,
        median_nearest_neighbor_m=median_nn,
        targets=targets,
        formed_component_count=len(component_ids),
        accepted_component_count=accepted_count,
        overlap_primitive_count=int(selected.size),
        retained_primitive_count=int(retained.size),
        noise_primitive_count=int(np.count_nonzero(labels < 0)),
        peak_count_before_nms=len(candidates),
        peak_count_after_nms=len(kept),
    )


def _retained_contact_field(
    points, normals, component, retained, retained_tree, w_rem, w_ret, sigma
):
    if retained_tree is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    accumulated: dict[int, float] = {}
    for removed_index in component:
        local = retained_tree.query_ball_point(points[removed_index], 3.0 * sigma)
        if not local:
            continue
        retained_indices = retained[np.asarray(local, dtype=np.int64)]
        displacement = points[retained_indices] - points[removed_index]
        kernel = np.exp(
            -np.einsum("ij,ij->i", displacement, displacement)
            / (2.0 * sigma * sigma)
        )
        pair_mass = (
            w_rem[removed_index]
            * w_ret[retained_indices]
            * kernel
            * np.abs(normals[retained_indices] @ normals[removed_index])
        )
        positive = pair_mass > 0.0
        for retained_index, value in zip(
            retained_indices[positive], pair_mass[positive], strict=True
        ):
            key = int(retained_index)
            accumulated[key] = accumulated.get(key, 0.0) + float(value)
    return (
        np.fromiter(accumulated.keys(), dtype=np.int64),
        np.fromiter(accumulated.values(), dtype=np.float64),
    )


def _refined_peaks(points, indices, rho, sigma, component_id):
    if not len(rho):
        return []
    field_points = points[indices]
    neighborhoods = cKDTree(field_points).query_ball_point(field_points, sigma)
    output = []
    for local_index, local_neighbors in enumerate(neighborhoods):
        neighbors = np.asarray(local_neighbors, dtype=np.int64)
        if rho[local_index] < np.max(rho[neighbors]):
            continue
        raw_xyz = field_points[local_index].copy()
        local = np.linalg.norm(field_points - raw_xyz, axis=1) <= sigma
        output.append(
            {
                "component_id": int(component_id),
                "primitive_index": int(indices[local_index]),
                "raw_xyz": raw_xyz,
                "refined_xyz": _weighted_centroid(field_points[local], rho[local]),
                "m_peak": float(rho[neighbors].sum()),
            }
        )
    return output


def _global_nms(candidates, radius):
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["m_peak"]),
            int(row["component_id"]),
            int(row["primitive_index"]),
        ),
    )
    if not ordered:
        return []
    raw = np.asarray([row["raw_xyz"] for row in ordered], dtype=np.float64)
    return [ordered[index] for index in _greedy_nms(raw, radius)]


def _greedy_nms(points, radius):
    cells: dict[tuple[int, int, int], list[int]] = {}
    kept: list[int] = []
    for index, point in enumerate(points):
        cell = tuple(np.floor(point / radius).astype(np.int64).tolist())
        suppress = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nearby = cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ())
                    if any(np.linalg.norm(point - points[other]) <= radius for other in nearby):
                        suppress = True
                        break
                if suppress:
                    break
            if suppress:
                break
        if not suppress:
            kept.append(index)
            cells.setdefault(cell, []).append(index)
    return kept


def _boundary_score(candidate, removed_points, removed_weights, sigma):
    xyz = np.asarray(candidate["refined_xyz"], dtype=np.float64)
    vector = removed_points - xyz
    distance = np.linalg.norm(vector, axis=1)
    unit = np.divide(
        vector,
        distance[:, None],
        out=np.zeros_like(vector),
        where=distance[:, None] > 0.0,
    )
    total = float(removed_weights.sum())
    one_sidedness = (
        float(np.linalg.norm(np.sum(removed_weights[:, None] * unit, axis=0)) / total)
        if total > 0.0
        else 0.0
    )
    boundary_distance = float(distance.min()) if len(distance) else float("inf")
    output = dict(candidate)
    output.update(
        one_sidedness=one_sidedness,
        boundary_distance_m=boundary_distance,
        boundary_score=float(
            one_sidedness * np.exp(-0.5 * (boundary_distance / sigma) ** 2)
        ),
    )
    return output


def _validate_inputs(mu, normals, delta, omega, overlap, source_indices):
    mu = np.asarray(mu, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=bool)
    if mu.ndim != 2 or mu.shape[1] != 3 or normals.shape != mu.shape:
        raise ValueError("mu and normals must both have shape (N, 3)")
    count = len(mu)
    if delta.shape != (count,) or omega.shape != (count,) or overlap.shape != (count,):
        raise ValueError("delta, omega, and overlap must have shape (N,)")
    if source_indices is None:
        source_indices = np.arange(count, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if source_indices.shape != (count,):
        raise ValueError("source_indices must have shape (N,)")
    if not (
        np.isfinite(mu).all()
        and np.isfinite(normals).all()
        and np.isfinite(delta).all()
        and np.isfinite(omega).all()
    ):
        raise ValueError("Inputs must be finite")
    if np.any(delta < 0.0) or np.any(omega < 0.0):
        raise ValueError("delta and omega must be non-negative")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Normals must be non-zero")
    return mu, normals / lengths[:, None], delta, omega, overlap, source_indices


def _validate_config(config):
    if not 0.0 <= config.delta_threshold <= 1.0:
        raise ValueError("delta_threshold must be in [0, 1]")
    if config.dbscan_eps_m <= 0.0 or config.dbscan_min_samples < 1:
        raise ValueError("DBSCAN settings must be positive")
    if config.normalized_removed_mass_min < 0.0:
        raise ValueError("normalized_removed_mass_min must be non-negative")
    if config.sigma_multiplier <= 0.0 or config.nms_radius_m <= 0.0:
        raise ValueError("Scale and NMS radius must be positive")


def _dbscan_labels(points, eps, min_samples):
    if not len(points):
        return np.empty(0, dtype=np.int64)
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)


def _median_nonself_nn(points):
    distance, _ = cKDTree(points).query(points, k=2, workers=-1)
    return float(np.median(distance[:, 1]))


def _weighted_centroid(points, weights):
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.mean(points, axis=0)
    return np.sum(points * weights[:, None], axis=0) / total
