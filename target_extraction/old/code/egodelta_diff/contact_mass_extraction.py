"""Crop-agnostic contact-mass interface extraction.

Stage 3 consumes no appearance features; appearance enters only through the
upstream change score.  Removed evidence lives on the pre-action primitives.
The estimator constructs the contact field on both sides of every interface,
detects its local modes, and returns ``(g_ret, g_rem)`` endpoint pairs.  A
downstream tool convention, not a crop/task branch, selects the endpoint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


Side = Literal["retained", "removed"]
Variant = Literal["primary", "E3-1", "E3-2", "E3-3", "E3-3b", "E3-4", "E3-6"]


@dataclass(frozen=True)
class ContactMassConfig:
    """Declared prediction-affecting values in the frozen estimator.

    The DBSCAN values are inherited unchanged from the legacy grouping stage.
    ``nms_radius_m`` is the evaluation/tool resolution, and
    ``deployment_offset_m`` is an agronomic domain constant that must be zero
    for evaluation.  The Gaussian query truncation at three sigma is
    structural.  There is deliberately no component-mass percentile filter.
    """

    sigma_multiplier: float = 2.0
    tau_c: float = 0.5
    dbscan_eps_m: float = 0.02
    dbscan_min_samples: int = 20
    nms_radius_m: float = 0.03
    deployment_offset_m: float = 0.005


@dataclass
class ContactMassGroup:
    """One DBSCAN component and its two contact fields."""

    component_id: int
    primitive_indices: np.ndarray
    source_indices: np.ndarray
    removed_mass: float
    removed_centroid: np.ndarray
    retained_rho_indices: np.ndarray = field(repr=False)
    retained_rho_values: np.ndarray = field(repr=False)
    removed_rho_indices: np.ndarray = field(repr=False)
    removed_rho_values: np.ndarray = field(repr=False)
    retained_peak_count: int
    removed_peak_count: int
    retained_neighbors_within_3sigma: int
    boundary_to_retained_m: float | None
    no_normal_contact_mass: float
    zero_mass_cause: str | None


@dataclass
class ContactMassComponent:
    """A peak-anchored interface pair.

    ``anchor_side`` records which side supplied the peak and NMS decision.
    Both endpoints are still returned.  The compatibility names
    ``predicted_xyz`` and ``conf_c`` mean the selected endpoint and its local
    peak-contact mass, respectively.
    """

    component_id: int
    source_component_id: int
    peak_id: int
    anchor_side: Side
    primitive_indices: np.ndarray
    source_indices: np.ndarray
    removed_mass: float
    removed_centroid: np.ndarray
    retained_xyz: np.ndarray | None
    removed_xyz: np.ndarray | None
    retained_raw_xyz: np.ndarray | None
    removed_raw_xyz: np.ndarray | None
    retained_m_peak: float | None
    removed_m_peak: float | None
    predicted_xyz: np.ndarray
    deployment_xyz: np.ndarray
    conf_c: float
    prominence: float
    pair_gap_m: float | None
    bidirectional_gap_m: float | None
    rho_indices: np.ndarray = field(repr=False)
    rho_values: np.ndarray = field(repr=False)
    retained_neighbors_within_3sigma: int
    no_normal_contact_mass: float
    zero_mass_cause: str | None = None

    @property
    def emitted(self) -> bool:
        return True

    def endpoint(self, side: Side) -> np.ndarray | None:
        """Return the mechanically selected endpoint without a task branch."""

        _validate_side(side)
        return self.retained_xyz if side == "retained" else self.removed_xyz


@dataclass
class ContactMassResult:
    sigma: float
    median_nearest_neighbor_distance: float
    variant: str
    components: list[ContactMassComponent]
    retained_targets: list[ContactMassComponent]
    removed_targets: list[ContactMassComponent]
    groups: list[ContactMassGroup]
    formed_component_count: int
    dropped_component_count: int
    dropped_components: list[dict[str, object]]
    overlap_primitive_count: int
    retained_primitive_count: int
    noise_primitive_count: int
    abstained_component_count: int
    e3_2_skipped_count: int = 0
    e3_2_skipped_fraction: float = 0.0
    e3_2_clipped_count: int = 0

    def targets(self, side: Side) -> list[ContactMassComponent]:
        """Select interface pairs by tool-side convention."""

        _validate_side(side)
        return self.retained_targets if side == "retained" else self.removed_targets

    def points(self, side: Side, *, deployment: bool = False) -> np.ndarray:
        """Return selected endpoint coordinates in score/NMS order."""

        targets = self.targets(side)
        key = "deployment_xyz" if deployment else "predicted_xyz"
        if not targets:
            return np.empty((0, 3), dtype=np.float64)
        if all(target.anchor_side == side for target in targets):
            return np.asarray([getattr(target, key) for target in targets], dtype=np.float64)
        return np.asarray([target.endpoint(side) for target in targets], dtype=np.float64)


def extract_contact_mass_targets(
    mu: np.ndarray,
    normals: np.ndarray,
    delta: np.ndarray,
    omega: np.ndarray,
    overlap: np.ndarray,
    source_indices: np.ndarray | None = None,
    config: ContactMassConfig | None = None,
    *,
    variant: Variant = "primary",
) -> ContactMassResult:
    """Extract refined contact modes on both sides of each interface.

    The cached score already contains reconstruction confidence.  The primary
    weights therefore partition confidence without recovering the unstable raw
    score: ``w_rem = delta`` and ``w_ret = omega - delta``.  Grouping remains
    on cached ``delta``.  ``variant='E3-3'`` only changes the compatibility
    ``components`` view to the removed side; both endpoint sets are computed in
    every primary/mirror call.
    """

    cfg = config or ContactMassConfig()
    _validate_config(cfg)
    mu, normals, delta, omega, overlap, source_indices = _validate_inputs(
        mu, normals, delta, omega, overlap, source_indices
    )
    selected = np.flatnonzero(overlap)
    if selected.size < 2:
        raise ValueError("Contact-mass extraction needs at least two overlap primitives")

    points = mu[selected]
    nrm = normals[selected]
    score = delta[selected]
    confidence = omega[selected]
    sources = source_indices[selected]

    median_nn = _median_nonself_nn(points)
    if not np.isfinite(median_nn) or median_nn <= 0.0:
        raise ValueError(f"Invalid scene scale from median nearest-neighbor distance: {median_nn}")
    sigma = float(cfg.sigma_multiplier * median_nn)

    w_rem, w_ret_base, skipped, clipped = _soft_weights(score, confidence, variant)
    retained_mask = score <= cfg.tau_c
    if variant == "E3-6":
        retained_mask = np.ones_like(retained_mask)
    w_ret = w_ret_base * retained_mask

    grouped_local = np.flatnonzero(score > cfg.tau_c)
    labels = _dbscan_labels(points[grouped_local], cfg.dbscan_eps_m, cfg.dbscan_min_samples)
    component_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    component_indices = [grouped_local[np.flatnonzero(labels == label)] for label in component_ids]
    noise_count = int(np.count_nonzero(labels < 0))

    retained_indices = np.flatnonzero(w_ret > 0.0)
    retained_tree = cKDTree(points[retained_indices]) if retained_indices.size else None
    groups: list[ContactMassGroup] = []
    retained_candidates: list[dict[str, object]] = []
    removed_candidates: list[dict[str, object]] = []
    local_retained: dict[int, list[dict[str, object]]] = {}
    local_removed: dict[int, list[dict[str, object]]] = {}

    use_normals = variant != "E3-1"
    for component_id, comp_idx in zip(component_ids, component_indices, strict=True):
        removed_centroid = _weighted_centroid(points[comp_idx], w_rem[comp_idx])
        if variant == "E3-3b":
            ret_idx = np.empty(0, dtype=np.int64)
            rho_ret = np.empty(0, dtype=np.float64)
            rem_idx, rho_rem, neighbor_count, no_normal_mass = _self_mass(
                points, nrm, comp_idx, w_rem, sigma
            )
        else:
            ret_idx, rho_ret, rem_idx, rho_rem, neighbor_count, no_normal_mass = _contact_fields(
                points,
                nrm,
                comp_idx,
                retained_indices,
                retained_tree,
                w_rem,
                w_ret,
                sigma,
                use_normals=use_normals,
            )
        ret_peaks = _refined_peaks(points, ret_idx, rho_ret, sigma, component_id, "retained")
        rem_peaks = _refined_peaks(points, rem_idx, rho_rem, sigma, component_id, "removed")
        _annotate_peak_prominence(points, ret_idx, rho_ret, sigma, ret_peaks)
        _annotate_peak_prominence(points, rem_idx, rho_rem, sigma, rem_peaks)
        local_retained[component_id] = ret_peaks
        local_removed[component_id] = rem_peaks
        retained_candidates.extend(ret_peaks)
        removed_candidates.extend(rem_peaks)
        contact_mass = float(rho_ret.sum())
        if contact_mass > 0.0 or float(rho_rem.sum()) > 0.0:
            cause = None
        elif neighbor_count == 0:
            cause = "no_retained_neighbors"
        elif no_normal_mass > 0.0:
            cause = "normal_suppression"
        else:
            cause = "other_zero_mass"
        if retained_tree is None or not len(comp_idx):
            boundary_to_retained = None
        else:
            nearest, _ = retained_tree.query(points[comp_idx], k=1)
            boundary_to_retained = float(np.min(nearest))
        groups.append(
            ContactMassGroup(
                component_id=component_id,
                primitive_indices=selected[comp_idx],
                source_indices=sources[comp_idx],
                removed_mass=float(w_rem[comp_idx].sum()),
                removed_centroid=removed_centroid,
                retained_rho_indices=selected[ret_idx],
                retained_rho_values=rho_ret,
                removed_rho_indices=selected[rem_idx],
                removed_rho_values=rho_rem,
                retained_peak_count=len(ret_peaks),
                removed_peak_count=len(rem_peaks),
                retained_neighbors_within_3sigma=neighbor_count,
                boundary_to_retained_m=boundary_to_retained,
                no_normal_contact_mass=no_normal_mass,
                zero_mass_cause=cause,
            )
        )

    kept_retained = _global_nms(retained_candidates, cfg.nms_radius_m)
    kept_removed = _global_nms(removed_candidates, cfg.nms_radius_m)
    group_by_id = {group.component_id: group for group in groups}
    retained_targets = _make_interface_pairs(
        kept_retained,
        local_removed,
        local_retained,
        group_by_id,
        selected,
        anchor_side="retained",
        deployment_offset_m=cfg.deployment_offset_m,
    )
    removed_targets = _make_interface_pairs(
        kept_removed,
        local_retained,
        local_removed,
        group_by_id,
        selected,
        anchor_side="removed",
        deployment_offset_m=cfg.deployment_offset_m,
    )
    components = removed_targets if variant in {"E3-3", "E3-3b"} else retained_targets
    abstained = sum(group.retained_peak_count == 0 and group.removed_peak_count == 0 for group in groups)
    return ContactMassResult(
        sigma=sigma,
        median_nearest_neighbor_distance=median_nn,
        variant=variant,
        components=components,
        retained_targets=retained_targets,
        removed_targets=removed_targets,
        groups=groups,
        formed_component_count=len(groups),
        dropped_component_count=0,
        dropped_components=[],
        overlap_primitive_count=int(selected.size),
        retained_primitive_count=int(retained_indices.size),
        noise_primitive_count=noise_count,
        abstained_component_count=int(abstained),
        e3_2_skipped_count=int(skipped),
        e3_2_skipped_fraction=float(skipped / len(points)),
        e3_2_clipped_count=int(clipped),
    )


def _soft_weights(score, confidence, variant):
    if np.any(score > confidence):
        worst = float(np.max(score - confidence))
        count = int(np.count_nonzero(score > confidence))
        raise ValueError(
            f"Cached delta exceeds omega for {count} primitives (max violation {worst:.9g}); "
            "omega = w_rem + w_ret would be invalid"
        )
    if variant == "E3-2":
        valid = confidence >= 0.01
        raw = np.zeros_like(score)
        raw[valid] = score[valid] / confidence[valid]
        clipped = int(np.count_nonzero(valid & ((raw < 0.0) | (raw > 1.0))))
        raw = np.clip(raw, 0.0, 1.0)
        return raw * valid, (1.0 - raw) * valid, int(np.count_nonzero(~valid)), clipped
    if variant == "E3-4":
        removed = score > 0.5
        return confidence * removed, confidence * ~removed, 0, 0
    return score.copy(), confidence - score, 0, 0


def _contact_fields(
    points, normals, component, retained, retained_tree, w_rem, w_ret, sigma, *, use_normals
):
    if retained_tree is None:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_i.copy(), empty_f.copy(), 0, 0.0
    retained_accum: dict[int, float] = {}
    removed_indices: list[int] = []
    removed_rho: list[float] = []
    neighbor_set: set[int] = set()
    no_normal_mass = 0.0
    for j in component:
        local = retained_tree.query_ball_point(points[j], 3.0 * sigma)
        if not local:
            continue
        ii = retained[np.asarray(local, dtype=np.int64)]
        diff = points[ii] - points[j]
        kernel = np.exp(-np.einsum("ij,ij->i", diff, diff) / (2.0 * sigma * sigma))
        base = w_rem[j] * w_ret[ii] * kernel
        no_normal_mass += float(base.sum())
        neighbor_set.update(int(i) for i in ii[base > 0.0])
        pair_mass = base * np.abs(normals[ii] @ normals[j]) if use_normals else base
        positive = pair_mass > 0.0
        if not np.any(positive):
            continue
        ii = ii[positive]
        pair_mass = pair_mass[positive]
        removed_indices.append(int(j))
        removed_rho.append(float(pair_mass.sum()))
        for i, value in zip(ii, pair_mass, strict=True):
            retained_accum[int(i)] = retained_accum.get(int(i), 0.0) + float(value)
    retained_idx = np.fromiter(retained_accum.keys(), dtype=np.int64)
    retained_rho = np.fromiter(retained_accum.values(), dtype=np.float64)
    return (
        retained_idx,
        retained_rho,
        np.asarray(removed_indices, dtype=np.int64),
        np.asarray(removed_rho, dtype=np.float64),
        len(neighbor_set),
        no_normal_mass,
    )


def _refined_peaks(points, indices, rho, sigma, component_id, side):
    peaks = _detect_peaks(points, indices, rho, sigma, component_id, side)
    field_points = points[indices]
    for peak in peaks:
        local = np.linalg.norm(field_points - peak["raw_xyz"], axis=1) <= sigma
        peak["refined_xyz"] = _weighted_centroid(field_points[local], rho[local])
    return peaks


def _annotate_peak_prominence(points, indices, rho, radius, peaks):
    """Attach parameter-free graph prominence to every local contact mode.

    The rho-support graph uses the same ``sigma`` neighborhood already used
    for local-mode detection.  Prominence is the peak height minus the highest
    saddle connecting its basin to a strictly higher peak.  The component-wide
    maximum uses zero as the field boundary.  No rejection threshold is
    applied here; downstream evaluation can sweep the continuous score.
    """

    values = _topographic_prominence(
        points,
        indices,
        rho,
        radius,
        [int(peak["primitive_index"]) for peak in peaks],
    )
    for peak in peaks:
        info = values[int(peak["primitive_index"])]
        peak["prominence"] = float(info["prominence"])
        peak["saddle_height"] = float(info["saddle_height"])
        peak["is_component_global_max"] = bool(info["is_global_max"])


def _topographic_prominence(points, indices, rho, radius, peak_global_indices):
    """Compute local-maximum prominence with a descending union-find sweep."""

    indices = np.asarray(indices, dtype=np.int64)
    rho = np.asarray(rho, dtype=np.float64)
    if not len(rho):
        return {}

    local_of_global = {int(value): pos for pos, value in enumerate(indices)}
    peak_local = [local_of_global[int(value)] for value in peak_global_indices]
    peak_ids_at_node: dict[int, list[int]] = defaultdict(list)
    for local, global_index in zip(peak_local, peak_global_indices, strict=True):
        peak_ids_at_node[local].append(int(global_index))

    neighborhoods = cKDTree(points[indices]).query_ball_point(points[indices], radius)
    order = np.argsort(-rho, kind="stable")
    parent = np.arange(len(rho), dtype=np.int64)
    active = np.zeros(len(rho), dtype=bool)
    summit = np.full(len(rho), -np.inf, dtype=np.float64)
    alive: list[set[int]] = [set() for _ in range(len(rho))]
    output: dict[int, dict[str, float | bool]] = {}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first, second, level):
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            return root_a
        height_a, height_b = summit[root_a], summit[root_b]
        if height_a < height_b or (height_a == height_b and root_a > root_b):
            root_a, root_b = root_b, root_a
            height_a, height_b = height_b, height_a
        parent[root_b] = root_a
        if height_a > height_b:
            for peak_id in alive[root_b]:
                height = float(rho[local_of_global[peak_id]])
                output[peak_id] = {
                    "peak_height": height,
                    "saddle_height": float(level),
                    "prominence": float(max(height - level, 0.0)),
                    "is_global_max": False,
                }
            alive[root_b].clear()
        else:
            alive[root_a].update(alive[root_b])
            alive[root_b].clear()
        summit[root_a] = max(height_a, height_b)
        return root_a

    for node in order.tolist():
        active[node] = True
        parent[node] = node
        if node in peak_ids_at_node:
            alive[node].update(peak_ids_at_node[node])
            summit[node] = float(rho[node])
        for neighbor in neighborhoods[node]:
            neighbor = int(neighbor)
            if neighbor != node and active[neighbor]:
                root = union(node, neighbor, float(rho[node]))
                if summit[root] == -np.inf:
                    summit[root] = float(rho[node])

    for peak_id in peak_global_indices:
        peak_id = int(peak_id)
        if peak_id not in output:
            height = float(rho[local_of_global[peak_id]])
            output[peak_id] = {
                "peak_height": height,
                "saddle_height": 0.0,
                "prominence": height,
                "is_global_max": True,
            }
    return output


def _detect_peaks(points, indices, rho, sigma, component_id, side):
    if not len(rho):
        return []
    tree = cKDTree(points[indices])
    neighborhoods = tree.query_ball_point(points[indices], sigma)
    output = []
    for local_idx, local_neighbors in enumerate(neighborhoods):
        neighbors = np.asarray(local_neighbors, dtype=np.int64)
        if rho[local_idx] < np.max(rho[neighbors]):
            continue
        output.append(
            {
                "component_id": int(component_id),
                "side": side,
                "primitive_index": int(indices[local_idx]),
                "raw_xyz": points[indices[local_idx]].copy(),
                "rho_at_peak": float(rho[local_idx]),
                "m_peak": float(rho[neighbors].sum()),
            }
        )
    return output


def _global_nms(candidates, radius):
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["m_peak"]), int(row["component_id"]), int(row["primitive_index"])),
    )
    if not ordered:
        return []
    raw = np.asarray([row["raw_xyz"] for row in ordered], dtype=np.float64)
    kept = [ordered[index] for index in _greedy_nms(raw, radius)]
    for rank, row in enumerate(kept, start=1):
        row["rank_after_nms"] = rank
    return kept


def _greedy_nms(points, radius):
    cells: dict[tuple[int, int, int], list[int]] = {}
    kept: list[int] = []
    for idx, point in enumerate(points):
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
            kept.append(idx)
            cells.setdefault(cell, []).append(idx)
    return kept


def _make_interface_pairs(
    anchors,
    opposite_by_component,
    same_by_component,
    groups,
    selected,
    *,
    anchor_side,
    deployment_offset_m,
):
    output = []
    for output_id, anchor in enumerate(anchors):
        source_component_id = int(anchor["component_id"])
        group = groups[source_component_id]
        opposite = opposite_by_component.get(source_component_id, [])
        paired = None
        bidirectional_gap = None
        if opposite:
            distances = [
                np.linalg.norm(candidate["refined_xyz"] - anchor["refined_xyz"])
                for candidate in opposite
            ]
            paired = opposite[int(np.argmin(distances))]
            same = same_by_component.get(source_component_id, [])
            if same:
                reverse = [
                    np.linalg.norm(candidate["refined_xyz"] - paired["refined_xyz"])
                    for candidate in same
                ]
                p_prime = same[int(np.argmin(reverse))]
                bidirectional_gap = float(
                    np.linalg.norm(anchor["refined_xyz"] - p_prime["refined_xyz"])
                )
        if anchor_side == "retained":
            retained, removed = anchor, paired
            rho_indices, rho_values = group.retained_rho_indices, group.retained_rho_values
        else:
            retained, removed = paired, anchor
            rho_indices, rho_values = group.removed_rho_indices, group.removed_rho_values
        retained_xyz = None if retained is None else retained["refined_xyz"].copy()
        removed_xyz = None if removed is None else removed["refined_xyz"].copy()
        predicted = anchor["refined_xyz"].copy()
        counterpart = removed_xyz if anchor_side == "retained" else retained_xyz
        deployment = _deployment_point(predicted, counterpart, group.removed_centroid, deployment_offset_m)
        pair_gap = (
            None
            if retained_xyz is None or removed_xyz is None
            else float(np.linalg.norm(retained_xyz - removed_xyz))
        )
        output.append(
            ContactMassComponent(
                component_id=output_id,
                source_component_id=source_component_id,
                peak_id=int(anchor["primitive_index"]),
                anchor_side=anchor_side,
                primitive_indices=group.primitive_indices,
                source_indices=group.source_indices,
                removed_mass=group.removed_mass,
                removed_centroid=group.removed_centroid,
                retained_xyz=retained_xyz,
                removed_xyz=removed_xyz,
                retained_raw_xyz=None if retained is None else retained["raw_xyz"].copy(),
                removed_raw_xyz=None if removed is None else removed["raw_xyz"].copy(),
                retained_m_peak=None if retained is None else float(retained["m_peak"]),
                removed_m_peak=None if removed is None else float(removed["m_peak"]),
                predicted_xyz=predicted,
                deployment_xyz=deployment,
                conf_c=float(anchor["m_peak"]),
                prominence=float(anchor["prominence"]),
                pair_gap_m=pair_gap,
                bidirectional_gap_m=bidirectional_gap,
                rho_indices=rho_indices,
                rho_values=rho_values,
                retained_neighbors_within_3sigma=group.retained_neighbors_within_3sigma,
                no_normal_contact_mass=group.no_normal_contact_mass,
            )
        )
    return output


def _self_mass(points, normals, component, w_rem, sigma):
    tree = cKDTree(points[component])
    accum = np.zeros(len(component), dtype=np.float64)
    for out_idx, j in enumerate(component):
        local = np.asarray(tree.query_ball_point(points[j], 3.0 * sigma), dtype=np.int64)
        ii = component[local]
        diff = points[ii] - points[j]
        kernel = np.exp(-np.einsum("ij,ij->i", diff, diff) / (2.0 * sigma * sigma))
        accum[out_idx] = w_rem[j] * np.sum(w_rem[ii] * kernel * np.abs(normals[ii] @ normals[j]))
    return component.copy(), accum, int(np.count_nonzero(accum)), float(accum.sum())


def _validate_inputs(mu, normals, delta, omega, overlap, source_indices):
    mu = np.asarray(mu, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=bool)
    if mu.ndim != 2 or mu.shape[1] != 3 or normals.shape != mu.shape:
        raise ValueError("mu and normals must both have shape (N, 3)")
    n = len(mu)
    if delta.shape != (n,) or omega.shape != (n,) or overlap.shape != (n,):
        raise ValueError("delta, omega, and overlap must have shape (N,)")
    if source_indices is None:
        source_indices = np.arange(n, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if source_indices.shape != (n,):
        raise ValueError("source_indices must have shape (N,)")
    finite = (
        np.isfinite(mu).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(delta)
        & np.isfinite(omega)
    )
    if not finite.all():
        raise ValueError("Contact-mass inputs must be finite")
    if np.any(delta < 0.0) or np.any(omega < 0.0):
        raise ValueError("delta and omega must be non-negative")
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths <= 0.0):
        raise ValueError("Normals must be non-zero")
    normals = normals / normal_lengths[:, None]
    return mu, normals, delta, omega, overlap, source_indices


def _validate_config(config):
    if config.sigma_multiplier <= 0.0 or config.dbscan_eps_m <= 0.0:
        raise ValueError("sigma_multiplier and dbscan_eps_m must be positive")
    if config.dbscan_min_samples < 1:
        raise ValueError("dbscan_min_samples must be positive")
    if config.nms_radius_m <= 0.0:
        raise ValueError("nms_radius_m must be positive")


def _validate_side(side):
    if side not in {"retained", "removed"}:
        raise ValueError(f"side must be 'retained' or 'removed', got {side!r}")


def _dbscan_labels(points, eps, min_samples):
    if not len(points):
        return np.empty(0, dtype=np.int64)
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)


def _median_nonself_nn(points: np.ndarray) -> float:
    distance, _ = cKDTree(points).query(points, k=2, workers=-1)
    return float(np.median(distance[:, 1]))


def _weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.mean(points, axis=0)
    return np.sum(points * weights[:, None], axis=0) / total


def _deployment_point(predicted, counterpart, removed_centroid, offset):
    if offset == 0.0:
        return predicted.copy()
    destination = counterpart if counterpart is not None else removed_centroid
    direction = destination - predicted
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return predicted.copy()
    return predicted + float(offset) * direction / length
