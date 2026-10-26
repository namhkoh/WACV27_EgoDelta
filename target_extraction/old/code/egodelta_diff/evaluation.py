"""Evaluation and failure diagnostics for contact-mass targets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .contact_mass_extraction import ContactMassResult
from .targets import SceneCutPointEvaluation, aggregate_cut_point_evaluations, evaluate_cut_points


def evaluate_contact_result(
    *,
    scene_id: str,
    task: str,
    result: ContactMassResult,
    mu: np.ndarray,
    delta: np.ndarray,
    omega: np.ndarray,
    overlap: np.ndarray,
    gt_points: np.ndarray,
    tau_c: float,
    threshold_m: float = 0.03,
) -> tuple[SceneCutPointEvaluation, list[dict[str, Any]]]:
    emitted = [component for component in result.components if component.emitted]
    predictions = np.asarray([c.predicted_xyz for c in emitted], dtype=np.float64).reshape((-1, 3))
    gt = np.asarray(gt_points, dtype=np.float64).reshape((-1, 3))
    evaluation = evaluate_cut_points(gt, predictions, scene=scene_id, threshold_m=threshold_m)
    matched_by_pred = {
        match.pred_index: match.gt_index for match in evaluation.matches if match.within_threshold
    }

    mu = np.asarray(mu, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    valid_retained = np.asarray(overlap, dtype=bool) & (delta <= tau_c)
    if np.any(delta > omega):
        raise ValueError("Cached delta exceeds omega; retained-mass partition is invalid")
    w_ret = (omega - delta) * valid_retained
    component_gt_sets: list[set[int]] = []
    gt_component_count = np.zeros(len(gt), dtype=np.int64)
    for component in result.components:
        comp_points = mu[component.primitive_indices]
        adjacent = {
            gt_idx
            for gt_idx, point in enumerate(gt)
            if len(comp_points) and np.min(np.linalg.norm(comp_points - point, axis=1)) <= 3.0 * result.sigma
        }
        component_gt_sets.append(adjacent)
        for gt_idx in adjacent:
            gt_component_count[gt_idx] += 1

    rows: list[dict[str, Any]] = []
    emitted_position = {component.component_id: idx for idx, component in enumerate(emitted)}
    for component in result.components:
        pred = component.predicted_xyz
        pred_idx = emitted_position.get(component.component_id)
        matched_gt_idx = matched_by_pred.get(pred_idx) if pred_idx is not None else None
        if pred is None:
            nearest_gt_idx, d_gt = (None, float("inf"))
        elif len(gt):
            distances = np.linalg.norm(gt - pred, axis=1)
            nearest_gt_idx = int(np.argmin(distances))
            d_gt = float(distances[nearest_gt_idx])
        else:
            nearest_gt_idx, d_gt = (None, float("inf"))
        diagnostic_gt_idx = matched_gt_idx if matched_gt_idx is not None else nearest_gt_idx
        diagnostic_gt = gt[diagnostic_gt_idx] if diagnostic_gt_idx is not None else None

        within_gt = (
            np.linalg.norm(mu - diagnostic_gt, axis=1) <= 3.0 * result.sigma
            if diagnostic_gt is not None
            else np.zeros(len(mu), dtype=bool)
        )
        m_gt_ret = float(w_ret[within_gt].sum())
        if diagnostic_gt is not None and len(component.rho_indices):
            rho_points = mu[component.rho_indices]
            rho_near = np.linalg.norm(rho_points - diagnostic_gt, axis=1) <= 3.0 * result.sigma
            m_gt_rho = float(component.rho_values[rho_near].sum())
        else:
            m_gt_rho = 0.0
        ext_norm = _rho_extent(component, mu, result.sigma)
        axis = _axis_diagnostics(diagnostic_gt, pred, mu, w_ret, result.sigma, matched_gt_idx is not None)
        comp_gt = component_gt_sets[component.component_id]
        n_c_per_gt = int(gt_component_count[diagnostic_gt_idx]) if diagnostic_gt_idx is not None else 0
        error_cm = float(np.linalg.norm(pred - diagnostic_gt) * 100.0) if pred is not None and diagnostic_gt is not None else None
        rows.append(
            {
                "scene_id": scene_id,
                "task": task,
                "variant": result.variant,
                "component_id": component.component_id,
                "conf_c": component.conf_c,
                "prominence": component.prominence,
                "n_primitives_in_component": int(len(component.primitive_indices)),
                "component_removed_mass": component.removed_mass,
                "sum_w_ret_within_3sigma": m_gt_ret,
                "m_gt_ret": m_gt_ret,
                "m_gt_rho": m_gt_rho,
                "ext_norm": ext_norm,
                "predicted_xyz": _point_or_none(pred),
                "removed_centroid_xyz": _point_or_none(component.removed_centroid),
                "matched_gt_xyz": _point_or_none(gt[matched_gt_idx]) if matched_gt_idx is not None else None,
                "diagnostic_gt_xyz": _point_or_none(diagnostic_gt),
                "error_cm": error_cm,
                "matched": matched_gt_idx is not None,
                "emitted": component.emitted,
                "zero_mass_cause": component.zero_mass_cause,
                "no_normal_contact_mass": component.no_normal_contact_mass,
                "retained_neighbors_within_3sigma": component.retained_neighbors_within_3sigma,
                "d_gt": d_gt,
                "d_rem": float(np.linalg.norm(pred - component.removed_centroid)) if pred is not None else None,
                "n_gt_in_c": len(comp_gt),
                "n_c_per_gt": n_c_per_gt,
                **axis,
            }
        )
    return evaluation, rows


def finalize_failure_labels(
    rows: list[dict[str, Any]],
    *,
    ret_cutoff: float = 0.1,
    rho_cutoff: float = 0.1,
    extent_cutoff: float = 3.0,
) -> dict[str, int]:
    """Normalize masses per task and derive deterministic D/C/A1/A2/B labels."""

    medians: dict[str, tuple[float, float, int]] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        matched = [row for row in rows if row["task"] == task and row["matched"]]
        ret = np.asarray([row["m_gt_ret"] for row in matched], dtype=np.float64)
        rho = np.asarray([row["m_gt_rho"] for row in matched], dtype=np.float64)
        medians[task] = (_positive_median(ret), _positive_median(rho), len(matched))

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        ret_med, rho_med, n_matched = medians[str(row["task"])]
        ret_norm = float(row["m_gt_ret"] / ret_med) if ret_med > 0.0 else float("inf")
        rho_norm = float(row["m_gt_rho"] / rho_med) if rho_med > 0.0 else float("inf")
        row["m_gt_ret_norm"] = ret_norm
        row["m_gt_rho_norm"] = rho_norm
        row["normalizer_n_matched"] = n_matched
        row["normalization_low_confidence"] = n_matched < 20
        label, rationale = classify_failure(
            row,
            ret_norm=ret_norm,
            rho_norm=rho_norm,
            ret_cutoff=ret_cutoff,
            rho_cutoff=rho_cutoff,
            extent_cutoff=extent_cutoff,
        )
        row["failure_class"] = label
        row["rationale"] = rationale
        row.setdefault("auto_label_confidence", _auto_label_confidence(row, label))
        row.setdefault("human_label", "")
        counts[label] += 1
    return dict(counts)


def classify_failure(row, *, ret_norm, rho_norm, ret_cutoff, rho_cutoff, extent_cutoff):
    if float(row.get("d_gt", float("inf"))) > 0.06:
        return "D", f"d_gt={row['d_gt']:.6g} > 0.06"
    if int(row.get("n_gt_in_c", 0)) >= 2 or int(row.get("n_c_per_gt", 0)) >= 2:
        return "C", f"n_gt_in_c={row.get('n_gt_in_c', 0)}, n_c_per_gt={row.get('n_c_per_gt', 0)}"
    if ret_norm < ret_cutoff:
        return "A1", f"m_gt_ret_norm={ret_norm:.6g} < {ret_cutoff:.6g}"
    if rho_norm < rho_cutoff:
        return "A2", f"m_gt_ret_norm={ret_norm:.6g} >= {ret_cutoff:.6g} and m_gt_rho_norm={rho_norm:.6g} < {rho_cutoff:.6g}"
    if float(row.get("ext_norm", 0.0)) > extent_cutoff:
        return "B", f"m_gt_rho_norm={rho_norm:.6g} >= {rho_cutoff:.6g} and ext_norm={row['ext_norm']:.6g} > {extent_cutoff:.6g}"
    return "unresolved", "no D/C/A1/A2/B rule fired"


def aggregate_evaluations(evaluations: Iterable[SceneCutPointEvaluation], threshold_m: float):
    return aggregate_cut_point_evaluations(
        list(evaluations), threshold_cm=int(round(threshold_m * 100.0))
    )


def mass_distribution(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not len(values):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    q = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "p25", "median", "p75", "max"), q.tolist())) | {"n": len(values)}


def cutoff_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for ret_cutoff in (0.05, 0.1, 0.2):
        for rho_cutoff in (0.05, 0.1, 0.2):
            for extent_cutoff in (2.0, 3.0, 4.0):
                counts: dict[str, int] = defaultdict(int)
                for row in rows:
                    label, _ = classify_failure(
                        row,
                        ret_norm=float(row["m_gt_ret_norm"]),
                        rho_norm=float(row["m_gt_rho_norm"]),
                        ret_cutoff=ret_cutoff,
                        rho_cutoff=rho_cutoff,
                        extent_cutoff=extent_cutoff,
                    )
                    counts[label] += 1
                output.append(
                    {"ret_cutoff": ret_cutoff, "rho_cutoff": rho_cutoff, "extent_cutoff": extent_cutoff, **counts}
                )
    return output


def _rho_extent(component, mu, sigma):
    if component.conf_c <= 0.0 or not len(component.rho_indices):
        return float("inf")
    distance = np.linalg.norm(mu[component.rho_indices] - component.predicted_xyz, axis=1)
    order = np.argsort(distance)
    cumulative = np.cumsum(component.rho_values[order])
    idx = int(np.searchsorted(cumulative, 0.8 * component.conf_c, side="left"))
    return float(distance[order[min(idx, len(order) - 1)]] / sigma)


def _axis_diagnostics(gt, pred, mu, w_ret, sigma, matched):
    base = {
        "along_err": None,
        "radial_err": None,
        "axis_world_up_dot": None,
        "axis_status": "not_matched" if not matched else "axis_undefined",
        "axis_retained_count": 0,
        "axis_eigenvalue_ratio": None,
    }
    if not matched or gt is None or pred is None:
        return base
    mask = (w_ret > 0.0) & (np.linalg.norm(mu - gt, axis=1) <= 3.0 * sigma)
    idx = np.flatnonzero(mask)
    base["axis_retained_count"] = int(len(idx))
    if len(idx) < 10:
        return base
    points, weights = mu[idx], w_ret[idx]
    center = np.sum(points * weights[:, None], axis=0) / weights.sum()
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ratio = float(eigenvalues[-1] / eigenvalues[-2]) if eigenvalues[-2] > 0.0 else float("inf")
    base["axis_eigenvalue_ratio"] = ratio
    if ratio < 2.0:
        base["axis_status"] = "axis_degenerate"
        return base
    axis = eigenvectors[:, -1]
    if axis[2] < 0.0:
        axis = -axis
    error = pred - gt
    along = float(error @ axis)
    radial = float(np.linalg.norm(error - along * axis))
    base.update(
        along_err=along,
        radial_err=radial,
        axis_world_up_dot=float(axis[2]),
        axis_status="ok",
    )
    return base


def _positive_median(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else 0.0


def _point_or_none(point):
    return None if point is None else [float(value) for value in point]


def _auto_label_confidence(row, label):
    if label == "unresolved":
        return "low"
    if label in {"A1", "A2", "B"}:
        return "medium"
    return "high"
