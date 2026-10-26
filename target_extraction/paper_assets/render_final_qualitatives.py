#!/usr/bin/env python
"""Render the frozen final Stage-3 qualitative panels.

This is a visualization-only replay.  It does not alter caches, configurations,
predictions, or evaluation outputs.  GT is used only to choose representative
matched/error-quantile panels and to annotate the figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egodelta_diff.contact_mass_extraction import (  # noqa: E402
    ExtractionConfig,
    _median_nonself_nn,
    _refined_peaks,
    _retained_contact_field,
)
from egodelta_diff.data import find_scene_targets, parse_gt_markdown  # noqa: E402
from egodelta_diff.removed_surface_extraction import (  # noqa: E402
    RemovedSurfaceConfig,
    _complete_partition,
    _dense_mode_cores,
    _gravity_surface_peak,
)
from egodelta_diff.stage2_cache import load_stage2_cache  # noqa: E402


@dataclass(frozen=True)
class Case:
    task: str
    kind: str
    quantile: str
    scene_id: str
    gt_index: int
    prediction_index: int
    error_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--final-runs", type=Path, default=ROOT / "final_runs"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_gt_markdown(args.gt)
    all_cases: list[Case] = []
    for task in ("deleafing", "harvesting"):
        rows = _read_csv(args.final_runs / f"{task}_all30" / "targets.csv")
        cases = _select_cases(task, rows, entries)
        expected_failures = 8 if task == "deleafing" else 3
        failures = [case for case in cases if case.kind == "failure"]
        if len(failures) != expected_failures:
            raise AssertionError(
                f"Expected {expected_failures} {task} failures, found {len(failures)}"
            )
        _render_task(args, task, rows, entries, cases)
        all_cases.extend(cases)

    _write_csv(
        args.output_dir / "qualitative_case_index.csv",
        [
            {
                "task": case.task,
                "kind": case.kind,
                "error_quantile": case.quantile,
                "scene_id": case.scene_id,
                "gt_index": case.gt_index,
                "prediction_index": case.prediction_index,
                "error_cm": case.error_m * 100.0,
            }
            for case in all_cases
        ],
    )
    manifest = {
        "scope": "visualization-only replay of frozen final predictions",
        "matched_cases_per_task": 5,
        "failure_cases": {"deleafing": 8, "harvesting": 3},
        "matched_quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
        "deleafing_field": "retained contact mass rho",
        "harvesting_field": (
            "visualization-only delta-weighted local removed density: "
            "delta_i times the number of component neighbors within sigma"
        ),
        "projection": "world X-Z",
        "gt_use": "case selection and overlay only; never prediction",
    }
    (args.output_dir / "qualitative_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _select_cases(task, rows, entries):
    by_scene: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    matches: list[Case] = []
    failures: list[Case] = []
    suffix = "leaf" if task == "deleafing" else "tomato"
    scene_ids = sorted(
        f"egodelta_{uuid.replace('-', '')[:12]}_{suffix}"
        for uuid, tasks in entries.items()
        if len(tasks[task])
    )
    for scene_id in scene_ids:
        gt = find_scene_targets(scene_id, entries)[2].astype(np.float64)
        scene_rows = by_scene.get(scene_id, [])
        pred = np.asarray(
            [[float(row[axis]) for axis in "xyz"] for row in scene_rows],
            dtype=np.float64,
        ).reshape((-1, 3))
        if len(pred):
            distance = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
            invalid = float(distance.max() + 2.0)
            rr, cc = linear_sum_assignment(np.where(distance <= 0.03, distance, invalid))
            valid = distance[rr, cc] <= 0.03
            assigned_gt = set()
            for pred_index, gt_index in zip(rr[valid], cc[valid], strict=True):
                assigned_gt.add(int(gt_index))
                matches.append(
                    Case(
                        task, "matched", "", scene_id, int(gt_index),
                        int(pred_index), float(distance[pred_index, gt_index])
                    )
                )
            for gt_index in range(len(gt)):
                if gt_index not in assigned_gt:
                    nearest = int(np.argmin(distance[:, gt_index]))
                    failures.append(
                        Case(
                            task, "failure", "", scene_id, gt_index, nearest,
                            float(distance[nearest, gt_index])
                        )
                    )
        else:
            failures.extend(
                Case(task, "failure", "", scene_id, i, -1, float("inf"))
                for i in range(len(gt))
            )

    ordered = sorted(matches, key=lambda case: case.error_m)
    quantiles = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    indices = np.rint(quantiles * (len(ordered) - 1)).astype(int)
    chosen = []
    for quantile, index in zip(quantiles, indices, strict=True):
        case = ordered[int(index)]
        chosen.append(
            Case(
                case.task, case.kind, f"q{quantile:.2f}", case.scene_id,
                case.gt_index, case.prediction_index, case.error_m
            )
        )
    return chosen + sorted(failures, key=lambda case: (case.scene_id, case.gt_index))


def _render_task(args, task, rows, entries, cases):
    by_scene: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    columns = 5
    rows_count = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(3.45 * columns, 3.25 * rows_count), squeeze=False
    )
    for axis, case in zip(axes.flat, cases, strict=False):
        cache = load_stage2_cache(args.cache_dir / f"{case.scene_id}.npz")
        gt = find_scene_targets(case.scene_id, entries)[2].astype(np.float64)
        scene_rows = by_scene.get(case.scene_id, [])
        if task == "deleafing":
            _render_deleafing(axis, cache, gt[case.gt_index], scene_rows, case)
        else:
            _render_harvesting(axis, cache, gt[case.gt_index], scene_rows, case)
    for axis in axes.flat[len(cases):]:
        axis.axis("off")
    title = (
        "Deleafing: retained primitives colored by contact mass $\\rho$"
        if task == "deleafing"
        else "Harvesting: removed primitives colored by local density"
    )
    figure.suptitle(title, fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(args.output_dir / f"{task}_qualitative_panels.png", dpi=220)
    figure.savefig(args.output_dir / f"{task}_qualitative_panels.pdf")
    plt.close(figure)


def _render_deleafing(axis, cache, gt, scene_rows, case):
    cfg = ExtractionConfig()
    selected = np.flatnonzero(cache.overlap)
    points = cache.mu[selected].astype(np.float64)
    normals = cache.normals[selected].astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    score = cache.delta[selected].astype(np.float64)
    omega = cache.omega[selected].astype(np.float64)
    sigma = cfg.sigma_multiplier * _median_nonself_nn(points)
    w_rem = score
    w_ret = (omega - score) * (score <= cfg.delta_threshold)
    removed = np.flatnonzero(score > cfg.delta_threshold)
    labels = DBSCAN(
        eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_samples, n_jobs=-1
    ).fit_predict(points[removed])
    components = {
        int(label): removed[np.flatnonzero(labels == label)]
        for label in np.unique(labels) if int(label) >= 0
    }
    desired_component = None
    if case.prediction_index >= 0 and case.prediction_index < len(scene_rows):
        desired_component = int(scene_rows[case.prediction_index]["component_id"])
    if desired_component not in components:
        desired_component = _nearest_component(points, components, gt)
    component = components[desired_component]
    retained = np.flatnonzero(w_ret > 0.0)
    field_indices, rho = _retained_contact_field(
        points, normals, component, retained, cKDTree(points[retained]),
        w_rem, w_ret, sigma
    )
    candidates = _refined_peaks(points, field_indices, rho, sigma, desired_component)
    predictions = _row_points(scene_rows)
    selected_xyz = (
        predictions[case.prediction_index] if case.prediction_index >= 0 else None
    )
    _background(axis, points, gt, selected_xyz, points[component])
    if len(field_indices):
        color = np.log1p(rho)
        axis.scatter(
            points[field_indices, 0], points[field_indices, 2], c=color,
            cmap="magma", s=5, alpha=0.9, linewidths=0, rasterized=True
        )
    axis.scatter(
        points[component, 0], points[component, 2], s=2, color="#35c9c9",
        alpha=0.18, linewidths=0, rasterized=True
    )
    if candidates:
        peaks = np.asarray([candidate["refined_xyz"] for candidate in candidates])
        axis.scatter(peaks[:, 0], peaks[:, 2], marker="x", s=33, color="#42f5e3", linewidths=1.0)
    _markers(axis, gt, selected_xyz)
    _finish_axis(axis, case, gt, selected_xyz, points[component])


def _render_harvesting(axis, cache, gt, scene_rows, case):
    cfg = RemovedSurfaceConfig()
    selected = np.flatnonzero(cache.overlap)
    points = cache.mu[selected].astype(np.float64)
    score = cache.delta[selected].astype(np.float64)
    sigma = cfg.sigma_multiplier * _median_nonself_nn(points)
    removed = np.flatnonzero(score > cfg.delta_threshold)
    labels = DBSCAN(
        eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_samples, n_jobs=-1
    ).fit_predict(points[removed])
    components = {
        int(label): removed[np.flatnonzero(labels == label)]
        for label in np.unique(labels) if int(label) >= 0
    }
    desired_component = None
    if case.prediction_index >= 0 and case.prediction_index < len(scene_rows):
        desired_component = int(scene_rows[case.prediction_index]["component_id"])
    if desired_component not in components:
        desired_component = _nearest_component(points, components, gt)
    component = components[desired_component]
    component_points = points[component]
    local_density = cKDTree(component_points).query_ball_point(
        component_points, sigma, return_length=True
    ).astype(np.float64) * score[component]
    cores = _dense_mode_cores(
        component_points, cfg.mode_min_cluster_size, cfg.dbscan_min_samples
    )
    units = _complete_partition(component_points, cores)
    peaks = []
    for unit in units:
        _, raw, _ = _gravity_surface_peak(
            component_points[unit], score[component][unit], sigma
        )
        peaks.append(raw)
    predictions = _row_points(scene_rows)
    selected_xyz = (
        predictions[case.prediction_index] if case.prediction_index >= 0 else None
    )
    _background(axis, points, gt, selected_xyz, component_points)
    axis.scatter(
        component_points[:, 0], component_points[:, 2], c=np.log1p(local_density),
        cmap="viridis", s=6, alpha=0.9, linewidths=0, rasterized=True
    )
    if peaks:
        peak_array = np.asarray(peaks)
        axis.scatter(peak_array[:, 0], peak_array[:, 2], marker="x", s=48,
                     color="#ffdd35", linewidths=1.4)
    _markers(axis, gt, selected_xyz)
    _finish_axis(axis, case, gt, selected_xyz, component_points)


def _background(axis, points, gt, selected_xyz, focus):
    center_items = [focus, gt.reshape(1, 3)]
    if selected_xyz is not None:
        center_items.append(selected_xyz.reshape(1, 3))
    bounds = np.concatenate(center_items, axis=0)
    low = np.percentile(bounds[:, [0, 2]], 1, axis=0) - 0.035
    high = np.percentile(bounds[:, [0, 2]], 99, axis=0) + 0.035
    nearby = (
        (points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
        & (points[:, 2] >= low[1]) & (points[:, 2] <= high[1])
    )
    indices = np.flatnonzero(nearby)
    if len(indices) > 9000:
        indices = indices[np.linspace(0, len(indices) - 1, 9000).astype(int)]
    axis.scatter(points[indices, 0], points[indices, 2], s=1, color="#adb5bd",
                 alpha=0.17, linewidths=0, rasterized=True)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])


def _markers(axis, gt, selected_xyz):
    if selected_xyz is not None:
        axis.scatter([selected_xyz[0]], [selected_xyz[2]], marker="*", s=125,
                     color="#ef233c", edgecolor="white", linewidth=0.8, zorder=8)
    axis.scatter([gt[0]], [gt[2]], marker="P", s=95, color="white",
                 edgecolor="black", linewidth=1.0, zorder=9)


def _finish_axis(axis, case, gt, selected_xyz, focus):
    label = case.quantile if case.kind == "matched" else "failure"
    error = "no prediction" if not np.isfinite(case.error_m) else f"{case.error_m*100:.2f} cm"
    axis.set_title(f"{label} | {case.scene_id[9:21]} | {error}", fontsize=9)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world X [m]", fontsize=8)
    axis.set_ylabel("world Z [m]", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.08)


def _nearest_component(points, components, gt):
    if not components:
        raise RuntimeError("No DBSCAN component available for qualitative panel")
    return min(
        components,
        key=lambda label: float(np.linalg.norm(points[components[label]] - gt, axis=1).min()),
    )


def _row_points(rows):
    return np.asarray(
        [[float(row[axis]) for axis in "xyz"] for row in rows], dtype=np.float64
    ).reshape((-1, 3))


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
