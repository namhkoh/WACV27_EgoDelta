#!/usr/bin/env python
"""Reproduce contact-mass experiments E1--E6.

The ``phase1`` command intentionally implements the mandatory checkpoint.  It
never launches Phase 2.  The later experiments require an explicit ``phase2``
invocation after review of ``interim_results.md``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egodelta_diff.contact_mass_extraction import ContactMassConfig, extract_contact_mass_targets
from egodelta_diff.contact_proposal import contact_config_from_dict, generate_contact_proposals
from egodelta_diff.data import discover_ready_pairs, find_scene_targets, parse_gt_markdown
from egodelta_diff.evaluation import (
    aggregate_evaluations,
    cutoff_sensitivity,
    evaluate_contact_result,
    finalize_failure_labels,
    mass_distribution,
)
from egodelta_diff.pipeline import EgoDeltaDiffPipeline
from egodelta_diff.stage2_cache import cache_digest, load_stage2_cache, save_stage2_cache
from egodelta_diff.targets import (
    evaluate_cut_points,
    rule_candidate_from_json,
    select_contact_guarded_rule_leaf_candidates,
)


PUBLISHED_F1 = {"harvesting": 0.9074, "deleafing": 0.8824, "all": 0.8910}
E3_IDS = {
    "E3-1": "no normal term",
    "E3-2": "omega = 1",
    "E3-3": "removed-side mirror",
    "E3-3b": "removed-on-removed self-mass",
    "E3-4": "binary delta in weights only",
    "E3-5": "Stage-2 overlap off",
    "E3-6": "no retained-side exclusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache", help="Run frozen Stage 2 and write continuous caches.")
    cache.add_argument("--pre-root", type=Path, required=True)
    cache.add_argument("--post-root", type=Path, required=True)
    cache.add_argument("--gt", type=Path, default=Path("gt.md"))
    cache.add_argument("--config", type=Path, default=Path("configs/objectness_full.yaml"))
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--num-shards", type=int, default=1)
    cache.add_argument("--shard-index", type=int, default=0)
    cache.add_argument("--limit", type=int)
    cache.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Exact scene ID to cache; repeat to select several. When provided, "
            "scene discovery does not read the point-GT file."
        ),
    )
    cache.add_argument("--skip-existing", action="store_true")
    cache.add_argument("--overlap-off", action="store_true", help="Phase-2 E3-5 cache only.")

    for name in ("phase1", "phase2"):
        run = sub.add_parser(name)
        run.add_argument("--cache-root", type=Path, required=True)
        run.add_argument("--gt", type=Path, default=Path("gt.md"))
        run.add_argument("--config", type=Path, default=Path("configs/objectness_full.yaml"))
        run.add_argument("--out-dir", type=Path, required=True)
        run.add_argument("--legacy-outputs-root", type=Path)
        run.add_argument("--allow-reoptimized-baseline", action="store_true")
        run.add_argument(
            "--published-mismatch-report",
            type=Path,
            help=(
                "Root-cause report that authorizes proceeding when the legacy "
                "rerun differs from the published F1 values. The report path "
                "and SHA-256 are recorded in the experiment artifacts."
            ),
        )
    sub.choices["phase2"].add_argument("--overlap-off-cache-root", type=Path)
    return parser.parse_args()


def cache_stage2(args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    config = yaml.safe_load(args.config.read_text())
    config.setdefault("render", {})["enabled"] = False
    config.setdefault("overlay", {})["enabled"] = False
    config.setdefault("compute", {})["device"] = args.device
    if args.scenes:
        # A deployable inference run must not inspect point annotations.  The
        # GT block is used only by optional artifact writers, but disabling it
        # here also makes that boundary explicit in the cache provenance.
        config.setdefault("gt", {})["enabled"] = False
    if args.overlap_off:
        config.setdefault("ablation", {})["use_view_overlap_filter"] = False
    pipeline = EgoDeltaDiffPipeline(config)
    pairs = discover_ready_pairs(args.pre_root, args.post_root)
    if args.scenes:
        requested = set(args.scenes)
        pairs = [pair for pair in pairs if pair.name in requested]
        missing = sorted(requested.difference(pair.name for pair in pairs))
        if missing:
            raise SystemExit(f"Requested scene pairs not found: {', '.join(missing)}")
    else:
        gt_entries = parse_gt_markdown(args.gt)
        pairs = [pair for pair in pairs if find_scene_targets(pair.name, gt_entries)]
    pairs = [p for idx, p in enumerate(pairs) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    args.cache_root.mkdir(parents=True, exist_ok=True)
    records = []
    for pair in pairs:
        path = args.cache_root / f"{pair.name}.npz"
        if args.skip_existing and path.exists():
            print(f"[cache] skip {pair.name}", flush=True)
            continue
        started = time.perf_counter()
        print(f"[cache] {pair.name}", flush=True)
        result = pipeline.run_pair(pair)
        save_stage2_cache(path, result)
        elapsed = time.perf_counter() - started
        records.append({"scene_id": pair.name, "seconds": elapsed, "sha256": cache_digest(path)})
        print(f"[cache] wrote {path} in {elapsed:.1f}s", flush=True)
    manifest = _base_manifest(args)
    manifest.update({"kind": "stage2_cache", "overlap_off": args.overlap_off, "records": records})
    if args.scenes and len(args.scenes) == 1:
        manifest_name = f"manifest_{args.scenes[0]}.json"
    else:
        manifest_name = f"manifest_shard_{args.shard_index:02d}.json"
    _write_json(args.cache_root / manifest_name, manifest)


def run_phase1(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    gt_entries = parse_gt_markdown(args.gt)
    config_dict = yaml.safe_load(args.config.read_text())
    caches = _load_selected_caches(args.cache_root, gt_entries)
    _verify_cache_population(caches, gt_entries)

    legacy_evals, legacy_rows = _legacy_evaluations(caches, gt_entries, config_dict)
    legacy_summary = _summaries(legacy_evals)
    original_summary = None
    if args.legacy_outputs_root:
        original_evals = _legacy_original_ply_evaluations(
            caches, gt_entries, config_dict, args.legacy_outputs_root
        )
        original_summary = _summaries(original_evals)
        if not _summaries_equal(legacy_summary, original_summary):
            _write_json(out / "legacy_reproduction_failure.json", {"cache": legacy_summary, "original_ply": original_summary})
            raise SystemExit("STOP: legacy cache and original removed_primitives.ply metrics differ")
    mismatch = _published_mismatch(legacy_summary)
    mismatch_report = None
    if mismatch and args.published_mismatch_report is not None:
        report_path = args.published_mismatch_report.resolve()
        if not report_path.is_file():
            raise SystemExit(f"Published-mismatch root-cause report does not exist: {report_path}")
        mismatch_report = {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    if mismatch and not args.allow_reoptimized_baseline and mismatch_report is None:
        _write_json(out / "legacy_reproduction_failure.json", {"summary": legacy_summary, "mismatch": mismatch})
        raise SystemExit("STOP: original-checkpoint legacy metrics do not reproduce published reference")

    legacy_reproducibility = {
        "acceptance_criterion": (
            "Legacy is re-run on the identical Stage-2 cache used by contact-mass, "
            "and any delta from the published numbers is root-caused and reported."
        ),
        "published_f1": PUBLISHED_F1,
        "rerun_f1": {task: legacy_summary[task]["f1"] for task in PUBLISHED_F1},
        "delta_f1": {
            task: legacy_summary[task]["f1"] - published for task, published in PUBLISHED_F1.items()
        },
        "mismatch": mismatch,
        "root_cause_report": mismatch_report,
        "reoptimized_baseline_override": bool(args.allow_reoptimized_baseline),
    }

    primary_evals, diagnostic_rows, primary_results = _contact_run(caches, gt_entries, ContactMassConfig(), "primary")
    finalize_failure_labels(diagnostic_rows)
    primary_summary = _summaries(primary_evals)

    mirror_evals, mirror_rows, mirror_results = _contact_run(caches, gt_entries, ContactMassConfig(), "E3-3")
    mirror_summary = _summaries(mirror_evals)
    paired = _paired_mirror(primary_results, mirror_results, gt_entries)

    sigma_rows = []
    for multiplier in (1.0, 1.5, 2.0, 3.0, 5.0):
        evals, _, _ = _contact_run(
            caches, gt_entries, ContactMassConfig(sigma_multiplier=multiplier, deployment_offset_m=0.0), "primary"
        )
        summary = _summaries(evals)
        for task, metrics in summary.items():
            sigma_rows.append({"sigma_multiplier": multiplier, "task": task, **metrics})
    sigma_ranges = _sweep_ranges(sigma_rows, "sigma_multiplier")

    zero = _zero_mass_summary(diagnostic_rows)
    raw_mass = {
        task: {
            field: mass_distribution([r for r in diagnostic_rows if r["task"] == task], field)
            for field in ("m_gt_ret", "m_gt_rho")
        }
        for task in ("harvesting", "deleafing")
    }
    consistency = _zero_mass_label_consistency(diagnostic_rows)
    payload = {
        "integrity": {
            "cache_count": len(caches),
            "legacy_cache_matches_original_ply": original_summary is None or _summaries_equal(legacy_summary, original_summary),
            "legacy_published_delta_root_caused": (
                not mismatch or mismatch_report is not None or args.allow_reoptimized_baseline
            ),
            "abstentions_count_as_false_negatives": True,
        },
        "legacy_reproducibility": legacy_reproducibility,
        "E1": {"contact_mass": primary_summary, "legacy_cache": legacy_summary, "legacy_original_ply": original_summary},
        "E3-3": {"primary": primary_summary, "mirror": mirror_summary, "paired": paired},
        "E2_sigma": {"rows": sigma_rows, "max_minus_min_f1": sigma_ranges},
        "zero_mass": zero,
        "raw_mass_distributions": raw_mass,
        "zero_mass_A1_A2_disagreements": consistency,
        "parameters": asdict(ContactMassConfig()),
        "radii": {
            "3_sigma": "structural Gaussian truncation; component adjacency and diagnostics",
            "0.03_m": "primary one-to-one evaluation threshold",
            "0.06_m": "diagnostic spurious-prediction threshold",
        },
    }
    _write_json(out / "interim_results.json", payload)
    _write_csv(out / "instance_diagnostics.csv", diagnostic_rows)
    _write_csv(out / "legacy_per_scene.csv", legacy_rows)
    _write_csv(out / "sigma_sweep.csv", sigma_rows)
    _write_csv(out / "E3-3_paired.csv", paired["rows"])
    _write_markdown_phase1(out / "interim_results.md", payload)
    _write_json(
        out / "run_manifest.json",
        _base_manifest(args)
        | {
            "cache_hashes": _cache_hashes(args.cache_root),
            "legacy_reproducibility": legacy_reproducibility,
        },
    )
    _plot_sigma(out / "sigma_sweep.png", sigma_rows)
    print(f"Phase-1 checkpoint written to {out}. STOP: explicit approval is required for phase2.")


def run_phase2(args: argparse.Namespace) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    gt_entries = parse_gt_markdown(args.gt)
    caches = _load_selected_caches(args.cache_root, gt_entries)
    experiment_rows: list[dict[str, Any]] = []

    def record(label, cfg, variant="primary", **kwargs):
        evals, _, _ = _contact_run(caches, gt_entries, cfg, variant, **kwargs)
        for task, metrics in _summaries(evals).items():
            experiment_rows.append({"experiment": label, "task": task, **metrics})

    for tau in (0.2, 0.3, 0.4, 0.5):
        record(f"E2_tau_{tau:g}", ContactMassConfig(tau_c=tau, deployment_offset_m=0.0))
    for multiplier in (1.0, 1.5, 2.0, 3.0, 5.0):
        record(
            f"E2_sigma_{multiplier:g}",
            ContactMassConfig(sigma_multiplier=multiplier, deployment_offset_m=0.0),
        )
    for variant in ("E3-1", "E3-2", "E3-4", "E3-6"):
        record(variant, ContactMassConfig(deployment_offset_m=0.0), variant=variant)
    if args.overlap_off_cache_root:
        off_caches = _load_selected_caches(args.overlap_off_cache_root, gt_entries)
        evals, _, _ = _contact_run(off_caches, gt_entries, ContactMassConfig(deployment_offset_m=0.0), "primary")
        for task, metrics in _summaries(evals).items():
            experiment_rows.append({"experiment": "E3-5", "task": task, **metrics})

    primary_evals, diagnostics, results = _contact_run(caches, gt_entries, ContactMassConfig(deployment_offset_m=0.0), "primary")
    finalize_failure_labels(diagnostics)
    thresholds = []
    for threshold in (0.005, 0.01, 0.02, 0.03):
        evals = _reevaluate_results(results, gt_entries, threshold)
        for task, metrics in _summaries(evals).items():
            thresholds.append({"threshold_m": threshold, "task": task, **metrics})
    pr_rows = _precision_recall(caches, gt_entries, results)
    sensitivity = cutoff_sensitivity(diagnostics)
    audit = _parameter_audit()
    _write_csv(out / "experiment_results.csv", experiment_rows)
    _write_csv(out / "tighter_thresholds.csv", thresholds)
    _write_csv(out / "precision_recall.csv", pr_rows)
    _write_csv(out / "cutoff_sensitivity.csv", sensitivity)
    _write_csv(out / "instance_diagnostics.csv", diagnostics)
    _write_json(out / "parameter_audit.json", audit)
    _render_qualitatives(out / "renderings", caches, results, gt_entries)
    payload = {
        "experiments": experiment_rows,
        "tighter_thresholds": thresholds,
        "precision_recall": pr_rows,
        "failure_counts": _counts(diagnostics, "failure_class"),
        "cutoff_sensitivity": sensitivity,
        "component_mass_percentile": "deleted from the frozen estimator",
        "parameter_audit": audit,
    }
    _write_json(out / "results.json", payload)
    _write_markdown_phase2(out / "results.md", payload)


def _contact_run(caches, gt_entries, cfg, variant, **kwargs):
    evaluations: dict[str, list[Any]] = {"harvesting": [], "deleafing": []}
    rows, results = [], {}
    eval_cfg = ContactMassConfig(**(asdict(cfg) | {"deployment_offset_m": 0.0}))
    for cache in caches:
        found = find_scene_targets(cache.scene_id, gt_entries)
        if found is None:
            continue
        _, task, gt = found
        result = extract_contact_mass_targets(
            cache.mu, cache.normals, cache.delta, cache.omega, cache.overlap,
            cache.source_indices, eval_cfg, variant=variant, **kwargs,
        )
        evaluation, scene_rows = evaluate_contact_result(
            scene_id=cache.scene_id, task=task, result=result, mu=cache.mu,
            delta=cache.delta, omega=cache.omega, overlap=cache.overlap,
            gt_points=gt, tau_c=eval_cfg.tau_c,
        )
        evaluations[task].append(evaluation)
        rows.extend(scene_rows)
        results[cache.scene_id] = result
    return evaluations, rows, results


def _legacy_evaluations(caches, gt_entries, config):
    proposal_cfg = contact_config_from_dict(config.get("proposal", {}))
    output = {"harvesting": [], "deleafing": []}
    rows = []
    for cache in caches:
        found = find_scene_targets(cache.scene_id, gt_entries)
        if found is None:
            continue
        _, task, gt = found
        proposals = generate_contact_proposals(
            points=cache.mu, scores=cache.delta, source_indices=cache.source_indices,
            rgb01=cache.rgb01, removed_mask=cache.delta > 0.5,
            cluster_labels=cache.legacy_cluster_labels, config=proposal_cfg,
            object_type="tomato" if task == "harvesting" else "leaf",
        )
        selected = _legacy_select(cache.scene_id, task, proposals)
        pred = np.asarray([p.representative_point for p in selected], dtype=np.float64).reshape((-1, 3))
        ev = evaluate_cut_points(gt, pred, scene=cache.scene_id, threshold_m=0.03)
        output[task].append(ev)
        rows.append({"scene_id": cache.scene_id, "task": task, **ev.to_metrics_dict(3)})
    return output, rows


def _legacy_original_ply_evaluations(caches, gt_entries, config, root):
    proposal_cfg = contact_config_from_dict(config.get("proposal", {}))
    output = {"harvesting": [], "deleafing": []}
    for cache in caches:
        found = find_scene_targets(cache.scene_id, gt_entries)
        if found is None:
            continue
        _, task, gt = found
        path = root / cache.scene_id / "removed_primitives.ply"
        vertex = PlyData.read(path)["vertex"].data
        source = np.asarray(vertex["source_index"], dtype=np.int64)
        score = np.asarray(vertex["score"], dtype=np.float32)
        cluster = np.asarray(vertex["cluster"], dtype=np.int32)
        by_source = {int(value): idx for idx, value in enumerate(cache.source_indices)}
        mask = np.zeros(len(cache.mu), dtype=bool)
        scores = np.zeros(len(cache.mu), dtype=np.float32)
        labels = np.full(len(cache.mu), -1, dtype=np.int32)
        for src, value, label in zip(source, score, cluster, strict=True):
            idx = by_source.get(int(src))
            if idx is not None:
                mask[idx], scores[idx], labels[idx] = True, value, label
        proposals = generate_contact_proposals(
            cache.mu, scores, cache.source_indices, cache.rgb01, mask, labels,
            proposal_cfg, "tomato" if task == "harvesting" else "leaf",
        )
        selected = _legacy_select(cache.scene_id, task, proposals)
        pred = np.asarray([p.representative_point for p in selected], dtype=np.float64).reshape((-1, 3))
        output[task].append(evaluate_cut_points(gt, pred, scene=cache.scene_id, threshold_m=0.03))
    return output


def _legacy_select(scene, task, proposals):
    if task == "harvesting":
        return sorted([p for p in proposals if p.primary], key=lambda p: (p.rank, p.proposal_id))
    candidates = [rule_candidate_from_json(scene, asdict(p)) for p in proposals]
    ids = {candidate.proposal_id for candidate in select_contact_guarded_rule_leaf_candidates(candidates)}
    return sorted([p for p in proposals if p.proposal_id in ids], key=lambda p: (p.rank, p.proposal_id))


def _summaries(evaluations):
    output = {}
    all_rows = []
    for task in ("harvesting", "deleafing"):
        rows = evaluations[task]
        all_rows.extend(rows)
        output[task] = _compact_aggregate(aggregate_evaluations(rows, 0.03), 3)
    output["all"] = _compact_aggregate(aggregate_evaluations(all_rows, 0.03), 3)
    return output


def _compact_aggregate(value, cm):
    return {
        "scenes": value["scene_count"], "gt": value["total_gt"], "pred": value["total_pred"],
        "tp": value["TP"], "fp": value["FP"], "fn": value["FN"],
        "precision": value[f"Micro_Precision@{cm}cm"], "recall": value[f"Micro_Recall@{cm}cm"],
        "f1": value[f"Micro_F1@{cm}cm"],
        "median_error_cm": _times100(value["matched_localization_distance"]["median"]),
    }


def _paired_mirror(primary, mirror, gt_entries):
    rows = []
    for scene in sorted(mirror):
        found = find_scene_targets(scene, gt_entries)
        gt = found[2] if found else np.empty((0, 3))
        for pair in mirror[scene].removed_targets:
            if pair.retained_xyz is None or pair.removed_xyz is None:
                continue
            a_gt = float(np.min(np.linalg.norm(gt - pair.retained_xyz, axis=1))) if len(gt) else float("inf")
            b_gt = float(np.min(np.linalg.norm(gt - pair.removed_xyz, axis=1))) if len(gt) else float("inf")
            a_rem = float(np.linalg.norm(pair.retained_xyz - pair.removed_centroid))
            b_rem = float(np.linalg.norm(pair.removed_xyz - pair.removed_centroid))
            rows.append({"scene_id": scene, "component_id": pair.source_component_id,
                         "peak_id": pair.peak_id,
                         "d_rem_primary_minus_mirror": a_rem - b_rem,
                         "d_gt_primary_minus_mirror": a_gt - b_gt,
                         "d_rem_primary": a_rem, "d_rem_mirror": b_rem,
                         "d_gt_primary": a_gt, "d_gt_mirror": b_gt})
    return {"rows": rows, "d_rem_difference": _median_iqr(rows, "d_rem_primary_minus_mirror"),
            "d_gt_difference": _median_iqr(rows, "d_gt_primary_minus_mirror"),
            "fraction_primary_farther_from_removed_centroid": float(np.mean([r["d_rem_primary_minus_mirror"] > 0 for r in rows])) if rows else None}


def _reevaluate_results(results, gt_entries, threshold):
    output = {"harvesting": [], "deleafing": []}
    for scene, result in results.items():
        found = find_scene_targets(scene, gt_entries)
        _, task, gt = found
        pred = np.asarray([c.predicted_xyz for c in result.components if c.emitted], dtype=np.float64).reshape((-1, 3))
        output[task].append(evaluate_cut_points(gt, pred, scene=scene, threshold_m=threshold))
    return output


def _precision_recall(caches, gt_entries, results):
    confidences = sorted({c.conf_c for result in results.values() for c in result.components if c.emitted})
    thresholds = [-float("inf")] + confidences
    rows = []
    for threshold in thresholds:
        evals = {"harvesting": [], "deleafing": []}
        emitted, unthresholded = 0, 0
        for cache in caches:
            found = find_scene_targets(cache.scene_id, gt_entries)
            _, task, gt = found
            components = [c for c in results[cache.scene_id].components if c.emitted]
            unthresholded += len(components)
            chosen = [c for c in components if c.conf_c >= threshold]
            emitted += len(chosen)
            pred = np.asarray([c.predicted_xyz for c in chosen], dtype=np.float64).reshape((-1, 3))
            evals[task].append(evaluate_cut_points(gt, pred, scene=cache.scene_id, threshold_m=0.03))
        summary = _summaries(evals)["all"]
        rows.append({"confidence_threshold": threshold, "yield": emitted / unthresholded if unthresholded else 0.0, **summary})
    return rows


def _parameter_audit():
    source = Path(__file__).resolve().parents[1] / "egodelta_diff" / "contact_mass_extraction.py"
    tree = ast.parse(source.read_text())
    structural_literals = sum(isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) for node in ast.walk(tree))
    return {
        "new": {"tuned": ["tau_c", "dbscan_eps_m", "dbscan_min_samples"],
                "derived": ["median_nearest_neighbor_distance", "sigma"],
                "domain": ["nms_radius_m", "deployment_offset_m"],
                "fixed_multiplier": ["sigma_multiplier"],
                "deleted": ["component_mass_percentile"],
                "structural_literal_occurrences_excluded": structural_literals},
        "legacy": {"status": "Table 6 and code audit required before publication", "config_source": "configs/objectness_full.yaml"},
    }


def _render_qualitatives(out, caches, results, gt_entries):
    out.mkdir(parents=True, exist_ok=True)
    per_task = {"harvesting": 0, "deleafing": 0}
    for cache in caches:
        found = find_scene_targets(cache.scene_id, gt_entries)
        _, task, gt = found
        if per_task[task] >= 10:
            continue
        result = results[cache.scene_id]
        if not result.components:
            continue
        component = max(result.components, key=lambda c: c.conf_c)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        retained = np.flatnonzero((cache.delta <= ContactMassConfig().tau_c) & cache.overlap)
        if len(retained) > 20000:
            retained = retained[:: max(1, len(retained) // 20000)]
        rho_map = {int(i): float(v) for i, v in zip(component.rho_indices, component.rho_values, strict=True)}
        colors = np.asarray([rho_map.get(int(i), 0.0) for i in retained])
        ax.scatter(*cache.mu[retained].T, c=colors, cmap="magma", s=1, alpha=0.45)
        ax.scatter(*cache.mu[component.primitive_indices].T, c="cyan", s=2, alpha=0.5, label="removed component")
        if component.emitted:
            ax.scatter(*component.predicted_xyz, c="lime", s=80, marker="x", label="prediction")
        if len(gt):
            ax.scatter(*gt.T, c="blue", s=55, marker="o", label="GT")
        ax.legend()
        ax.set_title(f"{cache.scene_id}: retained rho + removed component")
        fig.tight_layout()
        fig.savefig(out / f"{cache.scene_id}.png", dpi=180)
        plt.close(fig)
        per_task[task] += 1


def _load_selected_caches(root, gt_entries):
    return [cache for path in sorted(Path(root).glob("*.npz"))
            if find_scene_targets((cache := load_stage2_cache(path)).scene_id, gt_entries)]


def _verify_cache_population(caches, gt_entries):
    expected = {f"egodelta_{uuid.replace('-', '')[:12]}_{'tomato' if task == 'harvesting' else 'leaf'}"
                for uuid, tasks in gt_entries.items() for task, points in tasks.items() if len(points)}
    actual = {cache.scene_id for cache in caches}
    missing = sorted(expected - actual)
    if missing:
        raise SystemExit(f"Missing Stage-2 caches for {len(missing)} scenes: {missing}")


def _base_manifest(args):
    return {"timestamp_unix": time.time(), "command": sys.argv, "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": _git(["rev-parse", "HEAD"]), "git_status": _git(["status", "--short"])}


def _git(args):
    return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def _cache_hashes(root):
    return {path.name: cache_digest(path) for path in sorted(Path(root).glob("*.npz"))}


def _published_mismatch(summary):
    return {task: {"observed": summary[task]["f1"], "published": expected}
            for task, expected in PUBLISHED_F1.items() if abs(summary[task]["f1"] - expected) > 5e-4}


def _summaries_equal(left, right):
    return all(left[t][k] == right[t][k] for t in left for k in ("tp", "fp", "fn", "f1", "median_error_cm"))


def _sweep_ranges(rows, field):
    return {task: float(np.ptp([r["f1"] for r in rows if r["task"] == task]))
            for task in sorted({r["task"] for r in rows})}


def _zero_mass_summary(rows):
    output = {}
    for task in ("harvesting", "deleafing"):
        selected = [r for r in rows if r["task"] == task]
        zero = [r for r in selected if r["conf_c"] <= 0.0]
        output[task] = {"components": len(selected), "zero": len(zero),
                        "rate": len(zero) / len(selected) if selected else 0.0,
                        "causes": _counts(zero, "zero_mass_cause")}
    return output


def _zero_mass_label_consistency(rows):
    disagreements = []
    for row in rows:
        if row["conf_c"] > 0.0:
            continue
        shadow = "A1" if row["m_gt_ret_norm"] < 0.1 else ("A2" if row["m_gt_rho_norm"] < 0.1 else "other")
        expected = "A1" if row["zero_mass_cause"] == "no_retained_neighbors" else ("A2" if row["zero_mass_cause"] == "normal_suppression" else "other")
        if shadow != expected:
            disagreements.append({"scene_id": row["scene_id"], "component_id": row["component_id"], "shadow": shadow, "zero_cause": row["zero_mass_cause"]})
    return disagreements


def _counts(rows, field):
    output = {}
    for row in rows:
        key = str(row.get(field))
        output[key] = output.get(key, 0) + 1
    return output


def _median_iqr(rows, field):
    if not rows:
        return {"median": None, "q25": None, "q75": None}
    q = np.quantile([row[field] for row in rows], [0.25, 0.5, 0.75])
    return {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}


def _times100(value):
    return None if value is None else float(value) * 100.0


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default))


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _plot_sigma(path, rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    for task in ("harvesting", "deleafing", "all"):
        selected = [row for row in rows if row["task"] == task]
        ax.plot([r["sigma_multiplier"] for r in selected], [r["f1"] for r in selected], marker="o", label=task)
    ax.set(xlabel="sigma multiplier k", ylabel="F1 @ 3 cm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown_phase1(path, payload):
    lines = ["# Contact-mass Phase 1 checkpoint", "",
             "Stage 3 consumes no appearance features; appearance enters only through the upstream change score.", "",
             "Abstained zero-contact-mass components emit no prediction; their GT targets remain unmatched and count as false negatives.", "",
             "## Legacy reproducibility", "", payload["legacy_reproducibility"]["acceptance_criterion"], "",
             _legacy_reproducibility_table(payload["legacy_reproducibility"]), "",
             f"Root-cause report: `{payload['legacy_reproducibility']['root_cause_report']}`", "",
             "## E1", "", _summary_table(payload["E1"]), "", "## E3-3 retained-side versus fair mirror", "",
             _summary_table({"primary": payload["E3-3"]["primary"], "mirror": payload["E3-3"]["mirror"]}), "",
             f"Fraction primary farther from removed centroid: {payload['E3-3']['paired']['fraction_primary_farther_from_removed_centroid']}", "",
             "The mirror uses removed outer weights over the same component and sums over all retained primitives; this asymmetry is intentional because retained structure is not componentized.", "",
             "## E2 sigma sweep", "", f"Max-minus-min F1: `{payload['E2_sigma']['max_minus_min_f1']}`", "",
             "## Radii", "", "- `3 × sigma`: structural Gaussian truncation, adjacency, and diagnostics.", "- `3 cm`: primary matching threshold.", "- `6 cm`: diagnostic spurious-prediction threshold.", "",
             "## Checkpoint", "", "Execution stops here. Phase 2 requires explicit approval."]
    path.write_text("\n".join(lines) + "\n")


def _legacy_reproducibility_table(reproducibility):
    lines = [
        "| Task | Published F1 | Re-run F1 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for task in ("harvesting", "deleafing", "all"):
        lines.append(
            f"| {task} | {reproducibility['published_f1'][task]:.6f} | "
            f"{reproducibility['rerun_f1'][task]:.6f} | "
            f"{reproducibility['delta_f1'][task]:+.6f} |"
        )
    return "\n".join(lines)


def _summary_table(groups):
    lines = ["| Method | Task | P | R | F1 | Median error (cm) | Pred / GT |", "|---|---|---:|---:|---:|---:|---:|"]
    for method, summary in groups.items():
        if summary is None:
            continue
        for task, m in summary.items():
            lines.append(f"| {method} | {task} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['median_error_cm']} | {m['pred']} / {m['gt']} |")
    return "\n".join(lines)


def _write_markdown_phase2(path, payload):
    lines = ["# Contact-mass experiment results", "",
             "Stage 3 consumes no appearance features; appearance enters only through the upstream change score.", "",
             "The 3-sigma radius is structural Gaussian truncation; 3 cm is the primary evaluation gate; 6 cm is only the D diagnostic gate.", "",
             "## Artifacts", "", "Detailed experiment, threshold, PR, diagnostics, sensitivity, and audit tables accompany this file.", "",
             "## Constraint note", "", "No crop-specific branch or appearance input was added to the estimator."]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.command == "cache":
        cache_stage2(args)
    elif args.command == "phase1":
        run_phase1(args)
    else:
        run_phase2(args)


if __name__ == "__main__":
    main()
