#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egodelta_diff.data import discover_ready_pairs, find_scene_targets, parse_gt_markdown  # noqa: E402
from egodelta_diff.pipeline import EgoDeltaDiffPipeline  # noqa: E402
from egodelta_diff.targets import (  # noqa: E402
    DEFAULT_THRESHOLD_M,
    aggregate_cut_point_evaluations,
    evaluate_cut_points,
    rule_candidate_from_json,
    select_contact_guarded_rule_leaf_candidates,
)

DEFAULT_PRE_ROOT = Path("pre_models")
DEFAULT_POST_ROOT = Path("post_models")
DEFAULT_VARIANTS = ("full", "geometry_only", "without_confidence", "without_view_overlap")


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-config", type=Path, default=Path("configs/objectness_full.yaml"))
    parser.add_argument("--variant-config-dir", type=Path, default=Path("configs/ablations"))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--pre-root", type=Path, default=DEFAULT_PRE_ROOT)
    parser.add_argument("--post-root", type=Path, default=DEFAULT_POST_ROOT)
    parser.add_argument("--out-root", type=Path, default=Path("outputs_scoring_ablation"))
    parser.add_argument("--gt", type=Path, default=Path("gt.md"))
    parser.add_argument("--require-gt", action="store_true", help="Only run scenes present in the GT file.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--query-views", choices=["pre", "post", "all", "pre_added"], default=None)
    parser.add_argument("--max-primitives", type=int, default=None)
    parser.add_argument("--tree-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--disable-render", action="store_true")
    parser.add_argument("--disable-overlay", action="store_true")


def add_summarize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs_scoring_ablation"))
    parser.add_argument("--gt", type=Path, default=Path("gt.md"))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--threshold-m", type=float, default=DEFAULT_THRESHOLD_M)
    parser.add_argument("--summary-json", type=Path, default=Path("experiment_logs/scoring_ablation_summary.json"))
    parser.add_argument("--per-scene-csv", type=Path, default=Path("experiment_logs/scoring_ablation_per_scene.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("experiment_logs/scoring_ablation_summary.md"))
    parser.add_argument(
        "--selection-policy",
        choices=["paper", "primary", "all"],
        default="paper",
        help="paper uses leaf contact-guarded consensus and tomato primary proposals.",
    )
    parser.add_argument(
        "--check-reference-full",
        action="store_true",
        help="Fail if the full variant does not match --reference-target-json.",
    )
    parser.add_argument(
        "--reference-target-json",
        type=Path,
        default=None,
        help="Reference trajectory target JSON to use as the full-method baseline.",
    )


def run_ablation(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards")

    pairs = discover_ready_pairs(args.pre_root, args.post_root)
    if args.require_gt:
        gt_entries = parse_gt_markdown(args.gt)
        pairs = [pair for pair in pairs if find_scene_targets(pair.name, gt_entries) is not None]
    if args.num_shards > 1:
        pairs = [pair for idx, pair in enumerate(pairs) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No ready pre/post pairs selected.")

    base_config = load_yaml(args.base_config)
    variants = parse_variants(args.variants)
    for variant in variants:
        variant_config = load_yaml(args.variant_config_dir / f"{variant}.yaml")
        config = apply_cli_overrides(deep_merge(base_config, variant_config), args)
        pipeline = EgoDeltaDiffPipeline(config)
        for pair in pairs:
            out_dir = args.out_root / variant / pair.name
            if args.skip_existing and (out_dir / "diagnostics.json").exists() and (
                out_dir / "contact_target_proposals.json"
            ).exists():
                print(f"[ablation:{variant}] skip existing {pair.name} -> {out_dir}", flush=True)
                continue
            print(f"[ablation:{variant}] {pair.name} -> {out_dir}", flush=True)
            result = pipeline.run_pair(pair)
            pipeline.save(result, out_dir, query_views=args.query_views)


def summarize_ablation_cli(args: argparse.Namespace) -> None:
    if args.check_reference_full and args.reference_target_json is None:
        raise SystemExit("--check-reference-full requires --reference-target-json")
    report = summarize_ablation(
        outputs_root=args.outputs_root,
        gt_path=args.gt,
        variants=parse_variants(args.variants),
        threshold_m=args.threshold_m,
        selection_policy=args.selection_policy,
    )
    if args.check_reference_full:
        failures = check_reference_full(report, args.reference_target_json)
        report["reference_full_check"] = {
            "passed": not failures,
            "failures": failures,
            "reference_target_json": str(args.reference_target_json) if args.reference_target_json else None,
        }
    else:
        failures = []
    write_json(args.summary_json, report)
    write_csv(args.per_scene_csv, report["per_scene_rows"])
    write_markdown(args.markdown, report)
    if args.check_reference_full and failures:
        print("Reference full baseline check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    if args.check_reference_full:
        print("Reference full baseline check passed.", flush=True)
    print(json.dumps(report["variant_summaries"], indent=2, allow_nan=False))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.max_primitives is not None:
        config.setdefault("compute", {})["max_primitives"] = args.max_primitives
    if args.tree_workers is not None:
        config.setdefault("kernel", {})["tree_workers"] = args.tree_workers
    if args.device is not None:
        config.setdefault("compute", {})["device"] = args.device
    if args.allow_cpu_fallback:
        config.setdefault("compute", {})["allow_cpu_fallback"] = True
    if args.disable_render:
        config.setdefault("render", {})["enabled"] = False
    if args.disable_overlay:
        config.setdefault("overlay", {})["enabled"] = False
    return config

def summarize_ablation(
    *,
    outputs_root: Path,
    gt_path: Path,
    variants: list[str],
    threshold_m: float,
    selection_policy: str,
) -> dict[str, Any]:
    gt_entries = parse_gt_markdown(gt_path)
    threshold_cm = int(round(threshold_m * 100.0))
    precision_key = f"Precision@{threshold_cm}cm"
    recall_key = f"Recall@{threshold_cm}cm"
    f1_key = f"F1@{threshold_cm}cm"
    variant_summaries = []
    per_scene_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    organization_rows: list[dict[str, Any]] = []

    for variant in variants:
        variant_root = resolve_variant_root(outputs_root, variant, variants)
        evaluations_by_split: dict[str, list[Any]] = {"Leaf": [], "Tomato": [], "All": []}
        bbox_counts = {"Leaf": [0, 0], "Tomato": [0, 0], "All": [0, 0]}
        overlay_counts = {"Leaf": [0, 0], "Tomato": [0, 0], "All": [0, 0]}
        if not variant_root.exists():
            skipped.append({"variant": variant, "scene": "", "reason": "missing_variant_root"})
            variant_summaries.append(empty_variant_summary(variant, threshold_m))
            continue

        for scene_dir in sorted((path for path in variant_root.iterdir() if path.is_dir()), key=scene_sort_key):
            proposal_path = scene_dir / "contact_target_proposals.json"
            if not proposal_path.exists():
                skipped.append({"variant": variant, "scene": scene_dir.name, "reason": "missing_contact_target_proposals"})
                continue
            scene_targets = find_scene_targets(scene_dir.name, gt_entries)
            if scene_targets is None:
                skipped.append({"variant": variant, "scene": scene_dir.name, "reason": "missing_gt"})
                continue
            _, task, gt_points = scene_targets
            split = split_from_scene(scene_dir.name, task)
            proposals = json.loads(proposal_path.read_text()).get("proposals", [])
            selected = select_scene_proposals(scene_dir.name, split, proposals, selection_policy)
            pred_points = np.asarray(
                [proposal["representative_point"] for proposal in selected],
                dtype=np.float64,
            ).reshape((-1, 3))
            evaluation = evaluate_cut_points(
                gt_points=np.asarray(gt_points, dtype=np.float64),
                pred_points=pred_points,
                scene=scene_dir.name,
                threshold_m=threshold_m,
            )
            evaluations_by_split[split].append(evaluation)
            evaluations_by_split["All"].append(evaluation)

            cluster_by_id = load_cluster_index(scene_dir / "removed_clusters.json")
            bbox_count = sum(1 for proposal in selected if proposal_has_bbox(proposal, cluster_by_id))
            overlay_available = load_best_overlay_crop(scene_dir) is not None
            for key in (split, "All"):
                bbox_counts[key][0] += bbox_count
                bbox_counts[key][1] += len(selected)
                overlay_counts[key][0] += int(overlay_available)
                overlay_counts[key][1] += 1

            per_scene_rows.append(
                {
                    "variant": variant,
                    "scene": scene_dir.name,
                    "split": split,
                    "num_gt": evaluation.num_gt,
                    "num_pred": evaluation.num_pred,
                    "TP": evaluation.tp,
                    "FP": evaluation.fp,
                    "FN": evaluation.fn,
                    precision_key: evaluation.precision,
                    recall_key: evaluation.recall,
                    f1_key: evaluation.f1,
                    "matched_mean_distance_cm": cm_or_none(evaluation.matched_mean_distance),
                    "gt_to_nearest_mean_distance_cm": cm_or_none(evaluation.gt_to_nearest_mean_distance),
                    "target_bboxes": bbox_count,
                    "selected_targets": len(selected),
                    "overlay_available": overlay_available,
                }
            )

        variant_summary = {
            "variant": variant,
            "splits": {
                split: aggregate_row(split, evaluations_by_split[split], threshold_m)
                for split in ("Leaf", "Tomato", "All")
            },
        }
        variant_summaries.append(variant_summary)
        organization_rows.append(
            {
                "variant": variant,
                "splits": {
                    split: {
                        "target_bboxes": bbox_counts[split][0],
                        "selected_targets": bbox_counts[split][1],
                        "scenes_with_2d_overlay": overlay_counts[split][0],
                        "scenes": overlay_counts[split][1],
                    }
                    for split in ("Leaf", "Tomato", "All")
                },
            }
        )

    return {
        "metadata": {
            "outputs_root": str(outputs_root),
            "gt": str(gt_path),
            "threshold_m": float(threshold_m),
            "selection_policy": selection_policy,
        },
        "variant_summaries": variant_summaries,
        "organization_summaries": organization_rows,
        "per_scene_rows": per_scene_rows,
        "skipped": skipped,
    }


def resolve_variant_root(outputs_root: Path, variant: str, variants: list[str]) -> Path:
    variant_root = outputs_root / variant
    if variant_root.exists():
        return variant_root
    if len(variants) == 1 and any(outputs_root.glob("egodelta_*")):
        return outputs_root
    return variant_root


def select_scene_proposals(
    scene_name: str,
    split: str,
    proposals: list[dict[str, Any]],
    selection_policy: str,
) -> list[dict[str, Any]]:
    if selection_policy == "all":
        return sorted(proposals, key=proposal_sort_key)
    if selection_policy == "primary" or split == "Tomato":
        return sorted(
            [proposal for proposal in proposals if bool(proposal.get("primary", False))],
            key=proposal_sort_key,
        )
    candidates = [rule_candidate_from_json(scene_name, proposal) for proposal in proposals]
    selected = select_contact_guarded_rule_leaf_candidates(candidates)
    by_id = {
        int(proposal["proposal_id"]): proposal
        for proposal in proposals
        if "proposal_id" in proposal
    }
    return [by_id[candidate.proposal_id] for candidate in selected if candidate.proposal_id in by_id]


def aggregate_row(split: str, evaluations: list[Any], threshold_m: float) -> dict[str, Any]:
    aggregate = aggregate_cut_point_evaluations(evaluations, threshold_cm=int(round(threshold_m * 100.0)))
    threshold_cm = int(round(threshold_m * 100.0))
    return {
        "Split": split,
        "Scenes": aggregate["scene_count"],
        "GT": aggregate["total_gt"],
        "Pred": aggregate["total_pred"],
        "TP": aggregate["TP"],
        "FP": aggregate["FP"],
        "FN": aggregate["FN"],
        f"Precision@{threshold_cm}cm": aggregate[f"Micro_Precision@{threshold_cm}cm"],
        f"Recall@{threshold_cm}cm": aggregate[f"Micro_Recall@{threshold_cm}cm"],
        f"F1@{threshold_cm}cm": aggregate[f"Micro_F1@{threshold_cm}cm"],
        "matched_mean_distance_cm": cm_or_none(aggregate["matched_localization_distance"]["mean"]),
        "matched_median_distance_cm": cm_or_none(aggregate["matched_localization_distance"]["median"]),
        "gt_to_nearest_mean_distance_cm": cm_or_none(aggregate["gt_to_nearest_distance"]["mean"]),
        "gt_to_nearest_median_distance_cm": cm_or_none(aggregate["gt_to_nearest_distance"]["median"]),
    }


def empty_variant_summary(variant: str, threshold_m: float) -> dict[str, Any]:
    threshold_cm = int(round(threshold_m * 100.0))
    return {
        "variant": variant,
        "splits": {
            split: {
                "Split": split,
                "Scenes": 0,
                "GT": 0,
                "Pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                f"Precision@{threshold_cm}cm": 0.0,
                f"Recall@{threshold_cm}cm": 0.0,
                f"F1@{threshold_cm}cm": 0.0,
                "matched_mean_distance_cm": None,
                "matched_median_distance_cm": None,
                "gt_to_nearest_mean_distance_cm": None,
                "gt_to_nearest_median_distance_cm": None,
            }
            for split in ("Leaf", "Tomato", "All")
        },
    }


def load_cluster_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    clusters = json.loads(path.read_text())
    return {
        int(cluster["cluster_id"]): cluster
        for cluster in clusters
        if "cluster_id" in cluster
    }


def proposal_has_bbox(proposal: dict[str, Any], cluster_by_id: dict[int, dict[str, Any]]) -> bool:
    cluster_ids = [int(value) for value in proposal.get("support_cluster_ids", [])]
    return any(
        cluster_id in cluster_by_id
        and "aabb_min" in cluster_by_id[cluster_id]
        and "aabb_max" in cluster_by_id[cluster_id]
        for cluster_id in cluster_ids
    )


def load_best_overlay_crop(scene_dir: Path) -> dict[str, Any] | None:
    candidates = [
        scene_dir / "contact_proposal_overlays" / "gt_mask_cloud_overlay_tight" / "manifest.json",
        scene_dir / "contact_proposal_overlays" / "gt_mask_cloud_overlay" / "manifest.json",
        scene_dir / "contact_proposal_overlays" / "gt_mask_cloud" / "manifest.json",
    ]
    for path in candidates:
        if path.exists() and json.loads(path.read_text()):
            return {"manifest": str(path)}
    return None


def split_from_scene(scene_name: str, task: str) -> str:
    if scene_name.endswith("_tomato") or task == "harvesting":
        return "Tomato"
    return "Leaf"


def scene_sort_key(path: Path) -> tuple[str, str]:
    match = re.match(r"^egodelta_([0-9a-f]+)_(leaf|tomato)$", path.name)
    if match:
        return match.group(1), match.group(2)
    return path.name, ""


def proposal_sort_key(proposal: dict[str, Any]) -> tuple[int, int]:
    return int(proposal.get("rank", 0)), int(proposal.get("proposal_id", 0))


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    if not variants:
        raise SystemExit("--variants must name at least one variant")
    return variants


def check_reference_full(report: dict[str, Any], reference_target_json: Path | None = None) -> list[str]:
    if reference_target_json is None:
        return ["reference full baseline check requires --reference-target-json"]
    threshold_m = float(report["metadata"]["threshold_m"])
    if int(round(threshold_m * 100.0)) != 3:
        return ["reference full baseline check is only defined for the 3 cm threshold"]
    if str(report["metadata"].get("selection_policy")) != "paper":
        return ["reference full baseline check requires --selection-policy paper"]

    by_variant = {summary["variant"]: summary for summary in report["variant_summaries"]}
    full = by_variant.get("full")
    if full is None:
        return ["missing full variant summary"]

    expected_by_split = reference_full_baseline_from_target_json(
        reference_target_json,
        Path(str(report["metadata"]["gt"])),
        threshold_m,
    )

    failures: list[str] = []
    for split, expected in expected_by_split.items():
        actual = full["splits"].get(split)
        if actual is None:
            failures.append(f"{split}: missing split")
            continue
        for key in ("Scenes", "GT", "Pred", "TP", "FP", "FN"):
            if int(actual[key]) != int(expected[key]):
                failures.append(f"{split} {key}: got {actual[key]}, expected {expected[key]}")
        for key in ("Precision@3cm", "Recall@3cm", "F1@3cm"):
            if round(float(actual[key]), 4) != round(float(expected[key]), 4):
                failures.append(
                    f"{split} {key}: got {float(actual[key]):.4f}, expected {float(expected[key]):.4f}"
                )
        cm_checks = (
            ("matched_mean_distance_cm", "matched_mean_distance_cm"),
            ("gt_to_nearest_mean_distance_cm", "gt_to_nearest_mean_distance_cm"),
        )
        for actual_key, expected_key in cm_checks:
            expected_value = expected[expected_key]
            actual_cm = actual[actual_key]
            if expected_value is None:
                if actual_cm is not None:
                    failures.append(f"{split} {expected_key}: got {float(actual_cm):.2f} cm, expected none")
                continue
            expected_cm = float(expected_value)
            if actual_cm is None or round(float(actual_cm), 2) != round(expected_cm, 2):
                got = "" if actual_cm is None else f"{float(actual_cm):.2f} cm"
                failures.append(f"{split} {expected_key}: got {got}, expected {expected_cm:.2f} cm")
    return failures


def reference_full_baseline_from_target_json(
    path: Path,
    gt_path: Path,
    threshold_m: float,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    gt_entries = parse_gt_markdown(gt_path)
    by_scene: dict[str, list[list[float]]] = {}
    for row in payload.get("targets", []):
        scene = str(row["scene"])
        by_scene.setdefault(scene, []).append(
            [
                float(row["target_x"]),
                float(row["target_y"]),
                float(row["target_z"]),
            ]
        )

    evaluations_by_split: dict[str, list[Any]] = {"Leaf": [], "Tomato": [], "All": []}
    for scene, points in sorted(by_scene.items()):
        scene_targets = find_scene_targets(scene, gt_entries)
        if scene_targets is None:
            continue
        _, task, gt_points = scene_targets
        evaluation = evaluate_cut_points(
            gt_points=np.asarray(gt_points, dtype=np.float64),
            pred_points=np.asarray(points, dtype=np.float64).reshape((-1, 3)),
            scene=scene,
            threshold_m=threshold_m,
        )
        split = split_from_scene(scene, task)
        evaluations_by_split[split].append(evaluation)
        evaluations_by_split["All"].append(evaluation)

    return {
        split: aggregate_row(split, evaluations_by_split[split], threshold_m)
        for split in ("Leaf", "Tomato", "All")
    }


def cm_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if math.isfinite(value) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold_cm = int(round(float(report["metadata"]["threshold_m"]) * 100.0))
    precision_key = f"Precision@{threshold_cm}cm"
    recall_key = f"Recall@{threshold_cm}cm"
    f1_key = f"F1@{threshold_cm}cm"
    lines = [
        "# Scoring Ablation Summary",
        "",
        f"- Output root: `{report['metadata']['outputs_root']}`",
        f"- GT: `{report['metadata']['gt']}`",
        f"- Selection policy: `{report['metadata']['selection_policy']}`",
        "",
        f"| Variant | Split | Scenes | GT | Pred | TP | FP | FN | Precision@{threshold_cm}cm | Recall@{threshold_cm}cm | F1@{threshold_cm}cm | Matched Mean Dist. | Matched Median Dist. | GT-Nearest Mean Dist. |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant_summary in report["variant_summaries"]:
        variant = variant_summary["variant"]
        for split in ("Leaf", "Tomato", "All"):
            row = variant_summary["splits"][split]
            lines.append(
                "| {variant} | {split} | {scenes} | {gt} | {pred} | {tp} | {fp} | {fn} | {p:.4f} | {r:.4f} | {f1:.4f} | {mean} | {median} | {nearest} |".format(
                    variant=variant,
                    split=split,
                    scenes=row["Scenes"],
                    gt=row["GT"],
                    pred=row["Pred"],
                    tp=row["TP"],
                    fp=row["FP"],
                    fn=row["FN"],
                    p=float(row[precision_key]),
                    r=float(row[recall_key]),
                    f1=float(row[f1_key]),
                    mean=fmt_cm(row["matched_mean_distance_cm"]),
                    median=fmt_cm(row["matched_median_distance_cm"]),
                    nearest=fmt_cm(row["gt_to_nearest_mean_distance_cm"]),
                )
            )
    lines.extend(
        [
            "",
            "## Organization Check",
            "",
            "| Variant | Split | Target BBoxes | Scenes With 2D Overlay |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for organization in report["organization_summaries"]:
        variant = organization["variant"]
        for split in ("Leaf", "Tomato", "All"):
            row = organization["splits"][split]
            lines.append(
                "| {variant} | {split} | {bbox} / {targets} | {overlays} / {scenes} |".format(
                    variant=variant,
                    split=split,
                    bbox=row["target_bboxes"],
                    targets=row["selected_targets"],
                    overlays=row["scenes_with_2d_overlay"],
                    scenes=row["scenes"],
                )
            )
    if report["skipped"]:
        lines.extend(["", "## Skipped", ""])
        for item in report["skipped"]:
            lines.append(f"- `{item['variant']}` `{item['scene']}`: {item['reason']}")
    path.write_text("\n".join(lines) + "\n")


def fmt_cm(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f} cm"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize EGODELTA-Diff primitive-scoring ablations.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run scoring variants over pre/post model pairs.")
    add_run_args(run_parser)
    summarize_parser = subparsers.add_parser("summarize", help="Summarize existing ablation outputs.")
    add_summarize_args(summarize_parser)

    args = parser.parse_args()
    if args.command == "run":
        run_ablation(args)
    elif args.command == "summarize":
        summarize_ablation_cli(args)
    else:
        raise SystemExit(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
