#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egodelta_diff.contact_proposal import (  # noqa: E402
    contact_config_from_dict,
    evaluate_contact_proposal_sets,
    generate_contact_proposals,
    save_contact_eval_json,
    save_contact_proposals_json,
    save_contact_proposals_ply,
)
from egodelta_diff.data import find_scene_targets, parse_gt_markdown  # noqa: E402
from egodelta_diff.data import load_2dgs_ply  # noqa: E402
from egodelta_diff.targets import (  # noqa: E402
    evaluate_proposals,
    generate_target_proposals,
    proposal_config_from_dict,
    save_proposal_eval_json,
    save_target_proposals_json,
    save_target_proposals_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GT-free target proposals from removed_primitives.ply and evaluate them against gt.md when available."
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs_ready_full"))
    parser.add_argument("--gt", type=Path, default=Path("gt.md"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--write-proposals", action="store_true")
    parser.add_argument("--mode", choices=["contact", "baseline_nms"], default=None)

    parser.add_argument("--seed-score-min", type=float, default=None)
    parser.add_argument("--nms-radius-m", type=float, default=None)
    parser.add_argument("--support-radius-m", type=float, default=None)
    parser.add_argument("--min-support", type=int, default=None)
    parser.add_argument("--min-score-mass", type=float, default=None)
    parser.add_argument("--min-peak-score", type=float, default=None)
    parser.add_argument("--eval-gate-m", type=float, default=None)
    return parser.parse_args()


def load_removed_primitives(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(path)["vertex"].data
    points = np.stack(
        [np.asarray(vertex[field]).copy() for field in ("x", "y", "z")],
        axis=1,
    ).astype(np.float32)
    scores = np.asarray(vertex["score"]).copy().astype(np.float32)
    if "source_index" in vertex.dtype.names:
        source_indices = np.asarray(vertex["source_index"]).copy().astype(np.int64)
    else:
        source_indices = np.arange(len(points), dtype=np.int64)
    if {"red", "green", "blue"}.issubset(vertex.dtype.names or ()):
        rgb01 = np.stack(
            [np.asarray(vertex[field]).copy() for field in ("red", "green", "blue")],
            axis=1,
        ).astype(np.float32) / 255.0
    else:
        rgb01 = np.zeros((len(points), 3), dtype=np.float32)
    if "cluster" in vertex.dtype.names:
        cluster_labels = np.asarray(vertex["cluster"]).copy().astype(np.int32)
    else:
        cluster_labels = np.full(len(points), -1, dtype=np.int32)
    return points, scores, source_indices, rgb01, cluster_labels


def scene_sort_key(path: Path) -> tuple[str, str]:
    match = re.match(r"^egodelta_([0-9a-f]+)_(leaf|tomato)$", path.name)
    if match:
        return match.group(1), match.group(2)
    return path.name, ""


def load_proposal_cli_config(args: argparse.Namespace) -> tuple[str, dict[str, Any], float]:
    config = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    proposal_cfg = dict((config or {}).get("proposal", {}))
    mode = str(args.mode or proposal_cfg.get("mode", "contact"))
    overrides = {
        "seed_score_min": args.seed_score_min,
        "nms_radius_m": args.nms_radius_m,
        "support_radius_m": args.support_radius_m,
        "min_support": args.min_support,
        "min_score_mass": args.min_score_mass,
        "min_peak_score": args.min_peak_score,
    }
    for key, value in overrides.items():
        if value is not None:
            proposal_cfg.setdefault("baseline_nms", {})[key] = value
    eval_gate_m = float(
        args.eval_gate_m
        if args.eval_gate_m is not None
        else proposal_cfg.get("eval_gate_m", (config or {}).get("gt", {}).get("tolerance_m", 0.03))
    )
    proposal_cfg["mode"] = mode
    return mode, proposal_cfg, eval_gate_m


def evaluate_outputs(
    outputs_root: Path,
    gt_path: Path,
    proposal_cfg: dict[str, Any],
    eval_gate_m: float,
    write_proposals: bool = False,
) -> dict[str, Any]:
    gt_entries = parse_gt_markdown(gt_path)
    mode = str(proposal_cfg.get("mode", "contact"))
    contact_config = contact_config_from_dict(proposal_cfg)
    proposal_config = proposal_config_from_dict(proposal_cfg)
    rows: list[dict[str, Any]] = []

    for out_dir in sorted((p for p in outputs_root.iterdir() if p.is_dir()), key=scene_sort_key):
        removed_ply = out_dir / "removed_primitives.ply"
        if not removed_ply.exists():
            continue
        removed = load_removed_primitives(removed_ply)
        if mode == "contact":
            points, scores, source_indices, rgb01, removed_mask, cluster_labels = load_contact_arrays(out_dir, removed)
            proposals = generate_contact_proposals(
                points=points,
                scores=scores,
                source_indices=source_indices,
                rgb01=rgb01,
                removed_mask=removed_mask,
                cluster_labels=cluster_labels,
                config=contact_config,
                object_type=object_type_from_scene_name(out_dir.name),
            )
            metadata = {
                "scene_name": out_dir.name,
                "object_type": object_type_from_scene_name(out_dir.name),
                "mode": "contact",
                "config": contact_config.__dict__,
                "input_removed_primitive_count": int(removed_mask.sum()),
                "gt_free": True,
            }
            if write_proposals:
                save_contact_proposals_json(out_dir / "contact_target_proposals.json", proposals, metadata)
                save_contact_proposals_ply(out_dir / "contact_target_proposals.ply", proposals)
        else:
            points, scores, source_indices, _, _ = removed
            proposals = generate_target_proposals(points, scores, source_indices, proposal_config)
            metadata = {
                "scene_name": out_dir.name,
                "mode": "baseline_nms",
                "config": proposal_config.__dict__,
                "input_removed_primitive_count": int(len(points)),
                "gt_free": True,
            }
            if write_proposals:
                save_target_proposals_json(out_dir / "target_proposals.json", proposals, metadata)
                save_target_proposals_ply(out_dir / "target_proposals.ply", proposals)

        scene_targets = find_scene_targets(out_dir.name, gt_entries)
        row: dict[str, Any] = {
            "scene": out_dir.name,
            "proposal_status": "ok",
            "proposal_mode": mode,
        }
        if mode == "contact":
            primary_count = sum(1 for proposal in proposals if proposal.primary)
            row["proposal_count"] = int(primary_count)
            row["all_candidate_count"] = len(proposals)
        else:
            row["proposal_count"] = len(proposals)
        if scene_targets is None:
            row["gt_status"] = "missing"
        else:
            scene_uuid, task, gt_points = scene_targets
            if mode == "contact":
                metrics = evaluate_contact_proposal_sets(proposals, gt_points, gate_m=eval_gate_m)
            else:
                metrics = evaluate_proposals(proposals, gt_points, gate_m=eval_gate_m)
            metrics["scene_name"] = out_dir.name
            metrics["scene_uuid"] = scene_uuid
            metrics["task"] = task
            metrics["eval_gate_m"] = eval_gate_m
            row.update(
                {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"matches", "primary_metrics", "all_candidate_metrics"}
                }
            )
            if mode == "contact":
                all_metrics = metrics["all_candidate_metrics"]
                for key in ("proposal_count", "tp", "fp", "fn", "precision", "recall", "f1", "count_l1"):
                    row[f"all_candidate_{key}"] = all_metrics[key]
            row["gt_status"] = "ok"
            if write_proposals:
                if mode == "contact":
                    save_contact_eval_json(out_dir / "contact_proposal_eval.json", metrics)
                else:
                    save_proposal_eval_json(out_dir / "proposal_eval.json", metrics)
        rows.append(row)

    compact_config = contact_config.__dict__ if mode == "contact" else proposal_config.__dict__
    compact_config = {"mode": mode, **compact_config}
    return summarize_rows(outputs_root, gt_path, compact_config, eval_gate_m, rows)


def load_contact_arrays(
    out_dir: Path,
    removed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    removed_points, removed_scores, removed_source_indices, removed_rgb01, removed_clusters = removed
    diagnostics_path = out_dir / "diagnostics.json"
    if not diagnostics_path.exists():
        return (
            removed_points,
            removed_scores,
            removed_source_indices,
            removed_rgb01,
            np.ones(len(removed_points), dtype=bool),
            removed_clusters,
        )

    diagnostics = json.loads(diagnostics_path.read_text())
    pre_ply = diagnostics.get("pre_ply")
    if not pre_ply:
        return (
            removed_points,
            removed_scores,
            removed_source_indices,
            removed_rgb01,
            np.ones(len(removed_points), dtype=bool),
            removed_clusters,
        )

    g_pre = load_2dgs_ply(Path(pre_ply), device="cpu")
    points = g_pre.mu.detach().cpu().numpy().astype(np.float32)
    rgb01 = g_pre.rgb01.detach().cpu().numpy().astype(np.float32)
    source_indices = g_pre.source_indices.detach().cpu().numpy().astype(np.int64)
    scores = np.zeros(len(points), dtype=np.float32)
    removed_mask = np.zeros(len(points), dtype=bool)
    cluster_labels = np.full(len(points), -1, dtype=np.int32)

    valid = (removed_source_indices >= 0) & (removed_source_indices < len(points))
    valid_source = removed_source_indices[valid].astype(np.int64)
    scores[valid_source] = removed_scores[valid]
    removed_mask[valid_source] = True
    cluster_labels[valid_source] = removed_clusters[valid]
    return points, scores, source_indices, rgb01, removed_mask, cluster_labels


def summarize_rows(
    outputs_root: Path,
    gt_path: Path,
    proposal_cfg: dict[str, Any],
    eval_gate_m: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("gt_status") == "ok"]
    tp = sum(int(row["tp"]) for row in ok_rows)
    fp = sum(int(row["fp"]) for row in ok_rows)
    fn = sum(int(row["fn"]) for row in ok_rows)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    micro_f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    macro_f1 = float(np.mean([float(row["f1"]) for row in ok_rows])) if ok_rows else 0.0
    total_count_l1 = sum(int(row["count_l1"]) for row in ok_rows)
    exact_count = sum(1 for row in ok_rows if bool(row["exact_count"]))
    all_tp = sum(int(row.get("all_candidate_tp", 0)) for row in ok_rows)
    all_fp = sum(int(row.get("all_candidate_fp", 0)) for row in ok_rows)
    all_fn = sum(int(row.get("all_candidate_fn", 0)) for row in ok_rows)
    all_precision = float(all_tp / (all_tp + all_fp)) if (all_tp + all_fp) else 0.0
    all_recall = float(all_tp / (all_tp + all_fn)) if (all_tp + all_fn) else 0.0
    all_micro_f1 = (
        float(2.0 * all_precision * all_recall / (all_precision + all_recall))
        if (all_precision + all_recall)
        else 0.0
    )
    all_macro_f1_rows = [row for row in ok_rows if "all_candidate_f1" in row]
    all_macro_f1 = (
        float(np.mean([float(row["all_candidate_f1"]) for row in all_macro_f1_rows]))
        if all_macro_f1_rows
        else 0.0
    )
    tp_distance_rows = [row for row in ok_rows if np.isfinite(float(row["mean_distance_m"]))]
    mean_distance_cm = (
        float(np.mean([float(row["mean_distance_m"]) for row in tp_distance_rows]) * 100.0)
        if tp_distance_rows
        else float("inf")
    )
    max_distance_cm = (
        float(np.max([float(row["max_distance_m"]) for row in tp_distance_rows]) * 100.0)
        if tp_distance_rows
        else float("inf")
    )

    task_summaries = _task_summaries(ok_rows)
    return {
        "outputs_root": str(outputs_root),
        "gt": str(gt_path),
        "proposal_config": proposal_cfg,
        "eval_gate_m": eval_gate_m,
        "scene_count": len(rows),
        "ok_scene_count": len(ok_rows),
        "missing_gt_scene_count": len(rows) - len(ok_rows),
        "exact_count_scene_count": exact_count,
        "exact_count_rate": float(exact_count / len(ok_rows)) if ok_rows else 0.0,
        "total_count_l1": int(total_count_l1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "all_candidate_tp": int(all_tp),
        "all_candidate_fp": int(all_fp),
        "all_candidate_fn": int(all_fn),
        "all_candidate_micro_precision": all_precision,
        "all_candidate_micro_recall": all_recall,
        "all_candidate_micro_f1": all_micro_f1,
        "all_candidate_macro_f1": all_macro_f1,
        "mean_distance_cm": mean_distance_cm,
        "max_distance_cm": max_distance_cm,
        "task_summaries": task_summaries,
        "hard_cases": _hard_cases(ok_rows),
        "rows": rows,
    }


def _task_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for task in sorted({str(row.get("task", "unknown")) for row in rows}):
        task_rows = [row for row in rows if str(row.get("task", "unknown")) == task]
        if not task_rows:
            continue
        tp = sum(int(row["tp"]) for row in task_rows)
        fp = sum(int(row["fp"]) for row in task_rows)
        fn = sum(int(row["fn"]) for row in task_rows)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        output[task] = {
            "scene_count": len(task_rows),
            "gt_count": int(sum(int(row["gt_count"]) for row in task_rows)),
            "proposal_count": int(sum(int(row["proposal_count"]) for row in task_rows)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": float(np.mean([float(row["f1"]) for row in task_rows])),
            "count_l1": int(sum(int(row["count_l1"]) for row in task_rows)),
        }
    return output


def _hard_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_tomato = {
        "egodelta_35aa89c0e852_tomato",
        "egodelta_64aaf8574f83_tomato",
        "egodelta_7830a1180c98_tomato",
    }
    low_f1 = sorted(
        rows,
        key=lambda row: (
            float(row.get("f1", 0.0)),
            -int(row.get("fn", 0)),
            -int(row.get("fp", 0)),
        ),
    )[:10]
    tomato_rows = [
        row for row in rows if row.get("scene") in fixed_tomato or (row.get("task") == "harvesting" and float(row.get("f1", 0.0)) < 1.0)
    ]
    return {
        "low_f1": [_compact_hard_case(row) for row in low_f1],
        "tomato_focus": [_compact_hard_case(row) for row in sorted(tomato_rows, key=lambda row: str(row["scene"]))],
    }


def _compact_hard_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": row["scene"],
        "task": row.get("task"),
        "proposal_count": row.get("proposal_count"),
        "gt_count": row.get("gt_count"),
        "tp": row.get("tp"),
        "fp": row.get("fp"),
        "fn": row.get("fn"),
        "f1": row.get("f1"),
        "assignment_mean_distance_cm": (
            float(row["assignment_mean_distance_m"]) * 100.0
            if np.isfinite(float(row.get("assignment_mean_distance_m", float("inf"))))
            else float("inf")
        ),
    }


def object_type_from_scene_name(scene_name: str) -> str:
    if scene_name.endswith("_tomato"):
        return "tomato"
    return "leaf"


def main() -> None:
    args = parse_args()
    _, proposal_cfg, eval_gate_m = load_proposal_cli_config(args)
    summary = evaluate_outputs(
        outputs_root=args.outputs_root,
        gt_path=args.gt,
        proposal_cfg=proposal_cfg,
        eval_gate_m=eval_gate_m,
        write_proposals=args.write_proposals,
    )
    print(json.dumps(summary, indent=2))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

