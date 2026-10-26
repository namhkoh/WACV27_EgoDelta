#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_aria_change_baseline.evaluate_cut_points import (
    evaluate_cut_point_sets,
    load_prediction_points,
    parse_gt_label_file,
    save_evaluation_results,
    threshold_label_cm,
)


SCENES = [
    "35aa89c0-e852-4750-9608-3f79788f7ee9",
    "3b0e7a93-8e13-443f-b729-80e0926dcc42",
    "44fe03cd-d0a7-408a-8ca2-b9b302c912a0",
    "47d0e394-ae62-49e1-bcd6-2f9f236972d0",
    "4d5ebec9-3ebd-4c65-b01d-8d2a0c668df2",
    "5479d13e-b1ed-4539-aeb4-c42ce20dbf8f",
    "5d9ae0cf-5e76-43c8-af38-24547c88e5e5",
    "5ede33ac-5cf2-4a8d-88cc-ef38f9b1e5ff",
]

PROMPTS = {
    "side": "side stem of tomato plant, which is not a main stem and hung horizontally",
    "petiole": "tomato petiole||petiole||tomato leaf petiole||leaf petiole||tomato leaf stem",
    "petiole_short": "tomato petiole||tomato leaf petiole||leaf petiole",
    "combo": "tomato petiole||side stem of tomato plant, which is not a main stem and hung horizontally",
    "stem_combo": "tomato leaf stem||leaf petiole||side stem of tomato plant, which is not a main stem and hung horizontally",
}
DEFAULT_AUX_PROMPT = "main stem of tomato plant, without any leaves or branches"

DEFAULT_PARAMS: dict[str, Any] = {
    "deleafing_sam_score_thres": 0.12,
    "deleafing_min_mask_area_px": 50,
    "deleafing_max_mask_area_frac": 0.03,
    "deleafing_max_primary_masks_per_frame": 12,
    "deleafing_pipe_like_long_side_frac": 0.35,
    "deleafing_pipe_like_aspect_ratio": 6.0,
    "deleafing_primary_vertical_axis_thres": 0.90,
    "deleafing_primary_vertical_axis_max_thinness": 0.45,
    "deleafing_primary_vertical_axis_image_direction": "image_horizontal",
    "petiole_min_points_inside_mask": 4,
    "petiole_dbscan_min_samples": 3,
    "petiole_instance_min_frames": 1,
    "petiole_min_3d_major_extent_m": 0.04,
    "petiole_max_3d_major_extent_m": 0.55,
    "deleafing_aux_sam_score_thres": 0.35,
    "deleafing_aux_min_mask_area_px": 80,
    "deleafing_main_stem_min_verticality": 0.82,
    "deleafing_main_stem_min_height_frac": 0.18,
    "deleafing_main_stem_max_width_frac": 0.22,
    "deleafing_main_stem_min_aspect_ratio": 1.8,
    "deleafing_aux_min_vertical_extent_m": 0.08,
    "deleafing_aux_min_vertical_ratio": 1.25,
    "deleafing_max_aux_masks_per_frame": 8,
    "disappeared_score_thres": 0.0,
    "match_radius_m": 0.02,
}


def _trial(
    name: str,
    prompt_key: str = "side",
    **overrides: Any,
) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    return {
        "name": name,
        "prompt_key": prompt_key,
        "primary_prompts": PROMPTS[prompt_key],
        "auxiliary_prompts": DEFAULT_AUX_PROMPT,
        "params": params,
    }


TRIALS = [
    _trial("side_mr000_th000", "side", match_radius_m=0.00, disappeared_score_thres=0.00),
    _trial("side_mr001_th000", "side", match_radius_m=0.01, disappeared_score_thres=0.00),
    _trial("side_mr002_th000", "side", match_radius_m=0.02, disappeared_score_thres=0.00),
    _trial("side_mr003_th000", "side", match_radius_m=0.03, disappeared_score_thres=0.00),
    _trial("side_mr004_th000", "side", match_radius_m=0.04, disappeared_score_thres=0.00),
    _trial("side_mr005_th000", "side", match_radius_m=0.05, disappeared_score_thres=0.00),
    _trial("side_mr002_th0025", "side", match_radius_m=0.02, disappeared_score_thres=0.025),
    _trial("side_mr003_th0025", "side", match_radius_m=0.03, disappeared_score_thres=0.025),
    _trial("side_mr002_th005", "side", match_radius_m=0.02, disappeared_score_thres=0.05),
    _trial("side_mr003_th005", "side", match_radius_m=0.03, disappeared_score_thres=0.05),
    _trial("side_mr002_th010", "side", match_radius_m=0.02, disappeared_score_thres=0.10),
    _trial("side_mr003_th010", "side", match_radius_m=0.03, disappeared_score_thres=0.10),
    _trial("side_score008_mr002", "side", deleafing_sam_score_thres=0.08, match_radius_m=0.02),
    _trial("side_score008_mr003", "side", deleafing_sam_score_thres=0.08, match_radius_m=0.03),
    _trial("side_score016_mr002", "side", deleafing_sam_score_thres=0.16, match_radius_m=0.02),
    _trial("side_score016_mr003", "side", deleafing_sam_score_thres=0.16, match_radius_m=0.03),
    _trial("side_area025_mr002", "side", deleafing_min_mask_area_px=25, deleafing_max_primary_masks_per_frame=20, match_radius_m=0.02),
    _trial("side_area025_mr003", "side", deleafing_min_mask_area_px=25, deleafing_max_primary_masks_per_frame=20, match_radius_m=0.03),
    _trial("side_area080_mr002", "side", deleafing_min_mask_area_px=80, deleafing_max_primary_masks_per_frame=10, match_radius_m=0.02),
    _trial("side_area080_mr003", "side", deleafing_min_mask_area_px=80, deleafing_max_primary_masks_per_frame=10, match_radius_m=0.03),
    _trial("side_maxfrac005_mr002", "side", deleafing_max_mask_area_frac=0.05, match_radius_m=0.02),
    _trial("side_maxfrac005_mr003", "side", deleafing_max_mask_area_frac=0.05, match_radius_m=0.03),
    _trial("side_many_masks_mr002", "side", deleafing_max_primary_masks_per_frame=24, match_radius_m=0.02),
    _trial("side_many_masks_mr003", "side", deleafing_max_primary_masks_per_frame=24, match_radius_m=0.03),
    _trial("side_tight_pipe_mr002", "side", deleafing_pipe_like_long_side_frac=0.30, deleafing_pipe_like_aspect_ratio=5.0, match_radius_m=0.02),
    _trial("side_loose_pipe_mr002", "side", deleafing_pipe_like_long_side_frac=0.45, deleafing_pipe_like_aspect_ratio=8.0, match_radius_m=0.02),
    _trial("side_axis085_mr002", "side", deleafing_primary_vertical_axis_thres=0.85, match_radius_m=0.02),
    _trial("side_axis095_mr002", "side", deleafing_primary_vertical_axis_thres=0.95, match_radius_m=0.02),
    _trial("side_depth_sparse_mr002", "side", petiole_min_points_inside_mask=2, petiole_dbscan_min_samples=2, match_radius_m=0.02),
    _trial("side_depth_strict_mr002", "side", petiole_min_points_inside_mask=8, petiole_dbscan_min_samples=4, match_radius_m=0.02),
    _trial("side_extent030_045_mr002", "side", petiole_min_3d_major_extent_m=0.03, petiole_max_3d_major_extent_m=0.45, match_radius_m=0.02),
    _trial("side_extent020_070_mr002", "side", petiole_min_3d_major_extent_m=0.02, petiole_max_3d_major_extent_m=0.70, match_radius_m=0.02),
    _trial("petiole_mr001_th000", "petiole", match_radius_m=0.01, disappeared_score_thres=0.00),
    _trial("petiole_mr002_th000", "petiole", match_radius_m=0.02, disappeared_score_thres=0.00),
    _trial("petiole_mr003_th000", "petiole", match_radius_m=0.03, disappeared_score_thres=0.00),
    _trial("petiole_mr002_th005", "petiole", match_radius_m=0.02, disappeared_score_thres=0.05),
    _trial("petiole_mr003_th005", "petiole", match_radius_m=0.03, disappeared_score_thres=0.05),
    _trial("petiole_short_mr002", "petiole_short", match_radius_m=0.02),
    _trial("petiole_short_mr003", "petiole_short", match_radius_m=0.03),
    _trial("combo_mr001_th000", "combo", match_radius_m=0.01, disappeared_score_thres=0.00),
    _trial("combo_mr002_th000", "combo", match_radius_m=0.02, disappeared_score_thres=0.00),
    _trial("combo_mr003_th000", "combo", match_radius_m=0.03, disappeared_score_thres=0.00),
    _trial("combo_mr002_th005", "combo", match_radius_m=0.02, disappeared_score_thres=0.05),
    _trial("combo_mr003_th005", "combo", match_radius_m=0.03, disappeared_score_thres=0.05),
    _trial("stem_combo_mr002", "stem_combo", match_radius_m=0.02),
    _trial("stem_combo_mr003", "stem_combo", match_radius_m=0.03),
    _trial("side_aux_loose_mr002", "side", match_radius_m=0.02, deleafing_aux_sam_score_thres=0.25, deleafing_aux_min_mask_area_px=50),
    _trial("side_aux_strict_mr002", "side", match_radius_m=0.02, deleafing_aux_sam_score_thres=0.45, deleafing_aux_min_mask_area_px=120),
    _trial("side_stem_loose_mr002", "side", match_radius_m=0.02, deleafing_aux_min_vertical_extent_m=0.04, deleafing_aux_min_vertical_ratio=0.8),
    _trial("side_stem_strict_mr002", "side", match_radius_m=0.02, deleafing_aux_min_vertical_extent_m=0.12, deleafing_aux_min_vertical_ratio=1.6),
]

PRIORITY_20_TRIAL_NAMES = [
    "side_mr000_th000",
    "side_mr001_th000",
    "side_mr002_th000",
    "side_mr003_th000",
    "side_mr004_th000",
    "side_mr002_th0025",
    "side_mr002_th005",
    "side_score008_mr002",
    "side_score016_mr002",
    "side_area025_mr002",
    "side_area080_mr002",
    "side_many_masks_mr002",
    "side_tight_pipe_mr002",
    "side_loose_pipe_mr002",
    "side_depth_sparse_mr002",
    "side_extent020_070_mr002",
    "petiole_mr002_th000",
    "petiole_short_mr002",
    "combo_mr002_th000",
    "stem_combo_mr002",
]


def _ordered_trials() -> list[dict[str, Any]]:
    trial_by_name = {trial["name"]: trial for trial in TRIALS}
    ordered = [trial_by_name[name] for name in PRIORITY_20_TRIAL_NAMES if name in trial_by_name]
    priority_names = {trial["name"] for trial in ordered}
    ordered.extend(trial for trial in TRIALS if trial["name"] not in priority_names)
    return ordered


def _metric_key(tau_m: float, metric: str) -> str:
    return f"micro_{metric}@{threshold_label_cm(tau_m)}"


def _rank_key(row: dict[str, Any], tau_m: float) -> tuple:
    aggregate = row.get("aggregate") or {}
    matched_mean = aggregate.get("matched_mean_distance")
    nearest_mean = aggregate.get("gt_to_nearest_mean_distance")
    matched_rank = -float(matched_mean) if matched_mean is not None else float("-inf")
    nearest_rank = -float(nearest_mean) if nearest_mean is not None else float("-inf")
    return (
        int(aggregate.get("total_TP", -1)),
        float(aggregate.get(_metric_key(tau_m, "F1"), -1.0)),
        float(aggregate.get(_metric_key(tau_m, "Precision"), -1.0)),
        float(aggregate.get(_metric_key(tau_m, "Recall"), -1.0)),
        matched_rank,
        nearest_rank,
        -int(aggregate.get("total_num_pred", 10**9)),
    )


def _attempt_complete(attempt_root: Path, scene_ids: list[str]) -> bool:
    return all((attempt_root / scene_id / "final_predictions.json").exists() for scene_id in scene_ids)


def _attempt_command(args: argparse.Namespace, trial: dict[str, Any], attempt_root: Path) -> list[str]:
    params = trial["params"]
    cmd = [
        sys.executable,
        "scripts/run_deleaf_benchmark_batch.py",
        "--out-root",
        str(attempt_root),
        "--devices",
        args.devices,
        "--conda-env",
        args.conda_env,
        "--frame-stride",
        str(args.frame_stride),
        "--sam-batch-size",
        str(args.sam_batch_size),
        "--poll-s",
        str(args.poll_s),
        "--deleafing-sam-score-thres",
        str(params["deleafing_sam_score_thres"]),
        "--deleafing-min-mask-area-px",
        str(params["deleafing_min_mask_area_px"]),
        "--deleafing-max-mask-area-frac",
        str(params["deleafing_max_mask_area_frac"]),
        "--deleafing-max-primary-masks-per-frame",
        str(params["deleafing_max_primary_masks_per_frame"]),
        "--deleafing-pipe-like-long-side-frac",
        str(params["deleafing_pipe_like_long_side_frac"]),
        "--deleafing-pipe-like-aspect-ratio",
        str(params["deleafing_pipe_like_aspect_ratio"]),
        "--deleafing-primary-vertical-axis-thres",
        str(params["deleafing_primary_vertical_axis_thres"]),
        "--deleafing-primary-vertical-axis-max-thinness",
        str(params["deleafing_primary_vertical_axis_max_thinness"]),
        "--deleafing-primary-vertical-axis-image-direction",
        str(params["deleafing_primary_vertical_axis_image_direction"]),
        "--petiole-min-points-inside-mask",
        str(params["petiole_min_points_inside_mask"]),
        "--petiole-dbscan-min-samples",
        str(params["petiole_dbscan_min_samples"]),
        "--petiole-instance-min-frames",
        str(params["petiole_instance_min_frames"]),
        "--petiole-min-3d-major-extent-m",
        str(params["petiole_min_3d_major_extent_m"]),
        "--petiole-max-3d-major-extent-m",
        str(params["petiole_max_3d_major_extent_m"]),
        "--deleafing-aux-sam-score-thres",
        str(params["deleafing_aux_sam_score_thres"]),
        "--deleafing-aux-min-mask-area-px",
        str(params["deleafing_aux_min_mask_area_px"]),
        "--deleafing-main-stem-min-verticality",
        str(params["deleafing_main_stem_min_verticality"]),
        "--deleafing-main-stem-min-height-frac",
        str(params["deleafing_main_stem_min_height_frac"]),
        "--deleafing-main-stem-max-width-frac",
        str(params["deleafing_main_stem_max_width_frac"]),
        "--deleafing-main-stem-min-aspect-ratio",
        str(params["deleafing_main_stem_min_aspect_ratio"]),
        "--deleafing-aux-min-vertical-extent-m",
        str(params["deleafing_aux_min_vertical_extent_m"]),
        "--deleafing-aux-min-vertical-ratio",
        str(params["deleafing_aux_min_vertical_ratio"]),
        "--deleafing-max-aux-masks-per-frame",
        str(params["deleafing_max_aux_masks_per_frame"]),
        "--disappeared-score-thres",
        str(params["disappeared_score_thres"]),
        "--match-radius-m",
        str(params["match_radius_m"]),
    ]
    if args.force:
        cmd.append("--force")
    for scene_id in args.scene_id:
        cmd.extend(["--scene-id", scene_id])
    return cmd


def _evaluate_subset(args: argparse.Namespace, pred_root: Path, eval_root: Path) -> dict[str, Any]:
    gt_all = parse_gt_label_file(args.gt_labels, "deleafing")
    pred_all, warnings = load_prediction_points(pred_root, "deleafing", known_scene_ids=set(gt_all))
    gt = {scene_id: gt_all.get(scene_id, []) for scene_id in args.scene_id}
    pred = {scene_id: pred_all.get(scene_id, []) for scene_id in args.scene_id}
    per_scene, aggregate = evaluate_cut_point_sets(
        gt,
        pred,
        tau_m=args.tau_m,
        include_prediction_only_scenes=False,
    )
    save_evaluation_results(
        per_scene,
        aggregate,
        eval_root,
        metadata={
            "task": "deleafing",
            "tau_m": args.tau_m,
            "scene_ids": args.scene_id,
            "pred_root": str(pred_root),
            "warnings": warnings,
        },
    )
    return aggregate


def _write_best_report(out_root: Path, summary: list[dict[str, Any]], tau_m: float) -> dict[str, Any] | None:
    successful = [row for row in summary if row.get("status") == "ok" and row.get("aggregate")]
    if not successful:
        return None
    ranked = sorted(successful, key=lambda row: _rank_key(row, tau_m), reverse=True)
    best = ranked[0]
    report = {
        "ranking_rule": [
            "total_TP",
            _metric_key(tau_m, "F1"),
            _metric_key(tau_m, "Precision"),
            _metric_key(tau_m, "Recall"),
            "lower matched_mean_distance",
            "lower gt_to_nearest_mean_distance",
            "fewer predictions",
        ],
        "best": best,
        "top_10": ranked[:10],
    }
    (out_root / "best_trial.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 20 prioritized deleafing tuning trials on the 8-scene subset and report the best parameters."
    )
    parser.add_argument("--out-root", default="/tmp/sam_deleaf_tuning_8sample_20")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--conda-env", default="sam3_aria")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sam-batch-size", type=int, default=16)
    parser.add_argument("--poll-s", type=float, default=120.0)
    parser.add_argument("--tau-m", type=float, default=0.03)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--start-at", type=int, default=1, help="1-indexed first trial to run.")
    parser.add_argument("--force", action="store_true", help="Rerun complete trial outputs.")
    parser.add_argument(
        "--gt-labels",
        default="/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt",
    )
    parser.add_argument("--scene-id", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.scene_id = args.scene_id or list(SCENES)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "tuning_summary.json"
    trials_path = out_root / "trial_configs.json"
    summary: list[dict[str, Any]] = []
    if summary_path.exists() and not args.force:
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = []
    existing_by_attempt = {row.get("attempt"): row for row in summary}
    ordered_trials = _ordered_trials()
    trials = ordered_trials[args.start_at - 1 : args.max_trials]
    trials_path.write_text(json.dumps(ordered_trials[: args.max_trials], indent=2) + "\n")

    for trial_idx, trial in enumerate(trials, start=args.start_at):
        attempt = f"trial_{trial_idx:02d}_{trial['name']}"
        attempt_root = out_root / attempt
        eval_root = out_root / f"{attempt}_eval_{threshold_label_cm(args.tau_m)}_subset8"
        env = os.environ.copy()
        env["SAM_ARIA_DELEAFING_PRIMARY_PROMPTS"] = trial["primary_prompts"]
        env["SAM_ARIA_DELEAFING_AUXILIARY_PROMPTS"] = trial["auxiliary_prompts"]

        if not args.force and _attempt_complete(attempt_root, args.scene_id):
            print(f"===== {attempt} already complete; evaluating/resuming =====", flush=True)
            aggregate = _evaluate_subset(args, attempt_root, eval_root)
            row = {
                "attempt": attempt,
                "status": "ok",
                "returncode": 0,
                "elapsed_min": 0.0,
                "trial": trial,
                "eval_root": str(eval_root),
                "aggregate": aggregate,
                "resumed": True,
            }
        elif not args.force and existing_by_attempt.get(attempt, {}).get("status") == "ok":
            print(f"===== {attempt} already evaluated; skipping =====", flush=True)
            row = existing_by_attempt[attempt]
        else:
            cmd = _attempt_command(args, trial, attempt_root)
            started = time.time()
            print(f"===== {attempt} =====", flush=True)
            print("primary prompts:", env["SAM_ARIA_DELEAFING_PRIMARY_PROMPTS"], flush=True)
            print("params:", json.dumps(trial["params"], sort_keys=True), flush=True)
            print("command:", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, env=env)
            elapsed_min = (time.time() - started) / 60.0
            if proc.returncode != 0:
                row = {
                    "attempt": attempt,
                    "status": "failed",
                    "returncode": proc.returncode,
                    "elapsed_min": elapsed_min,
                    "trial": trial,
                }
                print(f"{attempt} failed with code {proc.returncode}", flush=True)
            else:
                aggregate = _evaluate_subset(args, attempt_root, eval_root)
                row = {
                    "attempt": attempt,
                    "status": "ok",
                    "returncode": 0,
                    "elapsed_min": elapsed_min,
                    "trial": trial,
                    "eval_root": str(eval_root),
                    "aggregate": aggregate,
                }

        existing_by_attempt[attempt] = row
        summary = list(existing_by_attempt.values())
        summary.sort(key=lambda item: str(item.get("attempt", "")))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        report = _write_best_report(out_root, summary, args.tau_m)
        if row.get("status") == "ok":
            aggregate = row["aggregate"]
            print(
                f"{attempt}: TP={aggregate['total_TP']} "
                f"Pred={aggregate['total_num_pred']} "
                f"P={aggregate[_metric_key(args.tau_m, 'Precision')]:.4f} "
                f"R={aggregate[_metric_key(args.tau_m, 'Recall')]:.4f} "
                f"F1={aggregate[_metric_key(args.tau_m, 'F1')]:.4f}",
                flush=True,
            )
        if report:
            best = report["best"]
            best_agg = best["aggregate"]
            print(
                f"current best: {best['attempt']} "
                f"TP={best_agg['total_TP']} "
                f"F1={best_agg[_metric_key(args.tau_m, 'F1')]:.4f}",
                flush=True,
            )

    print(f"Summary: {summary_path}", flush=True)
    print(f"Best: {out_root / 'best_trial.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
