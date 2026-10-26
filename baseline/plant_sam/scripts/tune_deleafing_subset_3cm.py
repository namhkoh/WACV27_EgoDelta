#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_aria_change_baseline.evaluate_cut_points import (
    evaluate_cut_point_sets,
    load_prediction_points,
    parse_gt_label_file,
    save_evaluation_results,
)


SCENES = [
    "3b0e7a93-8e13-443f-b729-80e0926dcc42",
    "4d5ebec9-3ebd-4c65-b01d-8d2a0c668df2",
    "5d9ae0cf-5e76-43c8-af38-24547c88e5e5",
    "5ede33ac-5cf2-4a8d-88cc-ef38f9b1e5ff",
]

CURRENT_PRIMARY = "side stem of tomato plant, which is not a main stem and hung horizontally"
PETIOLE_PRIMARY = "tomato petiole||petiole||tomato leaf petiole||leaf petiole||tomato leaf stem"
SIDE_BRANCH_PRIMARY = "tomato petiole||side stem of tomato plant, which is not a main stem and hung horizontally"
CURRENT_AUX = "main stem of tomato plant, without any leaves or branches"


ATTEMPTS = [
    {"name": "mr002_th000", "match_radius_m": 0.02, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY},
    {"name": "mr003_th000", "match_radius_m": 0.03, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY},
    {"name": "mr001_th000", "match_radius_m": 0.01, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY},
    {"name": "mr004_th000", "match_radius_m": 0.04, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY},
    {"name": "mr005_th000", "match_radius_m": 0.05, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY},
    {"name": "mr002_th0025", "match_radius_m": 0.02, "disappeared_score_thres": 0.025, "primary": CURRENT_PRIMARY},
    {"name": "mr003_th0025", "match_radius_m": 0.03, "disappeared_score_thres": 0.025, "primary": CURRENT_PRIMARY},
    {"name": "mr002_th005", "match_radius_m": 0.02, "disappeared_score_thres": 0.05, "primary": CURRENT_PRIMARY},
    {"name": "mr003_th005", "match_radius_m": 0.03, "disappeared_score_thres": 0.05, "primary": CURRENT_PRIMARY},
    {"name": "mr002_th010", "match_radius_m": 0.02, "disappeared_score_thres": 0.10, "primary": CURRENT_PRIMARY},
    {"name": "mr003_th010", "match_radius_m": 0.03, "disappeared_score_thres": 0.10, "primary": CURRENT_PRIMARY},
    {"name": "petiole_mr002_th000", "match_radius_m": 0.02, "disappeared_score_thres": 0.00, "primary": PETIOLE_PRIMARY},
    {"name": "petiole_mr003_th000", "match_radius_m": 0.03, "disappeared_score_thres": 0.00, "primary": PETIOLE_PRIMARY},
    {"name": "petiole_mr002_th005", "match_radius_m": 0.02, "disappeared_score_thres": 0.05, "primary": PETIOLE_PRIMARY},
    {"name": "petiole_mr003_th005", "match_radius_m": 0.03, "disappeared_score_thres": 0.05, "primary": PETIOLE_PRIMARY},
    {"name": "combo_mr002_th000", "match_radius_m": 0.02, "disappeared_score_thres": 0.00, "primary": SIDE_BRANCH_PRIMARY},
    {"name": "combo_mr003_th000", "match_radius_m": 0.03, "disappeared_score_thres": 0.00, "primary": SIDE_BRANCH_PRIMARY},
    {"name": "combo_mr002_th005", "match_radius_m": 0.02, "disappeared_score_thres": 0.05, "primary": SIDE_BRANCH_PRIMARY},
    {"name": "loose_area_mr002_th000", "match_radius_m": 0.02, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY, "min_area": 25, "max_masks": 20},
    {"name": "loose_area_mr003_th000", "match_radius_m": 0.03, "disappeared_score_thres": 0.00, "primary": CURRENT_PRIMARY, "min_area": 25, "max_masks": 20},
]


def _attempt_command(args: argparse.Namespace, attempt: dict, out_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_deleaf_benchmark_batch.py",
        "--out-root",
        str(out_root),
        "--devices",
        args.devices,
        "--frame-stride",
        str(args.frame_stride),
        "--sam-batch-size",
        str(args.sam_batch_size),
        "--force",
        "--poll-s",
        str(args.poll_s),
        "--disappeared-score-thres",
        str(attempt["disappeared_score_thres"]),
        "--match-radius-m",
        str(attempt["match_radius_m"]),
        "--deleafing-min-mask-area-px",
        str(attempt.get("min_area", args.deleafing_min_mask_area_px)),
        "--deleafing-max-primary-masks-per-frame",
        str(attempt.get("max_masks", args.deleafing_max_primary_masks_per_frame)),
    ]
    for scene_id in args.scene_id:
        cmd.extend(["--scene-id", scene_id])
    return cmd


def _evaluate_subset(args: argparse.Namespace, pred_root: Path, eval_root: Path) -> dict:
    gt_all = parse_gt_label_file(args.gt_labels, "deleafing")
    pred_all, warnings = load_prediction_points(
        pred_root,
        "deleafing",
        known_scene_ids=set(gt_all),
    )
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run up to 20 deleafing tuning attempts on the four-scene subset."
    )
    parser.add_argument("--out-root", default="/tmp/sam_deleaf_tuning_3cm")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sam-batch-size", type=int, default=16)
    parser.add_argument("--poll-s", type=float, default=120.0)
    parser.add_argument("--tau-m", type=float, default=0.03)
    parser.add_argument("--stop-at-tp", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--deleafing-min-mask-area-px", type=int, default=50)
    parser.add_argument("--deleafing-max-primary-masks-per-frame", type=int, default=12)
    parser.add_argument(
        "--gt-labels",
        default="/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt",
    )
    parser.add_argument("--scene-id", action="append", default=list(SCENES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "tuning_summary.json"
    summary: list[dict] = []

    for attempt_idx, attempt in enumerate(ATTEMPTS[: args.max_attempts], start=1):
        attempt_name = f"attempt_{attempt_idx:02d}_{attempt['name']}"
        attempt_root = out_root / attempt_name
        eval_root = out_root / f"{attempt_name}_eval_{int(round(args.tau_m * 100))}cm_subset4"
        env = os.environ.copy()
        env["SAM_ARIA_DELEAFING_PRIMARY_PROMPTS"] = attempt["primary"]
        env["SAM_ARIA_DELEAFING_AUXILIARY_PROMPTS"] = attempt.get("auxiliary", CURRENT_AUX)
        cmd = _attempt_command(args, attempt, attempt_root)

        started = time.time()
        print(f"===== {attempt_name} =====", flush=True)
        print("primary prompts:", env["SAM_ARIA_DELEAFING_PRIMARY_PROMPTS"], flush=True)
        print("command:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, env=env)
        elapsed_min = (time.time() - started) / 60.0
        if proc.returncode != 0:
            row = {
                "attempt": attempt_name,
                "returncode": proc.returncode,
                "elapsed_min": elapsed_min,
                "config": attempt,
                "status": "failed",
            }
            summary.append(row)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{attempt_name} failed with code {proc.returncode}", flush=True)
            continue

        aggregate = _evaluate_subset(args, attempt_root, eval_root)
        row = {
            "attempt": attempt_name,
            "returncode": proc.returncode,
            "elapsed_min": elapsed_min,
            "config": attempt,
            "eval_root": str(eval_root),
            "aggregate": aggregate,
            "status": "ok",
        }
        summary.append(row)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"{attempt_name}: TP={aggregate['total_TP']} "
            f"Pred={aggregate['total_num_pred']} "
            f"F1={aggregate[f'micro_F1@{int(round(args.tau_m * 100))}cm']:.4f} "
            f"eval={eval_root}",
            flush=True,
        )
        if int(aggregate["total_TP"]) >= args.stop_at_tp:
            print(f"Reached target TP >= {args.stop_at_tp}; stopping.", flush=True)
            break

    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
