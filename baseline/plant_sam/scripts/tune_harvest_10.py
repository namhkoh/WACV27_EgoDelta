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


DEFAULT_PRIMARY = "tomato fruit||ripe tomato||green tomato||tomato"
RIPE_PRIMARY = "tomato fruit||ripe tomato"
FRUIT_ONLY_PRIMARY = "tomato fruit"
GENERIC_PRIMARY = "tomato"

DEFAULT_AUX = (
    "tomato stem||tomato peduncle||tomato pedicel||fruit pedicel||fruit stem||"
    "stem connected to tomato||tomato calyx||calyx||tomato truss||tomato cluster stem"
)
PEDUNCLE_AUX = "tomato peduncle||tomato pedicel||fruit pedicel||stem connected to tomato||tomato calyx||calyx"


def _trial(
    name: str,
    primary_prompts: str = DEFAULT_PRIMARY,
    auxiliary_prompts: str = DEFAULT_AUX,
    use_aux_prompts: bool = True,
    sam_score_thres: float = 0.25,
    min_mask_area_px: int = 100,
    max_mask_area_frac: float = 0.40,
    min_visible_post_views: int = 3,
    disappeared_score_thres: float = 0.60,
    match_radius_m: float = 0.12,
    change_score_mode: str = "auto",
) -> dict[str, Any]:
    return {
        "name": name,
        "primary_prompts": primary_prompts,
        "auxiliary_prompts": auxiliary_prompts,
        "use_aux_prompts": use_aux_prompts,
        "params": {
            "sam_score_thres": sam_score_thres,
            "min_mask_area_px": min_mask_area_px,
            "max_mask_area_frac": max_mask_area_frac,
            "min_visible_post_views": min_visible_post_views,
            "disappeared_score_thres": disappeared_score_thres,
            "match_radius_m": match_radius_m,
            "change_score_mode": change_score_mode,
        },
    }


TRIALS = [
    _trial("baseline_default"),
    _trial("change040", disappeared_score_thres=0.40),
    _trial("change020", disappeared_score_thres=0.20),
    _trial("score015_change040", sam_score_thres=0.15, disappeared_score_thres=0.40),
    _trial("score035_change040", sam_score_thres=0.35, disappeared_score_thres=0.40),
    _trial("ripe_primary_change040", primary_prompts=RIPE_PRIMARY, disappeared_score_thres=0.40),
    _trial("fruit_only_change040", primary_prompts=FRUIT_ONLY_PRIMARY, disappeared_score_thres=0.40),
    _trial("generic_tomato_change040", primary_prompts=GENERIC_PRIMARY, disappeared_score_thres=0.40),
    _trial("peduncle_aux_change040", auxiliary_prompts=PEDUNCLE_AUX, disappeared_score_thres=0.40),
    _trial(
        "no_aux_proxy_change040",
        use_aux_prompts=False,
        disappeared_score_thres=0.40,
        max_mask_area_frac=0.25,
    ),
]


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


def _scene_id_from_manifest(path: Path) -> str:
    return path.parents[1].name


def _default_scene_ids(args: argparse.Namespace) -> list[str]:
    gt_all = parse_gt_label_file(args.gt_labels, "harvesting")
    manifest_scene_ids = {
        _scene_id_from_manifest(path)
        for path in Path(args.pairs_fast_root).glob("*/tomato_pair/manifest.json")
    }
    scene_ids = sorted(
        scene_id
        for scene_id, points in gt_all.items()
        if points and scene_id in manifest_scene_ids
    )
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]
    return scene_ids


def _attempt_complete(attempt_root: Path, scene_ids: list[str]) -> bool:
    return all((attempt_root / scene_id / "final_predictions.json").exists() for scene_id in scene_ids)


def _shard_command(
    args: argparse.Namespace,
    trial: dict[str, Any],
    attempt_root: Path,
    shard_index: int,
    num_shards: int,
    scene_ids: list[str],
) -> list[str]:
    params = trial["params"]
    cmd = [
        sys.executable,
        "scripts/run_harvest_benchmark_inprocess.py",
        "--pairs-fast-root",
        args.pairs_fast_root,
        "--out-root",
        str(attempt_root),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
        "--frame-stride",
        str(args.frame_stride),
        "--sam-batch-size",
        str(args.sam_batch_size),
        "--rgb-stream-id",
        args.rgb_stream_id,
        "--sam-score-thres",
        str(params["sam_score_thres"]),
        "--min-mask-area-px",
        str(params["min_mask_area_px"]),
        "--max-mask-area-frac",
        str(params["max_mask_area_frac"]),
        "--min-visible-post-views",
        str(params["min_visible_post_views"]),
        "--disappeared-score-thres",
        str(params["disappeared_score_thres"]),
        "--match-radius-m",
        str(params["match_radius_m"]),
        "--change-score-mode",
        str(params["change_score_mode"]),
    ]
    cmd.append("--use-aux-prompts" if trial["use_aux_prompts"] else "--no-aux-prompts")
    if args.force:
        cmd.append("--force")
    for scene_id in scene_ids:
        cmd.extend(["--scene-id", scene_id])
    return cmd


def _trial_env(trial: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["SAM_ARIA_HARVESTING_PRIMARY_PROMPTS"] = trial["primary_prompts"]
    env["SAM_ARIA_HARVESTING_AUXILIARY_PROMPTS"] = trial["auxiliary_prompts"]
    env.setdefault("SAM3_NATIVE_AMP_DTYPE", "bfloat16")
    env.setdefault("SAM3_DISABLE_ACT_CKPT", "1")
    return env


def _run_trial(
    args: argparse.Namespace,
    trial: dict[str, Any],
    attempt_root: Path,
    scene_ids: list[str],
) -> int:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("No devices provided.")

    env_base = _trial_env(trial)
    running: list[dict[str, Any]] = []
    for shard_index, device in enumerate(devices):
        env = dict(env_base)
        env["SAM3_DEVICE"] = f"cuda:{device}"
        log_path = attempt_root / f"tune_shard_{shard_index}_cuda{device}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = _shard_command(args, trial, attempt_root, shard_index, len(devices), scene_ids)
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(" ".join(cmd) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        running.append(
            {
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "device": device,
                "started": time.time(),
            }
        )
        print(f"  start shard={shard_index} cuda:{device} log={log_path}", flush=True)

    return_code = 0
    while running:
        time.sleep(args.poll_s)
        for item in list(running):
            code = item["process"].poll()
            if code is None:
                continue
            elapsed_min = (time.time() - item["started"]) / 60.0
            item["log_handle"].close()
            print(
                f"  done cuda:{item['device']} code={code} elapsed_min={elapsed_min:.1f}",
                flush=True,
            )
            if code != 0:
                return_code = code
                print(f"    log: {item['log_path']}", flush=True)
            running.remove(item)
    return return_code


def _evaluate_subset(
    args: argparse.Namespace,
    pred_root: Path,
    eval_root: Path,
    scene_ids: list[str],
) -> dict[str, Any]:
    gt_all = parse_gt_label_file(args.gt_labels, "harvesting")
    pred_all, warnings = load_prediction_points(pred_root, "harvesting", known_scene_ids=set(gt_all))
    gt = {scene_id: gt_all.get(scene_id, []) for scene_id in scene_ids}
    pred = {scene_id: pred_all.get(scene_id, []) for scene_id in scene_ids}
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
            "task": "harvesting",
            "tau_m": args.tau_m,
            "scene_ids": scene_ids,
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
        "best": ranked[0],
        "all_ranked": ranked,
    }
    (out_root / "best_trial.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 10 harvesting tuning trials and report the best cut-point parameters."
    )
    parser.add_argument("--out-root", default="/tmp/sam_harvest_tuning_10")
    parser.add_argument(
        "--pairs-fast-root",
        default="/path/to/data/mps/egodelta_benchmark_data/pairs_fast",
    )
    parser.add_argument(
        "--gt-labels",
        default="/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt",
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sam-batch-size", type=int, default=16)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--poll-s", type=float, default=120.0)
    parser.add_argument("--tau-m", type=float, default=0.03)
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--start-at", type=int, default=1, help="1-indexed first trial to run.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--limit-scenes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    scene_ids = args.scene_id or _default_scene_ids(args)
    if not scene_ids:
        raise SystemExit("No harvesting scenes found. Check --gt-labels and --pairs-fast-root.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "tuning_summary.json"
    trials_path = out_root / "trial_configs.json"
    scenes_path = out_root / "scene_ids.json"
    trials = TRIALS[args.start_at - 1 : args.max_trials]
    trials_path.write_text(json.dumps(TRIALS[: args.max_trials], indent=2) + "\n")
    scenes_path.write_text(json.dumps(scene_ids, indent=2) + "\n")

    summary: list[dict[str, Any]] = []
    if summary_path.exists() and not args.force:
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = []
    existing_by_attempt = {row.get("attempt"): row for row in summary}

    print(f"Harvesting scenes: {len(scene_ids)}", flush=True)
    print(f"Trials: {len(trials)}", flush=True)
    for trial_idx, trial in enumerate(trials, start=args.start_at):
        attempt = f"trial_{trial_idx:02d}_{trial['name']}"
        attempt_root = out_root / attempt
        eval_root = out_root / f"{attempt}_eval_{threshold_label_cm(args.tau_m)}"

        if not args.force and _attempt_complete(attempt_root, scene_ids):
            print(f"===== {attempt} already complete; evaluating/resuming =====", flush=True)
            aggregate = _evaluate_subset(args, attempt_root, eval_root, scene_ids)
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
            started = time.time()
            print(f"===== {attempt} =====", flush=True)
            print("primary prompts:", trial["primary_prompts"], flush=True)
            print("aux prompts:", trial["auxiliary_prompts"] if trial["use_aux_prompts"] else "disabled", flush=True)
            print("params:", json.dumps(trial["params"], sort_keys=True), flush=True)
            returncode = _run_trial(args, trial, attempt_root, scene_ids)
            elapsed_min = (time.time() - started) / 60.0
            if returncode != 0:
                row = {
                    "attempt": attempt,
                    "status": "failed",
                    "returncode": returncode,
                    "elapsed_min": elapsed_min,
                    "trial": trial,
                }
            else:
                aggregate = _evaluate_subset(args, attempt_root, eval_root, scene_ids)
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
