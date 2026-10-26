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
from sam_aria_change_baseline.task_prompts import TASK_PROMPTS


TASK_PAIR_DIR = {
    "deleafing": "leaf_pair",
    "harvesting": "tomato_pair",
}


def _scene_id_from_manifest(path: Path) -> str:
    return path.parents[1].name


def _scene_ids_for_task(args: argparse.Namespace, task: str) -> list[str]:
    gt_by_scene = parse_gt_label_file(args.gt_labels, task)
    pair_dir = TASK_PAIR_DIR[task]
    manifest_scene_ids = {
        _scene_id_from_manifest(path)
        for path in Path(args.pairs_fast_root).glob(f"*/{pair_dir}/manifest.json")
    }
    scene_ids = sorted(
        scene_id
        for scene_id, points in gt_by_scene.items()
        if points and scene_id in manifest_scene_ids
    )
    selected = set(getattr(args, f"{task}_scene_id") or [])
    if selected:
        scene_ids = [scene_id for scene_id in scene_ids if scene_id in selected]
    if args.limit_scenes is not None:
        scene_ids = scene_ids[: args.limit_scenes]
    return scene_ids


def _manifest_for_scene(args: argparse.Namespace, task: str, scene_id: str) -> Path:
    return Path(args.pairs_fast_root) / scene_id / TASK_PAIR_DIR[task] / "manifest.json"


def _task_out_root(args: argparse.Namespace, task: str) -> Path:
    return Path(args.out_root) / task / "predictions"


def _task_eval_root(args: argparse.Namespace, task: str) -> Path:
    return Path(args.out_root) / task / "evaluation"


def _pipeline_args(
    args: argparse.Namespace,
    task: str,
    manifest_path: Path,
    out_dir: Path,
) -> list[str]:
    if task == "deleafing":
        use_aux = not args.no_deleaf_aux
        batch_size = args.deleaf_sam_batch_size
    elif task == "harvesting":
        use_aux = not args.no_harvest_aux
        batch_size = args.harvest_sam_batch_size
    else:
        raise ValueError(f"Unsupported task {task!r}")

    cmd = [
        args.python,
        "-m",
        "sam_aria_change_baseline.run_pipeline",
        "--task",
        task,
        "--mode",
        "extracted_frames",
        "--pair-manifest",
        str(manifest_path),
        "--rgb-stream-id",
        args.rgb_stream_id,
        "--frame-stride",
        str(args.frame_stride),
        "--sam-batch-size",
        str(batch_size),
        "--sam-backend",
        args.sam_backend,
        "--skip-pipeline-visualizations",
        "--out",
        str(out_dir),
    ]
    if use_aux:
        cmd.append("--use-aux-prompts")

    if task == "deleafing":
        cmd.extend(
            [
                "--deleafing-sam-score-thres",
                str(args.deleafing_sam_score_thres),
                "--deleafing-min-mask-area-px",
                str(args.deleafing_min_mask_area_px),
                "--deleafing-max-mask-area-frac",
                str(args.deleafing_max_mask_area_frac),
                "--deleafing-max-primary-masks-per-frame",
                str(args.deleafing_max_primary_masks_per_frame),
                "--deleafing-pipe-like-long-side-frac",
                str(args.deleafing_pipe_like_long_side_frac),
                "--deleafing-pipe-like-aspect-ratio",
                str(args.deleafing_pipe_like_aspect_ratio),
                "--deleafing-primary-vertical-axis-thres",
                str(args.deleafing_primary_vertical_axis_thres),
                "--deleafing-primary-vertical-axis-max-thinness",
                str(args.deleafing_primary_vertical_axis_max_thinness),
                "--deleafing-primary-vertical-axis-image-direction",
                args.deleafing_primary_vertical_axis_image_direction,
                "--petiole-min-points-inside-mask",
                str(args.petiole_min_points_inside_mask),
                "--petiole-dbscan-min-samples",
                str(args.petiole_dbscan_min_samples),
                "--petiole-instance-min-frames",
                str(args.petiole_instance_min_frames),
                "--petiole-min-3d-major-extent-m",
                str(args.petiole_min_3d_major_extent_m),
                "--petiole-max-3d-major-extent-m",
                str(args.petiole_max_3d_major_extent_m),
                "--deleafing-aux-sam-score-thres",
                str(args.deleafing_aux_sam_score_thres),
                "--deleafing-aux-min-mask-area-px",
                str(args.deleafing_aux_min_mask_area_px),
                "--deleafing-main-stem-min-verticality",
                str(args.deleafing_main_stem_min_verticality),
                "--deleafing-main-stem-min-height-frac",
                str(args.deleafing_main_stem_min_height_frac),
                "--deleafing-main-stem-max-width-frac",
                str(args.deleafing_main_stem_max_width_frac),
                "--deleafing-main-stem-min-aspect-ratio",
                str(args.deleafing_main_stem_min_aspect_ratio),
                "--deleafing-aux-min-vertical-extent-m",
                str(args.deleafing_aux_min_vertical_extent_m),
                "--deleafing-aux-min-vertical-ratio",
                str(args.deleafing_aux_min_vertical_ratio),
                "--deleafing-max-aux-masks-per-frame",
                str(args.deleafing_max_aux_masks_per_frame),
                "--disappeared-score-thres",
                str(args.deleafing_disappeared_score_thres),
                "--match-radius-m",
                str(args.deleafing_match_radius_m),
            ]
        )
    else:
        cmd.extend(
            [
                "--sam-score-thres",
                str(args.harvest_sam_score_thres),
                "--min-mask-area-px",
                str(args.harvest_min_mask_area_px),
                "--max-mask-area-frac",
                str(args.harvest_max_mask_area_frac),
                "--min-visible-post-views",
                str(args.harvest_min_visible_post_views),
                "--disappeared-score-thres",
                str(args.harvest_disappeared_score_thres),
                "--match-radius-m",
                str(args.harvest_match_radius_m),
                "--change-score-mode",
                args.harvest_change_score_mode,
            ]
        )
    return cmd


def _child_env(args: argparse.Namespace, task: str, device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["SAM3_DEVICE"] = f"cuda:{device}"
    env.setdefault("SAM3_NATIVE_AMP_DTYPE", "bfloat16")
    env.setdefault("SAM3_DISABLE_ACT_CKPT", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not args.inherit_prompt_env:
        prompts = TASK_PROMPTS[task]
        env[f"SAM_ARIA_{task.upper()}_PRIMARY_PROMPTS"] = "||".join(prompts["primary"])
        env[f"SAM_ARIA_{task.upper()}_AUXILIARY_PROMPTS"] = "||".join(prompts["auxiliary"])
    return env


def _run_task(args: argparse.Namespace, task: str, scene_ids: list[str]) -> dict[str, Any]:
    out_root = _task_out_root(args, task)
    out_root.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("No devices provided.")

    queue: list[tuple[str, Path, Path]] = []
    for scene_id in scene_ids:
        manifest_path = _manifest_for_scene(args, task, scene_id)
        out_dir = out_root / scene_id
        if out_dir.joinpath("final_predictions.json").exists() and not args.force:
            print(f"[{task}] skip complete {scene_id}", flush=True)
            continue
        if not manifest_path.exists():
            print(f"[{task}] missing manifest for {scene_id}: {manifest_path}", flush=True)
            continue
        queue.append((scene_id, manifest_path, out_dir))

    running: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    completed: list[str] = []
    while queue or running:
        for device in devices:
            if device in running or not queue:
                continue
            scene_id, manifest_path, out_dir = queue.pop(0)
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "full_eval_run.log"
            cmd = _pipeline_args(args, task, manifest_path, out_dir)
            env = _child_env(args, task, device)
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write(" ".join(cmd) + "\n\n")
            log_handle.flush()
            process = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running[device] = {
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "scene_id": scene_id,
                "started": time.time(),
            }
            print(f"[{task}] start {scene_id} on cuda:{device}", flush=True)

        time.sleep(args.poll_s)
        for device, item in list(running.items()):
            process = item["process"]
            code = process.poll()
            if code is None:
                continue
            item["log_handle"].close()
            elapsed_min = (time.time() - item["started"]) / 60.0
            scene_id = item["scene_id"]
            print(
                f"[{task}] done {scene_id} on cuda:{device}: code={code} "
                f"elapsed_min={elapsed_min:.1f}",
                flush=True,
            )
            if code != 0:
                failures.append(
                    {
                        "scene_id": scene_id,
                        "returncode": int(code),
                        "log_path": str(item["log_path"]),
                    }
                )
            else:
                completed.append(scene_id)
            del running[device]

    return {
        "task": task,
        "num_selected_scenes": len(scene_ids),
        "num_completed_now": len(completed),
        "failures": failures,
        "prediction_root": str(out_root),
    }


def _evaluate_task(
    args: argparse.Namespace,
    task: str,
    scene_ids: list[str],
) -> dict[str, Any]:
    pred_root = _task_out_root(args, task)
    gt_all = parse_gt_label_file(args.gt_labels, task)
    pred_all, warnings = load_prediction_points(pred_root, task, known_scene_ids=set(gt_all))
    gt = {scene_id: gt_all.get(scene_id, []) for scene_id in scene_ids}
    pred = {scene_id: pred_all.get(scene_id, []) for scene_id in scene_ids}
    task_summary: dict[str, Any] = {
        "task": task,
        "scene_ids": scene_ids,
        "prediction_root": str(pred_root),
        "warnings": warnings,
        "thresholds": {},
    }
    for tau_m in args.eval_tau_m:
        per_scene, aggregate = evaluate_cut_point_sets(
            gt,
            pred,
            tau_m=float(tau_m),
            include_prediction_only_scenes=False,
        )
        label = threshold_label_cm(float(tau_m))
        eval_root = _task_eval_root(args, task) / label
        save_evaluation_results(
            per_scene,
            aggregate,
            eval_root,
            metadata={
                "task": task,
                "tau_m": float(tau_m),
                "scene_ids": scene_ids,
                "pred_root": str(pred_root),
                "gt_labels": args.gt_labels,
                "warnings": warnings,
            },
        )
        task_summary["thresholds"][label] = aggregate
        print(
            f"[{task}] {label}: TP={aggregate['total_TP']} "
            f"Pred={aggregate['total_num_pred']} GT={aggregate['total_num_gt']} "
            f"F1={aggregate[f'micro_F1@{label}']:.4f}",
            flush=True,
        )
    return task_summary


def _task_order(value: str) -> list[str]:
    if value == "both":
        return ["deleafing", "harvesting"]
    return [value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full 30-scene deleafing and harvesting evaluation on pairs_fast."
    )
    parser.add_argument("--task", choices=("both", "deleafing", "harvesting"), default="both")
    parser.add_argument("--out-root", default="/tmp/plant_sam_full_eval_current")
    parser.add_argument(
        "--pairs-fast-root",
        default="/path/to/data/mps/egodelta_benchmark_data/pairs_fast",
    )
    parser.add_argument(
        "--gt-labels",
        default="/path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sam-backend", choices=("native_sam3", "ultralytics_sam3"), default="native_sam3")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--deleaf-sam-batch-size", type=int, default=16)
    parser.add_argument("--harvest-sam-batch-size", type=int, default=1)
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--deleafing-scene-id", action="append")
    parser.add_argument("--harvesting-scene-id", action="append")
    parser.add_argument(
        "--eval-tau-m",
        type=float,
        action="append",
        default=None,
        help="Evaluation threshold in meters. Repeat to evaluate multiple thresholds; defaults to 0.03.",
    )
    parser.add_argument("--inherit-prompt-env", action="store_true")

    parser.add_argument("--no-deleaf-aux", action="store_true")
    parser.add_argument("--deleafing-sam-score-thres", type=float, default=0.16)
    parser.add_argument("--deleafing-min-mask-area-px", type=int, default=50)
    parser.add_argument("--deleafing-max-mask-area-frac", type=float, default=0.03)
    parser.add_argument("--deleafing-max-primary-masks-per-frame", type=int, default=12)
    parser.add_argument("--deleafing-pipe-like-long-side-frac", type=float, default=0.35)
    parser.add_argument("--deleafing-pipe-like-aspect-ratio", type=float, default=6.0)
    parser.add_argument("--deleafing-primary-vertical-axis-thres", type=float, default=0.90)
    parser.add_argument("--deleafing-primary-vertical-axis-max-thinness", type=float, default=0.45)
    parser.add_argument("--deleafing-primary-vertical-axis-image-direction", default="image_horizontal")
    parser.add_argument("--petiole-min-points-inside-mask", type=int, default=4)
    parser.add_argument("--petiole-dbscan-min-samples", type=int, default=3)
    parser.add_argument("--petiole-instance-min-frames", type=int, default=1)
    parser.add_argument("--petiole-min-3d-major-extent-m", type=float, default=0.04)
    parser.add_argument("--petiole-max-3d-major-extent-m", type=float, default=0.55)
    parser.add_argument("--deleafing-aux-sam-score-thres", type=float, default=0.35)
    parser.add_argument("--deleafing-aux-min-mask-area-px", type=int, default=80)
    parser.add_argument("--deleafing-main-stem-min-verticality", type=float, default=0.82)
    parser.add_argument("--deleafing-main-stem-min-height-frac", type=float, default=0.18)
    parser.add_argument("--deleafing-main-stem-max-width-frac", type=float, default=0.22)
    parser.add_argument("--deleafing-main-stem-min-aspect-ratio", type=float, default=1.8)
    parser.add_argument("--deleafing-aux-min-vertical-extent-m", type=float, default=0.08)
    parser.add_argument("--deleafing-aux-min-vertical-ratio", type=float, default=1.25)
    parser.add_argument("--deleafing-max-aux-masks-per-frame", type=int, default=8)
    parser.add_argument("--deleafing-disappeared-score-thres", type=float, default=0.0)
    parser.add_argument("--deleafing-match-radius-m", type=float, default=0.02)

    parser.add_argument("--no-harvest-aux", action="store_true")
    parser.add_argument("--harvest-sam-score-thres", type=float, default=0.25)
    parser.add_argument("--harvest-min-mask-area-px", type=int, default=100)
    parser.add_argument("--harvest-max-mask-area-frac", type=float, default=0.40)
    parser.add_argument("--harvest-min-visible-post-views", type=int, default=3)
    parser.add_argument("--harvest-disappeared-score-thres", type=float, default=0.60)
    parser.add_argument("--harvest-match-radius-m", type=float, default=0.12)
    parser.add_argument(
        "--harvest-change-score-mode",
        choices=("auto", "average", "sam_absence_only", "point_disappearance_only"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.eval_tau_m is None:
        args.eval_tau_m = [0.03]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, Any] = {
        "out_root": str(out_root),
        "pairs_fast_root": args.pairs_fast_root,
        "gt_labels": args.gt_labels,
        "tasks": {},
    }
    return_code = 0
    for task in _task_order(args.task):
        scene_ids = _scene_ids_for_task(args, task)
        scenes_path = out_root / task / "scene_ids.json"
        scenes_path.parent.mkdir(parents=True, exist_ok=True)
        scenes_path.write_text(json.dumps(scene_ids, indent=2) + "\n")
        print(f"[{task}] selected scenes: {len(scene_ids)}", flush=True)
        run_summary = None
        if not args.eval_only:
            run_summary = _run_task(args, task, scene_ids)
            if run_summary["failures"]:
                return_code = 1
        eval_summary = _evaluate_task(args, task, scene_ids)
        all_summary["tasks"][task] = {
            "scene_ids_path": str(scenes_path),
            "run": run_summary,
            "evaluation": eval_summary,
        }
    summary_path = out_root / "full_eval_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2) + "\n")
    print(f"Full evaluation summary: {summary_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
