from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tomato_cutpoints.config import deep_update, load_config, save_config
from tomato_cutpoints.eval.metrics import aggregate_cut_point_metrics, evaluate_cut_points, threshold_label_cm
from tomato_cutpoints.pipeline import run_pipeline
from tomato_cutpoints.registration.transforms import apply_transform
from tools.evaluate_cutpoints import (
    TASK_SPECS,
    estimate_full_to_multi_transform,
    input_paths,
    jsonable,
    parse_labels,
    parse_task_names,
    read_predictions,
    write_gt_points,
)


DEFAULT_UUIDS = [
    "35aa89c0-e852-4750-9608-3f79788f7ee9",
    "3b0e7a93-8e13-443f-b729-80e0926dcc42",
    "44fe03cd-d0a7-408a-8ca2-b9b302c912a0",
    "47d0e394-ae62-49e1-bcd6-2f9f236972d0",
    "4d5ebec9-3ebd-4c65-b01d-8d2a0c668df2",
    "5479d13e-b1ed-4539-aeb4-c42ce20dbf8f",
    "5d9ae0cf-5e76-43c8-af38-24547c88e5e5",
    "5ede33ac-5cf2-4a8d-88cc-ef38f9b1e5ff",
]

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _choice(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def _uniform(rng: np.random.Generator, low: float, high: float, ndigits: int = 4) -> float:
    return round(float(rng.uniform(low, high)), ndigits)


def sample_overrides(task: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample only existing detector parameters; no algorithmic switches are introduced."""
    nn_weight = _uniform(rng, 0.35, 0.75, 3)
    common: dict[str, Any] = {
        "quality": {
            "max_dist_std": _choice(rng, [0.01, 0.02, 0.03, 0.05]),
            "min_observations": int(_choice(rng, [1, 2, 3])),
        },
        "nn": {
            "threshold_m": _uniform(rng, 0.015, 0.04, 4),
            "auto_threshold": True,
            "auto_threshold_k_mad": _uniform(rng, 2.0, 6.0, 2),
            "stable_percentile": int(_choice(rng, [90, 92, 95, 97, 99])),
        },
        "m3c2": {
            "enabled": True,
            "cyl_radius": _choice(rng, [0.02, 0.03, 0.04, 0.05]),
            "max_distance": _choice(rng, [0.08, 0.10, 0.12, 0.15]),
            "threshold_m": _uniform(rng, 0.015, 0.05, 4),
            "use_lodetection": True,
            "min_samples1": int(_choice(rng, [2, 3, 4, 5])),
            "min_samples2": int(_choice(rng, [1, 2, 3])),
        },
        "scoring": {
            "nn_weight": nn_weight,
            "m3c2_weight": round(1.0 - nn_weight, 3),
            "density_weight": 0.0,
            "fusion": "weighted_average",
            "min_combined_score": _uniform(rng, 0.25, 0.75, 3),
        },
    }

    if task == "harvest":
        common["dbscan"] = {
            "eps_m": _choice(rng, [0.04, 0.05, 0.06, 0.08, 0.10, 0.12]),
            "min_samples": int(_choice(rng, [2, 3, 4, 5, 8])),
            "min_cluster_points": int(_choice(rng, [8, 10, 15, 20, 30, 40])),
            "max_cluster_points": 200000,
        }
        common["task_priors"] = {
            "harvest": {
                "prefer_compact": True,
                "min_diameter_m": _uniform(rng, 0.012, 0.035, 4),
                "max_diameter_m": _uniform(rng, 0.12, 0.25, 4),
                "min_points": int(_choice(rng, [10, 20, 30, 40, 60])),
                "point_count_weight": _uniform(rng, 0.10, 0.60, 3),
                "point_count_saturation": int(_choice(rng, [80, 120, 160, 200, 260, 320])),
                "max_targets": int(_choice(rng, [5, 8, 10, 12])),
                "min_target_score": _uniform(rng, 0.60, 1.50, 3),
                "split_merged": True,
                "split_trigger_diameter_m": _uniform(rng, 0.10, 0.20, 4),
                "split_max_parts": int(_choice(rng, [2, 3, 4])),
                "split_min_points": int(_choice(rng, [15, 20, 30, 40])),
                "split_min_center_distance_m": _uniform(rng, 0.035, 0.08, 4),
                "split_min_silhouette": _uniform(rng, 0.20, 0.50, 3),
                "split_max_subcluster_diameter_m": _uniform(rng, 0.10, 0.20, 4),
                "max_scene_robust_distance": _uniform(rng, 2.0, 4.5, 2),
            }
        }
    elif task == "deleaf":
        common["dbscan"] = {
            "eps_m": _choice(rng, [0.02, 0.025, 0.03, 0.04, 0.05, 0.06]),
            "min_samples": int(_choice(rng, [3, 5, 8, 10, 12])),
            "min_cluster_points": int(_choice(rng, [8, 12, 20, 30, 45, 60])),
            "max_cluster_points": 200000,
        }
        common["task_priors"] = {
            "deleaf": {
                "prefer_elongated": True,
                "min_length_m": _uniform(rng, 0.02, 0.12, 4),
                "max_targets": int(_choice(rng, [5, 8, 10, 12, 15])),
                "min_target_score": _uniform(rng, 0.50, 1.50, 3),
            }
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
    return common


def build_trial_config(task: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(TASK_SPECS[task]["config"])
    deep_update(cfg, deepcopy(overrides))
    cfg["task"] = task
    return cfg


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_progress_outputs(
    base_dir: Path,
    trial_rows: list[dict[str, Any]],
    per_scene_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    trial_docs: list[dict[str, Any]],
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    write_csv(base_dir / "trial_summary.csv", trial_rows)
    write_csv(base_dir / "per_scene_summary.csv", per_scene_rows)
    write_csv(base_dir / "failures.csv", failure_rows)
    (base_dir / "trials.json").write_text(
        json.dumps(jsonable(trial_docs), indent=2) + "\n",
        encoding="utf-8",
    )


def load_or_estimate_transform(
    cache_dir: Path,
    full_mps_root: Path,
    multi_slam_root: Path,
    uuid: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = cache_dir / f"{uuid}.json"
    if cache_path.exists():
        doc = json.loads(cache_path.read_text(encoding="utf-8"))
        return np.asarray(doc["T_target_from_source"], dtype=np.float64), doc
    transform, info = estimate_full_to_multi_transform(full_mps_root, multi_slam_root, uuid)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(jsonable(info), indent=2) + "\n", encoding="utf-8")
    return transform, info


def parse_uuid_list(value: str | None) -> list[str]:
    if value is None:
        return DEFAULT_UUIDS.copy()
    uuids = UUID_RE.findall(value)
    if not uuids:
        raise argparse.ArgumentTypeError("UUID list cannot be empty.")
    return uuids


def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    f1 = float(row.get("micro F1") or 0.0)
    gt_nearest = row.get("gt_to_nearest_mean_distance_cm")
    matched = row.get("matched_mean_distance_cm")
    return (
        f1,
        -float(gt_nearest) if gt_nearest is not None else -1.0e9,
        -float(matched) if matched is not None else -1.0e9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-search tuning for existing cutpoint detector parameters.")
    parser.add_argument("--labels", type=Path, default=Path("3d points labels.txt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark_splitted"))
    parser.add_argument("--full-mps-root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark"))
    parser.add_argument("--multi-slam-root", type=Path, default=Path("/path/to/data/mps/mps_multi_slam_split"))
    parser.add_argument("--out-dir", type=Path, default=Path("/path/to/data/plant_point/tuning"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Fast local directory for heavy per-trial outputs. Defaults to /tmp/<out-dir-name>_work.",
    )
    parser.add_argument("--trials", type=int, default=15, help="Number of independent random-search trials per task.")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threshold-m", type=float, default=0.03)
    parser.add_argument(
        "--tasks",
        type=parse_task_names,
        default=list(TASK_SPECS),
        help="Comma-separated tasks to tune: deleaf, harvest, or deleaf,harvest.",
    )
    parser.add_argument(
        "--uuids",
        type=parse_uuid_list,
        default=DEFAULT_UUIDS.copy(),
        help="Comma-separated UUID pool. Defaults to the 8 requested tuning scenes.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse a trial task run if its targets.json already exists.")
    parser.add_argument("--allow-failures", action="store_true", help="Score failed scene-task runs as empty predictions.")
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    missing_labels = [uuid for uuid in args.uuids if uuid not in labels]
    if missing_labels:
        raise ValueError(
            "Missing UUIDs in label file "
            f"{args.labels}: {', '.join(missing_labels)}. "
            "Use the label file that contains the requested tuning set."
        )

    rng = np.random.default_rng(args.seed)
    work_dir = args.work_dir or Path("/tmp") / f"{args.out_dir.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    transform_cache = work_dir / "frame_transform_cache"
    threshold_label = threshold_label_cm(args.threshold_m)

    trial_rows: list[dict[str, Any]] = []
    per_scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    trial_docs: list[dict[str, Any]] = []

    for task in args.tasks:
        spec = TASK_SPECS[task]
        for trial_idx in range(args.trials):
            trial_id = f"{task}_trial_{trial_idx:03d}"
            trial_dir = work_dir / task / f"trial_{trial_idx:03d}"
            trial_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max)))
            sampled_uuids = list(
                trial_rng.choice(args.uuids, size=min(args.sample_size, len(args.uuids)), replace=False)
            )
            sampled_uuids = [str(uuid) for uuid in sampled_uuids]

            overrides = sample_overrides(task, trial_rng)
            config = build_trial_config(task, overrides)
            config_path = trial_dir / "configs" / f"{task}.yaml"
            save_config(config, config_path)

            scene_docs: list[dict[str, Any]] = []
            trial_failures: list[dict[str, Any]] = []
            for uuid in sampled_uuids:
                gt_transform, frame_info = load_or_estimate_transform(
                    transform_cache,
                    args.full_mps_root,
                    args.multi_slam_root,
                    uuid,
                )
                run_dir = trial_dir / "runs" / uuid / task
                targets_path = run_dir / "targets.json"
                gt_original = np.asarray(labels[uuid][spec["label_key"]], dtype=np.float64).reshape(-1, 3)
                gt = apply_transform(gt_original, gt_transform)
                write_gt_points(
                    run_dir,
                    gt_original,
                    gt,
                    {
                        "labels_frame": "full-vrs",
                        "prediction_frame": "multi-slam",
                        **frame_info,
                    },
                )

                try:
                    if not args.resume or not targets_path.exists():
                        pre_points, pre_obs = input_paths(
                            args.dataset_root,
                            args.multi_slam_root,
                            "multi-slam",
                            uuid,
                            spec,
                            "pre",
                        )
                        post_points, post_obs = input_paths(
                            args.dataset_root,
                            args.multi_slam_root,
                            "multi-slam",
                            uuid,
                            spec,
                            "post",
                        )
                        for path in [pre_points, pre_obs, post_points, post_obs]:
                            if not path.exists():
                                raise FileNotFoundError(path)
                        run_pipeline(
                            task=task,
                            pre_points=pre_points,
                            post_points=post_points,
                            pre_observations=pre_obs,
                            post_observations=post_obs,
                            out_dir=run_dir,
                            config_path=config_path,
                            refine_icp=False,
                        )
                    pred = read_predictions(targets_path)
                    metrics = evaluate_cut_points(pred, gt, threshold_m=args.threshold_m)
                except Exception as exc:
                    if not args.allow_failures:
                        raise
                    metrics = evaluate_cut_points(np.empty((0, 3), dtype=np.float64), gt, threshold_m=args.threshold_m)
                    failure = {
                        "trial": trial_id,
                        "uuid": uuid,
                        "task": task,
                        "error": repr(exc),
                        "run_dir": str(run_dir),
                    }
                    trial_failures.append(failure)
                    failure_rows.append(failure)

                scene_doc = {"trial": trial_id, "uuid": uuid, "task": task, **metrics}
                scene_docs.append(scene_doc)
                per_scene_rows.append(
                    {
                        "trial": trial_id,
                        "uuid": uuid,
                        "task": task,
                        "num_gt": metrics["num_gt"],
                        "num_pred": metrics["num_pred"],
                        "TP": metrics["TP"],
                        "FP": metrics["FP"],
                        "FN": metrics["FN"],
                        f"Precision@{threshold_label}": metrics[f"Precision@{threshold_label}"],
                        f"Recall@{threshold_label}": metrics[f"Recall@{threshold_label}"],
                        f"F1@{threshold_label}": metrics[f"F1@{threshold_label}"],
                        "matched_mean_distance_cm": metrics["matched_mean_distance_cm"],
                        "gt_to_nearest_mean_distance_cm": metrics["gt_to_nearest_mean_distance_cm"],
                        "run_dir": str(run_dir),
                    }
                )

            aggregate = aggregate_cut_point_metrics(scene_docs, threshold_m=args.threshold_m)
            trial_row = {
                "trial": trial_id,
                "task": task,
                "sample_size": len(sampled_uuids),
                "num_scenes": aggregate["num_scenes"],
                "num_gt": aggregate["num_gt"],
                "num_pred": aggregate["num_pred"],
                "TP": aggregate["TP"],
                "FP": aggregate["FP"],
                "FN": aggregate["FN"],
                f"micro Precision@{threshold_label}": aggregate[f"micro Precision@{threshold_label}"],
                f"micro Recall@{threshold_label}": aggregate[f"micro Recall@{threshold_label}"],
                f"micro F1@{threshold_label}": aggregate[f"micro F1@{threshold_label}"],
                "micro F1": aggregate["micro F1"],
                "matched_mean_distance_cm": aggregate["matched_mean_distance_cm"],
                "gt_to_nearest_mean_distance_cm": aggregate["gt_to_nearest_mean_distance_cm"],
                "n_failures": len(trial_failures),
                "sampled_uuids": ",".join(sampled_uuids),
                "config_path": str(config_path),
                "trial_dir": str(trial_dir),
            }
            trial_rows.append(trial_row)
            trial_docs.append(
                {
                    "trial": trial_id,
                    "task": task,
                    "sampled_uuids": sampled_uuids,
                    "aggregate": aggregate,
                    "overrides": {task: overrides},
                    "config_paths": {task: str(config_path)},
                    "failures": trial_failures,
                }
            )
            print(
                f"{trial_id}: F1@{threshold_label}={aggregate[f'micro F1@{threshold_label}']:.4f} "
                f"P={aggregate[f'micro Precision@{threshold_label}']:.4f} "
                f"R={aggregate[f'micro Recall@{threshold_label}']:.4f} "
                f"gt_nearest_cm={aggregate['gt_to_nearest_mean_distance_cm']}",
                flush=True,
            )

            write_progress_outputs(work_dir, trial_rows, per_scene_rows, failure_rows, trial_docs)

    ranked = sorted(trial_rows, key=sort_key, reverse=True)
    write_csv(work_dir / "trial_summary_ranked.csv", ranked)
    for task in args.tasks:
        task_ranked = sorted([row for row in trial_rows if row["task"] == task], key=sort_key, reverse=True)
        write_csv(work_dir / f"trial_summary_ranked_{task}.csv", task_ranked)
        if not task_ranked:
            continue
        best_trial_id = task_ranked[0]["trial"]
        best_doc = next((doc for doc in trial_docs if doc["trial"] == best_trial_id), None)
        if best_doc is None:
            continue
        for best_dir in [work_dir / "best" / task, args.out_dir / "best" / task]:
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "best_trial.json").write_text(
                json.dumps(jsonable(best_doc), indent=2) + "\n",
                encoding="utf-8",
            )
            save_config(build_trial_config(task, best_doc["overrides"][task]), best_dir / f"best_{task}.yaml")
        for best_root in [work_dir / "best", args.out_dir / "best"]:
            best_root.mkdir(parents=True, exist_ok=True)
            save_config(build_trial_config(task, best_doc["overrides"][task]), best_root / f"best_{task}.yaml")
        print(f"\nBest {task} trial:")
        print(json.dumps(jsonable(best_doc["aggregate"]), indent=2))
        print(f"Best {task} config written to {args.out_dir / 'best' / f'best_{task}.yaml'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_progress_outputs(args.out_dir, trial_rows, per_scene_rows, failure_rows, trial_docs)
    write_csv(args.out_dir / "trial_summary_ranked.csv", ranked)
    for task in args.tasks:
        task_ranked = sorted([row for row in trial_rows if row["task"] == task], key=sort_key, reverse=True)
        write_csv(args.out_dir / f"trial_summary_ranked_{task}.csv", task_ranked)
    run_info = {
        "out_dir": args.out_dir,
        "work_dir": work_dir,
        "heavy_intermediate_outputs": str(work_dir),
        "final_summary_outputs": str(args.out_dir),
        "trials_per_task": args.trials,
        "total_trials": args.trials * len(args.tasks),
        "sample_size": args.sample_size,
        "threshold_m": args.threshold_m,
        "tasks": args.tasks,
        "uuids": args.uuids,
    }
    for base_dir in [work_dir, args.out_dir]:
        (base_dir / "run_info.json").write_text(json.dumps(jsonable(run_info), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
