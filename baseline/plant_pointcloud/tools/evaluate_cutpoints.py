from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from tomato_cutpoints.eval.metrics import aggregate_cut_point_metrics, evaluate_cut_points, threshold_label_cm
from tomato_cutpoints.pipeline import run_pipeline
from tomato_cutpoints.registration.transforms import apply_transform, best_fit_transform, identity_transform


TASK_SPECS = {
    "deleaf": {
        "label_key": "deleafing",
        "pre_stage": "pre_deleafing",
        "pre_mps": "mps_pre_deleafing_vrs",
        "post_stage": "post_deleafing_pre_harvesting",
        "post_mps": "mps_post_deleafing_pre_harvesting_vrs",
        "config": "configs/deleaf.yaml",
    },
    "harvest": {
        "label_key": "harvesting",
        "pre_stage": "post_deleafing_pre_harvesting",
        "pre_mps": "mps_post_deleafing_pre_harvesting_vrs",
        "post_stage": "post_harvesting",
        "post_mps": "mps_post_harvesting_vrs",
        "config": "configs/harvest.yaml",
    },
}


def parse_task_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("At least one task must be specified.")
    unknown = [name for name in names if name not in TASK_SPECS]
    if unknown:
        valid = ", ".join(TASK_SPECS)
        raise argparse.ArgumentTypeError(f"Unknown task(s): {', '.join(unknown)}. Valid tasks: {valid}")
    return names

UUID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.vrs")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_labels(path: Path) -> dict[str, dict[str, list[list[float]]]]:
    labels: dict[str, dict[str, list[list[float]]]] = {}
    current_uuid: str | None = None
    current_task: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        uuid_match = UUID_RE.search(line)
        if uuid_match:
            current_uuid = uuid_match.group(1)
            labels[current_uuid] = {"deleafing": [], "harvesting": []}
            current_task = None
            continue
        lowered = line.lower()
        if lowered in {"deleafing", "harvesting"}:
            current_task = lowered
            continue
        if current_uuid is None or current_task is None:
            continue
        if "[" not in line:
            continue
        values = [float(v) for v in FLOAT_RE.findall(line)]
        if len(values) >= 3:
            labels[current_uuid][current_task].append(values[:3])
    return labels


def slam_paths(dataset_root: Path, uuid: str, spec: dict[str, str], prefix: str) -> tuple[Path, Path]:
    stage = spec[f"{prefix}_stage"]
    mps = spec[f"{prefix}_mps"]
    slam = dataset_root / uuid / stage / mps / "slam"
    return slam / "semidense_points.csv.gz", slam / "semidense_observations.csv.gz"


def multi_slam_paths(multi_slam_root: Path, uuid: str, stage: str) -> tuple[Path, Path]:
    run_root = multi_slam_root / uuid
    mapping_path = run_root / "vrs_to_multi_slam.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing official Multi-SLAM mapping: {mapping_path}. "
            f"Run `aria_mps multi -i <split-uuid-dir> -o {run_root} -n <name>` first."
        )

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    stage_token = f"/{stage}/"
    folder_id: str | None = None
    for vrs_path, output_id in mapping.items():
        normalized = str(vrs_path).replace("\\", "/")
        if stage_token in normalized or normalized.endswith(f"/{stage}.vrs"):
            folder_id = str(output_id)
            break
    if folder_id is None:
        raise KeyError(f"Could not find split stage {stage!r} in {mapping_path}")

    slam = run_root / folder_id / "slam"
    return slam / "semidense_points.csv.gz", slam / "semidense_observations.csv.gz"


def full_mps_slam_dir(full_mps_root: Path, uuid: str) -> Path:
    slam = full_mps_root / f"mps_{uuid}_vrs" / "slam"
    if not slam.exists():
        raise FileNotFoundError(f"Missing full-VRS MPS SLAM directory: {slam}")
    return slam


def input_paths(
    dataset_root: Path,
    multi_slam_root: Path,
    mps_source: str,
    uuid: str,
    spec: dict[str, str],
    prefix: str,
) -> tuple[Path, Path]:
    if mps_source == "split":
        return slam_paths(dataset_root, uuid, spec, prefix)
    if mps_source == "multi-slam":
        return multi_slam_paths(multi_slam_root, uuid, spec[f"{prefix}_stage"])
    raise ValueError(f"Unknown mps_source: {mps_source}")


def read_predictions(targets_path: Path) -> np.ndarray:
    if not targets_path.exists():
        return np.empty((0, 3), dtype=np.float64)
    doc = json.loads(targets_path.read_text(encoding="utf-8"))
    points = []
    for target in doc.get("targets", []):
        cut = target.get("cut_point_world")
        if cut is not None and len(cut) == 3:
            points.append([float(v) for v in cut])
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def read_trajectory_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    positions: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trajectory {path} is missing columns: {sorted(missing)}")
        for row in reader:
            timestamps.append(int(row["tracking_timestamp_us"]))
            positions.append(
                [
                    float(row["tx_world_device"]),
                    float(row["ty_world_device"]),
                    float(row["tz_world_device"]),
                ]
            )
    return np.asarray(timestamps, dtype=np.int64), np.asarray(positions, dtype=np.float64).reshape(-1, 3)


def match_trajectory_positions(
    full_timestamps: np.ndarray,
    full_positions: np.ndarray,
    other_timestamps: np.ndarray,
    other_positions: np.ndarray,
    max_time_delta_us: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(full_timestamps)
    full_timestamps = full_timestamps[order]
    full_positions = full_positions[order]

    matched_full: list[np.ndarray] = []
    matched_other: list[np.ndarray] = []
    deltas: list[int] = []
    for ts, pos in zip(other_timestamps, other_positions):
        insert_at = int(np.searchsorted(full_timestamps, ts))
        candidates = []
        if insert_at < len(full_timestamps):
            candidates.append(insert_at)
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: abs(int(full_timestamps[idx]) - int(ts)))
        delta = abs(int(full_timestamps[best_idx]) - int(ts))
        if delta <= max_time_delta_us:
            matched_full.append(full_positions[best_idx])
            matched_other.append(pos)
            deltas.append(delta)

    if not matched_full:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(matched_full, dtype=np.float64),
        np.asarray(matched_other, dtype=np.float64),
        np.asarray(deltas, dtype=np.int64),
    )


def multi_slam_trajectory_paths(multi_slam_root: Path, uuid: str) -> list[Path]:
    run_root = multi_slam_root / uuid
    mapping_path = run_root / "vrs_to_multi_slam.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing official Multi-SLAM mapping: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    folder_ids = sorted({str(output_id) for output_id in mapping.values()})
    paths = [run_root / folder_id / "slam" / "closed_loop_trajectory.csv" for folder_id in folder_ids]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def estimate_full_to_multi_transform(
    full_mps_root: Path,
    multi_slam_root: Path,
    uuid: str,
    max_pairs: int = 50000,
) -> tuple[np.ndarray, dict[str, Any]]:
    full_traj = full_mps_slam_dir(full_mps_root, uuid) / "closed_loop_trajectory.csv"
    full_ts, full_pos = read_trajectory_positions(full_traj)

    all_full: list[np.ndarray] = []
    all_multi: list[np.ndarray] = []
    all_deltas: list[np.ndarray] = []
    for multi_traj in multi_slam_trajectory_paths(multi_slam_root, uuid):
        multi_ts, multi_pos = read_trajectory_positions(multi_traj)
        matched_full, matched_multi, deltas = match_trajectory_positions(full_ts, full_pos, multi_ts, multi_pos)
        if len(matched_full):
            all_full.append(matched_full)
            all_multi.append(matched_multi)
            all_deltas.append(deltas)

    if not all_full:
        raise RuntimeError(f"No timestamp-matched trajectory poses for {uuid}")

    source = np.vstack(all_full)
    target = np.vstack(all_multi)
    deltas = np.concatenate(all_deltas)
    if len(source) > max_pairs:
        indices = np.linspace(0, len(source) - 1, max_pairs, dtype=np.int64)
        source = source[indices]
        target = target[indices]
        deltas = deltas[indices]

    transform = best_fit_transform(source, target)
    residuals_m = np.linalg.norm(apply_transform(source, transform) - target, axis=1)
    info = {
        "method": "timestamp_matched_trajectory_kabsch",
        "source_frame": "full_vrs_mps_slam",
        "target_frame": "multi_slam_common_world",
        "full_trajectory": str(full_traj),
        "multi_trajectories": [str(path) for path in multi_slam_trajectory_paths(multi_slam_root, uuid)],
        "n_pose_matches": int(len(source)),
        "max_time_delta_us": int(np.max(deltas)) if len(deltas) else None,
        "median_time_delta_us": float(np.median(deltas)) if len(deltas) else None,
        "mean_residual_cm": float(np.mean(residuals_m) * 100.0),
        "median_residual_cm": float(np.median(residuals_m) * 100.0),
        "max_residual_cm": float(np.max(residuals_m) * 100.0),
        "T_target_from_source": transform.tolist(),
    }
    return transform, info


def write_gt_points(run_dir: Path, original_gt: np.ndarray, transformed_gt: np.ndarray, frame_info: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "gt_points_original": np.asarray(original_gt, dtype=np.float64).reshape(-1, 3).tolist(),
        "gt_points_prediction_frame": np.asarray(transformed_gt, dtype=np.float64).reshape(-1, 3).tolist(),
        "frame_info": frame_info,
    }
    (run_dir / "gt_points_prediction_frame.json").write_text(
        json.dumps(jsonable(doc), indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cut point predictions against manual MPS labels.")
    parser.add_argument("--labels", type=Path, default=Path("3d points labels.txt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark_splitted"))
    parser.add_argument("--full-mps-root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark"))
    parser.add_argument("--multi-slam-root", type=Path, default=Path("outputs/mps_multi_slam_split"))
    parser.add_argument(
        "--labels-frame",
        choices=["prediction", "full-vrs"],
        default="prediction",
        help="Coordinate frame of manual labels. Use full-vrs for labels made in full VRS MPS SLAM.",
    )
    parser.add_argument(
        "--mps-source",
        choices=["split", "multi-slam"],
        default="split",
        help="Use independent split MPS outputs, or official aria_mps multi outputs in a shared frame.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cutpoint_eval"))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing targets.json files.")
    parser.add_argument("--refine-icp", action="store_true", help="Align each post MPS cloud to the pre MPS cloud with ICP.")
    parser.add_argument("--strict", action="store_true", help="Fail the scene instead of using fallbacks such as NN-only M3C2 failure.")
    parser.add_argument("--threshold-m", type=float, default=0.02, help="Correct cut-point threshold in meters.")
    parser.add_argument(
        "--tasks",
        type=parse_task_names,
        default=list(TASK_SPECS),
        help="Comma-separated tasks to evaluate: deleaf, harvest, or deleaf,harvest.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N labeled UUIDs.")
    parser.add_argument("--uuids", type=str, default=None, help="Comma-separated UUIDs to evaluate. Defaults to all labels.")
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    uuids = [uuid.strip() for uuid in args.uuids.split(",") if uuid.strip()] if args.uuids else list(labels.keys())
    unknown_uuids = [uuid for uuid in uuids if uuid not in labels]
    if unknown_uuids:
        raise ValueError(f"UUIDs not found in labels: {', '.join(unknown_uuids)}")
    if args.limit is not None:
        uuids = uuids[: args.limit]
    threshold_label = threshold_label_cm(args.threshold_m)

    match_rows: list[dict[str, Any]] = []
    gt_nearest_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    scene_metric_docs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frame_transform_rows: list[dict[str, Any]] = []

    for trial_idx, uuid in enumerate(uuids, start=1):
        print(f"[{trial_idx}/{len(uuids)}] {uuid}", flush=True)
        gt_transform = identity_transform()
        frame_info: dict[str, Any] = {"labels_frame": args.labels_frame, "prediction_frame": args.mps_source}
        if args.labels_frame == "full-vrs":
            if args.mps_source != "multi-slam":
                raise ValueError("--labels-frame full-vrs currently requires --mps-source multi-slam")
            gt_transform, transform_info = estimate_full_to_multi_transform(args.full_mps_root, args.multi_slam_root, uuid)
            frame_info.update(transform_info)
            frame_transform_rows.append(
                {
                    "uuid": uuid,
                    "method": transform_info["method"],
                    "n_pose_matches": transform_info["n_pose_matches"],
                    "median_time_delta_us": transform_info["median_time_delta_us"],
                    "mean_residual_cm": transform_info["mean_residual_cm"],
                    "median_residual_cm": transform_info["median_residual_cm"],
                    "max_residual_cm": transform_info["max_residual_cm"],
                }
            )
        for task in args.tasks:
            spec = TASK_SPECS[task]
            gt_original = np.asarray(labels[uuid][spec["label_key"]], dtype=np.float64).reshape(-1, 3)
            gt = apply_transform(gt_original, gt_transform)
            run_dir = args.out_dir / "runs" / uuid / task
            targets_path = run_dir / "targets.json"

            try:
                write_gt_points(run_dir, gt_original, gt, frame_info)
                if not args.skip_existing or not targets_path.exists():
                    pre_points, pre_obs = input_paths(
                        args.dataset_root, args.multi_slam_root, args.mps_source, uuid, spec, "pre"
                    )
                    post_points, post_obs = input_paths(
                        args.dataset_root, args.multi_slam_root, args.mps_source, uuid, spec, "post"
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
                        config_path=Path(spec["config"]),
                        refine_icp=args.refine_icp and args.mps_source != "multi-slam",
                        strict=args.strict,
                    )
                pred = read_predictions(targets_path)
                metrics = evaluate_cut_points(pred, gt, threshold_m=args.threshold_m)
                scene_metric_docs.append({"uuid": uuid, "task": task, **metrics})
                trial_summary = {
                    "uuid": uuid,
                    "task": task,
                    "num_gt": metrics["num_gt"],
                    "num_pred": metrics["num_pred"],
                    "TP": metrics["TP"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1": metrics["F1"],
                    f"Precision@{threshold_label}": metrics[f"Precision@{threshold_label}"],
                    f"Recall@{threshold_label}": metrics[f"Recall@{threshold_label}"],
                    f"F1@{threshold_label}": metrics[f"F1@{threshold_label}"],
                    "matched_mean_distance": metrics["matched_mean_distance"],
                    "matched_mean_distance_cm": metrics["matched_mean_distance_cm"],
                    "gt_to_nearest_mean_distance": metrics["gt_to_nearest_mean_distance"],
                    "gt_to_nearest_mean_distance_cm": metrics["gt_to_nearest_mean_distance_cm"],
                    # Backward-compatible aliases for older notebooks/scripts.
                    "n_gt": metrics["num_gt"],
                    "n_pred": metrics["num_pred"],
                    "n_missing": metrics["FN"],
                    "n_extra": metrics["FP"],
                    "run_dir": str(run_dir),
                }
                trial_rows.append(trial_summary)
                for row in metrics["matches"]:
                    match_rows.append({"uuid": uuid, "task": task, **row})
                for row in metrics["gt_to_nearest"]:
                    gt_nearest_rows.append({"uuid": uuid, "task": task, **row})
                print(
                    f"  {task}: gt={len(gt)} pred={len(pred)} TP={metrics['TP']} FP={metrics['FP']} "
                    f"FN={metrics['FN']} F1@{threshold_label}={metrics['F1']:.3f} "
                    f"matched_mean_cm={metrics['matched_mean_distance_cm']}",
                    flush=True,
                )
            except Exception as exc:
                metrics = evaluate_cut_points(np.empty((0, 3), dtype=np.float64), gt, threshold_m=args.threshold_m)
                scene_metric_docs.append({"uuid": uuid, "task": task, **metrics})
                failures.append({"uuid": uuid, "task": task, "error": repr(exc), "run_dir": str(run_dir)})
                trial_rows.append(
                    {
                        "uuid": uuid,
                        "task": task,
                        "num_gt": metrics["num_gt"],
                        "num_pred": metrics["num_pred"],
                        "TP": metrics["TP"],
                        "FP": metrics["FP"],
                        "FN": metrics["FN"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1": metrics["F1"],
                        f"Precision@{threshold_label}": metrics[f"Precision@{threshold_label}"],
                        f"Recall@{threshold_label}": metrics[f"Recall@{threshold_label}"],
                        f"F1@{threshold_label}": metrics[f"F1@{threshold_label}"],
                        "matched_mean_distance": metrics["matched_mean_distance"],
                        "matched_mean_distance_cm": metrics["matched_mean_distance_cm"],
                        "gt_to_nearest_mean_distance": metrics["gt_to_nearest_mean_distance"],
                        "gt_to_nearest_mean_distance_cm": metrics["gt_to_nearest_mean_distance_cm"],
                        "n_gt": metrics["num_gt"],
                        "n_pred": metrics["num_pred"],
                        "n_missing": metrics["FN"],
                        "n_extra": metrics["FP"],
                        "run_dir": str(run_dir),
                    }
                )
                print(f"  {task}: FAILED {exc!r}", flush=True)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_rows.append({"task": "all", **aggregate_cut_point_metrics(scene_metric_docs, threshold_m=args.threshold_m)})
    for task in args.tasks:
        task_metrics = [row for row in scene_metric_docs if row["task"] == task]
        aggregate_rows.append({"task": task, **aggregate_cut_point_metrics(task_metrics, threshold_m=args.threshold_m)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "matches.csv", match_rows)
    write_csv(args.out_dir / "gt_to_nearest.csv", gt_nearest_rows)
    write_csv(args.out_dir / "trial_summary.csv", trial_rows)
    write_csv(args.out_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(args.out_dir / "frame_transforms.csv", frame_transform_rows)
    summary = {
        "labels": args.labels,
        "dataset_root": args.dataset_root,
        "full_mps_root": args.full_mps_root,
        "labels_frame": args.labels_frame,
        "mps_source": args.mps_source,
        "multi_slam_root": args.multi_slam_root,
        "threshold_m": args.threshold_m,
        "tasks": args.tasks,
        "n_label_entries": len(labels),
        "n_evaluated_entries": len(uuids),
        "aggregates": aggregate_rows,
        "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8")

    print("\nAggregate summary:")
    for row in aggregate_rows:
        print(row)
    if failures:
        print(f"\nFailures: {len(failures)}")
        for failure in failures:
            print(failure)


if __name__ == "__main__":
    main()
