#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PY = Path("/tmp/egodelta_1cm/manual_front_yolo_032/vendor_py")
if VENDOR_PY.exists():
    sys.path.insert(0, str(VENDOR_PY))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_ID = "yolo26l_box32_mps_pretrain_2dgt_finetune_035"
TASKS = ("deleafing", "harvesting")
BOX_SIZE = 32
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/yolo26l_box32_mps_pretrain_2dgt_finetune_035")
DEFAULT_REPORT = REPO_ROOT / "research" / "REPORTS" / "yolo26l_box32_mps_pretrain_2dgt_finetune_035.md"
DEFAULT_MODEL = REPO_ROOT / "yolo26l.pt"
DEFAULT_LABELS = Path("/tmp/egodelta/front/labels.jsonl")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
BASELINE_SUMMARY = Path(
    "/tmp/egodelta_1cm/manual_2d_gt_yolo26l_box_sweep_034/lanes_physical_devices_001/box32/summary.json"
)

MPS_DATA = {
    "deleafing": Path("/tmp/egodelta_1cm/aria_mps_2d_stride5_033/datasets/deleafing_box32/data.yaml"),
    "harvesting": Path("/tmp/egodelta_1cm/aria_mps_2d_stride5_033/datasets/harvesting_box32/data.yaml"),
}
MANUAL_DATA = {
    "deleafing": Path("/tmp/egodelta_1cm/manual_front_yolo_032/datasets/deleafing_box32/data.yaml"),
    "harvesting": Path("/tmp/egodelta_1cm/manual_front_yolo_032/datasets/harvesting_box32/data.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-large box32 MPS pretraining followed by manual 2D-GT fine-tuning."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--mps-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="3")
    parser.add_argument("--predict-batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--seed", type=int, default=35035)
    parser.add_argument("--threshold-m", type=float, default=0.020)
    return parser.parse_args()


def main() -> int:
    set_thread_limits()
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    local_vendor = ensure_local_vendor(args.out_root)
    sys.path.insert(0, str(local_vendor))
    if not args.model.exists():
        write_blocked(args, f"Missing requested checkpoint: {args.model}")
        return 2

    accepted_labels = write_accepted_only_labels(args.labels_jsonl, args.out_root / "labels_accepted_only.jsonl")
    datasets = validate_datasets(tasks)
    baseline = load_baseline()
    run_config = {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "pid": os.getpid(),
        "model": str(args.model.resolve()),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "accepted_only_labels_jsonl": str(accepted_labels.resolve()),
        "front_root": str(args.front_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "report_path": str(args.report_path.resolve()),
        "tasks": list(tasks),
        "box_size_px": BOX_SIZE,
        "mps_epochs": int(args.mps_epochs),
        "finetune_epochs": int(args.finetune_epochs),
        "initial_batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "workers": int(args.workers),
        "device": str(args.device),
        "predict_batch": int(args.predict_batch),
        "threshold_m": float(args.threshold_m),
        "thread_limits": thread_limits_payload(),
        "datasets": datasets,
        "validation_policy": "YOLO train runs with val=False. data.yaml val entries are compatibility aliases only.",
        "augmentation_policy": "All geometry/color augmentation disabled to match the manual 2D-GT-only box32 baseline lane.",
        "gpu_policy": "Explicit physical GPU device IDs only; no CUDA_VISIBLE_DEVICES masking.",
    }
    write_json(args.out_root / "run_config.json", run_config)

    results: list[dict[str, Any]] = []
    for task in tasks:
        result = run_task_chain(task, accepted_labels, args)
        results.append(result)
        write_json(args.out_root / "partial_summary.json", summary_payload(args, run_config, results, baseline))

    summary = summary_payload(args, run_config, results, baseline)
    write_json(args.out_root / "summary.json", summary)
    write_report(args.report_path, args, run_config, results, baseline, summary)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve())}, indent=2))
    return 0


def parse_tasks(text: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not tasks:
        raise ValueError("No tasks requested")
    bad = [task for task in tasks if task not in TASKS]
    if bad:
        raise ValueError(f"Unsupported task(s): {bad}; allowed={TASKS}")
    return tasks


def set_thread_limits() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def ensure_local_vendor(out_root: Path) -> Path:
    vendor = out_root / "vendor_py"
    vendor.mkdir(parents=True, exist_ok=True)
    polars_stub = vendor / "polars.py"
    if not polars_stub.exists():
        polars_stub.write_text(
            "\n".join(
                [
                    "import csv",
                    "",
                    "class _Frame:",
                    "    def __init__(self, rows):",
                    "        self._rows = rows",
                    "",
                    "    def to_dict(self, as_series=False):",
                    "        keys = []",
                    "        for row in self._rows:",
                    "            for key in row.keys():",
                    "                if key not in keys:",
                    "                    keys.append(key)",
                    "        return {key: [_coerce(row.get(key, '')) for row in self._rows] for key in keys}",
                    "",
                    "def _coerce(value):",
                    "    try:",
                    "        if value == '':",
                    "            return None",
                    "        return float(value)",
                    "    except Exception:",
                    "        return value",
                    "",
                    "def read_csv(path, infer_schema_length=None):",
                    "    with open(path, 'r', encoding='utf-8', newline='') as handle:",
                    "        return _Frame(list(csv.DictReader(handle)))",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return vendor


def thread_limits_payload() -> dict[str, str]:
    return {
        key: os.environ.get(key, "")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }


def validate_datasets(tasks: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in tasks:
        for stage, mapping in (("mps_pretrain", MPS_DATA), ("manual_finetune", MANUAL_DATA)):
            data_yaml = mapping[task]
            if not data_yaml.exists():
                raise FileNotFoundError(f"Missing {stage} data.yaml for {task}: {data_yaml}")
            dataset_dir = data_yaml.parent
            summary_path = dataset_dir / "dataset_summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {}
            box_size = summary.get("box_size_px")
            if box_size is not None and int(box_size) != BOX_SIZE:
                raise ValueError(f"{summary_path} reports box_size_px={box_size}, expected {BOX_SIZE}")
            out[f"{task}_{stage}"] = {
                "data_yaml": str(data_yaml.resolve()),
                "dataset_summary": str(summary_path.resolve()) if summary_path.exists() else None,
                "summary": compact_dataset_summary(summary),
            }
    return out


def compact_dataset_summary(summary: dict[str, Any]) -> dict[str, Any]:
    splits = summary.get("splits", {})
    return {
        "task": summary.get("task"),
        "box_size_px": summary.get("box_size_px"),
        "splits": {
            split: {
                "num_images": payload.get("num_images"),
                "num_labels": payload.get("num_labels"),
                "multi_label_images": payload.get("multi_label_images"),
                "max_labels_per_image": payload.get("max_labels_per_image"),
                "shrunk_box_labels": payload.get("shrunk_box_labels"),
            }
            for split, payload in splits.items()
        },
    }


def write_accepted_only_labels(labels_path: Path, out_path: Path) -> Path:
    rows = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("status")) == "accepted":
                rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out_path


def run_task_chain(task: str, labels_jsonl: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = utc_now()
    task_dir = args.out_root / "runs" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    mps_train_dir = task_dir / "mps_pretrain"
    mps_batch = train_with_retry(
        model_path=args.model,
        data_yaml=MPS_DATA[task],
        train_dir=mps_train_dir,
        task=task,
        stage="mps_pretrain",
        epochs=int(args.mps_epochs),
        args=args,
    )
    mps_weights = select_weights(mps_train_dir)

    finetune_train_dir = task_dir / "manual_finetune"
    finetune_batch = train_with_retry(
        model_path=mps_weights,
        data_yaml=MANUAL_DATA[task],
        train_dir=finetune_train_dir,
        task=task,
        stage="manual_finetune",
        epochs=int(args.finetune_epochs),
        args=args,
    )
    finetune_weights = select_weights(finetune_train_dir)

    eval_dir = task_dir / "eval_test"
    eval_summary = run_evaluator(task, finetune_weights, eval_dir, labels_jsonl, args)
    result = {
        "task": task,
        "box_size_px": BOX_SIZE,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "mps_pretrain": {
            "data_yaml": str(MPS_DATA[task].resolve()),
            "train_dir": str(mps_train_dir.resolve()),
            "weights": str(mps_weights.resolve()),
            "batch_used": int(mps_batch),
            "epochs": int(args.mps_epochs),
        },
        "manual_finetune": {
            "data_yaml": str(MANUAL_DATA[task].resolve()),
            "train_dir": str(finetune_train_dir.resolve()),
            "weights": str(finetune_weights.resolve()),
            "batch_used": int(finetune_batch),
            "epochs": int(args.finetune_epochs),
        },
        "eval_summary": str((eval_dir / "summary.json").resolve()),
        "metrics": eval_summary.get("metrics", {}),
        "threshold_reached": threshold_reached(task, eval_summary.get("metrics", {}), float(args.threshold_m)),
    }
    write_json(task_dir / "result_summary.json", result)
    return result


def train_with_retry(
    model_path: Path,
    data_yaml: Path,
    train_dir: Path,
    task: str,
    stage: str,
    epochs: int,
    args: argparse.Namespace,
) -> int:
    batch = max(1, int(args.batch))
    last_exc = None
    while batch >= 1:
        try:
            train_yolo(model_path, data_yaml, train_dir, task, stage, epochs, batch, args)
            return batch
        except RuntimeError as exc:
            text = str(exc).lower()
            last_exc = exc
            if "out of memory" not in text and "cuda" not in text:
                raise
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            write_json(train_dir.parent / f"{stage}_oom_retry.json", {"next_batch": batch, "error": str(last_exc), "at_utc": utc_now()})
    raise RuntimeError("Training failed") from last_exc


def train_yolo(
    model_path: Path,
    data_yaml: Path,
    train_dir: Path,
    task: str,
    stage: str,
    epochs: int,
    batch: int,
    args: argparse.Namespace,
) -> None:
    from ultralytics import YOLO  # type: ignore

    train_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path.resolve()))
    model.train(
        data=str(data_yaml.resolve()),
        epochs=int(epochs),
        batch=int(batch),
        imgsz=int(args.imgsz),
        workers=int(args.workers),
        device=str(args.device),
        project=str(train_dir.parent),
        name=train_dir.name,
        exist_ok=True,
        task="detect",
        val=False,
        plots=False,
        verbose=False,
        seed=int(args.seed) + (0 if task == "deleafing" else 1000) + (0 if stage == "mps_pretrain" else 100),
        patience=0,
        cache=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )


def select_weights(train_dir: Path) -> Path:
    best = train_dir / "weights" / "best.pt"
    last = train_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No trained weights found under {train_dir / 'weights'}")


def run_evaluator(task: str, weights: Path, eval_dir: Path, labels_jsonl: Path, args: argparse.Namespace) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_manual_front_yolo_032.py"),
        "--labels-jsonl",
        str(labels_jsonl),
        "--front-root",
        str(args.front_root),
        "--out-root",
        str(eval_dir),
        "--split",
        "test",
        "--tasks",
        task,
        f"--{task}-weights",
        str(weights),
        "--imgsz",
        str(int(args.imgsz)),
        "--conf",
        str(float(args.conf)),
        "--iou",
        str(float(args.iou)),
        "--max-det",
        str(int(args.max_det)),
        "--batch",
        str(int(args.predict_batch)),
        "--device",
        str(args.device),
        "--overlay-count",
        "0",
        "--depth-policy",
        "nearest-source-target",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.out_root / "vendor_py") + os.pathsep + env.get("PYTHONPATH", "")
    if VENDOR_PY.exists():
        env["PYTHONPATH"] = str(VENDOR_PY) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    return read_json(eval_dir / "summary.json")


def threshold_reached(task: str, metrics: dict[str, Any], threshold: float) -> bool:
    mean_m = nested(metrics.get(task, {}), "error_3d_m_with_miss_penalty", "mean")
    return bool(mean_m is not None and float(mean_m) < float(threshold))


def load_baseline() -> dict[str, Any]:
    if not BASELINE_SUMMARY.exists():
        return {"summary_path": str(BASELINE_SUMMARY), "available": False, "by_task": {}}
    payload = read_json(BASELINE_SUMMARY)
    by_task = {}
    for task, result in payload.get("best_by_task", {}).items():
        by_task[task] = {
            "weights": result.get("weights"),
            "center_error_px_mean": result.get("center_error_px_mean"),
            "center_error_px_median": result.get("center_error_px_median"),
            "center_error_px_p90": result.get("center_error_px_p90"),
            "oracle_depth_all_mean_m": result.get("oracle_depth_error_m_mean"),
            "oracle_depth_all_median_m": result.get("oracle_depth_error_m_median"),
            "oracle_depth_all_p90_m": result.get("oracle_depth_error_m_p90"),
            "miss_rate": result.get("miss_rate"),
            "box_size_px": result.get("box_size_px"),
        }
    return {"summary_path": str(BASELINE_SUMMARY.resolve()), "available": True, "by_task": by_task}


def summary_payload(
    args: argparse.Namespace,
    run_config: dict[str, Any],
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "status": "complete" if len(results) == len(run_config.get("tasks", [])) else "running",
        "run_config": run_config,
        "results": results,
        "comparison": comparison_payload(results, baseline, float(args.threshold_m)),
        "threshold_m": float(args.threshold_m),
        "target_reached_any": any(result.get("threshold_reached") for result in results),
        "metric_caveat": metric_caveat(),
        "baseline": baseline,
        "artifacts": {
            "root": str(args.out_root.resolve()),
            "run_config": str((args.out_root / "run_config.json").resolve()),
            "summary": str((args.out_root / "summary.json").resolve()),
            "report": str(args.report_path.resolve()),
        },
    }


def comparison_payload(results: list[dict[str, Any]], baseline: dict[str, Any], threshold: float) -> dict[str, Any]:
    rows = {}
    baseline_by_task = baseline.get("by_task", {})
    for result in results:
        task = str(result.get("task"))
        metrics = result.get("metrics", {}).get(task, {})
        all_m = nested(metrics, "error_3d_m_with_miss_penalty", "mean")
        base_m = baseline_by_task.get(task, {}).get("oracle_depth_all_mean_m")
        delta = None
        if all_m is not None and base_m is not None:
            delta = float(all_m) - float(base_m)
        rows[task] = {
            "mps_pretrain_2dgt_finetune": compact_metrics(task, result),
            "manual_2dgt_only_box32_baseline": baseline_by_task.get(task),
            "all_sample_mean_delta_vs_baseline_m": delta,
            "reached_lt_threshold_m": bool(all_m is not None and float(all_m) < threshold),
        }
    return rows


def compact_metrics(task: str, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {}).get(task, {})
    return {
        "weights": result.get("manual_finetune", {}).get("weights"),
        "mps_pretrain_weights": result.get("mps_pretrain", {}).get("weights"),
        "center_error_px_mean": nested(metrics, "center_error_px", "mean"),
        "center_error_px_median": nested(metrics, "center_error_px", "median"),
        "center_error_px_p90": nested(metrics, "center_error_px", "p90"),
        "oracle_depth_covered_mean_m": nested(metrics, "error_3d_m_covered_only", "mean"),
        "oracle_depth_all_mean_m": nested(metrics, "error_3d_m_with_miss_penalty", "mean"),
        "oracle_depth_all_median_m": nested(metrics, "error_3d_m_with_miss_penalty", "median"),
        "oracle_depth_all_p90_m": nested(metrics, "error_3d_m_with_miss_penalty", "p90"),
        "le_0p02m_all": nested(metrics, "error_3d_m_with_miss_penalty", "le_0p02m"),
        "miss_rate": metrics.get("miss_rate"),
        "num_samples": metrics.get("num_samples"),
        "num_missing_detection": metrics.get("num_missing_detection"),
    }


def write_report(
    path: Path,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# YOLO26L Box32 MPS Pretrain + 2D-GT Fine-tune 035",
        "",
        f"Date: {utc_now()}",
        "Status: complete",
        f"Output root: `{args.out_root.resolve()}`",
        f"Base model: `{args.model.resolve()}`",
        "",
        "## Protocol",
        "",
        "- Fixed bbox size: 32x32 only for both MPS pretraining and manual 2D-GT fine-tuning.",
        "- Task chains ran sequentially: MPS stride-5 projected dataset pretraining, then manual front-view 2D-GT fine-tuning, then manual 2D-GT test evaluation.",
        "- Training used YOLO `val=False`; `data.yaml` test entries were not used for model selection.",
        f"- Resource settings: physical GPU device `{args.device}`, workers={int(args.workers)}, initial batch={int(args.batch)}, predict batch={int(args.predict_batch)}, imgsz={int(args.imgsz)}, thread env={thread_limits_payload()}.",
        "- No CUDA_VISIBLE_DEVICES masking was used.",
        "- Augmentation was disabled to keep this comparable to the prior manual 2D-GT-only box32 lane.",
        "",
        "## Metric Caveat",
        "",
        metric_caveat(),
        "",
        "## Datasets",
        "",
        "| Task | Stage | Train labels | Train images | Test labels | Test images | Data yaml |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for task in run_config["tasks"]:
        for stage in ("mps_pretrain", "manual_finetune"):
            data = run_config["datasets"][f"{task}_{stage}"]
            splits = data.get("summary", {}).get("splits", {})
            train = splits.get("train", {})
            test = splits.get("test", {})
            lines.append(
                f"| {task} | {stage} | {fmt_int(train.get('num_labels'))} | {fmt_int(train.get('num_images'))} | "
                f"{fmt_int(test.get('num_labels'))} | {fmt_int(test.get('num_images'))} | `{data['data_yaml']}` |"
            )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Task | N | Miss | 2D mean px | 2D median px | 2D p90 px | Covered oracle-depth mean m | All-sample oracle-depth mean m | Median m | P90 m | <=2cm all | <0.02 m all mean? | Fine-tuned weights |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in results:
        task = str(result["task"])
        metrics = result.get("metrics", {}).get(task, {})
        px = metrics.get("center_error_px", {})
        covered_m = metrics.get("error_3d_m_covered_only", {})
        all_m = metrics.get("error_3d_m_with_miss_penalty", {})
        reached = "yes" if result.get("threshold_reached") else "no"
        lines.append(
            "| {task} | {n} | {miss} | {pxmean} | {pxmed} | {pxp90} | {covered} | {allmean} | {median} | {p90} | {le2} | {reached} | `{weights}` |".format(
                task=task,
                n=int(metrics.get("num_samples", 0)),
                miss=fmt_pct(metrics.get("miss_rate")),
                pxmean=fmt(px.get("mean"), 2),
                pxmed=fmt(px.get("median"), 2),
                pxp90=fmt(px.get("p90"), 2),
                covered=fmt(covered_m.get("mean"), 6),
                allmean=fmt(all_m.get("mean"), 6),
                median=fmt(all_m.get("median"), 6),
                p90=fmt(all_m.get("p90"), 6),
                le2=fmt_pct(all_m.get("le_0p02m")),
                reached=reached,
                weights=result.get("manual_finetune", {}).get("weights", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Baseline Context",
            "",
            "Previous 2D-GT-only YOLO26L box32 baseline from `research/REPORTS/manual_2d_gt_yolo26l_box_sweep_034.md`:",
            "",
            "| Task | Baseline all-sample oracle-depth mean m | Baseline miss | This all-sample mean m | Delta m |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    comparison = summary.get("comparison", {})
    for task in run_config["tasks"]:
        base = baseline.get("by_task", {}).get(task, {})
        current = comparison.get(task, {}).get("mps_pretrain_2dgt_finetune", {})
        delta = comparison.get(task, {}).get("all_sample_mean_delta_vs_baseline_m")
        lines.append(
            f"| {task} | {fmt(base.get('oracle_depth_all_mean_m'), 6)} | {fmt_pct(base.get('miss_rate'))} | "
            f"{fmt(current.get('oracle_depth_all_mean_m'), 6)} | {fmt(delta, 6)} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
        ]
    )
    for result in results:
        task = str(result["task"])
        lines.append(f"- {task} MPS-pretrained: `{result['mps_pretrain']['weights']}`")
        lines.append(f"- {task} fine-tuned: `{result['manual_finetune']['weights']}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in summary["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_caveat() -> str:
    return (
        "Manual labels are 2D-only. The 2D center error is direct pixel error against the manual click. "
        "The meter metric is oracle-depth lateral 3D error from `tools/evaluate_manual_front_yolo_032.py`, "
        "which backprojects the predicted and manual 2D points at the same nearest-source-target depth. "
        "This is not strict full 3D ground truth unless manual labels include per-click depth or 3D coordinates."
    )


def write_blocked(args: argparse.Namespace, reason: str) -> None:
    payload = {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "status": "blocked",
        "reason": reason,
        "expected_model": str(args.model),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "summary.json", payload)
    args.report_path.write_text(
        "\n".join(["# YOLO26L Box32 MPS Pretrain + 2D-GT Fine-tune 035", "", "Status: blocked", "", reason, ""]),
        encoding="utf-8",
    )


def any_reached(results: list[dict[str, Any]], threshold: float) -> bool:
    for result in results:
        task = str(result.get("task"))
        value = nested(result.get("metrics", {}).get(task, {}), "error_3d_m_with_miss_penalty", "mean")
        if value is not None and float(value) < threshold:
            return True
    return False


def nested(payload: dict[str, Any], key1: str, key2: str) -> Any:
    return payload.get(key1, {}).get(key2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.1f}%"


def fmt_int(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


if __name__ == "__main__":
    raise SystemExit(main())
