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

from tools import run_yolo26l_box32_mps_pretrain_2dgt_finetune_035 as base  # noqa: E402


TASK_ID = "yolo26l_box32_mps_pretrain_2dgt_besthp_harvest_042"
TASK = "harvesting"
BOX_SIZE = 32
TARGET_M = 0.020
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/yolo26l_box32_mps_pretrain_2dgt_besthp_harvest_042")
DEFAULT_REPORT = REPO_ROOT / "research" / "REPORTS" / "yolo26l_box32_mps_pretrain_2dgt_besthp_harvest_042.md"
DEFAULT_MODEL = REPO_ROOT / "yolo26l.pt"
DEFAULT_LABELS = Path("/tmp/egodelta/front/labels.jsonl")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
MPS_DATA = Path("/tmp/egodelta_1cm/aria_mps_2d_stride5_033/datasets/harvesting_box32/data.yaml")
MANUAL_DATA = Path("/tmp/egodelta_1cm/manual_front_yolo_032/datasets/harvesting_box32/data.yaml")
PRIOR_2DGT_ONLY_M = 0.014407
PRIOR_MPS_2DGT_M = 0.007425


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvesting YOLO26L box32 MPS pretrain then 2DGT fine-tune with best 037 eval settings."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--mps-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="2")
    parser.add_argument("--predict-batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.00001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42042)
    return parser.parse_args()


def main() -> int:
    base.set_thread_limits()
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    local_vendor = base.ensure_local_vendor(args.out_root)
    sys.path.insert(0, str(local_vendor))
    validate_inputs(args)

    accepted_labels = base.write_accepted_only_labels(args.labels_jsonl, args.out_root / "labels_accepted_only.jsonl")
    run_config = build_run_config(args, accepted_labels)
    write_json(args.out_root / "run_config.json", run_config)

    task_dir = args.out_root / "runs" / TASK
    mps_train_dir = task_dir / "mps_pretrain"
    mps_batch = base.train_with_retry(
        model_path=args.model,
        data_yaml=MPS_DATA,
        train_dir=mps_train_dir,
        task=TASK,
        stage="mps_pretrain",
        epochs=int(args.mps_epochs),
        args=args,
    )
    mps_weights = base.select_weights(mps_train_dir)

    finetune_train_dir = task_dir / "manual_finetune"
    finetune_batch = base.train_with_retry(
        model_path=mps_weights,
        data_yaml=MANUAL_DATA,
        train_dir=finetune_train_dir,
        task=TASK,
        stage="manual_finetune",
        epochs=int(args.finetune_epochs),
        args=args,
    )
    finetune_weights = base.select_weights(finetune_train_dir)

    selector_dir = task_dir / "selector_eval_test"
    selector_summary = run_selector_eval(finetune_weights, selector_dir, accepted_labels, args)
    topconf_dir = task_dir / "topconf_eval_test"
    topconf_summary = run_topconf_eval(finetune_weights, topconf_dir, accepted_labels, args)

    primary = find_selector_result(
        selector_summary,
        selector="nearest_highconf_train_median",
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        max_det=int(args.max_det),
    )
    best_deployable = selector_summary.get("best_deployable_by_task", {}).get(TASK, {})
    topconf_metrics = topconf_summary.get("metrics", {}).get(TASK, {})
    summary = {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "status": "complete",
        "task": TASK,
        "box_size_px": BOX_SIZE,
        "target_m": TARGET_M,
        "run_config": run_config,
        "training": {
            "mps_pretrain": {
                "data_yaml": str(MPS_DATA.resolve()),
                "train_dir": str(mps_train_dir.resolve()),
                "weights": str(mps_weights.resolve()),
                "epochs": int(args.mps_epochs),
                "batch_used": int(mps_batch),
            },
            "manual_finetune": {
                "data_yaml": str(MANUAL_DATA.resolve()),
                "train_dir": str(finetune_train_dir.resolve()),
                "weights": str(finetune_weights.resolve()),
                "epochs": int(args.finetune_epochs),
                "batch_used": int(finetune_batch),
            },
        },
        "evaluation": {
            "selector_summary": str((selector_dir / "summary.json").resolve()),
            "topconf_summary": str((topconf_dir / "summary.json").resolve()),
            "primary_selector": primary,
            "best_deployable_by_selector_grid": best_deployable,
            "top_confidence_metrics": compact_topconf_metrics(topconf_metrics),
        },
        "comparison": {
            "manual_2dgt_only_037_m": PRIOR_2DGT_ONLY_M,
            "prior_mps_pretrain_2dgt_035_m": PRIOR_MPS_2DGT_M,
            "primary_delta_vs_2dgt_only_m": maybe_delta(metric_mean(primary), PRIOR_2DGT_ONLY_M),
            "primary_delta_vs_prior_mps_2dgt_m": maybe_delta(metric_mean(primary), PRIOR_MPS_2DGT_M),
        },
        "target_reached": bool(metric_mean(primary) is not None and float(metric_mean(primary)) < TARGET_M),
        "metric_caveat": metric_caveat(),
        "artifacts": {
            "root": str(args.out_root.resolve()),
            "run_config": str((args.out_root / "run_config.json").resolve()),
            "summary": str((args.out_root / "summary.json").resolve()),
            "report": str(args.report_path.resolve()),
        },
    }
    write_json(args.out_root / "summary.json", summary)
    write_report(args.report_path, args, summary)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve())}, indent=2))
    return 0


def validate_inputs(args: argparse.Namespace) -> None:
    required = [args.model, args.labels_jsonl, MPS_DATA, MANUAL_DATA]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def build_run_config(args: argparse.Namespace, accepted_labels: Path) -> dict[str, Any]:
    return {
        "role": "worker-042 harvesting delegated by PI/orchestrator",
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "task": TASK,
        "model": str(args.model.resolve()),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "accepted_only_labels_jsonl": str(accepted_labels.resolve()),
        "front_root": str(args.front_root.resolve()),
        "out_root": str(args.out_root.resolve()),
        "box_size_px": BOX_SIZE,
        "mps_data_yaml": str(MPS_DATA.resolve()),
        "manual_data_yaml": str(MANUAL_DATA.resolve()),
        "mps_epochs": int(args.mps_epochs),
        "finetune_epochs": int(args.finetune_epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "predict_batch": int(args.predict_batch),
        "eval_selector": "nearest_highconf_train_median",
        "eval_conf": float(args.conf),
        "eval_iou": float(args.iou),
        "eval_max_det": int(args.max_det),
        "validation_policy": "YOLO train uses val=False. No validation split is used for checkpoint selection.",
        "augmentation_policy": "Geometry/color augmentation disabled, matching prior 035 fixed-box32 chain.",
        "gpu_policy": "Explicit physical GPU ID; no CUDA_VISIBLE_DEVICES masking.",
        "thread_limits": base.thread_limits_payload(),
        "training_commands_equivalent": equivalent_commands(args),
    }


def equivalent_commands(args: argparse.Namespace) -> list[str]:
    return [
        "YOLO(model=yolo26l.pt).train(data=aria_mps_2d_stride5_033/datasets/harvesting_box32/data.yaml, "
        f"epochs={int(args.mps_epochs)}, imgsz={int(args.imgsz)}, batch={int(args.batch)}, "
        f"workers={int(args.workers)}, device={args.device}, val=False, mosaic=0, mixup=0, copy_paste=0, "
        "degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0)",
        "YOLO(mps_pretrain_best.pt).train(data=manual_front_yolo_032/datasets/harvesting_box32/data.yaml, "
        f"epochs={int(args.finetune_epochs)}, imgsz={int(args.imgsz)}, batch={int(args.batch)}, "
        f"workers={int(args.workers)}, device={args.device}, val=False, same_disabled_augmentation=True)",
        "python tools/evaluate_yolo26l_box32_selectors_036.py "
        f"--tasks harvesting --harvesting-weights manual_finetune_best.pt --imgsz-list {int(args.imgsz)} "
        f"--conf-list {float(args.conf)} --max-det-list {int(args.max_det)} --device {args.device}",
    ]


def run_selector_eval(weights: Path, out_dir: Path, labels_jsonl: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_yolo26l_box32_selectors_036.py"),
        "--labels-jsonl",
        str(labels_jsonl),
        "--front-root",
        str(args.front_root),
        "--out-root",
        str(out_dir),
        "--split",
        "test",
        "--tasks",
        TASK,
        "--harvesting-weights",
        str(weights),
        "--checkpoint-tag",
        TASK_ID,
        "--imgsz-list",
        str(int(args.imgsz)),
        "--conf-list",
        str(float(args.conf)),
        "--max-det-list",
        str(int(args.max_det)),
        "--iou",
        str(float(args.iou)),
        "--batch",
        str(int(args.predict_batch)),
        "--device",
        str(args.device),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=eval_env(args.out_root), check=True)
    return read_json(out_dir / "summary.json")


def run_topconf_eval(weights: Path, out_dir: Path, labels_jsonl: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_manual_front_yolo_032.py"),
        "--labels-jsonl",
        str(labels_jsonl),
        "--front-root",
        str(args.front_root),
        "--out-root",
        str(out_dir),
        "--split",
        "test",
        "--tasks",
        TASK,
        "--harvesting-weights",
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
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=eval_env(args.out_root), check=True)
    return read_json(out_dir / "summary.json")


def eval_env(out_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(out_root / "vendor_py") + os.pathsep + env.get("PYTHONPATH", "")
    if VENDOR_PY.exists():
        env["PYTHONPATH"] = str(VENDOR_PY) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def find_selector_result(summary: dict[str, Any], selector: str, imgsz: int, conf: float, max_det: int) -> dict[str, Any]:
    for result in summary.get("results", []):
        if (
            result.get("task") == TASK
            and result.get("selector") == selector
            and int(result.get("imgsz", -1)) == int(imgsz)
            and abs(float(result.get("conf", -1.0)) - float(conf)) < 1.0e-12
            and int(result.get("max_det", -1)) == int(max_det)
        ):
            return result
    return {}


def compact_topconf_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_samples": metrics.get("num_samples"),
        "num_missing_detection": metrics.get("num_missing_detection"),
        "miss_rate": metrics.get("miss_rate"),
        "center_error_px_mean": nested(metrics, "center_error_px", "mean"),
        "center_error_px_median": nested(metrics, "center_error_px", "median"),
        "center_error_px_p90": nested(metrics, "center_error_px", "p90"),
        "oracle_depth_all_mean_m": nested(metrics, "error_3d_m_with_miss_penalty", "mean"),
        "oracle_depth_all_median_m": nested(metrics, "error_3d_m_with_miss_penalty", "median"),
        "oracle_depth_all_p90_m": nested(metrics, "error_3d_m_with_miss_penalty", "p90"),
        "le_0p02m": nested(metrics, "error_3d_m_with_miss_penalty", "le_0p02m"),
    }


def metric_mean(result: dict[str, Any]) -> float | None:
    value = nested(result.get("metrics", {}), "error_3d_m_with_miss_penalty", "mean")
    return float(value) if value is not None else None


def maybe_delta(value: float | None, baseline: float) -> float | None:
    if value is None:
        return None
    return float(value) - float(baseline)


def nested(payload: dict[str, Any], key1: str, key2: str) -> Any:
    return payload.get(key1, {}).get(key2)


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    primary = summary["evaluation"]["primary_selector"]
    best = summary["evaluation"]["best_deployable_by_selector_grid"]
    topconf = summary["evaluation"]["top_confidence_metrics"]
    lines = [
        "# YOLO26L Box32 MPS Pretrain + 2DGT Best-HP Harvest 042",
        "",
        f"Date: {utc_now()}",
        "Role: worker-042 harvesting, delegated by PI/orchestrator",
        "Status: complete",
        f"Output root: `{args.out_root.resolve()}`",
        "",
        "## Protocol",
        "",
        "- Task: harvesting only.",
        "- Fixed bbox size: 32x32.",
        "- Training: MPS stride-5 harvesting box32 pretrain, then manual 2DGT harvesting box32 fine-tune.",
        "- No validation set: YOLO training used `val=False`; the test set was used only after training.",
        f"- Best transferred evaluation settings: `imgsz={int(args.imgsz)}`, `conf={float(args.conf):.5f}`, `max_det={int(args.max_det)}`, selector `nearest_highconf_train_median`.",
        f"- Resource settings: physical GPU `{args.device}`, workers={int(args.workers)}, batch={int(args.batch)}, predict batch={int(args.predict_batch)}, thread limits {base.thread_limits_payload()}.",
        "",
        "## Training Commands",
        "",
    ]
    for command in summary["run_config"]["training_commands_equivalent"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- MPS pretrain: `{summary['training']['mps_pretrain']['weights']}`",
            f"- 2DGT fine-tune: `{summary['training']['manual_finetune']['weights']}`",
            "",
            "## Results",
            "",
            "| Eval policy | N | Miss | 2D mean px | 2D median px | 2D p90 px | All-sample oracle-depth mean m | Median m | P90 m | <=2cm | Target <2cm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            result_row("nearest_highconf_train_median", primary),
            topconf_row("top-confidence", topconf),
            best_selector_row("best deployable in same grid", best),
            "",
            "## Comparison",
            "",
            f"- 2DGT-only 037 best deployable: `{PRIOR_2DGT_ONLY_M:.6f} m`.",
            f"- Prior MPS pretrain + 2DGT fine-tune 035: `{PRIOR_MPS_2DGT_M:.6f} m`.",
            f"- Current primary selector delta vs 2DGT-only: `{fmt(summary['comparison']['primary_delta_vs_2dgt_only_m'], 6)} m`.",
            f"- Current primary selector delta vs prior MPS+2DGT: `{fmt(summary['comparison']['primary_delta_vs_prior_mps_2dgt_m'], 6)} m`.",
            "",
            "## Critical Analysis",
            "",
            critical_analysis(primary, best),
            "",
            "## Metric Caveat",
            "",
            metric_caveat(),
            "",
            "## Artifacts",
            "",
            f"- Summary: `{summary['artifacts']['summary']}`",
            f"- Selector evaluation: `{summary['evaluation']['selector_summary']}`",
            f"- Top-confidence evaluation: `{summary['evaluation']['topconf_summary']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_row(name: str, result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    px = metrics.get("center_error_px", {})
    m = metrics.get("error_3d_m_with_miss_penalty", {})
    miss = metrics.get("miss_rate")
    reached = "yes" if m.get("mean") is not None and float(m["mean"]) < TARGET_M else "no"
    return (
        f"| {name} | {fmt_int(metrics.get('num_samples'))} | {fmt_pct(miss)} | "
        f"{fmt(px.get('mean'), 2)} | {fmt(px.get('median'), 2)} | {fmt(px.get('p90'), 2)} | "
        f"{fmt(m.get('mean'), 6)} | {fmt(m.get('median'), 6)} | {fmt(m.get('p90'), 6)} | "
        f"{fmt_pct(m.get('le_0p02m'))} | {reached} |"
    )


def topconf_row(name: str, metrics: dict[str, Any]) -> str:
    reached = (
        "yes"
        if metrics.get("oracle_depth_all_mean_m") is not None
        and float(metrics["oracle_depth_all_mean_m"]) < TARGET_M
        else "no"
    )
    return (
        f"| {name} | {fmt_int(metrics.get('num_samples'))} | {fmt_pct(metrics.get('miss_rate'))} | "
        f"{fmt(metrics.get('center_error_px_mean'), 2)} | {fmt(metrics.get('center_error_px_median'), 2)} | "
        f"{fmt(metrics.get('center_error_px_p90'), 2)} | {fmt(metrics.get('oracle_depth_all_mean_m'), 6)} | "
        f"{fmt(metrics.get('oracle_depth_all_median_m'), 6)} | {fmt(metrics.get('oracle_depth_all_p90_m'), 6)} | "
        f"{fmt_pct(metrics.get('le_0p02m'))} | {reached} |"
    )


def best_selector_row(name: str, result: dict[str, Any]) -> str:
    label = f"{name}: {result.get('selector', '')}"
    return result_row(label, {"metrics": selector_compact_to_metrics(result)})


def selector_compact_to_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_samples": result.get("num_samples"),
        "miss_rate": result.get("miss_rate"),
        "center_error_px": {
            "mean": result.get("center_error_px_mean"),
            "median": result.get("center_error_px_median"),
            "p90": result.get("center_error_px_p90"),
        },
        "error_3d_m_with_miss_penalty": {
            "mean": result.get("oracle_depth_error_m_mean"),
            "median": result.get("oracle_depth_error_m_median"),
            "p90": result.get("oracle_depth_error_m_p90"),
            "le_0p02m": result.get("le_0p02m"),
        },
    }


def critical_analysis(primary: dict[str, Any], best: dict[str, Any]) -> str:
    primary_m = metric_mean(primary)
    best_m = best.get("oracle_depth_error_m_mean")
    if primary_m is not None and primary_m < TARGET_M:
        return (
            "The transferred 037 selector remains below 2 cm after MPS pretraining and 2DGT fine-tuning. "
            "For harvesting, the target is visually compact and the train-median high-confidence selector suppresses "
            "low-confidence false positives while preserving recall, so MPS pretraining is useful when it improves "
            "the detector's candidate localization without adding misses."
        )
    if best_m is not None and float(best_m) < TARGET_M:
        return (
            "The same-grid deployable selector can reach the 2 cm target, but the exact transferred selector does not. "
            "That means MPS pretraining changed confidence/ranking behavior enough that selector calibration should be "
            "reselected from train-only criteria before deployment."
        )
    return (
        "This run does not reach 2 cm. The likely failure mode is not missing detections alone, but ranking/calibration "
        "shift after MPS pretraining: MPS boxes come from projected 3D labels and may bias confidence toward nearby "
        "tomato-like candidates that are not the manual cut point. Next improvement should tune the selector on the "
        "manual train split or fine-tune longer/lower learning rate from the MPS checkpoint."
    )


def metric_caveat() -> str:
    return (
        "Manual labels are 2D-only. The meter metric is oracle-depth lateral 3D error: predicted and manual "
        "2D points are backprojected at the same nearest-source-target depth. This is not strict full 3D GT."
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
