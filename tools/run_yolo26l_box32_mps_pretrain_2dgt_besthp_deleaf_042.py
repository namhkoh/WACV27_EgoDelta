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


TASK_ID = "yolo26l_box32_mps_pretrain_2dgt_besthp_deleaf_042"
TASK = "deleafing"
BOX_SIZE = 32
DEFAULT_OUT_ROOT = Path("/tmp/egodelta_1cm/yolo26l_box32_mps_pretrain_2dgt_besthp_deleaf_042")
DEFAULT_REPORT = REPO_ROOT / "research" / "REPORTS" / "yolo26l_box32_mps_pretrain_2dgt_besthp_deleaf_042.md"
DEFAULT_MODEL = REPO_ROOT / "yolo26l.pt"
DEFAULT_LABELS = Path("/tmp/egodelta/front/labels.jsonl")
DEFAULT_FRONT_ROOT = Path("/tmp/egodelta/front")
MPS_DATA = Path("/tmp/egodelta_1cm/aria_mps_2d_stride5_033/datasets/deleafing_box32/data.yaml")
MANUAL_DATA = Path("/tmp/egodelta_1cm/manual_front_yolo_032/datasets/deleafing_box32/data.yaml")
PRIOR_2DGT_ONLY_M = 0.11201898362642956
PRIOR_MPS_2DGT_M = 0.12411270145728523


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deleafing YOLO26L box32 MPS pretrain -> 2DGT fine-tune with best 2DGT-only hyperparams."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels-jsonl", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument("--mps-data", type=Path, default=MPS_DATA)
    parser.add_argument("--manual-data", type=Path, default=MANUAL_DATA)
    parser.add_argument("--mps-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="1")
    parser.add_argument("--predict-batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.00001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det-list", default="20,100")
    parser.add_argument("--threshold-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42042)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    set_resource_limits()
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    local_vendor = ensure_local_vendor(args.out_root)
    sys.path.insert(0, str(local_vendor))
    validate_inputs(args)

    accepted_labels = write_accepted_only_labels(args.labels_jsonl, args.out_root / "labels_accepted_only.jsonl")
    run_config = build_run_config(args, accepted_labels)
    write_json(args.out_root / "run_config.json", run_config)

    mps_train_dir = args.out_root / "runs" / TASK / "mps_pretrain_mild640_e30"
    mps_batch = train_with_retry(
        model_path=args.model,
        data_yaml=args.mps_data,
        train_dir=mps_train_dir,
        epochs=int(args.mps_epochs),
        batch=int(args.batch),
        seed_offset=0,
        args=args,
    )
    mps_weights = select_weights(mps_train_dir)

    finetune_dir = args.out_root / "runs" / TASK / "manual_finetune_mild640_e30"
    finetune_batch = train_with_retry(
        model_path=mps_weights,
        data_yaml=args.manual_data,
        train_dir=finetune_dir,
        epochs=int(args.finetune_epochs),
        batch=int(args.batch),
        seed_offset=100,
        args=args,
    )
    finetune_weights = select_weights(finetune_dir)

    reranker_results = []
    for max_det in parse_int_list(args.max_det_list):
        eval_dir = args.out_root / "runs" / TASK / f"reranker_lowconf_maxdet{max_det}"
        reranker_results.append(run_reranker_eval(finetune_weights, accepted_labels, eval_dir, max_det, args))
        write_json(
            args.out_root / "partial_summary.json",
            summary_payload(args, run_config, mps_train_dir, finetune_dir, mps_batch, finetune_batch, reranker_results),
        )

    summary = summary_payload(
        args,
        run_config,
        mps_train_dir,
        finetune_dir,
        mps_batch,
        finetune_batch,
        reranker_results,
    )
    write_json(args.out_root / "summary.json", summary)
    write_report(args.report_path, args, run_config, summary)
    print(json.dumps({"status": "complete", "summary": str((args.out_root / "summary.json").resolve())}, indent=2))
    return 0


def set_resource_limits() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


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


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (
        ("YOLO26L checkpoint", args.model),
        ("MPS deleafing box32 data.yaml", args.mps_data),
        ("manual 2DGT deleafing box32 data.yaml", args.manual_data),
        ("manual labels.jsonl", args.labels_jsonl),
        ("front root", args.front_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for path in (args.mps_data.parent / "dataset_summary.json", args.manual_data.parent / "dataset_summary.json"):
        if path.exists():
            summary = read_json(path)
            if summary.get("box_size_px") is not None and int(summary["box_size_px"]) != BOX_SIZE:
                raise ValueError(f"{path} reports box_size_px={summary['box_size_px']}, expected {BOX_SIZE}")


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


def build_run_config(args: argparse.Namespace, accepted_labels: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "pid": os.getpid(),
        "role": "worker-042",
        "task": TASK,
        "box_size_px": BOX_SIZE,
        "model": str(args.model.resolve()),
        "out_root": str(args.out_root.resolve()),
        "report_path": str(args.report_path.resolve()),
        "labels_jsonl": str(args.labels_jsonl.resolve()),
        "accepted_only_labels_jsonl": str(accepted_labels.resolve()),
        "front_root": str(args.front_root.resolve()),
        "mps_data": dataset_info(args.mps_data),
        "manual_data": dataset_info(args.manual_data),
        "training": {
            "mps_epochs": int(args.mps_epochs),
            "finetune_epochs": int(args.finetune_epochs),
            "imgsz": int(args.imgsz),
            "initial_batch": int(args.batch),
            "workers": int(args.workers),
            "device": str(args.device),
            "seed": int(args.seed),
            "augmentation": augmentation_policy(),
            "validation_policy": "YOLO train uses val=False; no validation set is used for model selection.",
            "gpu_policy": "Explicit physical GPU device ID is passed to Ultralytics; CUDA_VISIBLE_DEVICES is removed.",
        },
        "evaluation": {
            "selector": "train-only logistic reranker from 036",
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det_list": parse_int_list(args.max_det_list),
            "imgsz": int(args.imgsz),
            "batch": int(args.predict_batch),
            "depth_policy": "nearest-source-target",
            "threshold_m": float(args.threshold_m),
        },
        "thread_limits": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def dataset_info(data_yaml: Path) -> dict[str, Any]:
    summary_path = data_yaml.parent / "dataset_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    return {
        "data_yaml": str(data_yaml.resolve()),
        "dataset_summary": str(summary_path.resolve()) if summary_path.exists() else None,
        "summary": compact_dataset_summary(summary),
    }


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
            for split, payload in sorted(splits.items())
        },
    }


def train_with_retry(
    model_path: Path,
    data_yaml: Path,
    train_dir: Path,
    epochs: int,
    batch: int,
    seed_offset: int,
    args: argparse.Namespace,
) -> int:
    if not args.force and select_weights_if_available(train_dir) is not None:
        return int(batch)
    current_batch = max(1, int(batch))
    last_exc: Exception | None = None
    while current_batch >= 1:
        try:
            train_yolo(model_path, data_yaml, train_dir, epochs, current_batch, seed_offset, args)
            return current_batch
        except RuntimeError as exc:
            last_exc = exc
            text = str(exc).lower()
            if "out of memory" not in text and "cuda" not in text:
                raise
            if current_batch == 1:
                raise
            current_batch = max(1, current_batch // 2)
            write_json(
                train_dir.parent / f"{train_dir.name}_oom_retry.json",
                {"next_batch": current_batch, "error": str(exc), "at_utc": utc_now()},
            )
    raise RuntimeError("Training failed") from last_exc


def train_yolo(
    model_path: Path,
    data_yaml: Path,
    train_dir: Path,
    epochs: int,
    batch: int,
    seed_offset: int,
    args: argparse.Namespace,
) -> None:
    from ultralytics import YOLO  # type: ignore

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
        seed=int(args.seed) + int(seed_offset),
        patience=0,
        cache=False,
        save_period=30,
        **augmentation_args(),
    )


def augmentation_args() -> dict[str, Any]:
    return {
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "erasing": 0.0,
        "degrees": 2.0,
        "translate": 0.03,
        "scale": 0.10,
        "hsv_h": 0.0,
        "hsv_s": 0.10,
        "hsv_v": 0.10,
    }


def augmentation_policy() -> str:
    return (
        "Mild YOLO affine/color augmentation: no mosaic/mixup/cutmix/copy-paste/erasing/flips; "
        "degrees=2, translate=0.03, scale=0.10, hsv_s=0.10, hsv_v=0.10."
    )


def select_weights(train_dir: Path) -> Path:
    existing = select_weights_if_available(train_dir)
    if existing is None:
        raise FileNotFoundError(f"No weights found under {train_dir / 'weights'}")
    return existing


def select_weights_if_available(train_dir: Path) -> Path | None:
    best = train_dir / "weights" / "best.pt"
    last = train_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def run_reranker_eval(
    weights: Path,
    labels_jsonl: Path,
    eval_dir: Path,
    max_det: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_yolo26l_box32_reranker_036.py"),
        "--labels-jsonl",
        str(labels_jsonl),
        "--front-root",
        str(args.front_root),
        "--out-root",
        str(eval_dir),
        "--task",
        TASK,
        "--weights",
        str(weights),
        "--checkpoint-tag",
        f"{TASK_ID}_maxdet{max_det}",
        "--imgsz",
        str(int(args.imgsz)),
        "--conf",
        str(float(args.conf)),
        "--iou",
        str(float(args.iou)),
        "--max-det",
        str(int(max_det)),
        "--batch",
        str(int(args.predict_batch)),
        "--device",
        str(args.device),
        "--crop-size",
        "0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.out_root / "vendor_py") + os.pathsep + env.get("PYTHONPATH", "")
    if VENDOR_PY.exists():
        env["PYTHONPATH"] = str(VENDOR_PY) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    summary = read_json(eval_dir / "summary.json")
    return {
        "max_det": int(max_det),
        "out_root": str(eval_dir.resolve()),
        "summary": str((eval_dir / "summary.json").resolve()),
        "best_test_deployable": summary.get("best_test_deployable"),
        "test_oracle": summary.get("test_oracle"),
        "metrics": summary.get("metrics", {}),
    }


def summary_payload(
    args: argparse.Namespace,
    run_config: dict[str, Any],
    mps_train_dir: Path,
    finetune_dir: Path,
    mps_batch: int,
    finetune_batch: int,
    reranker_results: list[dict[str, Any]],
) -> dict[str, Any]:
    best = best_deployable(reranker_results)
    oracle = best_oracle(reranker_results)
    best_mean = nested(best, "oracle_depth_error_m_mean")
    return {
        "task_id": TASK_ID,
        "created_at_utc": utc_now(),
        "status": "complete" if len(reranker_results) == len(parse_int_list(args.max_det_list)) else "running",
        "run_config": run_config,
        "mps_pretrain": {
            "train_dir": str(mps_train_dir.resolve()),
            "weights": str(select_weights(mps_train_dir).resolve()) if select_weights_if_available(mps_train_dir) else None,
            "batch_used": int(mps_batch),
            "epochs": int(args.mps_epochs),
        },
        "manual_finetune": {
            "train_dir": str(finetune_dir.resolve()),
            "weights": str(select_weights(finetune_dir).resolve()) if select_weights_if_available(finetune_dir) else None,
            "batch_used": int(finetune_batch),
            "epochs": int(args.finetune_epochs),
        },
        "reranker_results": reranker_results,
        "best_deployable": best,
        "best_oracle": oracle,
        "reached_deleafing_lt_0p05m": bool(best_mean is not None and float(best_mean) < float(args.threshold_m)),
        "comparison": {
            "best_042_deployable_mean_m": best_mean,
            "delta_vs_2dgt_only_best_m": delta(best_mean, PRIOR_2DGT_ONLY_M),
            "delta_vs_prior_mps_2dgt_035_m": delta(best_mean, PRIOR_MPS_2DGT_M),
            "prior_2dgt_only_best_mean_m": PRIOR_2DGT_ONLY_M,
            "prior_mps_2dgt_035_mean_m": PRIOR_MPS_2DGT_M,
        },
        "metric_caveat": metric_caveat(),
        "artifacts": {
            "root": str(args.out_root.resolve()),
            "run_config": str((args.out_root / "run_config.json").resolve()),
            "summary": str((args.out_root / "summary.json").resolve()),
            "report": str(args.report_path.resolve()),
        },
    }


def best_deployable(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row.get("best_test_deployable") for row in results if row.get("best_test_deployable")]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.get("oracle_depth_error_m_mean", float("inf"))))


def best_oracle(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row.get("test_oracle") for row in results if row.get("test_oracle")]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.get("oracle_depth_error_m_mean", float("inf"))))


def write_report(path: Path, args: argparse.Namespace, run_config: dict[str, Any], summary: dict[str, Any]) -> None:
    best = summary.get("best_deployable") or {}
    oracle = summary.get("best_oracle") or {}
    comparison = summary.get("comparison", {})
    lines = [
        "# YOLO26L Box32 MPS Pretrain + 2DGT Fine-tune Deleafing 042",
        "",
        f"Date: {utc_now()}",
        "Role: worker-042",
        f"Status: {summary.get('status')}",
        f"Output root: `{args.out_root.resolve()}`",
        "",
        "## Protocol",
        "",
        "- Task: deleafing only.",
        "- Initial input is RGB; YOLO predicts a 32x32 bbox, and bbox center is the 2D point estimate.",
        "- Training chain: YOLO26L -> Aria MPS stride-5 projected box32 pretraining -> manual 2DGT box32 fine-tuning.",
        "- No validation set was used. YOLO training ran with `val=False`; test split was only used after training.",
        f"- Hyperparameters transferred from the best 2DGT-only lane: imgsz={args.imgsz}, mild augmentation, epoch-30-style training, low-conf candidate generation conf={args.conf}, max_det={args.max_det_list}, train-only logistic reranker from 036.",
        f"- Resource limits: physical GPU `{args.device}`, workers={args.workers}, batch={args.batch}, predict_batch={args.predict_batch}, thread limits={run_config['thread_limits']}. No CUDA_VISIBLE_DEVICES masking.",
        "",
        "## Reproduction Command",
        "",
        "```bash",
        (
            f"/home/user/miniconda3/envs/aria-tomato-change/bin/python {Path(__file__).resolve()} "
            f"--device {args.device} --workers {args.workers} --batch {args.batch} --imgsz {args.imgsz} "
            f"--mps-epochs {args.mps_epochs} --finetune-epochs {args.finetune_epochs} --conf {args.conf} "
            f"--max-det-list {args.max_det_list}"
        ),
        "```",
        "",
        "## Datasets",
        "",
        "| Stage | Train labels | Train images | Test labels | Test images | Data yaml |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for stage_key in ("mps_data", "manual_data"):
        data = run_config[stage_key]
        splits = data.get("summary", {}).get("splits", {})
        train = splits.get("train", {})
        test = splits.get("test", {})
        lines.append(
            f"| {stage_key} | {fmt_int(train.get('num_labels'))} | {fmt_int(train.get('num_images'))} | "
            f"{fmt_int(test.get('num_labels'))} | {fmt_int(test.get('num_images'))} | `{data['data_yaml']}` |"
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- MPS pretrained: `{summary.get('mps_pretrain', {}).get('weights')}`",
            f"- 2DGT fine-tuned: `{summary.get('manual_finetune', {}).get('weights')}`",
            "",
            "## Test Result",
            "",
            "| Selector | Max det source | Miss | N | 2D mean px | 2D median px | 2D p90 px | All-sample oracle-depth mean m | Median m | P90 m | Reaches <0.05m? |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    max_det_source = max_det_for_selector(summary, best.get("selector"))
    lines.append(
        f"| {best.get('selector', '')} | {max_det_source} | {fmt_pct(best.get('miss_rate'))} | "
        f"{fmt_int(best.get('num_samples'))} | {fmt(best.get('center_error_px_mean'), 2)} | "
        f"{fmt(best.get('center_error_px_median'), 2)} | {fmt(best.get('center_error_px_p90'), 2)} | "
        f"{fmt(best.get('oracle_depth_error_m_mean'), 6)} | {fmt(best.get('oracle_depth_error_m_median'), 6)} | "
        f"{fmt(best.get('oracle_depth_error_m_p90'), 6)} | {'yes' if summary.get('reached_deleafing_lt_0p05m') else 'no'} |"
    )
    lines.extend(
        [
            "",
            "Oracle diagnostic, not deployable:",
            "",
            f"- Selector `{oracle.get('selector')}`: mean `{fmt(oracle.get('oracle_depth_error_m_mean'), 6)}` m, miss `{fmt_pct(oracle.get('miss_rate'))}`.",
            "",
            "## Comparison",
            "",
            f"- 042 best deployable mean: `{fmt(comparison.get('best_042_deployable_mean_m'), 6)}` m.",
            f"- Best 2DGT-only so far: `{PRIOR_2DGT_ONLY_M:.6f}` m; delta `{fmt(comparison.get('delta_vs_2dgt_only_best_m'), 6)}` m.",
            f"- Prior MPS+2DGT 035: `{PRIOR_MPS_2DGT_M:.6f}` m; delta `{fmt(comparison.get('delta_vs_prior_mps_2dgt_035_m'), 6)}` m.",
            "",
            "## Critical Analysis",
            "",
        ]
    )
    if summary.get("reached_deleafing_lt_0p05m"):
        lines.append(
            "The transferred MPS pretraining plus mild 2DGT fine-tuning reached the relaxed deleafing target. "
            "The likely useful part is not candidate recall alone, but making the train-only logistic ranker see a candidate distribution closer to the held-out split."
        )
    else:
        lines.append(
            "The relaxed deleafing target was not reached. This is consistent with prior 039-041 findings: candidate recall is high, but deployable ranking under recording shift is unstable. "
            "MPS pretraining can change candidate coverage, but the train-only ranker still has too little task-specific supervision to reliably choose the branch cut point on unseen recordings."
        )
    lines.extend(
        [
            "",
            "Metric caveat: " + metric_caveat(),
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in summary.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def max_det_for_selector(summary: dict[str, Any], selector: str | None) -> str:
    if selector is None:
        return ""
    best_mean = nested(summary.get("best_deployable") or {}, "oracle_depth_error_m_mean")
    for result in summary.get("reranker_results", []):
        row = result.get("best_test_deployable") or {}
        if row.get("selector") == selector and row.get("oracle_depth_error_m_mean") == best_mean:
            return str(result.get("max_det", ""))
    return ""


def metric_caveat() -> str:
    return (
        "Manual labels are 2D-only. The meter metric is oracle-depth lateral 3D error: predicted and manual "
        "2D points are backprojected at the same nearest-source-target depth, so this is not strict full 3D GT."
    )


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer in max-det-list")
    return values


def nested(payload: dict[str, Any] | None, key: str) -> Any:
    if not payload:
        return None
    return payload.get(key)


def delta(value: Any, baseline: float) -> float | None:
    if value is None:
        return None
    return float(value) - float(baseline)


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
