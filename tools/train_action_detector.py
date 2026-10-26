#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on generated action-point bboxes.")
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml from export_action_detector_dataset.py.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO detect checkpoint or model name.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--name", default="action_detector")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Detector training requires ultralytics. Install the plant_yolo training requirements "
            "or run in an environment that provides Ultralytics YOLO."
        ) from exc

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    model = YOLO(args.model)
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "cache": bool(args.cache),
        "task": "detect",
        "plots": True,
        "val": True,
        "resume": bool(args.resume),
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)
    print(f"Training run: {Path(args.project) / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
