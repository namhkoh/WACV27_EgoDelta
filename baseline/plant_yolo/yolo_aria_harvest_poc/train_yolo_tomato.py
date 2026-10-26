"""Train a YOLO11 tomato segmentation checkpoint for the PoC pipeline.

This module is intentionally separate from the Aria harvesting pipeline. The
pipeline remains inference-only; this script exists because the upstream
repository currently documents a release checkpoint URL that is not published.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.resume and args.resume_checkpoint:
        parser.error("--resume and --resume-checkpoint are mutually exclusive")

    best_path = train_model(args)
    if args.copy_to:
        copy_to = Path(args.copy_to)
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, copy_to)
        print(f"Copied best checkpoint to: {copy_to}")
    print(f"Best checkpoint: {best_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 tomato segmentation on a public YOLO segmentation dataset."
    )
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML path")
    parser.add_argument(
        "--model",
        default="x",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO11 segmentation model size",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Device, e.g. 0, 1, cpu")
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--name", default="tomato_yolo11_seg")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics image cache")
    parser.add_argument("--resume", action="store_true", help="Resume the latest run in project/name")
    parser.add_argument("--resume-checkpoint", help="Resume from a specific checkpoint path")
    parser.add_argument("--copy-to", default="models/best.pt", help="Copy best.pt here after training")
    return parser


def train_model(args: argparse.Namespace) -> Path:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError("Training requires torch. Install requirements-training-cu121.txt.") from exc
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        versions = {
            "torch": installed_version("torch"),
            "torchvision": installed_version("torchvision"),
            "ultralytics": installed_version("ultralytics"),
        }
        raise RuntimeError(
            "Failed to import Ultralytics. This is usually a PyTorch/TorchVision "
            f"version mismatch. Detected versions: {versions}. "
            "Repair the environment with: bash scripts/fix_torch_training_env.sh"
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    device = normalize_device(args.device, torch)
    if args.resume_checkpoint:
        model = YOLO(str(args.resume_checkpoint))
        resume: bool | str = True
    elif args.resume:
        model = YOLO(str(Path(args.project) / args.name / "weights" / "last.pt"))
        resume = True
    else:
        model = YOLO(f"yolo11{args.model}-seg.pt")
        resume = False

    print("YOLO11 Tomato Segmentation Training")
    print(f"  data: {data_path}")
    print(f"  base model: yolo11{args.model}-seg.pt")
    print(f"  epochs: {args.epochs}")
    print(f"  batch: {args.batch}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  device: {device}")
    print(f"  output: {Path(args.project) / args.name}")

    train_args: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "save_period": args.save_period,
        "cache": bool(args.cache),
        "save": True,
        "plots": True,
        "val": True,
        "amp": True,
    }
    if resume:
        train_args["resume"] = resume

    model.train(**train_args)
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best checkpoint was not found: {best_path}")
    return best_path


def normalize_device(device_arg: str | None, torch_module: Any) -> int | str:
    if device_arg is None:
        return 0 if torch_module.cuda.is_available() else "cpu"
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def installed_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


if __name__ == "__main__":
    raise SystemExit(main())
