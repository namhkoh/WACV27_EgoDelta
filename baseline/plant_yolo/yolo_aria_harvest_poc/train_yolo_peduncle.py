"""Train YOLO11 peduncle pose or segmentation models."""

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
    if args.task == "pose" and args.copy_to == "models/peduncle.pt":
        args.copy_to = "models/peduncle_pose.pt"
    if args.task == "segment" and args.copy_to == "models/peduncle.pt":
        args.copy_to = "models/peduncle_seg.pt"

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
        description="Train YOLO11 peduncle pose/keypoint or segmentation model."
    )
    parser.add_argument("--task", choices=["pose", "segment"], default="pose")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML path")
    parser.add_argument("--model-size", default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--pretrained", help="Explicit pretrained .pt path to start from")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="Device, e.g. 0, 1, cpu")
    parser.add_argument("--project", default="training_runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics image cache")
    parser.add_argument("--resume", action="store_true", help="Resume from project/name/weights/last.pt")
    parser.add_argument("--resume-checkpoint", help="Resume from a specific checkpoint path")
    parser.add_argument("--copy-to", default="models/peduncle.pt", help="Copy best.pt here after training")
    return parser


def train_model(args: argparse.Namespace) -> Path:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Convert/download annotations into "
            "Ultralytics YOLO pose or segment format first."
        )
    validate_dataset_paths(data_path, args.task)

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

    run_name = args.name or f"peduncle_{args.task}_yolo11{args.model_size}"
    device = normalize_device(args.device, torch)
    model = build_model(args, run_name)

    print("YOLO11 Peduncle Training")
    print(f"  task: {args.task}")
    print(f"  data: {data_path}")
    print(f"  base model: {model}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch: {args.batch}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  device: {device}")
    print(f"  output: {Path(args.project) / run_name}")

    train_args: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": args.project,
        "name": run_name,
        "workers": args.workers,
        "patience": args.patience,
        "save_period": args.save_period,
        "cache": bool(args.cache),
        "save": True,
        "plots": True,
        "val": True,
        "amp": True,
    }
    if args.resume or args.resume_checkpoint:
        train_args["resume"] = True

    model.train(**train_args)
    best_path = Path(args.project) / run_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best checkpoint was not found: {best_path}")
    return best_path


def build_model(args: argparse.Namespace, run_name: str) -> Any:
    from ultralytics import YOLO  # type: ignore

    if args.resume_checkpoint:
        return YOLO(str(args.resume_checkpoint))
    if args.resume:
        return YOLO(str(Path(args.project) / run_name / "weights" / "last.pt"))
    if args.pretrained:
        return YOLO(str(args.pretrained))
    suffix = "pose" if args.task == "pose" else "seg"
    return YOLO(f"yolo11{args.model_size}-{suffix}.pt")


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


def validate_dataset_paths(data_path: Path, task: str) -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        return

    with data_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        return

    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    missing: list[Path] = []
    for split in ("train", "val"):
        split_value = data.get(split)
        if not split_value:
            missing.append(root / f"images/{split}")
            continue
        split_paths = split_value if isinstance(split_value, list) else [split_value]
        for item in split_paths:
            split_path = Path(str(item))
            if not split_path.is_absolute():
                split_path = root / split_path
            if not split_path.exists():
                missing.append(split_path)

    if not missing:
        return

    missing_text = "\n".join(f"  - {path}" for path in missing)
    hint = ""
    if task == "pose":
        hint = (
            "\n\nFor the WUR/4TU peduncle keypoint dataset, run:\n"
            "  python scripts/convert_wur_peduncle_to_yolo_pose.py "
            "--src datasets/wur_peduncle_keypoints/keypoints_dataset "
            "--out datasets/peduncle_pose_yolo "
            "--yaml-out data/peduncle_pose.yaml\n"
        )
    raise FileNotFoundError(
        f"Dataset YAML exists, but the configured image path(s) were not found:\n"
        f"{missing_text}{hint}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
