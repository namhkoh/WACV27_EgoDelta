#!/usr/bin/env python3
"""Download the public Laboro Tomato dataset from Roboflow Universe.

Requires a Roboflow API key:

    export ROBOFLOW_API_KEY=...
    python scripts/download_laboro_tomato_dataset.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Laboro Tomato as YOLO segmentation data")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--workspace", default="jalals-lab")
    parser.add_argument("--project", default="laboro-tomato-kwpth")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument(
        "--format",
        default="yolov8",
        help="Roboflow export format. YOLOv8 segmentation format is compatible with Ultralytics YOLO11.",
    )
    parser.add_argument("--location", default="datasets/Laboro_Tomato")
    parser.add_argument("--yaml-out", default="data/laboro_tomato.yaml")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY or pass --api-key.")

    try:
        from roboflow import Roboflow  # type: ignore
    except Exception as exc:
        raise SystemExit("Install the roboflow package first: pip install roboflow") from exc

    location = Path(args.location)
    location.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)
    dataset = (
        rf.workspace(args.workspace)
        .project(args.project)
        .version(args.version)
        .download(args.format, location=str(location))
    )
    dataset_dir = Path(getattr(dataset, "location", location))
    yaml_path = find_dataset_yaml(dataset_dir)
    output_yaml = Path(args.yaml_out)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    rewrite_dataset_yaml(yaml_path, output_yaml, dataset_dir)
    print(f"Dataset downloaded to: {dataset_dir}")
    print(f"Ultralytics YAML written to: {output_yaml}")
    return 0


def find_dataset_yaml(dataset_dir: Path) -> Path:
    candidates = [dataset_dir / "data.yaml", dataset_dir / "data_config.yaml"]
    candidates.extend(sorted(dataset_dir.glob("*.yaml")))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No dataset YAML found under {dataset_dir}")


def rewrite_dataset_yaml(source_yaml: Path, output_yaml: Path, dataset_dir: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise SystemExit("Install PyYAML first: pip install pyyaml") from exc

    with source_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    data["path"] = str(dataset_dir.resolve())
    data.setdefault("train", "train/images")
    data.setdefault("val", "valid/images")
    if "test" not in data and (dataset_dir / "test" / "images").exists():
        data["test"] = "test/images"
    shutil.copy2(source_yaml, output_yaml.with_suffix(".source.yaml"))
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
