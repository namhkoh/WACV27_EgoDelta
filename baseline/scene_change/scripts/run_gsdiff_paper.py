#!/usr/bin/env python3
"""Run our paper-faithful GS-DIFF reimplementation on a PASLCD instance."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from egoscene.gsdiff import load_cameras, render_scalar, score_gaussian_pair
from egoscene.outputs import write_prediction, write_scene_metadata


def _latest_ply(model: Path, iteration: int) -> Path:
    path = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _pgsr_training_resolution(camera, maximum_width: int = 1600):
    """Match PGSR's default -1 resolution policy used during reconstruction."""
    if camera.width <= maximum_width:
        return camera
    scale = camera.width / maximum_width
    width = int(camera.width / scale)
    height = int(camera.height / scale)
    return replace(
        camera,
        width=width,
        height=height,
        fx=camera.fx / scale,
        fy=camera.fy / scale,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def _image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    union = true_positive + false_positive + false_negative
    return {
        "iou": 1.0 if union == 0 else true_positive / union,
        "f1": 1.0
        if 2 * true_positive + false_positive + false_negative == 0
        else 2 * true_positive / (2 * true_positive + false_positive + false_negative),
    }


def _evaluate_threshold(
    scores: dict[str, np.ndarray], gt_root: Path, threshold: float
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rows = []
    resized_masks = {}
    for name, score in sorted(scores.items()):
        gt_path = gt_root / f"{Path(name).stem}.png"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        with Image.open(gt_path) as handle:
            target = np.asarray(handle.convert("L")) > 127
        model_mask = score > threshold
        evaluated = np.asarray(
            Image.fromarray(model_mask.astype(np.uint8)).resize(
                (target.shape[1], target.shape[0]), resample=Image.Resampling.NEAREST
            )
        ).astype(bool)
        rows.append(_image_metrics(evaluated, target))
        resized_masks[name] = evaluated
    return (
        {key: float(np.mean([row[key] for row in rows])) for key in ("iou", "f1")},
        resized_masks,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=0)
    parser.add_argument("--max-primitives", type=int, default=0, help="Debug only")
    args = parser.parse_args()

    reference_model = args.root / "models" / "reference"
    inference_model = args.root / "models" / "inference"
    reference, inference, score_reference, score_inference, diagnostics = score_gaussian_pair(
        _latest_ply(reference_model, args.iteration),
        _latest_ply(inference_model, args.iteration),
        reference_model / "cameras.json",
        inference_model / "cameras.json",
        device=args.device,
        max_neighbors=args.max_neighbors or None,
        max_primitives=args.max_primitives or None,
    )

    query_cameras = [
        _pgsr_training_resolution(camera)
        for camera in load_cameras(inference_model / "cameras.json", args.device)
    ]
    scene_id = "Cantina_Instance_1"
    scores: dict[str, np.ndarray] = {}
    print(f"  rendering {len(query_cameras)} PASLCD inference views", flush=True)
    with torch.inference_mode():
        for index, camera in enumerate(query_cameras, start=1):
            reference_map = render_scalar(reference, score_reference, camera)
            inference_map = render_scalar(inference, score_inference, camera)
            score = torch.maximum(reference_map, inference_map).cpu().numpy()
            scores[camera.image_name] = score
            write_prediction(args.output, scene_id, camera.image_name, score, threshold=args.threshold)
            print(f"    [{index}/{len(query_cameras)}] {camera.image_name}", flush=True)

    fixed_metrics, evaluated_masks = _evaluate_threshold(scores, args.source / "gt_mask", args.threshold)
    eval_root = args.output / scene_id / "masks_eval_resolution"
    eval_root.mkdir(parents=True, exist_ok=True)
    for name, mask in evaluated_masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(eval_root / f"{Path(name).stem}.png")

    # This diagnostic is explicitly not a reported result: it tells us whether
    # a poor fixed-threshold score is caused by method failure or calibration.
    oracle_rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics, _ = _evaluate_threshold(scores, args.source / "gt_mask", float(threshold))
        oracle_rows.append({"threshold": float(threshold), **metrics})
    oracle = max(oracle_rows, key=lambda row: row["f1"])

    result = {
        "baseline": "gsdiff_paper_faithful_reimplementation",
        "paper": "arXiv:2605.07203v2",
        "reconstructor": "official PGSR",
        "dataset": "PASLCD/Cantina/Instance_1",
        "render_resolution_policy": "PGSR default: cap input width at 1600 pixels",
        "evaluation": "MV3DCD official per-image IoU/F1 averaging; predictions resized nearest-neighbor to GT",
        "threshold": args.threshold,
        "fixed_threshold_metrics": fixed_metrics,
        "oracle_threshold_diagnostic_not_for_reporting": oracle,
        **diagnostics,
    }
    write_scene_metadata(args.output, scene_id, result)
    (args.output / scene_id / "paper_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
