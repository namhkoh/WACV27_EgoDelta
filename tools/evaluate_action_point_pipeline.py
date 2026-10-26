#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_points.crop_dataset import ActionPointCropDataset
from action_points.manifest import load_action_manifests, split_csv, write_json
from action_points.models import ActionPointModelConfig, build_action_point_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the action-point head on ground-truth pseudo bboxes.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_split_json", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--context_pad", type=float, default=0.10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("Evaluation requires torch.") from exc

    manifests = load_action_manifests(args.manifest)
    dataset = ActionPointCropDataset(
        manifests,
        split=None if args.scenes else args.split,
        scenes=split_csv(args.scenes),
        scene_split_json=args.scene_split_json,
        image_size=args.image_size,
        context_pad=args.context_pad,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_action_point_model(ActionPointModelConfig(**ckpt.get("model_config", {}))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    errors_px = []
    errors_depth = []
    errors_3d_camera = []
    per_sample = []
    with torch.no_grad():
        for batch in loader:
            model_out = model(batch["image"].to(device), batch["bbox_features"].to(device))
            pred = model_out["xy"].detach().cpu()
            pred_depth = model_out.get("depth")
            if pred_depth is not None:
                pred_depth = pred_depth.detach().cpu()
            target = batch["target_xy"]
            for sample_idx, (rel_pred, rel_gt, sample) in enumerate(zip(pred, target, batch["samples"])):
                crop_box = padded_for_sample(sample, args.context_pad)
                width = max(crop_box[2] - crop_box[0], 1e-6)
                height = max(crop_box[3] - crop_box[1], 1e-6)
                pred_xy = [crop_box[0] + float(rel_pred[0]) * width, crop_box[1] + float(rel_pred[1]) * height]
                gt_xy = sample["target_2d"]
                err = math.hypot(pred_xy[0] - float(gt_xy[0]), pred_xy[1] - float(gt_xy[1]))
                errors_px.append(err)
                record = {
                    "sample_id": sample["sample_id"],
                    "image_path": sample["image_path"],
                    "scene_id": sample["scene_id"],
                    "instance_id": sample["instance_id"],
                    "predicted_2d": pred_xy,
                    "target_2d": gt_xy,
                    "error_px": err,
                }
                if pred_depth is not None and sample.get("target_depth") is not None:
                    depth = float(pred_depth[sample_idx].reshape(-1)[0])
                    target_depth = float(sample["target_depth"])
                    depth_err = abs(depth - target_depth)
                    errors_depth.append(depth_err)
                    record.update(
                        {
                            "predicted_depth": depth,
                            "target_depth": target_depth,
                            "depth_error": depth_err,
                        }
                    )
                    if sample.get("target_3d_camera") is not None:
                        pred_cam = backproject_camera(pred_xy, depth, sample["camera"])
                        target_cam = [float(v) for v in sample["target_3d_camera"]]
                        err_3d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pred_cam, target_cam)))
                        errors_3d_camera.append(err_3d)
                        record.update(
                            {
                                "predicted_3d_camera": pred_cam,
                                "target_3d_camera": target_cam,
                                "error_3d_camera": err_3d,
                            }
                        )
                per_sample.append(record)
    pixel_metrics = summarize_errors(errors_px)
    summary = {
        "mean_error_px": pixel_metrics["mean_error"],
        "median_error_px": pixel_metrics["median_error"],
        "p90_error_px": pixel_metrics["p90_error"],
    }
    if errors_depth:
        summary.update(prefix_metrics(summarize_errors(errors_depth), "depth"))
    if errors_3d_camera:
        summary.update(prefix_metrics(summarize_errors(errors_3d_camera), "camera_3d"))
    summary["num_samples"] = len(errors_px)
    summary["split"] = args.split
    out_dir = Path(args.out_dir)
    write_json(out_dir / "evaluation_summary.json", summary)
    write_json(out_dir / "evaluation_samples.json", per_sample)
    print(f"Saved evaluation: {out_dir / 'evaluation_summary.json'}")
    print(summary)
    return 0


def collate_batch(items: list[dict]) -> dict:
    import torch

    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "bbox_features": torch.stack([item["bbox_features"] for item in items], dim=0),
        "target_xy": torch.stack([item["target_xy"] for item in items], dim=0),
        "samples": [item["sample"] for item in items],
    }


def padded_for_sample(sample: dict, context_pad: float) -> list[float]:
    from action_points.crop_dataset import padded_crop_box

    return padded_crop_box(sample["bbox_xyxy"], sample["width"], sample["height"], context_pad)


def summarize_errors(errors: list[float]) -> dict:
    import numpy as np

    if not errors:
        return {"mean_error": None, "median_error": None, "p90_error": None}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "mean_error": float(arr.mean()),
        "median_error": float(np.median(arr)),
        "p90_error": float(np.percentile(arr, 90)),
    }


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def backproject_camera(point_xy: list[float], depth: float, camera: dict) -> list[float]:
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    x = (float(point_xy[0]) - cx) * depth / fx
    y = (float(point_xy[1]) - cy) * depth / fy
    return [x, y, float(depth)]


if __name__ == "__main__":
    raise SystemExit(main())
