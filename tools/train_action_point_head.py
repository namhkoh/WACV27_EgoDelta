#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
    parser = argparse.ArgumentParser(description="Train the bbox-conditioned action-point head.")
    parser.add_argument("--manifest", action="append", required=True, help="Action-point manifest JSON. Pass once per scene.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_split_json", default=None)
    parser.add_argument("--train_scenes", default=None, help="Comma-separated scene ids. Overrides split tags for train.")
    parser.add_argument("--val_scenes", default=None, help="Comma-separated scene ids. Overrides split tags for val.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--context_pad", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--predict_depth", action="store_true")
    parser.add_argument("--depth_loss_weight", type=float, default=0.10)
    parser.add_argument("--augment", action="store_true", help="Apply crop jitter and appearance augmentation to training crops.")
    parser.add_argument("--crop_jitter_shift", type=float, default=0.08)
    parser.add_argument("--crop_jitter_scale", type=float, default=0.25)
    parser.add_argument("--color_jitter", type=float, default=0.20)
    parser.add_argument("--blur_prob", type=float, default=0.10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("Action-point head training requires torch.") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = load_action_manifests(args.manifest)
    train_scenes = split_csv(args.train_scenes)
    val_scenes = split_csv(args.val_scenes)
    train_dataset = ActionPointCropDataset(
        manifests,
        split=None if train_scenes else "train",
        scenes=train_scenes,
        scene_split_json=args.scene_split_json,
        image_size=args.image_size,
        context_pad=args.context_pad,
        augment=args.augment,
        crop_jitter_shift=args.crop_jitter_shift,
        crop_jitter_scale=args.crop_jitter_scale,
        color_jitter=args.color_jitter,
        blur_prob=args.blur_prob,
    )
    val_dataset = None
    try:
        val_dataset = ActionPointCropDataset(
            manifests,
            split=None if val_scenes else "val",
            scenes=val_scenes,
            scene_split_json=args.scene_split_json,
            image_size=args.image_size,
            context_pad=args.context_pad,
        )
    except RuntimeError:
        val_dataset = None

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ActionPointModelConfig(
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        predict_depth=bool(args.predict_depth),
        dropout=args.dropout,
    )
    model = build_action_point_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )

    best_metric = math.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_loader, None, device, args) if val_loader is not None else {}
        metric = val_metrics.get("point_l1_px_proxy", train_metrics["point_l1_px_proxy"])
        item = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(item)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.5f} train_l1={train_metrics['point_l1_px_proxy']:.4f} "
            f"val_l1={val_metrics.get('point_l1_px_proxy', float('nan')):.4f}"
        )
        state = {
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "train_args": vars(args),
            "history": history,
        }
        torch.save(state, out_dir / "last.pt")
        if metric < best_metric:
            best_metric = metric
            best_state = copy.deepcopy(state)
            torch.save(best_state, out_dir / "best.pt")

    write_json(out_dir / "training_history.json", history)
    if best_state is None:
        torch.save({"model_state": model.state_dict(), "model_config": model_config.__dict__}, out_dir / "best.pt")
    print(f"Saved checkpoints: {out_dir / 'best.pt'} and {out_dir / 'last.pt'}")
    return 0


def collate_batch(items: list[dict]) -> dict:
    import torch

    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "bbox_features": torch.stack([item["bbox_features"] for item in items], dim=0),
        "target_xy": torch.stack([item["target_xy"] for item in items], dim=0),
        "target_depth": torch.stack([item["target_depth"] for item in items], dim=0),
        "samples": [item["sample"] for item in items],
    }


def run_epoch(model, loader, optimizer, device: str, args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    if loader is None:
        return {}
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_l1 = 0.0
    total_count = 0
    for batch in loader:
        image = batch["image"].to(device)
        bbox_features = batch["bbox_features"].to(device)
        target_xy = batch["target_xy"].to(device)
        target_depth = batch["target_depth"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(image, bbox_features)
            xy_loss = F.smooth_l1_loss(pred["xy"], target_xy)
            loss = xy_loss
            if args.predict_depth and "depth" in pred:
                loss = loss + float(args.depth_loss_weight) * F.smooth_l1_loss(pred["depth"], target_depth)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_count = int(image.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_count
        total_l1 += float(torch.abs(pred["xy"].detach() - target_xy).sum(dim=1).mean().cpu()) * batch_count
        total_count += batch_count
    return {
        "loss": total_loss / max(total_count, 1),
        "point_l1_px_proxy": total_l1 / max(total_count, 1),
        "num_samples": total_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
