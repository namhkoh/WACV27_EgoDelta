#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_aria_change_baseline.config import PipelineConfig
from sam_aria_change_baseline.io_aria import load_frames_metadata
from sam_aria_change_baseline.mask_postprocess import (
    filter_and_deduplicate_frame_masks,
    load_mask,
    save_mask,
)
from sam_aria_change_baseline.sam_backend import create_sam_backend
from sam_aria_change_baseline.visualization import save_sam_overlay


TRIALS: list[dict[str, Any]] = [
    {
        "name": "01_petiole_terms",
        "prompts": ["tomato petiole", "petiole", "tomato leaf petiole", "leaf petiole"],
        "score": 0.05,
        "min_area": 10,
    },
    {
        "name": "02_leaf_stem_terms",
        "prompts": ["tomato leaf stem", "leaf stem", "tomato leaf stalk", "leaf stalk"],
        "score": 0.05,
        "min_area": 10,
    },
    {
        "name": "03_attached_petiole",
        "prompts": [
            "tomato petiole attached to stem",
            "petiole attached to main stem",
            "petiole connected to tomato stem",
            "leaf stalk attached to tomato stem",
        ],
        "score": 0.03,
        "min_area": 8,
    },
    {
        "name": "04_lateral_stem",
        "prompts": [
            "tomato lateral stem",
            "lateral stem of tomato plant",
            "side stem of tomato plant",
            "small side stem",
        ],
        "score": 0.04,
        "min_area": 10,
    },
    {
        "name": "05_leaf_branch",
        "prompts": [
            "tomato leaf branch",
            "leaf branch",
            "branch holding tomato leaves",
            "green leaf branch",
        ],
        "score": 0.04,
        "min_area": 10,
    },
    {
        "name": "06_thin_green_stems",
        "prompts": [
            "thin green stem",
            "thin green leaf stem",
            "small green stem",
            "green petiole",
        ],
        "score": 0.04,
        "min_area": 8,
    },
    {
        "name": "07_tomato_plant_leaf_stem",
        "prompts": [
            "tomato plant petiole",
            "tomato plant leaf stem",
            "tomato plant leaf stalk",
            "tomato plant side stem",
        ],
        "score": 0.04,
        "min_area": 8,
    },
    {
        "name": "08_rachis_terms",
        "prompts": [
            "tomato leaf rachis",
            "leaf rachis",
            "compound leaf rachis",
            "tomato compound leaf stem",
        ],
        "score": 0.03,
        "min_area": 8,
    },
    {
        "name": "09_broad_petiole_low_threshold",
        "prompts": [
            "tomato petiole",
            "petiole",
            "tomato leaf stem",
            "leaf stem",
            "tomato leaf stalk",
            "tomato leaf rachis",
        ],
        "score": 0.01,
        "min_area": 6,
    },
    {
        "name": "10_combined_best_candidate",
        "prompts": [
            "tomato petiole",
            "tomato leaf stem",
            "leaf stem",
            "tomato leaf stalk",
            "tomato petiole attached to stem",
            "tomato leaf rachis",
            "tomato lateral stem",
            "thin green leaf stem",
        ],
        "score": 0.02,
        "min_area": 8,
    },
]


def _load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _record_from_mask(mask: Any, mask_id: str, mask_path: Path) -> dict[str, Any]:
    save_mask(mask.mask, mask_path)
    h, w = mask.mask.shape[:2]
    return {
        "mask_id": mask_id,
        "prompt": mask.prompt,
        "semantic_group": "primary",
        "target_class": "petiole_or_leaf_stem",
        "score": float(mask.score),
        "bbox_xyxy": [float(x) for x in mask.bbox_xyxy],
        "area_px": int(mask.area_px),
        "area_frac": float(mask.area_px / max(1, h * w)),
        "mask_path": str(mask_path),
        "source": mask.source,
        "raw_class_name": mask.raw_class_name,
    }


def _make_contact_sheet(image_paths: list[Path], out_path: Path, columns: int = 4) -> None:
    if not image_paths:
        return
    thumbs: list[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((360, 360))
        canvas = Image.new("RGB", (360, 386), (0, 0, 0))
        x = (360 - img.width) // 2
        canvas.paste(img, (x, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 366), path.stem, fill=(255, 255, 255))
        thumbs.append(canvas)
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 360, rows * 386), (20, 20, 20))
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * 360
        y = (idx // columns) * 386
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _shape_rank(record: dict[str, Any]) -> float:
    x1, y1, x2, y2 = [float(x) for x in record.get("bbox_xyxy", [0, 0, 0, 0])]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    thinness = min(width, height) / max(width, height)
    area = float(record.get("area_px", 0))
    extent = area / max(1.0, width * height)
    score = float(record.get("score", 0.0))
    return score + 0.35 * (1.0 - thinness) + 0.15 * (1.0 - min(1.0, extent))


def _cap_candidates(
    records: list[dict[str, Any]],
    max_per_prompt: int,
    max_per_frame: int,
    max_area_frac: float,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if int(record.get("area_px", 0)) <= 0:
            continue
        if float(record.get("area_frac", 0.0)) > max_area_frac:
            continue
        x1, y1, x2, y2 = [float(x) for x in record.get("bbox_xyxy", [0, 0, 0, 0])]
        bbox_width = max(1.0, x2 - x1)
        bbox_height = max(1.0, y2 - y1)
        bbox_long = max(bbox_width, bbox_height)
        bbox_short = min(bbox_width, bbox_height)
        thinness = bbox_short / bbox_long
        image_long = max(float(image_width), float(image_height))
        if bbox_width > 0.30 * float(image_width) and bbox_width / bbox_height > 5.0:
            continue
        if bbox_long > 0.48 * image_long and thinness < 0.10:
            continue
        if bbox_long > 0.75 * image_long:
            continue
        by_prompt.setdefault(str(record.get("prompt", "")), []).append(record)

    selected: dict[str, dict[str, Any]] = {}
    half_cap = max(1, max_per_prompt // 2)
    for prompt_records in by_prompt.values():
        by_score = sorted(prompt_records, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        by_shape = sorted(prompt_records, key=_shape_rank, reverse=True)
        for record in by_score[:half_cap] + by_shape[: max_per_prompt - half_cap]:
            selected[str(record["mask_id"])] = record

    capped = sorted(selected.values(), key=_shape_rank, reverse=True)
    if max_per_frame > 0:
        capped = capped[:max_per_frame]
    return capped


def _bank_prompts() -> list[str]:
    prompts: list[str] = []
    for trial in TRIALS:
        for prompt in trial["prompts"]:
            if prompt not in prompts:
                prompts.append(prompt)
    return prompts


def _run_sam_bank(
    frames: list[dict[str, Any]],
    backend_name: str,
    out_dir: Path,
    reuse_bank: bool,
) -> list[dict[str, Any]]:
    bank_path = out_dir / "sam_bank_predictions.json"
    if reuse_bank and bank_path.exists():
        with bank_path.open("r") as handle:
            return json.load(handle)

    prompts = _bank_prompts()
    min_score = min(float(trial["score"]) for trial in TRIALS)
    backend = create_sam_backend(backend_name)
    raw_mask_dir = out_dir / "sam_bank_masks"
    predictions: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prompt_bank.json").open("w") as handle:
        json.dump({"prompts": prompts, "min_score": min_score, "trials": TRIALS}, handle, indent=2)

    for frame_idx, frame in enumerate(frames):
        image = _load_image(frame["image_path"])
        sam_masks = backend.predict_image(image=image, text_prompts=prompts, conf_thres=min_score)
        frame_records: list[dict[str, Any]] = []
        for mask_idx, mask in enumerate(sam_masks):
            mask_id = f"{frame['frame_id']}_bank_{mask_idx:04d}"
            mask_path = raw_mask_dir / frame["frame_id"] / f"{mask_id}.npz"
            frame_records.append(_record_from_mask(mask, mask_id, mask_path))
        predictions.append(
            {
                "frame_id": frame["frame_id"],
                "timestamp_ns": int(frame["timestamp_ns"]),
                "image_path": frame["image_path"],
                "width": int(frame.get("width", image.shape[1])),
                "height": int(frame.get("height", image.shape[0])),
                "task": "deleafing",
                "split": "pre",
                "masks": frame_records,
            }
        )
        print(
            f"SAM bank: frame {frame_idx + 1}/{len(frames)} "
            f"{frame['frame_id']} masks={len(frame_records)}"
        )

    with bank_path.open("w") as handle:
        json.dump(predictions, handle, indent=2)
    return predictions


def _trial_config(trial: dict[str, Any], max_area_frac: float) -> PipelineConfig:
    return PipelineConfig(
        task="deleafing",
        deleafing_sam_score_thres=float(trial["score"]),
        deleafing_min_mask_area_px=int(trial["min_area"]),
        max_mask_area_frac=max_area_frac,
        mask_dedup_iou_thres=0.80,
    )


def _run_trial(
    bank_predictions: list[dict[str, Any]],
    trial: dict[str, Any],
    out_dir: Path,
    max_candidates_per_prompt: int,
    max_candidates_per_frame: int,
    max_area_frac: float,
) -> dict[str, Any]:
    trial_dir = out_dir / trial["name"]
    vis_dir = trial_dir / "visualizations"
    filtered_dir = trial_dir / "filtered_masks"
    prompts = set(trial["prompts"])
    config = _trial_config(trial, max_area_frac)
    trial_predictions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "name": trial["name"],
        "prompts": trial["prompts"],
        "score_threshold": float(trial["score"]),
        "min_area_px": int(trial["min_area"]),
        "max_area_frac": float(max_area_frac),
        "num_frames": int(len(bank_predictions)),
        "num_raw_candidate_masks": 0,
        "num_capped_candidate_masks": 0,
        "num_filtered_masks": 0,
        "frames_with_masks": 0,
        "per_prompt": {},
    }
    overlay_paths: list[Path] = []

    for frame_pred in bank_predictions:
        records = [record for record in frame_pred.get("masks", []) if record.get("prompt") in prompts]
        summary["num_raw_candidate_masks"] += len(records)
        records = _cap_candidates(
            records,
            max_per_prompt=max_candidates_per_prompt,
            max_per_frame=max_candidates_per_frame,
            max_area_frac=max_area_frac,
            image_width=int(frame_pred["width"]),
            image_height=int(frame_pred["height"]),
        )
        summary["num_capped_candidate_masks"] += len(records)
        filtered, stats = filter_and_deduplicate_frame_masks(
            records,
            config=config,
            task="deleafing",
            image_shape=(int(frame_pred["height"]), int(frame_pred["width"])),
            filtered_mask_dir=filtered_dir / frame_pred["frame_id"],
        )
        for record in filtered:
            for prompt in record.get("source_prompts", [record.get("prompt", "")]):
                prompt_stats = summary["per_prompt"].setdefault(prompt, {"filtered_masks": 0})
                prompt_stats["filtered_masks"] += 1
        if filtered:
            summary["frames_with_masks"] += 1
        summary["num_filtered_masks"] += len(filtered)
        image = _load_image(frame_pred["image_path"])
        masks = [load_mask(record) for record in filtered]
        labels = [
            f"{record.get('prompt', '')} {float(record.get('score', 0.0)):.2f}"
            for record in filtered
        ]
        overlay_path = vis_dir / f"{frame_pred['frame_id']}.jpg"
        save_sam_overlay(image=image, masks=masks, labels=labels, out_path=overlay_path)
        overlay_paths.append(overlay_path)
        new_frame = dict(frame_pred)
        new_frame["masks"] = filtered
        new_frame["postprocess_stats"] = stats
        trial_predictions.append(new_frame)

    trial_dir.mkdir(parents=True, exist_ok=True)
    with (trial_dir / "pre_predictions_filtered.json").open("w") as handle:
        json.dump(trial_predictions, handle, indent=2)
    with (trial_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _make_contact_sheet(overlay_paths, trial_dir / "contact_sheet.jpg", columns=4)
    print(
        f"{trial['name']}: filtered={summary['num_filtered_masks']} "
        f"frames_with_masks={summary['frames_with_masks']}/{summary['num_frames']}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM-only deleafing prompt trials.")
    parser.add_argument("--frames-json", default="outputs/deleaf/frames/pre_frames.json")
    parser.add_argument("--out", default="outputs/deleaf_sam_trials")
    parser.add_argument("--sam-backend", default="native_sam3")
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sample-frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates-per-prompt", type=int, default=12)
    parser.add_argument("--max-candidates-per-frame", type=int, default=160)
    parser.add_argument("--trial-max-area-frac", type=float, default=0.04)
    parser.add_argument("--reuse-bank", action="store_true")
    args = parser.parse_args()

    frames = load_frames_metadata(args.frames_json)
    frames = frames[:: max(1, int(args.frame_stride))]
    if args.sample_frames is not None and args.sample_frames < len(frames):
        rng = np.random.default_rng(args.seed)
        indices = sorted(int(idx) for idx in rng.choice(len(frames), size=args.sample_frames, replace=False))
        frames = [frames[idx] for idx in indices]
    if args.frame_limit is not None:
        frames = frames[: args.frame_limit]
    if not frames:
        raise SystemExit("No frames found.")
    print("Selected frames:")
    for frame in frames:
        print(f"  {frame['frame_id']} {frame['image_path']}")

    out_dir = Path(args.out)
    bank = _run_sam_bank(frames, args.sam_backend, out_dir, args.reuse_bank)
    summaries = [
        _run_trial(
            bank,
            trial,
            out_dir,
            max_candidates_per_prompt=args.max_candidates_per_prompt,
            max_candidates_per_frame=args.max_candidates_per_frame,
            max_area_frac=args.trial_max_area_frac,
        )
        for trial in TRIALS
    ]
    summaries.sort(key=lambda item: (item["frames_with_masks"], item["num_filtered_masks"]), reverse=True)
    with (out_dir / "trial_summaries.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
    print("Top trials:")
    for item in summaries[:5]:
        print(
            f"  {item['name']}: masks={item['num_filtered_masks']} "
            f"frames={item['frames_with_masks']}/{item['num_frames']}"
        )
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
