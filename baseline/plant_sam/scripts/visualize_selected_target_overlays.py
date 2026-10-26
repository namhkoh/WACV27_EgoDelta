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
from sam_aria_change_baseline.mps_geometry import AriaMpsGeometry


DEFAULT_REQUESTS = [
    ("egodelta_17ef081b9d6c_leaf", ["000372", "000444", "000516", "000356"]),
    ("egodelta_17ef081b9d6c_tomato", ["001378", "001374", "001350", "001352"]),
    ("egodelta_64aaf8574f83_leaf", ["000510", "000540", "000506", "000534"]),
    ("egodelta_64aaf8574f83_tomato", ["001090", "001092", "001084", "001098"]),
    ("egodelta_8a957d592cfc_leaf", ["000566", "000568", "000562", "000574"]),
    ("egodelta_8a957d592cfc_tomato", ["001384", "001378", "001340", "001344"]),
    ("egodelta_dd6fab2f6fc2_leaf", ["000576", "000578", "000542", "000540"]),
    ("egodelta_dd6fab2f6fc2_tomato", ["001262", "001260", "001256", "001254"]),
]


TASK_BY_SUFFIX = {
    "leaf": "deleafing",
    "tomato": "harvesting",
}

PAIR_DIR_BY_TASK = {
    "deleafing": "leaf_pair",
    "harvesting": "tomato_pair",
}


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_scene_id(short_id: str, predictions_root: Path) -> str:
    compact_short = short_id.replace("-", "")
    candidates = []
    for path in predictions_root.iterdir():
        if not path.is_dir():
            continue
        compact = path.name.replace("-", "")
        if compact.startswith(compact_short):
            candidates.append(path.name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No prediction directory found for scene prefix {short_id!r}.")
    raise RuntimeError(f"Ambiguous scene prefix {short_id!r}: {candidates}")


def _parse_request_name(name: str) -> tuple[str, str]:
    if not name.startswith("egodelta_"):
        raise ValueError(f"Unexpected scene request name {name!r}.")
    body = name.removeprefix("egodelta_")
    short, suffix = body.rsplit("_", 1)
    task = TASK_BY_SUFFIX[suffix]
    return short, task


def _load_frames(scene_dir: Path, split: str) -> list[dict[str, Any]]:
    path = scene_dir / "frames" / f"{split}_frames.json"
    return list(_read_json(path))


def _frame_by_index(frames: list[dict[str, Any]], frame_idx_text: str) -> dict[str, Any]:
    wanted = int(frame_idx_text)
    for frame in frames:
        if int(frame.get("source_frame_idx", -1)) == wanted:
            return frame
    available = [int(frame.get("source_frame_idx", -1)) for frame in frames]
    raise FileNotFoundError(
        f"Frame {wanted:06d} is not present in this prediction output. "
        f"Available range: {min(available) if available else 'none'}..{max(available) if available else 'none'}."
    )


def _source_root(entry: dict[str, Any]) -> Path:
    return Path(entry["source_root"])


def _make_geometry(
    *,
    manifest_path: Path,
    task: str,
    split: str,
    scene_id: str,
) -> AriaMpsGeometry:
    manifest = _read_json(manifest_path)
    datasets = {str(item.get("label")): item for item in manifest.get("datasets", [])}
    entry = datasets[split]
    root = _source_root(entry)
    return AriaMpsGeometry.from_pinhole_files(
        source_root=root,
        points_source=entry.get("points_source"),
        camera_json=root / "camera_pinhole.json",
        poses_csv=root / "poses.csv",
        config=PipelineConfig(task=task),
        world_frame_id=scene_id,
    )


def _load_instances(scene_dir: Path) -> dict[str, dict[str, Any]]:
    instances = _read_json(scene_dir / "lifted_3d" / "pre_instances.json")
    return {str(item["instance_id"]): item for item in instances}


def _prediction_targets(prediction: dict[str, Any], draw_candidates: bool) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    if draw_candidates:
        for point in prediction.get("candidate_target_points_3d_world") or []:
            points.append(np.asarray(point, dtype=float))
    if not points and prediction.get("target_point_3d_world") is not None:
        points.append(np.asarray(prediction["target_point_3d_world"], dtype=float))
    return points


def _collect_points_and_targets(
    scene_dir: Path,
    draw_candidates: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    final_payload = _read_json(scene_dir / "final_predictions.json")
    predictions = list(final_payload.get("predictions", []))
    instances = _load_instances(scene_dir)
    all_points: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    warnings: list[str] = []
    for pred in predictions:
        instance_id = str(pred.get("pre_instance_id", ""))
        instance = instances.get(instance_id)
        if instance is None:
            warnings.append(f"Missing pre instance for prediction {instance_id}.")
        else:
            points_path = instance.get("points_3d_path")
            if points_path and Path(points_path).exists():
                points = np.load(points_path).reshape((-1, 3))
                if len(points):
                    all_points.append(points)
            else:
                warnings.append(f"Missing instance points for {instance_id}: {points_path}")
        targets.extend(_prediction_targets(pred, draw_candidates))
    points_array = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=float)
    return points_array, targets, warnings


def _draw_projected_points(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    *,
    radius: int,
    fill: tuple[int, int, int],
) -> None:
    for x, y in uv:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _draw_target_x(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    size: int,
) -> None:
    outline = (43, 119, 158)
    fill = (255, 255, 255)
    for color, width in ((outline, 8), (fill, 4)):
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def _render_overlay(
    *,
    image_path: Path,
    frame: dict[str, Any],
    geometry: AriaMpsGeometry,
    object_points_world: np.ndarray,
    target_points_world: list[np.ndarray],
    out_path: Path,
    point_radius: int,
    target_size: int,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    num_projected_object_points = 0
    if len(object_points_world):
        uv, _, valid = geometry.project_world_points_to_frame(
            object_points_world,
            int(frame["timestamp_ns"]),
            width,
            height,
        )
        valid_uv = uv[valid]
        num_projected_object_points = int(len(valid_uv))
        _draw_projected_points(
            draw,
            valid_uv,
            radius=point_radius,
            fill=(255, 132, 0, 230),
        )

    num_projected_targets = 0
    if target_points_world:
        target_array = np.asarray(target_points_world, dtype=float).reshape((-1, 3))
        uv, _, valid = geometry.project_world_points_to_frame(
            target_array,
            int(frame["timestamp_ns"]),
            width,
            height,
        )
        for x, y in uv[valid]:
            _draw_target_x(draw, float(x), float(y), size=target_size)
        num_projected_targets = int(np.count_nonzero(valid))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {
        "image_path": str(image_path),
        "output_path": str(out_path),
        "num_object_points_total": int(len(object_points_world)),
        "num_object_points_projected": num_projected_object_points,
        "num_targets_total": int(len(target_points_world)),
        "num_targets_projected": num_projected_targets,
    }


def render_requested_overlays(args: argparse.Namespace) -> dict[str, Any]:
    eval_root = Path(args.eval_root)
    pairs_fast_root = Path(args.pairs_fast_root)
    out_dir = Path(args.out_dir)
    summary: dict[str, Any] = {
        "eval_root": str(eval_root),
        "pairs_fast_root": str(pairs_fast_root),
        "out_dir": str(out_dir),
        "items": [],
    }

    for request_name, view_ids in DEFAULT_REQUESTS:
        short_scene_id, task = _parse_request_name(request_name)
        predictions_root = eval_root / task / "predictions"
        scene_id = _resolve_scene_id(short_scene_id, predictions_root)
        scene_dir = predictions_root / scene_id
        pair_dir = PAIR_DIR_BY_TASK[task]
        manifest_path = pairs_fast_root / scene_id / pair_dir / "manifest.json"
        frames = _load_frames(scene_dir, args.split)
        geometry = _make_geometry(
            manifest_path=manifest_path,
            task=task,
            split=args.split,
            scene_id=scene_id,
        )
        object_points, targets, warnings = _collect_points_and_targets(
            scene_dir,
            draw_candidates=args.draw_candidate_targets,
        )
        request_summary = {
            "request_name": request_name,
            "task": task,
            "scene_id": scene_id,
            "warnings": warnings,
            "views": [],
        }
        for order_idx, view_id in enumerate(view_ids):
            frame = _frame_by_index(frames, view_id)
            image_path = Path(frame["image_path"])
            out_path = (
                out_dir
                / request_name
                / f"view_{order_idx:03d}_frame_{int(view_id):06d}_target_overlay.png"
            )
            stats = _render_overlay(
                image_path=image_path,
                frame=frame,
                geometry=geometry,
                object_points_world=object_points,
                target_points_world=targets,
                out_path=out_path,
                point_radius=args.point_radius,
                target_size=args.target_size,
            )
            request_summary["views"].append(stats)
            print(
                f"{request_name} frame_{int(view_id):06d}: "
                f"points {stats['num_object_points_projected']}/{stats['num_object_points_total']}, "
                f"targets {stats['num_targets_projected']}/{stats['num_targets_total']} -> {out_path}",
                flush=True,
            )
        summary["items"].append(request_summary)

    summary_path = out_dir / "selected_overlay_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render selected paper-style overlays with disappeared-instance lifted 3D points "
            "and final target points."
        )
    )
    parser.add_argument("--eval-root", default="/tmp/plant_sam_full_eval_current")
    parser.add_argument(
        "--pairs-fast-root",
        default="/path/to/data/mps/egodelta_benchmark_data/pairs_fast",
    )
    parser.add_argument("--out-dir", default="/tmp/plant_sam_selected_target_overlays")
    parser.add_argument("--split", choices=("pre", "post"), default="pre")
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=9)
    parser.add_argument(
        "--draw-candidate-targets",
        action="store_true",
        help="For ambiguous deleafing endpoints, draw all candidate target points instead of only the selected target.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    render_requested_overlays(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
