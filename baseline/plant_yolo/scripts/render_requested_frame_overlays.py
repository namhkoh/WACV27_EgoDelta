#!/usr/bin/env python3
"""Render selected frame overlays for final target/disappearance visualizations.

The default request list is the 8 rows from the current analysis request. The
script reads the full raw evaluation tree, projects each final removed target's
3D points plus action/cut point into the requested RGB frames, and writes the
result images under /tmp.

It does not rerun inference or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_aria_harvest_poc.config import PipelineConfig  # noqa: E402
from yolo_aria_harvest_poc.mps_geometry import MPSGeometry  # noqa: E402
from yolo_aria_harvest_poc.result_projection import (  # noqa: E402
    _project_for_overlay,
    _sample_points,
)


DEFAULT_EVAL_ROOT = "/tmp/egodelta/plant_yolo/full_eval_30_tuned_yoloe26s_3cm"
DEFAULT_PAIRS_ROOT = "/path/to/data/mps/egodelta_benchmark_data/pairs_fast"
DEFAULT_OUT = "/tmp/egodelta/plant_yolo/requested_frame_visualizations"

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

TASK_BY_KIND = {
    "leaf": "deleaf",
    "tomato": "harvest",
}

PAIR_BY_TASK = {
    "deleaf": "leaf_pair",
    "harvest": "tomato_pair",
}

TASK_DIR_BY_TASK = {
    "deleaf": "deleaf",
    "harvest": "harvest",
}

FINAL_FILE_BY_TASK = {
    "deleaf": "final_predictions_deleaf.json",
    "harvest": "final_predictions.json",
}

REMOVED_KEY_BY_TASK = {
    "deleaf": "removed_petioles",
    "harvest": "removed_tomatoes",
}


def main() -> int:
    args = build_arg_parser().parse_args()
    eval_root = Path(args.eval_root)
    pairs_root = Path(args.pairs_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    requests = load_requests(args.requests_csv)
    recording_index = build_recording_index(eval_root)
    manifest: list[dict[str, Any]] = []

    for scene_name, frame_numbers in requests:
        task = task_from_scene_name(scene_name)
        uuid = resolve_recording_id(scene_name, recording_index)
        sample_root = recording_index[uuid]
        task_out = sample_root / TASK_DIR_BY_TASK[task]
        pair_manifest = pairs_root / uuid / PAIR_BY_TASK[task] / "manifest.json"
        if not pair_manifest.exists():
            raise FileNotFoundError(f"Missing pair manifest: {pair_manifest}")

        geometry = MPSGeometry.from_pair_manifest(pair_manifest, geometry_config())
        frames_by_split = load_frames_by_split(task_out)
        targets = load_removed_targets(task_out, task, max_points_per_instance=args.max_points_per_instance)

        scene_out = out_root / scene_name
        scene_out.mkdir(parents=True, exist_ok=True)
        for view_index, frame_number in enumerate(frame_numbers):
            frame, split = select_frame(frames_by_split, frame_number, preferred_split=args.split)
            output_path = scene_out / f"view_{view_index:03d}_{frame['frame_id']}_target_overlay.jpg"
            entry = render_frame_overlay(
                frame=frame,
                split=split,
                task=task,
                targets=targets,
                geometry=geometry,
                output_path=output_path,
            )
            entry.update(
                {
                    "scene": scene_name,
                    "recording_id": uuid,
                    "task": task,
                    "requested_view": f"View {view_index:03d}",
                    "requested_frame_number": frame_number,
                    "output_path": str(output_path),
                }
            )
            manifest.append(entry)

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_root), "manifest": str(manifest_path), "num_images": len(manifest)}, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--pairs-root", default=DEFAULT_PAIRS_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--requests-csv",
        help="Optional CSV with columns Scene,View 000,View 001,View 002,View 003. Defaults to the built-in list.",
    )
    parser.add_argument(
        "--split",
        choices=["pre", "post", "auto"],
        default="pre",
        help="Frame split to search first. The requested examples are pre-frame overlays.",
    )
    parser.add_argument("--max-points-per-instance", type=int, default=1600)
    return parser


def load_requests(path: str | None) -> list[tuple[str, list[str]]]:
    if path is None:
        return DEFAULT_REQUESTS
    rows: list[tuple[str, list[str]]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scene = (row.get("Scene") or "").strip()
            if not scene:
                continue
            frames = [
                (row.get(f"View {idx:03d}") or "").strip()
                for idx in range(4)
            ]
            if any(not value for value in frames):
                raise ValueError(f"CSV row has missing view frame: {row}")
            rows.append((scene, frames))
    return rows


def build_recording_index(eval_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for sample_root in sorted(eval_root.glob("shard_*/*")):
        if not sample_root.is_dir():
            continue
        uuid = sample_root.name
        if (sample_root / "harvest").is_dir() or (sample_root / "deleaf").is_dir():
            index[uuid] = sample_root
    if not index:
        raise FileNotFoundError(f"No sample outputs found under {eval_root}/shard_*")
    return index


def task_from_scene_name(scene_name: str) -> str:
    kind = scene_name.rsplit("_", 1)[-1]
    try:
        return TASK_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"Scene name must end with _leaf or _tomato: {scene_name}") from exc


def resolve_recording_id(scene_name: str, recording_index: dict[str, Path]) -> str:
    compact_prefix = scene_name.rsplit("_", 1)[0].replace("egodelta_", "").replace("-", "")
    matches = [
        uuid
        for uuid in recording_index
        if uuid.replace("-", "").startswith(compact_prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve {scene_name}; matches={matches}")
    return matches[0]


def geometry_config() -> PipelineConfig:
    return PipelineConfig(
        image_view="pinhole",
        aria_camera_label="camera-rgb",
        rect_size=720,
        rect_focal=380.0,
    )


def load_frames_by_split(task_out: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "pre": load_json(task_out / "frames" / "pre_frames.json", default=[]),
        "post": load_json(task_out / "frames" / "post_frames.json", default=[]),
    }


def select_frame(
    frames_by_split: dict[str, list[dict[str, Any]]],
    frame_number: str,
    *,
    preferred_split: str,
) -> tuple[dict[str, Any], str]:
    search_order = ["pre", "post"] if preferred_split == "auto" else [preferred_split, other_split(preferred_split)]
    wanted_suffix = f"_{frame_number}"
    for split in search_order:
        for frame in frames_by_split.get(split, []):
            frame_id = str(frame.get("frame_id", ""))
            if frame_id == frame_number or frame_id.endswith(wanted_suffix):
                return frame, split
    raise FileNotFoundError(f"Could not find frame number {frame_number} in {search_order}")


def other_split(split: str) -> str:
    return "post" if split == "pre" else "pre"


def load_removed_targets(task_out: Path, task: str, *, max_points_per_instance: int) -> list[dict[str, Any]]:
    final = load_json(task_out / FINAL_FILE_BY_TASK[task], default={REMOVED_KEY_BY_TASK[task]: []})
    instances = load_task_pre_instances(task_out, task)
    instance_by_id = {item["instance_id"]: item for item in instances}
    targets = []
    for removed in final.get(REMOVED_KEY_BY_TASK[task], []):
        instance = instance_by_id.get(removed.get("pre_instance_id"))
        if instance is None:
            continue
        points = load_instance_points(instance, task_out)
        if len(points) == 0:
            continue
        action_point = task_action_point_value(removed, task)
        if action_point is None:
            continue
        targets.append(
            {
                "instance": instance,
                "points": _sample_points(points, max_points=max_points_per_instance),
                "action_point": np.asarray(action_point, dtype=float).reshape(1, 3),
            }
        )
    return targets


def load_task_pre_instances(task_out: Path, task: str) -> list[dict[str, Any]]:
    task_path = task_out / "lifted_3d" / task / "pre_instances.json"
    if task_path.exists():
        return load_json(task_path, default=[])
    return load_json(task_out / "lifted_3d" / "pre_instances.json", default=[])


def load_instance_points(instance: dict[str, Any], task_out: Path) -> np.ndarray:
    path_value = instance.get("points_3d_path")
    if not path_value:
        return np.zeros((0, 3), dtype=float)
    path = Path(path_value)
    if not path.is_absolute():
        path = task_out / path
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    return np.asarray(np.load(path), dtype=float).reshape(-1, 3)


def task_centroid_value(removed: dict[str, Any], task: str) -> list[float] | None:
    if task == "harvest":
        return removed.get("tomato_centroid_3d_world")
    return removed.get("petiole_centroid_3d_world")


def task_action_point_value(removed: dict[str, Any], task: str) -> list[float] | None:
    if task == "harvest":
        return removed.get("cut_point_3d_world") or removed.get("proxy_cut_point_3d_world")
    return removed.get("proxy_cut_point_3d_world") or removed.get("cut_point_3d_world")


def render_frame_overlay(
    *,
    frame: dict[str, Any],
    split: str,
    task: str,
    targets: list[dict[str, Any]],
    geometry: MPSGeometry,
    output_path: Path,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required to render overlays") from exc

    image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame image: {frame['image_path']}")

    T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
    projected_targets = []
    for target in targets:
        projected_points = _project_for_overlay(target["points"], T_world_rgb, geometry)
        projected_action = _project_for_overlay(target["action_point"], T_world_rgb, geometry)
        if not (
            projected_points["valid"].any()
            or projected_action["valid"].any()
        ):
            continue
        projected_targets.append(
            {
                "instance_id": target["instance"]["instance_id"],
                "projected_points": projected_points,
                "projected_action": projected_action,
            }
        )

    if projected_targets:
        overlay = draw_target_only_overlay(
            image,
            projected_targets=projected_targets,
        )
    else:
        overlay = image.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return {
        "split": split,
        "frame_id": frame["frame_id"],
        "frame_image_path": frame["image_path"],
        "num_total_targets": len(targets),
        "num_targets_drawn": len(projected_targets),
        "instance_ids": [item["instance_id"] for item in projected_targets],
    }


def draw_target_only_overlay(
    image_bgr: np.ndarray,
    *,
    projected_targets: list[dict[str, Any]],
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    for target in projected_targets:
        projected_points = target["projected_points"]
        projected_action = target["projected_action"]

        uv_points = projected_points["uv"][projected_points["valid"]]
        if len(uv_points):
            step = max(1, len(uv_points) // 1800)
            for uv in uv_points[::step]:
                cv2.circle(overlay, pt(uv), 2, (0, 128, 255), -1, lineType=cv2.LINE_AA)

        if projected_action["valid"].any():
            action_uv = projected_action["uv"][projected_action["valid"]][0]
            cv2.drawMarker(
                overlay,
                pt(action_uv),
                (170, 80, 0),
                cv2.MARKER_TILTED_CROSS,
                24,
                8,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                overlay,
                pt(action_uv),
                (255, 255, 255),
                cv2.MARKER_TILTED_CROSS,
                18,
                4,
                cv2.LINE_AA,
            )
    return overlay


def pt(uv: np.ndarray) -> tuple[int, int]:
    return (int(round(float(uv[0]))), int(round(float(uv[1]))))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
