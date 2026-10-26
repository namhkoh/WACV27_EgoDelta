#!/usr/bin/env python3
"""Project benchmark 3D GT labels into full-recording RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_one_sample import parse_labels  # noqa: E402
from yolo_aria_harvest_poc.config import PipelineConfig  # noqa: E402
from yolo_aria_harvest_poc.io_aria import (  # noqa: E402
    create_image_processor,
    _create_provider,
    _get_image_data_by_index,
    _get_num_frames,
    _image_to_numpy,
    _timestamp_for_frame,
)
from yolo_aria_harvest_poc.mps_geometry import (  # noqa: E402
    find_mps_file,
    load_closed_loop_trajectory,
    load_rgb_calibration,
    nearest_pose_index,
    project_world_points_to_rgb,
)


TASK_SEGMENTS = {
    "deleafing": "pre_deleafing",
    "harvesting": "post_deleafing_pre_harvesting",
}


TASK_COLORS_BGR = {
    "deleafing": (0, 255, 0),
    "harvesting": (0, 80, 255),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="/path/to/data/mps/egodelta_benchmark_splitted/3d points labels.txt")
    parser.add_argument("--split-root", default="/path/to/data/mps/egodelta_benchmark_splitted")
    parser.add_argument("--full-root", default="/path/to/data/mps/egodelta_benchmark")
    parser.add_argument("--out", default="outputs/gt_label_visualizations")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--image-view", choices=["raw", "egodelta_pinhole"], default="raw")
    parser.add_argument("--aria-camera-label", default="camera-rgb")
    parser.add_argument("--rect-size", type=int, default=720)
    parser.add_argument("--rect-focal", type=float, default=380.0)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recording-id", action="append")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    labels = parse_labels(args.labels)
    recording_ids = list(labels)
    if args.recording_id:
        wanted = set(args.recording_id)
        recording_ids = [item for item in recording_ids if item in wanted]
    if args.limit is not None:
        recording_ids = recording_ids[: args.limit]

    summary: dict[str, Any] = {
        "label_file": args.labels,
        "label_frame": "full_mps_world",
        "num_recordings": len(recording_ids),
        "recordings": [],
    }
    combined_images: list[Path] = []
    for index, recording_id in enumerate(recording_ids, start=1):
        print(f"[{index}/{len(recording_ids)}] Visualizing GT for {recording_id}", flush=True)
        item = visualize_recording(
            recording_id=recording_id,
            labels=labels[recording_id],
            split_root=Path(args.split_root),
            full_root=Path(args.full_root),
            out_root=out_root,
            rgb_stream_id=args.rgb_stream_id,
            image_view=args.image_view,
            aria_camera_label=args.aria_camera_label,
            rect_size=args.rect_size,
            rect_focal=args.rect_focal,
            frame_stride=args.frame_stride,
        )
        summary["recordings"].append(item)
        if item.get("combined_overlay_path"):
            combined_images.append(Path(item["combined_overlay_path"]))

    contact_sheet = write_contact_sheet(combined_images, out_root / "gt_contact_sheet.jpg")
    if contact_sheet is not None:
        summary["contact_sheet_path"] = str(contact_sheet)
    summary_path = out_root / "gt_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "contact_sheet_path": summary.get("contact_sheet_path")}, indent=2))
    return 0


def visualize_recording(
    *,
    recording_id: str,
    labels: dict[str, list[list[float]]],
    split_root: Path,
    full_root: Path,
    out_root: Path,
    rgb_stream_id: str,
    image_view: str,
    aria_camera_label: str,
    rect_size: int,
    rect_focal: float,
    frame_stride: int,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for GT visualization.") from exc

    timing = load_json(split_root / "timing_json" / f"{recording_id}.json")
    vrs_path = Path(timing.get("vrs_path") or full_root / f"{recording_id}.vrs")
    mps_dir = Path(timing.get("mps_dir") or full_root / f"mps_{recording_id}_vrs")
    trajectory_path = find_mps_file(mps_dir, ["closed_loop_trajectory.csv"])
    trajectory = load_closed_loop_trajectory(trajectory_path)
    config = PipelineConfig(
        rgb_stream_id=rgb_stream_id,
        image_view=image_view,
        aria_camera_label=aria_camera_label,
        rect_size=rect_size,
        rect_focal=rect_focal,
    )
    provider, stream_id = _create_provider(vrs_path, rgb_stream_id)
    image_processor = create_image_processor(provider, stream_id, config)
    calibration, T_device_rgb = load_rgb_calibration(
        vrs_path,
        rgb_stream_id,
        image_view=image_view,
        aria_camera_label=aria_camera_label,
        rect_size=rect_size,
        rect_focal=rect_focal,
    )
    num_frames = _get_num_frames(provider, stream_id)
    first_timestamp_ns = _timestamp_for_frame(provider, stream_id, 0)

    sample_dir = out_root / recording_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    task_summaries: dict[str, Any] = {}
    overlay_paths: list[Path] = []
    for task, segment_name in TASK_SEGMENTS.items():
        points = np.asarray(labels.get(task, []), dtype=float).reshape(-1, 3)
        segment = segment_window(timing, segment_name)
        if len(points) == 0:
            task_summaries[task] = {"num_gt": 0, "overlay_path": None}
            continue
        selected = select_best_frame(
            provider=provider,
            stream_id=stream_id,
            num_frames=num_frames,
            first_timestamp_ns=first_timestamp_ns,
            segment=segment,
            frame_stride=frame_stride,
            points_world=points,
            trajectory=trajectory,
            T_device_rgb=T_device_rgb,
            calibration=calibration,
            image_processor=image_processor,
        )
        overlay_path = sample_dir / f"{task}_gt_overlay.jpg"
        overlay = draw_gt_overlay(
            image_bgr=selected["image_bgr"],
            task=task,
            recording_id=recording_id,
            frame_index=selected["frame_index"],
            rel_s=selected["rel_s"],
            points_world=points,
            uv=selected["uv"],
            valid=selected["valid"],
        )
        cv2.imwrite(str(overlay_path), overlay)
        overlay_paths.append(overlay_path)
        task_summaries[task] = {
            "num_gt": int(len(points)),
            "segment": segment_name,
            "segment_start_s": segment[0],
            "segment_end_s": segment[1],
            "selected_frame_index": int(selected["frame_index"]),
            "selected_frame_rel_s": float(selected["rel_s"]),
            "visible_gt": int(selected["valid"].sum()),
            "overlay_path": str(overlay_path),
            "points_world": points.tolist(),
            "projected_uv": selected["uv"].tolist(),
            "projected_valid": selected["valid"].astype(bool).tolist(),
        }

    combined_path = write_side_by_side(overlay_paths, sample_dir / "gt_overlay_combined.jpg")
    return {
        "recording_id": recording_id,
        "vrs_path": str(vrs_path),
        "mps_dir": str(mps_dir),
        "tasks": task_summaries,
        "combined_overlay_path": str(combined_path) if combined_path is not None else None,
    }


def select_best_frame(
    *,
    provider: Any,
    stream_id: Any,
    num_frames: int,
    first_timestamp_ns: int,
    segment: tuple[float, float],
    frame_stride: int,
    points_world: np.ndarray,
    trajectory: Any,
    T_device_rgb: np.ndarray,
    calibration: Any,
    image_processor: Any,
) -> dict[str, Any]:
    start_s, end_s = segment
    best: dict[str, Any] | None = None
    step = max(1, frame_stride)
    for frame_index in range(0, num_frames, step):
        timestamp_ns = _timestamp_for_frame(provider, stream_id, frame_index)
        rel_s = (timestamp_ns - first_timestamp_ns) / 1e9
        if rel_s < start_s or rel_s > end_s:
            continue
        T_world_rgb = trajectory[nearest_pose_index(trajectory, timestamp_ns)].T_world_device @ T_device_rgb
        uv, _, valid = project_world_points_to_rgb(points_world, T_world_rgb, calibration)
        score = int(valid.sum())
        if best is None or score > int(best["valid"].sum()):
            best = {
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "rel_s": rel_s,
                "uv": uv,
                "valid": valid,
            }
            if score == len(points_world):
                break
    if best is None:
        mid_s = 0.5 * (start_s + end_s)
        frame_index = min(num_frames - 1, max(0, int(round(mid_s * 20.0))))
        timestamp_ns = _timestamp_for_frame(provider, stream_id, frame_index)
        rel_s = (timestamp_ns - first_timestamp_ns) / 1e9
        T_world_rgb = trajectory[nearest_pose_index(trajectory, timestamp_ns)].T_world_device @ T_device_rgb
        uv, _, valid = project_world_points_to_rgb(points_world, T_world_rgb, calibration)
        best = {
            "frame_index": frame_index,
            "timestamp_ns": timestamp_ns,
            "rel_s": rel_s,
            "uv": uv,
            "valid": valid,
        }
    image_data, _ = _get_image_data_by_index(provider, stream_id, int(best["frame_index"]))
    image = image_processor.transform(_image_to_numpy(image_data))
    if image.ndim == 3 and image.shape[2] == 3:
        try:
            import cv2  # type: ignore

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception:
            image_bgr = image
    else:
        image_bgr = image
    best["image_bgr"] = image_bgr
    return best


def draw_gt_overlay(
    *,
    image_bgr: np.ndarray,
    task: str,
    recording_id: str,
    frame_index: int,
    rel_s: float,
    points_world: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    color = TASK_COLORS_BGR[task]
    cv2.putText(
        overlay,
        f"{recording_id} | {task} GT | frame={frame_index} t={rel_s:.2f}s visible={int(valid.sum())}/{len(points_world)}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for idx, (point, is_valid) in enumerate(zip(uv, valid)):
        label = f"{task[:1].upper()}GT{idx}"
        if is_valid:
            xy = tuple(np.round(point).astype(int).tolist())
            cv2.drawMarker(overlay, xy, color, cv2.MARKER_CROSS, 28, 3)
            cv2.circle(overlay, xy, 11, color, 2)
            cv2.putText(
                overlay,
                label,
                (xy[0] + 8, xy[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        else:
            y = 70 + idx * 24
            cv2.putText(
                overlay,
                f"{label}: not visible",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return overlay


def segment_window(timing: dict[str, Any], segment_name: str) -> tuple[float, float]:
    for segment in timing.get("segments", []):
        if segment.get("phase") == segment_name or segment.get("label") == segment_name:
            return float(segment["start_s"]), float(segment["end_s"])
    raise KeyError(f"Missing segment {segment_name}")


def write_side_by_side(paths: list[Path], out_path: Path) -> Path | None:
    if not paths:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return None
    target_h = min(image.shape[0] for image in images)
    resized = []
    for image in images:
        scale = target_h / image.shape[0]
        resized.append(cv2.resize(image, (int(round(image.shape[1] * scale)), target_h)))
    combined = cv2.hconcat(resized) if len(resized) > 1 else resized[0]
    cv2.imwrite(str(out_path), combined)
    return out_path


def write_contact_sheet(paths: list[Path], out_path: Path) -> Path | None:
    if not paths:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (640, 240))
        cv2.putText(
            image,
            path.parent.name,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(image)
    if not thumbs:
        return None
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    canvas = np.full((rows * 240, cols * 640, 3), 255, dtype=np.uint8)
    for idx, image in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        canvas[row * 240 : (row + 1) * 240, col * 640 : (col + 1) * 640] = image
    cv2.imwrite(str(out_path), canvas)
    return out_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
