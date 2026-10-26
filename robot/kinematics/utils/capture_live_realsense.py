from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR
    ]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.harvest_target_selection import refine_red_fruit_center_surface


REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_DATASET = "a270c038"
DEFAULT_TASK_NAME = "live_realsense_capture"


def default_transform_json(dataset_name: str) -> Path:
    return (
        REPO_ROOT
        / "robot"
        / "calibration"
        / "session"
        / dataset_name
        / "T_robot_from_aria_selected_current_best_target_toolfit.json"
    )


def capture_realsense(
    *,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    serial: str | None,
) -> dict[str, Any]:
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    align = rs.align(rs.stream.color)

    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        frames = None
        for _ in range(max(1, int(warmup_frames))):
            frames = align.process(pipeline.wait_for_frames())
        if frames is None:
            raise RuntimeError("failed to capture RealSense frames")

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense returned incomplete depth/color frames")

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        return {
            "color_bgr": color_bgr,
            "depth_m": depth_m,
            "depth_scale": depth_scale,
            "intrinsics": {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
                "model": str(intrinsics.model),
                "coeffs": [float(value) for value in intrinsics.coeffs],
            },
            "frame_metadata": {
                "depth_frame_number": int(depth_frame.get_frame_number()),
                "color_frame_number": int(color_frame.get_frame_number()),
                "depth_timestamp_ms": float(depth_frame.get_timestamp()),
                "color_timestamp_ms": float(color_frame.get_timestamp()),
            },
        }
    finally:
        pipeline.stop()


def build_point_cloud(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
) -> dict[str, np.ndarray]:
    if stride < 1:
        raise ValueError("point cloud stride must be at least 1")
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= float(min_depth_m)) & (depth <= float(max_depth_m))
    rows, cols = np.indices(depth.shape)
    sample = valid & (rows % stride == 0) & (cols % stride == 0)

    z = depth[sample].astype(np.float64)
    u = cols[sample].astype(np.float64)
    v = rows[sample].astype(np.float64)
    x = (u - float(intrinsics["ppx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["ppy"])) * z / float(intrinsics["fy"])
    points = np.column_stack((x, y, z))
    rgb = color_bgr[:, :, ::-1].reshape(-1, 3)
    colors = rgb[sample.reshape(-1)].astype(np.float64) / 255.0
    uv = np.column_stack((u, v)).astype(np.float64)
    return {"points": points, "colors": np.clip(colors, 0.0, 1.0), "uv": uv}


def pick_pixel(color_bgr: np.ndarray) -> tuple[int, int]:
    window_name = "Live RealSense Pick"
    selected: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(int(x), int(y))]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("OpenCV point picking:")
    print("  Left click: select the visible tomato point")
    print("  Enter: accept the selected point")
    print("  Q or Esc: cancel")
    while True:
        display = color_bgr.copy()
        if selected:
            x, y = selected[0]
            cv2.drawMarker(display, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(display, (x, y), 5, (0, 255, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and selected:
            break
        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("pixel picking canceled")
    cv2.destroyWindow(window_name)
    return selected[0]


def point_from_pixel(
    *,
    pixel_xy: tuple[int, int],
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    radius_px: int,
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, Any]:
    x, y = pixel_xy
    height, width = depth_m.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"selected pixel outside image bounds: {(x, y)}")
    radius = max(0, int(radius_px))
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    patch = np.asarray(depth_m[y0:y1, x0:x1], dtype=float)
    valid = patch[np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)]
    if valid.size == 0:
        raise RuntimeError(f"no valid depth near selected pixel {(x, y)}")
    z = float(np.median(valid))
    point = np.asarray(
        [
            (float(x) - float(intrinsics["ppx"])) * z / float(intrinsics["fx"]),
            (float(y) - float(intrinsics["ppy"])) * z / float(intrinsics["fy"]),
            z,
        ],
        dtype=float,
    )
    return {
        "pixel_xy": [int(x), int(y)],
        "radius_px": int(radius),
        "valid_depth_count": int(valid.size),
        "depth_m": z,
        "point_realsense_m": [float(value) for value in point],
    }


def point_to_robot(point_realsense_m: np.ndarray, transform_json: Path | None) -> list[float] | None:
    if transform_json is None:
        return None
    if not transform_json.exists():
        raise FileNotFoundError(f"robot transform JSON not found: {transform_json}")
    data = json.loads(transform_json.read_text(encoding="utf-8"))
    transform = np.asarray(data["T_robot_from_realsense"], dtype=float).reshape(4, 4)
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = np.asarray(point_realsense_m, dtype=float).reshape(3)
    return [float(value) for value in (transform @ homogeneous)[:3]]


def write_capture_artifacts(
    *,
    dataset_name: str,
    run_label: str,
    runs_root: Path,
    capture: dict[str, Any],
    point_cloud: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, str]:
    artifact_name = f"{dataset_name}_{run_label}"
    npz_path = artifact_path(DEFAULT_TASK_NAME, artifact_name, suffix=".npz", runs_root=runs_root)
    jpg_path = artifact_path(DEFAULT_TASK_NAME, artifact_name, suffix=".jpg", runs_root=runs_root)
    json_path = artifact_path(DEFAULT_TASK_NAME, artifact_name, suffix=".json", runs_root=runs_root)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        points=point_cloud["points"],
        colors=point_cloud["colors"],
        uv=point_cloud["uv"],
        depth_m=capture["depth_m"],
        color_bgr=capture["color_bgr"],
    )
    cv2.imwrite(str(jpg_path), capture["color_bgr"])
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"npz": str(npz_path), "jpg": str(jpg_path), "json": str(json_path)}


def default_run_label() -> str:
    return "live_realsense_" + datetime.now().strftime("%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a live RealSense RGB-D point cloud for robot-frame probing.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--no-robot-transform", action="store_true")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--pointcloud-stride", type=int, default=2)
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--pick-radius-px", type=int, default=3)
    parser.add_argument(
        "--snap-red-fruit-center",
        action="store_true",
        help="Harvest mode: use the click for fruit identity and output that fruit's centered RGB-D surface target.",
    )
    parser.add_argument("--pixel-x", type=int, default=None)
    parser.add_argument("--pixel-y", type=int, default=None)
    args = parser.parse_args()
    if (args.pixel_x is None) != (args.pixel_y is None):
        parser.error("--pixel-x and --pixel-y must be provided together")
    if args.pick and args.pixel_x is not None:
        parser.error("--pick cannot be combined with --pixel-x/--pixel-y")
    if args.snap_red_fruit_center and not (args.pick or args.pixel_x is not None):
        parser.error("--snap-red-fruit-center requires --pick or --pixel-x/--pixel-y")

    started = time.time()
    capture = capture_realsense(
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        serial=args.serial,
    )
    point_cloud = build_point_cloud(
        color_bgr=capture["color_bgr"],
        depth_m=capture["depth_m"],
        intrinsics=capture["intrinsics"],
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        stride=args.pointcloud_stride,
    )
    transform_json = None if args.no_robot_transform else args.transform_json or default_transform_json(args.dataset_name)
    selected = None
    picked_point = None
    pixel_xy = None
    selection_source = None
    if args.pixel_x is not None:
        pixel_xy = (int(args.pixel_x), int(args.pixel_y))
        selection_source = "cli_pixel"
    elif args.pick:
        pixel_xy = pick_pixel(capture["color_bgr"])
        selection_source = "opencv_pick"
    if pixel_xy is not None:
        picked = point_from_pixel(
            pixel_xy=pixel_xy,
            depth_m=capture["depth_m"],
            intrinsics=capture["intrinsics"],
            radius_px=args.pick_radius_px,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        robot_point = point_to_robot(np.asarray(picked["point_realsense_m"], dtype=float), transform_json)
        picked_point = {**picked, "point_robot_m": robot_point, "selection_source": selection_source}
        selected = picked_point
        if args.snap_red_fruit_center:
            centered = refine_red_fruit_center_surface(
                {**point_cloud, "color_bgr": capture["color_bgr"]},
                picked["pixel_xy"],
                intrinsics=capture["intrinsics"],
            )
            centered_robot = point_to_robot(
                np.asarray(centered["point_realsense_m"], dtype=float),
                transform_json,
            )
            selected = {
                **centered,
                "point_robot_m": centered_robot,
                "selection_source": f"{selection_source}_red_fruit_center",
            }

    metadata = {
        "method": "live_realsense_rgbd_capture",
        "dataset_name": args.dataset_name,
        "run_label": args.run_label or default_run_label(),
        "frame": "realsense_color_optical_frame",
        "capture_seconds": float(time.time() - started),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "warmup_frames": int(args.warmup_frames),
        "min_depth_m": float(args.min_depth_m),
        "max_depth_m": float(args.max_depth_m),
        "pointcloud_stride": int(args.pointcloud_stride),
        "point_count": int(point_cloud["points"].shape[0]),
        "intrinsics": capture["intrinsics"],
        "depth_scale": float(capture["depth_scale"]),
        "frame_metadata": capture["frame_metadata"],
        "transform_json": None if transform_json is None else str(transform_json),
        "picked_point": picked_point,
        "selected_point": selected,
    }
    paths = write_capture_artifacts(
        dataset_name=args.dataset_name,
        run_label=metadata["run_label"],
        runs_root=args.runs_root,
        capture=capture,
        point_cloud=point_cloud,
        metadata=metadata,
    )
    payload = {**metadata, "artifacts": paths}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if selected and selected["point_robot_m"] is not None:
        print("--selected-point-robot-m " + ",".join(f"{value:.6f}" for value in selected["point_robot_m"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
