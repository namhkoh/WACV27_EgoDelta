from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = "a270c038"

def _load_fruit_center_refiner() -> Any:
    module_path = REPO_ROOT / "robot" / "kinematics" / "harvest_target_selection.py"
    spec = importlib.util.spec_from_file_location("egodelta_harvest_target_selection", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load target-selection module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.refine_red_fruit_center_surface


refine_red_fruit_center_surface = _load_fruit_center_refiner()


def default_transform_json(dataset_name: str) -> Path:
    return (
        REPO_ROOT
        / "robot"
        / "calibration"
        / "session"
        / dataset_name
        / "T_robot_from_aria_selected_current_best_target_toolfit.json"
    )


def load_capture_points(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = ("points", "colors", "uv")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(f"capture npz missing required arrays {missing}: {path}")
    result = {
        "points": np.asarray(data["points"], dtype=float).reshape(-1, 3),
        "colors": np.asarray(data["colors"], dtype=float).reshape(-1, 3),
        "uv": np.asarray(data["uv"], dtype=float).reshape(-1, 2),
    }
    for key in ("color_bgr", "depth_m"):
        if key in data.files:
            result[key] = np.asarray(data[key])
    return result


def parse_pixel_xy(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pixel must contain x,y")
    pixel = np.asarray([float(part) for part in parts], dtype=float)
    if not np.all(np.isfinite(pixel)):
        raise argparse.ArgumentTypeError("pixel must contain finite values")
    return pixel


def select_point_at_pixel(point_cloud: dict[str, np.ndarray], pixel_xy: np.ndarray) -> dict[str, Any]:
    uv = point_cloud["uv"]
    if uv.size == 0:
        raise RuntimeError("capture point cloud is empty")
    index = int(np.argmin(np.sum((uv - np.asarray(pixel_xy, dtype=float).reshape(1, 2)) ** 2, axis=1)))
    uv_selected = uv[index]
    return {
        "point_index": index,
        "pixel_xy": [int(round(float(uv_selected[0]))), int(round(float(uv_selected[1])))],
        "point_realsense_m": [float(value) for value in point_cloud["points"][index]],
    }


def pick_point(point_cloud: dict[str, np.ndarray], *, axis_size_m: float) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required only for interactive point picking") from exc

    points = point_cloud["points"]
    colors = np.clip(point_cloud["colors"], 0.0, 1.0)
    if points.size == 0:
        raise RuntimeError("capture point cloud is empty")

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    geometries: list[o3d.geometry.Geometry] = [cloud]
    if axis_size_m > 0.0:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size_m)))

    print("Open3D RealSense capture point picking:")
    print("  Shift + left click: pick one point")
    print("  Q or Esc: close after selecting")
    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name="Pick RealSense Capture Point")
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()

    picked = visualizer.get_picked_points()
    if not picked:
        raise RuntimeError("no point was picked")
    index = int(picked[-1])
    if index < 0 or index >= points.shape[0]:
        raise RuntimeError(f"picked index out of range: {index}")
    uv = point_cloud["uv"][index]
    return {
        "point_index": index,
        "pixel_xy": [int(round(float(uv[0]))), int(round(float(uv[1])))],
        "point_realsense_m": [float(value) for value in points[index]],
    }


def point_to_robot(point_realsense_m: np.ndarray, transform_json: Path) -> list[float]:
    data = json.loads(transform_json.read_text(encoding="utf-8"))
    transform = np.asarray(data["T_robot_from_realsense"], dtype=float).reshape(4, 4)
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = np.asarray(point_realsense_m, dtype=float).reshape(3)
    return [float(value) for value in (transform @ homogeneous)[:3]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick a point from a saved live RealSense capture NPZ.")
    parser.add_argument("capture_npz", type=Path)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--axis-size-m", type=float, default=0.1)
    parser.add_argument(
        "--selected-pixel-xy",
        type=parse_pixel_xy,
        default=None,
        help="Use an existing x,y click instead of reopening the Open3D picker.",
    )
    parser.add_argument(
        "--snap-red-fruit-center",
        action="store_true",
        help="Treat the click as fruit identity and snap it to that fruit's centered RGB-D surface target.",
    )
    parser.add_argument("--confirm-execute-key", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.capture_npz.exists():
        raise FileNotFoundError(f"capture npz not found: {args.capture_npz}")
    transform_json = args.transform_json or default_transform_json(args.dataset_name)
    if not transform_json.exists():
        raise FileNotFoundError(f"robot transform JSON not found: {transform_json}")

    point_cloud = load_capture_points(args.capture_npz)
    capture_metadata_path = args.capture_npz.with_suffix(".json")
    capture_metadata = (
        json.loads(capture_metadata_path.read_text(encoding="utf-8"))
        if capture_metadata_path.is_file()
        else {}
    )
    interactive_pointcloud_pick = args.selected_pixel_xy is None
    selected = (
        select_point_at_pixel(point_cloud, args.selected_pixel_xy)
        if args.selected_pixel_xy is not None
        else pick_point(point_cloud, axis_size_m=args.axis_size_m)
    )
    selection_source = "open3d_pointcloud_pick" if interactive_pointcloud_pick else "existing_pixel_pick"
    picked_point = {**selected, "selection_source": selection_source}
    if args.snap_red_fruit_center:
        selected = refine_red_fruit_center_surface(
            point_cloud,
            selected["pixel_xy"],
            intrinsics=capture_metadata.get("intrinsics"),
        )
        selected["selection_source"] = f"{selection_source}_red_fruit_center"
    else:
        selected["selection_source"] = selection_source
    selected["point_robot_m"] = point_to_robot(np.asarray(selected["point_realsense_m"], dtype=float), transform_json)
    execute_confirmed = False
    if args.confirm_execute_key:
        response = input("Press e then Enter to execute this selected point; anything else keeps dry-run: ")
        execute_confirmed = response.strip().lower() == "e"
    payload = {
        "capture_npz": str(args.capture_npz),
        "transform_json": str(transform_json),
        "execute_confirmed": execute_confirmed,
        "picked_point": picked_point,
        "selected_point": selected,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("--selected-point-robot-m " + ",".join(f"{value:.6f}" for value in selected["point_robot_m"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
