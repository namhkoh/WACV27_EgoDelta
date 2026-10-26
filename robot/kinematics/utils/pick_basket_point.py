from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = "a270c038"


def default_point_cloud(dataset_name: str) -> Path:
    session_root = REPO_ROOT / "output" / "policy_training" / dataset_name
    selected = _selected_alignment_cloud(dataset_name, session_root)
    if selected is not None:
        return selected

    ranked = sorted(
        (session_root / "current_realsense_alignment" / "feature_depth_search").glob(
            "rank*_frame_*/aria_cloud_aligned_to_realsense.ply"
        )
    )
    if ranked:
        return ranked[0]

    dense = session_root / f"{dataset_name}_harvest_pre_2dgs" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if dense.exists():
        return dense

    return session_root / f"{dataset_name}_harvest_pre_2dgs" / "input.ply"


def _selected_alignment_cloud(dataset_name: str, session_root: Path) -> Path | None:
    decision_paths = (
        session_root / "current_realsense_alignment" / "selected_current_alignment_decision_best_target.json",
        REPO_ROOT / "robot" / "calibration" / "session" / dataset_name / "selected_current_alignment_decision_best_target.json",
    )
    for decision_path in decision_paths:
        if not decision_path.exists():
            continue
        data = json.loads(decision_path.read_text(encoding="utf-8"))
        rank_dir = data.get("selected", {}).get("rank_dir")
        if not rank_dir:
            continue
        cloud = Path(rank_dir) / "aria_cloud_aligned_to_realsense.ply"
        if cloud.exists():
            return cloud
    return None


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        points, colors = _load_npz_points(path)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"point cloud is empty or unreadable: {path}")
    return cloud


def _load_npz_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    point_key = _first_existing_key(data.files, ("points", "xyz", "vertices", "point_cloud"))
    if point_key is None:
        raise RuntimeError(f"npz contains no supported point array: {path}")
    points = np.asarray(data[point_key], dtype=float).reshape(-1, 3)

    color_key = _first_existing_key(data.files, ("colors", "rgb"))
    if color_key is None:
        return points, None
    colors = np.asarray(data[color_key], dtype=float).reshape(-1, 3)
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    return points, np.clip(colors, 0.0, 1.0)


def _first_existing_key(keys: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(keys)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def default_transform_json(point_cloud_path: Path) -> Path | None:
    candidate = point_cloud_path.parent / "T_robot_from_aria_featuredepth_hanger.json"
    if candidate.exists():
        return candidate
    return None


def point_to_robot(point: np.ndarray, transform_json: Path | None) -> np.ndarray | None:
    if transform_json is None:
        return None
    data = json.loads(transform_json.read_text(encoding="utf-8"))
    matrix = np.asarray(data["T_robot_from_realsense"], dtype=float).reshape(4, 4)
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = np.asarray(point, dtype=float).reshape(3)
    return (matrix @ homogeneous)[:3]


def pick_point(cloud: o3d.geometry.PointCloud, *, axis_size_m: float) -> np.ndarray:
    geometries: list[o3d.geometry.Geometry] = [cloud]
    if axis_size_m > 0.0:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size_m))

    print("Open3D point picking:")
    print("  Shift + left click: pick a basket point")
    print("  Q or Esc: close the window after selecting one point")
    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name="Pick Basket Point")
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()
    picked = visualizer.get_picked_points()
    if not picked:
        raise RuntimeError("no point was picked")
    if len(picked) > 1:
        print(f"Multiple points picked; using the last one: #{picked[-1]}")

    points = np.asarray(cloud.points)
    index = int(picked[-1])
    if index < 0 or index >= len(points):
        raise RuntimeError(f"picked index out of range: {index}")
    return points[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick a basket point from the current Open3D point cloud.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--point-cloud", type=Path, default=None)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--no-robot-transform", action="store_true")
    parser.add_argument("--axis-size-m", type=float, default=0.1)
    args = parser.parse_args()

    path = args.point_cloud or default_point_cloud(args.dataset_name)
    if not path.exists():
        raise FileNotFoundError(f"point cloud not found: {path}")

    cloud = load_point_cloud(path)
    selected = pick_point(cloud, axis_size_m=args.axis_size_m)
    transform_json = None if args.no_robot_transform else args.transform_json or default_transform_json(path)
    selected_robot = point_to_robot(selected, transform_json)
    payload = {
        "point_cloud": str(path),
        "selected_point_cloud_frame": [float(value) for value in selected],
        "transform_json": None if transform_json is None else str(transform_json),
        "selected_point_robot_m": None if selected_robot is None else [float(value) for value in selected_robot],
    }
    print(json.dumps(payload, indent=2))
    if selected_robot is None:
        print("selected-point-cloud-frame " + ",".join(f"{value:.6f}" for value in selected))
    else:
        print("--basket-anchor-robot-m " + ",".join(f"{value:.6f}" for value in selected_robot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
