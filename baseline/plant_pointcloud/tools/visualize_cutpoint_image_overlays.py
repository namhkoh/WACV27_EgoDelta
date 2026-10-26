from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SCENE_TABLE = """Scene,View 000,View 001,View 002,View 003
egodelta_17ef081b9d6c_leaf,000372,000444,000516,000356
egodelta_17ef081b9d6c_tomato,001378,001374,001350,001352
egodelta_64aaf8574f83_leaf,000510,000540,000506,000534
egodelta_64aaf8574f83_tomato,001090,001092,001084,001098
egodelta_8a957d592cfc_leaf,000566,000568,000562,000574
egodelta_8a957d592cfc_tomato,001384,001378,001340,001344
egodelta_dd6fab2f6fc2_leaf,000576,000578,000542,000540
egodelta_dd6fab2f6fc2_tomato,001262,001260,001256,001254
"""


TASK_SPECS = {
    "leaf": {
        "task": "deleaf",
        "stage": "pre_deleafing",
        "run_root_arg": "deleaf_run_root",
    },
    "tomato": {
        "task": "harvest",
        "stage": "post_deleafing_pre_harvesting",
        "run_root_arg": "harvest_run_root",
    },
}


@dataclass(frozen=True)
class SceneRequest:
    scene: str
    uuid_prefix: str
    object_kind: str
    frame_numbers: list[int]


@dataclass
class Trajectory:
    timestamp_us: np.ndarray
    translation_world_device: np.ndarray
    rotation_world_device: np.ndarray


def import_projectaria() -> Any:
    try:
        from projectaria_tools.core import calibration, data_provider
    except ImportError as exc:
        raise SystemExit(
            "projectaria_tools is required. Run with the aria environment, e.g. "
            "`conda run -n aria-tomato-change python tools/visualize_cutpoint_image_overlays.py`."
        ) from exc
    return data_provider, calibration


def parse_scene_table(path: Path | None) -> list[SceneRequest]:
    text = path.read_text(encoding="utf-8") if path is not None else DEFAULT_SCENE_TABLE
    rows = list(csv.DictReader(StringIO(text)))
    requests: list[SceneRequest] = []
    for row in rows:
        scene = row["Scene"].strip()
        if not scene.startswith("egodelta_") or "_" not in scene.removeprefix("egodelta_"):
            raise ValueError(f"Unexpected scene name: {scene}")
        prefix_and_kind = scene.removeprefix("egodelta_")
        uuid_prefix, object_kind = prefix_and_kind.rsplit("_", 1)
        if object_kind not in TASK_SPECS:
            raise ValueError(f"Unsupported scene kind in {scene}: {object_kind}")
        frame_numbers = [
            int(value)
            for key, value in row.items()
            if key.startswith("View") and value is not None and value.strip()
        ]
        requests.append(
            SceneRequest(
                scene=scene,
                uuid_prefix=uuid_prefix,
                object_kind=object_kind,
                frame_numbers=frame_numbers,
            )
        )
    return requests


def resolve_uuid(split_root: Path, compact_prefix: str) -> str:
    normalized = compact_prefix.replace("-", "")
    matches = [
        path.name
        for path in split_root.iterdir()
        if path.is_dir() and path.name.replace("-", "").startswith(normalized)
    ]
    if not matches:
        raise FileNotFoundError(f"No split VRS folder matches prefix {compact_prefix} under {split_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple split VRS folders match prefix {compact_prefix}: {matches}")
    return matches[0]


def read_ascii_ply_xyz(path: Path, max_points: int | None = None, seed: int = 13) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        n_vertices = 0
        for line in f:
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                n_vertices = int(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if n_vertices <= 0:
            return np.empty((0, 3), dtype=np.float64)
        data = np.loadtxt(f, max_rows=n_vertices)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, :3]
    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[keep]
    return xyz


def read_cutpoints(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    doc = json.loads(path.read_text(encoding="utf-8"))
    points = [
        target["cut_point_world"]
        for target in doc.get("targets", [])
        if target.get("cut_point_world") is not None
    ]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def run_dir_for(uuid: str, object_kind: str, args: argparse.Namespace) -> Path:
    spec = TASK_SPECS[object_kind]
    run_root = getattr(args, spec["run_root_arg"])
    task = spec["task"]
    return Path(run_root) / "runs" / uuid / task


def vrs_path_for(uuid: str, object_kind: str, split_root: Path) -> Path:
    spec = TASK_SPECS[object_kind]
    stage = spec["stage"]
    return split_root / uuid / stage / f"{stage}.vrs"


def full_vrs_path_for(uuid: str, full_root: Path) -> Path:
    path = full_root / f"{uuid}.vrs"
    if not path.exists():
        raise FileNotFoundError(f"Missing full VRS: {path}")
    return path


def full_mps_trajectory_path(uuid: str, full_root: Path) -> Path:
    path = full_root / f"mps_{uuid}_vrs" / "slam" / "closed_loop_trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing full MPS trajectory: {path}")
    return path


def multi_slam_stage_dir(uuid: str, vrs_path: Path, multi_slam_root: Path) -> Path:
    root = multi_slam_root / uuid
    mapping_path = root / "vrs_to_multi_slam.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    vrs_resolved = str(vrs_path)
    folder_id = mapping.get(vrs_resolved)
    if folder_id is None:
        for key, value in mapping.items():
            if Path(key).name == vrs_path.name and Path(key).parent.name == vrs_path.parent.name:
                folder_id = value
                break
    if folder_id is None:
        raise KeyError(f"{vrs_path} is not listed in {mapping_path}")
    return root / str(folder_id)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def prediction_to_full_transform(run_dir: Path) -> np.ndarray:
    gt_frame_path = run_dir / "gt_points_prediction_frame.json"
    if not gt_frame_path.exists():
        raise FileNotFoundError(
            f"Missing {gt_frame_path}. Re-run evaluation with full-VRS GT alignment so the "
            "full-to-prediction transform is saved."
        )
    doc = json.loads(gt_frame_path.read_text(encoding="utf-8"))
    frame_info = doc.get("frame_info", {})
    full_to_prediction = frame_info.get("T_target_from_source")
    if full_to_prediction is None:
        raise KeyError(f"{gt_frame_path} does not contain frame_info.T_target_from_source")
    transform = np.asarray(full_to_prediction, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(transform)


def quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_trajectory(path: Path) -> Trajectory:
    timestamps: list[float] = []
    translations: list[list[float]] = []
    rotations: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["tracking_timestamp_us"]))
            translations.append(
                [
                    float(row["tx_world_device"]),
                    float(row["ty_world_device"]),
                    float(row["tz_world_device"]),
                ]
            )
            rotations.append(
                quat_xyzw_to_matrix(
                    float(row["qx_world_device"]),
                    float(row["qy_world_device"]),
                    float(row["qz_world_device"]),
                    float(row["qw_world_device"]),
                )
            )
    if not timestamps:
        raise ValueError(f"Empty trajectory: {path}")
    return Trajectory(
        timestamp_us=np.asarray(timestamps, dtype=np.float64),
        translation_world_device=np.asarray(translations, dtype=np.float64),
        rotation_world_device=np.asarray(rotations, dtype=np.float64),
    )


def nearest_pose(trajectory: Trajectory, timestamp_us: float) -> tuple[np.ndarray, np.ndarray, float]:
    idx = int(np.searchsorted(trajectory.timestamp_us, timestamp_us))
    candidates = [max(0, idx - 1), min(len(trajectory.timestamp_us) - 1, idx)]
    best = min(candidates, key=lambda i: abs(float(trajectory.timestamp_us[i]) - timestamp_us))
    delta_us = abs(float(trajectory.timestamp_us[best]) - timestamp_us)
    return (
        trajectory.translation_world_device[best],
        trajectory.rotation_world_device[best],
        delta_us,
    )


def camera_from_device_matrix(camera_calib: Any) -> np.ndarray:
    t_device_camera = np.asarray(camera_calib.get_transform_device_camera().to_matrix(), dtype=np.float64)
    return np.linalg.inv(t_device_camera)


def make_rectified_camera(provider: Any, camera_label: str, rect_size: int, rect_focal: float, calibration: Any) -> dict:
    orig_calib = provider.get_device_calibration().get_camera_calib(camera_label)
    t_device_camera_orig = orig_calib.get_transform_device_camera()
    linear_calib = calibration.get_linear_camera_calibration(
        rect_size,
        rect_size,
        rect_focal,
        f"{camera_label}-linear",
        t_device_camera_orig,
    )
    linear_rot_calib = calibration.rotate_camera_calib_cw90deg(linear_calib)
    return {
        "orig_calib": orig_calib,
        "linear_calib": linear_calib,
        "project_calib": linear_rot_calib,
        "t_camera_device": camera_from_device_matrix(linear_rot_calib),
    }


def image_data_to_rgb(image_data: Any, bgr_to_rgb: bool = False) -> np.ndarray:
    image = np.asarray(image_data.to_numpy_array())
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if bgr_to_rgb:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def rectify_image_data(image_data: Any, camera: dict, calibration: Any, bgr_to_rgb: bool = False) -> np.ndarray:
    image = image_data_to_rgb(image_data, bgr_to_rgb=bgr_to_rgb)
    rectified_unrotated = calibration.distort_by_calibration(
        image,
        camera["linear_calib"],
        camera["orig_calib"],
    )
    return np.ascontiguousarray(np.rot90(rectified_unrotated, k=-1))


def resolve_frame_indices(provider: Any, stream_id: Any, frame_numbers: list[int]) -> dict[int, tuple[int, int, bool]]:
    n_images = provider.get_num_data(stream_id)
    needed = set(frame_numbers)
    resolved: dict[int, tuple[int, int, bool]] = {}
    for index in range(n_images):
        _, record = provider.get_image_data_by_index(stream_id, index)
        actual_frame = int(record.frame_number)
        if actual_frame in needed and actual_frame not in resolved:
            resolved[actual_frame] = (index, actual_frame, True)
            if len(resolved) == len(needed):
                return resolved

    for frame_number in frame_numbers:
        if frame_number in resolved:
            continue
        if 0 <= frame_number < n_images:
            _, record = provider.get_image_data_by_index(stream_id, frame_number)
            resolved[frame_number] = (frame_number, int(record.frame_number), False)
            continue
        raise IndexError(
            f"Frame number {frame_number} not found, and it is not a valid local index "
            f"for stream with {n_images} frames."
        )
    return resolved


def world_to_camera(
    points_world: np.ndarray,
    translation_world_device: np.ndarray,
    rotation_world_device: np.ndarray,
    t_camera_device: np.ndarray,
) -> np.ndarray:
    if len(points_world) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points_device = (rotation_world_device.T @ (points_world - translation_world_device).T).T
    points_device_h = np.concatenate([points_device, np.ones((len(points_device), 1), dtype=np.float64)], axis=1)
    points_camera_h = (t_camera_device @ points_device_h.T).T
    return points_camera_h[:, :3]


def project_points(
    camera_calib: Any,
    image_shape: tuple[int, int],
    points_camera: np.ndarray,
    min_depth_m: float,
) -> np.ndarray:
    height, width = image_shape
    projected: list[np.ndarray] = []
    for point_camera in points_camera:
        if not np.all(np.isfinite(point_camera)) or point_camera[2] <= min_depth_m:
            continue
        try:
            uv = np.asarray(camera_calib.project(point_camera), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if len(uv) != 2 or not np.all(np.isfinite(uv)):
            continue
        u, v = float(uv[0]), float(uv[1])
        if 0.0 <= u < width and 0.0 <= v < height:
            projected.append(np.array([u, v], dtype=np.float64))
    if not projected:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(projected)


def draw_overlay(
    image: np.ndarray,
    lifted_uv: np.ndarray,
    cutpoint_uv: np.ndarray,
    out_path: Path,
    point_size: float,
    point_alpha: float,
    cutpoint_size: float,
    cutpoint_linewidth: float,
) -> None:
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 160.0, height / 160.0), dpi=160)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(image)
    if len(lifted_uv):
        ax.scatter(
            lifted_uv[:, 0],
            lifted_uv[:, 1],
            s=point_size,
            c="#ff8c00",
            alpha=point_alpha,
            marker=".",
            linewidths=0,
        )
    if len(cutpoint_uv):
        ax.scatter(
            cutpoint_uv[:, 0],
            cutpoint_uv[:, 1],
            s=cutpoint_size,
            c="white",
            marker="X",
            edgecolors="#1f77b4",
            linewidths=cutpoint_linewidth,
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def render_scene(
    request: SceneRequest,
    args: argparse.Namespace,
    data_provider: Any,
    calibration: Any,
) -> list[dict[str, Any]]:
    split_root = Path(args.split_root)
    full_root = Path(args.full_root)
    multi_slam_root = Path(args.multi_slam_root)
    uuid = resolve_uuid(split_root, request.uuid_prefix)
    split_vrs_path = vrs_path_for(uuid, request.object_kind, split_root)
    full_vrs_path = full_vrs_path_for(uuid, full_root)
    run_dir = run_dir_for(uuid, request.object_kind, args)
    plys_dir = run_dir / "plys"
    lifted_points = read_ascii_ply_xyz(plys_dir / args.points_ply, max_points=args.max_lifted_points)
    cutpoints = read_cutpoints(run_dir / "targets.json")
    t_full_prediction = prediction_to_full_transform(run_dir)
    lifted_points_full = apply_transform(lifted_points, t_full_prediction)
    cutpoints_full = apply_transform(cutpoints, t_full_prediction)

    provider = data_provider.create_vrs_data_provider(str(full_vrs_path))
    stream_id = provider.get_stream_id_from_label(args.stream_label)
    if args.rectify:
        camera = make_rectified_camera(
            provider,
            args.stream_label,
            args.rect_size,
            args.rect_focal,
            calibration,
        )
        camera_calib = camera["project_calib"]
        t_camera_device = camera["t_camera_device"]
    else:
        camera = None
        camera_calib = provider.get_device_calibration().get_camera_calib(args.stream_label)
        t_camera_device = camera_from_device_matrix(camera_calib)

    stage_dir = multi_slam_stage_dir(uuid, split_vrs_path, multi_slam_root)
    trajectory = read_trajectory(full_mps_trajectory_path(uuid, full_root))
    frame_indices = resolve_frame_indices(provider, stream_id, request.frame_numbers)

    manifest_rows: list[dict[str, Any]] = []
    for view_idx, frame_number in enumerate(request.frame_numbers):
        image_index, matched_frame_number, exact_frame_match = frame_indices[frame_number]
        image_data, image_record = provider.get_image_data_by_index(stream_id, image_index)
        if args.rectify:
            image = rectify_image_data(
                image_data,
                camera,
                calibration,
                bgr_to_rgb=args.bgr_to_rgb,
            )
        else:
            image = image_data_to_rgb(image_data, bgr_to_rgb=args.bgr_to_rgb)

        timestamp_us = float(image_record.capture_timestamp_ns) / 1000.0
        t_world_device, r_world_device, pose_dt_us = nearest_pose(trajectory, timestamp_us)
        lifted_camera = world_to_camera(lifted_points_full, t_world_device, r_world_device, t_camera_device)
        cutpoint_camera = world_to_camera(cutpoints_full, t_world_device, r_world_device, t_camera_device)
        lifted_uv = project_points(camera_calib, image.shape[:2], lifted_camera, args.min_depth_m)
        cutpoint_uv = project_points(camera_calib, image.shape[:2], cutpoint_camera, args.min_depth_m)

        out_path = (
            Path(args.out_dir)
            / request.scene
            / f"{request.scene}_view{view_idx:03d}_frame{frame_number:06d}.png"
        )
        draw_overlay(
            image=image,
            lifted_uv=lifted_uv,
            cutpoint_uv=cutpoint_uv,
            out_path=out_path,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            cutpoint_size=args.cutpoint_size,
            cutpoint_linewidth=args.cutpoint_linewidth,
        )
        row = {
            "scene": request.scene,
            "uuid": uuid,
            "object_kind": request.object_kind,
            "task": TASK_SPECS[request.object_kind]["task"],
            "reference_frame": "full_vrs_mps_slam",
            "vrs_path": str(full_vrs_path),
            "split_vrs_path": str(split_vrs_path),
            "multi_slam_stage_dir": str(stage_dir),
            "run_dir": str(run_dir),
            "points_ply": args.points_ply,
            "rectified": bool(args.rectify),
            "rect_size": int(args.rect_size) if args.rectify else None,
            "rect_focal": float(args.rect_focal) if args.rectify else None,
            "view": f"View {view_idx:03d}",
            "requested_frame_number": frame_number,
            "image_index": image_index,
            "matched_frame_number": matched_frame_number,
            "exact_frame_match": exact_frame_match,
            "capture_timestamp_us": timestamp_us,
            "nearest_pose_dt_us": pose_dt_us,
            "num_lifted_3d": int(len(lifted_points_full)),
            "num_cutpoints_3d": int(len(cutpoints_full)),
            "num_lifted_projected": int(len(lifted_uv)),
            "num_cutpoints_projected": int(len(cutpoint_uv)),
            "out_path": str(out_path),
        }
        manifest_rows.append(row)
        print(
            f"{request.scene} {row['view']} frame={frame_number} "
            f"lifted={len(lifted_uv)}/{len(lifted_points_full)} "
            f"cutpoints={len(cutpoint_uv)}/{len(cutpoints_full)} "
            f"pose_dt_us={pose_dt_us:.1f} -> {out_path}"
        )
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Project 3D disappeared-object candidates and predicted cutpoints onto rectified full-VRS Aria RGB frames. "
            "Prediction points are transformed from Multi-SLAM coordinates back to full-MPS coordinates."
        )
    )
    parser.add_argument("--scene-table", type=Path, default=None, help="CSV with Scene,View 000,... columns.")
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Optional comma-separated scene names to render from the table.",
    )
    parser.add_argument("--split-root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark_splitted"))
    parser.add_argument("--full-root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark"))
    parser.add_argument("--multi-slam-root", type=Path, default=Path("/path/to/data/mps/mps_multi_slam_split"))
    parser.add_argument("--deleaf-run-root", type=Path, default=Path("/tmp/tuned_strict_full_eval_deleaf_3cm"))
    parser.add_argument("--harvest-run-root", type=Path, default=Path("/tmp/tuned_strict_full_eval_harvest_3cm"))
    parser.add_argument("--out-dir", type=Path, default=Path("/path/to/data/plant_point/image_overlays"))
    parser.add_argument(
        "--points-ply",
        type=str,
        default="target_candidates.ply",
        help="PLY under each run's plys/ directory to draw as orange lifted 3D points.",
    )
    parser.add_argument("--stream-label", type=str, default="camera-rgb")
    parser.add_argument(
        "--no-rectify",
        dest="rectify",
        action="store_false",
        help="Render overlays on the raw fisheye RGB image instead of EgoDelta-style rectified pinhole views.",
    )
    parser.set_defaults(rectify=True)
    parser.add_argument("--rect-size", type=int, default=720, help="Rectified pinhole image width/height in pixels.")
    parser.add_argument("--rect-focal", type=float, default=380.0, help="Rectified pinhole focal length in pixels.")
    parser.add_argument("--max-lifted-points", type=int, default=25000)
    parser.add_argument("--min-depth-m", type=float, default=0.02)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--point-alpha", type=float, default=0.9)
    parser.add_argument("--cutpoint-size", type=float, default=180.0)
    parser.add_argument("--cutpoint-linewidth", type=float, default=2.2)
    parser.add_argument("--bgr-to-rgb", action="store_true")
    args = parser.parse_args()

    data_provider, calibration = import_projectaria()
    requests = parse_scene_table(args.scene_table)
    if args.scenes:
        wanted = {scene.strip() for scene in args.scenes.split(",") if scene.strip()}
        requests = [request for request in requests if request.scene in wanted]
        missing = sorted(wanted - {request.scene for request in requests})
        if missing:
            raise ValueError(f"Requested scenes are not in the scene table: {missing}")
    manifest: list[dict[str, Any]] = []
    for request in requests:
        manifest.extend(render_scene(request, args, data_provider, calibration))

    manifest_path = Path(args.out_dir) / "overlay_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
