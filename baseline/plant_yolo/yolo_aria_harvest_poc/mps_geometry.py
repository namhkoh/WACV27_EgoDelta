"""Aria MPS trajectory, calibration, point cloud loading, and projection."""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig


WORLD_FRAME_ERROR = (
    "Pre and post MPS are not in the same world frame. "
    "Run Aria Multi-SLAM or provide an alignment transform."
)


@dataclass
class PinholeCalibration:
    """Small calibration used by tests and as a fallback for simple cameras."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    label: str = "pinhole"

    def project(self, point_camera: np.ndarray) -> np.ndarray | None:
        z = float(point_camera[2])
        if z <= 0:
            return None
        return np.array(
            [
                self.fx * float(point_camera[0]) / z + self.cx,
                self.fy * float(point_camera[1]) / z + self.cy,
            ],
            dtype=float,
        )


@dataclass
class PoseRecord:
    timestamp_ns: int
    T_world_device: np.ndarray


@dataclass
class PointCloudStats:
    num_raw_points: int = 0
    num_filtered_points: int = 0
    filter_thresholds: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_raw_points": self.num_raw_points,
            "num_filtered_points": self.num_filtered_points,
            "filter_thresholds": self.filter_thresholds,
            "warnings": self.warnings,
        }


@dataclass
class MPSGeometry:
    trajectory: list[PoseRecord]
    points_world: np.ndarray
    rgb_camera_calibration: Any
    T_device_rgb_camera: np.ndarray
    point_cloud_stats: PointCloudStats
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_paths(
        cls,
        mps_dir: str | Path,
        vrs_path: str | Path | None,
        config: PipelineConfig,
    ) -> "MPSGeometry":
        mps_dir = Path(mps_dir)
        trajectory_path = find_mps_file(mps_dir, ["closed_loop_trajectory.csv"])
        point_path = find_mps_file(
            mps_dir,
            ["semidense_points.csv.gz", "global_points.csv.gz", "semidense_points.csv"],
        )
        trajectory = load_closed_loop_trajectory(trajectory_path)
        points_world, stats = load_global_point_cloud(
            point_path,
            inverse_distance_std_threshold=config.inv_dist_std_thres,
            distance_std_threshold=config.dist_std_thres,
        )
        calibration, T_device_rgb = load_rgb_calibration(
            vrs_path,
            config.rgb_stream_id,
            image_view=config.image_view,
            rect_size=config.rect_size,
            rect_focal=config.rect_focal,
            aria_camera_label=config.aria_camera_label,
        )
        metadata = load_mps_world_frame_metadata(mps_dir)
        return cls(
            trajectory=trajectory,
            points_world=points_world,
            rgb_camera_calibration=calibration,
            T_device_rgb_camera=T_device_rgb,
            point_cloud_stats=stats,
            metadata=metadata,
        )

    @classmethod
    def from_pair_manifest(
        cls,
        pair_manifest: str | Path,
        config: PipelineConfig | None = None,
    ) -> "MPSGeometry":
        """Load camera poses/calibration/points from an extracted pair manifest.

        The fast benchmark stores rectified pinhole frames plus COLMAP-style
        sparse files. COLMAP image poses are world-to-camera; this converts them
        to the pipeline's ``T_world_rgb`` convention.
        """

        pair_manifest = Path(pair_manifest)
        manifest = json.loads(pair_manifest.read_text(encoding="utf-8"))
        datasets = manifest.get("datasets", [])
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"Pair manifest has no datasets: {pair_manifest}")

        camera = _camera_from_pair_manifest(manifest, datasets[0])
        trajectory_by_timestamp: dict[int, PoseRecord] = {}
        for dataset in datasets:
            sparse_dir = Path(str(dataset["dataset_dir"])) / "sparse" / "0"
            for timestamp_ns, T_world_camera in _load_colmap_image_poses(sparse_dir / "images.txt").items():
                trajectory_by_timestamp[timestamp_ns] = PoseRecord(
                    timestamp_ns=int(timestamp_ns),
                    T_world_device=T_world_camera,
                )
        trajectory = sorted(trajectory_by_timestamp.values(), key=lambda item: item.timestamp_ns)
        if not trajectory:
            raise RuntimeError(f"No COLMAP image poses found in {pair_manifest}")

        point_path = _point_source_from_pair_manifest(datasets)
        points_world = _load_colmap_points(point_path)
        stats = PointCloudStats(
            num_raw_points=len(points_world),
            num_filtered_points=len(points_world),
            filter_thresholds={},
            warnings=[],
        )
        metadata = {
            "source": "pairs_fast",
            "pair_manifest": str(pair_manifest),
            "alignment": manifest.get("alignment"),
            "alignment_method": manifest.get("alignment_method"),
            "single_trajectory": manifest.get("single_trajectory"),
        }
        return cls(
            trajectory=trajectory,
            points_world=points_world,
            rgb_camera_calibration=camera,
            T_device_rgb_camera=np.eye(4, dtype=float),
            point_cloud_stats=stats,
            metadata=metadata,
        )

    def get_rgb_camera_pose_at_timestamp(self, timestamp_ns: int) -> np.ndarray:
        if not self.trajectory:
            raise RuntimeError("MPS trajectory is empty; cannot query camera pose")
        idx = nearest_pose_index(self.trajectory, timestamp_ns)
        return self.trajectory[idx].T_world_device @ self.T_device_rgb_camera


def find_mps_file(root: str | Path, candidate_names: Iterable[str]) -> Path:
    root = Path(root)
    names = set(candidate_names)
    for path in root.rglob("*"):
        if path.is_file() and path.name in names:
            return path
    raise FileNotFoundError(f"Could not find any of {sorted(names)} under {root}")


def nearest_pose_index(trajectory: list[PoseRecord], timestamp_ns: int) -> int:
    timestamps = np.array([p.timestamp_ns for p in trajectory], dtype=np.int64)
    idx = int(np.searchsorted(timestamps, int(timestamp_ns)))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    before = idx - 1
    if abs(int(timestamps[before]) - int(timestamp_ns)) <= abs(
        int(timestamps[idx]) - int(timestamp_ns)
    ):
        return before
    return idx


def project_world_points_to_rgb(
    points_world: np.ndarray,
    T_world_rgb: np.ndarray,
    rgb_camera_calibration: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into RGB pixels.

    Returns ``uv`` with shape ``(N, 2)``, camera ``depth`` as z, and a boolean
    mask requiring positive depth and image bounds.
    """

    points_world = np.asarray(points_world, dtype=float)
    if points_world.size == 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=bool),
        )
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")

    T_rgb_world = np.linalg.inv(np.asarray(T_world_rgb, dtype=float))
    homog = np.c_[points_world, np.ones(len(points_world), dtype=float)]
    points_camera = (T_rgb_world @ homog.T).T[:, :3]
    depth = points_camera[:, 2].copy()
    uv = np.full((len(points_world), 2), np.nan, dtype=float)
    positive_depth = depth > 1e-6

    valid_indices = np.where(positive_depth)[0]
    if len(valid_indices):
        uv[valid_indices] = _project_camera_points(
            points_camera[valid_indices], rgb_camera_calibration
        )

    width, height = get_calibration_image_size(rgb_camera_calibration)
    finite = np.isfinite(uv).all(axis=1)
    in_bounds = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 1] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] < height)
    )
    valid = positive_depth & in_bounds
    return uv, depth, valid


def load_closed_loop_trajectory(path: str | Path) -> list[PoseRecord]:
    path = Path(path)
    try:
        from projectaria_tools.core import mps  # type: ignore

        raw_trajectory = mps.read_closed_loop_trajectory(str(path))
        trajectory = [_pose_record_from_projectaria(item) for item in raw_trajectory]
        trajectory = [p for p in trajectory if p is not None]
        if trajectory:
            trajectory.sort(key=lambda p: p.timestamp_ns)
            return trajectory
    except Exception:
        pass

    trajectory = _load_trajectory_csv(path)
    trajectory.sort(key=lambda p: p.timestamp_ns)
    return trajectory


def load_global_point_cloud(
    path: str | Path,
    inverse_distance_std_threshold: float = 0.005,
    distance_std_threshold: float = 0.01,
) -> tuple[np.ndarray, PointCloudStats]:
    path = Path(path)
    stats = PointCloudStats(
        filter_thresholds={
            "inverse_distance_std_threshold": inverse_distance_std_threshold,
            "distance_std_threshold": distance_std_threshold,
        }
    )

    records: list[Any] | None = None
    try:
        from projectaria_tools.core import mps  # type: ignore

        records = list(mps.read_global_point_cloud(str(path)))
        stats.num_raw_points = len(records)
        filtered_records = _try_projectaria_confidence_filter(
            records, inverse_distance_std_threshold, distance_std_threshold
        )
        if filtered_records is not None:
            records = filtered_records
        points, manual_warning = _points_from_projectaria_records(
            records,
            inverse_distance_std_threshold=inverse_distance_std_threshold,
            distance_std_threshold=distance_std_threshold,
        )
        if manual_warning:
            stats.warnings.append(manual_warning)
        stats.num_filtered_points = len(points)
        return points, stats
    except Exception:
        records = None

    points, raw_count, warning = _load_points_csv(
        path,
        inverse_distance_std_threshold=inverse_distance_std_threshold,
        distance_std_threshold=distance_std_threshold,
    )
    stats.num_raw_points = raw_count
    stats.num_filtered_points = len(points)
    if warning:
        stats.warnings.append(warning)
    return points, stats


def _camera_from_pair_manifest(manifest: dict[str, Any], dataset: dict[str, Any]) -> PinholeCalibration:
    camera = dataset.get("camera") if isinstance(dataset.get("camera"), dict) else None
    if camera is None:
        cameras_path = Path(str(dataset["dataset_dir"])) / "sparse" / "0" / "cameras.txt"
        camera = _load_colmap_camera(cameras_path)
    return PinholeCalibration(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=float(camera["fx"]),
        fy=float(camera.get("fy", camera["fx"])),
        cx=float(camera["cx"]),
        cy=float(camera["cy"]),
        label=str(manifest.get("camera_label") or "pairs_fast_pinhole"),
    )


def _load_colmap_camera(path: Path) -> dict[str, float | int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        model = parts[1].upper()
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        if model == "PINHOLE" and len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        elif len(params) >= 3:
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Unsupported COLMAP camera line in {path}: {line}")
        return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    raise RuntimeError(f"No camera found in {path}")


def _load_colmap_image_poses(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        name = parts[9]
        if not _looks_like_image_name(name):
            continue
        qw, qx, qy, qz = [float(v) for v in parts[1:5]]
        tx, ty, tz = [float(v) for v in parts[5:8]]
        R_world_to_camera = _qvec_to_rotmat(qw, qx, qy, qz)
        t_world_to_camera = np.array([tx, ty, tz], dtype=float)
        T_camera_world = np.eye(4, dtype=float)
        T_camera_world[:3, :3] = R_world_to_camera
        T_camera_world[:3, 3] = t_world_to_camera
        T_world_camera = np.linalg.inv(T_camera_world)
        poses[_timestamp_from_image_name(name, int(parts[0]))] = T_world_camera
    return poses


def _looks_like_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _timestamp_from_image_name(name: str, fallback: int) -> int:
    stem = Path(name).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1])
    return int(fallback)


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0:
        raise ValueError("Zero-norm quaternion in COLMAP images.txt")
    qw, qx, qy, qz = (q / norm).tolist()
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=float,
    )


def _point_source_from_pair_manifest(datasets: list[dict[str, Any]]) -> Path:
    for dataset in datasets:
        point_source = dataset.get("points_source")
        if point_source:
            path = Path(str(point_source))
            if path.exists():
                return path
    for dataset in datasets:
        path = Path(str(dataset["dataset_dir"])) / "sparse" / "0" / "points3D.ply"
        if path.exists():
            return path
    raise FileNotFoundError("Could not find points_source or sparse/0/points3D.ply in pair manifest")


def _load_colmap_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_colmap_points_txt(path)
    if suffix == ".ply":
        return _load_points_ply(path)
    raise ValueError(f"Unsupported extracted point cloud format: {path}")


def _load_colmap_points_txt(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not points:
        raise RuntimeError(f"No 3D points found in {path}")
    return np.asarray(points, dtype=np.float32)


def _load_points_ply(path: Path) -> np.ndarray:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"open3d is required to read PLY point clouds: {path}") from exc
    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError(f"No 3D points found in {path}")
    return points.reshape(-1, 3)


def load_rgb_calibration(
    vrs_path: str | Path | None,
    rgb_stream_id: str,
    image_view: str = "raw",
    rect_size: int = 720,
    rect_focal: float = 380.0,
    aria_camera_label: str = "camera-rgb",
) -> tuple[Any, np.ndarray]:
    if vrs_path is None:
        raise ValueError("A VRS path is required to load the RGB camera calibration")
    try:
        from projectaria_tools.core import calibration as aria_calibration  # type: ignore
        from projectaria_tools.core import data_provider  # type: ignore
        from projectaria_tools.core.stream_id import StreamId  # type: ignore

        provider = data_provider.create_vrs_data_provider(str(vrs_path))
        stream_id = StreamId(rgb_stream_id)
        device_calib = provider.get_device_calibration()
        labels_to_try = _camera_labels_to_try(
            provider,
            device_calib,
            stream_id,
            rgb_stream_id,
            preferred_label=aria_camera_label,
        )
        camera_calib = None
        camera_label = None
        for label in labels_to_try:
            try:
                camera_calib = device_calib.get_camera_calib(label)
            except Exception:
                camera_calib = None
            if camera_calib is not None:
                camera_label = label
                break
        if camera_calib is None:
            raise RuntimeError(f"Could not find RGB camera calibration for stream {rgb_stream_id}")
        if image_view == "egodelta_pinhole":
            T_device_camera_orig = camera_calib.get_transform_device_camera()
            linear = aria_calibration.get_linear_camera_calibration(
                int(rect_size),
                int(rect_size),
                float(rect_focal),
                f"{camera_label}-linear",
                T_device_camera_orig,
            )
            linear_rot_calib = aria_calibration.rotate_camera_calib_cw90deg(linear)
            T_device_rgb = _transform_to_matrix(linear_rot_calib.get_transform_device_camera())
            return linear_rot_calib, T_device_rgb
        if image_view != "raw":
            raise ValueError("--image-view must be one of: raw, egodelta_pinhole")
        T_device_rgb = _transform_to_matrix(camera_calib.get_transform_device_camera())
        return camera_calib, T_device_rgb
    except Exception as exc:
        raise RuntimeError(
            "Failed to load RGB camera calibration from VRS with projectaria-tools. "
            "Install projectaria-tools and verify --rgb-stream-id."
        ) from exc


def load_mps_world_frame_metadata(mps_dir: str | Path) -> dict[str, Any]:
    """Best-effort metadata scan for graph/session/world-frame identifiers."""

    root = Path(mps_dir)
    interesting_keys = {
        "graph_uid",
        "graph_id",
        "session_uid",
        "session_id",
        "world_frame",
        "world_frame_uid",
        "map_uid",
        "map_id",
        "multi_slam_uid",
    }
    metadata: dict[str, Any] = {}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt", ".csv"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                _collect_metadata_keys(data, interesting_keys, metadata)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines()[:100]:
                    for key in interesting_keys:
                        if key in line:
                            metadata.setdefault(key, line.strip())
        except Exception:
            continue
    return metadata


def assert_same_world_frame(pre_metadata: dict[str, Any], post_metadata: dict[str, Any]) -> list[str]:
    """Raise on incompatible metadata and return warnings for unverifiable cases."""

    comparable_keys = sorted(set(pre_metadata).intersection(post_metadata))
    checked = []
    for key in comparable_keys:
        pre_value = pre_metadata[key]
        post_value = post_metadata[key]
        if pre_value is None or post_value is None:
            continue
        checked.append(key)
        if str(pre_value) != str(post_value):
            raise RuntimeError(WORLD_FRAME_ERROR)
    if not checked:
        return [
            "Could not find comparable MPS world-frame metadata; assuming pre/post "
            "MPS are already aligned as required by separate_recordings mode."
        ]
    return []


def get_calibration_image_size(calibration: Any) -> tuple[int, int]:
    for width_name, height_name in [
        ("width", "height"),
        ("image_width", "image_height"),
        ("w", "h"),
    ]:
        width = getattr(calibration, width_name, None)
        height = getattr(calibration, height_name, None)
        if width is not None and height is not None:
            return int(width), int(height)

    for method_name in ["get_image_size", "image_size"]:
        method = getattr(calibration, method_name, None)
        if callable(method):
            size = method()
            if len(size) >= 2:
                return int(size[0]), int(size[1])

    projection_params = getattr(calibration, "projection_params", None)
    if projection_params is not None and len(projection_params) >= 6:
        cx, cy = float(projection_params[2]), float(projection_params[3])
        return int(round(cx * 2)), int(round(cy * 2))

    return 1408, 1408


def get_calibration_focal_estimate(calibration: Any) -> float:
    for name in ["fx", "focal_length", "focal"]:
        value = getattr(calibration, name, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    if hasattr(calibration, "fx") and hasattr(calibration, "fy"):
        return float((calibration.fx + calibration.fy) * 0.5)
    method = getattr(calibration, "get_focal_lengths", None)
    if callable(method):
        values = method()
        if len(values) >= 2:
            return float((values[0] + values[1]) * 0.5)
    method = getattr(calibration, "get_projection_params", None)
    if callable(method):
        params = method()
        if len(params) >= 1:
            return float(params[0])
    return 700.0


def _project_camera_points(points_camera: np.ndarray, calibration: Any) -> np.ndarray:
    pinhole = _pinhole_projection_params(calibration)
    if pinhole is not None:
        fx, fy, cx, cy = pinhole
        z = points_camera[:, 2]
        uv = np.empty((len(points_camera), 2), dtype=float)
        uv[:, 0] = fx * points_camera[:, 0] / z + cx
        uv[:, 1] = fy * points_camera[:, 1] / z + cy
        return uv

    if callable(getattr(calibration, "project", None)):
        out = np.full((len(points_camera), 2), np.nan, dtype=float)
        for i, point in enumerate(points_camera):
            try:
                uv = calibration.project(point)
            except Exception:
                uv = None
            if uv is not None:
                uv_array = np.asarray(uv, dtype=float).reshape(-1)
                if len(uv_array) >= 2:
                    out[i] = uv_array[:2]
        return out

    fx = float(getattr(calibration, "fx", get_calibration_focal_estimate(calibration)))
    fy = float(getattr(calibration, "fy", fx))
    width, height = get_calibration_image_size(calibration)
    cx = float(getattr(calibration, "cx", width / 2.0))
    cy = float(getattr(calibration, "cy", height / 2.0))
    z = points_camera[:, 2]
    uv = np.empty((len(points_camera), 2), dtype=float)
    uv[:, 0] = fx * points_camera[:, 0] / z + cx
    uv[:, 1] = fy * points_camera[:, 1] / z + cy
    return uv


def _pinhole_projection_params(calibration: Any) -> tuple[float, float, float, float] | None:
    if isinstance(calibration, PinholeCalibration):
        return float(calibration.fx), float(calibration.fy), float(calibration.cx), float(calibration.cy)
    if all(hasattr(calibration, name) for name in ("fx", "fy", "cx", "cy")):
        return (
            float(getattr(calibration, "fx")),
            float(getattr(calibration, "fy")),
            float(getattr(calibration, "cx")),
            float(getattr(calibration, "cy")),
        )
    method = getattr(calibration, "get_projection_params", None)
    if callable(method):
        try:
            params = [float(v) for v in method()]
        except Exception:
            params = []
        if len(params) == 4:
            return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    projection_params = getattr(calibration, "projection_params", None)
    if projection_params is not None:
        params = [float(v) for v in projection_params]
        if len(params) == 4:
            return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    return None


def _camera_labels_to_try(
    provider: Any,
    device_calib: Any,
    stream_id: Any,
    stream_id_text: str,
    preferred_label: str | None = None,
) -> list[str]:
    labels: list[str] = []
    if preferred_label:
        labels.append(str(preferred_label))
    for method_name in ["get_label_from_stream_id", "get_stream_label"]:
        method = getattr(provider, method_name, None)
        if callable(method):
            try:
                label = method(stream_id)
                if label:
                    labels.append(str(label))
            except Exception:
                pass
    labels.extend([stream_id_text, "camera-rgb", "rgb", "camera_rgb"])
    method = getattr(device_calib, "get_camera_calibs", None)
    if callable(method):
        try:
            calibs = method()
            if isinstance(calibs, dict):
                labels.extend(str(k) for k in calibs.keys())
        except Exception:
            pass
    return list(dict.fromkeys(labels))


def _pose_record_from_projectaria(item: Any) -> PoseRecord | None:
    timestamp = _read_any_attr(item, ["tracking_timestamp", "timestamp_ns", "capture_timestamp_ns"])
    transform = _read_any_attr(
        item,
        [
            "transform_world_device",
            "T_world_device",
            "transform_world_cpf",
            "T_world_cpf",
        ],
    )
    if timestamp is None or transform is None:
        return None
    try:
        return PoseRecord(timestamp_ns=_timestamp_to_ns(timestamp), T_world_device=_transform_to_matrix(transform))
    except Exception:
        return None


def _timestamp_to_ns(value: Any) -> int:
    if hasattr(value, "total_seconds"):
        return int(round(float(value.total_seconds()) * 1e9))
    if hasattr(value, "count"):
        try:
            return int(value.count())
        except Exception:
            pass
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, float):
        if abs(value) < 1e6:
            return int(round(value * 1e9))
        if abs(value) < 1e12:
            return int(round(value * 1000.0))
        return int(round(value))
    value_int = int(value)
    if abs(value_int) < 1_000_000:
        return int(value_int * 1_000_000_000)
    if abs(value_int) < 1_000_000_000_000:
        return int(value_int * 1000)
    return value_int


def _transform_to_matrix(transform: Any) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=float)
    if isinstance(transform, np.ndarray):
        matrix = transform.astype(float)
        if matrix.shape == (4, 4):
            return matrix
    for method_name in ["to_matrix", "matrix", "as_matrix", "to_numpy_array"]:
        method = getattr(transform, method_name, None)
        if callable(method):
            matrix = np.asarray(method(), dtype=float)
            if matrix.shape == (4, 4):
                return matrix
    matrix_attr = getattr(transform, "matrix", None)
    if matrix_attr is not None and not callable(matrix_attr):
        matrix = np.asarray(matrix_attr, dtype=float)
        if matrix.shape == (4, 4):
            return matrix
    translation = _read_any_attr(transform, ["translation", "t"])
    rotation = _read_any_attr(transform, ["rotation", "R"])
    if translation is not None and rotation is not None:
        T = np.eye(4, dtype=float)
        T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
        R = np.asarray(rotation, dtype=float)
        if R.shape == (3, 3):
            T[:3, :3] = R
            return T
    raise ValueError(f"Unsupported transform representation: {type(transform)!r}")


def _load_trajectory_csv(path: Path) -> list[PoseRecord]:
    trajectory: list[PoseRecord] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(row for row in file if not row.lstrip().startswith("#"))
        for row in reader:
            timestamp = _row_timestamp_ns(row)
            T = _row_transform_world_device(row)
            trajectory.append(PoseRecord(timestamp_ns=timestamp, T_world_device=T))
    if not trajectory:
        raise RuntimeError(f"No trajectory poses found in {path}")
    return trajectory


def _row_timestamp_ns(row: dict[str, str]) -> int:
    for key in [
        "tracking_timestamp_ns",
        "timestamp_ns",
        "capture_timestamp_ns",
        "device_timestamp_ns",
    ]:
        if key in row and row[key] != "":
            return int(float(row[key]))
    for key in ["tracking_timestamp_us", "timestamp_us", "device_timestamp_us"]:
        if key in row and row[key] != "":
            return int(float(row[key]) * 1000)
    for key in ["tracking_timestamp_s", "timestamp_s"]:
        if key in row and row[key] != "":
            return int(float(row[key]) * 1e9)
    raise KeyError("Could not find timestamp column in trajectory CSV")


def _row_transform_world_device(row: dict[str, str]) -> np.ndarray:
    matrix_keys = [f"transform_world_device_{i}_{j}" for i in range(4) for j in range(4)]
    if all(key in row for key in matrix_keys):
        return np.array([float(row[key]) for key in matrix_keys], dtype=float).reshape(4, 4)

    tx = _row_float(row, ["tx_world_device", "tx", "translation_x", "position_x", "p_world_device_x"])
    ty = _row_float(row, ["ty_world_device", "ty", "translation_y", "position_y", "p_world_device_y"])
    tz = _row_float(row, ["tz_world_device", "tz", "translation_z", "position_z", "p_world_device_z"])
    qw = _row_float(row, ["qw_world_device", "qw", "q_w", "quat_w"], default=1.0)
    qx = _row_float(row, ["qx_world_device", "qx", "q_x", "quat_x"], default=0.0)
    qy = _row_float(row, ["qy_world_device", "qy", "q_y", "quat_y"], default=0.0)
    qz = _row_float(row, ["qz_world_device", "qz", "q_z", "quat_z"], default=0.0)
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quat_to_rot(qw, qx, qy, qz)
    T[:3, 3] = [tx, ty, tz]
    return T


def _row_float(row: dict[str, str], keys: list[str], default: float | None = None) -> float:
    for key in keys:
        if key in row and row[key] != "":
            return float(row[key])
    if default is not None:
        return default
    raise KeyError(f"Could not find any of {keys} in CSV row")


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        return np.eye(3, dtype=float)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _try_projectaria_confidence_filter(
    records: list[Any],
    inverse_distance_std_threshold: float,
    distance_std_threshold: float,
) -> list[Any] | None:
    try:
        from projectaria_tools.core.mps import utils as mps_utils  # type: ignore

        helper = getattr(mps_utils, "filter_points_from_confidence", None)
        if callable(helper):
            return list(
                helper(
                    records,
                    inverse_distance_std_threshold,
                    distance_std_threshold,
                )
            )
    except Exception:
        return None
    return None


def _points_from_projectaria_records(
    records: list[Any],
    inverse_distance_std_threshold: float,
    distance_std_threshold: float,
) -> tuple[np.ndarray, str | None]:
    points: list[np.ndarray] = []
    saw_confidence = False
    for record in records:
        point = _read_any_attr(record, ["position_world", "point_world", "position", "p_world"])
        if point is None:
            try:
                array_record = np.asarray(record, dtype=float).reshape(-1)
                if len(array_record) >= 3:
                    point = array_record[:3]
            except Exception:
                point = None
        if point is None:
            continue

        inv_std = _read_any_attr(
            record,
            ["inverse_distance_std", "inverse_distance_std_m", "inv_dist_std"],
        )
        dist_std = _read_any_attr(record, ["distance_std", "distance_std_m", "dist_std"])
        if inv_std is not None or dist_std is not None:
            saw_confidence = True
        if inv_std is not None and float(inv_std) > inverse_distance_std_threshold:
            continue
        if dist_std is not None and float(dist_std) > distance_std_threshold:
            continue
        points.append(np.asarray(point, dtype=float).reshape(3))
    warning = None
    if not saw_confidence:
        warning = "MPS point confidence fields were unavailable; point cloud was not confidence-filtered."
    return np.asarray(points, dtype=float).reshape(-1, 3), warning


def _load_points_csv(
    path: Path,
    inverse_distance_std_threshold: float,
    distance_std_threshold: float,
) -> tuple[np.ndarray, int, str | None]:
    opener = gzip.open if path.suffix == ".gz" else open
    points: list[list[float]] = []
    raw_count = 0
    saw_confidence = False
    with opener(path, "rt", encoding="utf-8", newline="") as file:  # type: ignore[arg-type]
        reader = csv.DictReader(row for row in file if not row.lstrip().startswith("#"))
        for row in reader:
            raw_count += 1
            x = _row_float(row, ["x", "px", "position_x", "p_world_x", "point_x"])
            y = _row_float(row, ["y", "py", "position_y", "p_world_y", "point_y"])
            z = _row_float(row, ["z", "pz", "position_z", "p_world_z", "point_z"])
            inv_std = _optional_row_float(
                row, ["inverse_distance_std", "inverse_distance_std_m", "inv_dist_std"]
            )
            dist_std = _optional_row_float(row, ["distance_std", "distance_std_m", "dist_std"])
            if inv_std is not None or dist_std is not None:
                saw_confidence = True
            if inv_std is not None and inv_std > inverse_distance_std_threshold:
                continue
            if dist_std is not None and dist_std > distance_std_threshold:
                continue
            points.append([x, y, z])
    warning = None
    if not saw_confidence:
        warning = "MPS point confidence fields were unavailable; point cloud was not confidence-filtered."
    return np.asarray(points, dtype=float).reshape(-1, 3), raw_count, warning


def _optional_row_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        if key in row and row[key] != "":
            return float(row[key])
    return None


def _read_any_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value() if callable(value) and name.startswith("get_") else value
    return None


def _collect_metadata_keys(data: Any, interesting_keys: set[str], out: dict[str, Any]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in interesting_keys and not isinstance(value, (dict, list)):
                out.setdefault(key, value)
            elif isinstance(value, (dict, list)):
                _collect_metadata_keys(value, interesting_keys, out)
    elif isinstance(data, list):
        for item in data:
            _collect_metadata_keys(item, interesting_keys, out)
