from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EGODELTA_RECTIFIED_RGB_FOCAL, EGODELTA_RECTIFIED_RGB_SIZE, PipelineConfig


@dataclass
class SimplePinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def project(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_camera = np.asarray(points_camera, dtype=float)
        z = points_camera[:, 2]
        valid = z > 1e-9
        uv = np.full((len(points_camera), 2), np.nan, dtype=float)
        uv[valid, 0] = self.fx * points_camera[valid, 0] / z[valid] + self.cx
        uv[valid, 1] = self.fy * points_camera[valid, 1] / z[valid] + self.cy
        return uv, valid


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _make_transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def _open_text_or_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _find_first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for candidate in root.rglob("*"):
        if candidate.name in names:
            return candidate
    return None


def _first_present(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    lowered = {k.lower(): k for k in row.keys()}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None and row.get(key, "") != "":
            return row[key]
    return None


def _timestamp_ns_from_row(row: dict[str, str]) -> int:
    ns = _first_present(
        row,
        (
            "timestamp_ns",
            "tracking_timestamp_ns",
            "capture_timestamp_ns",
            "device_timestamp_ns",
        ),
    )
    if ns is not None:
        return int(float(ns))
    us = _first_present(row, ("timestamp_us", "tracking_timestamp_us", "device_timestamp_us"))
    if us is not None:
        return int(float(us) * 1000)
    seconds = _first_present(row, ("timestamp_s", "time_s", "seconds"))
    if seconds is not None:
        return int(float(seconds) * 1e9)
    raise ValueError("Could not find timestamp column in closed-loop trajectory row.")


def _make_egodelta_rectified_camera_calibration(
    camera_calib: Any,
    rect_size: int = EGODELTA_RECTIFIED_RGB_SIZE,
    rect_focal: float = EGODELTA_RECTIFIED_RGB_FOCAL,
) -> Any:
    from projectaria_tools.core import calibration

    linear = calibration.get_linear_camera_calibration(
        int(rect_size),
        int(rect_size),
        float(rect_focal),
        "camera-rgb-linear",
        camera_calib.get_transform_device_camera(),
    )
    return calibration.rotate_camera_calib_cw90deg(linear)


class AriaMpsGeometry:
    def __init__(
        self,
        vrs_path: str | Path | None,
        mps_path: str | Path | None,
        rgb_stream_id: str,
        config: PipelineConfig,
    ) -> None:
        self.vrs_path = Path(vrs_path) if vrs_path is not None else None
        self.mps_path = Path(mps_path) if mps_path is not None else None
        self.rgb_stream_id = rgb_stream_id
        self.config = config
        self.warnings: list[str] = []
        self.point_stats: dict[str, Any] = {}
        self.world_frame_id: str | None = None

        self.trajectory: list[tuple[int, np.ndarray]] = []
        self.T_device_rgb = np.eye(4, dtype=float)
        self.camera_calibration: Any | None = None
        self.simple_camera: SimplePinholeCamera | None = None
        self.points_path: Path | None = None
        self._filtered_points: np.ndarray | None = None

        if self.mps_path is not None:
            self.trajectory = self._load_trajectory()
        self._load_rgb_calibration()

    @classmethod
    def from_synthetic(
        cls,
        points_world: np.ndarray,
        camera: SimplePinholeCamera,
        trajectory: list[tuple[int, np.ndarray]] | None = None,
        config: PipelineConfig | None = None,
    ) -> "AriaMpsGeometry":
        obj = cls.__new__(cls)
        obj.vrs_path = None
        obj.mps_path = None
        obj.rgb_stream_id = "synthetic"
        obj.config = config or PipelineConfig()
        obj.warnings = []
        obj.point_stats = {
            "num_raw_points": int(len(points_world)),
            "num_filtered_points": int(len(points_world)),
            "inv_dist_std_thres": obj.config.inv_dist_std_thres,
            "dist_std_thres": obj.config.dist_std_thres,
        }
        obj.world_frame_id = "synthetic"
        obj.trajectory = trajectory or [(0, np.eye(4, dtype=float))]
        obj.T_device_rgb = np.eye(4, dtype=float)
        obj.camera_calibration = None
        obj.simple_camera = camera
        obj.points_path = None
        obj._filtered_points = np.asarray(points_world, dtype=float)
        return obj

    @classmethod
    def from_pinhole_files(
        cls,
        *,
        source_root: str | Path,
        points_source: str | Path | None,
        camera_json: str | Path | None = None,
        poses_csv: str | Path | None = None,
        config: PipelineConfig | None = None,
        world_frame_id: str = "extracted_pinhole",
    ) -> "AriaMpsGeometry":
        source_root = Path(source_root)
        camera_json = Path(camera_json) if camera_json is not None else source_root / "camera_pinhole.json"
        poses_csv = Path(poses_csv) if poses_csv is not None else source_root / "poses.csv"
        camera_payload = json.loads(camera_json.read_text())
        camera = SimplePinholeCamera(
            width=int(camera_payload["width"]),
            height=int(camera_payload["height"]),
            fx=float(camera_payload["fx"]),
            fy=float(camera_payload["fy"]),
            cx=float(camera_payload["cx"]),
            cy=float(camera_payload["cy"]),
        )

        trajectory: list[tuple[int, np.ndarray]] = []
        with poses_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                timestamp_ns = int(float(row["timestamp_ns"]))
                values = [float(row[f"T{r}{c}"]) for r in range(4) for c in range(4)]
                trajectory.append((timestamp_ns, np.asarray(values, dtype=float).reshape(4, 4)))
        trajectory.sort(key=lambda item: item[0])
        if not trajectory:
            raise RuntimeError(f"No usable extracted-frame poses loaded from {poses_csv}.")

        obj = cls.__new__(cls)
        obj.vrs_path = None
        obj.mps_path = None
        obj.rgb_stream_id = "extracted_pinhole"
        obj.config = config or PipelineConfig()
        obj.warnings = []
        obj.point_stats = {}
        obj.world_frame_id = world_frame_id
        obj.trajectory = trajectory
        obj.T_device_rgb = np.eye(4, dtype=float)
        obj.camera_calibration = None
        obj.simple_camera = camera
        obj.points_path = Path(points_source) if points_source is not None else None
        obj._filtered_points = None
        return obj

    def _load_trajectory(self) -> list[tuple[int, np.ndarray]]:
        assert self.mps_path is not None
        trajectory_path = _find_first_existing(
            self.mps_path,
            (
                "closed_loop_trajectory.csv",
                "closed_loop_trajectory.csv.gz",
                "open_loop_trajectory.csv",
                "open_loop_trajectory.csv.gz",
            ),
        )
        if trajectory_path is None:
            self.warnings.append(f"No closed_loop_trajectory.csv found under {self.mps_path}.")
            return []

        poses: list[tuple[int, np.ndarray]] = []
        with _open_text_or_gzip(trajectory_path) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self.world_frame_id is None:
                    self.world_frame_id = _first_present(
                        row,
                        ("graph_uid", "multi_slam_uid", "world_frame_uid", "world_id", "map_uid"),
                    )
                try:
                    timestamp_ns = _timestamp_ns_from_row(row)
                    tx = float(
                        _first_present(
                            row,
                            ("tx_world_device", "tx_odometry_device", "position_x", "x", "tx"),
                        )
                        or 0.0
                    )
                    ty = float(
                        _first_present(
                            row,
                            ("ty_world_device", "ty_odometry_device", "position_y", "y", "ty"),
                        )
                        or 0.0
                    )
                    tz = float(
                        _first_present(
                            row,
                            ("tz_world_device", "tz_odometry_device", "position_z", "z", "tz"),
                        )
                        or 0.0
                    )
                    qw = float(
                        _first_present(row, ("qw_world_device", "qw_odometry_device", "qw", "w"))
                        or 1.0
                    )
                    qx = float(
                        _first_present(row, ("qx_world_device", "qx_odometry_device", "qx"))
                        or 0.0
                    )
                    qy = float(
                        _first_present(row, ("qy_world_device", "qy_odometry_device", "qy"))
                        or 0.0
                    )
                    qz = float(
                        _first_present(row, ("qz_world_device", "qz_odometry_device", "qz"))
                        or 0.0
                    )
                except Exception as exc:
                    self.warnings.append(f"Skipped malformed trajectory row: {exc}")
                    continue
                T = _make_transform(np.array([tx, ty, tz], dtype=float), _quat_to_matrix(qw, qx, qy, qz))
                poses.append((timestamp_ns, T))
        poses.sort(key=lambda item: item[0])
        if not poses:
            self.warnings.append(f"No usable poses loaded from {trajectory_path}.")
        return poses

    def _load_rgb_calibration(self) -> None:
        if self.vrs_path is None:
            return
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.stream_id import StreamId
        except Exception:
            self.warnings.append(
                "projectaria-tools is not installed; using a fallback pinhole camera when needed."
            )
            return

        try:
            provider = data_provider.create_vrs_data_provider(str(self.vrs_path))
            stream_id = StreamId(self.rgb_stream_id)
            device_calib = provider.get_device_calibration()
            labels = [self.rgb_stream_id, "camera-rgb", "camera_rgb", "rgb"]
            try:
                labels.insert(0, provider.get_label_from_stream_id(stream_id))
            except Exception:
                pass
            camera_calib = None
            for label in labels:
                try:
                    camera_calib = device_calib.get_camera_calib(label)
                    if camera_calib is not None:
                        break
                except Exception:
                    continue
            if camera_calib is None:
                self.warnings.append(
                    f"Could not find RGB camera calibration for stream {self.rgb_stream_id}."
                )
                return
            if self.config.use_undistorted_rgb:
                try:
                    camera_calib = _make_egodelta_rectified_camera_calibration(
                        camera_calib,
                        rect_size=self.config.undistort_rect_size,
                        rect_focal=self.config.undistort_rect_focal,
                    )
                except Exception as exc:
                    self.warnings.append(f"Failed to create undistorted RGB calibration: {exc}")
            self.camera_calibration = camera_calib
            for method in ("get_transform_device_camera", "get_transform_device_sensor"):
                if hasattr(camera_calib, method):
                    try:
                        transform = getattr(camera_calib, method)()
                        self.T_device_rgb = np.asarray(transform.to_matrix(), dtype=float)
                        break
                    except Exception:
                        continue
        except Exception as exc:
            self.warnings.append(f"Failed to load VRS RGB calibration: {exc}")

    def _default_camera_for_image(self, image_width: int, image_height: int) -> SimplePinholeCamera:
        focal = float(max(image_width, image_height))
        return SimplePinholeCamera(
            width=image_width,
            height=image_height,
            fx=focal,
            fy=focal,
            cx=(image_width - 1) / 2.0,
            cy=(image_height - 1) / 2.0,
        )

    def load_filtered_points(self) -> np.ndarray:
        if self._filtered_points is not None:
            return self._filtered_points
        if self.points_path is not None:
            self._filtered_points = self._load_points_file(self.points_path)
            return self._filtered_points
        if self.mps_path is None:
            self._filtered_points = np.empty((0, 3), dtype=float)
            return self._filtered_points

        point_path = _find_first_existing(
            self.mps_path,
            (
                "semidense_points.csv.gz",
                "semidense_points.csv",
                "semi_dense_points.csv.gz",
                "semi_dense_points.csv",
            ),
        )
        if point_path is None:
            self.warnings.append(f"No semidense_points.csv.gz found under {self.mps_path}.")
            self._filtered_points = np.empty((0, 3), dtype=float)
            return self._filtered_points

        points: list[list[float]] = []
        raw_count = 0
        saw_inv_dist_std = False
        saw_dist_std = False
        with _open_text_or_gzip(point_path) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_count += 1
                try:
                    x = float(_first_present(row, ("px_world", "x_world", "x", "px")) or "nan")
                    y = float(_first_present(row, ("py_world", "y_world", "y", "py")) or "nan")
                    z = float(_first_present(row, ("pz_world", "z_world", "z", "pz")) or "nan")
                    if not np.isfinite([x, y, z]).all():
                        continue
                    inv_dist_std = _first_present(row, ("inv_dist_std", "inverse_distance_std"))
                    dist_std = _first_present(row, ("dist_std", "distance_std"))
                    keep = True
                    if inv_dist_std is not None:
                        saw_inv_dist_std = True
                        keep = keep and float(inv_dist_std) <= self.config.inv_dist_std_thres
                    if dist_std is not None:
                        saw_dist_std = True
                        keep = keep and float(dist_std) <= self.config.dist_std_thres
                    if keep:
                        points.append([x, y, z])
                except Exception as exc:
                    self.warnings.append(f"Skipped malformed semidense point row: {exc}")
        if not saw_inv_dist_std:
            self.warnings.append("semidense point field inv_dist_std unavailable; skipped that filter.")
        if not saw_dist_std:
            self.warnings.append("semidense point field dist_std unavailable; skipped that filter.")
        self._filtered_points = np.asarray(points, dtype=float).reshape((-1, 3))
        self.point_stats = {
            "num_raw_points": int(raw_count),
            "num_filtered_points": int(len(self._filtered_points)),
            "inv_dist_std_thres": self.config.inv_dist_std_thres,
            "dist_std_thres": self.config.dist_std_thres,
        }
        return self._filtered_points

    def _load_points_file(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith(".csv.gz") or path.suffix.lower() == ".csv":
            return self._load_semidense_points_file(path)
        points: list[list[float]] = []
        raw_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                raw_count += 1
                parts = stripped.split()
                if len(parts) < 4:
                    self.warnings.append(f"Skipped malformed point row in {path}: {stripped}")
                    continue
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    self.warnings.append(f"Skipped non-numeric point row in {path}: {stripped}")
                    continue
                if np.isfinite([x, y, z]).all():
                    points.append([x, y, z])
        arr = np.asarray(points, dtype=float).reshape((-1, 3))
        self.point_stats = {
            "num_raw_points": int(raw_count),
            "num_filtered_points": int(len(arr)),
            "source": str(path),
            "format": "colmap_points3D_text",
            "confidence_filter": "already_filtered_or_unavailable",
        }
        return arr

    def _load_semidense_points_file(self, path: Path) -> np.ndarray:
        points: list[list[float]] = []
        raw_count = 0
        saw_inv_dist_std = False
        saw_dist_std = False
        with _open_text_or_gzip(path) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_count += 1
                try:
                    x = float(_first_present(row, ("px_world", "x_world", "x", "px")) or "nan")
                    y = float(_first_present(row, ("py_world", "y_world", "y", "py")) or "nan")
                    z = float(_first_present(row, ("pz_world", "z_world", "z", "pz")) or "nan")
                    if not np.isfinite([x, y, z]).all():
                        continue
                    inv_dist_std = _first_present(row, ("inv_dist_std", "inverse_distance_std"))
                    dist_std = _first_present(row, ("dist_std", "distance_std"))
                    keep = True
                    if inv_dist_std is not None:
                        saw_inv_dist_std = True
                        keep = keep and float(inv_dist_std) <= self.config.inv_dist_std_thres
                    if dist_std is not None:
                        saw_dist_std = True
                        keep = keep and float(dist_std) <= self.config.dist_std_thres
                    if keep:
                        points.append([x, y, z])
                except Exception as exc:
                    self.warnings.append(f"Skipped malformed semidense point row: {exc}")
        if not saw_inv_dist_std:
            self.warnings.append("semidense point field inv_dist_std unavailable; skipped that filter.")
        if not saw_dist_std:
            self.warnings.append("semidense point field dist_std unavailable; skipped that filter.")
        arr = np.asarray(points, dtype=float).reshape((-1, 3))
        self.point_stats = {
            "num_raw_points": int(raw_count),
            "num_filtered_points": int(len(arr)),
            "source": str(path),
            "inv_dist_std_thres": self.config.inv_dist_std_thres,
            "dist_std_thres": self.config.dist_std_thres,
        }
        return arr

    def save_point_stats(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w") as handle:
            json.dump(self.point_stats, handle, indent=2)

    def get_T_world_rgb(self, timestamp_ns: int) -> np.ndarray:
        if not self.trajectory:
            return self.T_device_rgb.copy()
        timestamps = np.array([item[0] for item in self.trajectory], dtype=np.int64)
        idx = int(np.argmin(np.abs(timestamps - int(timestamp_ns))))
        T_world_device = self.trajectory[idx][1]
        return T_world_device @ self.T_device_rgb

    def project_world_points_to_frame(
        self,
        points_world: np.ndarray,
        timestamp_ns: int,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points_world = np.asarray(points_world, dtype=float).reshape((-1, 3))
        if len(points_world) == 0:
            return (
                np.empty((0, 2), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=bool),
            )
        T_world_rgb = self.get_T_world_rgb(timestamp_ns)
        T_rgb_world = np.linalg.inv(T_world_rgb)
        points_h = np.c_[points_world, np.ones((len(points_world),), dtype=float)]
        points_rgb = (T_rgb_world @ points_h.T).T[:, :3]
        depth = points_rgb[:, 2].copy()

        uv = np.full((len(points_world), 2), np.nan, dtype=float)
        valid = depth > 1e-9
        if self.camera_calibration is not None and hasattr(self.camera_calibration, "project"):
            for idx, point in enumerate(points_rgb):
                if not valid[idx]:
                    continue
                try:
                    projected = self.camera_calibration.project(point)
                    if projected is not None:
                        uv[idx] = np.asarray(projected, dtype=float)[:2]
                except Exception:
                    valid[idx] = False
        else:
            camera = self.simple_camera or self._default_camera_for_image(image_width, image_height)
            uv_projected, project_valid = camera.project(points_rgb)
            uv = uv_projected
            valid &= project_valid

        valid &= np.isfinite(uv).all(axis=1)
        valid &= uv[:, 0] >= 0
        valid &= uv[:, 0] < image_width
        valid &= uv[:, 1] >= 0
        valid &= uv[:, 1] < image_height
        return uv, depth, valid


def ensure_same_world_frame(pre_geometry: AriaMpsGeometry, post_geometry: AriaMpsGeometry) -> None:
    pre_uid = pre_geometry.world_frame_id
    post_uid = post_geometry.world_frame_id
    if pre_uid and post_uid and pre_uid != post_uid:
        raise RuntimeError(
            "Pre and post MPS appear to be in different world frames. Run Aria Multi-SLAM "
            "or provide an alignment transform."
        )
