from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np

_COLMAP_LOADER = None


def _colmap_loader():
    global _COLMAP_LOADER
    if _COLMAP_LOADER is None:
        path = Path(__file__).resolve().parents[1] / "scene" / "colmap_loader.py"
        spec = importlib.util.spec_from_file_location("_egodelta_colmap_loader", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load COLMAP helper: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _COLMAP_LOADER = module
    return _COLMAP_LOADER


@dataclass(frozen=True)
class CameraView:
    image_id: int
    image_name: str
    image_path: Path
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    r_cw: np.ndarray
    t_cw: np.ndarray

    @property
    def k(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def t_world_camera(self) -> np.ndarray:
        t_cw = np.eye(4, dtype=np.float64)
        t_cw[:3, :3] = self.r_cw
        t_cw[:3, 3] = self.t_cw
        return np.linalg.inv(t_cw)

    def to_manifest_dict(self) -> dict:
        return {
            "image_id": int(self.image_id),
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "K": self.k.tolist(),
            "R_cw": self.r_cw.tolist(),
            "t_cw": self.t_cw.tolist(),
            "T_world_camera": self.t_world_camera.tolist(),
        }


def load_colmap_views(dataset_dir: str | Path, images: str = "images", sparse: str = "sparse/0") -> list[CameraView]:
    dataset_dir = Path(dataset_dir)
    sparse_dir = dataset_dir / sparse
    loader = _colmap_loader()
    try:
        extrinsics = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
        intrinsics = loader.read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    except Exception:
        extrinsics = loader.read_extrinsics_text(str(sparse_dir / "images.txt"))
        intrinsics = loader.read_intrinsics_text(str(sparse_dir / "cameras.txt"))

    views: list[CameraView] = []
    image_root = dataset_dir / images
    for key in sorted(extrinsics.keys()):
        extr = extrinsics[key]
        intr = intrinsics[extr.camera_id]
        fx, fy, cx, cy = _intrinsics_to_pinhole(intr.model, intr.params)
        name = Path(extr.name).name
        views.append(
            CameraView(
                image_id=int(extr.id),
                image_name=name,
                image_path=image_root / name,
                width=int(intr.width),
                height=int(intr.height),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                r_cw=np.asarray(loader.qvec2rotmat(extr.qvec), dtype=np.float64),
                t_cw=np.asarray(extr.tvec, dtype=np.float64).reshape(3),
            )
        )
    views.sort(key=lambda item: item.image_name)
    return views


def _intrinsics_to_pinhole(model: str, params: np.ndarray) -> tuple[float, float, float, float]:
    params = np.asarray(params, dtype=np.float64)
    if model == "PINHOLE":
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    if model == "SIMPLE_PINHOLE":
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    raise RuntimeError(f"Only PINHOLE and SIMPLE_PINHOLE COLMAP cameras are supported, got {model}")


def project_world_points(
    points_world: np.ndarray,
    view: CameraView,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if points_world.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            np.zeros((0, 3), dtype=np.float64),
        )

    points_cam = points_world @ view.r_cw.T + view.t_cw[None, :]
    depth = points_cam[:, 2].copy()
    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float64)
    positive = depth > min_depth
    z_safe = np.maximum(depth[positive], min_depth)
    uv[positive, 0] = view.fx * points_cam[positive, 0] / z_safe + view.cx
    uv[positive, 1] = view.fy * points_cam[positive, 1] / z_safe + view.cy
    valid = (
        positive
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= float(view.width - 1))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= float(view.height - 1))
    )
    return uv, depth, valid, points_cam


def project_single_point(
    point_world: np.ndarray,
    view: CameraView,
    min_depth: float = 1e-6,
) -> tuple[np.ndarray, float, bool, np.ndarray]:
    uv, depth, valid, points_cam = project_world_points(np.asarray(point_world, dtype=np.float64).reshape(1, 3), view, min_depth)
    return uv[0], float(depth[0]), bool(valid[0]), points_cam[0]
