from __future__ import annotations

from pathlib import Path

import numpy as np


def _as_uint8_colors(colors: np.ndarray, n: int) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = np.tile(colors.reshape(1, 3), (n, 1))
    if colors.max(initial=0) <= 1.0:
        colors = np.clip(colors * 255, 0, 255)
    return colors.astype(np.uint8)


def _write_ascii_ply(path: Path, xyz: np.ndarray, colors: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors_u8 = np.full((len(xyz), 3), 180, dtype=np.uint8)
    else:
        colors_u8 = _as_uint8_colors(colors, len(xyz))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, colors_u8):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_point_cloud_ply(path: str | Path, xyz: np.ndarray, color: tuple[float, float, float] | np.ndarray) -> None:
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(color, dtype=float)
    if len(xyz) == 0:
        _write_ascii_ply(path, xyz, colors)
        return
    try:
        import open3d as o3d  # type: ignore

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        if colors.ndim == 1:
            pc.colors = o3d.utility.Vector3dVector(np.tile(colors.reshape(1, 3), (len(xyz), 1)))
        else:
            pc.colors = o3d.utility.Vector3dVector(colors)
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), pc, write_ascii=True)
    except Exception:
        _write_ascii_ply(path, xyz, colors)


def view_run_plys(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise RuntimeError("Open3D is required for interactive viewing") from exc
    geoms = []
    for name in [
        "pre_filtered.ply",
        "post_filtered_aligned.ply",
        "removed_candidates.ply",
        "clustered_removed_candidates.ply",
        "target_candidates.ply",
        "cut_points.ply",
    ]:
        path = run_dir / "plys" / name
        if path.exists():
            geoms.append(o3d.io.read_point_cloud(str(path)))
    if not geoms:
        raise FileNotFoundError(f"No diagnostic PLY files found under {run_dir / 'plys'}")
    o3d.visualization.draw_geometries(geoms)
