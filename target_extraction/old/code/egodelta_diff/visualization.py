from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from .primitives import GaussianSet2D
from .data import Camera


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def scene_parts(scene_name: str) -> tuple[str, str]:
    parts = scene_name.split("_")
    if len(parts) < 3 or parts[0] != "egodelta":
        raise ValueError(f"Unsupported scene name: {scene_name}")
    return parts[1], parts[2]


def find_manifest_for_scene(scene_name: str, pairs_root: Path) -> Path | None:
    short_id, target = scene_parts(scene_name)
    pairs_root = Path(pairs_root)
    for uuid_dir in sorted(p for p in pairs_root.iterdir() if p.is_dir()):
        compact = uuid_dir.name.replace("-", "")
        if not compact.startswith(short_id):
            continue
        manifest = uuid_dir / f"{target}_pair" / "manifest.json"
        if manifest.exists():
            return manifest
    return None


def load_image_index(scene_name: str, label: str, pairs_root: Path) -> dict[str, Path]:
    manifest_path = find_manifest_for_scene(scene_name, pairs_root)
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text())
    dataset = None
    for item in manifest.get("datasets", []):
        if item.get("task") == scene_name and item.get("label") == label:
            dataset = item
            break
    if dataset is None:
        return {}

    source_root = Path(dataset["source_root"])
    frame_list = Path(dataset["frame_list"])
    images: dict[str, Path] = {}
    if frame_list.exists():
        frame_names = [line.strip() for line in frame_list.read_text().splitlines() if line.strip()]
    else:
        frame_names = []

    for frame_name in frame_names:
        image_path = _resolve_image_path(frame_name, source_root)
        if image_path is not None:
            images[image_path.stem] = image_path

    if not images:
        for image_path in source_root.iterdir():
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                images[image_path.stem] = image_path
    return images


def _resolve_image_path(frame_name: str, source_root: Path) -> Path | None:
    path = Path(frame_name)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(source_root / path)
    if path.suffix:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    else:
        for candidate in candidates:
            for suffix in IMAGE_SUFFIXES:
                with_suffix = candidate.with_suffix(suffix)
                if with_suffix.exists():
                    return with_suffix
    return None


def save_removed_point_overlays(
    out_dir: Path,
    scene_name: str,
    g_pre: GaussianSet2D,
    removed_mask: torch.BoolTensor,
    scores: torch.Tensor,
    cameras_by_label: dict[str, list[Camera]],
    pairs_root: Path,
    labels: list[str],
    max_views: int = 8,
    max_points_per_view: int = 30000,
    point_radius: int = 2,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    overlay_dir = out_dir / "removed_point_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    points = g_pre.mu.detach().cpu()
    scores_cpu = scores.detach().cpu()
    removed = removed_mask.detach().cpu() & (scores_cpu >= float(min_score))
    if not bool(removed.any()):
        (overlay_dir / "manifest.json").write_text("[]\n")
        return []

    removed_idx = torch.nonzero(removed, as_tuple=False).flatten()
    return save_removed_point_overlays_from_arrays(
        out_dir=out_dir,
        scene_name=scene_name,
        points=points[removed_idx],
        scores=scores_cpu[removed_idx],
        cameras_by_label=cameras_by_label,
        pairs_root=pairs_root,
        labels=labels,
        max_views=max_views,
        max_points_per_view=max_points_per_view,
        point_radius=point_radius,
    )


def save_removed_point_overlays_from_arrays(
    out_dir: Path,
    scene_name: str,
    points: torch.Tensor,
    scores: torch.Tensor,
    cameras_by_label: dict[str, list[Camera]],
    pairs_root: Path,
    labels: list[str],
    max_views: int = 8,
    max_points_per_view: int = 30000,
    point_radius: int = 2,
) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    overlay_dir = out_dir / "removed_point_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for label in labels:
        image_index = load_image_index(scene_name, label, pairs_root)
        cameras = cameras_by_label.get(label, [])
        selected = [camera for camera in cameras if camera.img_name in image_index]
        if not selected:
            selected = cameras

        for view_i, camera in enumerate(selected[:max_views]):
            image_path = image_index.get(camera.img_name)
            if image_path is None:
                continue
            record = _draw_overlay(
                image_path=image_path,
                out_path=overlay_dir / f"{label}_{view_i:03d}_{camera.img_name}.png",
                camera=camera.to("cpu"),
                points=points.detach().cpu(),
                scores=scores.detach().cpu(),
                max_points=max_points_per_view,
                radius=point_radius,
            )
            record.update({"label": label, "camera_id": camera.id, "img_name": camera.img_name})
            records.append(record)

    (overlay_dir / "manifest.json").write_text(json.dumps(records, indent=2))
    return records


def _draw_overlay(
    image_path: Path,
    out_path: Path,
    camera: Camera,
    points: torch.Tensor,
    scores: torch.Tensor,
    max_points: int,
    radius: int,
) -> dict[str, Any]:
    u, v, z = camera.project(points)
    visible = (
        (z > 0.01)
        & (u >= 0.0)
        & (u < camera.width)
        & (v >= 0.0)
        & (v < camera.height)
    )
    visible_idx = torch.nonzero(visible, as_tuple=False).flatten()
    if len(visible_idx) > max_points:
        keep = torch.topk(scores[visible_idx], k=max_points, largest=True).indices
        visible_idx = visible_idx[keep]

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for idx in visible_idx.tolist():
        score = float(scores[idx])
        color = _score_color(score)
        x = float(u[idx])
        y = float(v[idx])
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255, 180),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {
        "image": str(out_path),
        "source_image": str(image_path),
        "visible_removed_points": int(len(visible_idx)),
    }


def _score_color(score: float) -> tuple[int, int, int, int]:
    score = max(0.0, min(1.0, score))
    # Yellow to red. Alpha increases with confidence to keep dense overlays readable.
    red = 255
    green = int(220 * (1.0 - score))
    blue = 0
    alpha = int(100 + 155 * score)
    return red, green, blue, alpha

import os
from pathlib import Path
import tempfile


def _set_writable_cache_dir(env_name: str, directory_name: str) -> None:
    if os.environ.get(env_name):
        return
    cache_dir = Path(tempfile.gettempdir()) / f"{directory_name}-{os.getuid()}"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if os.access(cache_dir, os.W_OK):
        os.environ[env_name] = str(cache_dir)


_set_writable_cache_dir("MPLCONFIGDIR", "egodelta-matplotlib")
_set_writable_cache_dir("XDG_CACHE_HOME", "egodelta-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

from .primitives import GaussianSet2D
from .data import Camera


def _projection_matrix(znear: float, zfar: float, tanfovx: float, tanfovy: float, device, dtype) -> torch.Tensor:
    top = znear * tanfovy
    bottom = -top
    right = znear * tanfovx
    left = -right
    p = torch.zeros(4, 4, device=device, dtype=dtype)
    z_sign = 1.0
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def _view_matrix(camera: Camera, device, dtype) -> torch.Tensor:
    rotation = camera.rotation.to(device=device, dtype=dtype)
    center = camera.center.to(device=device, dtype=dtype)
    w2c = torch.eye(4, device=device, dtype=dtype)
    w2c[:3, :3] = rotation.T
    w2c[:3, 3] = -(rotation.T @ center)
    return w2c.T.contiguous()


def render_change_map(
    g: GaussianSet2D,
    delta: torch.Tensor,
    camera: Camera,
    znear: float = 0.01,
    zfar: float = 100.0,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    if g.device.type != "cuda":
        raise RuntimeError("diff_surfel_rasterization requires CUDA tensors")
    camera = camera.to(g.device)
    tanfovx = camera.width / (2.0 * camera.fx)
    tanfovy = camera.height / (2.0 * camera.fy)
    view = _view_matrix(camera, g.device, g.mu.dtype)
    proj = _projection_matrix(znear, zfar, tanfovx, tanfovy, g.device, g.mu.dtype).T.contiguous()
    full_proj = view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    settings = GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=torch.zeros(3, device=g.device, dtype=g.mu.dtype),
        scale_modifier=float(scale_modifier),
        viewmatrix=view,
        projmatrix=full_proj,
        sh_degree=0,
        campos=camera.center.to(g.device),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(settings)
    colors = delta.clamp(0, 1).view(-1, 1).expand(-1, 3).contiguous()
    means2d = torch.zeros_like(g.mu, requires_grad=False)
    image, _, _ = rasterizer(
        means3D=g.mu,
        means2D=means2d,
        opacities=g.opacity,
        colors_precomp=colors,
        scales=g.scales,
        rotations=g.quaternion,
    )
    return image.mean(dim=0).detach().clamp(0.0, 1.0)


def save_change_map(path: Path, change_map: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = change_map.detach().cpu().numpy()
    plt.imsave(path, arr, cmap="magma", vmin=0.0, vmax=1.0)

