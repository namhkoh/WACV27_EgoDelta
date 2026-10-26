#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tomato_cutpoints.config import load_config  # noqa: E402
from tomato_cutpoints.io.aria_mps import load_semidense_points  # noqa: E402
from tomato_cutpoints.io.export import write_json  # noqa: E402
from tomato_cutpoints.preprocess.quality import filter_quality  # noqa: E402
from tomato_cutpoints.registration.transforms import apply_transform, identity_transform, load_transform  # noqa: E402


@dataclass
class RenderView:
    name: str
    view_dir: np.ndarray
    up: np.ndarray


VIEWS = [
    RenderView("front", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    RenderView("top", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    RenderView("side", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    RenderView("iso", np.array([1.0, -1.0, 0.65]), np.array([0.0, 0.0, 1.0])),
]


def _parse_vec3(values: list[float] | None, name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 numbers")
    return arr


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vec / norm


def _basis(view_dir: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalize(view_dir.astype(np.float64))
    up_hint = _normalize(up_hint.astype(np.float64))
    right = np.cross(up_hint, forward)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right = _normalize(right)
    up = _normalize(np.cross(forward, right))
    return right, up, forward


def _load_xyz(points_path: Path, config: dict[str, Any], observations_path: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    max_points = int(config.get("quality", {}).get("max_points", 2_000_000))
    cloud = load_semidense_points(points_path, observations_path, max_points=max_points)
    _, xyz, report = filter_quality(cloud.points_df, config)
    return xyz, {"source_path": str(points_path), "frame_id": cloud.frame_id, "quality": report, "loader": cloud.metadata}


def _load_target_roi(target_json: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if target_json is None:
        return None, None, None, {}
    doc = json.loads(target_json.read_text(encoding="utf-8"))
    targets = doc.get("targets", [])
    if not targets:
        return None, None, None, {"warning": "target_json_has_no_targets"}
    target = targets[0]
    diag = target.get("diagnostics", {})
    bbox = diag.get("cluster_bbox")
    center = target.get("object_center_world") or target.get("cut_point_world")
    roi_info = {"target_json": str(target_json), "cluster_id": target.get("cluster_id")}
    if bbox and "min" in bbox and "max" in bbox:
        bbox_min = np.asarray(bbox["min"], dtype=np.float64)
        bbox_max = np.asarray(bbox["max"], dtype=np.float64)
        if bbox_min.shape == (3,) and bbox_max.shape == (3,):
            return bbox_min, bbox_max, np.asarray(center, dtype=np.float64) if center is not None else None, roi_info
    if center is not None:
        center_arr = np.asarray(center, dtype=np.float64)
        return None, None, center_arr, roi_info
    return None, None, None, {**roi_info, "warning": "target_json_has_no_bbox_or_center"}


def _roi_from_args(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    bbox_min, bbox_max, center, roi_info = _load_target_roi(args.target_json)
    arg_min = _parse_vec3(args.bbox_min, "--bbox-min")
    arg_max = _parse_vec3(args.bbox_max, "--bbox-max")
    arg_center = _parse_vec3(args.center, "--center")
    if arg_min is not None or arg_max is not None:
        if arg_min is None or arg_max is None:
            raise ValueError("--bbox-min and --bbox-max must be provided together")
        bbox_min, bbox_max = arg_min, arg_max
        center = (bbox_min + bbox_max) / 2.0
        roi_info = {"source": "bbox_args"}
    elif arg_center is not None:
        center = arg_center
        bbox_min = None
        bbox_max = None
        roi_info = {"source": "center_arg"}

    if bbox_min is not None and bbox_max is not None:
        margin = float(args.roi_margin)
        bbox_min = bbox_min - margin
        bbox_max = bbox_max + margin
        if center is None:
            center = (bbox_min + bbox_max) / 2.0
        roi_info["bbox_min"] = bbox_min.tolist()
        roi_info["bbox_max"] = bbox_max.tolist()
    elif center is not None:
        radius = float(args.radius)
        bbox_min = center - radius
        bbox_max = center + radius
        roi_info["center"] = center.tolist()
        roi_info["radius"] = radius
        roi_info["bbox_min"] = bbox_min.tolist()
        roi_info["bbox_max"] = bbox_max.tolist()
    return bbox_min, bbox_max, center, roi_info


def _mask_box(xyz: np.ndarray, bbox_min: np.ndarray | None, bbox_max: np.ndarray | None) -> np.ndarray:
    if bbox_min is None or bbox_max is None:
        return np.ones(len(xyz), dtype=bool)
    return np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)


def _expanded_bounds(points: np.ndarray, pad_fraction: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = span * pad_fraction
    return mins - pad, maxs + pad


def _depth_colors(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if len(depth) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.full((len(depth), 3), 180, dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], [2.0, 98.0])
    denom = max(hi - lo, 1e-9)
    t = np.clip((depth - lo) / denom, 0.0, 1.0)
    # Lightweight blue-green-yellow ramp without requiring a colormap at render time.
    r = np.clip(255.0 * (1.6 * t - 0.25), 0, 255)
    g = np.clip(255.0 * (1.2 - np.abs(t - 0.55) * 1.4), 0, 255)
    b = np.clip(255.0 * (1.2 - 1.7 * t), 0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def render_points(
    xyz: np.ndarray,
    out_png: Path,
    view: RenderView,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    image_size: int,
    point_radius: int,
    highlight_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    from PIL import Image

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    if len(xyz) == 0:
        Image.fromarray(image).save(out_png)
        return {"path": str(out_png), "n_points_rendered": 0}

    right, up, forward = _basis(view.view_dir, view.up)
    corners = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float64,
    )
    u_c = corners @ right
    v_c = corners @ up
    u_min, u_max = float(u_c.min()), float(u_c.max())
    v_min, v_max = float(v_c.min()), float(v_c.max())
    scale = 0.92 * image_size / max(u_max - u_min, v_max - v_min, 1e-9)
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)

    u = xyz @ right
    v = xyz @ up
    depth = xyz @ forward
    px = np.round((u - u_mid) * scale + image_size / 2).astype(np.int64)
    py = np.round(image_size / 2 - (v - v_mid) * scale).astype(np.int64)
    valid = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
    order = np.argsort(depth[valid])
    valid_indices = np.where(valid)[0][order]
    colors = _depth_colors(depth)
    if highlight_mask is not None:
        highlight_mask = np.asarray(highlight_mask, dtype=bool)
        colors[highlight_mask] = np.array([230, 40, 35], dtype=np.uint8)

    rr = int(max(point_radius, 0))
    for idx in valid_indices:
        x = int(px[idx])
        y = int(py[idx])
        color = colors[idx]
        if rr <= 0:
            image[y, x] = color
        else:
            y0, y1 = max(0, y - rr), min(image_size, y + rr + 1)
            x0, x1 = max(0, x - rr), min(image_size, x + rr + 1)
            image[y0:y1, x0:x1] = color
    Image.fromarray(image).save(out_png)
    return {"path": str(out_png), "n_points_rendered": int(len(valid_indices))}


def _make_contact_sheet(image_paths: list[Path], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(p).convert("RGB") for p in image_paths]
    if not images:
        return
    w, h = images[0].size
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    label_h = 24
    sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, img) in enumerate(zip(image_paths, images)):
        row, col = divmod(i, cols)
        x = col * w
        y = row * (h + label_h)
        draw.text((x + 8, y + 4), path.stem, fill=(0, 0, 0))
        sheet.paste(img, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _roi_stats(name: str, xyz: np.ndarray, roi_mask: np.ndarray) -> dict[str, Any]:
    roi_xyz = xyz[roi_mask]
    if len(roi_xyz) == 0:
        return {
            "name": name,
            "n_total": int(len(xyz)),
            "n_roi": 0,
            "roi_density_points_per_liter": 0.0,
            "bbox_min": None,
            "bbox_max": None,
        }
    mins, maxs = roi_xyz.min(axis=0), roi_xyz.max(axis=0)
    volume_m3 = float(np.prod(np.maximum(maxs - mins, 1e-4)))
    return {
        "name": name,
        "n_total": int(len(xyz)),
        "n_roi": int(len(roi_xyz)),
        "roi_density_points_per_liter": float(len(roi_xyz) / max(volume_m3 * 1000.0, 1e-9)),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render pre/post Aria MPS semi-dense point clouds to PNG images and "
            "check whether a tomato ROI is represented by points."
        )
    )
    parser.add_argument("--pre-points", type=Path, required=True)
    parser.add_argument("--post-points", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/check_recon"))
    parser.add_argument("--config", type=Path, default=Path("configs/harvest.yaml"))
    parser.add_argument("--pre-observations", type=Path)
    parser.add_argument("--post-observations", type=Path)
    parser.add_argument("--post-to-pre-transform", type=Path)
    parser.add_argument("--target-json", type=Path, help="Optional targets.json; first target bbox is used as tomato ROI.")
    parser.add_argument("--center", type=float, nargs=3, help="Tomato ROI center in pre-action world coordinates.")
    parser.add_argument("--radius", type=float, default=0.18, help="ROI half-width if --center is used.")
    parser.add_argument("--bbox-min", type=float, nargs=3, help="Tomato ROI bbox min in pre-action world coordinates.")
    parser.add_argument("--bbox-max", type=float, nargs=3, help="Tomato ROI bbox max in pre-action world coordinates.")
    parser.add_argument("--roi-margin", type=float, default=0.04, help="Margin added to target-json or bbox ROI.")
    parser.add_argument("--image-size", type=int, default=1200)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--max-render-points", type=int, default=250_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre_xyz, pre_meta = _load_xyz(args.pre_points, cfg, args.pre_observations)
    post_xyz, post_meta = _load_xyz(args.post_points, cfg, args.post_observations)
    transform = load_transform(args.post_to_pre_transform) if args.post_to_pre_transform else identity_transform()
    post_xyz = apply_transform(post_xyz, transform)

    bbox_min, bbox_max, center, roi_info = _roi_from_args(args)
    pre_roi_mask = _mask_box(pre_xyz, bbox_min, bbox_max)
    post_roi_mask = _mask_box(post_xyz, bbox_min, bbox_max)

    if bbox_min is not None and bbox_max is not None:
        render_min, render_max = _expanded_bounds(np.vstack([bbox_min, bbox_max]), pad_fraction=0.35)
        pre_render = pre_xyz[_mask_box(pre_xyz, render_min, render_max)]
        post_render = post_xyz[_mask_box(post_xyz, render_min, render_max)]
        pre_highlight = _mask_box(pre_render, bbox_min, bbox_max)
        post_highlight = _mask_box(post_render, bbox_min, bbox_max)
    else:
        combined = np.vstack([pre_xyz, post_xyz]) if len(pre_xyz) and len(post_xyz) else pre_xyz
        render_min, render_max = _expanded_bounds(combined, pad_fraction=0.08)
        pre_render = pre_xyz
        post_render = post_xyz
        pre_highlight = None
        post_highlight = None

    rng = np.random.default_rng(13)
    if len(pre_render) > args.max_render_points:
        idx = np.sort(rng.choice(len(pre_render), size=args.max_render_points, replace=False))
        pre_render = pre_render[idx]
        if pre_highlight is not None:
            pre_highlight = pre_highlight[idx]
    if len(post_render) > args.max_render_points:
        idx = np.sort(rng.choice(len(post_render), size=args.max_render_points, replace=False))
        post_render = post_render[idx]
        if post_highlight is not None:
            post_highlight = post_highlight[idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_outputs: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for view in VIEWS:
        for label, xyz, highlight in (
            ("pre", pre_render, pre_highlight),
            ("post", post_render, post_highlight),
        ):
            out_png = args.out_dir / f"{label}_{view.name}.png"
            render_outputs.append(
                {
                    "cloud": label,
                    "view": view.name,
                    **render_points(
                        xyz,
                        out_png,
                        view,
                        render_min,
                        render_max,
                        int(args.image_size),
                        int(args.point_radius),
                        highlight,
                    ),
                }
            )
            image_paths.append(out_png)
    _make_contact_sheet(image_paths, args.out_dir / "recon_contact_sheet.png")

    pre_stats = _roi_stats("pre", pre_xyz, pre_roi_mask)
    post_stats = _roi_stats("post", post_xyz, post_roi_mask)
    pre_count = pre_stats["n_roi"]
    post_count = post_stats["n_roi"]
    report = {
        "inputs": {
            "pre_points": str(args.pre_points),
            "post_points": str(args.post_points),
            "config": str(args.config),
            "post_to_pre_transform": str(args.post_to_pre_transform) if args.post_to_pre_transform else None,
        },
        "metadata": {"pre": pre_meta, "post": post_meta},
        "roi": roi_info,
        "render_bounds": {"min": render_min.tolist(), "max": render_max.tolist()},
        "stats": {
            "pre": pre_stats,
            "post": post_stats,
            "pre_to_post_roi_count_ratio": float(pre_count / max(post_count, 1)),
            "post_to_pre_roi_count_ratio": float(post_count / max(pre_count, 1)),
        },
        "interpretation": {
            "pre_roi_has_points": bool(pre_count >= 10),
            "post_roi_has_few_points": bool(post_count <= max(5, 0.35 * pre_count)),
            "note": (
                "These renders are geometric point-cloud projections, not RGB reconstruction. "
                "A tomato can only be verified by visible point support in the tomato ROI."
            ),
        },
        "renders": render_outputs,
        "contact_sheet": str(args.out_dir / "recon_contact_sheet.png"),
    }
    write_json(args.out_dir / "recon_report.json", report)

    print(f"Wrote renders to {args.out_dir}")
    print(f"Contact sheet: {args.out_dir / 'recon_contact_sheet.png'}")
    print(f"Report: {args.out_dir / 'recon_report.json'}")
    print(f"ROI counts: pre={pre_count} post={post_count}")
    if pre_count < 10:
        print("WARNING: pre-scene has very few points in the tomato ROI.")
    if post_count > max(5, 0.35 * pre_count):
        print("WARNING: post-scene still has substantial points in the tomato ROI.")


if __name__ == "__main__":
    main()
