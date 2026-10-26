from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_points(instance: dict[str, Any]) -> np.ndarray:
    path = instance.get("points_3d_path")
    if path and Path(path).exists():
        try:
            return np.asarray(np.load(path), dtype=float).reshape((-1, 3))
        except Exception:
            pass
    centroid = np.asarray(instance.get("centroid_3d", [0.0, 0.0, 0.0]), dtype=float)
    return centroid.reshape((1, 3))


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8, copy=False)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return boundary & mask


def _mask_principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size < 3:
        return None
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    center = np.mean(coords, axis=0)
    centered = coords - center.reshape((1, 2))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return None
    axis = axis / axis_norm
    projected = centered @ axis
    if len(projected) < 5:
        length = float(projected.max() - projected.min())
    else:
        length = float(np.quantile(projected, 0.95) - np.quantile(projected, 0.05))
    verticality = float(abs(axis[1]))
    return center, axis, max(6.0, 0.5 * length), verticality


def save_sam_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    labels: list[str],
    out_path: str | Path,
    draw_principal_axes: bool = False,
) -> None:
    colors = np.array(
        [
            [230, 57, 70],
            [42, 157, 143],
            [244, 162, 97],
            [69, 123, 157],
            [233, 196, 106],
            [131, 56, 236],
        ],
        dtype=np.float32,
    )
    rgb = _as_rgb(image).astype(np.float32)
    overlay = rgb.copy()
    for idx, mask in enumerate(masks):
        if mask.size == 0:
            continue
        color = colors[idx % len(colors)]
        mask_bool = np.asarray(mask, dtype=bool)
        overlay[mask_bool] = 0.55 * overlay[mask_bool] + 0.45 * color
        boundary = _mask_boundary(mask_bool)
        overlay[boundary] = color
    pil = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    if draw_principal_axes:
        for idx, mask in enumerate(masks):
            axis_info = _mask_principal_axis(mask)
            if axis_info is None:
                continue
            center, axis, half_length, verticality = axis_info
            p0 = center - axis * half_length
            p1 = center + axis * half_length
            color = tuple(colors[idx % len(colors)].astype(int))
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(255, 255, 255), width=4)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=2)
            cx, cy = float(center[0]), float(center[1])
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255), outline=color)
            text_xy = (cx + 5, cy + 5)
            draw.text(text_xy, f"v={verticality:.2f}", fill=(255, 255, 255))
    for idx, label in enumerate(labels[:10]):
        draw.text((8, 8 + idx * 14), label[:80], fill=tuple(colors[idx % len(colors)].astype(int)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def save_lifting_debug_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    uv_inside: np.ndarray,
    uv_foreground: np.ndarray,
    out_path: str | Path,
    label: str,
) -> None:
    rgb = _as_rgb(image).astype(np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    rgb[mask_bool] = 0.70 * rgb[mask_bool] + 0.30 * np.array([255, 210, 0], dtype=np.float32)
    boundary = _mask_boundary(mask_bool)
    rgb[boundary] = np.array([255, 210, 0], dtype=np.float32)
    pil = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    for uv in np.asarray(uv_inside).reshape((-1, 2)):
        if np.isfinite(uv).all():
            x, y = float(uv[0]), float(uv[1])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 180, 255))
    for uv in np.asarray(uv_foreground).reshape((-1, 2)):
        if np.isfinite(uv).all():
            x, y = float(uv[0]), float(uv[1])
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(20, 240, 80))
    draw.rectangle((4, 4, min(900, pil.width - 4), 24), fill=(0, 0, 0))
    draw.text((8, 8), label[:120], fill=(255, 255, 255))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def _draw_labeled_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    label: str,
) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=3)
    draw.line((x - radius - 4, y, x + radius + 4, y), fill=outline, width=2)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill=outline, width=2)
    if label:
        text_xy = (x + radius + 6, y - radius - 4)
        try:
            bbox = draw.textbbox(text_xy, label)
        except Exception:
            bbox = (text_xy[0], text_xy[1], text_xy[0] + 8 * len(label), text_xy[1] + 14)
        draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
        draw.text(text_xy, label, fill=outline)


def save_target_overlays_on_frames(
    frames: list[dict[str, Any]],
    geometry: Any,
    predictions: list[dict[str, Any]],
    split: str,
    out_dir: str | Path,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Project final 3D target points back into Aria RGB frames and draw overlays."""
    overlay_dir = Path(out_dir) / "visualizations" / "target_overlay" / split
    overlay_dir.mkdir(parents=True, exist_ok=True)
    stats = {"split": split, "frames_written": 0, "projected_targets": 0}
    if not predictions:
        return stats

    def short_target_type(prediction: dict[str, Any]) -> str:
        point_type = str(prediction.get("target_point_type", "target"))
        if "peduncle" in point_type or "stem" in point_type:
            return "peduncle"
        if "proxy" in point_type:
            return "proxy"
        if "centroid" in point_type:
            return "centroid"
        return "target"

    selected_frames = frames if max_frames is None else frames[:max_frames]
    for frame in selected_frames:
        image_path = frame.get("image_path")
        if not image_path or not Path(image_path).exists():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width = int(frame.get("width", image.width))
        height = int(frame.get("height", image.height))
        timestamp_ns = int(frame["timestamp_ns"])

        for idx, pred in enumerate(predictions):
            points: list[list[float]] = []
            labels: list[str] = []
            target = pred.get("target_point_3d_world")
            centroid = pred.get("object_centroid_3d_world")
            if target is not None:
                points.append([float(v) for v in target])
                labels.append(f"cut {idx}: {pred.get('target_point_type', 'target')}")
            if centroid is not None:
                points.append([float(v) for v in centroid])
                labels.append(f"obj {idx}")
            if not points:
                continue

            uv, _, valid = geometry.project_world_points_to_frame(
                np.asarray(points, dtype=float),
                timestamp_ns,
                width,
                height,
            )
            target_xy = None
            centroid_xy = None
            centroid_idx = 1 if target is not None else 0
            if centroid is not None and centroid_idx < len(valid) and valid[centroid_idx]:
                centroid_xy = (float(uv[centroid_idx, 0]), float(uv[centroid_idx, 1]))
            if target is not None and valid[0]:
                target_xy = (float(uv[0, 0]), float(uv[0, 1]))
                stats["projected_targets"] += 1
            if target_xy is not None and centroid_xy is not None:
                draw.line((centroid_xy[0], centroid_xy[1], target_xy[0], target_xy[1]), fill=(255, 255, 255), width=2)
            if centroid_xy is not None:
                _draw_labeled_marker(
                    draw,
                    centroid_xy,
                    radius=6,
                    fill=(255, 255, 255),
                    outline=(30, 70, 255),
                    label=f"obj{idx}",
                )
            if target_xy is not None:
                _draw_labeled_marker(
                    draw,
                    target_xy,
                    radius=9,
                    fill=(25, 75, 255),
                    outline=(255, 255, 255),
                    label=f"cut{idx} {short_target_type(pred)}",
                )

        draw.rectangle((4, 4, min(image.width - 4, 760), 30), fill=(0, 0, 0))
        draw.text((8, 10), f"{split} target overlay: blue=cut point, white=object centroid", fill=(255, 255, 255))
        image.save(overlay_dir / f"{frame['frame_id']}.jpg")
        stats["frames_written"] += 1
    return stats


def _sample_points_for_overlay(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def _draw_projected_points(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 2,
) -> int:
    count = 0
    for point, is_valid in zip(np.asarray(uv).reshape((-1, 2)), np.asarray(valid, dtype=bool)):
        if not is_valid or not np.isfinite(point).all():
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        count += 1
    return count


def _status_short(status: str) -> str:
    if status == "disappeared":
        return "gone"
    if status == "still_present":
        return "present"
    if status == "unknown_not_enough_post_visibility":
        return "unknown"
    if status == "still_present_or_uncertain":
        return "uncertain"
    return status or "n/a"


def save_change_debug_overlays_on_frames(
    frames: list[dict[str, Any]],
    geometry: Any,
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    change_candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    split: str,
    out_dir: str | Path,
    max_frames: int | None = None,
    max_points_per_instance: int = 500,
) -> dict[str, Any]:
    """Project 3D change decisions into RGB frames and save PNG overlays."""
    overlay_dir = Path(out_dir) / "visualizations" / f"change_debug_{split}"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "split": split,
        "frames_written": 0,
        "projected_pre_instances": 0,
        "projected_post_instances": 0,
        "projected_targets": 0,
    }
    if not frames:
        return stats

    candidates_by_pre = {item.get("pre_instance_id"): item for item in change_candidates}
    pre_primary = [item for item in pre_instances if item.get("semantic_group", "primary") == "primary"]
    post_primary = [item for item in post_instances if item.get("semantic_group", "primary") == "primary"]
    disappeared_pre = [
        item
        for item in pre_primary
        if candidates_by_pre.get(item.get("instance_id"), {}).get("status") == "disappeared"
    ]
    uncertain_pre = [
        item
        for item in pre_primary
        if candidates_by_pre.get(item.get("instance_id"), {}).get("status")
        in ("still_present_or_uncertain", "unknown_not_enough_post_visibility")
    ]

    if split == "pre":
        instances_to_draw = [(instance, candidates_by_pre.get(instance.get("instance_id"), {})) for instance in pre_primary]
        post_instances_to_draw = post_primary
    else:
        instances_to_draw = [
            (instance, candidates_by_pre.get(instance.get("instance_id"), {}))
            for instance in disappeared_pre + uncertain_pre
        ]
        post_instances_to_draw = post_primary

    selected_frames = frames if max_frames is None else frames[:max_frames]
    for frame in selected_frames:
        image_path = frame.get("image_path")
        if not image_path or not Path(image_path).exists():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width = int(frame.get("width", image.width))
        height = int(frame.get("height", image.height))
        timestamp_ns = int(frame["timestamp_ns"])

        for instance in post_instances_to_draw:
            points = _sample_points_for_overlay(_load_points(instance), max_points_per_instance)
            if len(points):
                uv, _, valid = geometry.project_world_points_to_frame(points, timestamp_ns, width, height)
                projected = _draw_projected_points(draw, uv, valid, (42, 157, 143), radius=2)
                if projected:
                    stats["projected_post_instances"] += 1
            centroid = _instance_centroid(instance)
            if centroid is not None:
                uv, _, valid = geometry.project_world_points_to_frame(
                    centroid.reshape((1, 3)), timestamp_ns, width, height
                )
                if len(valid) and valid[0]:
                    _draw_labeled_marker(
                        draw,
                        (float(uv[0, 0]), float(uv[0, 1])),
                        radius=5,
                        fill=(42, 157, 143),
                        outline=(255, 255, 255),
                        label="post",
                    )

        for idx, (instance, candidate) in enumerate(instances_to_draw):
            status = str(candidate.get("status", "not_evaluated"))
            color = _change_status_color(status)
            points = _sample_points_for_overlay(_load_points(instance), max_points_per_instance)
            if len(points):
                uv, _, valid = geometry.project_world_points_to_frame(points, timestamp_ns, width, height)
                projected = _draw_projected_points(draw, uv, valid, color, radius=3 if status == "disappeared" else 2)
                if projected:
                    stats["projected_pre_instances"] += 1
            centroid = _instance_centroid(instance)
            if centroid is not None:
                uv, _, valid = geometry.project_world_points_to_frame(
                    centroid.reshape((1, 3)), timestamp_ns, width, height
                )
                if len(valid) and valid[0]:
                    score = candidate.get("final_change_score")
                    score_text = "" if score is None else f" {float(score):.2f}"
                    _draw_labeled_marker(
                        draw,
                        (float(uv[0, 0]), float(uv[0, 1])),
                        radius=8 if status == "disappeared" else 6,
                        fill=color,
                        outline=(255, 255, 255),
                        label=f"{idx}:{_status_short(status)}{score_text}",
                    )

        for idx, pred in enumerate(predictions):
            target = pred.get("target_point_3d_world")
            centroid = pred.get("object_centroid_3d_world")
            points = []
            labels = []
            if centroid is not None:
                points.append([float(x) for x in centroid])
                labels.append(f"obj{idx}")
            if target is not None:
                points.append([float(x) for x in target])
                labels.append(f"cut{idx}")
            if not points:
                continue
            uv, _, valid = geometry.project_world_points_to_frame(
                np.asarray(points, dtype=float), timestamp_ns, width, height
            )
            if centroid is not None and len(valid) >= 1 and valid[0]:
                _draw_labeled_marker(
                    draw,
                    (float(uv[0, 0]), float(uv[0, 1])),
                    radius=5,
                    fill=(255, 255, 255),
                    outline=(30, 70, 255),
                    label=labels[0],
                )
            target_idx = 1 if centroid is not None else 0
            if target is not None and target_idx < len(valid) and valid[target_idx]:
                _draw_labeled_marker(
                    draw,
                    (float(uv[target_idx, 0]), float(uv[target_idx, 1])),
                    radius=9,
                    fill=(30, 70, 255),
                    outline=(255, 255, 255),
                    label=f"cut{idx}",
                )
                stats["projected_targets"] += 1

        draw.rectangle((4, 4, min(image.width - 4, 980), 52), fill=(0, 0, 0))
        if split == "pre":
            title = "3D change overlay pre: yellow=disappeared, red=present pre, green=post projection, blue=cut"
        else:
            title = "3D change overlay post: green=post, yellow=disappeared pre projection, purple/orange=uncertain"
        draw.text((8, 10), title, fill=(255, 255, 255))
        draw.text((8, 30), f"{frame.get('frame_id', 'frame')}", fill=(255, 255, 255))
        image.save(overlay_dir / f"{frame.get('frame_id', 'frame')}.png")
        stats["frames_written"] += 1
    return stats


def write_point_cloud_ply(points: np.ndarray, colors: np.ndarray, path: str | Path) -> None:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    colors = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
    if len(colors) != len(points):
        colors = np.repeat(np.array([[255, 255, 255]], dtype=np.uint8), len(points), axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _sample_sphere(center: np.ndarray, radius: float = 0.012, rings: int = 8, sectors: int = 16) -> np.ndarray:
    pts: list[list[float]] = []
    center = np.asarray(center, dtype=float)
    for i in range(rings):
        phi = math.pi * (i + 0.5) / rings
        for j in range(sectors):
            theta = 2.0 * math.pi * j / sectors
            pts.append(
                (
                    center
                    + radius
                    * np.array(
                        [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)]
                    )
                ).tolist()
            )
    return np.asarray(pts, dtype=float)


def _sample_line(start: np.ndarray, end: np.ndarray, spacing: float = 0.006) -> np.ndarray:
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    length = float(np.linalg.norm(end - start))
    if length < 1e-9:
        return start.reshape((1, 3))
    count = int(np.clip(np.ceil(length / max(1e-6, spacing)), 2, 256))
    alpha = np.linspace(0.0, 1.0, count).reshape((-1, 1))
    return (1.0 - alpha) * start.reshape((1, 3)) + alpha * end.reshape((1, 3))


def _instance_centroid(instance: dict[str, Any]) -> np.ndarray | None:
    centroid = instance.get("centroid_3d")
    if centroid is None:
        return None
    point = np.asarray(centroid, dtype=float).reshape(-1)
    if point.size != 3 or not np.isfinite(point).all():
        return None
    return point


def _change_status_color(status: str) -> tuple[int, int, int]:
    if status == "disappeared":
        return (255, 220, 0)
    if status == "still_present":
        return (230, 57, 70)
    if status == "unknown_not_enough_post_visibility":
        return (255, 140, 0)
    if status == "still_present_or_uncertain":
        return (190, 80, 255)
    return (230, 57, 70)


def _append_points(
    chunks: list[np.ndarray],
    colors: list[np.ndarray],
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) == 0:
        return
    chunks.append(points)
    colors.append(np.repeat(np.array([color], dtype=np.uint8), len(points), axis=0))


def _stack_points(chunks: list[np.ndarray], colors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not chunks:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(chunks), np.vstack(colors)


def write_change_debug_plys(
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    change_candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write 3D pre/post change visualizations for inspecting disappearance decisions."""
    vis_dir = Path(out_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    candidates_by_pre = {item.get("pre_instance_id"): item for item in change_candidates}
    post_by_id = {item.get("instance_id"): item for item in post_instances}

    scene_chunks: list[np.ndarray] = []
    scene_colors: list[np.ndarray] = []
    centroid_chunks: list[np.ndarray] = []
    centroid_colors: list[np.ndarray] = []
    disappeared_chunks: list[np.ndarray] = []
    disappeared_colors: list[np.ndarray] = []

    status_counts: dict[str, int] = {}
    for instance in pre_instances:
        if instance.get("semantic_group", "primary") != "primary":
            continue
        candidate = candidates_by_pre.get(instance.get("instance_id"), {})
        status = str(candidate.get("status", "not_evaluated"))
        status_counts[status] = status_counts.get(status, 0) + 1
        color = _change_status_color(status)
        points = _load_points(instance)
        _append_points(scene_chunks, scene_colors, points, color)
        if status == "disappeared":
            _append_points(disappeared_chunks, disappeared_colors, points, color)

        centroid = _instance_centroid(instance)
        if centroid is not None:
            radius = 0.016 if status == "disappeared" else 0.011
            sphere = _sample_sphere(centroid, radius=radius)
            _append_points(scene_chunks, scene_colors, sphere, color)
            _append_points(centroid_chunks, centroid_colors, sphere, color)

            matched_post_id = candidate.get("matched_post_instance_id")
            matched_post = post_by_id.get(matched_post_id)
            post_centroid = _instance_centroid(matched_post) if matched_post is not None else None
            if post_centroid is not None:
                line = _sample_line(centroid, post_centroid)
                _append_points(scene_chunks, scene_colors, line, color)
                _append_points(centroid_chunks, centroid_colors, line, color)

    for instance in post_instances:
        if instance.get("semantic_group", "primary") != "primary":
            continue
        points = _load_points(instance)
        _append_points(scene_chunks, scene_colors, points, (42, 157, 143))
        _append_points(disappeared_chunks, disappeared_colors, points, (42, 157, 143))
        centroid = _instance_centroid(instance)
        if centroid is not None:
            sphere = _sample_sphere(centroid, radius=0.011)
            _append_points(scene_chunks, scene_colors, sphere, (42, 157, 143))
            _append_points(centroid_chunks, centroid_colors, sphere, (42, 157, 143))

    for pred in predictions:
        if pred.get("target_point_3d_world") is not None:
            sphere = _sample_sphere(np.asarray(pred["target_point_3d_world"], dtype=float), radius=0.014)
            _append_points(scene_chunks, scene_colors, sphere, (30, 70, 255))
            _append_points(centroid_chunks, centroid_colors, sphere, (30, 70, 255))
            _append_points(disappeared_chunks, disappeared_colors, sphere, (30, 70, 255))
        if pred.get("object_centroid_3d_world") is not None:
            sphere = _sample_sphere(np.asarray(pred["object_centroid_3d_world"], dtype=float), radius=0.010)
            _append_points(scene_chunks, scene_colors, sphere, (255, 255, 255))
            _append_points(centroid_chunks, centroid_colors, sphere, (255, 255, 255))
            _append_points(disappeared_chunks, disappeared_colors, sphere, (255, 255, 255))

    scene_points, scene_rgb = _stack_points(scene_chunks, scene_colors)
    centroid_points, centroid_rgb = _stack_points(centroid_chunks, centroid_colors)
    disappeared_points, disappeared_rgb = _stack_points(disappeared_chunks, disappeared_colors)
    scene_path = vis_dir / "change_debug_scene.ply"
    centroid_path = vis_dir / "change_centroids_and_links.ply"
    disappeared_path = vis_dir / "disappeared_vs_post_scene.ply"
    write_point_cloud_ply(scene_points, scene_rgb, scene_path)
    write_point_cloud_ply(centroid_points, centroid_rgb, centroid_path)
    write_point_cloud_ply(disappeared_points, disappeared_rgb, disappeared_path)

    legend = {
        "pre_disappeared": "yellow",
        "pre_still_present": "red",
        "pre_unknown_not_enough_post_visibility": "orange",
        "pre_still_present_or_uncertain": "purple",
        "pre_not_evaluated_or_other": "red",
        "post_instances": "green",
        "target_point": "blue sphere",
        "object_centroid": "white sphere",
        "centroid_links": "sampled colored points from pre centroid to matched/nearest post centroid",
    }
    summary = {
        "paths": {
            "change_debug_scene": str(scene_path),
            "change_centroids_and_links": str(centroid_path),
            "disappeared_vs_post_scene": str(disappeared_path),
        },
        "status_counts": status_counts,
        "legend": legend,
        "num_change_candidates": int(len(change_candidates)),
        "num_predictions": int(len(predictions)),
    }
    with (vis_dir / "change_debug_summary.json").open("w") as handle:
        import json

        json.dump(summary, handle, indent=2)
    return summary


def write_visualization_plys(
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    change_candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    disappeared_ids = {
        item["pre_instance_id"] for item in change_candidates if item.get("status") == "disappeared"
    }

    def collect(instances: list[dict[str, Any]], color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        chunks = [_load_points(instance) for instance in instances]
        if not chunks:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
        pts = np.vstack(chunks)
        colors = np.repeat(np.array([color], dtype=np.uint8), len(pts), axis=0)
        return pts, colors

    pre_primary = [i for i in pre_instances if i.get("semantic_group", "primary") == "primary"]
    post_primary = [i for i in post_instances if i.get("semantic_group", "primary") == "primary"]
    aux_instances = [i for i in pre_instances + post_instances if i.get("semantic_group") == "auxiliary"]
    disappeared = [i for i in pre_instances if i.get("instance_id") in disappeared_ids]

    pre_pts, pre_cols = collect(pre_primary, (230, 57, 70))
    post_pts, post_cols = collect(post_primary, (42, 157, 143))
    aux_pts, aux_cols = collect(aux_instances, (0, 220, 220))
    dis_pts, dis_cols = collect(disappeared, (255, 220, 0))

    write_point_cloud_ply(pre_pts, pre_cols, out_dir / "pre_instances.ply")
    write_point_cloud_ply(post_pts, post_cols, out_dir / "post_instances.ply")
    write_point_cloud_ply(dis_pts, dis_cols, out_dir / "disappeared_instances.ply")

    target_chunks: list[np.ndarray] = []
    target_cols: list[np.ndarray] = []
    for pred in predictions:
        if pred.get("target_point_3d_world") is not None:
            sphere = _sample_sphere(np.asarray(pred["target_point_3d_world"], dtype=float), radius=0.012)
            target_chunks.append(sphere)
            target_cols.append(np.repeat(np.array([[30, 70, 255]], dtype=np.uint8), len(sphere), axis=0))
        if pred.get("object_centroid_3d_world") is not None:
            sphere = _sample_sphere(np.asarray(pred["object_centroid_3d_world"], dtype=float), radius=0.01)
            target_chunks.append(sphere)
            target_cols.append(np.repeat(np.array([[255, 255, 255]], dtype=np.uint8), len(sphere), axis=0))
    if target_chunks:
        target_pts = np.vstack(target_chunks)
        target_colors = np.vstack(target_cols)
    else:
        target_pts = np.empty((0, 3))
        target_colors = np.empty((0, 3), dtype=np.uint8)
    write_point_cloud_ply(target_pts, target_colors, out_dir / "target_points.ply")

    all_pts = np.vstack([x for x in (pre_pts, post_pts, aux_pts, dis_pts, target_pts) if len(x)]) if any(
        len(x) for x in (pre_pts, post_pts, aux_pts, dis_pts, target_pts)
    ) else np.empty((0, 3))
    all_cols = np.vstack([x for x in (pre_cols, post_cols, aux_cols, dis_cols, target_colors) if len(x)]) if any(
        len(x) for x in (pre_cols, post_cols, aux_cols, dis_cols, target_colors)
    ) else np.empty((0, 3), dtype=np.uint8)
    write_point_cloud_ply(all_pts, all_cols, out_dir / "all_debug_scene.ply")


def save_summary_figure(
    task: str,
    summary: dict[str, Any],
    predictions: list[dict[str, Any]],
    out_path: str | Path,
) -> None:
    lines = [
        f"Task: {task}",
        f"Pre masks: {summary.get('num_pre_sam_masks_filtered', 0)}",
        f"Post masks: {summary.get('num_post_sam_masks_filtered', 0)}",
        f"Pre 3D instances: {summary.get('num_pre_instances', 0)}",
        f"Post 3D instances: {summary.get('num_post_instances', 0)}",
        f"Disappeared: {summary.get('num_disappeared_candidates', 0)}",
    ]
    for idx, pred in enumerate(predictions[:6]):
        lines.append(f"Target {idx}: {np.round(pred.get('target_point_3d_world', []), 4).tolist()}")
    image = Image.new("RGB", (900, 420), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = None
    y = 30
    for line in lines:
        draw.text((30, y), line, fill=(20, 24, 30), font=font)
        y += 38
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
