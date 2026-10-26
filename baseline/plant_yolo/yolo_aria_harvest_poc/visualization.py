"""Debug image and PLY visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only in minimal envs
    cv2 = None


def draw_yolo_overlay(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    masks: list[np.ndarray],
) -> np.ndarray:
    cv2_mod = _require_cv2()
    overlay = image_bgr.copy()
    color = (0, 0, 255)
    for det, mask in zip(detections, masks):
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2_mod.findContours(mask_u8, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
        cv2_mod.drawContours(overlay, contours, -1, color, 2)
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        cv2_mod.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"tomato {det['confidence']:.2f}"
        cv2_mod.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2_mod.LINE_AA,
        )
        tint = np.zeros_like(overlay)
        tint[mask_u8 > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.18, 0.0)
    return overlay


def draw_lifting_debug(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    uv_inside: np.ndarray,
    uv_foreground: np.ndarray,
) -> np.ndarray:
    cv2_mod = _require_cv2()
    overlay = image_bgr.copy()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2_mod.findContours(mask_u8, cv2_mod.RETR_EXTERNAL, cv2_mod.CHAIN_APPROX_SIMPLE)
    cv2_mod.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    for uv in np.asarray(uv_inside, dtype=float)[:: max(1, len(uv_inside) // 600 or 1)]:
        if np.isfinite(uv).all():
            cv2_mod.circle(overlay, (int(round(uv[0])), int(round(uv[1]))), 2, (255, 200, 0), -1)
    for uv in np.asarray(uv_foreground, dtype=float)[:: max(1, len(uv_foreground) // 600 or 1)]:
        if np.isfinite(uv).all():
            cv2_mod.circle(overlay, (int(round(uv[0])), int(round(uv[1]))), 3, (255, 0, 255), -1)
    return overlay


def write_visualization_plys(
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    disappearance_results: list[dict[str, Any]],
    cut_points: list[dict[str, Any]],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    vis_dir = out / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    disappeared_ids = {
        item["pre_instance_id"] for item in disappearance_results if item.get("status") == "disappeared"
    }

    pre_points = _instances_to_colored_points(pre_instances, out, np.array([255, 0, 0], dtype=np.uint8))
    post_points = _instances_to_colored_points(post_instances, out, np.array([0, 180, 0], dtype=np.uint8))
    disappeared_points = _instances_to_colored_points(
        [item for item in pre_instances if item["instance_id"] in disappeared_ids],
        out,
        np.array([255, 230, 0], dtype=np.uint8),
    )
    centroid_points = _centroid_points(pre_instances, disappeared_ids)
    cut_point_cloud = _cut_point_vertices(cut_points)

    _write_point_ply(vis_dir / "pre_instances.ply", *pre_points)
    _write_point_ply(vis_dir / "post_instances.ply", *post_points)
    _write_point_ply(vis_dir / "disappeared_tomatoes.ply", *disappeared_points)
    _write_point_ply(vis_dir / "proxy_cut_points.ply", *cut_point_cloud)

    all_points = []
    all_colors = []
    for points, colors in [pre_points, post_points, disappeared_points, centroid_points, cut_point_cloud]:
        if len(points):
            all_points.append(points)
            all_colors.append(colors)
    if all_points:
        _write_point_ply(vis_dir / "all_debug_scene.ply", np.vstack(all_points), np.vstack(all_colors))
    else:
        _write_point_ply(vis_dir / "all_debug_scene.ply", np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))


def write_summary_image(
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    disappearance_results: list[dict[str, Any]],
    final_predictions: dict[str, Any],
    out_dir: str | Path,
) -> None:
    cv2_mod = _require_cv2()
    canvas = np.full((720, 1200, 3), 255, dtype=np.uint8)
    lines = [
        "YOLO + Aria MPS Harvesting PoC",
        f"Pre tomato instances: {len(pre_instances)}",
        f"Post tomato instances: {len(post_instances)}",
        f"Disappeared tomatoes: {sum(1 for r in disappearance_results if r.get('status') == 'disappeared')}",
        f"Final removed tomatoes: {len(final_predictions.get('removed_tomatoes', []))}",
        "",
        "Final predictions:",
    ]
    for item in final_predictions.get("removed_tomatoes", [])[:12]:
        centroid = _fmt_vec(item.get("tomato_centroid_3d_world", []))
        cut = _fmt_vec(item.get("cut_point_3d_world") or item.get("proxy_cut_point_3d_world", []))
        lines.append(
            f"{item.get('pre_instance_id')} score={item.get('disappearance_score', 0):.2f} "
            f"conf={item.get('confidence', 0):.2f} centroid={centroid} cut={cut}"
        )
    if len(final_predictions.get("removed_tomatoes", [])) > 12:
        lines.append("...")

    y = 44
    for idx, line in enumerate(lines):
        scale = 0.82 if idx == 0 else 0.56
        thickness = 2 if idx == 0 else 1
        cv2_mod.putText(
            canvas,
            line,
            (32, y),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            scale,
            (25, 25, 25),
            thickness,
            cv2_mod.LINE_AA,
        )
        y += 40 if idx == 0 else 30

    path = Path(out_dir) / "visualizations" / "summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2_mod.imwrite(str(path), canvas)


def write_task_visualization_plys(
    task_name: str,
    target_semantic_class: str,
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    disappearance_results: list[dict[str, Any]],
    action_points: list[dict[str, Any]],
    out_dir: str | Path,
    main_stem_axis: dict[str, Any] | None = None,
) -> None:
    out = Path(out_dir)
    vis_dir = out / "visualizations" / task_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    disappeared_ids = {
        item["pre_instance_id"] for item in disappearance_results if item.get("status") == "disappeared"
    }
    color_pre = np.array([255, 140, 0] if task_name == "deleaf" else [220, 0, 0], dtype=np.uint8)
    pre_points = _instances_to_colored_points(pre_instances, out, color_pre)
    post_points = _instances_to_colored_points(post_instances, out, np.array([0, 180, 0], dtype=np.uint8))
    disappeared_points = _instances_to_colored_points(
        [item for item in pre_instances if item["instance_id"] in disappeared_ids],
        out,
        np.array([255, 230, 0], dtype=np.uint8),
    )
    action_point_cloud = _generic_action_point_vertices(action_points)
    stem_cloud = _main_stem_axis_vertices(main_stem_axis)

    if task_name == "deleaf":
        prefix = "petiole"
        disappeared_name = "disappeared_petioles.ply"
        action_name = "deleaf_cut_points.ply"
        all_name = "all_deleaf_debug_scene.ply"
    else:
        prefix = "tomato"
        disappeared_name = "disappeared_tomatoes.ply"
        action_name = "proxy_cut_points.ply"
        all_name = "all_harvest_debug_scene.ply"
    _write_point_ply(vis_dir / f"pre_{prefix}_instances.ply", *pre_points)
    _write_point_ply(vis_dir / f"post_{prefix}_instances.ply", *post_points)
    _write_point_ply(vis_dir / disappeared_name, *disappeared_points)
    _write_point_ply(vis_dir / action_name, *action_point_cloud)
    if task_name == "deleaf":
        _write_point_ply(vis_dir / "main_stem_axis.ply", *stem_cloud)

    all_points = []
    all_colors = []
    for points, colors in [pre_points, post_points, disappeared_points, action_point_cloud, stem_cloud]:
        if len(points):
            all_points.append(points)
            all_colors.append(colors)
    if all_points:
        _write_point_ply(vis_dir / all_name, np.vstack(all_points), np.vstack(all_colors))
    else:
        _write_point_ply(vis_dir / all_name, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))


def write_task_summary_image(
    task_name: str,
    target_semantic_class: str,
    pre_instances: list[dict[str, Any]],
    post_instances: list[dict[str, Any]],
    disappearance_results: list[dict[str, Any]],
    final_predictions: dict[str, Any],
    out_dir: str | Path,
) -> None:
    cv2_mod = _require_cv2()
    canvas = np.full((720, 1200, 3), 255, dtype=np.uint8)
    if task_name == "deleaf":
        final_key = "removed_petioles"
    else:
        final_key = "removed_tomatoes"
    lines = [
        "Aria Pre/Post Change Detection",
        f"Task: {task_name}",
        f"Pre {target_semantic_class} instances: {len(pre_instances)}",
        f"Post {target_semantic_class} instances: {len(post_instances)}",
        f"Disappeared candidates: {sum(1 for r in disappearance_results if r.get('status') == 'disappeared')}",
        f"Final removed targets: {len(final_predictions.get(final_key, []))}",
    ]
    y = 44
    for idx, line in enumerate(lines):
        cv2_mod.putText(
            canvas,
            line,
            (32, y),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.82 if idx == 0 else 0.58,
            (25, 25, 25),
            2 if idx == 0 else 1,
            cv2_mod.LINE_AA,
        )
        y += 40 if idx == 0 else 30
    path = Path(out_dir) / "visualizations" / task_name / "summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2_mod.imwrite(str(path), canvas)


def _instances_to_colored_points(
    instances: list[dict[str, Any]],
    out_dir: Path,
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    for instance in instances:
        points = _load_instance_points(instance, out_dir)
        if points.size == 0:
            continue
        points_all.append(points)
        colors_all.append(np.repeat(color.reshape(1, 3), len(points), axis=0))
    if not points_all:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(points_all), np.vstack(colors_all).astype(np.uint8)


def _centroid_points(
    instances: list[dict[str, Any]],
    highlighted_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for instance in instances:
        centroid = np.asarray(instance.get("centroid_3d", []), dtype=float)
        if centroid.shape != (3,):
            continue
        radius = 0.012 if instance["instance_id"] in highlighted_ids else 0.008
        sphere = _sphere_vertices(centroid, radius)
        points.append(sphere)
        colors.append(np.repeat(np.array([[255, 255, 255]], dtype=np.uint8), len(sphere), axis=0))
    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(points), np.vstack(colors)


def _cut_point_vertices(cut_points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for item in cut_points:
        center = np.asarray(item.get("cut_point_3d_world", []), dtype=float)
        if center.shape != (3,):
            continue
        sphere = _sphere_vertices(center, 0.015)
        points.append(sphere)
        colors.append(np.repeat(np.array([[0, 90, 255]], dtype=np.uint8), len(sphere), axis=0))
    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(points), np.vstack(colors)


def _generic_action_point_vertices(action_points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for item in action_points:
        value = (
            item.get("cut_point_3d_world")
            or item.get("removal_point_3d_world")
            or item.get("proxy_cut_point_3d_world")
        )
        center = np.asarray(value or [], dtype=float)
        if center.shape != (3,):
            continue
        sphere = _sphere_vertices(center, 0.015)
        points.append(sphere)
        colors.append(np.repeat(np.array([[0, 90, 255]], dtype=np.uint8), len(sphere), axis=0))
    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(points), np.vstack(colors)


def _main_stem_axis_vertices(main_stem_axis: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if not main_stem_axis:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    point = np.asarray(main_stem_axis.get("point_on_axis", []), dtype=float)
    direction = np.asarray(main_stem_axis.get("direction", []), dtype=float)
    if point.shape != (3,) or direction.shape != (3,):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    direction = direction / norm
    samples = np.linspace(-0.4, 0.4, 80).reshape(-1, 1)
    points = point.reshape(1, 3) + samples * direction.reshape(1, 3)
    colors = np.repeat(np.array([[160, 50, 180]], dtype=np.uint8), len(points), axis=0)
    return points, colors


def _sphere_vertices(center: np.ndarray, radius: float, rings: int = 8, sectors: int = 16) -> np.ndarray:
    vertices = []
    for r in range(rings + 1):
        phi = np.pi * r / rings
        for s in range(sectors):
            theta = 2 * np.pi * s / sectors
            vertices.append(
                center
                + radius
                * np.array(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ],
                    dtype=float,
                )
            )
    return np.asarray(vertices, dtype=float)


def _load_instance_points(instance: dict[str, Any], out_dir: Path) -> np.ndarray:
    path_value = instance.get("points_3d_path")
    if not path_value:
        return np.zeros((0, 3), dtype=float)
    path = Path(path_value)
    if not path.is_absolute():
        path = out_dir / path
    if not path.exists():
        return np.zeros((0, 3), dtype=float)
    points = np.load(path)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(colors) != len(points):
        colors = np.repeat(np.array([[255, 255, 255]], dtype=np.uint8), len(points), axis=0)

    try:
        import open3d as o3d  # type: ignore

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
        if o3d.io.write_point_cloud(str(path), cloud, write_ascii=True):
            return
    except Exception:
        pass

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _fmt_vec(values: list[float]) -> str:
    if len(values) != 3:
        return "[]"
    return "[" + ", ".join(f"{float(v):.2f}" for v in values) + "]"


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for image visualizations.")
    return cv2
