"""Project final 3D harvesting results back onto RGB frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import should_write_visualization
from .mps_geometry import MPSGeometry, project_world_points_to_rgb


def write_projected_result_overlays(
    out_dir: str | Path,
    pre_geometry: MPSGeometry,
    post_geometry: MPSGeometry,
    max_frames_per_split: int = 20,
    max_points_per_instance: int = 1600,
    visualization_stride: int = 30,
) -> dict[str, Any]:
    """Write image overlays of removed tomato 3D points on RGB frames."""

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required to write projected result overlays.") from exc

    out = Path(out_dir)
    final_predictions = _load_json(out / "final_predictions.json", default={"removed_tomatoes": []})
    pre_instances = _load_json(out / "lifted_3d" / "pre_instances.json", default=[])
    frames = {
        "pre": _load_json(out / "frames" / "pre_frames.json", default=[]),
        "post": _load_json(out / "frames" / "post_frames.json", default=[]),
    }
    instance_by_id = {item["instance_id"]: item for item in pre_instances}
    overlay_root = out / "visualizations" / "projected_results"
    overlay_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"overlays": []}
    for removed in final_predictions.get("removed_tomatoes", []):
        instance = instance_by_id.get(removed.get("pre_instance_id"))
        if instance is None:
            continue
        points = _load_instance_points(instance, out)
        if len(points) == 0:
            continue
        sampled_points = _sample_points(points, max_points=max_points_per_instance)
        centroid = np.asarray(removed["tomato_centroid_3d_world"], dtype=float).reshape(1, 3)
        cut_value = removed.get("cut_point_3d_world") or removed.get("proxy_cut_point_3d_world")
        if cut_value is None:
            continue
        cut_point = np.asarray(cut_value, dtype=float).reshape(1, 3)

        pre_selected = _select_frames_for_instance(
            frames["pre"],
            instance,
            max_frames=max_frames_per_split,
            visualization_stride=visualization_stride,
        )
        post_selected = _select_evenly(
            frames["post"],
            max_frames=max_frames_per_split,
            visualization_stride=visualization_stride,
        )
        for split, selected_frames, geometry in [
            ("pre", pre_selected, pre_geometry),
            ("post", post_selected, post_geometry),
        ]:
            split_dir = overlay_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for frame in selected_frames:
                image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
                projected_points = _project_for_overlay(sampled_points, T_world_rgb, geometry)
                projected_centroid = _project_for_overlay(centroid, T_world_rgb, geometry)
                projected_cut = _project_for_overlay(cut_point, T_world_rgb, geometry)
                overlay = draw_projected_result_overlay(
                    image,
                    projected_points=projected_points,
                    projected_centroid=projected_centroid,
                    projected_cut=projected_cut,
                    split=split,
                    instance_id=instance["instance_id"],
                )
                output_path = split_dir / f"{frame['frame_id']}_{instance['instance_id']}.jpg"
                cv2.imwrite(str(output_path), overlay)
                summary["overlays"].append(
                    {
                        "split": split,
                        "frame_id": frame["frame_id"],
                        "instance_id": instance["instance_id"],
                        "image_path": str(output_path),
                        "num_projected_points": int(projected_points["valid"].sum()),
                        "centroid_visible": bool(projected_centroid["valid"].any()),
                        "proxy_cut_point_visible": bool(projected_cut["valid"].any()),
                    }
                )

    summary_path = overlay_root / "projected_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_task_projected_result_overlays(
    out_dir: str | Path,
    task_name: str,
    pre_geometry: MPSGeometry,
    post_geometry: MPSGeometry,
    max_frames_per_split: int = 20,
    max_points_per_instance: int = 1600,
    visualization_stride: int = 30,
) -> dict[str, Any]:
    """Write RGB overlays for removed targets from a task-generic run."""

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required to write projected result overlays.") from exc

    out = Path(out_dir)
    final_predictions, removed_key = _load_task_final_predictions(out, task_name)
    instances = _load_task_pre_instances(out, task_name)
    frames = {
        "pre": _load_json(out / "frames" / "pre_frames.json", default=[]),
        "post": _load_json(out / "frames" / "post_frames.json", default=[]),
    }
    instance_by_id = {item["instance_id"]: item for item in instances}
    overlay_root = out / "visualizations" / task_name / "projected_removed_targets"
    overlay_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"task": task_name, "overlays": [], "combined_overlays": []}
    targets: list[dict[str, Any]] = []
    for removed in final_predictions.get(removed_key, []):
        instance = instance_by_id.get(removed.get("pre_instance_id"))
        if instance is None:
            continue
        points = _load_instance_points(instance, out)
        if len(points) == 0:
            continue
        sampled_points = _sample_points(points, max_points=max_points_per_instance)
        centroid_value = _task_centroid_value(removed, task_name)
        action_value = _task_action_point_value(removed, task_name)
        if centroid_value is None or action_value is None:
            continue
        centroid = np.asarray(centroid_value, dtype=float).reshape(1, 3)
        action_point = np.asarray(action_value, dtype=float).reshape(1, 3)
        targets.append(
            {
                "instance": instance,
                "points": sampled_points,
                "centroid": centroid,
                "action_point": action_point,
            }
        )
        pre_selected = _select_frames_for_instance(
            frames["pre"],
            instance,
            max_frames=max_frames_per_split,
            visualization_stride=visualization_stride,
        )
        post_selected = _select_evenly(
            frames["post"],
            max_frames=max_frames_per_split,
            visualization_stride=visualization_stride,
        )

        for split, selected_frames, geometry in [
            ("pre", pre_selected, pre_geometry),
            ("post", post_selected, post_geometry),
        ]:
            split_dir = overlay_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for frame in selected_frames:
                image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
                projected_points = _project_for_overlay(sampled_points, T_world_rgb, geometry)
                projected_centroid = _project_for_overlay(centroid, T_world_rgb, geometry)
                projected_action = _project_for_overlay(action_point, T_world_rgb, geometry)
                overlay = draw_task_projected_result_overlay(
                    image,
                    projected_points=projected_points,
                    projected_centroid=projected_centroid,
                    projected_action=projected_action,
                    split=split,
                    instance_id=instance["instance_id"],
                    task_name=task_name,
                )
                output_path = split_dir / f"{frame['frame_id']}_{instance['instance_id']}.jpg"
                cv2.imwrite(str(output_path), overlay)
                summary["overlays"].append(
                    {
                        "task": task_name,
                        "split": split,
                        "frame_id": frame["frame_id"],
                        "instance_id": instance["instance_id"],
                        "image_path": str(output_path),
                        "num_projected_points": int(projected_points["valid"].sum()),
                        "centroid_visible": bool(projected_centroid["valid"].any()),
                        "action_point_visible": bool(projected_action["valid"].any()),
                    }
                )

    summary["combined_overlays"] = _write_combined_task_projected_overlays(
        cv2_mod=cv2,
        targets=targets,
        frames=frames,
        pre_geometry=pre_geometry,
        post_geometry=post_geometry,
        overlay_root=overlay_root,
        task_name=task_name,
        max_frames_per_split=max_frames_per_split,
        visualization_stride=visualization_stride,
    )
    summary_path = overlay_root / "projected_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def draw_projected_result_overlay(
    image_bgr: np.ndarray,
    projected_points: dict[str, np.ndarray],
    projected_centroid: dict[str, np.ndarray],
    projected_cut: dict[str, np.ndarray],
    split: str,
    instance_id: str,
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    uv_points = projected_points["uv"][projected_points["valid"]]
    if len(uv_points):
        hull = _convex_hull(uv_points)
        if hull is not None:
            cv2.polylines(overlay, [hull], isClosed=True, color=(0, 255, 255), thickness=2)
        step = max(1, len(uv_points) // 900)
        for uv in uv_points[::step]:
            cv2.circle(overlay, _pt(uv), 2, (0, 220, 255), -1, lineType=cv2.LINE_AA)

    if projected_centroid["valid"].any():
        centroid_uv = projected_centroid["uv"][projected_centroid["valid"]][0]
        cv2.drawMarker(
            overlay,
            _pt(centroid_uv),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=26,
            thickness=3,
        )
        cv2.putText(
            overlay,
            "tomato_centroid",
            (_pt(centroid_uv)[0] + 10, _pt(centroid_uv)[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if projected_cut["valid"].any():
        cut_uv = projected_cut["uv"][projected_cut["valid"]][0]
        cv2.drawMarker(
            overlay,
            _pt(cut_uv),
            (255, 80, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=30,
            thickness=3,
        )
        cv2.circle(overlay, _pt(cut_uv), 8, (255, 80, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            "proxy_cut_point_from_tomato_top_surface",
            (_pt(cut_uv)[0] + 10, _pt(cut_uv)[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 80, 0),
            2,
            cv2.LINE_AA,
        )

    title = f"{split} projection: {instance_id}"
    cv2.rectangle(overlay, (12, 12), (620, 96), (0, 0, 0), -1)
    cv2.putText(overlay, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        overlay,
        "yellow: disappeared tomato 3D points  white: centroid  blue: cut point",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def draw_task_projected_result_overlay(
    image_bgr: np.ndarray,
    projected_points: dict[str, np.ndarray],
    projected_centroid: dict[str, np.ndarray],
    projected_action: dict[str, np.ndarray],
    split: str,
    instance_id: str,
    task_name: str,
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    uv_points = projected_points["uv"][projected_points["valid"]]
    if len(uv_points):
        hull = _convex_hull(uv_points)
        if hull is not None:
            cv2.polylines(overlay, [hull], isClosed=True, color=(0, 255, 255), thickness=2)
        step = max(1, len(uv_points) // 900)
        for uv in uv_points[::step]:
            cv2.circle(overlay, _pt(uv), 2, (0, 220, 255), -1, lineType=cv2.LINE_AA)

    centroid_label, action_label = _task_overlay_labels(task_name)
    if projected_centroid["valid"].any():
        centroid_uv = projected_centroid["uv"][projected_centroid["valid"]][0]
        cv2.drawMarker(
            overlay,
            _pt(centroid_uv),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=26,
            thickness=3,
        )
        cv2.putText(
            overlay,
            centroid_label,
            (_pt(centroid_uv)[0] + 10, _pt(centroid_uv)[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if projected_action["valid"].any():
        action_uv = projected_action["uv"][projected_action["valid"]][0]
        cv2.drawMarker(
            overlay,
            _pt(action_uv),
            (255, 80, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=30,
            thickness=3,
        )
        cv2.circle(overlay, _pt(action_uv), 8, (255, 80, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            action_label,
            (_pt(action_uv)[0] + 10, _pt(action_uv)[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 80, 0),
            2,
            cv2.LINE_AA,
        )

    title = f"{split} projection: {task_name} {instance_id}"
    cv2.rectangle(overlay, (12, 12), (760, 96), (0, 0, 0), -1)
    cv2.putText(overlay, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        overlay,
        "yellow: removed target 3D points  white: centroid  blue: action/cut point",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _write_combined_task_projected_overlays(
    cv2_mod: Any,
    targets: list[dict[str, Any]],
    frames: dict[str, list[dict[str, Any]]],
    pre_geometry: MPSGeometry,
    post_geometry: MPSGeometry,
    overlay_root: Path,
    task_name: str,
    max_frames_per_split: int,
    visualization_stride: int,
) -> list[dict[str, Any]]:
    if not targets:
        return []

    entries: list[dict[str, Any]] = []
    for split, selected_frames, geometry in [
        (
            "pre",
            _select_evenly(
                frames["pre"],
                max_frames=max_frames_per_split,
                visualization_stride=visualization_stride,
            ),
            pre_geometry,
        ),
        (
            "post",
            _select_evenly(
                frames["post"],
                max_frames=max_frames_per_split,
                visualization_stride=visualization_stride,
            ),
            post_geometry,
        ),
    ]:
        split_dir = overlay_root / "combined" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for frame in selected_frames:
            image = cv2_mod.imread(str(frame["image_path"]), cv2_mod.IMREAD_COLOR)
            if image is None:
                continue
            T_world_rgb = geometry.get_rgb_camera_pose_at_timestamp(int(frame["timestamp_ns"]))
            projected_targets: list[dict[str, Any]] = []
            for target in targets:
                instance = target["instance"]
                projected_points = _project_for_overlay(target["points"], T_world_rgb, geometry)
                projected_centroid = _project_for_overlay(target["centroid"], T_world_rgb, geometry)
                projected_action = _project_for_overlay(target["action_point"], T_world_rgb, geometry)
                if not (
                    projected_points["valid"].any()
                    or projected_centroid["valid"].any()
                    or projected_action["valid"].any()
                ):
                    continue
                projected_targets.append(
                    {
                        "instance_id": instance["instance_id"],
                        "projected_points": projected_points,
                        "projected_centroid": projected_centroid,
                        "projected_action": projected_action,
                    }
                )
            if not projected_targets:
                continue
            overlay = draw_task_multi_projected_result_overlay(
                image,
                projected_targets=projected_targets,
                split=split,
                task_name=task_name,
            )
            output_path = split_dir / f"{frame['frame_id']}_all_removed_targets.jpg"
            cv2_mod.imwrite(str(output_path), overlay)
            entries.append(
                {
                    "task": task_name,
                    "split": split,
                    "frame_id": frame["frame_id"],
                    "image_path": str(output_path),
                    "num_targets_drawn": int(len(projected_targets)),
                    "instance_ids": [item["instance_id"] for item in projected_targets],
                }
            )
    return entries


def draw_task_multi_projected_result_overlay(
    image_bgr: np.ndarray,
    projected_targets: list[dict[str, Any]],
    split: str,
    task_name: str,
) -> np.ndarray:
    import cv2  # type: ignore

    overlay = image_bgr.copy()
    centroid_label, action_label = _task_overlay_labels(task_name)
    for target in projected_targets:
        instance_id = target["instance_id"]
        projected_points = target["projected_points"]
        projected_centroid = target["projected_centroid"]
        projected_action = target["projected_action"]

        uv_points = projected_points["uv"][projected_points["valid"]]
        if len(uv_points):
            hull = _convex_hull(uv_points)
            if hull is not None:
                cv2.polylines(overlay, [hull], isClosed=True, color=(0, 255, 255), thickness=2)
            step = max(1, len(uv_points) // 500)
            for uv in uv_points[::step]:
                cv2.circle(overlay, _pt(uv), 2, (0, 220, 255), -1, lineType=cv2.LINE_AA)

        if projected_centroid["valid"].any():
            centroid_uv = projected_centroid["uv"][projected_centroid["valid"]][0]
            cv2.drawMarker(
                overlay,
                _pt(centroid_uv),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=3,
            )
            cv2.putText(
                overlay,
                f"{instance_id} {centroid_label}",
                (_pt(centroid_uv)[0] + 8, _pt(centroid_uv)[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if projected_action["valid"].any():
            action_uv = projected_action["uv"][projected_action["valid"]][0]
            cv2.drawMarker(
                overlay,
                _pt(action_uv),
                (255, 80, 0),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=28,
                thickness=3,
            )
            cv2.circle(overlay, _pt(action_uv), 8, (255, 80, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"{instance_id} {action_label}",
                (_pt(action_uv)[0] + 8, _pt(action_uv)[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 80, 0),
                1,
                cv2.LINE_AA,
            )

    title = f"{split} combined projection: {task_name} ({len(projected_targets)} targets)"
    cv2.rectangle(overlay, (12, 12), (820, 96), (0, 0, 0), -1)
    cv2.putText(overlay, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        overlay,
        "yellow: removed target 3D points  white: centroid  blue: action/cut point",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _project_for_overlay(
    points_world: np.ndarray,
    T_world_rgb: np.ndarray,
    geometry: MPSGeometry,
) -> dict[str, np.ndarray]:
    uv, depth, valid = project_world_points_to_rgb(
        points_world,
        T_world_rgb,
        geometry.rgb_camera_calibration,
    )
    return {"uv": uv, "depth": depth, "valid": valid}


def _select_frames_for_instance(
    frames: list[dict[str, Any]],
    instance: dict[str, Any],
    max_frames: int,
    visualization_stride: int,
) -> list[dict[str, Any]]:
    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    source_ids = [fid for fid in instance.get("source_frame_ids", []) if fid in frame_by_id]
    if source_ids:
        return _select_evenly(
            [frame_by_id[fid] for fid in source_ids],
            max_frames=max_frames,
            visualization_stride=visualization_stride,
        )
    return _select_evenly(frames, max_frames=max_frames, visualization_stride=visualization_stride)


def _select_evenly(
    frames: list[dict[str, Any]],
    max_frames: int,
    visualization_stride: int,
) -> list[dict[str, Any]]:
    frames = [
        frame for frame_index, frame in enumerate(frames)
        if should_write_visualization(frame_index, visualization_stride)
    ]
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[int(idx)] for idx in sorted(set(indices.tolist()))]


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return points[indices]


def _load_instance_points(instance: dict[str, Any], out_dir: Path) -> np.ndarray:
    path = Path(instance["points_3d_path"])
    if not path.is_absolute():
        path = out_dir / path
    return np.asarray(np.load(path), dtype=float).reshape(-1, 3)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_task_final_predictions(out: Path, task_name: str) -> tuple[dict[str, Any], str]:
    if task_name == "harvest":
        return _load_json(out / "final_predictions.json", default={"removed_tomatoes": []}), "removed_tomatoes"
    if task_name == "deleaf":
        return (
            _load_json(out / "final_predictions_deleaf.json", default={"removed_petioles": []}),
            "removed_petioles",
        )
    raise ValueError(f"Unsupported task for result projection: {task_name}")


def _load_task_pre_instances(out: Path, task_name: str) -> list[dict[str, Any]]:
    task_path = out / "lifted_3d" / task_name / "pre_instances.json"
    if task_path.exists():
        return _load_json(task_path, default=[])
    return _load_json(out / "lifted_3d" / "pre_instances.json", default=[])


def _task_centroid_value(removed: dict[str, Any], task_name: str) -> list[float] | None:
    if task_name == "harvest":
        return removed.get("tomato_centroid_3d_world")
    if task_name == "deleaf":
        return removed.get("petiole_centroid_3d_world")
    return None


def _task_action_point_value(removed: dict[str, Any], task_name: str) -> list[float] | None:
    if task_name == "harvest":
        return removed.get("cut_point_3d_world") or removed.get("proxy_cut_point_3d_world")
    if task_name == "deleaf":
        return removed.get("proxy_cut_point_3d_world") or removed.get("cut_point_3d_world")
    return None


def _task_overlay_labels(task_name: str) -> tuple[str, str]:
    if task_name == "deleaf":
        return "petiole_centroid", "proxy_cut_point_from_petiole_base"
    return "tomato_centroid", "proxy_cut_point_from_tomato_top_surface"


def _convex_hull(uv_points: np.ndarray) -> np.ndarray | None:
    import cv2  # type: ignore

    if len(uv_points) < 3:
        return None
    points = np.round(uv_points).astype(np.int32).reshape(-1, 1, 2)
    return cv2.convexHull(points)


def _pt(uv: np.ndarray) -> tuple[int, int]:
    return (int(round(float(uv[0]))), int(round(float(uv[1]))))
