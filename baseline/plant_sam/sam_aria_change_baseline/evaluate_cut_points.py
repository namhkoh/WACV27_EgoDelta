from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_TAU_M = 0.02
POINT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def threshold_label_cm(tau_m: float) -> str:
    tau_cm = float(tau_m) * 100.0
    rounded = round(tau_cm)
    if abs(tau_cm - rounded) < 1e-9:
        return f"{int(rounded)}cm"
    return f"{tau_cm:g}cm"


def normalize_scene_id(value: str | Path) -> str:
    text = str(value).strip().replace("\\", "/")
    name = Path(text).name if "/" in text else text
    for suffix in (".vrs", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("mps_"):
        name = name[4:]
    if name.endswith("_vrs"):
        name = name[:-4]
    return name


def _as_point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        point = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(point).all():
        return None
    return point


def _unique_points(points: list[list[float]], eps: float = 1e-9) -> list[list[float]]:
    unique: list[list[float]] = []
    for point in points:
        arr = np.asarray(point, dtype=float)
        if any(np.linalg.norm(arr - np.asarray(existing, dtype=float)) <= eps for existing in unique):
            continue
        unique.append(point)
    return unique


def parse_gt_label_file(path: str | Path, task: str = "deleafing") -> dict[str, list[list[float]]]:
    """Parse the benchmark text file containing per-scene 3D task labels."""
    gt_by_scene: dict[str, list[list[float]]] = {}
    current_scene: str | None = None
    current_task: str | None = None
    task = task.lower()

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            scene_match = re.match(r"^\d+\.\s+(.+)$", line)
            if scene_match:
                current_scene = normalize_scene_id(scene_match.group(1))
                gt_by_scene.setdefault(current_scene, [])
                current_task = None
                continue
            lowered = line.lower()
            if lowered in {"deleafing", "harvesting", "clip_removal", "clip removal"}:
                current_task = lowered.replace(" ", "_")
                continue
            numbers = [float(item) for item in POINT_RE.findall(line)]
            if current_scene is not None and current_task == task and len(numbers) >= 3:
                point = numbers[:3]
                if np.isfinite(point).all():
                    gt_by_scene.setdefault(current_scene, []).append(point)

    return {scene: _unique_points(points) for scene, points in gt_by_scene.items()}


def _extract_gt_points_from_json_obj(obj: Any, task: str) -> list[list[float]]:
    """Best-effort support for future JSON GT schemas."""
    points: list[list[float]] = []
    task = task.lower()

    def visit(value: Any, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            if str(value.get("task", "")).lower() not in {"", task}:
                return
            for key in (
                "gt_cut_points_3d_world",
                "gt_cut_points",
                "cut_points_3d_world",
                "cut_points",
                "points_3d_world",
            ):
                if key in value:
                    visit(value[key], key_path + (key,))
            for key, child in value.items():
                key_l = str(key).lower()
                if "harvest" in key_l and task != "harvesting":
                    continue
                if "deleaf" in key_l and task != "deleafing":
                    continue
                if key in {"task"}:
                    continue
                visit(child, key_path + (str(key),))
            return
        if isinstance(value, list):
            point = _as_point(value)
            if point is not None and any(
                token in "_".join(key_path).lower()
                for token in ("cut", "point", "gt", task)
            ):
                points.append(point)
                return
            for child in value:
                visit(child, key_path)

    visit(obj)
    return _unique_points(points)


def load_gt_points(
    timing_json_dir: str | Path | None = None,
    gt_labels_path: str | Path | None = None,
    task: str = "deleafing",
) -> tuple[dict[str, list[list[float]]], list[str]]:
    warnings: list[str] = []
    label_path = Path(gt_labels_path) if gt_labels_path is not None else None
    if label_path is None and timing_json_dir is not None:
        timing_dir = Path(timing_json_dir)
        candidate = timing_dir.parent / "3d points labels.txt"
        if candidate.exists():
            label_path = candidate
    if label_path is not None and label_path.exists():
        gt_by_scene = parse_gt_label_file(label_path, task=task)
        if not gt_by_scene:
            warnings.append(f"No GT points parsed from {label_path}.")
        return gt_by_scene, warnings

    if timing_json_dir is None:
        warnings.append("No GT label file or timing JSON directory was provided.")
        return {}, warnings

    gt_by_scene: dict[str, list[list[float]]] = {}
    for json_path in sorted(Path(timing_json_dir).glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        scene_id = normalize_scene_id(data.get("recording_id", json_path.stem))
        gt_by_scene[scene_id] = _extract_gt_points_from_json_obj(data, task)
    warnings.append(
        "GT label text file was not found; used best-effort timing JSON parsing instead."
    )
    return gt_by_scene, warnings


def _scene_id_from_prediction_json(
    data: dict[str, Any],
    path: Path,
    pred_root: Path | None,
    known_scene_ids: set[str] | None,
    scene_id_override: str | None,
) -> str:
    if scene_id_override:
        return normalize_scene_id(scene_id_override)
    for key in ("recording_id", "scene_id", "vrs_id"):
        if data.get(key):
            return normalize_scene_id(str(data[key]))
    for key in ("vrs_path", "pre_vrs", "post_vrs"):
        if data.get(key):
            return normalize_scene_id(str(data[key]))
    if known_scene_ids:
        candidates = [path.parent.name]
        if pred_root is not None:
            try:
                candidates.extend(path.relative_to(pred_root).parts)
            except ValueError:
                pass
        for part in reversed(candidates):
            scene_id = normalize_scene_id(part)
            if scene_id in known_scene_ids:
                return scene_id
    return normalize_scene_id(path.parent.name)


def extract_prediction_points(
    data: dict[str, Any],
    task: str = "deleafing",
    include_candidate_points: bool = False,
) -> list[list[float]]:
    points: list[list[float]] = []
    task = task.lower()
    for prediction in data.get("predictions", []):
        if str(prediction.get("task", task)).lower() != task:
            continue
        if include_candidate_points:
            for point in prediction.get("candidate_target_points_3d_world", []):
                parsed = _as_point(point)
                if parsed is not None:
                    points.append(parsed)
        parsed = _as_point(prediction.get("target_point_3d_world"))
        if parsed is not None:
            points.append(parsed)
    return _unique_points(points)


def load_prediction_points(
    pred_root: str | Path,
    task: str = "deleafing",
    known_scene_ids: set[str] | None = None,
    include_candidate_points: bool = False,
    prediction_filename: str = "final_predictions.json",
    scene_id_override: str | None = None,
) -> tuple[dict[str, list[list[float]]], list[str]]:
    warnings: list[str] = []
    root = Path(pred_root)
    if root.is_file():
        prediction_paths = [root]
        root_for_ids = root.parent
    else:
        prediction_paths = sorted(root.rglob(prediction_filename))
        root_for_ids = root
    if not prediction_paths:
        warnings.append(f"No {prediction_filename} files found under {root}.")

    pred_by_scene: dict[str, list[list[float]]] = {}
    for path in prediction_paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            warnings.append(f"Skipped unreadable prediction file {path}: {exc}")
            continue
        scene_id = _scene_id_from_prediction_json(
            data=data,
            path=path,
            pred_root=root_for_ids,
            known_scene_ids=known_scene_ids,
            scene_id_override=scene_id_override if len(prediction_paths) == 1 else None,
        )
        points = extract_prediction_points(
            data,
            task=task,
            include_candidate_points=include_candidate_points,
        )
        if scene_id in pred_by_scene:
            pred_by_scene[scene_id].extend(points)
            pred_by_scene[scene_id] = _unique_points(pred_by_scene[scene_id])
        else:
            pred_by_scene[scene_id] = points
    return pred_by_scene, warnings


def _linear_sum_assignment(cost: np.ndarray, invalid_cost: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except Exception:
        # Greedy fallback over valid pair costs; exact Hungarian is used when scipy is available.
        max_valid_cost = float(invalid_cost) if invalid_cost is not None else math.inf
        pairs = sorted(
            (
                (float(cost[row, col]), row, col)
                for row in range(cost.shape[0])
                for col in range(cost.shape[1])
                if float(cost[row, col]) < max_valid_cost
            ),
            key=lambda item: item[0],
        )
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        rows: list[int] = []
        cols: list[int] = []
        for _, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            rows.append(row)
            cols.append(col)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def match_cut_points(
    pred_points: list[list[float]] | np.ndarray,
    gt_points: list[list[float]] | np.ndarray,
    tau_m: float = DEFAULT_TAU_M,
) -> dict[str, Any]:
    pred = np.asarray(pred_points, dtype=float).reshape((-1, 3))
    gt = np.asarray(gt_points, dtype=float).reshape((-1, 3))
    if len(pred) == 0 or len(gt) == 0:
        return {"matches": [], "matched_distances": []}

    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    invalid_cost = max(float(tau_m) * 1000.0, 1e6)
    cost = np.where(distances <= tau_m, distances, invalid_cost)
    rows, cols = _linear_sum_assignment(cost, invalid_cost=invalid_cost)
    matches: list[dict[str, Any]] = []
    matched_distances: list[float] = []
    for row, col in zip(rows, cols):
        distance = float(distances[int(row), int(col)])
        if distance <= tau_m:
            matches.append({"pred_index": int(row), "gt_index": int(col), "distance_m": distance})
            matched_distances.append(distance)
    return {"matches": matches, "matched_distances": matched_distances}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    num_pred = tp + fp
    num_gt = tp + fn
    precision = float(tp / num_pred) if num_pred > 0 else (1.0 if num_gt == 0 else 0.0)
    recall = float(tp / num_gt) if num_gt > 0 else 1.0
    f1 = 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))
    return precision, recall, f1


def evaluate_scene(
    scene_id: str,
    pred_points: list[list[float]],
    gt_points: list[list[float]],
    tau_m: float = DEFAULT_TAU_M,
) -> dict[str, Any]:
    match_result = match_cut_points(pred_points, gt_points, tau_m=tau_m)
    matched_distances = [float(item) for item in match_result["matched_distances"]]
    tp = len(matched_distances)
    num_pred = len(pred_points)
    num_gt = len(gt_points)
    fp = num_pred - tp
    fn = num_gt - tp
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    threshold_label = threshold_label_cm(tau_m)

    gt_to_nearest: list[float] = []
    if num_gt > 0 and num_pred > 0:
        pred = np.asarray(pred_points, dtype=float).reshape((-1, 3))
        gt = np.asarray(gt_points, dtype=float).reshape((-1, 3))
        gt_to_nearest = np.min(np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2), axis=1).tolist()

    return {
        "scene_id": scene_id,
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        f"Precision@{threshold_label}": precision,
        f"Recall@{threshold_label}": recall,
        f"F1@{threshold_label}": f1,
        "matched_mean_distance": _mean(matched_distances),
        "gt_to_nearest_mean_distance": _mean([float(item) for item in gt_to_nearest]),
        "matched_distances": matched_distances,
        "gt_to_nearest_distances": [float(item) for item in gt_to_nearest],
        "matches": match_result["matches"],
    }


def aggregate_scene_metrics(per_scene: list[dict[str, Any]], tau_m: float = DEFAULT_TAU_M) -> dict[str, Any]:
    total_tp = int(sum(item["TP"] for item in per_scene))
    total_fp = int(sum(item["FP"] for item in per_scene))
    total_fn = int(sum(item["FN"] for item in per_scene))
    precision, recall, f1 = _precision_recall_f1(total_tp, total_fp, total_fn)
    threshold_label = threshold_label_cm(tau_m)

    matched_distances = [
        float(distance)
        for item in per_scene
        for distance in item.get("matched_distances", [])
        if math.isfinite(float(distance))
    ]
    gt_to_nearest_distances = [
        float(distance)
        for item in per_scene
        for distance in item.get("gt_to_nearest_distances", [])
        if math.isfinite(float(distance))
    ]
    num_gt_without_prediction = int(
        sum(item["num_gt"] for item in per_scene if item["num_gt"] > 0 and item["num_pred"] == 0)
    )
    return {
        "tau_m": float(tau_m),
        "num_scenes": int(len(per_scene)),
        "total_num_gt": int(sum(item["num_gt"] for item in per_scene)),
        "total_num_pred": int(sum(item["num_pred"] for item in per_scene)),
        "total_TP": total_tp,
        "total_FP": total_fp,
        "total_FN": total_fn,
        f"micro_Precision@{threshold_label}": precision,
        f"micro_Recall@{threshold_label}": recall,
        f"micro_F1@{threshold_label}": f1,
        "matched_mean_distance": _mean(matched_distances),
        "matched_median_distance": _median(matched_distances),
        "gt_to_nearest_mean_distance": _mean(gt_to_nearest_distances),
        "gt_to_nearest_median_distance": _median(gt_to_nearest_distances),
        "num_matched_distances": int(len(matched_distances)),
        "num_gt_to_nearest_distances": int(len(gt_to_nearest_distances)),
        "num_gt_without_prediction": num_gt_without_prediction,
    }


def evaluate_cut_point_sets(
    gt_by_scene: dict[str, list[list[float]]],
    pred_by_scene: dict[str, list[list[float]]],
    tau_m: float = DEFAULT_TAU_M,
    include_prediction_only_scenes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_ids = set(gt_by_scene)
    if include_prediction_only_scenes:
        scene_ids.update(pred_by_scene)
    per_scene = [
        evaluate_scene(
            scene_id=scene_id,
            pred_points=pred_by_scene.get(scene_id, []),
            gt_points=gt_by_scene.get(scene_id, []),
            tau_m=tau_m,
        )
        for scene_id in sorted(scene_ids)
    ]
    return per_scene, aggregate_scene_metrics(per_scene, tau_m=tau_m)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_per_scene_csv(per_scene: list[dict[str, Any]], path: Path, tau_m: float) -> None:
    threshold_label = threshold_label_cm(tau_m)
    fieldnames = [
        "scene_id",
        "num_gt",
        "num_pred",
        "TP",
        "FP",
        "FN",
        f"Precision@{threshold_label}",
        f"Recall@{threshold_label}",
        f"F1@{threshold_label}",
        "matched_mean_distance",
        "gt_to_nearest_mean_distance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_scene:
            writer.writerow({key: item.get(key, "") if item.get(key) is not None else "" for key in fieldnames})


def save_evaluation_results(
    per_scene: list[dict[str, Any]],
    aggregate: dict[str, Any],
    out_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_scene_json": str(out_path / "per_scene_metrics.json"),
        "per_scene_csv": str(out_path / "per_scene_metrics.csv"),
        "aggregate_json": str(out_path / "aggregate_metrics.json"),
        "combined_json": str(out_path / "cutpoint_eval_results.json"),
    }
    tau_m = float(aggregate.get("tau_m", DEFAULT_TAU_M))
    with Path(paths["per_scene_json"]).open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(per_scene), handle, indent=2)
    _write_per_scene_csv(per_scene, Path(paths["per_scene_csv"]), tau_m=tau_m)
    with Path(paths["aggregate_json"]).open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(aggregate), handle, indent=2)
    with Path(paths["combined_json"]).open("w", encoding="utf-8") as handle:
        json.dump(
            _json_ready({"metadata": metadata or {}, "aggregate": aggregate, "per_scene": per_scene}),
            handle,
            indent=2,
        )
    return paths


def _load_timing_scene_metadata(timing_json_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    if timing_json_dir is None:
        return {}
    timing_dir = Path(timing_json_dir)
    if not timing_dir.exists():
        return {}
    scenes: dict[str, dict[str, Any]] = {}
    for path in sorted(timing_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        scene_id = normalize_scene_id(data.get("recording_id", path.stem))
        scenes[scene_id] = data
    return scenes


def _prediction_output_dirs(
    pred_root: str | Path,
    known_scene_ids: set[str],
    prediction_filename: str,
) -> dict[str, Path]:
    root = Path(pred_root)
    paths = [root] if root.is_file() else sorted(root.rglob(prediction_filename))
    mapping: dict[str, Path] = {}
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        scene_id = _scene_id_from_prediction_json(
            data=data,
            path=path,
            pred_root=root.parent if root.is_file() else root,
            known_scene_ids=known_scene_ids,
            scene_id_override=None,
        )
        mapping[scene_id] = path.parent
    return mapping


def _draw_point_marker(
    draw: ImageDraw.ImageDraw,
    uv: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    radius: int = 7,
    cross: bool = False,
) -> None:
    x, y = float(uv[0]), float(uv[1])
    if cross:
        draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=3)
        draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=3)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    draw.text((x + radius + 3, y - radius - 2), label, fill=color)


def _select_visible_frames(
    frames: list[dict[str, Any]],
    geometry: Any,
    points: np.ndarray,
    max_frames: int,
) -> list[tuple[int, dict[str, Any], np.ndarray, np.ndarray]]:
    scored: list[tuple[int, int, dict[str, Any], np.ndarray, np.ndarray]] = []
    for order, frame in enumerate(frames):
        if points.size == 0:
            visible = np.zeros((0,), dtype=bool)
            uv = np.zeros((0, 2), dtype=float)
        else:
            uv, _, visible = geometry.project_world_points_to_frame(
                points_world=points,
                timestamp_ns=int(frame["timestamp_ns"]),
                image_width=int(frame["width"]),
                image_height=int(frame["height"]),
            )
        scored.append((int(np.sum(visible)), -order, frame, uv, visible))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        (visible_count, frame, uv, visible)
        for visible_count, _, frame, uv, visible in scored[: max(1, max_frames)]
        if visible_count > 0
    ]


def _select_first_middle_final_frames(
    frames: list[dict[str, Any]],
    geometry: Any,
    points: np.ndarray,
    max_frames: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    if not frames:
        return []
    candidates = [
        ("first", 0),
        ("middle", len(frames) // 2),
        ("final", len(frames) - 1),
    ]
    selected: list[tuple[str, int, dict[str, Any]]] = []
    seen_indices: set[int] = set()
    for label, idx in candidates:
        idx = int(np.clip(idx, 0, len(frames) - 1))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        frame = frames[idx]
        if points.size == 0:
            visible_count = 0
        else:
            _, _, visible = geometry.project_world_points_to_frame(
                points_world=points,
                timestamp_ns=int(frame["timestamp_ns"]),
                image_width=int(frame["width"]),
                image_height=int(frame["height"]),
            )
            visible_count = int(np.sum(visible))
        selected.append((label, visible_count, frame))
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def _load_extracted_projection_context(
    scene_out: Path,
    task: str,
) -> tuple[list[dict[str, Any]], Any] | None:
    from .config import PipelineConfig
    from .io_aria import load_frames_metadata
    from .mps_geometry import AriaMpsGeometry

    frames_json = scene_out / "frames" / "pre_frames.json"
    if not frames_json.exists():
        return None
    frames = load_frames_metadata(frames_json)
    if not frames:
        return None
    image_path = Path(frames[0].get("image_path", ""))
    source_root = image_path.parent
    camera_json = source_root / "camera_pinhole.json"
    poses_csv = source_root / "poses.csv"
    if not camera_json.exists() or not poses_csv.exists():
        return None
    config = PipelineConfig(task=task, use_undistorted_rgb=True)
    geometry = AriaMpsGeometry.from_pinhole_files(
        source_root=source_root,
        points_source=None,
        camera_json=camera_json,
        poses_csv=poses_csv,
        config=config,
        world_frame_id=scene_out.name,
    )
    return frames, geometry


def _load_legacy_projection_context(
    scene_id: str,
    scene_out: Path,
    timing: dict[str, Any],
    task: str,
    rgb_stream_id: str,
) -> tuple[list[dict[str, Any]], Any, bool]:
    from .config import PipelineConfig
    from .io_aria import load_frames_metadata
    from .mps_geometry import AriaMpsGeometry

    frames_root = scene_out / "frames_undistorted"
    use_undistorted = True
    if not (frames_root / "pre_frames.json").exists():
        frames_root = scene_out / "frames"
        use_undistorted = False
    frames_json = frames_root / "pre_frames.json"
    if not frames_json.exists():
        raise FileNotFoundError(f"{scene_id}: missing pre frame metadata under {frames_root}.")
    vrs_path = timing.get("vrs_path")
    mps_path = timing.get("mps_dir")
    if not vrs_path or not mps_path:
        raise RuntimeError(f"{scene_id}: timing JSON lacks vrs_path or mps_dir.")
    config = PipelineConfig(
        task=task,
        rgb_stream_id=rgb_stream_id,
        use_undistorted_rgb=use_undistorted,
    )
    frames = load_frames_metadata(frames_json)
    geometry = AriaMpsGeometry(vrs_path, mps_path, rgb_stream_id, config)
    return frames, geometry, use_undistorted


def save_evaluation_image_overlays(
    per_scene: list[dict[str, Any]],
    gt_by_scene: dict[str, list[list[float]]],
    pred_by_scene: dict[str, list[list[float]]],
    pred_root: str | Path,
    out_dir: str | Path,
    timing_json_dir: str | Path | None,
    task: str,
    rgb_stream_id: str,
    prediction_filename: str = "final_predictions.json",
    max_scenes: int | None = None,
    max_frames_per_scene: int = 3,
) -> dict[str, Any]:
    out_path = Path(out_dir) / "evaluation_overlays"
    out_path.mkdir(parents=True, exist_ok=True)
    timing_by_scene = _load_timing_scene_metadata(timing_json_dir)
    output_dirs = _prediction_output_dirs(pred_root, set(gt_by_scene) | set(pred_by_scene), prediction_filename)
    rows = sorted(
        per_scene,
        key=lambda item: (item.get("num_gt", 0) + item.get("num_pred", 0), item.get("scene_id", "")),
        reverse=True,
    )
    if max_scenes is not None and max_scenes > 0:
        rows = rows[:max_scenes]

    stats = {
        "output_dir": str(out_path),
        "num_scenes_attempted": int(len(rows)),
        "num_scenes_written": 0,
        "num_images_written": 0,
        "warnings": [],
        "frame_selection": "first_middle_final",
        "projection_context_counts": {"extracted_pinhole": 0, "legacy_vrs_mps": 0},
    }
    for row in rows:
        scene_id = str(row["scene_id"])
        scene_out = output_dirs.get(scene_id)
        timing = timing_by_scene.get(scene_id, {})
        if scene_out is None:
            stats["warnings"].append(f"{scene_id}: no prediction output directory found.")
            continue
        try:
            context = _load_extracted_projection_context(scene_out, task)
            projection_context = "extracted_pinhole"
            if context is None:
                frames, geometry, _ = _load_legacy_projection_context(
                    scene_id=scene_id,
                    scene_out=scene_out,
                    timing=timing,
                    task=task,
                    rgb_stream_id=rgb_stream_id,
                )
                projection_context = "legacy_vrs_mps"
            else:
                frames, geometry = context
            stats["projection_context_counts"][projection_context] += 1
        except Exception as exc:
            stats["warnings"].append(f"{scene_id}: failed to initialize visualization geometry: {exc}")
            continue

        gt = np.asarray(gt_by_scene.get(scene_id, []), dtype=float).reshape((-1, 3))
        pred = np.asarray(pred_by_scene.get(scene_id, []), dtype=float).reshape((-1, 3))
        points = np.vstack([gt, pred]) if len(gt) and len(pred) else (gt if len(gt) else pred)
        selected = _select_first_middle_final_frames(frames, geometry, points, max_frames=max_frames_per_scene)
        if not selected:
            stats["warnings"].append(f"{scene_id}: no frames available for evaluation overlay.")
            continue

        scene_dir = out_path / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        matched_pred = {int(match["pred_index"]) for match in row.get("matches", [])}
        matched_gt = {int(match["gt_index"]) for match in row.get("matches", [])}

        scene_written = 0
        for frame_label, visible_count, frame in selected:
            image_path = Path(frame["image_path"])
            if not image_path.exists():
                stats["warnings"].append(f"{scene_id}: missing frame image {image_path}.")
                continue
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            gt_uv, _, gt_valid = geometry.project_world_points_to_frame(
                gt,
                int(frame["timestamp_ns"]),
                int(frame["width"]),
                int(frame["height"]),
            )
            pred_uv, _, pred_valid = geometry.project_world_points_to_frame(
                pred,
                int(frame["timestamp_ns"]),
                int(frame["width"]),
                int(frame["height"]),
            )
            for match in row.get("matches", []):
                pi = int(match["pred_index"])
                gi = int(match["gt_index"])
                if pi < len(pred_uv) and gi < len(gt_uv) and pred_valid[pi] and gt_valid[gi]:
                    draw.line(
                        [tuple(gt_uv[gi].tolist()), tuple(pred_uv[pi].tolist())],
                        fill=(0, 170, 255),
                        width=2,
                    )
            for gi, uv in enumerate(gt_uv):
                if gt_valid[gi]:
                    color = (0, 255, 0) if gi in matched_gt else (80, 255, 80)
                    _draw_point_marker(draw, uv, color, f"GT{gi}", radius=7, cross=False)
            for pi, uv in enumerate(pred_uv):
                if pred_valid[pi]:
                    color = (0, 170, 255) if pi in matched_pred else (255, 60, 60)
                    _draw_point_marker(draw, uv, color, f"P{pi}", radius=7, cross=pi not in matched_pred)
            draw.rectangle((0, 0, int(frame["width"]), 54), fill=(0, 0, 0))
            draw.text(
                (8, 6),
                (
                    f"{scene_id} {frame_label}:{frame['frame_id']} | "
                    f"GT={row['num_gt']} Pred={row['num_pred']} "
                    f"TP/FP/FN={row['TP']}/{row['FP']}/{row['FN']} "
                    "green=GT red=FP blue=TP"
                ),
                fill=(255, 255, 255),
            )
            draw.text(
                (8, 28),
                f"visible points in frame: {visible_count} | projection: {projection_context}",
                fill=(220, 220, 220),
            )
            output_path = scene_dir / f"{frame_label}_{frame['frame_id']}_eval_overlay.jpg"
            image.save(output_path, quality=92)
            scene_written += 1
            stats["num_images_written"] += 1
        if scene_written:
            stats["num_scenes_written"] += 1
    with (out_path / "overlay_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(stats), handle, indent=2)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 3D cut-point predictions with one-to-one matching.")
    parser.add_argument("--task", default="deleafing", choices=("deleafing", "harvesting"))
    parser.add_argument("--pred-root", required=True, help="Prediction file or directory containing final_predictions.json files.")
    parser.add_argument("--out", required=True, help="Directory for per-scene and aggregate metrics.")
    parser.add_argument("--timing-json-dir", default="/path/to/data/mps/egodelta_benchmark_splitted/timing_json")
    parser.add_argument("--gt-labels", help="Path to the benchmark 3D point label text file.")
    parser.add_argument("--tau-m", type=float, default=DEFAULT_TAU_M)
    parser.add_argument("--prediction-filename", default="final_predictions.json")
    parser.add_argument("--scene-id", help="Scene id override when --pred-root is a single prediction file.")
    parser.add_argument(
        "--include-candidate-points",
        action="store_true",
        help="Also evaluate candidate_target_points_3d_world entries as predictions.",
    )
    parser.add_argument(
        "--exclude-prediction-only-scenes",
        action="store_true",
        help="Ignore prediction files whose scene id is not present in GT.",
    )
    parser.add_argument("--visualize-images", action="store_true", help="Save GT-vs-pred cut-point overlays on pre RGB frames.")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--vis-max-scenes", type=int, default=0, help="Maximum scenes to visualize; 0 means all scenes.")
    parser.add_argument(
        "--vis-max-frames-per-scene",
        type=int,
        default=3,
        help="Maximum evaluation overlay frames per scene. The default writes first, middle, and final pre frames.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    gt_by_scene, gt_warnings = load_gt_points(
        timing_json_dir=args.timing_json_dir,
        gt_labels_path=args.gt_labels,
        task=args.task,
    )
    pred_by_scene, pred_warnings = load_prediction_points(
        pred_root=args.pred_root,
        task=args.task,
        known_scene_ids=set(gt_by_scene),
        include_candidate_points=args.include_candidate_points,
        prediction_filename=args.prediction_filename,
        scene_id_override=args.scene_id,
    )
    scene_id_override = normalize_scene_id(args.scene_id) if args.scene_id else None
    if scene_id_override and Path(args.pred_root).is_file():
        gt_by_scene = {scene_id_override: gt_by_scene.get(scene_id_override, [])}
        pred_by_scene = {scene_id_override: pred_by_scene.get(scene_id_override, [])}

    per_scene, aggregate = evaluate_cut_point_sets(
        gt_by_scene=gt_by_scene,
        pred_by_scene=pred_by_scene,
        tau_m=args.tau_m,
        include_prediction_only_scenes=not args.exclude_prediction_only_scenes,
    )
    metadata = {
        "task": args.task,
        "tau_m": float(args.tau_m),
        "pred_root": str(args.pred_root),
        "timing_json_dir": str(args.timing_json_dir) if args.timing_json_dir else None,
        "gt_labels": str(args.gt_labels) if args.gt_labels else None,
        "include_candidate_points": bool(args.include_candidate_points),
        "warnings": gt_warnings + pred_warnings,
    }
    paths = save_evaluation_results(per_scene, aggregate, args.out, metadata=metadata)
    overlay_stats = None
    if args.visualize_images:
        overlay_stats = save_evaluation_image_overlays(
            per_scene=per_scene,
            gt_by_scene=gt_by_scene,
            pred_by_scene=pred_by_scene,
            pred_root=args.pred_root,
            out_dir=args.out,
            timing_json_dir=args.timing_json_dir,
            task=args.task,
            rgb_stream_id=args.rgb_stream_id,
            prediction_filename=args.prediction_filename,
            max_scenes=args.vis_max_scenes if args.vis_max_scenes > 0 else None,
            max_frames_per_scene=args.vis_max_frames_per_scene,
        )
        metadata["evaluation_overlay_stats"] = overlay_stats
        save_evaluation_results(per_scene, aggregate, args.out, metadata=metadata)
    print("===== Cut-Point Evaluation =====")
    print(f"Task: {args.task}")
    print(f"Tau: {args.tau_m:.4f} m")
    print(f"Scenes: {aggregate['num_scenes']}")
    print(f"Total GT / Pred: {aggregate['total_num_gt']} / {aggregate['total_num_pred']}")
    threshold_label = threshold_label_cm(args.tau_m)
    print(
        "Micro P/R/F1: "
        f"{aggregate[f'micro_Precision@{threshold_label}']:.4f} / "
        f"{aggregate[f'micro_Recall@{threshold_label}']:.4f} / "
        f"{aggregate[f'micro_F1@{threshold_label}']:.4f}"
    )
    print(f"Matched mean / median distance: {aggregate['matched_mean_distance']} / {aggregate['matched_median_distance']}")
    print(
        "GT-nearest mean / median distance: "
        f"{aggregate['gt_to_nearest_mean_distance']} / {aggregate['gt_to_nearest_median_distance']}"
    )
    print(f"Output: {paths['combined_json']}")
    if overlay_stats is not None:
        print(f"Evaluation overlays: {overlay_stats['output_dir']} ({overlay_stats['num_images_written']} images)")
    for warning in metadata["warnings"]:
        print(f"Warning: {warning}")
    if overlay_stats is not None:
        for warning in overlay_stats.get("warnings", []):
            print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
