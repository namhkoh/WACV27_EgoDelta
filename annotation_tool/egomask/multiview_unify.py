from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from .ply import read_gaussian_projection_fields, read_vertex_fields
from .projection import Camera, build_scene_depth, rasterize_components
from .storage import AnnotationStore


DEFAULT_SIGMAS = (0.35, 0.50, 0.65, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25)
DEFAULT_TRUSTED = ("annotator_3", "annotator_4")
DEFAULT_TRANSFER = ("annotator_1", "annotator_2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit each scene's Gaussian footprint to trusted human silhouettes and create "
            "non-destructive mask overrides for projected-mask-style views."
        )
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, help="CSV containing task_id and reason columns")
    parser.add_argument("--trusted-annotator", action="append", dest="trusted_annotators")
    parser.add_argument("--transfer-annotator", action="append", dest="transfer_annotators")
    parser.add_argument(
        "--sigma-grid",
        default=",".join(str(value) for value in DEFAULT_SIGMAS),
        help="Comma-separated positive footprint_sigma candidates",
    )
    parser.add_argument("--scene", action="append", default=[], help="Limit to this scene ID; repeatable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_multiview_overrides(
            args.run,
            args.output,
            trusted_annotators=args.trusted_annotators or DEFAULT_TRUSTED,
            transfer_annotators=args.transfer_annotators or DEFAULT_TRANSFER,
            review_queue=args.review_queue,
            sigmas=_parse_sigmas(args.sigma_grid),
            selected_scenes=set(args.scene),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Created {len(payload['overrides'])} non-destructive overrides across "
        f"{len(payload['calibration'])} scenes in {Path(args.output).resolve()}"
    )
    return 0


def build_multiview_overrides(
    run_root: str | Path,
    output: str | Path,
    *,
    trusted_annotators: tuple[str, ...] | list[str] = DEFAULT_TRUSTED,
    transfer_annotators: tuple[str, ...] | list[str] = DEFAULT_TRANSFER,
    review_queue: str | Path | None = None,
    sigmas: tuple[float, ...] | list[float] = DEFAULT_SIGMAS,
    selected_scenes: set[str] | None = None,
) -> dict:
    store = AnnotationStore(run_root)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    trusted = tuple(dict.fromkeys(trusted_annotators))
    transfer = tuple(dict.fromkeys(transfer_annotators))
    known_annotators = {row["id"] for row in store.manifest["annotators"]}
    unknown = set(trusted + transfer).difference(known_annotators)
    if unknown:
        raise ValueError(f"Unknown annotators: {sorted(unknown)}")
    overlap = set(trusted).intersection(transfer)
    if overlap:
        raise ValueError(f"Trusted and transfer annotators overlap: {sorted(overlap)}")
    sigmas = tuple(sorted(set(float(value) for value in sigmas)))
    if not sigmas or any(not np.isfinite(value) or value <= 0 for value in sigmas):
        raise ValueError("Footprint sigma candidates must be finite and positive")

    review_rows, review_metadata = _load_review_queue(review_queue)
    review_by_task = {row["task_id"]: row for row in review_rows}
    known_tasks = {task["task_id"] for task in store.manifest["tasks"]}
    unknown_review = sorted(set(review_by_task).difference(known_tasks))
    if unknown_review:
        raise ValueError(f"Review queue contains unknown tasks: {unknown_review[:5]}")

    scene_rows = {row["scene_id"]: row for row in store.manifest["scenes"]}
    tasks_by_scene: dict[str, list[dict]] = {}
    for task in store.manifest["tasks"]:
        tasks_by_scene.setdefault(task["scene_id"], []).append(task)
    scenes = sorted(tasks_by_scene)
    if selected_scenes:
        missing = sorted(selected_scenes.difference(scenes))
        if missing:
            raise ValueError(f"Unknown scene IDs: {missing}")
        scenes = [scene for scene in scenes if scene in selected_scenes]

    calibration_rows: list[dict] = []
    override_rows: list[dict] = []
    masks_root = output / "masks"
    for scene_index, scene_id in enumerate(scenes, start=1):
        tasks = sorted(tasks_by_scene[scene_id], key=lambda row: row["image_name"])
        calibration_tasks = [
            task
            for task in tasks
            if task["assigned_to"] in trusted and task["task_id"] not in review_by_task
        ]
        override_tasks = [
            task
            for task in tasks
            if task["assigned_to"] in transfer or task["task_id"] in review_by_task
        ]
        if len(calibration_tasks) < 2:
            raise ValueError(
                f"{scene_id}: need at least two unflagged trusted views, found {len(calibration_tasks)}"
            )

        components = _load_components(tasks[0])
        scene_points = _load_scene_points(scene_rows[scene_id])
        required_ids = {task["task_id"] for task in calibration_tasks + override_tasks}
        cameras: dict[str, Camera] = {}
        depth_maps: dict[str, np.ndarray] = {}
        for task in tasks:
            if task["task_id"] not in required_ids:
                continue
            camera = _camera_from_task(task)
            cameras[task["task_id"]] = camera
            depth_maps[task["task_id"]] = build_scene_depth(scene_points, camera)

        scores = np.empty((len(calibration_tasks), len(sigmas)), dtype=np.float64)
        for task_index, task in enumerate(calibration_tasks):
            target = _load_source_mask(store, task)
            for sigma_index, sigma in enumerate(sigmas):
                rendered, _ = rasterize_components(
                    components,
                    cameras[task["task_id"]],
                    scene_depth=depth_maps[task["task_id"]],
                    footprint_sigma=sigma,
                )
                scores[task_index, sigma_index] = _iou(np.asarray(rendered) > 0, target)

        best_index, leave_one_out = choose_footprint_sigma(scores, sigmas)
        best_sigma = sigmas[best_index]
        initial_index = min(range(len(sigmas)), key=lambda index: abs(sigmas[index] - 2.25))
        calibration_rows.append(
            {
                "scene_id": scene_id,
                "task_kind": tasks[0]["task_kind"],
                "trusted_view_count": len(calibration_tasks),
                "trusted_task_ids": [task["task_id"] for task in calibration_tasks],
                "best_sigma": best_sigma,
                "initial_sigma": sigmas[initial_index],
                "initial_mean_iou": float(scores[:, initial_index].mean()),
                "fitted_mean_iou": float(scores[:, best_index].mean()),
                "leave_one_out_mean_iou": float(np.mean(leave_one_out)),
            }
        )

        for task in override_tasks:
            rendered, _ = rasterize_components(
                components,
                cameras[task["task_id"]],
                scene_depth=depth_maps[task["task_id"]],
                footprint_sigma=best_sigma,
            )
            binary = Image.fromarray((np.asarray(rendered) > 0).astype(np.uint8), mode="L")
            relative_mask = Path("masks") / f"{task['task_id']}.png"
            _save_png_atomic(output / relative_mask, binary)
            review = review_by_task.get(task["task_id"])
            reason = "transfer calibrated 3D footprint into projected-mask-style view"
            if review:
                reason = f"replace queued mask: {review.get('reason', review.get('action', 'review'))}"
            source_annotation = store.annotation_mask_path(task["assigned_to"], task["task_id"])
            override_rows.append(
                {
                    "task_id": task["task_id"],
                    "mask": str(relative_mask),
                    "reason": reason,
                    "source_annotator": task["assigned_to"],
                    "source_annotation_sha256": _sha256(source_annotation),
                    "scene_id": scene_id,
                    "calibrated_footprint_sigma": best_sigma,
                }
            )
        print(
            f"[{scene_index}/{len(scenes)}] {scene_id}: sigma={best_sigma:.2f}, "
            f"trusted={len(calibration_tasks)}, overrides={len(override_tasks)}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "multiview_gaussian_footprint_calibration",
        "status": "candidate_requires_visual_adjudication_before_final_freeze",
        "source_run": str(store.run_root),
        "source_manifest_sha256": _sha256(store.run_root / "manifest.json"),
        "trusted_annotators": list(trusted),
        "transfer_annotators": list(transfer),
        "sigma_grid": list(sigmas),
        "review_queue": review_metadata,
        "calibration": calibration_rows,
        "overrides": override_rows,
    }
    _write_json_atomic(output / "overrides.json", payload)
    return payload


def choose_footprint_sigma(
    scores: np.ndarray,
    sigmas: tuple[float, ...] | list[float],
) -> tuple[int, list[float]]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(sigmas) or scores.shape[0] < 2:
        raise ValueError("scores must contain at least two views and one column per sigma")
    best_index = int(np.argmax(scores.mean(axis=0)))
    leave_one_out: list[float] = []
    for held_out in range(scores.shape[0]):
        training = np.delete(scores, held_out, axis=0).mean(axis=0)
        selected = int(np.argmax(training))
        leave_one_out.append(float(scores[held_out, selected]))
    return best_index, leave_one_out


def _load_components(task: dict) -> list[dict]:
    components = []
    for row in task["components"]:
        fields = read_gaussian_projection_fields(row["source_ply"])
        components.append(
            {
                "xyz": np.column_stack((fields["x"], fields["y"], fields["z"])),
                "world_scale": fields["world_scale"],
                "alpha": fields["alpha"],
            }
        )
    return components


def _load_scene_points(scene: dict) -> np.ndarray:
    fields = read_vertex_fields(scene["source_pre_ply"], ("x", "y", "z"))
    return np.column_stack((fields["x"], fields["y"], fields["z"]))


def _camera_from_task(task: dict) -> Camera:
    row = task["camera"]
    return Camera(
        image_name=task["image_name"],
        width=int(row["width"]),
        height=int(row["height"]),
        fx=float(row["fx"]),
        fy=float(row["fy"]),
        cx=float(row["cx"]),
        cy=float(row["cy"]),
        center=np.asarray(row["camera_center_world"], dtype=np.float64),
        rotation=np.asarray(row["rotation_camera_to_world"], dtype=np.float64),
    )


def _load_source_mask(store: AnnotationStore, task: dict) -> np.ndarray:
    path = store.annotation_mask_path(task["assigned_to"], task["task_id"])
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _load_review_queue(path: str | Path | None) -> tuple[list[dict], dict | None]:
    if path is None:
        return [], None
    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "task_id" not in rows[0]:
        raise ValueError(f"Review queue has no task_id column: {path}")
    return rows, {"path": str(path), "sha256": _sha256(path), "task_count": len(rows)}


def _parse_sigmas(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --sigma-grid: {value!r}") from exc


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, target).sum() / union)


def _save_png_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
