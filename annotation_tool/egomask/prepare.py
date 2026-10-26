from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile

import numpy as np

from .ply import read_gaussian_projection_fields, read_vertex_fields
from .projection import (
    Camera,
    build_scene_depth,
    camera_to_manifest,
    projection_counts,
    rasterize_components,
    select_temporal_views,
)


DEFAULT_DATA_ROOT = Path("/path/to/data/mps/egodelta_benchmark_data")
DEFAULT_ORACLE_ROOT = DEFAULT_DATA_ROOT / "oracle_component_masks_gt_adjacent_dbscan_e478073"
DEFAULT_ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3", "annotator_4")
PALETTE = (
    "#20d6c7",
    "#ff9f43",
    "#b084ff",
    "#8bd450",
    "#ff5d8f",
    "#36a2ff",
    "#ffd23f",
    "#f06543",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare projected EgoDelta instance masks and balanced annotation assignments."
    )
    parser.add_argument("--output", type=Path, required=True, help="New or resumable annotation run directory")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--images-per-scene", type=int, default=12)
    parser.add_argument("--annotators", nargs="+", default=list(DEFAULT_ANNOTATORS))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--min-visible-points", type=int, default=20)
    parser.add_argument("--no-occlusion-filter", action="store_true")
    parser.add_argument("--scene", action="append", default=[], help="Prepare only this scene ID; repeatable")
    parser.add_argument("--limit-scenes", type=int, help="Developer smoke-test limit")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already generated initial masks in an interrupted run directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_run(
            output=args.output,
            data_root=args.data_root,
            oracle_root=args.oracle_root,
            images_per_scene=args.images_per_scene,
            annotators=tuple(args.annotators),
            workers=args.workers,
            min_visible_points=args.min_visible_points,
            use_occlusion_filter=not args.no_occlusion_filter,
            selected_scenes=set(args.scene),
            limit_scenes=args.limit_scenes,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Prepared {len(manifest['tasks'])} tasks in {args.output.resolve()}")
    for annotator in manifest["annotators"]:
        print(f"  {annotator['id']}: {annotator['assigned_tasks']} tasks")
    print("Start the server with:")
    print(f"  python -m egomask.serve --run {args.output.resolve()} --host 0.0.0.0 --port 8765")
    return 0


def prepare_run(
    *,
    output: Path,
    data_root: Path,
    oracle_root: Path,
    images_per_scene: int = 12,
    annotators: tuple[str, ...] = DEFAULT_ANNOTATORS,
    workers: int = 1,
    min_visible_points: int = 20,
    use_occlusion_filter: bool = True,
    selected_scenes: set[str] | None = None,
    limit_scenes: int | None = None,
    resume: bool = False,
) -> dict:
    output = output.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    oracle_root = oracle_root.expanduser().resolve()
    _validate_arguments(images_per_scene, annotators, workers)
    _prepare_output_directory(output, resume)

    oracle_manifest_path = oracle_root / "manifest.json"
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    scenes = sorted(oracle_manifest["scenes"], key=lambda row: row["scene_id"])
    if selected_scenes:
        known = {row["scene_id"] for row in scenes}
        missing = sorted(selected_scenes - known)
        if missing:
            raise ValueError(f"Unknown scene IDs: {', '.join(missing)}")
        scenes = [row for row in scenes if row["scene_id"] in selected_scenes]
    if limit_scenes is not None:
        if limit_scenes < 1:
            raise ValueError("--limit-scenes must be positive")
        scenes = scenes[:limit_scenes]
    if not scenes:
        raise ValueError("No scenes selected")

    worker_specs = [
        {
            "scene_id": row["scene_id"],
            "data_root": str(data_root),
            "oracle_root": str(oracle_root),
            "output": str(output),
            "images_per_scene": images_per_scene,
            "min_visible_points": min_visible_points,
            "use_occlusion_filter": use_occlusion_filter,
            "resume": resume,
        }
        for row in scenes
    ]

    scene_results: list[dict] = []
    if workers == 1:
        for index, spec in enumerate(worker_specs, start=1):
            result = _prepare_scene(spec)
            scene_results.append(result)
            print(f"[{index}/{len(worker_specs)}] {result['scene_id']}: {len(result['tasks'])} masks", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_prepare_scene, spec): spec["scene_id"] for spec in worker_specs}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                scene_results.append(result)
                print(f"[{index}/{len(worker_specs)}] {result['scene_id']}: {len(result['tasks'])} masks", flush=True)
    scene_results.sort(key=lambda row: row["scene_id"])

    annotator_rows = [
        {"id": annotator, "token": secrets.token_urlsafe(24), "assigned_tasks": 0}
        for annotator in annotators
    ]
    tasks: list[dict] = []
    for scene_index, result in enumerate(scene_results):
        scene_tasks = sorted(result.pop("tasks"), key=lambda row: row["image_name"])
        for task_index, task in enumerate(scene_tasks):
            assignment_index = (task_index + scene_index) % len(annotator_rows)
            annotator_rows[assignment_index]["assigned_tasks"] += 1
            task["assigned_to"] = annotator_rows[assignment_index]["id"]
            tasks.append(task)

    manifest = {
        "schema_version": 1,
        "tool": "egomask",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "oracle_root": str(oracle_root),
        "oracle_manifest_sha256": _sha256(oracle_manifest_path),
        "oracle_export_git_head": oracle_manifest.get("export_git_head"),
        "configuration": {
            "images_per_scene": images_per_scene,
            "min_visible_points": min_visible_points,
            "occlusion_filter": use_occlusion_filter,
            "selection": "mask-visible temporal bins",
            "assignment": "scene-rotated round robin",
            "mask_encoding": "8-bit PNG; 0=background, component_id+1=instance label",
        },
        "annotators": annotator_rows,
        "scenes": scene_results,
        "tasks": tasks,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    _write_access_links(output, annotator_rows)
    return manifest


def _prepare_scene(spec: dict) -> dict:
    scene_id = spec["scene_id"]
    data_root = Path(spec["data_root"])
    oracle_root = Path(spec["oracle_root"])
    output = Path(spec["output"])
    scene_root = oracle_root / "scenes" / scene_id
    metadata = json.loads((scene_root / "metadata.json").read_text(encoding="utf-8"))
    scene_uuid = metadata["scene_uuid"]
    task_kind = metadata["task"]
    selection_name = "pre_frames.txt" if task_kind == "deleafing" else "tomato_pre_frames.txt"
    selection_path = data_root / "pinhole" / scene_uuid / "selections" / selection_name
    selected_names = [line.strip() for line in selection_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected_set = set(selected_names)
    image_root = data_root / "pinhole" / scene_uuid / "frames_rect"

    camera_path = data_root / "pre_models" / scene_id / "cameras.json"
    camera_items = json.loads(camera_path.read_text(encoding="utf-8"))
    cameras = [Camera.from_json(item) for item in camera_items]
    cameras = [camera for camera in cameras if camera.image_name in selected_set]
    cameras.sort(key=lambda camera: camera.image_name)
    if len(cameras) != len(selected_names):
        camera_names = {camera.image_name for camera in cameras}
        missing = sorted(selected_set - camera_names)
        raise ValueError(f"{scene_id}: camera JSON is missing selected frames: {missing[:5]}")
    for camera in cameras:
        if not (image_root / camera.image_name).is_file():
            raise FileNotFoundError(f"{scene_id}: missing image {image_root / camera.image_name}")

    components: list[dict] = []
    component_rows: list[dict] = []
    for component_meta in sorted(metadata["components"], key=lambda row: row["oracle_component_id"]):
        component_id = int(component_meta["oracle_component_id"])
        matches = sorted((scene_root / "components").glob(f"component_{component_id:03d}_*.ply"))
        if len(matches) != 1:
            raise ValueError(f"{scene_id}: expected one PLY for component {component_id}, found {len(matches)}")
        fields = read_gaussian_projection_fields(matches[0])
        components.append(
            {
                "xyz": np.column_stack((fields["x"], fields["y"], fields["z"])),
                "world_scale": fields["world_scale"],
                "alpha": fields["alpha"],
            }
        )
        label = component_id + 1
        component_rows.append(
            {
                "component_id": component_id,
                "label": label,
                "color": PALETTE[(label - 1) % len(PALETTE)],
                "adjacent_gt_indices": component_meta["adjacent_gt_indices"],
                "primitive_count": component_meta["primitive_count"],
                "source_ply": str(matches[0]),
            }
        )

    counts_by_view = [projection_counts(components, camera) for camera in cameras]
    selected_indices = select_temporal_views(
        cameras,
        counts_by_view,
        int(spec["images_per_scene"]),
        int(spec["min_visible_points"]),
    )

    source_pre_ply = Path(metadata["source_pre_ply"])
    if not source_pre_ply.is_file():
        source_pre_ply = data_root / "pre_models" / scene_id / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    scene_points: np.ndarray | None = None
    if spec["use_occlusion_filter"]:
        full = read_vertex_fields(source_pre_ply, ("x", "y", "z"))
        scene_points = np.column_stack((full["x"], full["y"], full["z"]))

    task_rows: list[dict] = []
    mask_dir = output / "initial_masks" / scene_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    for selected_rank, camera_index in enumerate(selected_indices):
        camera = cameras[camera_index]
        mask_path = mask_dir / f"{Path(camera.image_name).stem}.png"
        projection_meta: dict
        if spec["resume"] and mask_path.is_file():
            from PIL import Image

            with Image.open(mask_path) as prior:
                if prior.size != (camera.width, camera.height):
                    raise ValueError(f"{mask_path}: existing mask dimensions do not match camera")
                array = np.asarray(prior.convert("L"))
            projection_meta = {
                "visible_pixels": int(np.count_nonzero(array)),
                "pixel_counts": {
                    str(row["label"]): int(np.count_nonzero(array == row["label"]))
                    for row in component_rows
                },
                "resumed": True,
            }
        else:
            depth_map = build_scene_depth(scene_points, camera) if scene_points is not None else None
            mask, projection_meta = rasterize_components(components, camera, scene_depth=depth_map)
            _save_png_atomic(mask_path, mask)

        task_id = f"{scene_id}__{Path(camera.image_name).stem}"
        task_rows.append(
            {
                "task_id": task_id,
                "scene_id": scene_id,
                "scene_uuid": scene_uuid,
                "task_kind": task_kind,
                "image_name": camera.image_name,
                "source_image": str(image_root / camera.image_name),
                "initial_mask": str(mask_path.relative_to(output)),
                "selected_rank": selected_rank,
                "source_view_index": camera_index,
                "projection_counts": counts_by_view[camera_index],
                "projection": projection_meta,
                "camera": camera_to_manifest(camera),
                "components": component_rows,
            }
        )

    return {
        "scene_id": scene_id,
        "scene_uuid": scene_uuid,
        "task_kind": task_kind,
        "component_count": len(component_rows),
        "gt_count": metadata["gt_count"],
        "source_camera_count": len(cameras),
        "selected_image_count": len(task_rows),
        "selection_file": str(selection_path),
        "camera_file": str(camera_path),
        "source_pre_ply": str(source_pre_ply),
        "tasks": task_rows,
    }


def _validate_arguments(images_per_scene: int, annotators: tuple[str, ...], workers: int) -> None:
    if images_per_scene < 1:
        raise ValueError("images_per_scene must be positive")
    if not annotators:
        raise ValueError("At least one annotator is required")
    if len(set(annotators)) != len(annotators):
        raise ValueError("Annotator IDs must be unique")
    for annotator in annotators:
        if not annotator or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in annotator):
            raise ValueError(f"Unsafe annotator ID {annotator!r}; use letters, digits, '_' or '-'")
    if workers < 1:
        raise ValueError("workers must be positive")


def _prepare_output_directory(output: Path, resume: bool) -> None:
    if output.exists():
        entries = list(output.iterdir())
        if entries and not resume:
            raise FileExistsError(f"{output} is not empty; use a new directory or pass --resume")
        if (output / "manifest.json").exists():
            raise FileExistsError(f"{output} already contains a completed manifest")
    output.mkdir(parents=True, exist_ok=True)


def _write_access_links(output: Path, annotators: list[dict]) -> None:
    lines = [
        "# Private annotator access paths",
        "# Append each path to the server address, for example http://SERVER:8765/?token=...",
        "# Do not swap links: each token is bound to one non-overlapping assignment.",
        "",
    ]
    lines.extend(f"{row['id']}: /?token={row['token']}" for row in annotators)
    path = output / "ACCESS_LINKS.txt"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_png_atomic(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
