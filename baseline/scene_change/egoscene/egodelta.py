from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from annotation_tool.egomask.ply import read_gaussian_projection_fields
from annotation_tool.egomask.projection import Camera, build_scene_depth, rasterize_components
from target_extraction.egodelta_diff import (
    EndpointSide,
    ExtractionConfig,
    RemovedSurfaceConfig,
    cache_digest,
    extract_for_endpoint_side,
    load_stage2_cache,
)

from .benchmark import SceneSpec, select_point_cloud
from .outputs import write_prediction, write_scene_metadata


ENDPOINT_BY_TASK = {
    "deleafing": EndpointSide.RETAINED,
    "harvesting": EndpointSide.REMOVED,
}


def camera_from_task(task: dict[str, Any]) -> Camera:
    payload = task["camera"]
    return Camera(
        image_name=str(task["image_name"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        center=np.asarray(payload["camera_center_world"], dtype=np.float64).reshape(3),
        rotation=np.asarray(payload["rotation_camera_to_world"], dtype=np.float64).reshape(3, 3),
    )


def projection_components(
    targets: list[Any],
    cache_mu: np.ndarray,
    source_indices: np.ndarray,
    ply_fields: dict[str, np.ndarray],
    *,
    coordinate_tolerance_m: float = 1.0e-5,
) -> list[dict[str, np.ndarray]]:
    """Map accepted Stage-3 component indices back to full pre-model PLY rows."""

    cache_mu = np.asarray(cache_mu, dtype=np.float64).reshape(-1, 3)
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if len(cache_mu) != len(source_indices):
        raise ValueError("Stage-2 mu and source_indices must have equal length")
    full_xyz = np.column_stack((ply_fields["x"], ply_fields["y"], ply_fields["z"]))
    components: list[dict[str, np.ndarray]] = []
    for target in targets:
        cache_indices = np.asarray(target.primitive_indices, dtype=np.int64).reshape(-1)
        if not len(cache_indices):
            continue
        if cache_indices.min() < 0 or cache_indices.max() >= len(cache_mu):
            raise IndexError("Stage-3 primitive index is outside the Stage-2 cache")
        rows = source_indices[cache_indices]
        if rows.min() < 0 or rows.max() >= len(full_xyz):
            raise IndexError("Stage-2 source index is outside the pre-model PLY")
        error = float(np.max(np.linalg.norm(full_xyz[rows] - cache_mu[cache_indices], axis=1)))
        if error > coordinate_tolerance_m:
            raise ValueError(
                "Stage-2 source-index mapping does not match the pre-model PLY "
                f"(max coordinate error {error:.9g} m)"
            )
        components.append(
            {
                "xyz": full_xyz[rows],
                "world_scale": np.asarray(ply_fields["world_scale"])[rows],
                "alpha": np.asarray(ply_fields["alpha"])[rows],
            }
        )
    return components


def run_egodelta_scene(
    scene: SceneSpec,
    cache_root: str | Path,
    output_root: str | Path,
    *,
    annotator: str | None = None,
) -> dict[str, Any]:
    """Decode an EgoDelta Stage-2 cache and render its removed components."""

    cache_path = Path(cache_root) / f"{scene.scene_id}.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing EgoDelta Stage-2 cache: {cache_path}")
    cache = load_stage2_cache(cache_path)
    if cache.scene_id != scene.scene_id:
        raise ValueError(f"Cache scene mismatch: {cache.scene_id} != {scene.scene_id}")
    try:
        endpoint = ENDPOINT_BY_TASK[scene.task_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported EgoDelta task kind: {scene.task_kind}") from exc

    retained_config = ExtractionConfig()
    removed_config = RemovedSurfaceConfig()
    result = extract_for_endpoint_side(
        endpoint,
        mu=cache.mu,
        normals=cache.normals,
        delta=cache.delta,
        omega=cache.omega,
        overlap=cache.overlap,
        source_indices=cache.source_indices,
        retained_config=retained_config,
        removed_config=removed_config,
    )

    pre_ply = select_point_cloud(scene.pre_model)
    ply_fields = read_gaussian_projection_fields(pre_ply)
    full_xyz = np.column_stack((ply_fields["x"], ply_fields["y"], ply_fields["z"]))
    components = projection_components(
        result.targets,
        cache.mu,
        cache.source_indices,
        ply_fields,
    )
    queries = [
        task
        for task in scene.queries
        if annotator is None or str(task.get("assigned_to")) == annotator
    ]
    if not queries:
        raise ValueError(f"No benchmark queries for annotator {annotator!r} in {scene.scene_id}")

    print(
        f"  endpoint={endpoint.value}; targets={len(result.targets)}; "
        f"component primitives={sum(len(x['xyz']) for x in components)}",
        flush=True,
    )
    predictions: list[dict[str, Any]] = []
    for task in queries:
        camera = camera_from_task(task)
        scene_depth = build_scene_depth(full_xyz, camera)
        label_map, projection = rasterize_components(
            components,
            camera,
            scene_depth=scene_depth,
        )
        probability = (np.asarray(label_map, dtype=np.uint8) > 0).astype(np.float32)
        paths = write_prediction(output_root, scene.scene_id, task["image_name"], probability)
        predictions.append(
            {
                "task_id": task["task_id"],
                "image_name": task["image_name"],
                "projection": projection,
                **paths,
            }
        )

    config = retained_config if endpoint is EndpointSide.RETAINED else removed_config
    diagnostics = {
        "method": "egodelta",
        "implementation": "final mechanics-conditioned Stage-3 decoder",
        "task_kind": scene.task_kind,
        "endpoint_side": endpoint.value,
        "inference_is_gt_free": True,
        "query_protocol": "fixed annotation-manifest cameras; component annotations are not read",
        "annotator_query_filter": annotator,
        "stage2_cache": str(cache_path.resolve()),
        "stage2_cache_sha256": cache_digest(cache_path),
        "stage2_config": cache.metadata.get("stage2_diagnostics", {}).get("ablation", {}),
        "stage3_config": asdict(config),
        "formed_component_count": int(result.formed_component_count),
        "accepted_component_count": int(result.accepted_component_count),
        "target_count": len(result.targets),
        "projected_component_count": len(components),
        "projected_primitive_count": int(sum(len(item["xyz"]) for item in components)),
        "pre_model_ply": str(pre_ply.resolve()),
        "projection": {
            "rasterizer": "EgoMask Gaussian-footprint projection",
            "full_pre_model_depth_test": True,
            "binary_threshold": 0.5,
        },
        "predictions": predictions,
    }
    write_scene_metadata(output_root, scene.scene_id, diagnostics)
    print(f"  rendered {len(predictions)} benchmark masks; complete", flush=True)
    return diagnostics
