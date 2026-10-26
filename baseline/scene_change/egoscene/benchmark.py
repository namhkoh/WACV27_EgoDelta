from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    scene_uuid: str
    task_kind: str
    task_suffix: str
    pre_model: Path
    post_model: Path
    pair_manifest: Path
    pre_dataset: dict[str, Any]
    post_dataset: dict[str, Any]
    queries: tuple[dict[str, Any], ...]


def load_annotation_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported annotation manifest schema: {payload.get('schema_version')}")
    if not payload.get("tasks") or not payload.get("scenes"):
        raise ValueError(f"Annotation manifest has no tasks/scenes: {path}")
    return payload


def discover_scenes(
    annotation_manifest: str | Path,
    *,
    pairs_root: str | Path,
    pre_models_root: str | Path,
    post_models_root: str | Path,
    scene_ids: Iterable[str] | None = None,
) -> list[SceneSpec]:
    manifest = load_annotation_manifest(annotation_manifest)
    pairs_root = Path(pairs_root).expanduser().resolve()
    pre_models_root = Path(pre_models_root).expanduser().resolve()
    post_models_root = Path(post_models_root).expanduser().resolve()
    requested = set(scene_ids or ())

    queries_by_scene: dict[str, list[dict[str, Any]]] = {}
    for task in manifest["tasks"]:
        queries_by_scene.setdefault(task["scene_id"], []).append(task)

    specs: list[SceneSpec] = []
    for row in manifest["scenes"]:
        scene_id = str(row["scene_id"])
        if requested and scene_id not in requested:
            continue
        suffix = _task_suffix(scene_id)
        pair_manifest = pairs_root / row["scene_uuid"] / f"{suffix}_pair" / "manifest.json"
        if not pair_manifest.is_file():
            raise FileNotFoundError(f"Pair manifest not found for {scene_id}: {pair_manifest}")
        pair_payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
        datasets = {item["label"]: item for item in pair_payload.get("datasets", [])}
        if set(datasets) != {"pre", "post"}:
            raise ValueError(f"Expected pre/post datasets in {pair_manifest}, got {sorted(datasets)}")
        pre_model = pre_models_root / scene_id
        post_model = post_models_root / scene_id
        for model in (pre_model, post_model):
            if not (model / "cameras.json").is_file():
                raise FileNotFoundError(f"Missing cameras.json: {model}")
            select_point_cloud(model)
        queries = tuple(sorted(queries_by_scene.get(scene_id, ()), key=lambda x: x["image_name"]))
        if not queries:
            raise ValueError(f"No evaluation queries found for {scene_id}")
        specs.append(
            SceneSpec(
                scene_id=scene_id,
                scene_uuid=str(row["scene_uuid"]),
                task_kind=str(row["task_kind"]),
                task_suffix=suffix,
                pre_model=pre_model,
                post_model=post_model,
                pair_manifest=pair_manifest,
                pre_dataset=datasets["pre"],
                post_dataset=datasets["post"],
                queries=queries,
            )
        )
    if requested:
        missing = requested.difference(x.scene_id for x in specs)
        if missing:
            raise KeyError(f"Unknown scene IDs: {sorted(missing)}")
    return sorted(specs, key=lambda x: x.scene_id)


def select_point_cloud(model_dir: str | Path, preferred_iteration: int = 30000) -> Path:
    model_dir = Path(model_dir)
    preferred = model_dir / "point_cloud" / f"iteration_{preferred_iteration}" / "point_cloud.ply"
    if preferred.is_file():
        return preferred
    candidates = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No trained point cloud found under {model_dir}")
    return max(candidates, key=lambda p: int(p.parent.name.rsplit("_", 1)[-1]))


def selected_frame_names(dataset: dict[str, Any], budget: int | None = None) -> list[str]:
    names = [line.strip() for line in Path(dataset["frame_list"]).read_text().splitlines() if line.strip()]
    if budget is None or budget <= 0 or budget >= len(names):
        return names
    if budget == 1:
        return [names[len(names) // 2]]
    # Deterministic coverage of the trajectory, including both endpoints.
    indices = [round(i * (len(names) - 1) / (budget - 1)) for i in range(budget)]
    return [names[i] for i in indices]


def source_image(dataset: dict[str, Any], name: str) -> Path:
    path = Path(dataset["source_root"]) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _task_suffix(scene_id: str) -> str:
    suffix = scene_id.rsplit("_", 1)[-1]
    if suffix not in {"leaf", "tomato"}:
        raise ValueError(f"Unsupported scene ID: {scene_id}")
    return suffix
