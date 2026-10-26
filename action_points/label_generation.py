from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from action_points.colmap import CameraView, load_colmap_views, project_single_point, project_world_points


AUTO_MASK_CLUSTER_KEYS = (
    ("leaf_instance_candidate", "leaf_instance_id"),
    ("tomato_instance_candidate", "tomato_instance_id"),
    ("tomato_fruit_candidate", "tomato_instance_id"),
    ("optimized_candidate", "post_opt_cluster_id"),
)


@dataclass
class LabelBuildConfig:
    scene_id: str
    task_type: str
    class_name: str = "target"
    split: str = "train"
    min_visible_points: int = 8
    min_visible_fraction: float = 0.01
    min_bbox_size_px: float = 8.0
    bbox_percentile: float = 2.0
    bbox_pad_px: float = 8.0
    bbox_pad_fraction: float = 0.08
    target_pad_px: float = 6.0
    include_target_in_bbox: bool = True
    min_target_confidence: float = 0.0
    max_views_per_instance: int = 0
    max_projected_points_record: int = 64


@dataclass
class TargetInstance:
    instance_id: int
    source_cluster_id: int | None
    task_type: str
    class_name: str
    target_xyz: np.ndarray
    target_confidence: float
    component_points: np.ndarray
    component_size: int
    metadata: dict


def build_action_point_manifest(
    dataset_dir: str | Path,
    presence_npz: str | Path,
    summary_json: str | Path | None,
    config: LabelBuildConfig,
    images: str = "images",
    sparse: str = "sparse/0",
    mask_key: str | None = None,
    cluster_key: str | None = None,
) -> dict:
    views = load_colmap_views(dataset_dir, images=images, sparse=sparse)
    instances, keys = load_target_instances(
        presence_npz=presence_npz,
        summary_json=summary_json,
        task_type=config.task_type,
        class_name=config.class_name,
        min_confidence=config.min_target_confidence,
        mask_key=mask_key,
        cluster_key=cluster_key,
    )
    samples = []
    for instance in instances:
        instance_samples = []
        for view in views:
            sample = project_instance_sample(instance, view, config)
            if sample is not None:
                instance_samples.append(sample)
        if config.max_views_per_instance > 0 and len(instance_samples) > config.max_views_per_instance:
            instance_samples = _subsample_evenly(instance_samples, config.max_views_per_instance)
        samples.extend(instance_samples)

    return {
        "version": 1,
        "type": "egodelta_action_point_manifest",
        "scene_id": config.scene_id,
        "task_type": config.task_type,
        "class_names": sorted({sample["class_name"] for sample in samples} | {config.class_name}),
        "source": {
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "presence_npz": str(Path(presence_npz).resolve()),
            "summary_json": None if summary_json is None else str(Path(summary_json).resolve()),
            "mask_key": keys["mask_key"],
            "cluster_key": keys["cluster_key"],
        },
        "label_config": config.__dict__,
        "num_views": len(views),
        "num_instances": len(instances),
        "num_samples": len(samples),
        "samples": samples,
    }


def load_target_instances(
    presence_npz: str | Path,
    summary_json: str | Path | None,
    task_type: str,
    class_name: str = "target",
    min_confidence: float = 0.0,
    mask_key: str | None = None,
    cluster_key: str | None = None,
) -> tuple[list[TargetInstance], dict[str, str]]:
    presence = np.load(presence_npz)
    if "xyz" not in presence.files:
        raise RuntimeError(f"{presence_npz} has no xyz array")
    xyz = presence["xyz"].astype(np.float32)
    mask_key, cluster_key = resolve_mask_cluster_keys(presence, mask_key, cluster_key)
    selected = presence[mask_key].astype(bool)
    cluster_id = presence[cluster_key].astype(np.int64)
    summary_targets = _load_summary_targets(summary_json) if summary_json is not None else {}

    array_targets = _load_array_targets(presence)
    if summary_targets:
        target_items = summary_targets
    elif array_targets:
        target_items = array_targets
    else:
        raise RuntimeError(
            f"No target_xyz entries found in {summary_json or presence_npz}. "
            "Run an EgoDelta component target extractor first."
        )

    instances: list[TargetInstance] = []
    for instance_id, item in sorted(target_items.items()):
        confidence = float(item.get("target_confidence", 1.0))
        if confidence < min_confidence:
            continue
        component_mask = selected & (cluster_id == int(instance_id))
        if not component_mask.any():
            continue
        target_xyz = np.asarray(item["target_xyz"], dtype=np.float32).reshape(3)
        source_cluster_id = item.get("source_cluster_id")
        instances.append(
            TargetInstance(
                instance_id=int(instance_id),
                source_cluster_id=None if source_cluster_id is None else int(source_cluster_id),
                task_type=task_type,
                class_name=class_name,
                target_xyz=target_xyz,
                target_confidence=confidence,
                component_points=xyz[component_mask],
                component_size=int(component_mask.sum()),
                metadata={key: value for key, value in item.items() if key != "target_xyz"},
            )
        )
    return instances, {"mask_key": mask_key, "cluster_key": cluster_key}


def resolve_mask_cluster_keys(presence: np.lib.npyio.NpzFile, mask_key: str | None, cluster_key: str | None) -> tuple[str, str]:
    if mask_key is not None and cluster_key is not None:
        _require_npz_key(presence, mask_key)
        _require_npz_key(presence, cluster_key)
        return mask_key, cluster_key
    for candidate_mask, candidate_cluster in AUTO_MASK_CLUSTER_KEYS:
        if candidate_mask in presence.files and candidate_cluster in presence.files:
            return candidate_mask, candidate_cluster
    raise RuntimeError(
        "Could not infer component mask/cluster keys. Pass --mask-key and --cluster-key. "
        f"Available keys: {presence.files}"
    )


def project_instance_sample(instance: TargetInstance, view: CameraView, config: LabelBuildConfig) -> dict | None:
    uv, _depth, valid, _points_cam = project_world_points(instance.component_points, view)
    visible_count = int(valid.sum())
    if visible_count < config.min_visible_points:
        return None
    visible_fraction = visible_count / max(instance.component_size, 1)
    if visible_fraction < config.min_visible_fraction:
        return None

    target_uv, target_depth, target_valid, target_cam = project_single_point(instance.target_xyz, view)
    if not target_valid:
        return None

    component_bbox = bbox_from_projected_points(
        uv[valid],
        width=view.width,
        height=view.height,
        percentile=config.bbox_percentile,
        pad_px=config.bbox_pad_px,
        pad_fraction=config.bbox_pad_fraction,
    )
    if component_bbox is None:
        return None
    bbox = list(component_bbox)
    if config.include_target_in_bbox:
        bbox = union_bbox_with_point(bbox, target_uv, view.width, view.height, config.target_pad_px)
    if not bbox_is_usable(bbox, config.min_bbox_size_px):
        return None

    bbox_w = max(float(bbox[2] - bbox[0]), 1e-6)
    bbox_h = max(float(bbox[3] - bbox[1]), 1e-6)
    target_rel = [
        float((target_uv[0] - bbox[0]) / bbox_w),
        float((target_uv[1] - bbox[1]) / bbox_h),
    ]
    sample_id = stable_sample_id(config.scene_id, view.image_name, instance.instance_id)
    sample = {
        "sample_id": sample_id,
        "scene_id": config.scene_id,
        "split": config.split,
        "task_type": instance.task_type,
        "class_name": instance.class_name,
        "instance_id": int(instance.instance_id),
        "source_cluster_id": instance.source_cluster_id,
        "image_name": view.image_name,
        "image_path": str(view.image_path.resolve()),
        "width": int(view.width),
        "height": int(view.height),
        "camera": view.to_manifest_dict(),
        "bbox_xyxy": [float(v) for v in bbox],
        "component_bbox_xyxy": [float(v) for v in component_bbox],
        "bbox_xywh_norm": xyxy_to_xywh_norm(bbox, view.width, view.height),
        "target_2d": [float(target_uv[0]), float(target_uv[1])],
        "target_rel_xy": target_rel,
        "target_depth": float(target_depth),
        "target_3d_world": [float(v) for v in instance.target_xyz.tolist()],
        "target_3d_camera": [float(v) for v in target_cam.tolist()],
        "target_confidence": float(instance.target_confidence),
        "visible_component_points": visible_count,
        "component_point_count": int(instance.component_size),
        "visible_fraction": float(visible_fraction),
        "metadata": instance.metadata,
    }
    if config.max_projected_points_record > 0:
        sample["projected_points_preview"] = projected_points_preview(
            uv[valid],
            max_points=config.max_projected_points_record,
            seed=sample_id,
        )
    return sample


def bbox_from_projected_points(
    uv: np.ndarray,
    width: int,
    height: int,
    percentile: float = 2.0,
    pad_px: float = 8.0,
    pad_fraction: float = 0.08,
) -> list[float] | None:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] == 0:
        return None
    lo = float(np.clip(percentile, 0.0, 45.0))
    hi = 100.0 - lo
    x0, y0 = np.percentile(uv, lo, axis=0)
    x1, y1 = np.percentile(uv, hi, axis=0)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pad = float(pad_px) + float(pad_fraction) * max(float(x1 - x0), float(y1 - y0))
    return clip_bbox([x0 - pad, y0 - pad, x1 + pad, y1 + pad], width, height)


def union_bbox_with_point(bbox: list[float], point_uv: np.ndarray, width: int, height: int, pad_px: float) -> list[float]:
    x, y = [float(v) for v in point_uv[:2]]
    return clip_bbox(
        [
            min(float(bbox[0]), x - pad_px),
            min(float(bbox[1]), y - pad_px),
            max(float(bbox[2]), x + pad_px),
            max(float(bbox[3]), y + pad_px),
        ],
        width,
        height,
    )


def clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    return [
        float(np.clip(bbox[0], 0.0, max(float(width - 1), 0.0))),
        float(np.clip(bbox[1], 0.0, max(float(height - 1), 0.0))),
        float(np.clip(bbox[2], 0.0, max(float(width - 1), 0.0))),
        float(np.clip(bbox[3], 0.0, max(float(height - 1), 0.0))),
    ]


def bbox_is_usable(bbox: list[float], min_size_px: float) -> bool:
    return float(bbox[2] - bbox[0]) >= min_size_px and float(bbox[3] - bbox[1]) >= min_size_px


def xyxy_to_xywh_norm(bbox: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return [
        float(((x0 + x1) * 0.5) / max(width, 1)),
        float(((y0 + y1) * 0.5) / max(height, 1)),
        float((x1 - x0) / max(width, 1)),
        float((y1 - y0) / max(height, 1)),
    ]


def projected_points_preview(uv: np.ndarray, max_points: int, seed: str) -> list[list[float]]:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if uv.shape[0] <= max_points:
        chosen = uv
    else:
        seed_int = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        chosen = uv[np.sort(rng.choice(uv.shape[0], max_points, replace=False))]
    return [[float(x), float(y)] for x, y in chosen]


def stable_sample_id(scene_id: str, image_name: str, instance_id: int) -> str:
    stem = Path(image_name).stem
    digest = hashlib.sha1(f"{scene_id}:{stem}:{instance_id}".encode("utf-8")).hexdigest()[:10]
    return f"{scene_id}_{stem}_i{int(instance_id):03d}_{digest}"


def _load_summary_targets(summary_json: str | Path | None) -> dict[int, dict]:
    if summary_json is None:
        return {}
    data = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    targets = {}
    for item in data.get("clusters", []):
        target_xyz = item.get("target_xyz")
        if target_xyz is None:
            continue
        cluster_id = int(item["cluster_id"])
        confidence = item.get("target_confidence")
        if confidence is None and isinstance(item.get("target"), dict):
            confidence = item["target"].get("confidence")
        targets[cluster_id] = {
            "target_xyz": target_xyz,
            "target_confidence": 1.0 if confidence is None else float(confidence),
            "source_cluster_id": item.get("source_cluster_id"),
            "method": item.get("target", {}).get("method") if isinstance(item.get("target"), dict) else item.get("method"),
        }
    return targets


def _load_array_targets(presence: np.lib.npyio.NpzFile) -> dict[int, dict]:
    if "interaction_target_xyz" not in presence.files or "interaction_target_instance_id" not in presence.files:
        return {}
    target_xyz = np.asarray(presence["interaction_target_xyz"], dtype=np.float32).reshape(-1, 3)
    target_ids = np.asarray(presence["interaction_target_instance_id"], dtype=np.int64).reshape(-1)
    confidence = (
        np.asarray(presence["interaction_target_confidence"], dtype=np.float32).reshape(-1)
        if "interaction_target_confidence" in presence.files
        else np.ones((target_xyz.shape[0],), dtype=np.float32)
    )
    source_ids = (
        np.asarray(presence["interaction_target_source_cluster_id"], dtype=np.int64).reshape(-1)
        if "interaction_target_source_cluster_id" in presence.files
        else np.full((target_xyz.shape[0],), -1, dtype=np.int64)
    )
    return {
        int(instance_id): {
            "target_xyz": xyz.tolist(),
            "target_confidence": float(conf),
            "source_cluster_id": None if int(source_id) < 0 else int(source_id),
        }
        for instance_id, xyz, conf, source_id in zip(target_ids, target_xyz, confidence, source_ids)
    }


def _require_npz_key(presence: np.lib.npyio.NpzFile, key: str) -> None:
    if key not in presence.files:
        raise RuntimeError(f"{key} is absent from NPZ. Available keys: {presence.files}")


def _subsample_evenly(samples: list[dict], count: int) -> list[dict]:
    if len(samples) <= count:
        return samples
    idx = np.linspace(0, len(samples) - 1, count).round().astype(np.int64)
    return [samples[int(i)] for i in idx]
