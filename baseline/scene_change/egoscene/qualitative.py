from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHOD_ORDER = ("3DGS-CD", "MV3DCD", "GS-DIFF", "EgoDelta")
TASK_GROUPS = (("deleafing", "Deleafing"), ("harvesting", "Harvesting"))


def select_stratified_tasks(
    tasks: list[dict[str, Any]],
    *,
    scenes_per_kind: int = 5,
    seed: int = 2027,
) -> dict[str, list[dict[str, Any]]]:
    """Select distinct scene-interventions and one fixed view from each."""

    if scenes_per_kind < 1:
        raise ValueError("scenes_per_kind must be positive")
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for kind, _ in TASK_GROUPS:
        by_scene: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            if task["task_kind"] == kind:
                by_scene.setdefault(task["scene_id"], []).append(task)
        scene_ids = sorted(by_scene)
        if len(scene_ids) < scenes_per_kind:
            raise ValueError(
                f"Need {scenes_per_kind} {kind} scenes, found {len(scene_ids)}"
            )
        chosen_scenes = rng.sample(scene_ids, scenes_per_kind)
        selected[kind] = [
            rng.choice(sorted(by_scene[scene_id], key=lambda item: item["task_id"]))
            for scene_id in chosen_scenes
        ]
    return selected


def create_figure3_comparison(
    gt_export: str | Path,
    prediction_roots: Mapping[str, str | Path],
    output_stem: str | Path,
    *,
    scenes_per_kind: int = 5,
    seed: int = 2027,
    overlay_alpha: float = 0.42,
) -> dict[str, Any]:
    """Create a Figure-3-style RGB-overlay/binary-mask comparison grid."""

    gt_export = Path(gt_export).expanduser().resolve()
    output_stem = Path(output_stem).expanduser().resolve()
    roots = {name: Path(path).expanduser().resolve() for name, path in prediction_roots.items()}
    missing_methods = [name for name in METHOD_ORDER if name not in roots]
    if missing_methods:
        raise ValueError(f"Missing prediction roots for: {', '.join(missing_methods)}")
    dataset_path = gt_export / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected = select_stratified_tasks(
        dataset["tasks"], scenes_per_kind=scenes_per_kind, seed=seed
    )

    columns_per_group = len(METHOD_ORDER) + 1
    figure, axes = plt.subplots(
        scenes_per_kind,
        columns_per_group * len(TASK_GROUPS),
        figsize=(11.0, 8.25),
        squeeze=False,
    )
    column_labels = (*METHOD_ORDER, "Input / GT")
    records: list[dict[str, Any]] = []
    for group_index, (kind, group_label) in enumerate(TASK_GROUPS):
        column_offset = group_index * columns_per_group
        for column, label in enumerate(column_labels):
            axes[0, column_offset + column].set_title(label, fontsize=8, pad=4)
        for row, task in enumerate(selected[kind]):
            rgb = np.asarray(Image.open(task["source_image"]).convert("RGB"))
            gt = _read_mask(gt_export / task["label_map"], rgb.shape[:2])
            row_record: dict[str, Any] = {
                "group": group_label,
                "task_id": task["task_id"],
                "scene_id": task["scene_id"],
                "image_name": task["image_name"],
                "source_image": task["source_image"],
                "ground_truth": str((gt_export / task["label_map"]).resolve()),
                "predictions": {},
            }
            for method_index, method in enumerate(METHOD_ORDER):
                mask_path = (
                    roots[method]
                    / task["scene_id"]
                    / "masks"
                    / f"{Path(task['image_name']).stem}.png"
                )
                mask = _read_mask(mask_path, rgb.shape[:2])
                panel = _stack_overlay_and_mask(rgb, mask, alpha=overlay_alpha)
                axes[row, column_offset + method_index].imshow(panel)
                row_record["predictions"][method] = str(mask_path)
            gt_panel = np.concatenate((rgb, _binary_rgb(gt)), axis=0)
            axes[row, column_offset + len(METHOD_ORDER)].imshow(gt_panel)
            axes[row, column_offset].set_ylabel(
                f"{group_label[0]}{row + 1}", fontsize=7, rotation=0, labelpad=9
            )
            records.append(row_record)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.text(0.255, 0.985, "Deleafing", ha="center", va="top", fontsize=10, weight="bold")
    figure.text(0.745, 0.985, "Harvesting", ha="center", va="top", fontsize=10, weight="bold")
    figure.subplots_adjust(left=0.025, right=0.995, bottom=0.015, top=0.945, wspace=0.025, hspace=0.035)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(figure)

    metadata = {
        "schema_version": 1,
        "layout_inspiration": "Figure 3 of arXiv:2412.03911v1",
        "layout": (
            "two task-kind blocks; method RGB overlay above standalone binary mask; "
            "input RGB above human ground truth"
        ),
        "selection_policy": (
            "deterministic stratified random selection of distinct scene-interventions, "
            "then one fixed benchmark view per selected intervention"
        ),
        "seed": seed,
        "scenes_per_kind": scenes_per_kind,
        "gt_export": str(gt_export),
        "gt_dataset_sha256": _sha256(dataset_path),
        "prediction_roots": {key: str(value) for key, value in roots.items()},
        "method_order": list(METHOD_ORDER),
        "selected_tasks": records,
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    metadata_path = output_stem.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask: {path}")
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != expected_shape:
        raise ValueError(f"Mask/image shape mismatch for {path}: {mask.shape} != {expected_shape}")
    return mask


def _stack_overlay_and_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0, 1]")
    overlay = np.asarray(rgb, dtype=np.float32).copy()
    color = np.asarray((255.0, 67.0, 86.0), dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    boundary = _inner_boundary(mask)
    overlay[boundary] = color
    return np.concatenate((np.clip(overlay, 0, 255).astype(np.uint8), _binary_rgb(mask)), axis=0)


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    eroded = np.zeros_like(mask)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~eroded


def _binary_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((np.asarray(mask, dtype=np.uint8) * 255)[..., None], 3, axis=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
