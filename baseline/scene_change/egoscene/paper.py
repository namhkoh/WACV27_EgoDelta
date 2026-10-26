"""Adapters for validating baselines on the datasets used in their papers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .colmap import (
    ColmapImage,
    ColmapPoint3D,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
)


def _link(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f"Unexpected symlink target at {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    destination.symlink_to(source)


def _split_color_estimates(
    points: list[ColmapPoint3D],
    images: list[ColmapImage],
    images_root: Path,
) -> list[ColmapPoint3D]:
    """Estimate initialization RGB from only the selected split's observations."""
    point_ids = {point.point_id for point in points}
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for record in images:
        with Image.open(images_root / record.name) as handle:
            pixels = np.asarray(handle.convert("RGB"))
        height, width = pixels.shape[:2]
        for x, y, point_id in record.points2d:
            if point_id not in point_ids:
                continue
            column = min(max(int(round(x)), 0), width - 1)
            row = min(max(int(round(y)), 0), height - 1)
            sums.setdefault(point_id, np.zeros(3, dtype=np.float64))
            sums[point_id] += pixels[row, column]
            counts[point_id] = counts.get(point_id, 0) + 1
    return [
        replace(
            point,
            rgb=tuple(int(x) for x in np.rint(sums[point.point_id] / counts[point.point_id]).clip(0, 255)),
        )
        if point.point_id in counts
        else point
        for point in points
    ]


def prepare_paslcd_pgsr(
    source: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Create independent reference/inference PGSR inputs in one COLMAP frame.

    PASLCD distributes a joint COLMAP model.  We retain its common camera frame,
    but filter cameras, point tracks, and point initialization colors so each
    PGSR reconstruction receives observations from only one capture state.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    sparse = source / "sparse" / "0"
    images_root = source / "images"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points = read_points3d_binary(sparse / "points3D.bin")
    partitions = {
        "reference": [record for record in images if "_test_" not in record.name],
        "inference": [record for record in images if "_test_" in record.name],
    }
    result: dict[str, object] = {"source": str(source), "splits": {}}
    for split, records in partitions.items():
        root = output / split
        split_images = root / "images"
        split_sparse = root / "sparse"
        split_images.mkdir(parents=True, exist_ok=True)
        split_sparse.mkdir(parents=True, exist_ok=True)
        for record in records:
            _link(images_root / record.name, split_images / record.name)
        image_ids = {record.image_id for record in records}
        split_points = [point for point in points if any(image_id in image_ids for image_id, _ in point.track)]
        split_points = _split_color_estimates(split_points, records, images_root)
        write_cameras_txt(split_sparse / "cameras.txt", cameras)
        write_images_txt(split_sparse / "images.txt", records, reindex=False)
        point_count = write_points3d_txt(split_sparse / "points3D.txt", split_points, image_ids=image_ids)
        result["splits"][split] = {
            "images": len(records),
            "points": point_count,
            "image_ids": sorted(image_ids),
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
