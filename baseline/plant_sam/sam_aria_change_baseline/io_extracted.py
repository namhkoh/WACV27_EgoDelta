from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_frames_metadata(frames: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(frames, handle, indent=2)


def _frame_stem(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.isdigit():
        return f"frame_{int(text):06d}"
    return Path(text).stem


def read_frame_list(path: str | Path) -> list[str]:
    frames: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            stem = _frame_stem(stripped)
            if stem:
                frames.append(stem)
    return frames


def _find_image(source_root: Path, stem: str) -> Path:
    direct = source_root / stem
    if direct.suffix.lower() in IMAGE_EXTS and direct.exists():
        return direct
    matches = [source_root / f"{stem}{suffix}" for suffix in IMAGE_EXTS]
    existing = [path for path in matches if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(f"Ambiguous extracted-frame image candidates for {stem}: {existing}")
    recursive = sorted(
        path for path in source_root.rglob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTS
    )
    if len(recursive) == 1:
        return recursive[0]
    if len(recursive) > 1:
        raise RuntimeError(f"Ambiguous extracted-frame image candidates for {stem}: {recursive[:5]}")
    raise FileNotFoundError(f"Could not find extracted frame {stem} under {source_root}.")


def load_pose_index(poses_csv: str | Path) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    with Path(poses_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stem = f"frame_{int(row['frame_idx']):06d}"
            poses[stem] = {
                "timestamp_ns": int(float(row["timestamp_ns"])),
                "frame_idx": int(row["frame_idx"]),
            }
    return poses


def build_extracted_frame_metadata(
    frame_stems: list[str],
    *,
    split: str,
    source_root: str | Path,
    poses_csv: str | Path,
    camera: dict[str, Any],
    frame_stride: int = 1,
) -> tuple[list[dict[str, Any]], list[str]]:
    source_root = Path(source_root)
    pose_index = load_pose_index(poses_csv)
    warnings: list[str] = []
    frames: list[dict[str, Any]] = []
    stride = max(1, int(frame_stride))
    for order_idx, stem in enumerate(frame_stems):
        if order_idx % stride != 0:
            continue
        pose = pose_index.get(stem)
        if pose is None:
            warnings.append(f"Skipped {split} frame {stem}: no pose in {poses_csv}.")
            continue
        image_path = _find_image(source_root, stem)
        frames.append(
            {
                "frame_id": f"{split}_{stem}",
                "split": split,
                "timestamp_ns": int(pose["timestamp_ns"]),
                "image_path": str(image_path),
                "width": int(camera["width"]),
                "height": int(camera["height"]),
                "is_undistorted": True,
                "camera_model": str(camera.get("model", "pinhole")),
                "image_rotation": "cw90" if str(camera.get("model", "")).endswith("cw90") else "none",
                "source_frame_name": image_path.name,
                "source_frame_stem": stem,
                "source_frame_idx": int(pose["frame_idx"]),
            }
        )
    return frames, warnings


def prepare_frames_from_pair_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    frame_stride: int = 1,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    """Create the pipeline frame JSONs from an already-extracted pre/post pair manifest."""
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    datasets = {str(item.get("label")): item for item in manifest.get("datasets", [])}
    if "pre" not in datasets or "post" not in datasets:
        raise RuntimeError(f"{manifest_path} must contain pre and post dataset entries.")

    if not manifest.get("single_trajectory", False):
        transform = manifest.get("T_pre_post")
        if transform is not None and not np.allclose(
            np.asarray(transform, dtype=float),
            np.eye(4, dtype=float),
            atol=1e-8,
        ):
            raise RuntimeError(
                "Extracted pre/post pair is not in one shared trajectory and has a non-identity "
                "alignment transform. Provide already aligned extracted frames for this baseline."
            )

    warnings: list[str] = []
    frames_dir = Path(out_dir) / "frames"
    pre_entry = datasets["pre"]
    post_entry = datasets["post"]
    pre_camera = dict(pre_entry["camera"])
    post_camera = dict(post_entry["camera"])
    if pre_camera != post_camera:
        warnings.append("Pre and post extracted-frame camera intrinsics differ; using per-split metadata.")

    def source_root(entry: dict[str, Any]) -> Path:
        return Path(entry["source_root"])

    def poses_csv(entry: dict[str, Any]) -> Path:
        return source_root(entry) / "poses.csv"

    def camera_json(entry: dict[str, Any]) -> Path:
        return source_root(entry) / "camera_pinhole.json"

    pre_camera_payload = _read_json(camera_json(pre_entry)) if camera_json(pre_entry).exists() else pre_camera
    post_camera_payload = _read_json(camera_json(post_entry)) if camera_json(post_entry).exists() else post_camera
    pre_frames, pre_warnings = build_extracted_frame_metadata(
        read_frame_list(pre_entry["frame_list"]),
        split="pre",
        source_root=source_root(pre_entry),
        poses_csv=poses_csv(pre_entry),
        camera=pre_camera_payload,
        frame_stride=frame_stride,
    )
    post_frames, post_warnings = build_extracted_frame_metadata(
        read_frame_list(post_entry["frame_list"]),
        split="post",
        source_root=source_root(post_entry),
        poses_csv=poses_csv(post_entry),
        camera=post_camera_payload,
        frame_stride=frame_stride,
    )
    warnings.extend(pre_warnings)
    warnings.extend(post_warnings)

    pre_json = frames_dir / "pre_frames.json"
    post_json = frames_dir / "post_frames.json"
    _write_frames_metadata(pre_frames, pre_json)
    _write_frames_metadata(post_frames, post_json)

    metadata = {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "pre_entry": pre_entry,
        "post_entry": post_entry,
        "pre_source_root": str(source_root(pre_entry)),
        "post_source_root": str(source_root(post_entry)),
        "pre_poses_csv": str(poses_csv(pre_entry)),
        "post_poses_csv": str(poses_csv(post_entry)),
        "pre_camera_json": str(camera_json(pre_entry)),
        "post_camera_json": str(camera_json(post_entry)),
        "pre_points_source": str(pre_entry["points_source"]),
        "post_points_source": str(post_entry["points_source"]),
    }
    return pre_json, post_json, pre_frames, post_frames, metadata, warnings
