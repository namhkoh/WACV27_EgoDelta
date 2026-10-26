"""Project Aria VRS RGB frame extraction utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass
class FrameRecord:
    frame_id: str
    split: str
    timestamp_ns: int
    image_path: str
    original_width: int
    original_height: int
    source_frame_index: int
    image_view: str = "raw"
    camera_label: str | None = None
    camera_model: str | None = None
    rect_size: int | None = None
    rect_focal: float | None = None
    source_original_width: int | None = None
    source_original_height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameImageProcessor:
    image_view: str
    rgb_stream_id: str
    camera_label: str | None = None
    camera_model: str = "raw"
    rect_size: int | None = None
    rect_focal: float | None = None
    rgb_calib_orig: Any = None
    linear_calib: Any = None
    output_calib: Any = None

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.image_view == "raw":
            return np.ascontiguousarray(image)
        if self.image_view != "egodelta_pinhole":
            raise ValueError(f"Unsupported image_view: {self.image_view}")
        try:
            from projectaria_tools.core import calibration  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "projectaria-tools calibration is required for --image-view egodelta_pinhole"
            ) from exc
        rect_unrotated = calibration.distort_by_calibration(
            image,
            self.linear_calib,
            self.rgb_calib_orig,
        )
        rect_rotated = np.rot90(np.asarray(rect_unrotated), k=-1)
        return np.ascontiguousarray(rect_rotated)

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_view": self.image_view,
            "rgb_stream_id": self.rgb_stream_id,
            "camera_label": self.camera_label,
            "camera_model": self.camera_model,
            "rect_size": self.rect_size,
            "rect_focal": self.rect_focal,
        }
        if self.output_calib is not None:
            payload["camera_calibration"] = _camera_calibration_payload(self.output_calib)
        return payload


def extract_frames_single_recording(
    vrs_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
    pre_start_s: float,
    pre_end_s: float,
    post_start_s: float,
    post_end_s: float,
) -> dict[str, list[dict[str, Any]]]:
    provider, stream_id = _create_provider(vrs_path, config.rgb_stream_id)
    image_processor = create_image_processor(provider, stream_id, config)
    _write_camera_view_json(out_dir, "single_recording", image_processor)
    num_frames = _get_num_frames(provider, stream_id)
    if num_frames == 0:
        raise RuntimeError(f"No frames found for RGB stream {config.rgb_stream_id}")

    first_timestamp_ns = _timestamp_for_frame(provider, stream_id, 0)
    split_counts = {"pre": 0, "post": 0}
    records = {"pre": [], "post": []}

    for frame_index in range(num_frames):
        timestamp_ns = _timestamp_for_frame(provider, stream_id, frame_index)
        rel_s = (timestamp_ns - first_timestamp_ns) / 1e9
        split = None
        if pre_start_s <= rel_s <= pre_end_s:
            split = "pre"
        elif post_start_s <= rel_s <= post_end_s:
            split = "post"
        if split is None:
            continue
        if split_counts[split] % max(1, config.frame_stride) != 0:
            split_counts[split] += 1
            continue
        split_counts[split] += 1
        record = _save_frame(
            provider,
            stream_id,
            frame_index,
            timestamp_ns,
            split,
            out_dir,
            image_processor,
        )
        records[split].append(record.to_dict())

    _write_frame_json(out_dir, "pre", records["pre"])
    _write_frame_json(out_dir, "post", records["post"])
    return records


def extract_frames_separate_recordings(
    pre_vrs_path: str | Path,
    post_vrs_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
) -> dict[str, list[dict[str, Any]]]:
    records = {
        "pre": extract_frames_from_recording(pre_vrs_path, out_dir, config, "pre"),
        "post": extract_frames_from_recording(post_vrs_path, out_dir, config, "post"),
    }
    return records


def extract_frames_from_recording(
    vrs_path: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
    split: str,
) -> list[dict[str, Any]]:
    provider, stream_id = _create_provider(vrs_path, config.rgb_stream_id)
    image_processor = create_image_processor(provider, stream_id, config)
    _write_camera_view_json(out_dir, split, image_processor)
    num_frames = _get_num_frames(provider, stream_id)
    records: list[dict[str, Any]] = []
    for frame_index in range(0, num_frames, max(1, config.frame_stride)):
        timestamp_ns = _timestamp_for_frame(provider, stream_id, frame_index)
        record = _save_frame(
            provider,
            stream_id,
            frame_index,
            timestamp_ns,
            split,
            out_dir,
            image_processor,
        )
        records.append(record.to_dict())
    _write_frame_json(out_dir, split, records)
    return records


def load_extracted_frames_from_pair_manifest(
    pair_manifest: str | Path,
    out_dir: str | Path,
    config: PipelineConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Load pre/post frame records from an already-extracted pair manifest.

    The fast benchmark dataset stores rectified images and COLMAP metadata on
    disk already. This loader records those image paths directly instead of
    extracting or undistorting frames from VRS.
    """

    pair_manifest = Path(pair_manifest)
    manifest = json.loads(pair_manifest.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Pair manifest has no datasets: {pair_manifest}")

    records = {"pre": [], "post": []}
    for dataset in datasets:
        label = str(dataset.get("label", "")).lower()
        if label not in records:
            continue
        records[label] = _records_from_extracted_dataset(
            dataset=dataset,
            split=label,
            config=config,
        )
    if not records["pre"] or not records["post"]:
        raise ValueError(f"Pair manifest must contain pre and post datasets: {pair_manifest}")

    _write_frame_json(out_dir, "pre", records["pre"])
    _write_frame_json(out_dir, "post", records["post"])
    _write_extracted_camera_view_json(out_dir, manifest)
    return records


def _records_from_extracted_dataset(
    dataset: dict[str, Any],
    split: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    frame_list_path = Path(str(dataset["frame_list"]))
    source_root = Path(str(dataset["source_root"]))
    camera = dataset.get("camera", {})
    width = int(camera.get("width", config.rect_size))
    height = int(camera.get("height", config.rect_size))
    frame_names = [
        line.strip()
        for line in frame_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    stride = max(1, int(config.frame_stride))
    records: list[dict[str, Any]] = []
    for sampled_index, frame_name in enumerate(frame_names[::stride]):
        image_path = Path(frame_name)
        if not image_path.is_absolute():
            image_path = source_root / frame_name
        if not image_path.exists():
            raise FileNotFoundError(f"Extracted frame does not exist: {image_path}")
        source_frame_index = _source_frame_index_from_name(image_path.name, sampled_index)
        frame_id = f"{split}_{source_frame_index:06d}"
        records.append(
            FrameRecord(
                frame_id=frame_id,
                split=split,
                timestamp_ns=int(source_frame_index),
                image_path=str(image_path),
                original_width=width,
                original_height=height,
                source_frame_index=int(source_frame_index),
                image_view="pinhole",
                camera_label=None,
                camera_model="pinhole",
                rect_size=width if width == height else None,
                rect_focal=float(camera.get("fx", config.rect_focal)),
            ).to_dict()
        )
    return records


def _source_frame_index_from_name(name: str, fallback: int) -> int:
    stem = Path(name).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1])
    return int(fallback)


def _write_extracted_camera_view_json(out_dir: str | Path, manifest: dict[str, Any]) -> None:
    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera = {}
    for dataset in manifest.get("datasets", []):
        if isinstance(dataset, dict) and isinstance(dataset.get("camera"), dict):
            camera = dataset["camera"]
            break
    payload = {
        "image_view": "pinhole",
        "camera_model": "pinhole",
        "camera_calibration": camera,
        "source": "already_extracted_frames",
    }
    (frames_dir / "extracted_camera_view.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def create_image_processor(provider: Any, stream_id: Any, config: PipelineConfig) -> FrameImageProcessor:
    image_view = str(config.image_view or "raw")
    if image_view == "raw":
        return FrameImageProcessor(
            image_view="raw",
            rgb_stream_id=config.rgb_stream_id,
            camera_label=None,
            camera_model="raw",
        )
    if image_view != "egodelta_pinhole":
        raise ValueError("--image-view must be one of: raw, egodelta_pinhole")

    try:
        from projectaria_tools.core import calibration  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "projectaria-tools calibration is required for --image-view egodelta_pinhole"
        ) from exc

    device_calib = provider.get_device_calibration()
    camera_label = _resolve_camera_label(
        provider=provider,
        device_calib=device_calib,
        stream_id=stream_id,
        rgb_stream_id=config.rgb_stream_id,
        preferred_label=config.aria_camera_label,
    )
    rgb_calib_orig = device_calib.get_camera_calib(camera_label)
    T_device_camera_orig = rgb_calib_orig.get_transform_device_camera()
    linear = calibration.get_linear_camera_calibration(
        int(config.rect_size),
        int(config.rect_size),
        float(config.rect_focal),
        f"{camera_label}-linear",
        T_device_camera_orig,
    )
    linear_rot_calib = calibration.rotate_camera_calib_cw90deg(linear)
    return FrameImageProcessor(
        image_view="egodelta_pinhole",
        rgb_stream_id=config.rgb_stream_id,
        camera_label=camera_label,
        camera_model="pinhole_cw90",
        rect_size=int(config.rect_size),
        rect_focal=float(config.rect_focal),
        rgb_calib_orig=rgb_calib_orig,
        linear_calib=linear,
        output_calib=linear_rot_calib,
    )


def _create_provider(vrs_path: str | Path, rgb_stream_id: str) -> tuple[Any, Any]:
    try:
        from projectaria_tools.core import data_provider  # type: ignore
        from projectaria_tools.core.stream_id import StreamId  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "projectaria-tools is required for VRS frame extraction. "
            "Install requirements.txt before running the pipeline."
        ) from exc

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Failed to create VRS data provider for {vrs_path}")
    return provider, StreamId(rgb_stream_id)


def _get_num_frames(provider: Any, stream_id: Any) -> int:
    for method_name in ["get_num_data", "get_num_data_for_stream"]:
        method = getattr(provider, method_name, None)
        if callable(method):
            try:
                return int(method(stream_id))
            except TypeError:
                try:
                    return int(method(stream_id, None))
                except Exception:
                    pass
            except Exception:
                pass
    raise RuntimeError("Could not query number of RGB frames from the VRS provider")


def _timestamp_for_frame(provider: Any, stream_id: Any, frame_index: int) -> int:
    _, record = _get_image_data_by_index(provider, stream_id, frame_index)
    timestamp = _extract_timestamp_ns(record)
    if timestamp is None:
        raise RuntimeError(f"Could not read timestamp for frame index {frame_index}")
    return timestamp


def _save_frame(
    provider: Any,
    stream_id: Any,
    frame_index: int,
    timestamp_ns: int,
    split: str,
    out_dir: str | Path,
    image_processor: FrameImageProcessor,
) -> FrameRecord:
    image_data, _ = _get_image_data_by_index(provider, stream_id, frame_index)
    source_image = _image_to_numpy(image_data)
    source_height, source_width = source_image.shape[:2]
    image = image_processor.transform(source_image)
    height, width = image.shape[:2]

    frame_id = f"{split}_{frame_index:06d}"
    image_dir = Path(out_dir) / "frames" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{frame_id}.jpg"
    if image.ndim == 3 and image.shape[2] == 3:
        image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_write = image
    ok = cv2.imwrite(str(image_path), image_to_write)
    if not ok:
        raise RuntimeError(f"Failed to write frame image: {image_path}")

    return FrameRecord(
        frame_id=frame_id,
        split=split,
        timestamp_ns=int(timestamp_ns),
        image_path=str(image_path),
        original_width=int(width),
        original_height=int(height),
        source_frame_index=int(frame_index),
        image_view=image_processor.image_view,
        camera_label=image_processor.camera_label,
        camera_model=image_processor.camera_model,
        rect_size=image_processor.rect_size,
        rect_focal=image_processor.rect_focal,
        source_original_width=int(source_width),
        source_original_height=int(source_height),
    )


def _get_image_data_by_index(provider: Any, stream_id: Any, frame_index: int) -> tuple[Any, Any]:
    method = getattr(provider, "get_image_data_by_index", None)
    if not callable(method):
        raise RuntimeError("VRS provider does not expose get_image_data_by_index")
    result = method(stream_id, int(frame_index))
    if not isinstance(result, tuple):
        raise RuntimeError("Unexpected projectaria image data return type")
    if len(result) >= 2:
        return result[0], result[1]
    raise RuntimeError("Unexpected projectaria image data tuple length")


def _extract_timestamp_ns(record: Any) -> int | None:
    for name in [
        "capture_timestamp_ns",
        "device_timestamp_ns",
        "timestamp_ns",
        "tracking_timestamp_ns",
    ]:
        value = getattr(record, name, None)
        if value is not None:
            return int(value() if callable(value) else value)
    for name in ["capture_timestamp", "timestamp"]:
        value = getattr(record, name, None)
        if value is not None:
            raw = value() if callable(value) else value
            if hasattr(raw, "total_seconds"):
                return int(round(raw.total_seconds() * 1e9))
            return int(raw)
    return None


def _image_to_numpy(image_data: Any) -> np.ndarray:
    for method_name in ["to_numpy_array", "numpy", "as_numpy_array"]:
        method = getattr(image_data, method_name, None)
        if callable(method):
            image = np.asarray(method())
            if image.ndim in {2, 3}:
                return image
    image = np.asarray(image_data)
    if image.ndim in {2, 3}:
        return image
    raise RuntimeError("Could not convert Project Aria image data to a numpy array")


def _write_frame_json(out_dir: str | Path, split: str, records: list[dict[str, Any]]) -> None:
    path = Path(out_dir) / "frames" / f"{split}_frames.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _write_camera_view_json(
    out_dir: str | Path,
    split: str,
    image_processor: FrameImageProcessor,
) -> None:
    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"{split}_camera_view.json"
    path.write_text(json.dumps(image_processor.metadata(), indent=2), encoding="utf-8")


def _resolve_camera_label(
    provider: Any,
    device_calib: Any,
    stream_id: Any,
    rgb_stream_id: str,
    preferred_label: str | None,
) -> str:
    labels: list[str] = []
    if preferred_label:
        labels.append(str(preferred_label))
    for method_name in ["get_label_from_stream_id", "get_stream_label"]:
        method = getattr(provider, method_name, None)
        if callable(method):
            try:
                label = method(stream_id)
                if label:
                    labels.append(str(label))
            except Exception:
                pass
    labels.extend([rgb_stream_id, "camera-rgb", "rgb", "camera_rgb"])
    method = getattr(device_calib, "get_camera_calibs", None)
    if callable(method):
        try:
            calibs = method()
            if isinstance(calibs, dict):
                labels.extend(str(k) for k in calibs.keys())
        except Exception:
            pass
    for label in dict.fromkeys(labels):
        try:
            if device_calib.get_camera_calib(label) is not None:
                return label
        except Exception:
            continue
    raise RuntimeError(f"Could not find RGB camera calibration for stream {rgb_stream_id}")


def _camera_calibration_payload(camera_calib: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for method_name, key in [
        ("get_image_size", "image_size"),
        ("get_focal_lengths", "focal_lengths"),
        ("get_principal_point", "principal_point"),
    ]:
        method = getattr(camera_calib, method_name, None)
        if callable(method):
            try:
                value = method()
                payload[key] = np.asarray(value).tolist()
            except Exception:
                pass
    transform = getattr(camera_calib, "get_transform_device_camera", None)
    if callable(transform):
        try:
            payload["T_device_camera"] = np.asarray(transform().to_matrix()).tolist()
        except Exception:
            pass
    projection_params = getattr(camera_calib, "get_projection_params", None)
    if callable(projection_params):
        try:
            payload["projection_params"] = np.asarray(projection_params()).tolist()
        except Exception:
            pass
    return payload
