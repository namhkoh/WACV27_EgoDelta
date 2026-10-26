from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import EGODELTA_RECTIFIED_RGB_FOCAL, EGODELTA_RECTIFIED_RGB_SIZE


def _require_projectaria() -> tuple[Any, Any]:
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except Exception as exc:
        raise RuntimeError(
            "projectaria-tools is required to extract RGB frames from VRS. "
            "Install it with scripts/setup_sam3_native.sh or scripts/setup_sam3_ultralytics.sh."
        ) from exc
    return data_provider, StreamId


def _get_timestamp_ns(record: Any) -> int:
    for name in (
        "capture_timestamp_ns",
        "device_timestamp_ns",
        "tracking_timestamp_ns",
        "timestamp_ns",
    ):
        value = getattr(record, name, None)
        if value is not None:
            return int(value)
    if hasattr(record, "capture_timestamp"):
        ts = record.capture_timestamp
        if hasattr(ts, "total_seconds"):
            return int(ts.total_seconds() * 1e9)
    raise RuntimeError("Could not read timestamp from Project Aria image record.")


def _image_to_numpy(image: Any) -> np.ndarray:
    if hasattr(image, "to_numpy_array"):
        array = image.to_numpy_array()
    else:
        array = np.asarray(image)
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8, copy=False)


def _get_image_data_by_index(provider: Any, stream_id: Any, index: int) -> tuple[np.ndarray, int]:
    data = provider.get_image_data_by_index(stream_id, index)
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        image = data[0]
        record = data[1]
    else:
        image = getattr(data, "image", data)
        record = getattr(data, "record", data)
    return _image_to_numpy(image), _get_timestamp_ns(record)


def _get_num_images(provider: Any, stream_id: Any) -> int:
    for method in ("get_num_data", "get_num_images", "get_num_records"):
        if hasattr(provider, method):
            try:
                return int(getattr(provider, method)(stream_id))
            except Exception:
                continue
    raise RuntimeError("Could not determine number of RGB frames in VRS.")


def _get_rgb_camera_calibration(provider: Any, stream_id: Any, rgb_stream_id: str) -> Any:
    device_calib = provider.get_device_calibration()
    labels = [rgb_stream_id, "camera-rgb", "camera_rgb", "rgb"]
    try:
        labels.insert(0, provider.get_label_from_stream_id(stream_id))
    except Exception:
        pass
    for label in labels:
        try:
            camera_calib = device_calib.get_camera_calib(label)
            if camera_calib is not None:
                return camera_calib
        except Exception:
            continue
    raise RuntimeError(f"Could not find RGB camera calibration for stream {rgb_stream_id}.")


def _make_egodelta_linear_camera_calibration(
    camera_calib: Any,
    rect_size: int = EGODELTA_RECTIFIED_RGB_SIZE,
    rect_focal: float = EGODELTA_RECTIFIED_RGB_FOCAL,
) -> Any:
    from projectaria_tools.core import calibration

    return calibration.get_linear_camera_calibration(
        int(rect_size),
        int(rect_size),
        float(rect_focal),
        "camera-rgb-linear",
        camera_calib.get_transform_device_camera(),
    )


def _undistort_image_egodelta(image: np.ndarray, raw_calib: Any, linear_calib: Any) -> np.ndarray:
    from projectaria_tools.core import calibration

    expected_width = int(raw_calib.get_image_size()[0])
    expected_height = int(raw_calib.get_image_size()[1])
    if image.shape[0] != expected_height or image.shape[1] != expected_width:
        raise RuntimeError(
            f"Cannot undistort image shape {image.shape[:2]} with calibration size "
            f"{expected_width}x{expected_height}."
        )
    rect_unrotated = calibration.distort_by_calibration(image, linear_calib, raw_calib)
    return np.rot90(np.asarray(rect_unrotated, dtype=np.uint8), k=-1).copy()


def write_frames_metadata(frames: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        json.dump(frames, handle, indent=2)


def load_frames_metadata(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "frames" in data:
        return list(data["frames"])
    return list(data)


def extract_rgb_frames(
    vrs_path: str | Path,
    out_dir: str | Path,
    split: str,
    rgb_stream_id: str = "214-1",
    frame_stride: int = 10,
    time_window_s: tuple[float, float] | None = None,
    t0_ns: int | None = None,
    image_ext: str = ".jpg",
    undistort: bool = False,
    undistort_rect_size: int = EGODELTA_RECTIFIED_RGB_SIZE,
    undistort_rect_focal: float = EGODELTA_RECTIFIED_RGB_FOCAL,
) -> tuple[list[dict[str, Any]], int]:
    data_provider, StreamId = _require_projectaria()
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    stream_id = StreamId(rgb_stream_id)
    raw_calib = None
    linear_calib = None
    if undistort:
        raw_calib = _get_rgb_camera_calibration(provider, stream_id, rgb_stream_id)
        linear_calib = _make_egodelta_linear_camera_calibration(
            raw_calib,
            rect_size=undistort_rect_size,
            rect_focal=undistort_rect_focal,
        )
    num_images = _get_num_images(provider, stream_id)
    if num_images == 0:
        return [], 0 if t0_ns is None else t0_ns

    _, first_timestamp_ns = _get_image_data_by_index(provider, stream_id, 0)
    if t0_ns is None:
        t0_ns = first_timestamp_ns

    frames_root = "frames_undistorted" if undistort else "frames"
    split_dir = Path(out_dir) / frames_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    window_frame_index = 0

    for index in range(num_images):
        image, timestamp_ns = _get_image_data_by_index(provider, stream_id, index)
        relative_time_s = (timestamp_ns - t0_ns) / 1e9
        if time_window_s is not None and not (time_window_s[0] <= relative_time_s <= time_window_s[1]):
            continue
        if window_frame_index % frame_stride != 0:
            window_frame_index += 1
            continue
        window_frame_index += 1
        frame_id = f"{split}_{index:06d}"
        if undistort:
            assert raw_calib is not None and linear_calib is not None
            image = _undistort_image_egodelta(image, raw_calib, linear_calib)
        image_path = split_dir / f"{frame_id}{image_ext}"
        Image.fromarray(image).save(image_path)
        height, width = image.shape[:2]
        frames.append(
            {
                "frame_id": frame_id,
                "split": split,
                "timestamp_ns": int(timestamp_ns),
                "relative_time_s": float(relative_time_s),
                "image_path": str(image_path),
                "width": int(width),
                "height": int(height),
                "is_undistorted": bool(undistort),
                "camera_model": "egodelta_pinhole_cw90" if undistort else "raw",
                "rectified_rgb_size": int(undistort_rect_size) if undistort else None,
                "rectified_rgb_focal": float(undistort_rect_focal) if undistort else None,
                "image_rotation": "cw90" if undistort else "none",
            }
        )
    return frames, t0_ns


def extract_single_recording_frames(
    vrs_path: str | Path,
    out_dir: str | Path,
    rgb_stream_id: str,
    frame_stride: int,
    pre_start_s: float,
    pre_end_s: float,
    post_start_s: float,
    post_end_s: float,
    undistort: bool = False,
    undistort_rect_size: int = EGODELTA_RECTIFIED_RGB_SIZE,
    undistort_rect_focal: float = EGODELTA_RECTIFIED_RGB_FOCAL,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    pre_frames, t0_ns = extract_rgb_frames(
        vrs_path=vrs_path,
        out_dir=out_dir,
        split="pre",
        rgb_stream_id=rgb_stream_id,
        frame_stride=frame_stride,
        time_window_s=(pre_start_s, pre_end_s),
        t0_ns=None,
        undistort=undistort,
        undistort_rect_size=undistort_rect_size,
        undistort_rect_focal=undistort_rect_focal,
    )
    post_frames, _ = extract_rgb_frames(
        vrs_path=vrs_path,
        out_dir=out_dir,
        split="post",
        rgb_stream_id=rgb_stream_id,
        frame_stride=frame_stride,
        time_window_s=(post_start_s, post_end_s),
        t0_ns=t0_ns,
        undistort=undistort,
        undistort_rect_size=undistort_rect_size,
        undistort_rect_focal=undistort_rect_focal,
    )
    frames_dir = Path(out_dir) / ("frames_undistorted" if undistort else "frames")
    pre_json = frames_dir / "pre_frames.json"
    post_json = frames_dir / "post_frames.json"
    write_frames_metadata(pre_frames, pre_json)
    write_frames_metadata(post_frames, post_json)
    return pre_json, post_json, pre_frames, post_frames


def extract_separate_recording_frames(
    pre_vrs_path: str | Path,
    post_vrs_path: str | Path,
    out_dir: str | Path,
    rgb_stream_id: str,
    frame_stride: int,
    undistort: bool = False,
    undistort_rect_size: int = EGODELTA_RECTIFIED_RGB_SIZE,
    undistort_rect_focal: float = EGODELTA_RECTIFIED_RGB_FOCAL,
) -> tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    pre_frames, _ = extract_rgb_frames(
        vrs_path=pre_vrs_path,
        out_dir=out_dir,
        split="pre",
        rgb_stream_id=rgb_stream_id,
        frame_stride=frame_stride,
        undistort=undistort,
        undistort_rect_size=undistort_rect_size,
        undistort_rect_focal=undistort_rect_focal,
    )
    post_frames, _ = extract_rgb_frames(
        vrs_path=post_vrs_path,
        out_dir=out_dir,
        split="post",
        rgb_stream_id=rgb_stream_id,
        frame_stride=frame_stride,
        undistort=undistort,
        undistort_rect_size=undistort_rect_size,
        undistort_rect_focal=undistort_rect_focal,
    )
    frames_dir = Path(out_dir) / ("frames_undistorted" if undistort else "frames")
    pre_json = frames_dir / "pre_frames.json"
    post_json = frames_dir / "post_frames.json"
    write_frames_metadata(pre_frames, pre_json)
    write_frames_metadata(post_frames, post_json)
    return pre_json, post_json, pre_frames, post_frames
