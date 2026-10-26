from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a live RealSense color stream as trial evidence.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-duration-s", type=float, default=180.0)
    args = parser.parse_args()

    output = args.output.resolve()
    metadata = (args.metadata or output.with_suffix(output.suffix + ".json")).resolve()
    ready_file = None if args.ready_file is None else args.ready_file.resolve()
    stop_file = None if args.stop_file is None else args.stop_file.resolve()
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    for path in (output, metadata, partial):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite video artifact: {path}")
    if ready_file is not None and ready_file.exists():
        ready_file.unlink()
    if stop_file is not None and stop_file.exists():
        stop_file.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    writer = cv2.VideoWriter(
        str(partial),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open MP4 writer: {partial}")

    started_at = utc_now()
    started_monotonic = time.monotonic()
    frame_count = 0
    first_timestamp_ms: float | None = None
    last_timestamp_ms: float | None = None
    error: str | None = None
    pipeline_started = False
    try:
        pipeline.start(config)
        pipeline_started = True
        while not STOP_REQUESTED:
            if stop_file is not None and stop_file.exists():
                break
            if time.monotonic() - started_monotonic >= float(args.max_duration_s):
                break
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            writer.write(image)
            timestamp_ms = float(color_frame.get_timestamp())
            first_timestamp_ms = timestamp_ms if first_timestamp_ms is None else first_timestamp_ms
            last_timestamp_ms = timestamp_ms
            frame_count += 1
            if frame_count == 1 and ready_file is not None:
                write_json(
                    ready_file,
                    {
                        "ok": True,
                        "started_at_utc": started_at,
                        "output": str(output),
                        "first_frame_timestamp_ms": timestamp_ms,
                    },
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        writer.release()
        if pipeline_started:
            pipeline.stop()

    ended_at = utc_now()
    elapsed_s = max(0.0, time.monotonic() - started_monotonic)
    ok = error is None and frame_count > 0 and partial.is_file() and partial.stat().st_size > 0
    if ok:
        partial.replace(output)
    record = {
        "schema": "egodelta_realsense_trial_video_v1",
        "ok": bool(ok),
        "output": str(output),
        "partial_output": str(partial),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_s": elapsed_s,
        "frame_count": int(frame_count),
        "requested_fps": int(args.fps),
        "observed_fps": 0.0 if elapsed_s <= 0.0 else float(frame_count) / elapsed_s,
        "width": int(args.width),
        "height": int(args.height),
        "first_frame_timestamp_ms": first_timestamp_ms,
        "last_frame_timestamp_ms": last_timestamp_ms,
        "error": error,
    }
    write_json(metadata, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
