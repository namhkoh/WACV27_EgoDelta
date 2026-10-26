"""Persistent multi-camera RealSense capture for openpi rollouts.

openpi (pi0/pi0.5) needs a synchronized multi-view RGB observation on every
control tick. The existing ``robot/kinematics/utils/capture_live_realsense.py``
is single-shot (it opens AND stops a fresh pipeline per call, with a 30-frame
warmup) which is fine for one-time goal selection but far too slow for a control
loop. This module keeps one ``rs.pipeline`` open per camera for the whole rollout
and reads all of them once per tick.

Design choices (kept deliberately close to capture_live_realsense.py):
  * Each camera is selected by explicit serial (``config.enable_device(serial)``),
    because with 3 RealSense attached the default-device order is non-deterministic.
  * Color is requested as ``rgb8`` so frames come out RGB directly (openpi/pi0
    expects RGB; the legacy capture used bgr8 and flipped later).
  * Depth is optional and OFF by default -- openpi consumes RGB only, and three
    color+depth streams on one USB controller routinely exceed bandwidth.
  * Frames across separate pipelines are NOT hardware-synchronized; we read them
    back-to-back within one tick and stamp each with its sensor timestamp so the
    caller can check skew.

Run standalone to discover attached cameras::

    python -m robot.policy.openpi.multicam_capture --list
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pyrealsense2 as rs


def list_devices() -> list[dict[str, str]]:
    """Enumerate connected RealSense devices (serial, name, product line)."""
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append(
            {
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
                "product_line": dev.get_info(rs.camera_info.product_line),
            }
        )
    return devices


class RealSenseCamera:
    """One persistent RealSense color (+ optional depth) stream, opened by serial."""

    def __init__(
        self,
        *,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ) -> None:
        self.serial = serial
        self.enable_depth = enable_depth
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self._align: rs.align | None = rs.align(rs.stream.color)
        else:
            self._align = None
        self._profile = self._pipeline.start(config)
        self._depth_scale = (
            float(self._profile.get_device().first_depth_sensor().get_depth_scale())
            if enable_depth
            else 1.0
        )

    def warmup(self, frames: int = 30) -> None:
        for _ in range(max(0, int(frames))):
            self._pipeline.wait_for_frames()

    def read(self) -> dict[str, Any]:
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"RealSense {self.serial} returned no color frame")
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        out: dict[str, Any] = {
            "color_rgb": np.asanyarray(color_frame.get_data()).copy(),  # HxWx3 uint8 RGB
            "intrinsics": {
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
            },
            "color_timestamp_ms": float(color_frame.get_timestamp()),
            "color_frame_number": int(color_frame.get_frame_number()),
        }
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                out["depth_m"] = (
                    np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
                )
        return out

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


class MultiCamRealSense:
    """Opens several named RealSense cameras by serial and reads them per tick.

    ``serials`` maps a logical name (e.g. ``"base"``, ``"left_wrist"``,
    ``"right_wrist"``) to a device serial. Use as a context manager so every
    pipeline is closed even on error.
    """

    def __init__(
        self,
        serials: dict[str, str],
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
        warmup_frames: int = 30,
    ) -> None:
        if not serials:
            raise ValueError("serials must map at least one name -> serial")
        self.cameras: dict[str, RealSenseCamera] = {}
        try:
            for name, serial in serials.items():
                self.cameras[name] = RealSenseCamera(
                    serial=serial,
                    width=width,
                    height=height,
                    fps=fps,
                    enable_depth=enable_depth,
                )
            # Warm up all streams once (auto-exposure settle) before the loop.
            for cam in self.cameras.values():
                cam.warmup(warmup_frames)
        except Exception:
            self.close()
            raise

    def read(self) -> dict[str, dict[str, Any]]:
        """Read every camera once. Returns {name: frame_dict}. Soft-synchronized."""
        return {name: cam.read() for name, cam in self.cameras.items()}

    def close(self) -> None:
        for cam in self.cameras.values():
            cam.close()
        self.cameras = {}

    def __enter__(self) -> "MultiCamRealSense":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense multi-camera utility.")
    parser.add_argument("--list", action="store_true", help="List attached RealSense devices and exit.")
    parser.add_argument("--probe", nargs="+", metavar="NAME=SERIAL",
                        help="Open the given name=serial cameras and print one frame's shape/timestamp.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.list or not args.probe:
        devs = list_devices()
        if not devs:
            print("No RealSense devices found.")
        for d in devs:
            print(f"serial={d['serial']}  product_line={d['product_line']}  name={d['name']}")
        if not args.probe:
            return

    serials = {}
    for item in args.probe:
        name, _, serial = item.partition("=")
        serials[name] = serial
    with MultiCamRealSense(serials, width=args.width, height=args.height, fps=args.fps) as cams:
        t0 = time.perf_counter()
        frames = cams.read()
        dt_ms = (time.perf_counter() - t0) * 1e3
        print(f"read {len(frames)} cameras in {dt_ms:.1f} ms")
        ts = {name: f["color_timestamp_ms"] for name, f in frames.items()}
        for name, f in frames.items():
            print(f"  {name}: color_rgb {f['color_rgb'].shape} {f['color_rgb'].dtype}  ts={f['color_timestamp_ms']:.1f}ms")
        if len(ts) > 1:
            spread = max(ts.values()) - min(ts.values())
            print(f"  inter-camera timestamp spread: {spread:.1f} ms")


if __name__ == "__main__":
    main()
