"""Real-time tiled viewer for the RB-Y1 RealSense cameras (base D455 + 2 wrist D405).

Shows all configured cameras side-by-side in one live OpenCV window. Useful for
aiming cameras and for confirming which D405 serial is on which arm: wave a hand in
front of one arm's wrist camera and see which tile moves.

IMPORTANT: a RealSense device can only be streamed by ONE process at a time. Close
any running openpi_rollout before launching this viewer (and vice versa). To see the
camera views DURING a rollout, run the rollout with `--show-cameras` instead.

Run:
    python -m robot.policy.openpi.multicam_view                  # all cams in config
    python -m robot.policy.openpi.multicam_view --serials base=BASE_CAMERA_SERIAL right_wrist=LEFT_WRIST_CAMERA_SERIAL
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from robot.policy.openpi.multicam_capture import MultiCamRealSense, list_devices

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "config" / "realsense_cameras.json"


def _draw_label(bgr: np.ndarray, text: str, *, y: int = 0) -> None:
    import cv2

    cv2.rectangle(bgr, (0, y), (bgr.shape[1], y + 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1, cv2.LINE_AA)


def tile_frames(named_rgb: dict[str, np.ndarray], *, tile_height: int = 360,
                subtitles: dict[str, str] | None = None) -> np.ndarray:
    """Resize each RGB frame to a common height, convert to BGR, label, and hstack.

    Returns a single BGR image ready for cv2.imshow. Shared by the standalone viewer
    and the rollout's --show-cameras display.
    """
    import cv2

    tiles = []
    for name, rgb in named_rgb.items():
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR for OpenCV
        h, w = bgr.shape[:2]
        tw = max(1, int(round(w * tile_height / h)))
        bgr = cv2.resize(bgr, (tw, tile_height))
        _draw_label(bgr, name, y=0)
        if subtitles and name in subtitles:
            _draw_label(bgr, subtitles[name], y=tile_height - 22)
        tiles.append(bgr)
    if not tiles:
        return np.zeros((tile_height, tile_height, 3), dtype=np.uint8)
    return np.hstack(tiles)


class CameraStreamer:
    """Background thread that continuously reads a MultiCamRealSense and shows a live
    tiled window.

    The display stays smooth even while the main (control) thread is blocked in a
    multi-second robot motion command, because reading + imshow + waitKey all happen
    in this thread. The control loop pulls the most recent frames via ``latest()``
    instead of calling ``cams.read()`` itself (a RealSense pipeline must only be read
    from one thread). The window is owned start-to-finish by this thread, which is the
    arrangement OpenCV's HighGUI needs.
    """

    def __init__(self, cams, *, window: str = "openpi cameras (live, q to close)", tile_height: int = 360) -> None:
        self._cams = cams
        self._window = window
        self._tile_height = tile_height
        self._latest: dict | None = None
        self._status = ""
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-streamer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import cv2

        while not self._stop_evt.is_set():
            try:
                frames = self._cams.read()
            except Exception:
                continue
            with self._lock:
                self._latest = frames
                status = self._status
            named = {name: f["color_rgb"] for name, f in frames.items()}
            subs = {name: f"{self._cams.cameras[name].serial}  {status}" for name in named}
            grid = tile_frames(named, tile_height=self._tile_height, subtitles=subs)
            cv2.imshow(self._window, grid)
            cv2.waitKey(1)

    def latest(self, timeout: float = 10.0) -> dict:
        """Return the most recently captured frames dict (waits for the first frame)."""
        deadline = time.perf_counter() + timeout
        while True:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            if time.perf_counter() > deadline:
                raise RuntimeError("camera streamer produced no frames within timeout")
            time.sleep(0.01)

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> None:
    import cv2

    p = argparse.ArgumentParser(description="Live tiled RealSense viewer.")
    p.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    p.add_argument("--serials", nargs="+", default=None,
                   help="name=serial pairs; overrides the camera config (e.g. base=333.. rw=409..).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tile-height", type=int, default=360)
    p.add_argument("--warmup-frames", type=int, default=10)
    args = p.parse_args()

    if args.serials:
        serials = dict(s.split("=", 1) for s in args.serials)
    else:
        data = json.loads(Path(args.camera_config).read_text(encoding="utf-8"))
        serials = {k: v for k, v in data.items() if not k.startswith("_")}

    print("Attached RealSense devices:")
    for d in list_devices():
        print(f"  serial={d['serial']}  {d['name']}")
    print(f"Opening: {serials}")
    print("Tip: wave a hand in front of an arm's wrist cam to see which tile moves -> that's its serial.")

    cams = MultiCamRealSense(serials, width=args.width, height=args.height, fps=args.fps,
                             warmup_frames=args.warmup_frames)
    window = "RB-Y1 cameras (q / ESC to quit)"
    last = time.perf_counter()
    fps_est = 0.0
    try:
        while True:
            frames = cams.read()
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt)
            named = {name: f["color_rgb"] for name, f in frames.items()}
            subs = {name: f"{cams.cameras[name].serial}  {fps_est:4.1f} fps" for name, f in frames.items()}
            grid = tile_frames(named, tile_height=args.tile_height, subtitles=subs)
            cv2.imshow(window, grid)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cams.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
