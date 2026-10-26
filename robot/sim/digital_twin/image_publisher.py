"""In-Isaac-Sim viewport image publisher (ZMQ PUB) via non-destructive viewport capture.

Runs *inside* a live Isaac Sim 5.1 Kit process. Captures the active viewport's
already-rendered color AOV with ``omni.kit.viewport.utility.capture_viewport_to_buffer``
-- this does NOT create/resize a render product, so it never alters the viewport
(unlike binding an ``isaacsim.sensors.camera`` Camera to the viewport render product,
which forces its resolution). It works in a live GUI whether or not the timeline is
playing, and does not need the replicator SDG/orchestrator. Each captured frame is
JPEG-encoded and published over a ZMQ PUB socket so a host-side client can grab
snapshots on demand to visually verify the scene/robot.

Wire format (single ZMQ frame, so ZMQ_CONFLATE works end-to-end):
    [ 4-byte big-endian header length | header JSON (utf-8) | JPEG bytes ]
header JSON = {"w","h","fmt":"jpg","frame","t"}

Designed to be exec'd in the Kit global scope (via the isaacsim.code_editor.vscode
socket). Idempotent: the live publisher is stored on ``builtins`` so re-exec'ing
this file does not rebind the socket or duplicate the update subscription.

Public API (each prints a one-line status that the code-editor socket returns):
    start_image_publisher(endpoint="tcp://127.0.0.1:5556", fps=10.0, jpeg_quality=80)
    stop_image_publisher()
    image_publisher_status()
"""
import builtins
import ctypes
import json
import struct
import time

_SINGLETON_KEY = "_ISAAC_IMG_PUB"


def _log(msg):
    print("[image_publisher] " + str(msg))


def _ensure_zmq():
    try:
        import zmq  # noqa: F401
        return
    except ImportError:
        _log("pyzmq not found; installing via omni.kit.pipapi (one-time)...")
        import omni.kit.pipapi
        omni.kit.pipapi.install("pyzmq")
        import zmq  # noqa: F401
        _log("pyzmq installed: " + zmq.__version__)


def _read_capture_buffer(buffer, buffer_size):
    """Return a uint8 numpy copy of the capture buffer (PyCapsule or int pointer)."""
    import numpy as np

    ptr = None
    try:
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
    except Exception:
        try:
            ptr = int(buffer)
        except Exception:
            ptr = None
    if not ptr:
        return None
    raw = (ctypes.c_byte * buffer_size).from_address(int(ptr))
    return np.frombuffer(raw, dtype=np.uint8).copy()


class _ImagePublisher:
    def __init__(self, endpoint, fps, jpeg_quality):
        self.endpoint = endpoint
        self.period = 1.0 / float(fps) if fps and fps > 0 else 0.0
        self.jpeg_quality = int(jpeg_quality)

        self._socket = None
        self._sub = None
        self._vp = None
        self._last = 0.0
        self._cap_pending = False
        self._cap_since = 0.0

        self.frame = 0
        self.sent = 0
        self.dropped = 0
        self.empty = 0
        self.errors = 0
        self.last_error = ""
        self.last_shape = None
        self.last_format = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        import zmq
        from omni.kit.viewport.utility import get_active_viewport

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self.endpoint)
        self._socket = sock
        _log("PUB bound at " + self.endpoint)

        self._vp = get_active_viewport()
        if self._vp is None:
            raise RuntimeError("no active viewport")

        import omni.kit.app
        self._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="isaac_image_publisher"
        )
        _log("viewport-capture publisher started (non-destructive)")
        return self

    def stop(self):
        self._sub = None
        try:
            if self._socket is not None:
                self._socket.close(0)
        except Exception:
            pass
        self._socket = None
        self._vp = None

    # -- per-frame ---------------------------------------------------------
    def _on_update(self, _e):
        if self._socket is None or self._vp is None:
            return
        now = time.monotonic()
        # recover if a scheduled capture never called back (e.g. viewport not rendering)
        if self._cap_pending:
            if (now - self._cap_since) > 2.0:
                self._cap_pending = False
            else:
                return
        if self.period and (now - self._last) < self.period:
            return
        self._last = now
        try:
            from omni.kit.viewport.utility import capture_viewport_to_buffer
            self._cap_pending = True
            self._cap_since = now
            capture_viewport_to_buffer(self._vp, self._on_capture)
        except Exception as e:  # noqa: BLE001
            self._cap_pending = False
            self.errors += 1
            self.last_error = "schedule: " + repr(e)

    def _on_capture(self, *args):
        try:
            buffer = args[0]
            buffer_size = int(args[1])
            width = int(args[2])
            height = int(args[3])
            self.last_format = str(args[4]) if len(args) > 4 else None
            if width <= 0 or height <= 0 or buffer_size <= 0:
                self.empty += 1
                return

            import numpy as np  # noqa: F401
            import cv2
            import zmq

            flat = _read_capture_buffer(buffer, buffer_size)
            if flat is None or flat.size < width * height * 3:
                self.empty += 1
                return
            ch = buffer_size // (width * height)
            if ch >= 4:
                img = flat[: width * height * 4].reshape(height, width, 4)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif ch == 3:
                img = flat[: width * height * 3].reshape(height, width, 3)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                self.empty += 1
                return
            self.last_shape = (height, width, ch)

            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                self.errors += 1
                return
            payload = enc.tobytes()
            hdr = json.dumps(
                {"w": width, "h": height, "fmt": "jpg", "frame": self.frame, "t": time.time()}
            ).encode("utf-8")
            buf = struct.pack(">I", len(hdr)) + hdr + payload
            try:
                self._socket.send(buf, zmq.NOBLOCK)
                self.frame += 1
                self.sent += 1
            except zmq.Again:
                self.dropped += 1
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            self.last_error = "capture: " + repr(e)
        finally:
            self._cap_pending = False

    # -- introspection -----------------------------------------------------
    def status(self):
        return {
            "running": self._socket is not None and self._sub is not None,
            "endpoint": self.endpoint,
            "method": "capture_viewport_to_buffer",
            "fps": (round(1.0 / self.period, 2) if self.period else "uncapped"),
            "frames_sent": self.sent,
            "dropped_backpressure": self.dropped,
            "empty_frames": self.empty,
            "errors": self.errors,
            "last_error": self.last_error,
            "last_shape": self.last_shape,
            "last_format": self.last_format,
        }


# -- module-level API ------------------------------------------------------
def start_image_publisher(endpoint="tcp://127.0.0.1:5556", fps=10.0, jpeg_quality=80):
    existing = builtins.__dict__.get(_SINGLETON_KEY)
    if existing is not None and existing._socket is not None:
        _log("already running; returning status (call stop_image_publisher() to reset)")
        print(json.dumps(existing.status()))
        return existing

    _ensure_zmq()
    pub = _ImagePublisher(endpoint, fps, jpeg_quality)
    try:
        pub.start()
    except Exception as e:  # noqa: BLE001
        pub.stop()
        _log("FAILED to start: " + repr(e))
        raise
    builtins.__dict__[_SINGLETON_KEY] = pub
    _log("started")
    print(json.dumps(pub.status()))
    return pub


def stop_image_publisher():
    existing = builtins.__dict__.get(_SINGLETON_KEY)
    if existing is None:
        _log("no publisher running")
        return
    existing.stop()
    builtins.__dict__[_SINGLETON_KEY] = None
    _log("stopped")


def image_publisher_status():
    existing = builtins.__dict__.get(_SINGLETON_KEY)
    print(json.dumps(existing.status() if existing is not None else {"running": False}))
