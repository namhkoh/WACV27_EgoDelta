"""Host-side ZMQ snapshot client for the in-Isaac-Sim image_publisher.

Connects (SUB) to the publisher's PUB endpoint, grabs the latest frame (ZMQ_CONFLATE),
and writes the JPEG straight to a file. Requires only ``pyzmq``.

Usage:
    python snapshot_client.py --out shot.jpg
    python snapshot_client.py --endpoint tcp://127.0.0.1:5556 --out shot.jpg --timeout 5
"""
import argparse
import json
import os
import struct
import sys

# Snapshots are written here by default.
SNAP_DIR = r"C:\path\to\projects\EgoDelta\robot\sim\snapshots"


def grab(endpoint: str, out_path: str, timeout: float) -> dict:
    import zmq

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.CONFLATE, 1)  # only keep the most recent frame
    sub.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(endpoint)
    try:
        buf = sub.recv()
    finally:
        sub.close(0)

    hlen = struct.unpack(">I", buf[:4])[0]
    hdr = json.loads(buf[4 : 4 + hlen].decode("utf-8"))
    payload = buf[4 + hlen :]
    with open(out_path, "wb") as f:
        f.write(payload)
    hdr["bytes"] = len(payload)
    hdr["saved"] = out_path
    return hdr


def main() -> int:
    ap = argparse.ArgumentParser(description="Grab one snapshot from the Isaac Sim image publisher.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    ap.add_argument("--out", default=os.path.join(SNAP_DIR, "snapshot.jpg"),
                    help="output path; a bare filename is placed in the snapshots dir")
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args()
    # a bare filename (no directory) goes into the snapshots dir
    if not os.path.dirname(a.out):
        a.out = os.path.join(SNAP_DIR, a.out)
    try:
        info = grab(a.endpoint, a.out, a.timeout)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("SNAPSHOT_ERR: " + repr(e) + "\n")
        return 1
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
