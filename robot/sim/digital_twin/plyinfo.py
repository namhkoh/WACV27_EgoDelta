"""Inspect an INRIA-format 3D Gaussian Splat .ply: extent, distance from origin, scale.

Reads only the xyz (+ scale) columns via a memmap, so it handles the 158 MB splat
without loading all 62 float columns into memory at once.

Usage: python plyinfo.py "C:\\path\\to\\model.ply"
"""
import re
import sys

import numpy as np


def main(path):
    with open(path, "rb") as f:
        head = f.read(16384)
    marker = b"end_header\n"
    end = head.index(marker) + len(marker)
    header = head[:end].decode("ascii", "replace")
    nverts = int(re.search(r"element vertex (\d+)", header).group(1))
    props = re.findall(r"property\s+\w+\s+(\w+)", header)
    nprops = len(props)
    print("file:", path)
    print("vertices:", nverts, "| props/vertex:", nprops)
    print("first props:", props[:6], "| last props:", props[-7:])

    data = np.memmap(path, dtype="<f4", mode="r", offset=end)
    need = nverts * nprops
    data = np.asarray(data[:need]).reshape(nverts, nprops)

    xyz = data[:, :3].astype(np.float64)
    cen = xyz.mean(axis=0)
    mn, mx = xyz.min(axis=0), xyz.max(axis=0)
    radius = np.sqrt((xyz ** 2).sum(axis=1))
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)

    print("centroid:           ", np.round(cen, 4).tolist())
    print("bbox min:           ", np.round(mn, 4).tolist())
    print("bbox max:           ", np.round(mx, 4).tolist())
    print("bbox extent:        ", np.round(mx - mn, 4).tolist())
    print("robust extent 1-99%:", np.round(hi - lo, 4).tolist())
    print("centroid |dist| from origin:", round(float(np.linalg.norm(cen)), 3))
    print("max |pos| from origin:      ", round(float(radius.max()), 3))
    print("99pct |pos| from origin:    ", round(float(np.percentile(radius, 99)), 3))

    sidx = [i for i, p in enumerate(props) if p.startswith("scale_")]
    if sidx:
        es = np.exp(data[:, sidx].astype(np.float64))  # 3DGS stores log-scale
        print("gaussian size (exp scale) median:", np.round(np.median(es, axis=0), 5).tolist())
        print("gaussian size (exp scale) p95:   ", np.round(np.percentile(es, 95, axis=0), 5).tolist())

    # far-from-origin verdict (NuRec 5.1.0 artifact bug triggers >~300 m)
    if np.linalg.norm(cen) > 50 or radius.max() > 300:
        print("VERDICT: FAR from origin -> recenter before NuRec conversion.")
    else:
        print("VERDICT: near origin; recentering optional.")


if __name__ == "__main__":
    main(sys.argv[1])
