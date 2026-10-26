"""Find red tomato clusters in the leveled greenhouse splat frame.

Decodes each gaussian's DC color (RGB = 0.5 + C0*f_dc) and opacity (sigmoid),
keeps reddish + opaque points in the plant region, applies the orient.json
leveling, and clusters them into tomatoes. Reports each cluster's world centroid
+ size so we can place harvestable rigid bodies there.
"""
import json
import re

import numpy as np

PLY = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_crop.ply"
ORIENT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/orient.json"
C0 = 0.28209479177387814


def simple_cluster(pts, eps, min_samples):
    """Grid-based connected-components clustering (no sklearn dependency)."""
    keys = np.floor(pts / eps).astype(np.int64)
    from collections import defaultdict
    cell = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        cell[k].append(i)
    labels = np.full(len(pts), -1, dtype=int)
    cid = 0
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    visited_cells = set()
    for k0 in list(cell.keys()):
        if k0 in visited_cells:
            continue
        stack = [k0]
        comp = []
        while stack:
            k = stack.pop()
            if k in visited_cells:
                continue
            visited_cells.add(k)
            if k not in cell:
                continue
            comp.extend(cell[k])
            for d in neigh:
                nk = (k[0] + d[0], k[1] + d[1], k[2] + d[2])
                if nk in cell and nk not in visited_cells:
                    stack.append(nk)
        if len(comp) >= min_samples:
            for i in comp:
                labels[i] = cid
            cid += 1
    return labels


def main():
    with open(PLY, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    nv = int(re.search(r"element vertex (\d+)", header).group(1))
    props = re.findall(r"property\s+\w+\s+(\w+)", header)
    npp = len(props)
    data = np.frombuffer(raw, dtype="<f4", offset=end, count=nv * npp).reshape(nv, npp)

    xyz = data[:, :3].astype(np.float64)
    idx = {p: i for i, p in enumerate(props)}
    fdc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float64)
    opacity = data[:, idx["opacity"]].astype(np.float64)

    rgb = np.clip(0.5 + C0 * fdc, 0.0, 1.0)
    op = 1.0 / (1.0 + np.exp(-opacity))
    Rr, Gg, Bb = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    red = (Rr > 0.45) & ((Rr - np.maximum(Gg, Bb)) > 0.12) & (op > 0.25)

    o = json.load(open(ORIENT))
    Rm, t = np.array(o["R"]), np.array(o["t"])
    P = (Rm @ xyz.T).T + t

    core = (np.abs(P[:, 0]) < 3.0) & (np.abs(P[:, 1]) < 3.0) & (P[:, 2] > 0.05) & (P[:, 2] < 2.0)
    sel = red & core
    pts = P[sel]
    print("red+opaque points in plant region:", len(pts))
    if len(pts) == 0:
        return

    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=0.04, min_samples=25).fit_predict(pts)
        print("clustering: sklearn DBSCAN")
    except Exception:
        labels = simple_cluster(pts, eps=0.04, min_samples=25)
        print("clustering: grid fallback")

    clusters = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        cl = pts[labels == c]
        clusters.append((len(cl), cl.mean(0), cl.min(0), cl.max(0)))
    clusters.sort(key=lambda x: -x[0])

    print("found", len(clusters), "tomato clusters (sorted by size):")
    for i, (n, cen, mn, mx) in enumerate(clusters):
        size = mx - mn
        print("  T{}: n={:5d}  centroid=[{:.3f}, {:.3f}, {:.3f}]  size=[{:.3f}, {:.3f}, {:.3f}]".format(
            i, n, cen[0], cen[1], cen[2], size[0], size[1], size[2]))


if __name__ == "__main__":
    main()
