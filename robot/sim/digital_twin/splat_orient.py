"""Estimate a leveling transform for a 3DGS .ply: rotate so the dominant floor
plane is horizontal (normal -> +Z) and sits at Z=0.

RANSAC-fits the largest plane in the gaussian centers (assumed to be the floor),
orients its normal up (toward the scene bulk), and writes a 4x4 transform (column
-vector convention: p' = R @ p + t) to JSON for the in-sim apply step.

Usage: python splat_orient.py crop.ply orient.json
"""
import json
import re
import sys

import numpy as np


def load_xyz(ply):
    with open(ply, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    nv = int(re.search(r"element vertex (\d+)", header).group(1))
    npp = len(re.findall(r"property\s+\w+\s+(\w+)", header))
    data = np.frombuffer(raw, dtype="<f4", offset=end, count=nv * npp).reshape(nv, npp)
    return data[:, :3].astype(np.float64)


def ransac_plane(pts, thresh, iters, rng):
    best = None
    n_pts = len(pts)
    for _ in range(iters):
        i = rng.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[i]
        nrm = np.cross(p2 - p1, p3 - p1)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -nrm.dot(p1)
        cnt = int((np.abs(pts.dot(nrm) + d) < thresh).sum())
        if best is None or cnt > best[0]:
            best = (cnt, nrm, d)
    _, nrm, d = best
    inl = np.abs(pts.dot(nrm) + d) < thresh
    P = pts[inl]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    nrm = vt[-1]
    return nrm, c, inl


def main(ply, out, up_axis=None):
    xyz = load_xyz(ply)
    rng = np.random.default_rng(0)
    sub = xyz[rng.choice(len(xyz), min(120000, len(xyz)), replace=False)]

    lo99 = np.percentile(sub, 1, 0)
    hi99 = np.percentile(sub, 99, 0)
    ext = hi99 - lo99
    # The vertical axis is the one with the smallest extent (scene wider than tall),
    # unless explicitly overridden.
    if up_axis is None:
        up_axis = int(np.argmin(ext))
    print("robust extent", np.round(ext, 2).tolist(), "| inferred up-axis =", "xyz"[up_axis])

    # This capture is Y-DOWN (INRIA/COLMAP): the FLOOR is the MAX-coord slab along
    # the up-axis, and "up" points toward the scene bulk (decreasing Y).
    coord = sub[:, up_axis]
    slab_lo = hi99[up_axis] - 0.15 * ext[up_axis]
    floor_pts = sub[coord > slab_lo]
    thresh = 0.01 * float(np.linalg.norm(ext))
    nrm, c, inl = ransac_plane(floor_pts, thresh, 1500, rng)
    print("floor slab {} pts; plane inliers {} ({:.1f}%) normal {}".format(
        len(floor_pts), int(inl.sum()), 100 * inl.sum() / len(floor_pts), np.round(nrm, 3).tolist()))

    # Up = the floor normal pointing toward the scene bulk
    if nrm.dot(sub.mean(0) - c) < 0:
        nrm = -nrm

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(nrm, z)
    s = np.linalg.norm(v)
    ct = float(nrm.dot(z))
    if s < 1e-9:
        R = np.eye(3) if ct > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - ct) / (s * s))

    # Put the floor centroid at the world origin (floor at Z=0, centered)
    t = -(R @ c)

    # The capture frame may be Y-down (INRIA/COLMAP): verify the scene bulk ends
    # up ABOVE the floor; if it's below, the scene is upside down -> flip 180 deg
    # about X (keeps the floor horizontal at Z=0).
    pts_w = (R @ sub.T).T + t
    if np.median(pts_w[:, 2]) < 0:
        R = np.diag([1.0, -1.0, -1.0]) @ R
        t = -(R @ c)
        print("scene was below floor -> flipped 180 about X (was upside down)")

    print("R", np.round(R, 3).tolist())
    print("t", np.round(t, 3).tolist())

    json.dump(
        {"R": R.tolist(), "t": t.tolist(), "normal_up": nrm.tolist(), "inlier_frac": float(inl.sum() / len(sub))},
        open(out, "w"),
    )
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
