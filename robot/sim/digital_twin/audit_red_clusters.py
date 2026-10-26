"""Audit what red (tomato) gaussian clusters remain in a splat .ply, in the leveled
world frame, and report each cluster's distance to the T4 (center-gutter) centroid.

Tells us definitively whether leftover red splat sits in the center gutter (should be
gone -> physics fruit replace it) or only on the other gutters (intentionally kept).
"""
import json
import re
import sys

import numpy as np

PLY = sys.argv[1] if len(sys.argv) > 1 else r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_crop_notom.ply"
ORIENT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/orient.json"
C0 = 0.28209479177387814
T4 = np.array([-0.141, -0.283, 0.893])

with open(PLY, "rb") as f:
    raw = f.read()
end = raw.index(b"end_header\n") + len(b"end_header\n")
header = raw[:end].decode("ascii", "replace")
nv = int(re.search(r"element vertex (\d+)", header).group(1))
props = re.findall(r"property\s+\w+\s+(\w+)", header)
idx = {p: i for i, p in enumerate(props)}
data = np.frombuffer(raw, dtype="<f4", offset=end, count=nv * len(props)).reshape(nv, len(props))

xyz = data[:, :3].astype(np.float64)
fdc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float64)
op = 1.0 / (1.0 + np.exp(-data[:, idx["opacity"]].astype(np.float64)))
rgb = np.clip(0.5 + C0 * fdc, 0.0, 1.0)
red = (rgb[:, 0] > 0.45) & ((rgb[:, 0] - np.maximum(rgb[:, 1], rgb[:, 2])) > 0.12) & (op > 0.25)

o = json.load(open(ORIENT))
P = (np.array(o["R"]) @ xyz.T).T + np.array(o["t"])
pts = P[red]
print("total gaussians:", nv, "| red:", int(red.sum()))

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.05, min_samples=20).fit(pts)
labels = db.labels_
clusters = []
for lb in sorted(set(labels)):
    if lb == -1:
        continue
    c = pts[labels == lb]
    cen = c.mean(0)
    d = float(np.linalg.norm(cen - T4))
    clusters.append((lb, cen, len(c), d))
clusters.sort(key=lambda x: x[3])
print("\nred clusters (sorted by distance to T4 center {}):".format(np.round(T4, 3)))
for lb, cen, n, d in clusters:
    tag = "  <-- CENTER GUTTER (T4)" if d < 0.25 else ""
    print("  cl{:>2}  centroid=[{:+.3f}, {:+.3f}, {:+.3f}]  n={:>4}  dist_to_T4={:.3f}m{}".format(
        lb, cen[0], cen[1], cen[2], n, d, tag))
near = pts[np.linalg.norm(pts - T4, axis=1) < 0.25]
print("\nred gaussians within 0.25m of T4 center (leftover center-gutter splat):", len(near))
