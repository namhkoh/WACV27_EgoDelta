"""Carve a clean empty pocket at each physics-tomato location in the splat .ply.

The two physics fruit replace the middle-gutter splat tomatoes, so when one detaches
the spot must be TRULY empty -- no leftover red blob, no specular-highlight ghost, no
view-dependent (SH) reddish gaussians. A strict-red filter leaves those remnants, which
show through once the fruit drops. So here we remove EVERY non-green gaussian inside a
tight sphere around each fruit centre (kills the blob + highlights + shaded edges), plus
any strongly-red gaussian in a slightly larger halo (catches stray red floaters). Green
vine foliage passing through the pocket is preserved. Writes a new .ply.
"""
import json
import re

import numpy as np

PLY = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_crop.ply"
ORIENT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/orient.json"
TOMATOES = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/tomatoes.json"
OUT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_crop_notom.ply"
C0 = 0.28209479177387814
RAD_POCKET = 0.075   # clean-pocket radius: remove all non-green within this of each fruit
RAD_HALO = 0.12      # larger radius: also remove strongly-red gaussians (highlight/floater spread)

with open(PLY, "rb") as f:
    raw = f.read()
end = raw.index(b"end_header\n") + len(b"end_header\n")
header = raw[:end].decode("ascii", "replace")
nv = int(re.search(r"element vertex (\d+)", header).group(1))
props = re.findall(r"property\s+\w+\s+(\w+)", header)
npp = len(props)
idx = {p: i for i, p in enumerate(props)}
data = np.frombuffer(raw, dtype="<f4", offset=end, count=nv * npp).reshape(nv, npp)

xyz = data[:, :3].astype(np.float64)
fdc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float64)
rgb = np.clip(0.5 + C0 * fdc, 0.0, 1.0)
R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
green = (G > R + 0.04) & (G > B)                                  # clearly green foliage -> keep
red_strong = (R > 0.42) & (R - np.maximum(G, B) > 0.05)          # any reddish -> remove in halo

o = json.load(open(ORIENT))
P = (np.array(o["R"]) @ xyz.T).T + np.array(o["t"])               # leveled world frame

centers = [np.array([f["x"], f["y"], f["z"]]) for f in json.load(open(TOMATOES))]
remove = np.zeros(nv, dtype=bool)
for c in centers:
    d = np.linalg.norm(P - c, axis=1)
    remove |= (d < RAD_POCKET) & (~green)       # carve pocket, keep green vine
    remove |= (d < RAD_HALO) & red_strong       # kill reddish spread/floaters
keep = ~remove
print("fruit centres:", [list(np.round(c, 3)) for c in centers])
print("removing {} gaussians (pocket+halo); keeping {} / {}".format(int(remove.sum()), int(keep.sum()), nv))

kept = data[keep]
new_header = re.sub(r"element vertex \d+", "element vertex {}".format(kept.shape[0]), header)
with open(OUT, "wb") as f:
    f.write(new_header.encode("ascii"))
    f.write(np.ascontiguousarray(kept, dtype="<f4").tobytes())
print("wrote", OUT)
