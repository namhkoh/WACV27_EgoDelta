"""Split the middle-gutter truss (T4) into two individual fruit (tomato0, tomato1).

Extracts the red gaussian points near the T4 centroid, KMeans-splits them into 2,
and writes each fruit's world centroid + estimated radius to assets/tomatoes.json.
"""
import json
import re

import numpy as np

PLY = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_crop.ply"
ORIENT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/orient.json"
OUT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/tomatoes.json"
C0 = 0.28209479177387814
T4 = np.array([-0.141, -0.283, 0.893])


def main():
    with open(PLY, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    nv = int(re.search(r"element vertex (\d+)", header).group(1))
    props = re.findall(r"property\s+\w+\s+(\w+)", header)
    data = np.frombuffer(raw, dtype="<f4", offset=end, count=nv * len(props)).reshape(nv, len(props))
    idx = {p: i for i, p in enumerate(props)}

    xyz = data[:, :3].astype(np.float64)
    fdc = data[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]].astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-data[:, idx["opacity"]].astype(np.float64)))
    rgb = np.clip(0.5 + C0 * fdc, 0.0, 1.0)
    red = (rgb[:, 0] > 0.45) & ((rgb[:, 0] - np.maximum(rgb[:, 1], rgb[:, 2])) > 0.12) & (op > 0.25)

    o = json.load(open(ORIENT))
    P = (np.array(o["R"]) @ xyz.T).T + np.array(o["t"])
    pts = P[red]
    t4pts = pts[np.linalg.norm(pts - T4, axis=1) < 0.13]
    print("T4 region red points:", len(t4pts))

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(t4pts)

    fruit = []
    for i in range(2):
        c = t4pts[km.labels_ == i]
        cen = c.mean(0)
        r = float(np.percentile(np.linalg.norm(c - cen, axis=1), 85))
        fruit.append({"name": "tomato%d" % i, "x": float(cen[0]), "y": float(cen[1]), "z": float(cen[2]),
                      "radius": round(min(max(r, 0.025), 0.05), 4), "n": int(len(c))})
    # order tomato0 = left (smaller x), tomato1 = right
    fruit.sort(key=lambda f: f["x"])
    fruit[0]["name"], fruit[1]["name"] = "tomato0", "tomato1"
    for f in fruit:
        print("  {}: centroid=[{:.3f}, {:.3f}, {:.3f}] radius={} n={}".format(
            f["name"], f["x"], f["y"], f["z"], f["radius"], f["n"]))
    sep = np.linalg.norm(np.array([fruit[0]["x"], fruit[0]["y"], fruit[0]["z"]]) -
                         np.array([fruit[1]["x"], fruit[1]["y"], fruit[1]["z"]]))
    print("  separation:", round(float(sep), 3), "m")
    json.dump(fruit, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
