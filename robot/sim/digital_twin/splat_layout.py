"""Find the plant-wall location in the leveled greenhouse frame and compute a robot
placement that matches the real setup: 21 cm in front of the wall, facing it.

Leveled frame (orient.json applied): Z up, floor ~ Z=0. The plant wall (trellis +
vines) is the dense structure ABOVE the floor; the robot stands on the open floor
in front of it. Writes placement (x, y, yaw_deg) to assets/robot_place.json.
"""
import json
import re

import numpy as np

PLY = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/greenhouse_crop.ply"
ORIENT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/orient.json"
OUT = r"C:/path/to/projects/EgoDelta/robot/sim/digital_twin/assets/robot_place.json"
GUTTER_DIST_M = 0.20  # robot base distance IN FRONT OF the bed front edge (gutter)


def main():
    with open(PLY, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    nv = int(re.search(r"element vertex (\d+)", header).group(1))
    npp = len(re.findall(r"property\s+\w+\s+(\w+)", header))
    data = np.frombuffer(raw, dtype="<f4", offset=end, count=nv * npp).reshape(nv, npp)
    xyz = data[:, :3].astype(np.float64)

    o = json.load(open(ORIENT))
    R, t = np.array(o["R"]), np.array(o["t"])
    P = (R @ xyz.T).T + t  # leveled

    # dense core only
    m = (np.abs(P[:, 0]) < 3.0) & (np.abs(P[:, 1]) < 3.0) & (P[:, 2] > -0.2) & (P[:, 2] < 2.5)
    P = P[m]
    print("core points:", len(P), "| z range", round(P[:, 2].min(), 2), round(P[:, 2].max(), 2))

    # plant wall = points well above the floor
    wall = P[(P[:, 2] > 0.5) & (P[:, 2] < 2.0)]
    xy = wall[:, :2]
    c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    wall_dir = vt[0]
    wall_normal = vt[1]
    print("wall centroid xy:", np.round(c, 3).tolist())
    print("wall dir:", np.round(wall_dir, 3).tolist(), "| normal:", np.round(wall_normal, 3).tolist())

    # the open-floor side is the side of the wall line with fewer above-floor points
    floor = P[(P[:, 2] > -0.1) & (P[:, 2] < 0.2)][:, :2]
    side_wall = np.sign((xy - c) @ wall_normal)
    side_floor = np.sign((floor - c) @ wall_normal)
    # open side = where floor points dominate over wall points
    open_pos = (side_floor > 0).sum() / max(1, len(side_floor)) - (side_wall > 0).sum() / max(1, len(side_wall))
    open_normal = wall_normal if open_pos > 0 else -wall_normal
    print("open-floor side normal:", np.round(open_normal, 3).tolist())

    # gutter = the bed's FRONT edge (the wall points furthest toward the open floor)
    d = (xy - c) @ open_normal
    front_off = float(np.percentile(d, 92))  # robust front edge
    gutter = c + open_normal * front_off
    print("bed front-edge offset:", round(front_off, 3), "| gutter xy:", np.round(gutter, 3).tolist())
    # robot stands GUTTER_DIST in front of the gutter, centered along the wall
    robot_xy = gutter + open_normal * GUTTER_DIST_M
    # facing the wall = -open_normal
    yaw = float(np.degrees(np.arctan2(-open_normal[1], -open_normal[0])))
    print("robot xy:", np.round(robot_xy, 3).tolist(), "| yaw_deg:", round(yaw, 1))

    json.dump({"x": float(robot_xy[0]), "y": float(robot_xy[1]), "z": 0.0, "yaw_deg": yaw,
               "wall_centroid": c.tolist(), "open_normal": open_normal.tolist()}, open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
