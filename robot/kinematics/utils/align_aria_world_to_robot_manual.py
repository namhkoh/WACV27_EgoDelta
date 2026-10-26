"""Manual-correspondence Aria-world -> robot alignment (Kabsch).

Cross-modal auto-registration (FPFH/ICP) of the sparse Aria semidense cloud to the dense
RealSense bed view does not converge reliably. This tool sidesteps that: you click the
SAME rigid physical landmarks (rod tops, frame corners) in BOTH clouds, in the SAME order,
and it solves the exact rigid transform that maps Aria SLAM world -> robot frame. Works
even if the plants moved, since it uses the rigid bed structure.

Writes a 'world_direct' T_robot_from_aria alignment json (no anchor frame).

Run with the `aria` conda env python (needs open3d + a display):
  C:/path/to/user/miniconda3/envs/aria/python.exe robot/kinematics/utils/align_aria_world_to_robot_manual.py

Picking: in each window, SHIFT+LEFT-CLICK each landmark (same physical points, same order),
then press Q to close. Pick at least --min-correspondences (default 4), non-collinear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation


def load_semidense_world(path: Path, dist_std_max: float, max_points: int) -> np.ndarray:
    df = pd.read_csv(path, compression="infer")
    if "dist_std" in df.columns:
        df = df[df["dist_std"] <= dist_std_max]
    xyz = df[["px_world", "py_world", "pz_world"]].to_numpy(dtype=float)
    if len(xyz) > max_points:
        idx = np.random.RandomState(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def load_realsense_robot(npz_path: Path, extrinsic_json: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    pts = np.asarray(data["points_realsense_m"], dtype=float).reshape(-1, 3)
    colors = np.asarray(data["colors_rgb"], dtype=float).reshape(-1, 3) / 255.0 if "colors_rgb" in data.files else None
    T = np.asarray(json.loads(Path(extrinsic_json).read_text())["T_robot_from_realsense"], dtype=float).reshape(4, 4)
    robot = (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    return robot, colors


def hand_region_center_world(hand_csv: Path, slam_csv: Path, side: str) -> np.ndarray | None:
    try:
        hand = pd.read_csv(hand_csv)
        conf = hand[f"{side}_tracking_confidence"].to_numpy(float)
        m = conf >= 0.5
        hts = hand["tracking_timestamp_us"].to_numpy(float)[m]
        wrist = hand[[f"tx_{side}_device_wrist", f"ty_{side}_device_wrist", f"tz_{side}_device_wrist"]].to_numpy(float)[m]
        slam = pd.read_csv(slam_csv)
        order = np.argsort(slam["tracking_timestamp_us"].to_numpy(float))
        sts = slam["tracking_timestamp_us"].to_numpy(float)[order]
        st = slam[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy(float)[order]
        sq = slam[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy(float)[order]
        centers = []
        for ts_i, w in zip(hts, wrist):
            j = int(np.clip(np.searchsorted(sts, ts_i), 0, len(sts) - 1))
            centers.append(st[j] + Rotation.from_quat(sq[j]).as_matrix() @ w)
        return np.median(np.asarray(centers), axis=0)
    except Exception as exc:  # noqa: BLE001
        print(f"(could not compute hand-region crop center: {exc}; showing full cloud)")
        return None


def color_by_height(xyz: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    t = (z - z.min()) / (np.ptp(z) + 1e-9)
    return np.c_[t, 0.4 + 0.5 * (1 - t), 1 - t]


def pick_points(xyz: np.ndarray, colors: np.ndarray, title: str) -> list[int]:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=900)
    vis.add_geometry(pc)
    vis.run()  # blocks; user shift+clicks then presses Q
    vis.destroy_window()
    return vis.get_picked_points()


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform mapping source P -> target Q (no scale). Returns R, t, rms."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cQ - R @ cP
    resid = (R @ P.T).T + t - Q
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return R, t, rms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "C:/path/to/user/aria/handtrack/mps_0d058720-9302-4563-8764-3a30da628444_vrs"
    ap.add_argument("--semidense", type=Path, default=Path(f"{base}/slam/semidense_points.csv.gz"))
    ap.add_argument("--slam", type=Path, default=Path(f"{base}/slam/closed_loop_trajectory.csv"))
    ap.add_argument("--hand-tracking", type=Path, default=Path(f"{base}/hand_tracking/hand_tracking_results.csv"))
    ap.add_argument("--realsense-npz", type=Path, default=Path("data/policy_training/a270c038/current_realsense/realsense_scene_head_current.npz"))
    ap.add_argument("--realsense-extrinsic", type=Path, default=Path("data/policy_training/a270c038/current_realsense/T_robot_from_realsense_head2_current.json"))
    ap.add_argument("--hand-side", default="right")
    ap.add_argument("--dist-std-max", type=float, default=0.05)
    ap.add_argument("--max-points", type=int, default=300000)
    ap.add_argument("--crop-radius-m", type=float, default=1.5, help="Crop Aria cloud around the hand region for easier picking; 0 disables.")
    ap.add_argument("--min-correspondences", type=int, default=4)
    ap.add_argument("--output-json", type=Path, default=Path("robot/calibration/session/0d058720/T_robot_from_aria_manual_world_direct.json"))
    args = ap.parse_args()

    aria = load_semidense_world(args.semidense, args.dist_std_max, args.max_points)
    if args.crop_radius_m > 0:
        c = hand_region_center_world(args.hand_tracking, args.slam, args.hand_side)
        if c is not None:
            d = np.linalg.norm(aria - c, axis=1)
            aria = aria[d <= args.crop_radius_m]
    rs_robot, rs_colors = load_realsense_robot(args.realsense_npz, args.realsense_extrinsic)
    if rs_colors is None:
        rs_colors = color_by_height(rs_robot)

    print("\n" + "=" * 70)
    print("STEP 1/2 — ARIA cloud. SHIFT+LEFT-CLICK each rigid landmark (rod tops,")
    print("frame corners) in a fixed order you'll remember, then press Q.")
    print("=" * 70)
    aria_idx = pick_points(aria, color_by_height(aria), "ARIA cloud — pick landmarks (shift+click), then Q")
    print(f"  picked {len(aria_idx)} Aria points")

    print("\n" + "=" * 70)
    print("STEP 2/2 — REALSENSE cloud (robot frame). Click the SAME landmarks in the")
    print("SAME order, then press Q.")
    print("=" * 70)
    rs_idx = pick_points(rs_robot, rs_colors, "REALSENSE cloud — pick SAME landmarks SAME order, then Q")
    print(f"  picked {len(rs_idx)} RealSense points")

    if len(aria_idx) != len(rs_idx):
        print(f"\nERROR: count mismatch (Aria {len(aria_idx)} vs RealSense {len(rs_idx)}). Re-run and pick the same number, same order.")
        return 2
    if len(aria_idx) < args.min_correspondences:
        print(f"\nERROR: need >= {args.min_correspondences} correspondences, got {len(aria_idx)}. Re-run.")
        return 2

    P = aria[np.asarray(aria_idx, dtype=int)]       # source: Aria world
    Q = rs_robot[np.asarray(rs_idx, dtype=int)]      # target: robot frame
    R, t, rms = kabsch(P, Q)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "version": "0.1", "method": "manual_correspondence_kabsch", "mode": "world_direct",
        "dataset": "0d058720", "T_robot_from_aria": T.tolist(),
        "n_correspondences": int(len(P)), "fit_rms_m": rms,
        "aria_points_world_m": P.tolist(), "robot_points_m": Q.tolist(),
        "inputs": {"semidense": str(args.semidense), "realsense_npz": str(args.realsense_npz),
                   "realsense_extrinsic": str(args.realsense_extrinsic)},
        "notes": ["world_direct: T_robot_from_aria maps Aria SLAM world straight to robot; no anchor frame.",
                  "fit_rms_m is the correspondence residual; want < ~0.02 m for a good pick."],
    }
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output_json}")
    print(f"T_robot_from_aria=\n{np.round(T,3)}")
    print(f"\nfit RMS = {rms*1000:.1f} mm over {len(P)} correspondences  "
          f"=> {'GOOD' if rms < 0.02 else 'OK' if rms < 0.04 else 'HIGH (repick more carefully)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
