#!/usr/bin/env python3
"""Extract Aria RGB frames as the same pinhole view used by semidense_geo_baseline.

The output is a single rectified trajectory folder:

  <out>/frames_raw/              cw90-rotated native Aria RGB frames
  <out>/frames_rect/             cw90-rotated 720x720 pinhole RGB frames
    frame_000000.jpg, ...
    timestamps.csv
    poses.csv                    T_world_camera for the rectified camera
    camera_pinhole.json
    frame_index.csv              compact index for manual pre/post selection
    selection_contact_sheet.jpg  coarse visual index

Frame numbers are the original VRS RGB stream indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from projectaria_tools.core import calibration, data_provider
from scipy.spatial.transform import Rotation, Slerp


def provider_num_data(provider, stream_id):
    if hasattr(provider, "get_num_data"):
        return provider.get_num_data(stream_id)
    return provider.getNumData(stream_id)


def load_trajectory(mps_dir):
    path = Path(mps_dir) / "slam" / "closed_loop_trajectory.csv"
    rows = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No poses found in {path}")
    ts = np.asarray([int(r["tracking_timestamp_us"]) for r in rows], dtype=np.int64)
    t = np.asarray(
        [[float(r["tx_world_device"]), float(r["ty_world_device"]), float(r["tz_world_device"])] for r in rows],
        dtype=np.float64,
    )
    q = np.asarray(
        [[float(r["qx_world_device"]), float(r["qy_world_device"]), float(r["qz_world_device"]), float(r["qw_world_device"])] for r in rows],
        dtype=np.float64,
    )
    order = np.argsort(ts)
    return ts[order], t[order], q[order]


def interp_pose(ts_traj, t_traj, q_traj, ts_query_us):
    tmin, tmax = ts_traj[0], ts_traj[-1]
    valid = (ts_query_us >= tmin) & (ts_query_us <= tmax)
    out_t = np.full((ts_query_us.size, 3), np.nan, dtype=np.float64)
    if not valid.any():
        return out_t, None, valid

    qts = ts_query_us[valid]
    idx = np.searchsorted(ts_traj, qts).clip(1, len(ts_traj) - 1)
    i0 = idx - 1
    denom = np.maximum(ts_traj[idx] - ts_traj[i0], 1)
    alpha = (qts - ts_traj[i0]) / denom
    out_t[valid] = t_traj[i0] + alpha[:, None] * (t_traj[idx] - t_traj[i0])

    slerp = Slerp(ts_traj, Rotation.from_quat(q_traj))
    rots = slerp(np.clip(qts, tmin, tmax))
    return out_t, (rots, valid), valid


def T_world_camera(ts_ns, ts_us_traj, t_traj, q_traj, T_device_camera):
    ts_us = np.asarray([ts_ns // 1000], dtype=np.int64)
    t_wd, rot_pack, valid = interp_pose(ts_us_traj, t_traj, q_traj, ts_us)
    if rot_pack is None or not valid[0]:
        return None
    R_wd = rot_pack[0].as_matrix()[0]
    T_wd = np.eye(4, dtype=np.float64)
    T_wd[:3, :3] = R_wd
    T_wd[:3, 3] = t_wd[0]
    return T_wd @ T_device_camera


def save_camera_json(calib, path, model_label):
    fx, fy = calib.get_focal_lengths()
    cx, cy = calib.get_principal_point()
    width, height = calib.get_image_size()
    payload = {
        "model": model_label,
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "T_device_camera": calib.get_transform_device_camera().to_matrix().tolist(),
    }
    if model_label.startswith("fisheye"):
        payload["projection_params"] = calib.get_projection_params().tolist()
    path.write_text(json.dumps(payload, indent=2) + "\n")


def image_record_timestamp_ns(record):
    if hasattr(record, "capture_timestamp_ns"):
        return int(record.capture_timestamp_ns)
    if hasattr(record, "captureTimestampNs"):
        return int(record.captureTimestampNs)
    raise AttributeError("Could not read capture timestamp from Aria image record")


def make_contact_sheet(rect_dir, frames, out_path, thumb_size=144, cols=6):
    if not frames:
        return
    rows = int(np.ceil(len(frames) / cols))
    label_h = 20
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for pos, frame_idx in enumerate(frames):
        path = rect_dir / f"frame_{frame_idx:06d}.jpg"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_size, thumb_size))
        x = (pos % cols) * thumb_size + (thumb_size - img.width) // 2
        y = (pos // cols) * (thumb_size + label_h)
        sheet.paste(img, (x, y))
        draw.text(((pos % cols) * thumb_size + 4, y + thumb_size), f"{frame_idx:06d}", fill=(235, 235, 235), font=font)
    sheet.save(out_path, quality=92)


def extract(args):
    out_root = args.out_root
    raw_dir = args.raw_dir or out_root / "frames_raw"
    rect_dir = args.rect_dir or out_root / "frames_rect"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rect_dir.mkdir(parents=True, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(str(args.vrs))
    rgb_id = provider.get_stream_id_from_label(args.camera_label)
    rgb_calib_orig = provider.get_device_calibration().get_camera_calib(args.camera_label)
    T_device_camera_orig = rgb_calib_orig.get_transform_device_camera()

    rgb_calib_rot = calibration.rotate_camera_calib_cw90deg(rgb_calib_orig)
    linear = calibration.get_linear_camera_calibration(
        args.rect_size,
        args.rect_size,
        args.rect_focal,
        f"{args.camera_label}-linear",
        T_device_camera_orig,
    )
    linear_rot_calib = calibration.rotate_camera_calib_cw90deg(linear)
    T_device_camera_rect = linear_rot_calib.get_transform_device_camera().to_matrix()

    save_camera_json(rgb_calib_rot, raw_dir / "camera_fisheye.json", "fisheye624_cw90")
    save_camera_json(linear_rot_calib, rect_dir / "camera_pinhole.json", "pinhole_cw90")

    ts_us_traj, t_traj, q_traj = load_trajectory(args.mps_dir)
    n_total = provider_num_data(provider, rgb_id)
    selected_indices = list(range(0, n_total, args.stride))

    n_with_pose = 0
    contact_indices = []
    with (raw_dir / "timestamps.csv").open("w", newline="") as raw_ts, \
            (rect_dir / "timestamps.csv").open("w", newline="") as rect_ts, \
            (rect_dir / "poses.csv").open("w", newline="") as poses_fp, \
            (rect_dir / "frame_index.csv").open("w", newline="") as index_fp:
        raw_writer = csv.writer(raw_ts)
        rect_writer = csv.writer(rect_ts)
        pose_writer = csv.writer(poses_fp)
        index_writer = csv.writer(index_fp)
        raw_writer.writerow(["frame_idx", "vrs_idx", "timestamp_ns"])
        rect_writer.writerow(["frame_idx", "vrs_idx", "timestamp_ns"])
        pose_writer.writerow(["frame_idx", "timestamp_ns", *[f"T{r}{c}" for r in range(4) for c in range(4)]])
        index_writer.writerow(["frame_idx", "image_name", "timestamp_ns", "has_pose"])

        for out_i, idx in enumerate(selected_indices):
            image_data, record = provider.get_image_data_by_index(rgb_id, idx)
            ts_ns = image_record_timestamp_ns(record)
            img_orig = image_data.to_numpy_array()

            img_raw_rot = np.rot90(img_orig, k=-1)
            img_rect_unrot = calibration.distort_by_calibration(img_orig, linear, rgb_calib_orig)
            img_rect = np.rot90(img_rect_unrot, k=-1)

            fname = f"frame_{idx:06d}.jpg"
            Image.fromarray(img_raw_rot).save(raw_dir / fname, quality=args.jpg_quality)
            Image.fromarray(img_rect).save(rect_dir / fname, quality=args.jpg_quality)

            raw_writer.writerow([idx, idx, ts_ns])
            rect_writer.writerow([idx, idx, ts_ns])

            T_wc = T_world_camera(ts_ns, ts_us_traj, t_traj, q_traj, T_device_camera_rect)
            has_pose = T_wc is not None
            if has_pose:
                n_with_pose += 1
                pose_writer.writerow([idx, ts_ns, *[f"{x:.8f}" for x in T_wc.flatten()]])
            index_writer.writerow([idx, fname, ts_ns, int(has_pose)])

            if args.preview_stride > 0 and out_i % args.preview_stride == 0:
                contact_indices.append(idx)
            if (out_i + 1) % 100 == 0 or out_i == len(selected_indices) - 1:
                print(f"frame {out_i + 1}/{len(selected_indices)} from VRS {idx} (poses: {n_with_pose})", flush=True)

    if args.preview_stride > 0:
        make_contact_sheet(rect_dir, contact_indices, rect_dir / "selection_contact_sheet.jpg")

    manifest = {
        "vrs": str(args.vrs.resolve()),
        "mps_dir": str(args.mps_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "rect_dir": str(rect_dir.resolve()),
        "camera_label": args.camera_label,
        "rect_size": args.rect_size,
        "rect_focal": args.rect_focal,
        "stride": args.stride,
        "n_source_frames": n_total,
        "n_extracted_frames": len(selected_indices),
        "n_with_pose": n_with_pose,
    }
    (out_root / "extract_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Aria VRS RGB as baseline-compatible pinhole views")
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--mps_dir", type=Path, required=True, help="MPS folder containing slam/closed_loop_trajectory.csv")
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--rect_dir", type=Path, default=None)
    parser.add_argument("--camera_label", default="camera-rgb")
    parser.add_argument("--stride", type=int, default=1, help="Extract every Nth VRS RGB frame")
    parser.add_argument("--rect_size", type=int, default=720)
    parser.add_argument("--rect_focal", type=float, default=380.0)
    parser.add_argument("--jpg_quality", type=int, default=90)
    parser.add_argument("--preview_stride", type=int, default=30, help="Every Nth extracted frame goes into the contact sheet; 0 disables it")
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    return args


def main():
    extract(parse_args())


if __name__ == "__main__":
    main()
