#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import read_points3D_binary, rotmat2qvec


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def frame_stem(token, numeric_format):
    token = token.strip()
    if not token or token.startswith("#"):
        return None
    token = token.replace(",", " ").split()[0].strip()
    if token.lower() in {"frame_idx", "frame", "image", "image_name", "name"}:
        return None
    if token.isdigit():
        return numeric_format.format(int(token))
    return Path(token).stem


def read_frame_list(path, numeric_format):
    names = []
    seen = set()
    with open(path, "r") as fp:
        for line in fp:
            name = frame_stem(line, numeric_format)
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    if not names:
        raise ValueError(f"No frames found in {path}")
    return names


def replace_path(path, overwrite):
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def find_image(images_dir, stem):
    candidates = [images_dir / f"{stem}{ext}" for ext in IMAGE_EXTS]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(p for p in images_dir.glob(f"{stem}.*") if p.suffix in IMAGE_EXTS)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous image candidates for {stem}: {matches[:5]}")
    raise FileNotFoundError(f"Could not find image for {stem} in {images_dir}")


def load_poses(poses_csv):
    poses = {}
    with open(poses_csv, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            stem = f"frame_{int(row['frame_idx']):06d}"
            vals = [float(row[f"T{r}{c}"]) for r in range(4) for c in range(4)]
            poses[stem] = np.asarray(vals, dtype=np.float64).reshape(4, 4)
    return poses


def write_cameras(camera_json, out_path):
    camera = json.loads(Path(camera_json).read_text())
    out_path.write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {camera['width']} {camera['height']} "
        f"{camera['fx']} {camera['fy']} {camera['cx']} {camera['cy']}\n"
    )
    return camera


def write_images(frame_names, poses, images_dir, dataset_dir, out_path, copy_images):
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    missing_pose = [stem for stem in frame_names if stem not in poses]
    if missing_pose:
        raise RuntimeError(f"Missing poses for {len(missing_pose)} frames, first: {missing_pose[:5]}")
    resolved_images = [(stem, find_image(images_dir, stem)) for stem in frame_names]
    with open(out_path, "w") as out:
        out.write("# Image list with two lines of data per image:\n")
        out.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        out.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for stem, src in resolved_images:
            dst_name = src.name
            dst = image_dir / dst_name
            if not dst.exists() and not dst.is_symlink():
                if copy_images:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())
            t_cw = np.linalg.inv(poses[stem])
            q = rotmat2qvec(t_cw[:3, :3])
            t = t_cw[:3, 3]
            count += 1
            out.write(
                f"{count} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} 1 {dst_name}\n\n"
            )
    return count


def write_points_from_semidense(path, out_path, point_stride, max_points, inv_max, dist_max):
    count = 0
    kept = 0
    with gzip.open(path, "rt", newline="") as fp, open(out_path, "w") as out:
        reader = csv.DictReader(fp)
        required = {"px_world", "py_world", "pz_world"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path} is missing required semidense columns: {sorted(required)}")
        out.write("# 3D point list with one line of data per point:\n")
        out.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for row in reader:
            if "inv_dist_std" in row and row["inv_dist_std"] and float(row["inv_dist_std"]) > inv_max:
                continue
            if "dist_std" in row and row["dist_std"] and float(row["dist_std"]) > dist_max:
                continue
            kept += 1
            if (kept - 1) % max(1, point_stride) != 0:
                continue
            count += 1
            out.write(f"{count} {row['px_world']} {row['py_world']} {row['pz_world']} 128 128 128 1.0\n")
            if count >= max_points:
                break
    return count


def write_points_from_ply(path, out_path, point_stride, max_points):
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    names = {prop.name for prop in vertices.properties}
    count = 0
    with open(out_path, "w") as out:
        out.write("# 3D point list with one line of data per point:\n")
        out.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for idx in range(0, len(vertices), max(1, point_stride)):
            v = vertices[idx]
            if {"red", "green", "blue"}.issubset(names):
                rgb = (int(v["red"]), int(v["green"]), int(v["blue"]))
            else:
                rgb = (128, 128, 128)
            count += 1
            out.write(f"{count} {float(v['x'])} {float(v['y'])} {float(v['z'])} {rgb[0]} {rgb[1]} {rgb[2]} 1.0\n")
            if count >= max_points:
                break
    return count


def write_points_from_colmap_txt(path, out_path, point_stride, max_points):
    count = 0
    kept = 0
    with open(path) as fp, open(out_path, "w") as out:
        out.write("# 3D point list with one line of data per point:\n")
        out.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise RuntimeError(f"Malformed COLMAP point line in {path}: {stripped}")
            kept += 1
            if (kept - 1) % max(1, point_stride) != 0:
                continue
            count += 1
            out.write(f"{count} {' '.join(parts[1:8])}\n")
            if count >= max_points:
                break
    return count


def write_points_from_colmap_binary(path, out_path, point_stride, max_points):
    xyz, rgb, error = read_points3D_binary(str(path))
    count = 0
    with open(out_path, "w") as out:
        out.write("# 3D point list with one line of data per point:\n")
        out.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for idx in range(0, xyz.shape[0], max(1, point_stride)):
            count += 1
            c = rgb[idx].astype(int)
            out.write(
                f"{count} {xyz[idx, 0]} {xyz[idx, 1]} {xyz[idx, 2]} "
                f"{c[0]} {c[1]} {c[2]} {float(error[idx])}\n"
            )
            if count >= max_points:
                break
    return count


def resolve_points_source(args):
    if args.points:
        return Path(args.points)
    candidates = [
        args.source_root / "semidense_points.csv.gz",
        args.source_root / "slam" / "semidense_points.csv.gz",
        args.source_root / "points3D.txt",
        args.source_root / "sparse" / "0" / "points3D.txt",
        args.source_root / "points3D.ply",
        args.source_root / "sparse" / "0" / "points3D.ply",
        args.source_root / "points3D.bin",
        args.source_root / "sparse" / "0" / "points3D.bin",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Pass --points; no default point source found")


def write_points(points_source, out_path, args):
    suffixes = "".join(points_source.suffixes).lower()
    if suffixes.endswith(".csv.gz"):
        return write_points_from_semidense(
            points_source,
            out_path,
            args.point_stride,
            args.max_points,
            args.inv_dist_std_max,
            args.dist_std_max,
        )
    if points_source.suffix.lower() == ".ply":
        return write_points_from_ply(points_source, out_path, args.point_stride, args.max_points)
    if points_source.suffix.lower() == ".bin":
        return write_points_from_colmap_binary(points_source, out_path, args.point_stride, args.max_points)
    return write_points_from_colmap_txt(points_source, out_path, args.point_stride, args.max_points)


def build_one(label, frame_list, args, poses, camera, points_source):
    dataset_dir = args.out_root / f"{args.task}_{label}"
    replace_path(dataset_dir, args.overwrite)
    sparse_dir = dataset_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    write_cameras(args.camera_json, sparse_dir / "cameras.txt")
    n_images = write_images(frame_list, poses, args.images_dir, dataset_dir, sparse_dir / "images.txt", args.copy_images)
    n_points = write_points(points_source, sparse_dir / "points3D.txt", args)
    return {
        "name": f"{args.task}_{label}",
        "task": args.task,
        "label": label,
        "dataset_dir": str(dataset_dir.resolve()),
        "single_trajectory": True,
        "frame_list": str(args.pre_frames if label == "pre" else args.post_frames),
        "source_root": str(args.source_root.resolve()),
        "points_source": str(points_source.resolve()),
        "camera": {
            "width": camera["width"],
            "height": camera["height"],
            "fx": camera["fx"],
            "fy": camera["fy"],
            "cx": camera["cx"],
            "cy": camera["cy"],
        },
        "num_images_with_pose": n_images,
        "num_init_points": n_points,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build pre/post 2DGS datasets from one Aria trajectory")
    parser.add_argument("--source_root", type=Path, required=True, help="Folder with rectified images, poses.csv, camera_pinhole.json")
    parser.add_argument("--images_dir", type=Path, default=None)
    parser.add_argument("--poses_csv", type=Path, default=None)
    parser.add_argument("--camera_json", type=Path, default=None)
    parser.add_argument("--points", type=Path, default=None)
    parser.add_argument("--pre_frames", type=Path, required=True)
    parser.add_argument("--post_frames", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out_root", type=Path, default=Path("data/corl2026_2dgs_single"))
    parser.add_argument("--point_stride", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--inv_dist_std_max", type=float, default=0.05)
    parser.add_argument("--dist_std_max", type=float, default=0.10)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--numeric_frame_format", default="frame_{:06d}", help="Format used for numeric frame-list entries")
    return parser.parse_args()


def main():
    args = parse_args()
    args.images_dir = args.images_dir or args.source_root
    args.poses_csv = args.poses_csv or args.source_root / "poses.csv"
    args.camera_json = args.camera_json or args.source_root / "camera_pinhole.json"
    if args.point_stride <= 0:
        raise ValueError("--point_stride must be positive")
    if args.max_points <= 0:
        raise ValueError("--max_points must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)

    pre_frames = read_frame_list(args.pre_frames, args.numeric_frame_format)
    post_frames = read_frame_list(args.post_frames, args.numeric_frame_format)
    poses = load_poses(args.poses_csv)
    camera = json.loads(args.camera_json.read_text())
    points_source = resolve_points_source(args)

    entries = [
        build_one("pre", pre_frames, args, poses, camera, points_source),
        build_one("post", post_frames, args, poses, camera, points_source),
    ]
    manifest = {
        "single_trajectory": True,
        "alignment": "identity",
        "alignment_method": "identity_single_trajectory",
        "T_pre_post": np.eye(4).tolist(),
        "datasets": entries,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
