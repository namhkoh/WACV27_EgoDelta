#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    0: (255, 170, 30),
    1: (80, 170, 255),
    2: (255, 80, 180),
    3: (80, 220, 100),
    4: (210, 210, 80),
    5: (255, 120, 80),
    6: (160, 120, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project selected Gaussian components and interaction targets onto COLMAP images."
    )
    parser.add_argument("--dataset_dir", required=True, help="COLMAP dataset directory with images/ and sparse/0.")
    parser.add_argument("--presence_npz", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_key", default="leaf_instance_candidate")
    parser.add_argument("--cluster_key", default="leaf_instance_id")
    parser.add_argument("--selected_clusters", default="all")
    parser.add_argument("--images", default="images")
    parser.add_argument("--sparse", default="sparse/0")
    parser.add_argument("--frame_stride", default=6, type=int)
    parser.add_argument("--max_frames", default=18, type=int)
    parser.add_argument("--max_points_per_cluster", default=1800, type=int)
    parser.add_argument("--point_radius", default=2, type=int)
    parser.add_argument("--point_color_mode", choices=("cluster", "yellow"), default="yellow")
    parser.add_argument("--target_radius", default=8, type=int)
    parser.add_argument("--hide_target_labels", action="store_true")
    parser.add_argument("--crop_pad", default=100, type=int)
    parser.add_argument("--min_crop", default=360, type=int)
    parser.add_argument("--max_crop", default=980, type=int)
    parser.add_argument("--zoom_size", default=720, type=int)
    return parser.parse_args()


def qvec2rotmat(qvec):
    q0, q1, q2, q3 = [float(v) for v in qvec]
    return np.asarray(
        [
            [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q3 * q1 + 2 * q0 * q2],
            [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
            [2 * q3 * q1 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2],
        ],
        dtype=np.float64,
    )


def load_camera(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = np.asarray([float(v) for v in parts[4:]], dtype=np.float64)
        if model != "PINHOLE":
            raise RuntimeError(f"Only PINHOLE cameras are supported, got {model}")
        fx, fy, cx, cy = params
        return {
            "camera_id": camera_id,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
    raise RuntimeError(f"No camera found in {path}")


def load_images(path):
    images = []
    lines = Path(path).read_text().splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.asarray([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        images.append(
            {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "R": qvec2rotmat(qvec),
                "camera_id": camera_id,
                "name": name,
            }
        )
        if idx < len(lines):
            idx += 1
    images.sort(key=lambda item: item["image_id"])
    return images


def parse_cluster_ids(spec, available):
    spec = str(spec).strip().lower()
    if spec in ("all", "-1", ""):
        return sorted(int(v) for v in available if int(v) >= 0)
    requested = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    missing = sorted(set(requested) - {int(v) for v in available})
    if missing:
        raise RuntimeError(f"Requested clusters {missing} are not present")
    return requested


def sample_points(points, max_points, seed):
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(points.shape[0], max_points, replace=False)]


def load_targets(path, selected_clusters):
    data = json.loads(Path(path).read_text())
    selected = set(selected_clusters)
    targets = []
    for item in data.get("clusters", []):
        cluster_id = int(item["cluster_id"])
        target = item.get("target_xyz")
        if cluster_id not in selected or target is None:
            continue
        targets.append(
            {
                "cluster_id": cluster_id,
                "source_cluster_id": int(item.get("source_cluster_id", -1)),
                "point": np.asarray(target, dtype=np.float64),
            }
        )
    return targets


def project_points(points_world, image, camera):
    points_cam = points_world.astype(np.float64) @ image["R"].T + image["tvec"][None, :]
    z = points_cam[:, 2]
    z_safe = np.maximum(z, 1e-8)
    u = camera["fx"] * (points_cam[:, 0] / z_safe) + camera["cx"]
    v = camera["fy"] * (points_cam[:, 1] / z_safe) + camera["cy"]
    valid = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u <= camera["width"] - 1.0)
        & (v >= 0.0)
        & (v <= camera["height"] - 1.0)
    )
    return u, v, valid


def crop_box(us, vs, valid, width, height, args):
    if not valid.any():
        return (0, 0, width, height)
    x0 = float(us[valid].min())
    x1 = float(us[valid].max())
    y0 = float(vs[valid].min())
    y1 = float(vs[valid].max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) + 2.0 * args.crop_pad
    side = min(max(side, args.min_crop), args.max_crop, max(width, height))
    left = int(round(max(0.0, min(cx - side * 0.5, width - side))))
    top = int(round(max(0.0, min(cy - side * 0.5, height - side))))
    return (left, top, int(round(left + side)), int(round(top + side)))


def draw_point(draw, x, y, color, radius):
    if radius <= 1:
        draw.point((x, y), fill=color)
        return
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def make_contact_sheet(paths, out_path, cols=2, thumb_w=420):
    if not paths:
        return
    font = ImageFont.load_default()
    label_h = 18
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        thumb_h = int(round(thumb_w * img.height / img.width))
        thumbs.append((path, img.resize((thumb_w, thumb_h))))
    thumb_h = max(img.height for _, img in thumbs)
    rows = int(math.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (path, img) in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + thumb_h), path.stem[:52], fill=(235, 235, 235), font=font)
    canvas.save(out_path, quality=92)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    points_dir = out_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    camera = load_camera(dataset_dir / args.sparse / "cameras.txt")
    images = load_images(dataset_dir / args.sparse / "images.txt")
    images = images[:: args.frame_stride][: args.max_frames]
    if not images:
        raise RuntimeError("No images selected")

    presence = np.load(args.presence_npz)
    xyz = presence["xyz"].astype(np.float64)
    if args.mask_key not in presence.files:
        raise RuntimeError(f"{args.mask_key} is absent from {args.presence_npz}")
    if args.cluster_key not in presence.files:
        raise RuntimeError(f"{args.cluster_key} is absent from {args.presence_npz}")
    selected_all = presence[args.mask_key].astype(bool)
    cluster_id = presence[args.cluster_key].astype(np.int64)
    available = sorted(int(v) for v in np.unique(cluster_id[selected_all]) if int(v) >= 0)
    selected_clusters = parse_cluster_ids(args.selected_clusters, available)
    selected = selected_all & np.isin(cluster_id, np.asarray(selected_clusters, dtype=np.int64))
    if not selected.any():
        raise RuntimeError("No selected Gaussians to visualize")

    cluster_payload = []
    for cid in selected_clusters:
        mask = selected & (cluster_id == cid)
        points = sample_points(xyz[mask], args.max_points_per_cluster, 1700 + cid)
        if points.shape[0]:
            cluster_payload.append((cid, points))
    target_items = load_targets(args.summary_json, selected_clusters)
    target_points = (
        np.asarray([item["point"] for item in target_items], dtype=np.float64)
        if target_items
        else np.zeros((0, 3), dtype=np.float64)
    )

    output_paths = []
    metadata = []
    font = ImageFont.load_default()
    for frame_idx, image in enumerate(images):
        image_path = dataset_dir / args.images / image["name"]
        base_img = Image.open(image_path).convert("RGB")

        all_u = []
        all_v = []
        all_valid = []
        projected = []
        for cid, points in cluster_payload:
            u, v, valid = project_points(points, image, camera)
            all_u.append(u)
            all_v.append(v)
            all_valid.append(valid)
            projected.append((cid, u, v, valid))
        if target_points.shape[0]:
            tu, tv, tvalid = project_points(target_points, image, camera)
            all_u.append(tu)
            all_v.append(tv)
            all_valid.append(tvalid)
        else:
            tu = tv = np.zeros((0,), dtype=np.float64)
            tvalid = np.zeros((0,), dtype=bool)
        box = crop_box(
            np.concatenate(all_u),
            np.concatenate(all_v),
            np.concatenate(all_valid),
            base_img.width,
            base_img.height,
            args,
        )
        left, top, right, bottom = box
        crop = base_img.crop(box).resize((args.zoom_size, args.zoom_size), Image.Resampling.BILINEAR)
        sx = args.zoom_size / max(1, right - left)
        sy = args.zoom_size / max(1, bottom - top)
        draw = ImageDraw.Draw(crop)

        visible_points = 0
        for cid, u, v, valid in projected:
            color = (255, 190, 30) if args.point_color_mode == "yellow" else COLORS.get(cid % len(COLORS), (255, 190, 30))
            visible_points += int(valid.sum())
            for uu, vv, ok in zip(u, v, valid):
                if not ok:
                    continue
                x = int(round((float(uu) - left) * sx))
                y = int(round((float(vv) - top) * sy))
                draw_point(draw, x, y, color, args.point_radius)

        for item, uu, vv, ok in zip(target_items, tu, tv, tvalid):
            if not ok:
                continue
            x = int(round((float(uu) - left) * sx))
            y = int(round((float(vv) - top) * sy))
            color = COLORS.get(item["cluster_id"] % len(COLORS), (255, 255, 255))
            r = args.target_radius
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255), width=3)
            draw.line((x - r - 5, y, x + r + 5, y), fill=color, width=3)
            draw.line((x, y - r - 5, x, y + r + 5), fill=color, width=3)
            if not args.hide_target_labels:
                label = f"T{item['cluster_id']}"
                draw.text((x + r + 4, y - r), label, fill=(255, 255, 255), font=font)

        out_path = points_dir / f"{frame_idx:03d}_{Path(image['name']).stem}_points.jpg"
        crop.save(out_path, quality=92)
        output_paths.append(out_path)
        metadata.append(
            {
                "frame": image["name"],
                "crop_box": list(box),
                "visible_selected_points": visible_points,
                "visible_targets": int(tvalid.sum()),
                "point_image": str(out_path),
            }
        )

    make_contact_sheet(output_paths, out_dir / "points_contact_sheet.jpg")
    (out_dir / "visualization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved point contact sheet: {out_dir / 'points_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
