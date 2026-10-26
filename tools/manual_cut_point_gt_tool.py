#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


TASKS = {
    "deleafing": {
        "json_name": "deleafing_cut_point_gt.json",
        "target_phase": "pre_deleafing",
        "comparison_phase": "post_deleafing_pre_harvesting",
        "label": "deleafing_cut_point",
    },
    "tomato_harvesting": {
        "json_name": "tomato_harvesting_cut_point_gt.json",
        "target_phase": "post_deleafing_pre_harvesting",
        "comparison_phase": "post_harvesting",
        "label": "tomato_harvesting_cut_point",
    },
}

DARK_PURPLE = (52, 0, 88)
CYAN = (35, 215, 255)
YELLOW = (255, 215, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass
class Trajectory:
    timestamps_us: np.ndarray
    translations: np.ndarray
    rotations: Rotation
    slerp: Slerp


@dataclass
class Frame:
    image: Image.Image
    timestamp_ns: int
    rel_s: float
    T_world_camera: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual GT cut-point helper. This tool does not select labels. "
            "It only renders review sheets and converts explicit human/agent "
            "chosen pixels into nearest fresh-MPS 3D points."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split_root", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark_splitted"))
    common.add_argument("--out_dir", type=Path, default=Path("/path/to/data/mps/egodelta_benchmark_splitted/cut_point_gt_manual"))
    common.add_argument("--camera_label", default="camera-rgb")
    common.add_argument("--rect_size", type=int, default=720)
    common.add_argument("--rect_focal", type=float, default=380.0)
    common.add_argument("--inv_dist_std_max", type=float, default=0.05)
    common.add_argument("--dist_std_max", type=float, default=0.10)

    render = sub.add_parser("render", parents=[common])
    render.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    render.add_argument("--records", nargs="*", default=None)
    render.add_argument("--frames_per_phase", type=int, default=6)
    render.add_argument("--thumb", type=int, default=360)
    render.add_argument("--overwrite", action="store_true")

    finalize = sub.add_parser("finalize", parents=[common])
    finalize.add_argument("--labels", type=Path, required=True)
    finalize.add_argument("--projection_radius_px", type=float, default=36.0)
    finalize.add_argument("--local_depth_radius_m", type=float, default=0.14)
    finalize.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "render":
        render_review_sheets(args)
    elif args.cmd == "finalize":
        finalize_labels(args)
    return 0


def render_review_sheets(args: argparse.Namespace) -> None:
    root = args.out_dir / "manual_review_sheets"
    root.mkdir(parents=True, exist_ok=True)
    records = discover_records(args.split_root, args.records)
    manifest = {
        "type": "manual_cut_point_review_manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Review sheets only. No labels are selected by this tool.",
        "records": [],
    }
    for idx, rec_dir in enumerate(records, 1):
        rec_id = rec_dir.name
        print(f"render [{idx}/{len(records)}] {rec_id}", flush=True)
        rec_manifest = {"recording_id": rec_id, "tasks": {}}
        for task in args.tasks:
            out_path = root / task / f"{rec_id}.jpg"
            meta_path = root / task / f"{rec_id}.json"
            if out_path.exists() and meta_path.exists() and not args.overwrite:
                rec_manifest["tasks"][task] = {"sheet": str(out_path.resolve()), "metadata": str(meta_path.resolve())}
                continue
            sheet, meta = make_review_sheet(rec_dir, task, args)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(out_path, quality=92)
            write_json(meta_path, meta)
            rec_manifest["tasks"][task] = {"sheet": str(out_path.resolve()), "metadata": str(meta_path.resolve())}
        manifest["records"].append(rec_manifest)
    write_json(root / "manifest.json", manifest)


def make_review_sheet(rec_dir: Path, task: str, args: argparse.Namespace) -> tuple[Image.Image, dict]:
    spec = TASKS[task]
    target = load_phase(rec_dir, spec["target_phase"], args)
    comparison = load_phase(rec_dir, spec["comparison_phase"], args)
    phases = [(spec["target_phase"], target), (spec["comparison_phase"], comparison)]
    panels = []
    panel_meta = []
    panel_id = 0
    for phase_name, phase in phases:
        for frame in sample_phase_frames(phase, args.frames_per_phase):
            img = frame.image.copy()
            draw = ImageDraw.Draw(img)
            draw_text(draw, f"{panel_id:02d} {task} | {phase_name} | {frame.rel_s:.1f}s")
            img.thumbnail((args.thumb, args.thumb))
            panels.append(img.copy())
            panel_meta.append(
                {
                    "panel_id": panel_id,
                    "task": task,
                    "recording_id": rec_dir.name,
                    "phase": phase_name,
                    "timestamp_ns": int(frame.timestamp_ns),
                    "relative_time_s": float(frame.rel_s),
                    "panel_width": int(img.width),
                    "panel_height": int(img.height),
                    "display_note": "uv_px in manual labels is measured in the full-size 720x720 rectified image, not this thumbnail.",
                }
            )
            panel_id += 1

    cols = args.frames_per_phase
    label_h = 34
    width = cols * args.thumb
    rows = 2
    height = rows * (args.thumb + label_h)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, img in enumerate(panels):
        row = i // cols
        col = i % cols
        x = col * args.thumb + (args.thumb - img.width) // 2
        y = row * (args.thumb + label_h)
        canvas.paste(img, (x, y))
        item = panel_meta[i]
        y_label = row * (args.thumb + label_h) + args.thumb
        color = (38, 88, 52) if item["phase"] == spec["target_phase"] else (90, 53, 120)
        draw.rectangle([col * args.thumb, y_label, (col + 1) * args.thumb, y_label + label_h], fill=color)
        draw.text((col * args.thumb + 5, y_label + 4), f"{item['panel_id']:02d} {item['relative_time_s']:.1f}s", fill=WHITE, font=font)
        draw.text((col * args.thumb + 5, y_label + 18), item["phase"][:34], fill=WHITE, font=font)
    meta = {
        "recording_id": rec_dir.name,
        "task": task,
        "target_phase": spec["target_phase"],
        "comparison_phase": spec["comparison_phase"],
        "rectified_image_size": [args.rect_size, args.rect_size],
        "panels": panel_meta,
    }
    return canvas, meta


def finalize_labels(args: argparse.Namespace) -> None:
    manual = read_json(args.labels)
    labels = expand_manual_labels(manual.get("labels", []))
    grouped: dict[str, list[dict]] = {task: [] for task in TASKS}
    for item in labels:
        task = item["task"]
        if task not in TASKS:
            raise ValueError(f"Unknown task {task}")
        grouped[task].append(item)

    for task, items in grouped.items():
        if not items:
            continue
        spec = TASKS[task]
        records_payload = []
        by_record: dict[str, list[dict]] = {}
        for item in items:
            by_record.setdefault(item["recording_id"], []).append(item)
        for rec_id, rec_items in sorted(by_record.items()):
            rec_dir = args.split_root / rec_id
            target = load_phase(rec_dir, spec["target_phase"], args)
            comparison = load_phase(rec_dir, spec["comparison_phase"], args)
            cut_points = []
            visual_paths = []
            for rank, item in enumerate(rec_items):
                selected = selected_phase_for_label(item, target, comparison, spec)
                frame = frame_for_label(selected, item)
                uv = label_uv(item)
                xyz, depth_m, support = depth_from_uv(selected.points_world, frame, selected.camera, uv, args.projection_radius_px)
                cut_id = f"{rec_id}:{task}:{rank:02d}"
                cut = {
                    "cut_point_id": cut_id,
                    "rank": rank,
                    "label": spec["label"],
                    "xyz_world_m": [float(v) for v in xyz],
                    "coordinate_phase": selected.phase,
                    "coordinate_mps_dir": str(selected.mps_dir.resolve()),
                    "annotation_status": "manual_visual_label",
                    "annotator": "codex_visual_inspection",
                    "confidence": float(item.get("confidence", 1.0)),
                    "source": {
                        "selected_phase": selected.phase,
                        "selected_timestamp_ns": int(frame.timestamp_ns),
                        "selected_relative_time_s": float(frame.rel_s),
                        "selected_uv_rectified_px": [float(uv[0]), float(uv[1])],
                        "camera_depth_m": float(depth_m),
                        "mps_projection_support_points": int(support),
                        "note": item.get("note", ""),
                    },
                }
                vis_path = args.out_dir / "visualizations" / task / rec_id / f"{rank:02d}_{task}.jpg"
                draw_final_visualization(cut, target, comparison, selected, frame, uv, xyz, depth_m, vis_path, args)
                cut["sample_visualization"] = str(vis_path.resolve())
                cut_points.append(cut)
                visual_paths.append(str(vis_path.resolve()))
            records_payload.append(
                {
                    "recording_id": rec_id,
                    "task": task,
                    "status": "ok" if cut_points else "no_manual_labels",
                    "target_phase": spec["target_phase"],
                    "comparison_phase": spec["comparison_phase"],
                    "coordinate_phase": "per_cut_point",
                    "target_mps_dir": str(target.mps_dir.resolve()),
                    "comparison_mps_dir": str(comparison.mps_dir.resolve()),
                    "cut_points": cut_points,
                    "visualizations": visual_paths,
                }
            )
        payload = {
            "type": "egodelta_cut_point_ground_truth",
            "annotation_mode": "manual_visual_labeling_by_codex",
            "task": task,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manual_label_file": str(args.labels.resolve()),
            "coordinate_frame": (
                "Each xyz_world_m is in the fresh split-MPS world frame named by that "
                "cut point's coordinate_phase. World frames are not shared across "
                "recordings or phases."
            ),
            "records": records_payload,
            "summary": summarize(records_payload),
        }
        write_json(args.out_dir / spec["json_name"], payload)
        make_contact_sheet(args.out_dir / "visualizations" / task)


def expand_manual_labels(labels: Iterable[dict]) -> list[dict]:
    expanded = []
    for item in labels:
        points = item.get("points")
        if points is None:
            expanded.append(item)
            continue
        base = {k: v for k, v in item.items() if k != "points"}
        for idx, uv in enumerate(points):
            out = dict(base)
            out["uv_rectified_px"] = uv
            out["point_index"] = idx
            expanded.append(out)
    return expanded


@dataclass
class Phase:
    recording_id: str
    phase: str
    vrs_path: Path
    mps_dir: Path
    provider: object
    rgb_id: object
    camera: dict
    trajectory: Trajectory
    points_world: np.ndarray
    start_ns: int
    end_ns: int
    frames_cache: dict[int, Frame]


def selected_phase_for_label(item: dict, target: Phase, comparison: Phase, spec: dict) -> Phase:
    phase_name = item.get("phase")
    if not phase_name and "panel_id" in item and item.get("panel_metadata"):
        meta = read_json(Path(item["panel_metadata"]))
        panel = next(p for p in meta["panels"] if int(p["panel_id"]) == int(item["panel_id"]))
        phase_name = panel["phase"]
    if not phase_name:
        phase_name = spec["target_phase"]
    if phase_name == target.phase:
        return target
    if phase_name == comparison.phase:
        return comparison
    raise ValueError(
        f"Manual label phase {phase_name!r} is not one of "
        f"{target.phase!r} or {comparison.phase!r}: {item}"
    )


def discover_records(split_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        return [split_root / rec for rec in requested]
    records = []
    for p in sorted(split_root.iterdir()):
        if not p.is_dir() or p.name in {"contact_sheets", "timing_json", "cut_point_gt", "cut_point_gt_manual"}:
            continue
        if all((p / phase / f"{phase}.vrs").exists() for phase in ("pre_deleafing", "post_deleafing_pre_harvesting", "post_harvesting")):
            records.append(p)
    return records


def load_phase(rec_dir: Path, phase: str, args: argparse.Namespace) -> Phase:
    vrs_path = rec_dir / phase / f"{phase}.vrs"
    mps_dir = rec_dir / phase / f"mps_{phase}_vrs"
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if not provider:
        raise RuntimeError(f"Could not open {vrs_path}")
    rgb_id = provider.get_stream_id_from_label(args.camera_label)
    camera = make_rectified_camera(provider, args.camera_label, args.rect_size, args.rect_focal)
    trajectory = load_trajectory(mps_dir / "slam" / "closed_loop_trajectory.csv")
    points_world = load_semidense_points(mps_dir / "slam" / "semidense_points.csv.gz", args.inv_dist_std_max, args.dist_std_max)
    times = read_json(vrs_path.parent / "phase_times.json")
    phase_start_ns = int(times["start_device_time_ns"])
    phase_end_ns = int(times["end_device_time_ns"])
    traj_start_ns = int(trajectory.timestamps_us[0] * 1000)
    traj_end_ns = int(trajectory.timestamps_us[-1] * 1000)
    start_ns = max(phase_start_ns, traj_start_ns)
    end_ns = min(phase_end_ns, traj_end_ns)
    if end_ns <= start_ns:
        start_ns, end_ns = traj_start_ns, traj_end_ns
    return Phase(
        recording_id=rec_dir.name,
        phase=phase,
        vrs_path=vrs_path,
        mps_dir=mps_dir,
        provider=provider,
        rgb_id=rgb_id,
        camera=camera,
        trajectory=trajectory,
        points_world=points_world,
        start_ns=start_ns,
        end_ns=end_ns,
        frames_cache={},
    )


def sample_phase_frames(phase: Phase, n: int) -> list[Frame]:
    start, end = phase.start_ns, phase.end_ns
    margin = min(int((end - start) * 0.08), int(0.8e9))
    a, b = start + margin, end - margin
    if b <= a:
        a, b = start, end
    return [load_frame_at(phase, int(ts)) for ts in np.linspace(a, b, max(2, n))]


def frame_for_label(phase: Phase, item: dict) -> Frame:
    if "timestamp_ns" in item:
        return load_frame_at(phase, int(item["timestamp_ns"]))
    if "relative_time_s" in item:
        return load_frame_at(phase, int(phase.start_ns + float(item["relative_time_s"]) * 1e9))
    if "panel_id" in item:
        meta_path = item.get("panel_metadata")
        if not meta_path:
            raise ValueError("panel_id labels require panel_metadata path")
        meta = read_json(Path(meta_path))
        panel = next(p for p in meta["panels"] if int(p["panel_id"]) == int(item["panel_id"]))
        return load_frame_at(phase, int(panel["timestamp_ns"]))
    raise ValueError(f"Label must include timestamp_ns, relative_time_s, or panel_id: {item}")


def label_uv(item: dict) -> np.ndarray:
    if "uv_rectified_px" not in item:
        raise ValueError(f"Manual label missing uv_rectified_px: {item}")
    uv = np.asarray(item["uv_rectified_px"], dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError(f"Invalid uv_rectified_px: {item}")
    return uv


def load_frame_at(phase: Phase, timestamp_ns: int) -> Frame:
    if timestamp_ns in phase.frames_cache:
        return phase.frames_cache[timestamp_ns]
    image_data, record = phase.provider.get_image_data_by_time_ns(
        phase.rgb_id,
        int(timestamp_ns),
        TimeDomain.DEVICE_TIME,
        TimeQueryOptions.CLOSEST,
    )
    ts_ns = image_record_timestamp_ns(record)
    T_wd = pose_at_or_nearest(phase.trajectory, ts_ns // 1000)
    if T_wd is None:
        raise RuntimeError(f"No MPS pose for {phase.vrs_path} at {ts_ns}")
    T_wc = T_wd @ phase.camera["T_device_camera"]
    image = rectify_image(image_data, phase.camera)
    frame = Frame(image=image, timestamp_ns=ts_ns, rel_s=(ts_ns - phase.start_ns) / 1e9, T_world_camera=T_wc)
    phase.frames_cache[timestamp_ns] = frame
    return frame


def depth_from_uv(
    points_world: np.ndarray,
    frame: Frame,
    camera: dict,
    uv: np.ndarray,
    radius_px: float,
) -> tuple[np.ndarray, float, int]:
    proj_uv, proj_depth, valid = project_world_points(points_world, frame.T_world_camera, camera)
    valid_ids = np.flatnonzero(valid)
    if len(valid_ids) == 0:
        raise RuntimeError("No projected MPS points visible in selected frame")
    tree = cKDTree(proj_uv[valid_ids])
    ids = tree.query_ball_point(uv, radius_px)
    if not ids:
        dist, idx = tree.query(uv, k=1)
        if not np.isfinite(dist):
            raise RuntimeError("No nearest projected MPS point found")
        ids = [int(idx)]
    world_ids = valid_ids[np.asarray(ids, dtype=np.int64)]
    distances = np.linalg.norm(proj_uv[world_ids] - uv[None, :], axis=1)
    order = np.argsort(distances)[: min(16, len(distances))]
    world_ids = world_ids[order]
    distances = distances[order]
    weights = 1.0 / np.maximum(distances, 2.0)
    weights /= max(float(weights.sum()), 1e-8)
    xyz = (points_world[world_ids] * weights[:, None]).sum(axis=0)
    depth = float(np.average(proj_depth[world_ids], weights=weights))
    return xyz, depth, int(len(world_ids))


def draw_final_visualization(
    cut: dict,
    target: Phase,
    comparison: Phase,
    selected: Phase,
    frame: Frame,
    uv: np.ndarray,
    xyz: np.ndarray,
    depth_m: float,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    target_frame = frame if selected.phase == target.phase else nearest_relative_frame(target, frame.rel_s)
    comparison_frame = frame if selected.phase == comparison.phase else nearest_relative_frame(comparison, frame.rel_s)
    depth_frame = frame
    panels = []
    panel_specs = [
        ("target-present context", target, target_frame),
        ("post comparison context", comparison, comparison_frame),
        ("fresh MPS depth support", selected, depth_frame),
    ]
    for title, phase_obj, panel_frame in panel_specs:
        img = panel_frame.image.copy()
        draw = ImageDraw.Draw(img)
        if title.startswith("fresh"):
            draw_local_depth(draw, selected.points_world, xyz, depth_frame.T_world_camera, selected.camera, args.local_depth_radius_m)
            proj_uv, proj_depth, ok = project_world_point(xyz, depth_frame.T_world_camera, selected.camera)
            if ok:
                draw_target(draw, proj_uv)
                draw_panel_text(draw, title, cut["cut_point_id"], proj_depth)
            else:
                draw_target(draw, uv)
                draw_panel_text(draw, title, cut["cut_point_id"], depth_m)
        else:
            if phase_obj.phase == selected.phase:
                draw_target(draw, uv)
                draw_panel_text(draw, title + " | selected label", cut["cut_point_id"], depth_m)
            else:
                draw_panel_text(draw, title, cut["cut_point_id"], None)
        panels.append(img)
    canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), (18, 18, 18))
    x = 0
    for img in panels:
        canvas.paste(img, (x, 0))
        x += img.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def nearest_relative_frame(phase: Phase, rel_s: float) -> Frame:
    ts = int(phase.start_ns + rel_s * 1e9)
    ts = max(phase.start_ns, min(phase.end_ns, ts))
    return load_frame_at(phase, ts)


def draw_local_depth(
    draw: ImageDraw.ImageDraw,
    points_world: np.ndarray,
    xyz: np.ndarray,
    T_world_camera: np.ndarray,
    camera: dict,
    radius_m: float,
) -> None:
    if len(points_world) == 0:
        return
    tree = cKDTree(points_world)
    ids = tree.query_ball_point(xyz, radius_m)
    if len(ids) > 1000:
        ids = list(np.asarray(ids)[np.linspace(0, len(ids) - 1, 1000).astype(np.int64)])
    if not ids:
        return
    pts = points_world[np.asarray(ids, dtype=np.int64)]
    uv, _z, valid = project_world_points(pts, T_world_camera, camera)
    for u, v in uv[valid]:
        draw.ellipse([u - 1.7, v - 1.7, u + 1.7, v + 1.7], fill=CYAN)


def draw_target(draw: ImageDraw.ImageDraw, uv: np.ndarray) -> None:
    u, v = float(uv[0]), float(uv[1])
    r = 14
    draw.ellipse([u - r, v - r, u + r, v + r], outline=WHITE, width=6)
    draw.line([u - r - 4, v, u + r + 4, v], fill=WHITE, width=6)
    draw.line([u, v - r - 4, u, v + r + 4], fill=WHITE, width=6)
    draw.ellipse([u - r, v - r, u + r, v + r], outline=DARK_PURPLE, width=4)
    draw.line([u - r - 4, v, u + r + 4, v], fill=DARK_PURPLE, width=4)
    draw.line([u, v - r - 4, u, v + r + 4], fill=DARK_PURPLE, width=4)


def draw_text(draw: ImageDraw.ImageDraw, text: str) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([8, 8, bbox[2] + 18, bbox[3] + 18], fill=BLACK)
    draw.text((13, 13), text, fill=WHITE, font=font)


def draw_panel_text(draw: ImageDraw.ImageDraw, title: str, cut_id: str, depth_m: float | None) -> None:
    depth = "depth n/a" if depth_m is None or not np.isfinite(depth_m) else f"depth {depth_m:.3f}m"
    draw_text(draw, f"{title} | {cut_id.split(':')[-1]} | {depth}")


def make_contact_sheet(vis_root: Path) -> None:
    paths = sorted(vis_root.glob("*/*.jpg"))
    if not paths:
        return
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((900, 300))
        thumbs.append(img.copy())
    width = max(img.width for img in thumbs)
    height = sum(img.height for img in thumbs)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for img in thumbs:
        sheet.paste(img, (0, y))
        y += img.height
    sheet.save(vis_root / "contact_sheet_all.jpg", quality=92)


def load_trajectory(path: Path) -> Trajectory:
    rows = []
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            try:
                ts = int(row["tracking_timestamp_us"])
                t = [float(row["tx_world_device"]), float(row["ty_world_device"]), float(row["tz_world_device"])]
                q = [
                    float(row["qx_world_device"]),
                    float(row["qy_world_device"]),
                    float(row["qz_world_device"]),
                    float(row["qw_world_device"]),
                ]
            except Exception:
                continue
            if ts > 0 and np.isfinite(t).all() and np.isfinite(q).all():
                rows.append((ts, t, q))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise RuntimeError(f"No trajectory loaded from {path}")
    ts = np.asarray([r[0] for r in rows], dtype=np.int64)
    t = np.asarray([r[1] for r in rows], dtype=np.float64)
    q = np.asarray([r[2] for r in rows], dtype=np.float64)
    keep = np.r_[True, ts[1:] != ts[:-1]]
    ts, t, q = ts[keep], t[keep], q[keep]
    rotations = Rotation.from_quat(q)
    return Trajectory(ts, t, rotations, Slerp(ts, rotations))


def pose_at(trajectory: Trajectory, timestamp_us: int) -> np.ndarray | None:
    ts = int(timestamp_us)
    if ts < int(trajectory.timestamps_us[0]) or ts > int(trajectory.timestamps_us[-1]):
        return None
    idx = int(np.searchsorted(trajectory.timestamps_us, ts).clip(1, len(trajectory.timestamps_us) - 1))
    i0 = idx - 1
    denom = max(int(trajectory.timestamps_us[idx] - trajectory.timestamps_us[i0]), 1)
    alpha = (ts - int(trajectory.timestamps_us[i0])) / denom
    t = trajectory.translations[i0] + alpha * (trajectory.translations[idx] - trajectory.translations[i0])
    r = trajectory.slerp([ts]).as_matrix()[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def pose_at_or_nearest(trajectory: Trajectory, timestamp_us: int) -> np.ndarray | None:
    ts = int(timestamp_us)
    if ts < int(trajectory.timestamps_us[0]):
        ts = int(trajectory.timestamps_us[0])
    elif ts > int(trajectory.timestamps_us[-1]):
        ts = int(trajectory.timestamps_us[-1])
    return pose_at(trajectory, ts)


def load_semidense_points(path: Path, inv_std_max: float, dist_std_max: float) -> np.ndarray:
    points = []
    with gzip.open(path, "rt", newline="") as fp:
        for row in csv.DictReader(fp):
            try:
                xyz = [float(row["px_world"]), float(row["py_world"]), float(row["pz_world"])]
                inv_std = float(row.get("inv_dist_std", "nan"))
                dist_std = float(row.get("dist_std", "nan"))
            except Exception:
                continue
            if not np.isfinite(xyz).all():
                continue
            if np.isfinite(inv_std) and inv_std > inv_std_max:
                continue
            if np.isfinite(dist_std) and dist_std > dist_std_max:
                continue
            points.append(xyz)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def make_rectified_camera(provider, camera_label: str, rect_size: int, rect_focal: float) -> dict:
    rgb_calib_orig = provider.get_device_calibration().get_camera_calib(camera_label)
    T_device_camera_orig = rgb_calib_orig.get_transform_device_camera()
    linear = calibration.get_linear_camera_calibration(
        rect_size,
        rect_size,
        rect_focal,
        f"{camera_label}-linear",
        T_device_camera_orig,
    )
    linear_rot = calibration.rotate_camera_calib_cw90deg(linear)
    fx, fy = linear_rot.get_focal_lengths()
    cx, cy = linear_rot.get_principal_point()
    width, height = linear_rot.get_image_size()
    return {
        "orig_calib": rgb_calib_orig,
        "linear_calib": linear,
        "linear_rot_calib": linear_rot,
        "T_device_camera": linear_rot.get_transform_device_camera().to_matrix(),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": int(width),
        "height": int(height),
    }


def rectify_image(image_data, camera: dict) -> Image.Image:
    img_orig = image_data.to_numpy_array()
    img_rect_unrot = calibration.distort_by_calibration(img_orig, camera["linear_calib"], camera["orig_calib"])
    img_rect = np.rot90(img_rect_unrot, k=-1).copy()
    return Image.fromarray(img_rect).convert("RGB")


def project_world_point(point_world: np.ndarray, T_world_camera: np.ndarray, camera: dict) -> tuple[np.ndarray, float, bool]:
    T_cw = np.linalg.inv(T_world_camera)
    p = np.ones(4, dtype=np.float64)
    p[:3] = point_world
    pc = (T_cw @ p)[:3]
    z = float(pc[2])
    uv = np.asarray([np.nan, np.nan], dtype=np.float64)
    ok = False
    if z > 1e-6:
        uv[0] = camera["fx"] * pc[0] / z + camera["cx"]
        uv[1] = camera["fy"] * pc[1] / z + camera["cy"]
        ok = np.isfinite(uv).all() and 0 <= uv[0] < camera["width"] and 0 <= uv[1] < camera["height"]
    return uv, z, bool(ok)


def project_world_points(points_world: np.ndarray, T_world_camera: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=bool)
    T_cw = np.linalg.inv(T_world_camera)
    homog = np.c_[points_world, np.ones(len(points_world), dtype=np.float64)]
    pc = (T_cw @ homog.T).T[:, :3]
    z = pc[:, 2]
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    ok = z > 1e-6
    uv[ok, 0] = camera["fx"] * pc[ok, 0] / z[ok] + camera["cx"]
    uv[ok, 1] = camera["fy"] * pc[ok, 1] / z[ok] + camera["cy"]
    valid = ok & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < camera["height"])
    return uv, z, valid


def image_record_timestamp_ns(record) -> int:
    if hasattr(record, "capture_timestamp_ns"):
        return int(record.capture_timestamp_ns)
    if hasattr(record, "captureTimestampNs"):
        return int(record.captureTimestampNs)
    raise AttributeError("Could not read capture timestamp from image record")


def summarize(records: list[dict]) -> dict:
    points = [cp for r in records for cp in r.get("cut_points", [])]
    return {
        "num_records": len(records),
        "num_records_ok": sum(1 for r in records if r.get("status") == "ok"),
        "num_cut_points": len(points),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
