from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_cutpoints import parse_labels, parse_task_names


TASKS = {
    "deleaf": {"label_key": "deleafing", "title": "Deleafing"},
    "harvest": {"label_key": "harvesting", "title": "Harvesting"},
}


def read_ascii_ply(path: Path, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64), None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        n_vertices = 0
        properties: list[str] = []
        for line in f:
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                n_vertices = int(stripped.split()[-1])
            elif stripped.startswith("property"):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if n_vertices <= 0:
            return np.empty((0, 3), dtype=np.float64), None
        data = np.loadtxt(f, max_rows=n_vertices)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, :3]
    colors = None
    if data.shape[1] >= 6:
        colors = data[:, 3:6] / 255.0

    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(13)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
        if colors is not None:
            colors = colors[idx]
    return xyz, colors


def read_predicted_cutpoints(targets_path: Path) -> np.ndarray:
    if not targets_path.exists():
        return np.empty((0, 3), dtype=np.float64)
    targets = json.loads(targets_path.read_text(encoding="utf-8")).get("targets", [])
    points = [target["cut_point_world"] for target in targets if target.get("cut_point_world") is not None]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def read_gt_points(run_dir: Path, labels: dict[str, Any], uuid: str, label_key: str) -> np.ndarray:
    transformed_path = run_dir / "gt_points_prediction_frame.json"
    if transformed_path.exists():
        doc = json.loads(transformed_path.read_text(encoding="utf-8"))
        points = doc.get("gt_points_prediction_frame", [])
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.asarray(labels[uuid][label_key], dtype=np.float64).reshape(-1, 3)


def resolve_run_dir(eval_root: Path, uuid: str, task: str) -> Path:
    runs_layout = eval_root / "runs" / uuid / task
    if runs_layout.exists():
        return runs_layout
    return eval_root / uuid / task


def match_errors(pred: np.ndarray, gt: np.ndarray) -> list[float]:
    if len(pred) == 0 or len(gt) == 0:
        return []
    dist_cm = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2) * 100.0
    pred_idx, gt_idx = linear_sum_assignment(dist_cm)
    return [float(dist_cm[p, g]) for p, g in zip(pred_idx, gt_idx)]


def axis_limits(*arrays: np.ndarray) -> dict[str, tuple[float, float]]:
    valid = [arr.reshape(-1, 3) for arr in arrays if arr is not None and len(arr)]
    if not valid:
        return {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}
    pts = np.vstack(valid)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = 0.08 * span
    return {
        "x": (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        "y": (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
        "z": (float(mins[2] - pad[2]), float(maxs[2] + pad[2])),
    }


def local_axis_limits(center_points: np.ndarray, radius_m: float) -> dict[str, tuple[float, float]]:
    points = np.asarray(center_points, dtype=np.float64).reshape(-1, 3)
    center = points.mean(axis=0) if len(points) else np.zeros(3, dtype=np.float64)
    radius = max(float(radius_m), 1e-3)
    return {
        "x": (float(center[0] - radius), float(center[0] + radius)),
        "y": (float(center[1] - radius), float(center[1] + radius)),
        "z": (float(center[2] - radius), float(center[2] + radius)),
    }


def scatter_projection(
    ax: plt.Axes,
    projection: tuple[int, int],
    limits: dict[str, tuple[float, float]],
    scene: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
) -> None:
    names = ["x", "y", "z"]
    a, b = projection
    if len(scene):
        ax.scatter(scene[:, a], scene[:, b], s=0.15, c="#1f4f93", alpha=0.12, linewidths=0)
    if len(target):
        ax.scatter(target[:, a], target[:, b], s=1.4, c="#d62728", alpha=0.55, linewidths=0)
    if len(gt):
        ax.scatter(gt[:, a], gt[:, b], s=18, c="#00d7ff", marker=".", linewidths=0, label="GT")
    if len(pred):
        ax.scatter(
            pred[:, a],
            pred[:, b],
            s=18,
            c="#ffe119",
            marker=".",
            linewidths=0,
            label="Pred",
        )
    ax.set_xlim(*limits[names[a]])
    ax.set_ylim(*limits[names[b]])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(names[a], fontsize=8)
    ax.set_ylabel(names[b], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.25, alpha=0.25)


def plot_uuid(
    uuid: str,
    labels: dict[str, Any],
    eval_root: Path,
    out_path: Path,
    max_scene_points: int,
    tasks: list[str],
    zoom_radius_m: float | None = None,
) -> bool:
    n_rows = len(tasks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.2 * n_rows), dpi=160)
    axes_arr = np.asarray(axes).reshape(n_rows, 3)
    fig.suptitle(uuid, fontsize=13)
    made_any = False

    projections = [((0, 2), "X-Z"), ((0, 1), "X-Y"), ((2, 1), "Z-Y")]
    for row, task in enumerate(tasks):
        spec = TASKS[task]
        run_dir = resolve_run_dir(eval_root, uuid, task)
        scene, _ = read_ascii_ply(run_dir / "plys" / "post_filtered_aligned.ply", max_points=max_scene_points)
        target, _ = read_ascii_ply(run_dir / "plys" / "target_candidates.ply", max_points=None)
        pred = read_predicted_cutpoints(run_dir / "targets.json")
        gt = read_gt_points(run_dir, labels, uuid, spec["label_key"])
        errors = match_errors(pred, gt)
        made_any = made_any or len(pred) > 0 or len(gt) > 0 or len(target) > 0
        if zoom_radius_m is not None and (len(pred) or len(gt)):
            limits = local_axis_limits(np.vstack([arr for arr in [gt, pred] if len(arr)]), zoom_radius_m)
        else:
            limits = axis_limits(scene, target, pred, gt)
        err_text = "no matches" if not errors else f"mean {np.mean(errors):.1f} cm, med {np.median(errors):.1f} cm"
        if zoom_radius_m is not None:
            err_text = f"{err_text}, zoom +/- {zoom_radius_m * 100.0:g} cm"
        count_text = f"GT {len(gt)} / Pred {len(pred)}"
        for col, (proj, proj_name) in enumerate(projections):
            title = f"{spec['title']} {proj_name} | {count_text} | {err_text}" if col == 0 else proj_name
            scatter_projection(axes_arr[row, col], proj, limits, scene, target, pred, gt, title)
            if row == 0 and col == 2:
                axes_arr[row, col].legend(loc="upper right", fontsize=7)

    if not made_any:
        plt.close(fig)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    return True


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 5) -> None:
    if not image_paths:
        return
    import matplotlib.image as mpimg

    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.1), dpi=160)
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, path in zip(axes_arr.ravel(), image_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem, fontsize=6)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cut point evaluation PNGs for manual inspection.")
    parser.add_argument("--labels", type=Path, default=Path("3d points labels.txt"))
    parser.add_argument("--eval-root", type=Path, default=Path("outputs/cutpoint_eval"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cutpoint_eval/visualizations"))
    parser.add_argument("--max-scene-points", type=int, default=50000)
    parser.add_argument(
        "--uuids",
        type=str,
        default=None,
        help="Comma-separated UUIDs to render. Defaults to all labels.",
    )
    parser.add_argument(
        "--zoom-radius-m",
        type=float,
        default=None,
        help="Use a local +/- radius around GT and predicted cut points, in meters.",
    )
    parser.add_argument(
        "--tasks",
        type=parse_task_names,
        default=list(TASKS),
        help="Comma-separated tasks to visualize: deleaf, harvest, or deleaf,harvest.",
    )
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="Skip UUIDs without targets.json files for the selected tasks.",
    )
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    uuids = [uuid.strip() for uuid in args.uuids.split(",") if uuid.strip()] if args.uuids else list(labels)
    image_paths: list[Path] = []
    skipped: list[str] = []
    for uuid in uuids:
        if uuid not in labels:
            skipped.append(uuid)
            continue
        if args.only_complete:
            missing = [
                task
                for task in args.tasks
                if not (resolve_run_dir(args.eval_root, uuid, task) / "targets.json").exists()
            ]
            if missing:
                skipped.append(uuid)
                continue
        out_path = args.out_dir / f"{uuid}.png"
        if plot_uuid(uuid, labels, args.eval_root, out_path, args.max_scene_points, args.tasks, args.zoom_radius_m):
            image_paths.append(out_path)

    make_contact_sheet(image_paths, args.out_dir / "contact_sheet.png")
    print(f"Wrote {len(image_paths)} per-UUID PNGs to {args.out_dir}")
    print(f"Wrote contact sheet to {args.out_dir / 'contact_sheet.png'}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete UUIDs")


if __name__ == "__main__":
    main()
