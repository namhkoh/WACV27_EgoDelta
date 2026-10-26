from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

from .primitives import GaussianSet2D, normalize_quaternion, quaternion_to_matrix


@dataclass(frozen=True)
class Camera:
    id: int
    img_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    center: torch.Tensor
    rotation: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.center.device

    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        rotation = self.rotation.to(device=points.device, dtype=points.dtype)
        center = self.center.to(device=points.device, dtype=points.dtype)
        return (points - center) @ rotation

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pc = self.world_to_camera(points)
        z = pc[:, 2]
        u = self.fx * pc[:, 0] / z.clamp_min(1.0e-12) + self.cx
        v = self.fy * pc[:, 1] / z.clamp_min(1.0e-12) + self.cy
        return u, v, z

    def to(self, device: torch.device | str) -> "Camera":
        return Camera(
            id=self.id,
            img_name=self.img_name,
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            center=self.center.to(device),
            rotation=self.rotation.to(device),
        )


@dataclass(frozen=True)
class ScenePair:
    name: str
    pre_model: Path
    post_model: Path
    pre_ply: Path
    post_ply: Path
    pre_cameras: Path
    post_cameras: Path


def _field(vertex: np.ndarray, name: str) -> np.ndarray:
    if name not in vertex.dtype.names:
        raise KeyError(f"PLY vertex field '{name}' is missing")
    return np.asarray(vertex[name], dtype=np.float32)


def _stack_fields(vertex: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([_field(vertex, name) for name in names], axis=1)


def load_2dgs_ply(path: Path, device: torch.device | str = "cpu") -> GaussianSet2D:
    path = Path(path)
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())

    mu = torch.from_numpy(_stack_fields(vertex, ["x", "y", "z"])).float()
    dc = torch.from_numpy(_stack_fields(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])).float()
    scale_raw = torch.from_numpy(_stack_fields(vertex, ["scale_0", "scale_1"])).float()
    quat_raw = torch.from_numpy(_stack_fields(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])).float()
    opacity_raw = torch.from_numpy(_field(vertex, "opacity")).float()

    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        sh_flat = torch.from_numpy(_stack_fields(vertex, rest_names)).float()
        sh_rest = sh_flat.reshape(len(vertex), -1, 3)
    else:
        sh_rest = torch.empty((len(vertex), 0, 3), dtype=torch.float32)

    quat = normalize_quaternion(quat_raw)
    rotation = quaternion_to_matrix(quat)
    normal = rotation[:, :, 2]

    if {"nx", "ny", "nz"}.issubset(names):
        stored_normal = torch.from_numpy(_stack_fields(vertex, ["nx", "ny", "nz"])).float()
        norm = stored_normal.norm(dim=-1, keepdim=True)
        valid = norm.squeeze(-1) > 1.0e-8
        normal[valid] = stored_normal[valid] / norm[valid]

    g = GaussianSet2D(
        mu=mu,
        rotation=rotation,
        quaternion=quat,
        scales=torch.exp(scale_raw),
        normal=normal,
        opacity=torch.sigmoid(opacity_raw),
        dc_color=dc,
        sh_rest=sh_rest,
        source_indices=torch.arange(len(vertex), dtype=torch.long),
    )
    return g.to(device)


def _camera_from_json(item: dict[str, Any], device: torch.device | str) -> Camera:
    width = int(item["width"])
    height = int(item["height"])
    fx = float(item["fx"]) if "fx" in item else float(_focal_from_fov(item.get("fovx"), width))
    fy = float(item["fy"]) if "fy" in item else float(_focal_from_fov(item.get("fovy"), height))
    cx = float(item.get("cx", width / 2.0))
    cy = float(item.get("cy", height / 2.0))
    center = torch.tensor(item["position"], dtype=torch.float32, device=device)
    rotation = torch.tensor(item["rotation"], dtype=torch.float32, device=device)
    return Camera(
        id=int(item.get("id", 0)),
        img_name=str(item.get("img_name", item.get("image_name", ""))),
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        center=center,
        rotation=rotation,
    )


def _focal_from_fov(fov: float | None, pixels: int) -> float:
    if fov is None:
        raise KeyError("Camera JSON must contain fx/fy or fovx/fovy")
    return 0.5 * pixels / np.tan(0.5 * float(fov))


def load_cameras(path: Path, device: torch.device | str = "cpu") -> list[Camera]:
    path = Path(path)
    if path.is_dir():
        if (path / "cameras.json").exists():
            path = path / "cameras.json"
        elif (path / "sparse" / "0").exists():
            path = path / "sparse" / "0"

    if path.is_file() and path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "cameras" in data:
            data = data["cameras"]
        cameras = [_camera_from_json(item, device) for item in data]
        if not cameras:
            raise ValueError(f"No cameras found in {path}")
        return cameras

    if path.is_dir():
        raise NotImplementedError(
            f"COLMAP directory loading is not implemented for {path}; export cameras.json first."
        )

    raise FileNotFoundError(f"Camera path does not exist: {path}")


def select_point_cloud(model_dir: Path, preferred_iteration: int | None = None) -> Path:
    model_dir = Path(model_dir)
    candidates = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        direct = model_dir / "point_cloud.ply"
        if direct.exists():
            return direct
        raise FileNotFoundError(f"No point_cloud.ply found under {model_dir}")

    def iteration(path: Path) -> int:
        match = re.search(r"iteration_(\d+)", str(path))
        return int(match.group(1)) if match else -1

    if preferred_iteration is not None:
        for candidate in candidates:
            if iteration(candidate) == preferred_iteration:
                return candidate
    return max(candidates, key=iteration)


def make_scene_pair(pre_model: Path, post_model: Path) -> ScenePair:
    pre_model = Path(pre_model)
    post_model = Path(post_model)
    return ScenePair(
        name=pre_model.name,
        pre_model=pre_model,
        post_model=post_model,
        pre_ply=select_point_cloud(pre_model, preferred_iteration=30000),
        post_ply=select_point_cloud(post_model),
        pre_cameras=pre_model / "cameras.json",
        post_cameras=post_model / "cameras.json",
    )


def discover_ready_pairs(pre_root: Path, post_root: Path) -> list[ScenePair]:
    pre_root = Path(pre_root)
    post_root = Path(post_root)
    pairs: list[ScenePair] = []
    for pre_model in sorted(p for p in pre_root.iterdir() if p.is_dir()):
        post_model = post_root / pre_model.name
        if not post_model.is_dir():
            continue
        try:
            pair = make_scene_pair(pre_model, post_model)
        except FileNotFoundError:
            continue
        if pair.pre_cameras.exists() and pair.post_cameras.exists():
            pairs.append(pair)
    return pairs

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.optimize import linear_sum_assignment

from .primitives import GaussianSet2D


TASK_BY_SUFFIX = {
    "leaf": "deleafing",
    "tomato": "harvesting",
}


@dataclass(frozen=True)
class TargetMatch:
    target_index: int
    gt_point: list[float]
    selected_point: list[float]
    source_index: int
    score: float
    distance_m: float
    within_tolerance: bool


def parse_gt_markdown(path: Path) -> dict[str, dict[str, np.ndarray]]:
    entries: dict[str, dict[str, list[list[float]]]] = {}
    current_uuid: str | None = None
    current_task: str | None = None
    header_re = re.compile(r"^\d+\.\s+([0-9a-fA-F-]+)\.vrs$")

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = header_re.match(line)
        if header:
            current_uuid = header.group(1).lower()
            entries[current_uuid] = {"deleafing": [], "harvesting": []}
            current_task = None
            continue
        if line in {"deleafing", "harvesting"}:
            current_task = line
            continue
        if line.startswith("[") and current_uuid and current_task:
            point = json.loads(line)
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"Invalid GT point in {path}: {line}")
            entries[current_uuid][current_task].append([float(x) for x in point])

    return {
        scene_uuid: {
            task: np.asarray(points, dtype=np.float32).reshape((-1, 3))
            for task, points in tasks.items()
        }
        for scene_uuid, tasks in entries.items()
    }


def find_scene_targets(
    scene_name: str,
    gt_entries: dict[str, dict[str, np.ndarray]],
) -> tuple[str, str, np.ndarray] | None:
    match = re.match(r"^egodelta_([0-9a-fA-F]+)_(leaf|tomato)$", scene_name)
    if not match:
        return None
    compact_prefix = match.group(1).lower()
    task = TASK_BY_SUFFIX[match.group(2)]
    scene_matches = [
        scene_uuid
        for scene_uuid in gt_entries
        if scene_uuid.replace("-", "").startswith(compact_prefix)
    ]
    if not scene_matches:
        return None
    scene_uuid = scene_matches[0]
    return scene_uuid, task, gt_entries[scene_uuid][task]


def match_targets_to_points(
    gt_points: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    source_indices: np.ndarray,
    tolerance_m: float = 0.03,
) -> list[TargetMatch]:
    if len(gt_points) == 0:
        return []
    if len(points) < len(gt_points):
        raise ValueError(
            f"Need at least {len(gt_points)} candidate points, found {len(points)}"
        )

    distances = np.linalg.norm(gt_points[:, None, :] - points[None, :, :], axis=2)
    target_rows, point_cols = linear_sum_assignment(distances)
    ordered = sorted(zip(target_rows.tolist(), point_cols.tolist()), key=lambda x: x[0])
    matches: list[TargetMatch] = []
    for target_idx, point_idx in ordered:
        distance = float(distances[target_idx, point_idx])
        matches.append(
            TargetMatch(
                target_index=int(target_idx),
                gt_point=gt_points[target_idx].astype(float).tolist(),
                selected_point=points[point_idx].astype(float).tolist(),
                source_index=int(source_indices[point_idx]),
                score=float(scores[point_idx]),
                distance_m=distance,
                within_tolerance=distance <= tolerance_m,
            )
        )
    return matches


def match_removed_targets(
    g_pre: GaussianSet2D,
    removed_mask: torch.BoolTensor,
    scores: torch.Tensor,
    gt_points: np.ndarray,
    min_score: float = 0.5,
    tolerance_m: float = 0.03,
) -> list[TargetMatch]:
    candidate_mask = removed_mask & (scores >= float(min_score))
    idx = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    points = g_pre.mu[idx].detach().cpu().numpy().astype(np.float32)
    candidate_scores = scores[idx].detach().cpu().numpy().astype(np.float32)
    source_indices = g_pre.source_indices[idx].detach().cpu().numpy().astype(np.int64)
    return match_targets_to_points(
        gt_points=gt_points,
        points=points,
        scores=candidate_scores,
        source_indices=source_indices,
        tolerance_m=tolerance_m,
    )


def summarize_matches(matches: list[TargetMatch], gt_count: int) -> dict[str, float | int]:
    if not matches:
        return {
            "gt_count": int(gt_count),
            "selected_target_count": 0,
            "mean_distance_m": float("inf"),
            "max_distance_m": float("inf"),
            "within_tolerance_count": 0,
        }
    distances = np.asarray([match.distance_m for match in matches], dtype=np.float32)
    return {
        "gt_count": int(gt_count),
        "selected_target_count": int(len(matches)),
        "mean_distance_m": float(distances.mean()),
        "max_distance_m": float(distances.max()),
        "within_tolerance_count": int(sum(match.within_tolerance for match in matches)),
    }


def save_target_matches(path: Path, matches: list[TargetMatch], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "matches": [asdict(match) for match in matches],
    }
    path.write_text(json.dumps(payload, indent=2))


def save_selected_targets_ply(path: Path, matches: list[TargetMatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("score", "f4"),
        ("target_index", "i4"),
        ("source_index", "i4"),
        ("distance_m", "f4"),
    ]
    vertices = np.empty(len(matches), dtype=dtype)
    for i, match in enumerate(matches):
        point = np.asarray(match.selected_point, dtype=np.float32)
        vertices["x"][i], vertices["y"][i], vertices["z"][i] = point
        vertices["red"][i] = 255 if not match.within_tolerance else 0
        vertices["green"][i] = 255 if match.within_tolerance else 0
        vertices["blue"][i] = 0
        vertices["score"][i] = np.float32(match.score)
        vertices["target_index"][i] = np.int32(match.target_index)
        vertices["source_index"][i] = np.int32(match.source_index)
        vertices["distance_m"][i] = np.float32(match.distance_m)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)

