"""Paper-faithful GS-DIFF reimplementation for EgoDelta.

Equations follow arXiv:2605.07203v2, Algorithm 1. The loader accepts both
ordinary 3DGS (three scales) and EgoDelta 2DGS surfels (two scales); a 2D
surfel has zero covariance along its normal for scoring and a tiny positive
thickness only when passed to the standard 3DGS rasterizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch

from .benchmark import SceneSpec, select_point_cloud
from .outputs import write_prediction, write_scene_metadata


@dataclass(frozen=True)
class Camera:
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    center: torch.Tensor
    rotation_c2w: torch.Tensor

    def to(self, device: torch.device | str) -> "Camera":
        return Camera(
            self.image_name,
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.center.to(device),
            self.rotation_c2w.to(device),
        )

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation = self.rotation_c2w.to(points)
        center = self.center.to(points)
        camera_points = (points - center) @ rotation
        z = camera_points[:, 2]
        safe_z = torch.where(z.abs() < 1.0e-12, torch.full_like(z, 1.0e-12), z)
        u = self.fx * camera_points[:, 0] / safe_z + self.cx
        v = self.fy * camera_points[:, 1] / safe_z + self.cy
        return u, v, z


@dataclass
class GaussianSet:
    means: torch.Tensor
    quaternions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    normals: torch.Tensor
    opacity: torch.Tensor
    dc_color: torch.Tensor
    source_indices: torch.Tensor
    is_surfel: bool

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def covariance(self) -> torch.Tensor:
        diagonal = torch.diag_embed(self.scales.square())
        return self.rotations @ diagonal @ self.rotations.transpose(-1, -2)

    def __getitem__(self, item: Any) -> "GaussianSet":
        return GaussianSet(
            means=self.means[item],
            quaternions=self.quaternions[item],
            rotations=self.rotations[item],
            scales=self.scales[item],
            normals=self.normals[item],
            opacity=self.opacity[item],
            dc_color=self.dc_color[item],
            source_indices=self.source_indices[item],
            is_surfel=self.is_surfel,
        )


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(-1)
    matrix = torch.empty(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - w * z)
    matrix[..., 0, 2] = 2 * (x * z + w * y)
    matrix[..., 1, 0] = 2 * (x * y + w * z)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - w * x)
    matrix[..., 2, 0] = 2 * (x * z - w * y)
    matrix[..., 2, 1] = 2 * (y * z + w * x)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def load_gaussians(path: str | Path, device: torch.device | str = "cuda") -> GaussianSet:
    vertex = PlyData.read(Path(path))["vertex"].data
    names = set(vertex.dtype.names or ())

    def stack(prefix: str) -> np.ndarray:
        fields = sorted((x for x in names if x.startswith(prefix)), key=lambda x: int(x.rsplit("_", 1)[-1]))
        if not fields:
            raise KeyError(f"No {prefix}* fields in {path}")
        return np.stack([np.asarray(vertex[x], dtype=np.float32) for x in fields], axis=1)

    means = np.stack([np.asarray(vertex[x], dtype=np.float32) for x in ("x", "y", "z")], axis=1)
    dc = np.stack([np.asarray(vertex[x], dtype=np.float32) for x in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=1)
    scale_raw = stack("scale_")
    is_surfel = scale_raw.shape[1] == 2
    if is_surfel:
        # Exact 2D surfel covariance for primitive scoring.
        scale_raw = np.concatenate([scale_raw, np.full((len(scale_raw), 1), -np.inf, np.float32)], axis=1)
    if scale_raw.shape[1] != 3:
        raise ValueError(f"Expected two or three scale fields in {path}, got {scale_raw.shape[1]}")
    quaternion = torch.from_numpy(stack("rot_")).float().to(device)
    if quaternion.shape[1] != 4:
        raise ValueError(f"Expected WXYZ quaternion in {path}")
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    rotations = quaternion_to_matrix(quaternion)
    scales = torch.exp(torch.from_numpy(scale_raw).float().to(device))
    smallest_axis = scales.argmin(dim=-1)
    normals = torch.gather(rotations, 2, smallest_axis[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    if {"nx", "ny", "nz"}.issubset(names):
        stored = torch.from_numpy(
            np.stack([np.asarray(vertex[x], dtype=np.float32) for x in ("nx", "ny", "nz")], axis=1)
        ).to(device)
        valid = stored.norm(dim=-1) > 1.0e-8
        normals[valid] = torch.nn.functional.normalize(stored[valid], dim=-1)
    opacity_raw = torch.from_numpy(np.asarray(vertex["opacity"], dtype=np.float32)).to(device)
    return GaussianSet(
        means=torch.from_numpy(means).to(device),
        quaternions=quaternion,
        rotations=rotations,
        scales=scales,
        normals=normals,
        opacity=torch.sigmoid(opacity_raw),
        dc_color=torch.from_numpy(dc).to(device),
        source_indices=torch.arange(len(vertex), device=device),
        is_surfel=is_surfel,
    )


def load_cameras(path: str | Path, device: torch.device | str = "cuda") -> list[Camera]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = []
    for row in rows:
        width, height = int(row["width"]), int(row["height"])
        cameras.append(
            Camera(
                image_name=str(row.get("img_name", row.get("image_name", ""))),
                width=width,
                height=height,
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row.get("cx", width / 2.0)),
                cy=float(row.get("cy", height / 2.0)),
                center=torch.tensor(row["position"], dtype=torch.float32, device=device),
                rotation_c2w=torch.tensor(row["rotation"], dtype=torch.float32, device=device),
            )
        )
    return cameras


def camera_from_task(task: dict[str, Any], device: torch.device | str = "cuda") -> Camera:
    row = task["camera"]
    return Camera(
        image_name=task["image_name"],
        width=int(row["width"]),
        height=int(row["height"]),
        fx=float(row["fx"]),
        fy=float(row["fy"]),
        cx=float(row["cx"]),
        cy=float(row["cy"]),
        center=torch.tensor(row["camera_center_world"], dtype=torch.float32, device=device),
        rotation_c2w=torch.tensor(row["rotation_camera_to_world"], dtype=torch.float32, device=device),
    )


def finite_mask(gaussians: GaussianSet) -> torch.Tensor:
    tensors = (
        gaussians.means,
        gaussians.quaternions,
        gaussians.scales,
        gaussians.normals,
        gaussians.opacity[:, None],
        gaussians.dc_color,
    )
    mask = torch.ones(len(gaussians), dtype=torch.bool, device=gaussians.device)
    for tensor in tensors:
        mask &= torch.isfinite(tensor).all(dim=tuple(range(1, tensor.ndim)))
    return mask


def shared_frustum_mask(
    gaussians: GaussianSet,
    other_cameras: list[Camera],
    *,
    near: float = 0.01,
    far: float = 100.0,
    chunk_size: int = 50_000,
) -> torch.Tensor:
    visible_any = torch.zeros(len(gaussians), dtype=torch.bool, device=gaussians.device)
    for camera in other_cameras:
        for start in range(0, len(gaussians), chunk_size):
            end = min(start + chunk_size, len(gaussians))
            u, v, z = camera.project(gaussians.means[start:end])
            visible_any[start:end] |= (
                (z >= near) & (z <= far) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            )
    return visible_any


def nearest_neighbor_drift(a: GaussianSet, b: GaussianSet, quantile: float = 0.75) -> tuple[float, float]:
    tree = cKDTree(b.means.detach().cpu().numpy())
    _, indices_np = tree.query(a.means.detach().cpu().numpy(), k=1, workers=-1)
    indices = torch.as_tensor(indices_np, dtype=torch.long, device=a.device)
    displacement = b.means[indices] - a.means
    signed_normal = (displacement * a.normals).sum(-1)
    normal = signed_normal.abs()
    tangent = (displacement - signed_normal[:, None] * a.normals).norm(dim=-1)
    # GS-DIFF Eq. 1 defines normal drift as an absolute displacement before
    # applying the upper-quartile statistic in Eq. 2.
    return float(torch.quantile(normal, quantile).square()), float(torch.quantile(tangent, quantile).square())


def symmetric_drift(a: GaussianSet, b: GaussianSet, quantile: float = 0.75) -> tuple[float, float]:
    an, at = nearest_neighbor_drift(a, b, quantile)
    bn, bt = nearest_neighbor_drift(b, a, quantile)
    return 0.5 * (an + bn), 0.5 * (at + bt)


def compute_fim(
    gaussians: GaussianSet,
    cameras: list[Camera],
    *,
    near: float = 0.01,
    far: float = 100.0,
    chunk_size: int = 50_000,
) -> torch.Tensor:
    fim = torch.zeros((len(gaussians), 3, 3), device=gaussians.device, dtype=gaussians.means.dtype)
    identity = torch.eye(3, device=gaussians.device, dtype=gaussians.means.dtype)
    for camera in cameras:
        center = camera.center.to(gaussians.means)
        for start in range(0, len(gaussians), chunk_size):
            end = min(start + chunk_size, len(gaussians))
            points = gaussians.means[start:end]
            ray = points - center
            distance2 = ray.square().sum(-1).clamp_min(1.0e-12)
            direction = ray / distance2.sqrt()[:, None]
            u, v, z = camera.project(points)
            visible = (z >= near) & (z <= far) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            projection = identity - direction[:, :, None] * direction[:, None, :]
            fim[start:end] += projection * (visible / distance2)[:, None, None]
    return fim


def effective_covariance(
    gaussians: GaussianSet,
    fim: torch.Tensor,
    normal_drift2: float,
    tangent_drift2: float,
) -> tuple[torch.Tensor, float]:
    identity = torch.eye(3, device=gaussians.device, dtype=gaussians.means.dtype)
    nn_t = gaussians.normals[:, :, None] * gaussians.normals[:, None, :]
    ambiguity = tangent_drift2 * identity + (normal_drift2 - tangent_drift2) * nn_t
    inflated = gaussians.covariance + ambiguity
    fim_pinv = torch.linalg.pinv(fim, hermitian=True)
    numerator = torch.diagonal(inflated, dim1=-2, dim2=-1).sum(-1).median()
    denominator_values = torch.diagonal(fim_pinv, dim1=-2, dim2=-1).sum(-1)
    valid = torch.isfinite(denominator_values) & (denominator_values > 1.0e-12)
    scale = float((numerator / denominator_values[valid].median().clamp_min(1.0e-12)).item()) if valid.any() else 0.0
    return inflated + scale * fim_pinv, scale


def directional_color_bandwidth(
    a: GaussianSet,
    b: GaussianSet,
    covariance_a: torch.Tensor,
    covariance_b: torch.Tensor,
) -> float:
    tree = cKDTree(b.means.detach().cpu().numpy())
    _, indices_np = tree.query(a.means.detach().cpu().numpy(), k=1, workers=-1)
    indices = torch.as_tensor(indices_np, dtype=torch.long, device=a.device)
    displacement = a.means - b.means[indices]
    pair_covariance = covariance_a + covariance_b[indices]
    solved = torch.linalg.solve(
        pair_covariance + 1.0e-8 * torch.eye(3, device=a.device), displacement[:, :, None]
    ).squeeze(-1)
    geometry_weight = torch.exp(-0.5 * (displacement * solved).sum(-1).clamp_min(0))
    color_distance2 = (a.dc_color - b.dc_color[indices]).square().sum(-1)
    # GS-DIFF Eq. 9 takes the median of the geometry-weighted squared color
    # differences. This is not the weighted median of the raw differences.
    weighted_distance2 = geometry_weight * color_distance2
    return float(torch.quantile(weighted_distance2, 0.5).clamp_min(1.0e-8))


def symmetric_color_bandwidth(
    a: GaussianSet,
    b: GaussianSet,
    covariance_a: torch.Tensor,
    covariance_b: torch.Tensor,
) -> float:
    return 0.5 * (
        directional_color_bandwidth(a, b, covariance_a, covariance_b)
        + directional_color_bandwidth(b, a, covariance_b, covariance_a)
    )


def primitive_scores(
    a: GaussianSet,
    b: GaussianSet,
    covariance_a: torch.Tensor,
    covariance_b: torch.Tensor,
    color_bandwidth2: float,
    *,
    eta: float = 3.0,
    query_chunk_size: int = 1024,
    max_neighbors: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Batched cuSolver eigendecomposition can fail for very large collections
    # of tiny matrices on otherwise valid inputs; this one-off radius estimate
    # is inexpensive and substantially more robust on CPU.
    largest_eigenvalue = torch.linalg.eigvalsh(covariance_a.detach().cpu()).amax(-1).to(a.device)
    radii = eta * largest_eigenvalue.clamp_min(1.0e-12).sqrt()
    footprint2 = torch.diagonal(covariance_a, dim1=-2, dim2=-1).sum(-1).clamp_min(1.0e-12)
    local_color_bandwidth2 = color_bandwidth2 * torch.maximum(
        footprint2 / footprint2.median().clamp_min(1.0e-12), torch.ones_like(footprint2)
    )
    points_a = a.means.detach().cpu().numpy()
    points_b = b.means.detach().cpu().numpy()
    tree = cKDTree(points_b)
    best_geo = torch.zeros(len(a), dtype=a.means.dtype, device=a.device)
    best_app = torch.zeros_like(best_geo)
    identity = torch.eye(3, dtype=a.means.dtype, device=a.device)

    for start in range(0, len(a), query_chunk_size):
        end = min(start + query_chunk_size, len(a))
        neighbors = tree.query_ball_point(
            points_a[start:end], radii[start:end].detach().cpu().numpy(), workers=-1
        )
        query_parts: list[np.ndarray] = []
        neighbor_parts: list[np.ndarray] = []
        for offset, candidates in enumerate(neighbors):
            if max_neighbors and len(candidates) > max_neighbors:
                array = np.asarray(candidates, dtype=np.int64)
                distance2 = ((points_b[array] - points_a[start + offset]) ** 2).sum(axis=1)
                candidates = array[np.argpartition(distance2, max_neighbors - 1)[:max_neighbors]].tolist()
            if candidates:
                query_parts.append(np.full(len(candidates), offset, dtype=np.int64))
                neighbor_parts.append(np.asarray(candidates, dtype=np.int64))
        if not query_parts:
            continue
        local_query = torch.from_numpy(np.concatenate(query_parts)).to(a.device)
        neighbor_index = torch.from_numpy(np.concatenate(neighbor_parts)).to(a.device)
        global_query = local_query + start
        displacement = a.means[global_query] - b.means[neighbor_index]
        pair_covariance = covariance_a[global_query] + covariance_b[neighbor_index] + 1.0e-8 * identity
        solved = torch.linalg.solve(pair_covariance, displacement[:, :, None]).squeeze(-1)
        geo = torch.exp(-0.5 * (displacement * solved).sum(-1).clamp_min(0))
        color_distance2 = (a.dc_color[global_query] - b.dc_color[neighbor_index]).square().sum(-1)
        app = torch.exp(-color_distance2 / (2 * local_color_bandwidth2[global_query].clamp_min(1.0e-8)))
        local_geo = torch.zeros(end - start, dtype=geo.dtype, device=a.device)
        local_app = torch.zeros_like(local_geo)
        local_geo.scatter_reduce_(0, local_query, geo, reduce="amax", include_self=True)
        local_app.scatter_reduce_(0, local_query, app, reduce="amax", include_self=True)
        best_geo[start:end] = local_geo
        best_app[start:end] = local_app
    return 1.0 - best_geo, 1.0 - best_app


def observability_confidence(fim: torch.Tensor, quantile: float = 0.25) -> torch.Tensor:
    trace = torch.diagonal(fim, dim1=-2, dim2=-1).sum(-1).clamp_min(1.0e-12)
    reference = torch.quantile(trace, quantile).clamp_min(1.0e-12)
    return torch.sigmoid(torch.log(trace) - torch.log(reference))


def aggregate_scores(geometry: torch.Tensor, appearance: torch.Tensor, fim: torch.Tensor) -> torch.Tensor:
    return observability_confidence(fim) * (geometry + appearance).clamp(max=1.0)


def render_scalar(
    gaussians: GaussianSet,
    scores: torch.Tensor,
    camera: Camera,
    *,
    surfel_thickness_ratio: float = 1.0e-3,
) -> torch.Tensor:
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError("GS-DIFF rendering requires gsplat==1.4.0 (run setup_envs.sh)") from exc
    camera = camera.to(gaussians.device)
    rotation_w2c = camera.rotation_c2w.T
    view = torch.eye(4, device=gaussians.device, dtype=gaussians.means.dtype)
    view[:3, :3] = rotation_w2c
    view[:3, 3] = -(rotation_w2c @ camera.center)
    intrinsics = torch.tensor(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        device=gaussians.device,
        dtype=gaussians.means.dtype,
    )
    scales = gaussians.scales.clone()
    if gaussians.is_surfel:
        scales[:, 2] = scales[:, :2].amin(-1).clamp_min(1.0e-6) * surfel_thickness_ratio
    rendered, _, _ = rasterization(
        means=gaussians.means,
        quats=gaussians.quaternions,
        scales=scales.clamp_min(1.0e-8),
        opacities=gaussians.opacity,
        colors=scores[:, None].repeat(1, 3),
        viewmats=view[None],
        Ks=intrinsics[None],
        width=camera.width,
        height=camera.height,
        backgrounds=torch.zeros((1, 3), device=gaussians.device),
        packed=False,
        sh_degree=None,
    )
    return rendered[0, :, :, 0].clamp(0.0, 1.0)


def score_gaussian_pair(
    reference_ply: str | Path,
    inference_ply: str | Path,
    reference_cameras_json: str | Path,
    inference_cameras_json: str | Path,
    *,
    device: str = "cuda",
    max_neighbors: int | None = None,
    max_primitives: int | None = None,
) -> tuple[GaussianSet, GaussianSet, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run the paper's bidirectional primitive scoring on two reconstructions."""
    torch_device = torch.device(device)
    print(f"  loading reference/inference primitives on {torch_device}", flush=True)
    reference = load_gaussians(reference_ply, torch_device)
    inference = load_gaussians(inference_ply, torch_device)
    cameras_reference = load_cameras(reference_cameras_json, torch_device)
    cameras_inference = load_cameras(inference_cameras_json, torch_device)
    reference = reference[finite_mask(reference)]
    inference = inference[finite_mask(inference)]
    if max_primitives:
        reference = _deterministic_sample(reference, max_primitives, seed=0)
        inference = _deterministic_sample(inference, max_primitives, seed=1)
    raw_counts = (len(reference), len(inference))
    reference = reference[shared_frustum_mask(reference, cameras_inference)]
    inference = inference[shared_frustum_mask(inference, cameras_reference)]
    if not len(reference) or not len(inference):
        raise RuntimeError("Shared-frustum filtering emptied a reconstruction")
    print(
        f"  shared frustum: {len(reference):,} reference / {len(inference):,} inference primitives",
        flush=True,
    )

    print("  estimating reconstruction drift, FIM and adaptive bandwidths", flush=True)
    normal_drift2, tangent_drift2 = symmetric_drift(reference, inference)
    fim_reference = compute_fim(reference, cameras_reference)
    fim_inference = compute_fim(inference, cameras_inference)
    covariance_reference, fim_scale_reference = effective_covariance(
        reference, fim_reference, normal_drift2, tangent_drift2
    )
    covariance_inference, fim_scale_inference = effective_covariance(
        inference, fim_inference, normal_drift2, tangent_drift2
    )
    color_bandwidth2 = symmetric_color_bandwidth(
        reference, inference, covariance_reference, covariance_inference
    )
    print("  scoring reference-to-inference primitives", flush=True)
    geometry_reference, appearance_reference = primitive_scores(
        reference,
        inference,
        covariance_reference,
        covariance_inference,
        color_bandwidth2,
        max_neighbors=max_neighbors,
    )
    print("  scoring inference-to-reference primitives", flush=True)
    geometry_inference, appearance_inference = primitive_scores(
        inference,
        reference,
        covariance_inference,
        covariance_reference,
        color_bandwidth2,
        max_neighbors=max_neighbors,
    )
    score_reference = aggregate_scores(geometry_reference, appearance_reference, fim_reference)
    score_inference = aggregate_scores(geometry_inference, appearance_inference, fim_inference)
    diagnostics = {
        "raw_or_sampled_primitive_counts": {"reference": raw_counts[0], "inference": raw_counts[1]},
        "shared_frustum_primitive_counts": {"reference": len(reference), "inference": len(inference)},
        "normal_drift": normal_drift2**0.5,
        "tangent_drift": tangent_drift2**0.5,
        "color_bandwidth": color_bandwidth2**0.5,
        "fim_injection_scale": {
            "reference": fim_scale_reference,
            "inference": fim_scale_inference,
        },
        "max_neighbors": max_neighbors,
        "max_primitives_debug_only": max_primitives,
    }
    return reference, inference, score_reference, score_inference, diagnostics


def run_gsdiff_scene(
    scene: SceneSpec,
    output_root: str | Path,
    *,
    device: str = "cuda",
    threshold: float = 0.5,
    max_neighbors: int | None = None,
    max_primitives: int | None = None,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    pre, post, score_pre, score_post, pair_diagnostics = score_gaussian_pair(
        select_point_cloud(scene.pre_model),
        select_point_cloud(scene.post_model),
        scene.pre_model / "cameras.json",
        scene.post_model / "cameras.json",
        device=device,
        max_neighbors=max_neighbors,
        max_primitives=max_primitives,
    )

    print(f"  rendering {len(scene.queries)} frozen benchmark views", flush=True)
    predictions = []
    for task in scene.queries:
        camera = camera_from_task(task, torch_device)
        pre_map = render_scalar(pre, score_pre, camera)
        post_map = render_scalar(post, score_post, camera)
        final = torch.maximum(pre_map, post_map).detach().cpu().numpy()
        paths = write_prediction(output_root, scene.scene_id, task["image_name"], final, threshold=threshold)
        predictions.append({"task_id": task["task_id"], "image_name": task["image_name"], **paths})

    diagnostics = {
        "baseline": "gsdiff_reimplementation",
        "paper": "arXiv:2605.07203v2",
        "implementation": "Algorithm 1, adapted to existing EgoDelta 2DGS surfel reconstructions",
        "threshold": threshold,
        **pair_diagnostics,
        "predictions": predictions,
    }
    write_scene_metadata(output_root, scene.scene_id, diagnostics)
    print("  complete", flush=True)
    return diagnostics


def _deterministic_sample(gaussians: GaussianSet, count: int, seed: int) -> GaussianSet:
    if len(gaussians) <= count:
        return gaussians
    generator = torch.Generator(device=gaussians.device).manual_seed(seed)
    indices = torch.randperm(len(gaussians), generator=generator, device=gaussians.device)[:count].sort().values
    return gaussians[indices]
