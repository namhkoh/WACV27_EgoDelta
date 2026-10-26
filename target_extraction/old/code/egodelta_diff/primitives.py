from __future__ import annotations

from dataclasses import dataclass

import torch

SH_C0 = 0.28209479177387814


@dataclass
class GaussianSet2D:
    mu: torch.Tensor
    rotation: torch.Tensor
    quaternion: torch.Tensor
    scales: torch.Tensor
    normal: torch.Tensor
    opacity: torch.Tensor
    dc_color: torch.Tensor
    sh_rest: torch.Tensor
    source_indices: torch.Tensor

    def __len__(self) -> int:
        return int(self.mu.shape[0])

    def __getitem__(self, item) -> "GaussianSet2D":
        return GaussianSet2D(
            mu=self.mu[item],
            rotation=self.rotation[item],
            quaternion=self.quaternion[item],
            scales=self.scales[item],
            normal=self.normal[item],
            opacity=self.opacity[item],
            dc_color=self.dc_color[item],
            sh_rest=self.sh_rest[item],
            source_indices=self.source_indices[item],
        )

    def to(self, device: torch.device | str) -> "GaussianSet2D":
        return GaussianSet2D(
            mu=self.mu.to(device),
            rotation=self.rotation.to(device),
            quaternion=self.quaternion.to(device),
            scales=self.scales.to(device),
            normal=self.normal.to(device),
            opacity=self.opacity.to(device),
            dc_color=self.dc_color.to(device),
            sh_rest=self.sh_rest.to(device),
            source_indices=self.source_indices.to(device),
        )

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @property
    def cov_3d(self) -> torch.Tensor:
        s = torch.zeros(len(self), 3, 3, device=self.device, dtype=self.mu.dtype)
        s[:, 0, 0] = self.scales[:, 0].square()
        s[:, 1, 1] = self.scales[:, 1].square()
        return self.rotation @ s @ self.rotation.transpose(-1, -2)

    @property
    def rgb01(self) -> torch.Tensor:
        return torch.clamp(self.dc_color * SH_C0 + 0.5, 0.0, 1.0)


def normalize_quaternion(q: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    out = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    out[..., 0, 0] = ww + xx - yy - zz
    out[..., 0, 1] = 2.0 * (xy - wz)
    out[..., 0, 2] = 2.0 * (xz + wy)
    out[..., 1, 0] = 2.0 * (xy + wz)
    out[..., 1, 1] = ww - xx + yy - zz
    out[..., 1, 2] = 2.0 * (yz - wx)
    out[..., 2, 0] = 2.0 * (xz - wy)
    out[..., 2, 1] = 2.0 * (yz + wx)
    out[..., 2, 2] = ww - xx - yy + zz
    return out


def build_effective_covariance(
    g: GaussianSet2D,
    h: torch.Tensor,
    u_n2: float,
    u_t2: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    i3 = torch.eye(3, device=g.device, dtype=g.mu.dtype).expand(len(g), -1, -1)
    nn_t = g.normal.unsqueeze(-1) * g.normal.unsqueeze(-2)
    u = u_t2 * i3 + (u_n2 - u_t2) * nn_t
    sigma_tilde = g.cov_3d + u

    h_pinv = torch.linalg.pinv(h)
    trace_sigma = torch.diagonal(sigma_tilde, dim1=-2, dim2=-1).sum(-1)
    trace_pinv = torch.diagonal(h_pinv, dim1=-2, dim2=-1).sum(-1)
    finite = torch.isfinite(trace_pinv) & torch.isfinite(trace_sigma) & (trace_pinv > eps)
    if finite.any():
        scale = (trace_sigma[finite].median() / trace_pinv[finite].median().clamp_min(eps)).item()
    else:
        scale = 0.0

    sigma_eff = sigma_tilde + scale * h_pinv
    diag = torch.diagonal(sigma_eff, dim1=-2, dim2=-1).sum(-1)
    stats = {
        "fim_injection_scale": float(scale),
        "sigma_eff_trace_median": float(diag.median().detach().cpu()),
        "sigma_eff_trace_q01": float(torch.quantile(diag, 0.01).detach().cpu()),
        "sigma_eff_trace_q99": float(torch.quantile(diag, 0.99).detach().cpu()),
    }
    return sigma_eff, stats


def sample_gaussians(g: GaussianSet2D, max_primitives: int | None, seed: int = 0) -> GaussianSet2D:
    if max_primitives is None or len(g) <= max_primitives:
        return g
    generator = torch.Generator(device=g.device)
    generator.manual_seed(seed)
    idx = torch.randperm(len(g), generator=generator, device=g.device)[:max_primitives]
    idx = torch.sort(idx).values
    return g[idx]

import torch



def compute_fim(
    g: GaussianSet2D,
    cameras: list[Camera],
    chunk_size: int = 50000,
    near: float = 0.01,
    far: float = 100.0,
) -> torch.Tensor:
    h = torch.zeros(len(g), 3, 3, device=g.device, dtype=g.mu.dtype)
    eye = torch.eye(3, device=g.device, dtype=g.mu.dtype)
    for camera in cameras:
        camera = camera.to(g.device)
        center = camera.center.to(device=g.device, dtype=g.mu.dtype)
        for start in range(0, len(g), chunk_size):
            end = min(start + chunk_size, len(g))
            mu = g.mu[start:end]
            ray = mu - center
            d2 = ray.square().sum(-1).clamp_min(1.0e-12)
            v_hat = ray / d2.sqrt().unsqueeze(-1)
            u, v, z = camera.project(mu)
            visible = (
                (z >= near)
                & (z <= far)
                & (u >= 0.0)
                & (u < camera.width)
                & (v >= 0.0)
                & (v < camera.height)
            )
            proj = eye.expand(end - start, -1, -1) - v_hat.unsqueeze(-1) * v_hat.unsqueeze(-2)
            h[start:end] += proj / d2.view(-1, 1, 1) * visible.float().view(-1, 1, 1)
    return h

import torch



def in_shared_frustum(
    g: GaussianSet2D,
    cameras_other: list[Camera],
    near: float = 0.01,
    far: float = 100.0,
    chunk_size: int = 50000,
) -> torch.BoolTensor:
    mask = torch.zeros(len(g), dtype=torch.bool, device=g.device)
    for camera in cameras_other:
        camera = camera.to(g.device)
        for start in range(0, len(g), chunk_size):
            end = min(start + chunk_size, len(g))
            if bool(mask[start:end].all()):
                continue
            u, v, z = camera.project(g.mu[start:end])
            visible = (
                (z >= near)
                & (z <= far)
                & (u >= 0.0)
                & (u < camera.width)
                & (v >= 0.0)
                & (v < camera.height)
            )
            mask[start:end] |= visible
    return mask

import numpy as np
import torch
from scipy.spatial import cKDTree



def _tree(points: torch.Tensor) -> cKDTree:
    return cKDTree(points.detach().cpu().numpy())


def estimate_geometric_drift(
    g_a: GaussianSet2D,
    g_b: GaussianSet2D,
    quantile: float = 0.75,
) -> tuple[float, float]:
    tree = _tree(g_b.mu)
    _, idx_np = tree.query(g_a.mu.detach().cpu().numpy(), k=1, workers=-1)
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=g_a.device)
    delta = g_b.mu[idx] - g_a.mu
    d_signed = (delta * g_a.normal).sum(-1)
    d_n = d_signed.abs()
    d_t = (delta - d_signed.unsqueeze(-1) * g_a.normal).norm(dim=-1)
    u_n2 = float(torch.quantile(d_n, quantile).detach().cpu().item() ** 2)
    u_t2 = float(torch.quantile(d_t, quantile).detach().cpu().item() ** 2)
    return u_n2, u_t2


def estimate_geometric_drift_symmetric(
    g_pre: GaussianSet2D,
    g_post: GaussianSet2D,
    quantile: float = 0.75,
) -> tuple[float, float, dict[str, float]]:
    un2_ab, ut2_ab = estimate_geometric_drift(g_pre, g_post, quantile)
    un2_ba, ut2_ba = estimate_geometric_drift(g_post, g_pre, quantile)
    un2 = 0.5 * (un2_ab + un2_ba)
    ut2 = 0.5 * (ut2_ab + ut2_ba)
    return un2, ut2, {
        "u_n": float(un2 ** 0.5),
        "u_t": float(ut2 ** 0.5),
        "u_n_pre_to_post": float(un2_ab ** 0.5),
        "u_t_pre_to_post": float(ut2_ab ** 0.5),
        "u_n_post_to_pre": float(un2_ba ** 0.5),
        "u_t_post_to_pre": float(ut2_ba ** 0.5),
    }


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    values = values.flatten()
    weights = weights.flatten().clamp_min(0)
    finite = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not finite.any():
        return torch.quantile(values[torch.isfinite(values)], q)
    values = values[finite]
    weights = weights[finite]
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = torch.cumsum(weights, dim=0)
    cutoff = q * cdf[-1]
    idx = torch.searchsorted(cdf, cutoff).clamp(max=len(values) - 1)
    return values[idx]


def estimate_color_bandwidth(
    g_a: GaussianSet2D,
    g_b: GaussianSet2D,
    sigma_eff_a: torch.Tensor,
    sigma_eff_b: torch.Tensor,
    quantile: float = 0.5,
    jitter: float = 1.0e-8,
) -> float:
    tree = _tree(g_b.mu)
    _, idx_np = tree.query(g_a.mu.detach().cpu().numpy(), k=1, workers=-1)
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=g_a.device)

    c_diff = (g_a.dc_color - g_b.dc_color[idx]).square().sum(-1)
    delta = g_a.mu - g_b.mu[idx]
    m = sigma_eff_a + sigma_eff_b[idx]
    eye = torch.eye(3, device=g_a.device, dtype=g_a.mu.dtype).expand_as(m)
    m = m + jitter * eye
    solved = torch.linalg.solve(m, delta.unsqueeze(-1)).squeeze(-1)
    d_maha = (delta * solved).sum(-1)
    weights = torch.exp(-0.5 * d_maha.clamp_min(0.0))
    value = weighted_quantile(c_diff, weights, quantile)
    return float(value.clamp_min(jitter).detach().cpu().item())


def tensor_histogram(values: torch.Tensor, bins: int = 10) -> dict[str, list[float]]:
    values_np = values.detach().cpu().float().numpy()
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values_np, bins=bins, range=(float(values_np.min()), float(values_np.max())))
    return {"bins": edges.tolist(), "counts": counts.astype(int).tolist()}

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm



def _select_neighbor_indices(
    center: torch.Tensor,
    candidate_indices: list[int],
    points_b: torch.Tensor,
    max_neighbors: int,
) -> torch.Tensor:
    idx = torch.as_tensor(candidate_indices, dtype=torch.long, device=points_b.device)
    if len(idx) <= max_neighbors:
        return idx
    d2 = (points_b[idx] - center).square().sum(-1)
    keep = torch.topk(d2, k=max_neighbors, largest=False).indices
    return idx[keep]


def score_primitives(
    g_a: GaussianSet2D,
    g_b: GaussianSet2D,
    sigma_eff_a: torch.Tensor,
    sigma_eff_b: torch.Tensor,
    sigma_c2: float,
    eta: float = 3.0,
    max_neighbors: int = 64,
    query_chunk_size: int = 4096,
    jitter: float = 1.0e-8,
    tree_workers: int = -1,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CUDA batched eigvalsh can fail on very large batches of tiny matrices with
    # some driver/cuSolver combinations. Radius setup is one-off, so do it on CPU.
    eig = torch.linalg.eigvalsh(sigma_eff_a.detach().cpu()).to(g_a.device)
    radius = eta * eig.max(dim=-1).values.clamp_min(jitter).sqrt()

    h2_i = torch.diagonal(sigma_eff_a, dim1=-2, dim2=-1).sum(-1).clamp_min(jitter)
    h2_med = h2_i.median().clamp_min(jitter)
    sigma_ci2 = float(max(sigma_c2, jitter)) * torch.clamp(h2_i / h2_med, min=1.0)

    tree = cKDTree(g_b.mu.detach().cpu().numpy())
    delta_geo = torch.ones(len(g_a), device=g_a.device, dtype=g_a.mu.dtype)
    delta_app = torch.ones(len(g_a), device=g_a.device, dtype=g_a.mu.dtype)
    eye = torch.eye(3, device=g_a.device, dtype=g_a.mu.dtype)

    points_np = g_a.mu.detach().cpu().numpy()
    radii_np = radius.detach().cpu().numpy()
    chunk_starts = range(0, len(g_a), query_chunk_size)
    iterator = chunk_starts
    if show_progress:
        iterator = tqdm(iterator, desc="score primitives", leave=False)

    for start in iterator:
        end = min(start + query_chunk_size, len(g_a))
        neighbor_lists = tree.query_ball_point(
            points_np[start:end],
            radii_np[start:end],
            workers=tree_workers,
        )
        for offset, nbrs in enumerate(neighbor_lists):
            i = start + offset
            if not nbrs:
                continue
            idx = _select_neighbor_indices(g_a.mu[i], nbrs, g_b.mu, max_neighbors)
            delta_pos = g_a.mu[i].unsqueeze(0) - g_b.mu[idx]
            m = sigma_eff_a[i].unsqueeze(0) + sigma_eff_b[idx]
            m = m + jitter * eye.unsqueeze(0)
            solved = torch.linalg.solve(m, delta_pos.unsqueeze(-1)).squeeze(-1)
            d_maha = (delta_pos * solved).sum(-1)
            k_geo = torch.exp(-0.5 * d_maha.clamp_min(0.0))
            delta_geo[i] = 1.0 - k_geo.max()

            c_diff = (g_a.dc_color[i].unsqueeze(0) - g_b.dc_color[idx]).square().sum(-1)
            k_app = torch.exp(-c_diff / (2.0 * sigma_ci2[i].clamp_min(jitter)))
            delta_app[i] = 1.0 - k_app.max()

    return delta_geo.clamp(0.0, 1.0), delta_app.clamp(0.0, 1.0)


def score_primitives_fixed_radius(
    g_a: GaussianSet2D,
    g_b: GaussianSet2D,
    sigma_c2: float,
    radius_m: float,
    max_neighbors: int = 64,
    query_chunk_size: int = 4096,
    jitter: float = 1.0e-8,
    tree_workers: int = -1,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score primitive mismatch with an isotropic fixed-radius neighborhood."""
    radius_m = float(max(radius_m, jitter))
    sigma_c2 = float(max(sigma_c2, jitter))
    tree = cKDTree(g_b.mu.detach().cpu().numpy())
    delta_geo = torch.ones(len(g_a), device=g_a.device, dtype=g_a.mu.dtype)
    delta_app = torch.ones(len(g_a), device=g_a.device, dtype=g_a.mu.dtype)

    points_np = g_a.mu.detach().cpu().numpy()
    chunk_starts = range(0, len(g_a), query_chunk_size)
    iterator = chunk_starts
    if show_progress:
        iterator = tqdm(iterator, desc="score primitives fixed radius", leave=False)

    for start in iterator:
        end = min(start + query_chunk_size, len(g_a))
        neighbor_lists = tree.query_ball_point(points_np[start:end], radius_m, workers=tree_workers)
        for offset, nbrs in enumerate(neighbor_lists):
            i = start + offset
            if not nbrs:
                continue
            idx = _select_neighbor_indices(g_a.mu[i], nbrs, g_b.mu, max_neighbors)
            delta_pos = g_a.mu[i].unsqueeze(0) - g_b.mu[idx]
            d2 = delta_pos.square().sum(-1)
            k_geo = torch.exp(-0.5 * d2 / (radius_m * radius_m))
            delta_geo[i] = 1.0 - k_geo.max()

            c_diff = (g_a.dc_color[i].unsqueeze(0) - g_b.dc_color[idx]).square().sum(-1)
            k_app = torch.exp(-c_diff / (2.0 * sigma_c2))
            delta_app[i] = 1.0 - k_app.max()

    return delta_geo.clamp(0.0, 1.0), delta_app.clamp(0.0, 1.0)


def compute_confidence(h: torch.Tensor, q_ref: float = 0.25) -> torch.Tensor:
    trace = torch.diagonal(h, dim1=-2, dim2=-1).sum(-1).clamp_min(1.0e-8)
    log_trace = torch.log(trace)
    ref = torch.quantile(log_trace, q_ref)
    return torch.sigmoid(log_trace - ref)


def aggregate_score(
    delta_geo: torch.Tensor,
    delta_app: torch.Tensor,
    omega: torch.Tensor,
    mode: str = "full",
    use_confidence_weight: bool = True,
) -> torch.Tensor:
    if mode == "full":
        evidence = torch.clamp(delta_geo + delta_app, max=1.0)
    elif mode == "geometry_only":
        evidence = delta_geo
    elif mode == "appearance_only":
        evidence = delta_app
    else:
        raise ValueError(
            f"Unsupported scoring mode {mode!r}; expected full, geometry_only, or appearance_only"
        )
    if not use_confidence_weight:
        omega = torch.ones_like(omega)
    return omega * evidence.clamp(0.0, 1.0)


def score_stats(values: torch.Tensor) -> dict[str, float | list[int] | list[float]]:
    v = values.detach().cpu().float()
    hist_counts, hist_edges = np.histogram(v.numpy(), bins=10, range=(0.0, 1.0))
    return {
        "min": float(v.min()),
        "median": float(v.median()),
        "mean": float(v.mean()),
        "max": float(v.max()),
        "hist_bins": hist_edges.tolist(),
        "hist_counts": hist_counts.astype(int).tolist(),
    }

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN



@dataclass
class ClusterInfo:
    cluster_id: int
    primitive_count: int
    source_indices: list[int]
    centroid: list[float]
    aabb_min: list[float]
    aabb_max: list[float]
    volume: float
    axis_length_m: float
    mean_color: list[float]
    mean_score: float


def extract_removed_primitives(delta_pre: torch.Tensor, threshold: float = 0.5) -> torch.BoolTensor:
    return delta_pre > threshold


def cluster_removed(
    g_pre: GaussianSet2D,
    removed_mask: torch.BoolTensor,
    scores: torch.Tensor,
    eps: float = 0.02,
    min_samples: int = 20,
    min_volume: float = 1.0e-5,
    max_volume: float = 1.0e-2,
    min_axis_length_m: float = 0.0,
    keep_large_axis_components: bool = False,
) -> tuple[list[ClusterInfo], torch.Tensor]:
    removed_idx = torch.nonzero(removed_mask, as_tuple=False).flatten()
    labels_full = torch.full((len(g_pre),), -1, dtype=torch.long, device=g_pre.device)
    if len(removed_idx) == 0:
        return [], labels_full

    points = g_pre.mu[removed_idx].detach().cpu().numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    labels_full[removed_idx] = torch.as_tensor(labels, dtype=torch.long, device=g_pre.device)

    clusters: list[ClusterInfo] = []
    rgb = g_pre.rgb01.detach().cpu().numpy()
    source_indices = g_pre.source_indices.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    for label in sorted(set(int(x) for x in labels if x >= 0)):
        local = np.where(labels == label)[0]
        pts = points[local]
        aabb_min = pts.min(axis=0)
        aabb_max = pts.max(axis=0)
        extent = aabb_max - aabb_min
        volume = float(np.prod(extent))
        axis_length = _pca_axis_length(pts)
        long_enough = min_axis_length_m > 0.0 and axis_length >= min_axis_length_m
        too_small = volume < min_volume and not long_enough
        too_large = max_volume > 0.0 and volume > max_volume and not (
            keep_large_axis_components and long_enough
        )
        if too_small or too_large:
            labels_full[removed_idx[torch.as_tensor(local, device=g_pre.device)]] = -1
            continue
        global_idx = removed_idx.detach().cpu().numpy()[local]
        clusters.append(
            ClusterInfo(
                cluster_id=len(clusters),
                primitive_count=int(len(local)),
                source_indices=source_indices[global_idx].astype(int).tolist(),
                centroid=pts.mean(axis=0).astype(float).tolist(),
                aabb_min=aabb_min.astype(float).tolist(),
                aabb_max=aabb_max.astype(float).tolist(),
                volume=volume,
                axis_length_m=axis_length,
                mean_color=rgb[global_idx].mean(axis=0).astype(float).tolist(),
                mean_score=float(scores_np[global_idx].mean()),
            )
        )
        labels_full[removed_idx[torch.as_tensor(local, device=g_pre.device)]] = clusters[-1].cluster_id
    return clusters, labels_full


def _pca_axis_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    if len(vh) == 0:
        return 0.0
    axis = vh[0].astype(np.float32)
    projection = centered @ axis
    return float(projection.max() - projection.min())


def save_clusters_json(path: Path, clusters: list[ClusterInfo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(cluster) for cluster in clusters], indent=2))


def save_primitives_ply(
    path: Path,
    g: GaussianSet2D,
    mask: torch.BoolTensor,
    scores: torch.Tensor,
    cluster_labels: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if cluster_labels is None:
        cluster_labels = torch.full((len(g),), -1, dtype=torch.long, device=g.device)

    mu = g.mu[idx].detach().cpu().numpy().astype(np.float32)
    normal = g.normal[idx].detach().cpu().numpy().astype(np.float32)
    rgb = (g.rgb01[idx].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    score = scores[idx].detach().cpu().numpy().astype(np.float32)
    labels = cluster_labels[idx].detach().cpu().numpy().astype(np.int32)
    source_indices = g.source_indices[idx].detach().cpu().numpy().astype(np.int32)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("score", "f4"),
        ("cluster", "i4"),
        ("source_index", "i4"),
    ]
    vertices = np.empty(len(idx), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = mu[:, 0], mu[:, 1], mu[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = normal[:, 0], normal[:, 1], normal[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertices["score"] = score
    vertices["cluster"] = labels
    vertices["source_index"] = source_indices
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)

