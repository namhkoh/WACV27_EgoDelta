from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .primitives import (
    ClusterInfo,
    GaussianSet2D,
    aggregate_score,
    build_effective_covariance,
    cluster_removed,
    compute_confidence,
    compute_fim,
    estimate_color_bandwidth,
    estimate_geometric_drift_symmetric,
    extract_removed_primitives,
    in_shared_frustum,
    sample_gaussians,
    save_clusters_json,
    save_primitives_ply,
    score_primitives,
    score_primitives_fixed_radius,
    score_stats,
)
from .contact_proposal import (
    contact_config_from_dict,
    evaluate_contact_proposal_sets,
    generate_contact_proposals,
    save_contact_eval_json,
    save_contact_proposals_json,
    save_contact_proposals_ply,
)
from .data import (
    Camera,
    ScenePair,
    find_scene_targets,
    load_2dgs_ply,
    load_cameras,
    match_removed_targets,
    parse_gt_markdown,
    save_selected_targets_ply,
    save_target_matches,
    summarize_matches,
)
from .visualization import render_change_map, save_change_map, save_removed_point_overlays
from .targets import (
    evaluate_proposals,
    generate_target_proposals,
    proposal_config_from_dict,
    save_proposal_eval_json,
    save_target_proposals_json,
    save_target_proposals_ply,
)


@dataclass
class EgoDeltaDiffResult:
    g_pre: GaussianSet2D
    g_post: GaussianSet2D
    cameras_pre: list[Camera]
    cameras_post: list[Camera]
    delta_pre: torch.Tensor
    delta_post: torch.Tensor
    omega_pre: torch.Tensor
    omega_post: torch.Tensor
    overlap_pre: torch.BoolTensor
    overlap_post: torch.BoolTensor
    removed_mask: torch.BoolTensor
    clusters: list[ClusterInfo]
    cluster_labels: torch.Tensor
    diagnostics: dict[str, Any]


class EgoDeltaDiffPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        compute_cfg = config.get("compute", {})
        requested = str(compute_cfg.get("device", "cuda"))
        allow_cpu_fallback = bool(compute_cfg.get("allow_cpu_fallback", False))
        if requested.startswith("cuda") and not _can_allocate_cuda(requested):
            if allow_cpu_fallback:
                requested = "cpu"
            else:
                raise RuntimeError(
                    f"Config requested {requested}, but a CUDA tensor could not be allocated. "
                    "Run in a CUDA-visible shell or set compute.device=cpu and "
                    "render.enabled=false for CPU-only scoring."
                )
        self.device = torch.device(requested)

    def run_pair(self, pair: ScenePair) -> EgoDeltaDiffResult:
        result = self.run(pair.pre_ply, pair.post_ply, pair.pre_cameras, pair.post_cameras)
        result.diagnostics["scene_name"] = pair.name
        return result

    def run(
        self,
        pre_ply: Path,
        post_ply: Path,
        pre_cameras: Path,
        post_cameras: Path,
    ) -> EgoDeltaDiffResult:
        cfg = self.config
        compute_cfg = cfg.get("compute", {})
        frustum_cfg = cfg.get("frustum", {})
        drift_cfg = cfg.get("drift", {})
        kernel_cfg = cfg.get("kernel", {})
        change_cfg = cfg.get("change", {})
        cluster_cfg = cfg.get("cluster", {})
        ablation_cfg = cfg.get("ablation", {})
        scoring_mode = str(ablation_cfg.get("scoring_mode", "full"))
        use_confidence_weight = bool(ablation_cfg.get("use_confidence_weight", True))
        use_view_overlap_filter = bool(ablation_cfg.get("use_view_overlap_filter", True))
        matching_mode = str(ablation_cfg.get("matching_mode", "adaptive_covariance"))
        fixed_radius_m = float(ablation_cfg.get("fixed_radius_m", 0.03))
        if scoring_mode not in {"full", "geometry_only", "appearance_only"}:
            raise ValueError(
                "ablation.scoring_mode must be one of full, geometry_only, or appearance_only"
            )
        if matching_mode not in {"adaptive_covariance", "fixed_radius"}:
            raise ValueError(
                "ablation.matching_mode must be adaptive_covariance or fixed_radius"
            )

        chunk_size = int(compute_cfg.get("chunk_size", 50000))
        diagnostics: dict[str, Any] = {
            "pre_ply": str(pre_ply),
            "post_ply": str(post_ply),
            "pre_cameras": str(pre_cameras),
            "post_cameras": str(post_cameras),
            "device": str(self.device),
            "ablation": {
                "scoring_mode": scoring_mode,
                "use_confidence_weight": use_confidence_weight,
                "use_view_overlap_filter": use_view_overlap_filter,
                "matching_mode": matching_mode,
                "fixed_radius_m": fixed_radius_m,
            },
        }
        scene_name_hint = _scene_name_from_path(pre_ply)
        object_type_hint = _object_type_from_scene_name(scene_name_hint) if scene_name_hint else "unknown"
        if scene_name_hint:
            diagnostics["scene_name"] = scene_name_hint

        g_pre = load_2dgs_ply(pre_ply, self.device)
        g_post = load_2dgs_ply(post_ply, self.device)
        cameras_pre = load_cameras(pre_cameras, self.device)
        cameras_post = load_cameras(post_cameras, self.device)
        diagnostics["n_primitives_pre_raw"] = len(g_pre)
        diagnostics["n_primitives_post_raw"] = len(g_post)
        diagnostics["n_cameras_pre"] = len(cameras_pre)
        diagnostics["n_cameras_post"] = len(cameras_post)

        finite_pre = _finite_gaussian_mask(g_pre)
        finite_post = _finite_gaussian_mask(g_post)
        diagnostics["finite_pre_kept"] = int(finite_pre.sum().detach().cpu())
        diagnostics["finite_post_kept"] = int(finite_post.sum().detach().cpu())
        diagnostics["finite_pre_ratio"] = float(finite_pre.float().mean().detach().cpu())
        diagnostics["finite_post_ratio"] = float(finite_post.float().mean().detach().cpu())
        g_pre = g_pre[finite_pre]
        g_post = g_post[finite_post]
        if len(g_pre) == 0 or len(g_post) == 0:
            raise RuntimeError("Finite-value filtering removed all primitives from pre or post model")

        diagnostics["scale_pre_median"] = float(g_pre.scales.median().detach().cpu())
        diagnostics["scale_post_median"] = float(g_post.scales.median().detach().cpu())

        max_primitives = compute_cfg.get("max_primitives")
        if max_primitives is not None:
            seed = int(compute_cfg.get("random_seed", 0))
            g_pre = sample_gaussians(g_pre, int(max_primitives), seed)
            g_post = sample_gaussians(g_post, int(max_primitives), seed + 1)
            diagnostics["max_primitives_applied"] = int(max_primitives)

        if use_view_overlap_filter:
            mask_pre = in_shared_frustum(
                g_pre,
                cameras_post,
                near=float(frustum_cfg.get("near", 0.01)),
                far=float(frustum_cfg.get("far", 100.0)),
                chunk_size=chunk_size,
            )
            mask_post = in_shared_frustum(
                g_post,
                cameras_pre,
                near=float(frustum_cfg.get("near", 0.01)),
                far=float(frustum_cfg.get("far", 100.0)),
                chunk_size=chunk_size,
            )
        else:
            mask_pre = torch.ones(len(g_pre), dtype=torch.bool, device=self.device)
            mask_post = torch.ones(len(g_post), dtype=torch.bool, device=self.device)
        diagnostics["frustum_pre_kept"] = int(mask_pre.sum().detach().cpu())
        diagnostics["frustum_post_kept"] = int(mask_post.sum().detach().cpu())
        diagnostics["frustum_pre_ratio"] = float(mask_pre.float().mean().detach().cpu())
        diagnostics["frustum_post_ratio"] = float(mask_post.float().mean().detach().cpu())
        g_pre = g_pre[mask_pre]
        g_post = g_post[mask_post]
        if len(g_pre) == 0 or len(g_post) == 0:
            raise RuntimeError("Frustum filtering removed all primitives from pre or post model")

        u_n2, u_t2, drift_stats = estimate_geometric_drift_symmetric(
            g_pre,
            g_post,
            quantile=float(drift_cfg.get("geometric_quantile", 0.75)),
        )
        diagnostics.update(drift_stats)

        h_pre = compute_fim(
            g_pre,
            cameras_pre,
            chunk_size=chunk_size,
            near=float(frustum_cfg.get("near", 0.01)),
            far=float(frustum_cfg.get("far", 100.0)),
        )
        h_post = compute_fim(
            g_post,
            cameras_post,
            chunk_size=chunk_size,
            near=float(frustum_cfg.get("near", 0.01)),
            far=float(frustum_cfg.get("far", 100.0)),
        )
        diagnostics["fim_pre_trace_median"] = float(torch.diagonal(h_pre, dim1=-2, dim2=-1).sum(-1).median().detach().cpu())
        diagnostics["fim_post_trace_median"] = float(torch.diagonal(h_post, dim1=-2, dim2=-1).sum(-1).median().detach().cpu())

        sigma_eff_pre, sigma_stats_pre = build_effective_covariance(g_pre, h_pre, u_n2, u_t2)
        sigma_eff_post, sigma_stats_post = build_effective_covariance(g_post, h_post, u_n2, u_t2)
        diagnostics["sigma_eff_pre"] = sigma_stats_pre
        diagnostics["sigma_eff_post"] = sigma_stats_post

        sigma_c2_pre = estimate_color_bandwidth(
            g_pre,
            g_post,
            sigma_eff_pre,
            sigma_eff_post,
            quantile=float(drift_cfg.get("photometric_quantile", 0.5)),
            jitter=float(kernel_cfg.get("matrix_jitter", 1.0e-8)),
        )
        sigma_c2_post = estimate_color_bandwidth(
            g_post,
            g_pre,
            sigma_eff_post,
            sigma_eff_pre,
            quantile=float(drift_cfg.get("photometric_quantile", 0.5)),
            jitter=float(kernel_cfg.get("matrix_jitter", 1.0e-8)),
        )
        sigma_c2 = 0.5 * (sigma_c2_pre + sigma_c2_post)
        diagnostics["sigma_c"] = float(sigma_c2**0.5)
        diagnostics["sigma_c2_pre_to_post"] = float(sigma_c2_pre)
        diagnostics["sigma_c2_post_to_pre"] = float(sigma_c2_post)

        score_kwargs = {
            "max_neighbors": int(kernel_cfg.get("max_neighbors", 64)),
            "query_chunk_size": int(kernel_cfg.get("query_chunk_size", 4096)),
            "jitter": float(kernel_cfg.get("matrix_jitter", 1.0e-8)),
            "tree_workers": int(kernel_cfg.get("tree_workers", -1)),
        }
        if matching_mode == "adaptive_covariance":
            delta_geo_pre, delta_app_pre = score_primitives(
                g_pre,
                g_post,
                sigma_eff_pre,
                sigma_eff_post,
                sigma_c2,
                eta=float(kernel_cfg.get("eta", 3.0)),
                **score_kwargs,
            )
            delta_geo_post, delta_app_post = score_primitives(
                g_post,
                g_pre,
                sigma_eff_post,
                sigma_eff_pre,
                sigma_c2,
                eta=float(kernel_cfg.get("eta", 3.0)),
                **score_kwargs,
            )
        else:
            delta_geo_pre, delta_app_pre = score_primitives_fixed_radius(
                g_pre,
                g_post,
                sigma_c2,
                fixed_radius_m,
                **score_kwargs,
            )
            delta_geo_post, delta_app_post = score_primitives_fixed_radius(
                g_post,
                g_pre,
                sigma_c2,
                fixed_radius_m,
                **score_kwargs,
            )
        omega_pre = compute_confidence(h_pre, float(drift_cfg.get("confidence_quantile", 0.25)))
        omega_post = compute_confidence(h_post, float(drift_cfg.get("confidence_quantile", 0.25)))
        delta_pre = aggregate_score(
            delta_geo_pre,
            delta_app_pre,
            omega_pre,
            mode=scoring_mode,
            use_confidence_weight=use_confidence_weight,
        )
        delta_post = aggregate_score(
            delta_geo_post,
            delta_app_post,
            omega_post,
            mode=scoring_mode,
            use_confidence_weight=use_confidence_weight,
        )

        diagnostics["delta_geo_pre"] = score_stats(delta_geo_pre)
        diagnostics["delta_app_pre"] = score_stats(delta_app_pre)
        diagnostics["delta_pre"] = score_stats(delta_pre)
        diagnostics["delta_post"] = score_stats(delta_post)
        diagnostics["omega_pre"] = score_stats(omega_pre)
        diagnostics["omega_post"] = score_stats(omega_post)

        removed_mask = extract_removed_primitives(delta_pre, float(change_cfg.get("binary_threshold", 0.5)))
        cluster_min_axis_length_m = float(cluster_cfg.get("min_axis_length_m", 0.0))
        keep_large_axis_components = bool(cluster_cfg.get("keep_large_axis_components", False))
        if object_type_hint == "leaf":
            cluster_min_axis_length_m = float(
                cluster_cfg.get("leaf_min_axis_length_m", cluster_min_axis_length_m)
            )
            keep_large_axis_components = bool(
                cluster_cfg.get("leaf_keep_large_axis_components", True)
            )
        clusters, cluster_labels = cluster_removed(
            g_pre,
            removed_mask,
            delta_pre,
            eps=float(cluster_cfg.get("eps", 0.02)),
            min_samples=int(cluster_cfg.get("min_samples", 20)),
            min_volume=float(cluster_cfg.get("min_volume", 1.0e-5)),
            max_volume=float(cluster_cfg.get("max_volume", 1.0e-2)),
            min_axis_length_m=cluster_min_axis_length_m,
            keep_large_axis_components=keep_large_axis_components,
        )
        diagnostics["removed_primitive_count"] = int(removed_mask.sum().detach().cpu())
        diagnostics["cluster_count"] = len(clusters)
        diagnostics["cluster_min_axis_length_m"] = cluster_min_axis_length_m
        diagnostics["cluster_keep_large_axis_components"] = keep_large_axis_components

        return EgoDeltaDiffResult(
            g_pre=g_pre,
            g_post=g_post,
            cameras_pre=cameras_pre,
            cameras_post=cameras_post,
            delta_pre=delta_pre,
            delta_post=delta_post,
            omega_pre=omega_pre,
            omega_post=omega_post,
            # The Gaussian sets returned by Stage 2 have already been restricted
            # to the selected overlap population.  Keeping the mask explicit
            # makes the Stage-3 cache contract unambiguous and lets the
            # overlap-off rerun use the identical interface.
            overlap_pre=torch.ones(len(g_pre), dtype=torch.bool, device=self.device),
            overlap_post=torch.ones(len(g_post), dtype=torch.bool, device=self.device),
            removed_mask=removed_mask,
            clusters=clusters,
            cluster_labels=cluster_labels,
            diagnostics=diagnostics,
        )

    def save(self, result: EgoDeltaDiffResult, out_dir: Path, query_views: str | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_primitives_ply(
            out_dir / "removed_primitives.ply",
            result.g_pre,
            result.removed_mask,
            result.delta_pre,
            result.cluster_labels,
        )
        save_clusters_json(out_dir / "removed_clusters.json", result.clusters)

        diagnostics = dict(result.diagnostics)
        scene_name = str(diagnostics.get("scene_name", out_dir.name))
        gt_entries = None
        scene_targets = None
        gt_cfg = self.config.get("gt", {})
        if bool(gt_cfg.get("enabled", False)):
            try:
                gt_path = Path(gt_cfg.get("path", "gt.md"))
                gt_entries = parse_gt_markdown(gt_path)
                scene_targets = find_scene_targets(scene_name, gt_entries)
                if scene_targets is None:
                    diagnostics["gt_status"] = "missing"
                else:
                    scene_uuid, task, gt_points = scene_targets
                    tolerance_m = float(gt_cfg.get("tolerance_m", 0.03))
                    min_score = float(gt_cfg.get("min_score", self.config.get("change", {}).get("binary_threshold", 0.5)))
                    matches = match_removed_targets(
                        result.g_pre,
                        result.removed_mask,
                        result.delta_pre,
                        gt_points,
                        min_score=min_score,
                        tolerance_m=tolerance_m,
                    )
                    metadata: dict[str, object] = {
                        "scene_name": scene_name,
                        "scene_uuid": scene_uuid,
                        "task": task,
                        "gt_path": str(gt_path),
                        "min_score": min_score,
                        "tolerance_m": tolerance_m,
                    }
                    save_target_matches(out_dir / "target_matches.json", matches, metadata)
                    save_selected_targets_ply(out_dir / "selected_targets.ply", matches)
                    diagnostics["gt_status"] = "ok"
                    diagnostics["gt_task"] = task
                    diagnostics["gt_scene_uuid"] = scene_uuid
                    diagnostics["gt_min_score"] = min_score
                    diagnostics["gt_tolerance_m"] = tolerance_m
                    diagnostics["gt_metrics"] = summarize_matches(matches, len(gt_points))
            except Exception as exc:
                diagnostics["gt_status"] = "failed"
                diagnostics["gt_error"] = repr(exc)

        proposal_cfg = self.config.get("proposal", {})
        if bool(proposal_cfg.get("enabled", True)):
            try:
                mode = str(proposal_cfg.get("mode", "contact"))
                gate_m = float(proposal_cfg.get("eval_gate_m", gt_cfg.get("tolerance_m", 0.03)))
                if mode in {"contact", "both"}:
                    contact_config = contact_config_from_dict(proposal_cfg)
                    points = result.g_pre.mu.detach().cpu().numpy()
                    scores = result.delta_pre.detach().cpu().numpy()
                    source_indices = result.g_pre.source_indices.detach().cpu().numpy()
                    rgb01 = result.g_pre.rgb01.detach().cpu().numpy()
                    removed_mask = result.removed_mask.detach().cpu().numpy()
                    cluster_labels = result.cluster_labels.detach().cpu().numpy()
                    object_type = _object_type_from_scene_name(scene_name)
                    contact_proposals = generate_contact_proposals(
                        points=points,
                        scores=scores,
                        source_indices=source_indices,
                        rgb01=rgb01,
                        removed_mask=removed_mask,
                        cluster_labels=cluster_labels,
                        config=contact_config,
                        object_type=object_type,
                    )
                    contact_metadata: dict[str, object] = {
                        "scene_name": scene_name,
                        "object_type": object_type,
                        "mode": "contact",
                        "config": contact_config.__dict__,
                        "input_removed_primitive_count": int(removed_mask.sum()),
                        "gt_free": True,
                    }
                    save_contact_proposals_json(
                        out_dir / "contact_target_proposals.json",
                        contact_proposals,
                        contact_metadata,
                    )
                    save_contact_proposals_ply(out_dir / "contact_target_proposals.ply", contact_proposals)
                    diagnostics["contact_proposal_status"] = "ok"
                    diagnostics["contact_proposal_count"] = len(contact_proposals)
                    if scene_targets is not None:
                        scene_uuid, task, gt_points = scene_targets
                        contact_eval = evaluate_contact_proposal_sets(contact_proposals, gt_points, gate_m=gate_m)
                        contact_eval["scene_name"] = scene_name
                        contact_eval["scene_uuid"] = scene_uuid
                        contact_eval["task"] = task
                        contact_eval["eval_gate_m"] = gate_m
                        save_contact_eval_json(out_dir / "contact_proposal_eval.json", contact_eval)
                        diagnostics["contact_proposal_eval"] = {
                            key: value
                            for key, value in contact_eval.items()
                            if key not in {"matches", "primary_metrics", "all_candidate_metrics"}
                        }

                if mode in {"baseline_nms", "score_nms", "both"}:
                    proposal_config = proposal_config_from_dict(proposal_cfg)
                    candidate_idx = torch.nonzero(result.removed_mask, as_tuple=False).flatten()
                    points = result.g_pre.mu[candidate_idx].detach().cpu().numpy()
                    scores = result.delta_pre[candidate_idx].detach().cpu().numpy()
                    source_indices = result.g_pre.source_indices[candidate_idx].detach().cpu().numpy()
                    proposals = generate_target_proposals(
                        points=points,
                        scores=scores,
                        source_indices=source_indices,
                        config=proposal_config,
                    )
                    proposal_metadata: dict[str, object] = {
                        "scene_name": scene_name,
                        "mode": "baseline_nms",
                        "config": proposal_config.__dict__,
                        "input_removed_primitive_count": int(len(points)),
                        "gt_free": True,
                    }
                    proposal_stem = "" if mode != "both" else "baseline_"
                    save_target_proposals_json(out_dir / f"{proposal_stem}target_proposals.json", proposals, proposal_metadata)
                    save_target_proposals_ply(out_dir / f"{proposal_stem}target_proposals.ply", proposals)
                    diagnostics["baseline_proposal_status"] = "ok"
                    diagnostics["baseline_proposal_count"] = len(proposals)
                    if scene_targets is not None:
                        scene_uuid, task, gt_points = scene_targets
                        proposal_eval = evaluate_proposals(proposals, gt_points, gate_m=gate_m)
                        proposal_eval["scene_name"] = scene_name
                        proposal_eval["scene_uuid"] = scene_uuid
                        proposal_eval["task"] = task
                        proposal_eval["eval_gate_m"] = gate_m
                        save_proposal_eval_json(out_dir / f"{proposal_stem}proposal_eval.json", proposal_eval)
                        diagnostics["baseline_proposal_eval"] = {
                            key: value
                            for key, value in proposal_eval.items()
                            if key != "matches"
                        }

                diagnostics["proposal_status"] = "ok"
                diagnostics["proposal_mode"] = mode
            except Exception as exc:
                diagnostics["proposal_status"] = "failed"
                diagnostics["proposal_error"] = repr(exc)

        overlay_cfg = self.config.get("overlay", {})
        if bool(overlay_cfg.get("enabled", True)):
            try:
                records = save_removed_point_overlays(
                    out_dir=out_dir,
                    scene_name=scene_name,
                    g_pre=result.g_pre,
                    removed_mask=result.removed_mask,
                    scores=result.delta_pre,
                    cameras_by_label={"pre": result.cameras_pre, "post": result.cameras_post},
                    pairs_root=Path(overlay_cfg.get("pairs_root", "pairs_fast")),
                    labels=list(overlay_cfg.get("labels", ["pre", "post"])),
                    max_views=int(overlay_cfg.get("max_views", 8)),
                    max_points_per_view=int(overlay_cfg.get("max_points_per_view", 30000)),
                    point_radius=int(overlay_cfg.get("point_radius", 2)),
                    min_score=float(overlay_cfg.get("min_score", 0.5)),
                )
                diagnostics["overlay_status"] = "ok"
                diagnostics["overlay_count"] = len(records)
            except Exception as exc:
                diagnostics["overlay_status"] = "failed"
                diagnostics["overlay_error"] = repr(exc)

        render_cfg = self.config.get("render", {})
        if query_views is None:
            query_views = str(render_cfg.get("query_views", "post"))
        if bool(render_cfg.get("enabled", True)):
            try:
                cameras = _select_query_cameras(result, query_views)
                max_views = int(render_cfg.get("max_views", 8))
                for view_idx, camera in enumerate(cameras[:max_views]):
                    g = result.g_pre if query_views != "pre_added" else result.g_post
                    delta = result.delta_pre if query_views != "pre_added" else result.delta_post
                    change_map = render_change_map(
                        g,
                        delta,
                        camera,
                        znear=float(render_cfg.get("near", 0.01)),
                        zfar=float(render_cfg.get("far", 100.0)),
                        scale_modifier=float(render_cfg.get("scale_modifier", 1.0)),
                    )
                    save_change_map(out_dir / f"change_map_view_{view_idx:03d}.png", change_map)
                diagnostics["render_status"] = "ok"
                diagnostics["rendered_view_count"] = min(len(cameras), max_views)
            except Exception as exc:  # Rendering should not block primitive-level outputs.
                diagnostics["render_status"] = "failed"
                diagnostics["render_error"] = repr(exc)
        else:
            diagnostics["render_status"] = "disabled"

        (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


def _select_query_cameras(result: EgoDeltaDiffResult, query_views: str) -> list[Camera]:
    if query_views == "pre":
        return result.cameras_pre
    if query_views in {"post", "pre_added"}:
        return result.cameras_post
    if query_views == "all":
        return result.cameras_post + result.cameras_pre
    raise ValueError(f"Unsupported query_views={query_views!r}; expected pre, post, pre_added, or all")


def _object_type_from_scene_name(scene_name: str) -> str:
    if scene_name.endswith("_tomato"):
        return "tomato"
    if scene_name.endswith("_leaf"):
        return "leaf"
    return "leaf"


def _scene_name_from_path(path: Path) -> str | None:
    path = Path(path)
    for item in [path, *path.parents]:
        if item.name.startswith("egodelta_"):
            return item.name
    return None


def _finite_gaussian_mask(g: GaussianSet2D) -> torch.BoolTensor:
    return (
        _finite_rows(g.mu)
        & _finite_rows(g.rotation)
        & _finite_rows(g.quaternion)
        & _finite_rows(g.scales)
        & _finite_rows(g.normal)
        & _finite_rows(g.opacity)
        & _finite_rows(g.dc_color)
    )


def _finite_rows(values: torch.Tensor) -> torch.BoolTensor:
    finite = torch.isfinite(values)
    if finite.ndim == 1:
        return finite
    return finite.flatten(1).all(dim=-1)


def _can_allocate_cuda(device: str) -> bool:
    try:
        torch.empty(1, device=torch.device(device))
        return True
    except Exception:
        return False
