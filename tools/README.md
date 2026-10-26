# Tools

This directory is now organized around the latest CoRL2026 EgoDelta pipeline.

## Main Pipeline

Use `run_corl2026_latest_pipeline.py` for the current leaf/deleafing flow. It runs:

1. `score_gaussian_multiview_residuals.py`
2. `optimize_gaussian_presence_post.py`
3. `prune_gaussian_mask_components.py`
4. `extract_leaf_component_targets.py --target_method validated_mask_endpoint`
5. `visualize_component_points.py`

The canonical parameters are documented in `../corl2026_latest_pipeline_runbook.md`.

## Core Stage Tools

- `score_gaussian_multiview_residuals.py`: Gaussian-level multiview change evidence.
- `optimize_gaussian_presence_post.py`: post-scene opacity-drop optimization on the fixed pre Gaussian set.
- `prune_gaussian_mask_components.py`: conservative selected-mask cleanup.
- `extract_leaf_component_targets.py`: split the removed leaf mask into components, reject weak tomato-adjacent false positives, and extract interaction targets.
- `visualize_component_points.py`: final point-cloud projection sanity check for leaf and tomato instances.
- `extract_leaf_connection_paths.py`, `extract_leaf_ray_targets.py`, `visualize_leaf_mask_support_and_points.py`: retained for gs4/backbone ablations and renderer-based diagnostics.

## Data Preparation and QC

- `extract_aria_pinhole_views.py`: extract Aria pinhole images.
- `build_single_trajectory_split.py`: split one trajectory into pre/post COLMAP-style datasets.
- `build_separate_trajectory_manifest.py`: support separate pre/post captures.
- `visualize_post_alignment.py`: inspect pre/post alignment.
- `visualize_gaussian_cluster_on_images.py`: inspect Stage 1 residual clusters.
- `visualize_presence_opt_verification.py`: inspect Stage 2 opacity optimization.

## Tomato-Specific Tools

Tomato harvesting uses residuals as a change seed and red fruit geometry for object extent. The current gs5-validated flow is:

1. `recluster_gaussian_residual_scores.py`: adjust residual candidate quantile/cluster labels without rerendering.
2. `filter_tomato_fruit_mask.py --score_seeded_red_components`: keep red fruit components that contain residual seeds.
3. `split_tomato_fruit_instances.py --auto`: split merged fruit components using adaptive local graph connectivity and reject tiny low-support components.
4. `extract_tomato_fruit_shell_targets.py`: choose harvest targets from nearby green/calyx contact Gaussians.
5. `visualize_component_points.py`: project final fruit/leaf instances and target points onto COLMAP images.

## Validation

- `evaluate_leaf_backbone_targets.py`: compare target rules against manual labels.

## 2DGS Core

These root-level upstream entrypoints are intentionally kept because new EgoDelta datasets need a trained pre-scene model:

- `../train.py`: train the pre-scene 2DGS model.
- `../render.py`: render trained models for reconstruction QC.
- `../metrics.py`: optional image metric evaluation.
- `../view.py`: optional network viewer.

## Removed Legacy

The old mask-completion trials, gs3 one-off shell runner, old generic target extractor, manual-label pack/diagnostic scripts, and old Aria one-shot converter have been removed from this branch. Baselines are intentionally kept under `../baseline/`.
