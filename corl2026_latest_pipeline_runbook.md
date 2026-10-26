# CoRL2026 Latest Pipeline Runbook

Use this file when applying the current verified pipeline to a new dataset. It is intentionally operational: set the paths, run the stages, inspect the listed outputs.

For method rationale and historical notes, see `corl2026_change_detection_algorithm.md`.

## Current Status

The latest verified result is the `gs5` component-level leaf and tomato pipeline:

- removed mask: residual scoring -> constrained opacity-drop optimization -> conservative component pruning
- leaf interaction target: split the fixed removed mask into connected components, reject weak tomato-adjacent false positives, then choose a validated mask endpoint using retained green plant support
- default leaf target method: `validated_mask_endpoint` in `extract_leaf_component_targets.py`
- tomato interaction target: residual-seeded red fruit instances + nearby green/calyx shell contact in `extract_tomato_fruit_shell_targets.py`

Do not treat retained support, connection paths, or visualization support masks as the final removed-object mask. The final leaf removed mask is `leaf_instance_candidate`; the robot target is `interaction_target_xyz`.

## One-Command Runner

For the current leaf pipeline, prefer the wrapper below unless you need to debug a specific stage:

```bash
python tools/run_corl2026_latest_pipeline.py \
  --task ${TASK} \
  --data_root ${DATA_ROOT} \
  --single_trajectory \
  --pre_model ${PRE_MODEL} \
  --out_root ${OUT_ROOT} \
  --object_type leaf
```

Use `--dry_run` to print the exact stage commands without running them. Use `--stages score,optimize` to run a subset.

The individual stage commands below are kept as the debugging and reproducibility reference.

## Required Inputs

Before starting from this runbook, make sure these exist:

- aligned pre/post 2DGS dataset directories under `DATA_ROOT`
- a trained pre-scene 2DGS model at `PRE_MODEL`
- post frames belonging to the target phase
- verified alignment between pre scene and post images

If `PRE_MODEL` does not exist yet, train it in this repository:

```bash
python train.py \
  -s <pre_dataset_dir> \
  -m ${PRE_MODEL} \
  -r 2 \
  --quiet
```

For single-trajectory data, identity alignment is expected. If the dataset uses a selected post-frame list, pass both `--single_trajectory` and `--post_frame_list`.

Recommended variables:

```bash
TASK=gs5_leaf
DATA_ROOT=data/corl2026_2dgs_gs5/leaf_pair
PRE_MODEL=output/corl2026_2dgs_gs5/gs5_leaf_pre
OUT_ROOT=output/corl2026_2dgs_gs5
PRE_DATASET_DIR=${DATA_ROOT}/${TASK}_pre

SCORE_DIR=${OUT_ROOT}/${TASK}_scores
OPT_DIR=${OUT_ROOT}/${TASK}_presence_losscolor
PRUNE_DIR=${OUT_ROOT}/${TASK}_component_pruned
TARGET_DIR=${OUT_ROOT}/${TASK}_component_targets_validated_mask_endpoint
VIS_DIR=${TARGET_DIR}_points_pre_color
```

## Stage 1: Gaussian Residual Scoring

For leaf/deleafing targets, use plant ROI expansion and a permissive cluster setting. The latest verified leaf setting is:

```bash
python tools/score_gaussian_multiview_residuals.py \
  --task ${TASK} \
  --data_root ${DATA_ROOT} \
  --single_trajectory \
  --pre_model ${PRE_MODEL} \
  --model_path ${SCORE_DIR} \
  --resolution 2 \
  --frame_stride 1 \
  --score_quantile 0.95 \
  --cluster_eps 0.055 \
  --cluster_min_samples 3 \
  --cluster_max_bbox_diag 0.8 \
  --cluster_max_planarity 200.0 \
  --plant_roi_mode expanded \
  --plant_roi_max_components 3
```

If this is not a leaf task:

- clip: consider `--appearance_roi_mode dark` as a prior, but verify visually.
- tomato: consider `--appearance_roi_mode red` as a prior, but verify visually.
- if the prior suppresses true positives, rerun with no appearance prior and rely on residual + opacity optimization.

Expected outputs:

- `${SCORE_DIR}/gaussian_multiview_residual_scores.npz`
- `${SCORE_DIR}/gaussian_multiview_residual_scores.ply`
- `${SCORE_DIR}/multiview_residual_clusters.json`
- `${SCORE_DIR}/score_config.json`

Minimum checks:

- residual clusters should lie on plausible removed foreground structures
- broad wall/floor/lighting residuals should not dominate
- `score_config.json` should report the intended alignment method and post camera count

## Stage 2: Constrained Opacity-Drop Optimization

Use the latest verified opacity optimization with loss color correction:

```bash
python tools/optimize_gaussian_presence_post.py \
  --task ${TASK} \
  --data_root ${DATA_ROOT} \
  --single_trajectory \
  --pre_model ${PRE_MODEL} \
  --score_npz ${SCORE_DIR}/gaussian_multiview_residual_scores.npz \
  --cluster_ids all \
  --out_dir ${OPT_DIR} \
  --resolution 2 \
  --frame_stride 1 \
  --iterations 800 \
  --drop_sparsity_weight 0.0001 \
  --loss_color_correction \
  --candidate_expand_radius 0.06 \
  --graph_expand_geodesic_max 0.5 \
  --graph_expand_edge_radius 0.025 \
  --graph_expand_max_nodes 30000 \
  --graph_expand_green_margin -0.05 \
  --graph_expand_residual_min 0.05 \
  --select_drop_min 0.2 \
  --select_opacity_drop_min 0.02 \
  --max_selected_bbox_diag 0.8 \
  --max_selected_planarity 200.0 \
  --post_cluster_eps 0.06
```

Expected outputs:

- `${OPT_DIR}/presence_optimization_scores.npz`
- `${OPT_DIR}/presence_optimization_scores.ply`
- `${OPT_DIR}/presence_optimization_summary.json`
- `${OPT_DIR}/post_optimization_clusters.json`
- `${OPT_DIR}/projection_images/`

Minimum checks:

- `loss_improvement` should be positive
- selected clusters should project to removed structures in `projection_images`
- if a cluster crosses to retained neighboring plant, do not fix it by mask completion; inspect Stage 3 pruning and Stage 5 target extraction separately

## Stage 3: Conservative Component Pruning

Use pruning to remove small, low-score fragments only:

```bash
python tools/prune_gaussian_mask_components.py \
  --presence_npz ${OPT_DIR}/presence_optimization_scores.npz \
  --score_npz ${SCORE_DIR}/gaussian_multiview_residual_scores.npz \
  --out_dir ${PRUNE_DIR} \
  --selected_clusters all \
  --component_radius 0.018 \
  --min_component_size 120 \
  --min_component_ratio 0.08 \
  --min_median_score 0.25
```

Expected outputs:

- `${PRUNE_DIR}/component_pruned_presence_scores.npz`
- `${PRUNE_DIR}/component_pruned_mask.ply`
- `${PRUNE_DIR}/component_pruned_summary.json`

Minimum checks:

- pruning should remove only small isolated fragments
- keep plausible leaf/stem subcomponents even when they are thin
- this stage should not try to extend the mask to a main stem joint

## Tomato Harvest Variant

Tomato targets should not use the leaf connection stages. Use residuals only to seed changed red fruit components, then recover fruit extent and target points from pre-scene tomato geometry. This avoids over-relying on weak residuals when one tomato is sparsely reconstructed.

Canonical gs5-tested sequence:

```bash
python tools/recluster_gaussian_residual_scores.py \
  --score_npz ${SCORE_DIR}/gaussian_multiview_residual_scores.npz \
  --out_dir ${SCORE_DIR}_q80_recluster \
  --score_quantile 0.80 \
  --cluster_eps 0.06 \
  --cluster_min_samples 4 \
  --cluster_max_bbox_diag 0.8 \
  --cluster_max_planarity 0

python tools/filter_tomato_fruit_mask.py \
  --presence_npz ${OPT_DIR}/presence_optimization_scores.npz \
  --score_npz ${SCORE_DIR}_q80_recluster/gaussian_multiview_residual_scores.npz \
  --out_dir ${OUT_ROOT}/${TASK}_fruit_seeded_red_components \
  --score_seeded_red_components \
  --component_radius 0.035

python tools/split_tomato_fruit_instances.py \
  --presence_npz ${OUT_ROOT}/${TASK}_fruit_seeded_red_components/tomato_fruit_mask.npz \
  --out_dir ${OUT_ROOT}/${TASK}_fruit_auto_graph_instances \
  --auto \
  --source_cluster largest \
  --min_instance_gaussians 50 \
  --min_instance_fraction 0.08 \
  --min_split_fraction 0.12

python tools/extract_tomato_fruit_shell_targets.py \
  --presence_npz ${OUT_ROOT}/${TASK}_fruit_auto_graph_instances/tomato_fruit_mask.npz \
  --score_npz ${SCORE_DIR}_q80_recluster/gaussian_multiview_residual_scores.npz \
  --out_dir ${OUT_ROOT}/${TASK}_fruit_shell_targets \
  --mask_key tomato_fruit_candidate \
  --cluster_key tomato_instance_id \
  --selected_clusters all \
  --search_radius 0.09 \
  --min_contact_distance 0.002 \
  --component_radius 0.018 \
  --min_component_size 2 \
  --target_points 12 \
  --green_margin 0.0

python tools/visualize_component_points.py \
  --dataset_dir <pre_dataset_dir> \
  --presence_npz ${OUT_ROOT}/${TASK}_fruit_shell_targets/tomato_component_targets.npz \
  --summary_json ${OUT_ROOT}/${TASK}_fruit_shell_targets/tomato_component_targets_summary.json \
  --out_dir ${OUT_ROOT}/${TASK}_fruit_shell_targets_points_pre \
  --mask_key tomato_instance_candidate \
  --cluster_key tomato_instance_id \
  --selected_clusters all \
  --point_color_mode cluster
```

Interpretation:

- `tomato_fruit_candidate`: final removed fruit mask.
- `tomato_instance_id`: automatically split fruit instances.
- `interaction_target_xyz`: green/calyx contact points for robot interaction.
- tiny residual-only red components are rejected when they do not meet the minimum fruit support.

This is still a tomato-specific prior, not a class-agnostic removal method. It is appropriate when the task is ripe tomato harvesting and the target class is known.

## Stage 4: Leaf Component Targets

For leaf/deleafing, split the fixed removed mask into physical components and extract one interaction target per accepted component:

```bash
python tools/extract_leaf_component_targets.py \
  --target_method validated_mask_endpoint \
  --presence_npz ${PRUNE_DIR}/component_pruned_presence_scores.npz \
  --score_npz ${SCORE_DIR}/gaussian_multiview_residual_scores.npz \
  --out_dir ${TARGET_DIR} \
  --selected_clusters all
```

Expected outputs:

- `${TARGET_DIR}/leaf_component_targets.npz`
- `${TARGET_DIR}/leaf_component_targets.ply`
- `${TARGET_DIR}/leaf_component_targets_summary.json`

Use these fields for downstream robot code:

- `leaf_instance_candidate`: final accepted removed leaf/stem component mask
- `leaf_instance_id`: per-component instance id
- `interaction_target_instance_id`
- `interaction_target_xyz`
- `interaction_target_confidence`
- `interaction_target_method`

Components rejected as weak residual evidence or tomato-adjacent false positives are recorded in `rejected_components` and do not receive robot targets.

## Stage 5: Visualization

Use pre-scene cameras for target inspection:

```bash
python tools/visualize_component_points.py \
  --dataset_dir ${PRE_DATASET_DIR} \
  --presence_npz ${TARGET_DIR}/leaf_component_targets.npz \
  --summary_json ${TARGET_DIR}/leaf_component_targets_summary.json \
  --out_dir ${VIS_DIR} \
  --mask_key leaf_instance_candidate \
  --cluster_key leaf_instance_id \
  --selected_clusters all \
  --frame_stride 6 \
  --max_frames 18 \
  --point_color_mode cluster
```

Expected output:

- `${VIS_DIR}/points_contact_sheet.jpg`

What to inspect:

- point view: accepted Gaussians should follow the removed leaf/stem geometry
- target marker: should lie on the removed component endpoint selected by retained plant support, not on tomato stems or retained neighboring plant
- rejected components should be explainable in `leaf_component_targets_summary.json`

## Ablation and Regression Checks

The gs4 connection/backbone tools are retained for historical ablation. If using `evaluate_leaf_backbone_targets.py`, first generate connection paths with `extract_leaf_connection_paths.py` and pass that summary as `--connection_summary_json`.

The current production gs5 leaf target is `extract_leaf_component_targets.py --target_method validated_mask_endpoint`.

## Dataset-Specific Notes

Leaf/deleafing:

- multiple targets are expected
- use plant ROI expansion
- final removed mask should remain conservative
- target point comes from a validated removed-mask endpoint supported by retained green plant geometry, not mask completion

Clip:

- dark/black appearance prior can help candidate selection
- target extraction is task-specific; do not use leaf junction logic by default
- verify that dark supports, shadows, and frame hardware are not selected

Tomato:

- red appearance prior can help candidate selection
- robot target should likely be the fruit-stem junction, not the fruit centroid
- current leaf backbone rule should not be reused blindly for tomato without a tomato-specific retained-stem rule

## Common Failure Modes

If the robot/frame is selected:

- check pre/post alignment first
- inspect per-frame exposure and residual images
- ensure color correction is enabled in Stage 2
- tighten appearance or plant ROI only after verifying that it does not remove the target

If thin stems are missing:

- confirm pre 2DGS actually reconstructed the thin stem
- lower pruning aggressiveness before changing target extraction
- inspect pointcloud visualization, not only rendered support masks

If the target is short of the junction:

- keep the removed mask unchanged
- use `backbone_gap_or_rise`
- inspect retained candidates and connection path
- do not extend the removed mask unless visual evidence shows the removed object was actually reconstructed there

## Minimal Hand-Off Checklist

For another agent, provide:

- `TASK`
- `DATA_ROOT`
- `PRE_MODEL`
- post-frame list path if used
- object type: leaf, clip, or tomato
- the output root
- whether manual labels exist

Then run Stages 1-6 in order and report:

- number of residual clusters
- number of opacity-selected Gaussians
- number removed by pruning
- final selected clusters
- target summary, if leaf or tomato
- visualization contact sheet paths
