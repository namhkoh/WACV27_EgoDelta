# CoRL2026 Single-Trajectory 2DGS Change Detection Algorithm

This document summarizes the current working pipeline for detecting removed objects from a single Aria trajectory using a pre-trained 2D Gaussian Splatting scene.

The current preferred direction is not to rely on a hand-crafted color mask. Instead, we keep the pre-scene Gaussians fixed and score each Gaussian by multi-view evidence from post images.

For direct execution on a new dataset, start from `corl2026_latest_pipeline_runbook.md`.

## Goal

Given:

- a single Aria trajectory that contains both pre and post segments,
- manually selected pre frames and post frames,
- an MPS trajectory and semidense point cloud,
- a pre-scene 2DGS model trained from pre frames,

find the pre-scene Gaussians that correspond to objects that disappeared in the post segment, and produce interaction targets for the robot.

The target cases are clip removal, deleafing, and tomato harvesting. Leaf masks and leaf interaction targets are treated as separate outputs because the removed-mask signal is photometric, while the interaction target is geometric and should lie on the retained plant connection.

## Current Verified Method: gs4 Leaf Targets

The latest verified leaf pipeline is:

1. Train or reuse the pre-scene 2DGS from the pre segment.
2. Score frozen pre Gaussians with multi-view residuals from the post segment.
3. Run constrained opacity-drop optimization on candidate Gaussians, using optional per-frame affine color correction in the loss.
4. Keep the removed Gaussian mask fixed after conservative component pruning.
5. Infer the leaf interaction target from retained, stable, green plant Gaussians near the removed-mask endpoint ray.

The important design choice is that Stage 4 and Stage 5 are not forced to share one mask. The removed mask should describe the disappeared leaf or stem surface. The robot target should describe where that removed part connected to retained plant structure.

### Removed Mask

The current mask path is:

```bash
python tools/optimize_gaussian_presence_post.py \
  --loss_color_correction \
  ...

python tools/prune_gaussian_mask_components.py \
  ...
```

`--loss_color_correction` applies robust per-frame affine color correction to the rendered RGB before the opacity optimization loss. On `gs4` leaf data this kept the final target accuracy stable while reducing small unstable components after pruning.

The mask is intentionally not completed toward the main stem. Completion made some clusters look better in projection, but it also crossed into retained neighboring stems in several views. The final mask therefore remains the conservative opacity/residual-selected removed region.

### Leaf Interaction Target

The current production target extractor is:

```bash
python tools/extract_leaf_ray_targets.py \
  --target_method backbone_gap_or_rise \
  --presence_npz output/corl2026_2dgs_gs4/gs4_leaf_losscolor_component_pruned/component_pruned_presence_scores.npz \
  --score_npz output/corl2026_2dgs_gs4/gs4_leaf_trial_q95_scores/gaussian_multiview_residual_scores.npz \
  --connection_summary_json output/corl2026_2dgs_gs4/gs4_leaf_losscolor_connection_paths/leaf_connection_paths_summary.json \
  --selected_clusters 0,1,2,3 \
  --out_dir output/corl2026_2dgs_gs4/gs4_leaf_backbone_targets
```

`backbone_gap_or_rise` replaces the earlier fixed `0.10 m` along-ray prior as the default. For each leaf cluster:

1. Use the removed-mask endpoint and direction estimated by `extract_leaf_connection_paths.py`.
2. Search a tube around that ray for retained Gaussians that are:
   - green-like plant Gaussians,
   - not part of the removed mask,
   - low opacity/presence-change,
   - visible in enough post views.
3. Bin these retained Gaussians along the ray and build a local backbone profile.
4. Use connected components in the retained tube to keep the dominant retained plant component.
5. Choose the first reliable connection evidence:
   - the first node after a large gap in the dominant component, or
   - the component entry if it is already dense enough, or
   - the first density rise in the dominant component.

The fixed `ray_prior` mode remains available for comparison and fallback, but it is no longer the recommended default.

### gs4 Label Validation

Validation used four manually labeled leaf junction proxies from `gs4`.

| Method | Mean error | Max error | Note |
| --- | ---: | ---: | --- |
| `ray_prior_0.10m` | `2.84 cm` | `5.68 cm` | previous fixed-distance rule |
| `largest_component_entry` | `4.44 cm` | `11.72 cm` | often too early or wrong branch |
| `density_first_peak` | `6.51 cm` | `12.30 cm` | often too far into dense retained plant |
| `backbone_score_peak` | `4.83 cm` | `8.88 cm` | picks strongest node, not connection onset |
| `backbone_gap_or_rise` | `0.97 cm` | `1.61 cm` | current default |

Production check output:

- `output/corl2026_2dgs_gs4/gs4_leaf_backbone_targets_production_check/leaf_ray_targets_summary.json`
- `output/corl2026_2dgs_gs4/gs4_leaf_backbone_targets_production_check/leaf_ray_targets.ply`
- `output/corl2026_2dgs_gs4/gs4_leaf_backbone_targets_production_check/leaf_ray_targets.npz`

Visualization from the same verified method:

- `output/corl2026_2dgs_gs4/gs4_leaf_backbone_gap_or_rise_visual_pre/points_contact_sheet.jpg`
- `output/corl2026_2dgs_gs4/gs4_leaf_backbone_gap_or_rise_visual_pre/support_mask_contact_sheet.jpg`

This result is strong enough to use as the current method, but it is still validated on only four labeled `gs4` leaves. More datasets should be added before claiming generality.

The following `gs_new` sections are kept as historical run details and implementation reference.

## Data Preparation

### 1. Extract Pinhole RGB Views

Raw Aria RGB is fisheye/native. We convert it to the same pinhole convention used by the previous `semidense_geo_baseline` pipeline:

- rectified pinhole image size: `720 x 720`
- focal length: `380`
- image rotation: clockwise 90 degrees
- camera pose: `T_world_camera` using the rotated pinhole camera frame

Tool:

```bash
python tools/extract_aria_pinhole_views.py \
  --vrs /path/to/workspace/CoRL2026/data/gs_new/28c1aee4-b27a-4977-ae26-502521a46ae2.vrs \
  --mps_dir /path/to/workspace/CoRL2026/data/gs_new/mps_28c1aee4-b27a-4977-ae26-502521a46ae2_vrs \
  --out_root /path/to/workspace/CoRL2026/data/gs_new/pinhole_view \
  --stride 1 \
  --preview_stride 20
```

Output:

- `frames_rect/frame_*.jpg`
- `frames_rect/poses.csv`
- `frames_rect/camera_pinhole.json`
- `frames_rect/frame_index.csv`
- `frames_rect/selection_contact_sheet.jpg`

For the current `gs_new` data:

- total extracted RGB frames: `1040`
- frames with pose: `1022`
- pose-less frames: `0-16`, `1039`

### 2. Select Pre/Post Frames

Current selection:

- pre: 120 frames from pose-valid frames before `545`, actual selected range `21-543`
- post: 120 frames from pose-valid frames between `665` and `1030`, actual selected range `665-1028`

Selection rule used in the current run:

1. Split each candidate range into 120 temporal buckets.
2. In each bucket, choose the frame with highest simple image sharpness score.
3. Exclude frames without MPS pose.

Output:

- `data/corl2026_2dgs_single/selections/gs_new_pre_frames.txt`
- `data/corl2026_2dgs_single/selections/gs_new_post_frames.txt`
- selected-frame contact sheets under `data/corl2026_2dgs_single/selections/`

### 3. Build 2DGS Datasets

Because pre and post are from the same trajectory, the pre/post world frames are already shared. The dataset manifest uses identity alignment.

Tool:

```bash
python tools/build_single_trajectory_split.py \
  --source_root /path/to/workspace/CoRL2026/data/gs_new/pinhole_view/frames_rect \
  --points /path/to/workspace/CoRL2026/data/gs_new/mps_28c1aee4-b27a-4977-ae26-502521a46ae2_vrs/slam/semidense_points.csv.gz \
  --pre_frames data/corl2026_2dgs_single/selections/gs_new_pre_frames.txt \
  --post_frames data/corl2026_2dgs_single/selections/gs_new_post_frames.txt \
  --task gs_new \
  --out_root data/corl2026_2dgs_single \
  --overwrite
```

Output:

- `data/corl2026_2dgs_single/gs_new_pre`
- `data/corl2026_2dgs_single/gs_new_post`
- `data/corl2026_2dgs_single/manifest.json`

Current dataset stats:

- pre images: `120`
- post images: `120`
- init points: `41048`
- camera: `PINHOLE 720 720 380.0 380.0 359.5 359.5`
- alignment: `identity_single_trajectory`

## Pre Scene Optimization

Train or provide a pre-scene 2DGS model from the selected pre frames. This branch keeps only the EgoDelta runtime that loads and renders an existing pre-scene model; run 2DGS training in a dedicated upstream 2DGS worktree if a new model is needed.

Current result:

- final train L1: `0.00904`
- final train PSNR: `37.56 dB`
- final Gaussian count: `104550`
- checkpoint: `output/corl2026_2dgs_single/gs_new_pre/point_cloud/iteration_30000/point_cloud.ply`

Quality checks:

- render preview: `output/corl2026_2dgs_single/preview/gs_new_pre_gt_vs_render.jpg`
- post alignment overlay: `output/corl2026_2dgs_single/alignment_overlay_gs_new/`

The current visual check indicates:

- pre splatting is good,
- post alignment and projected point cloud look reasonable.

## Change Detection Direction

### Archived Gate-Based Idea

An earlier experimental implementation froze the pre Gaussians and optimized a per-Gaussian presence gate. That code path has been removed from the working codebase because it depended on unreliable image-space masks before we had a stable 3D candidate set.

The archived idea was:

```text
gated_opacity_i = pre_opacity_i * presence_i
removal_score_i = 1 - presence_i
```

It matched the desired modeling direction, but it needed reliable evidence telling the optimizer where opacity should drop. The initial implementation used image-space masks:

- `M_rgb`: RGB residual between pre render and post image
- `M_free`: free-space evidence from post depth
- `M_new_occlusion`: new occlusion evidence from post depth

Without usable post-only depth, `M_free` and `M_new_occlusion` are empty or unreliable. Then the final mask depends mostly on RGB residual, which also fires on ceiling, lighting changes, exposure mismatch, and render artifacts. This is why the RGB-only gate result found background regions.

### Current Preferred Method: Gaussian-Level Multi-View Residual Scoring

Instead of first building a 2D mask, we score each pre Gaussian directly.

Tool:

```bash
python tools/score_gaussian_multiview_residuals.py \
  --task gs_new \
  --data_root data/corl2026_2dgs_single \
  --pre_model output/corl2026_2dgs_single/gs_new_pre \
  --model_path output/corl2026_2dgs_single/multiview_residual_gs_new_all_q99 \
  --resolution 2 \
  --frame_stride 1 \
  --score_quantile 0.99
```

### Algorithm

Let the frozen pre scene contain Gaussians:

```text
G_i = {x_i, scale_i, rotation_i, color_i, opacity_i}
```

For each post camera view:

1. Render the frozen pre 2DGS model into the post camera.
2. Apply robust per-channel affine color correction from pre render to post RGB.
3. Compute a residual map:

```text
R_v(p) = mean_abs(pre_corrected_v(p) - post_rgb_v(p))
```

4. Project every pre Gaussian center `x_i` into the post camera.
5. Keep only visible Gaussians:

```text
rasterizer radius_i > 0
sampled pre alpha > alpha_min
abs(projected_center_depth_i - rendered_depth_at_projection_i) < depth_margin
```

6. Sample the residual map at the projected Gaussian center.
7. Compute a per-view high-residual threshold by residual quantile.
8. Accumulate Gaussian-level evidence:

```text
visibility_count_i
residual_mean_i
residual_excess_mean_i = mean(max(residual_i_v - view_threshold_v, 0))
high_residual_ratio_i = # high residual observations / visibility_count_i
```

Then compute a 3D density prior:

```text
density_count_i = number of active pre Gaussians within radius r
```

Final score:

```text
score_i =
  normalized_residual_excess_i
  * normalized_high_residual_ratio_i
  * normalized_density_i
```

Gaussians are set to zero score if:

- pre opacity is too low,
- visibility count is too low.

Candidate selection:

1. Select top scoring Gaussians by score quantile.
2. Run DBSCAN in 3D.
3. Filter clusters by:
   - maximum bbox diagonal,
   - planarity ratio,
   - minimum cluster size.

This is intended to suppress broad planar background residuals such as ceiling/wall changes and keep compact dense foreground changes.

### Current Multi-View Result

Output:

- `output/corl2026_2dgs_single/multiview_residual_gs_new_all_q99/gaussian_multiview_residual_scores.npz`
- `output/corl2026_2dgs_single/multiview_residual_gs_new_all_q99/gaussian_multiview_residual_scores.ply`
- `output/corl2026_2dgs_single/multiview_residual_gs_new_all_q99/multiview_residual_clusters.json`

Current result:

- post views used: `120`
- candidate Gaussians: `230`
- final compact clusters: `1`
- cluster size: `54`
- cluster centroid: `[-0.8266, 0.7309, -0.2189]`
- bbox diagonal: `0.098 m`
- mean residual: `0.140`
- mean visibility count: `119.8 / 120`

The selected cluster is compact, dense, and consistently high residual across almost all post views.

### Image-Space Visualization

To inspect where the detected 3D cluster appears in post images:

```bash
python tools/visualize_gaussian_cluster_on_images.py \
  --task gs_new \
  --data_root data/corl2026_2dgs_single \
  --score_npz output/corl2026_2dgs_single/multiview_residual_gs_new_all_q99/gaussian_multiview_residual_scores.npz \
  --out_dir output/corl2026_2dgs_single/multiview_residual_gs_new_all_q99/cluster_projection_images \
  --resolution 2 \
  --max_frames 24 \
  --cluster_id 0 \
  --draw_candidates
```

Output:

- `cluster_0_contact_sheet.jpg`
- individual post image overlays
- `cluster_0_projection_metadata.json`

Overlay convention:

- orange: final cluster members
- red: high-score candidates before cluster filtering

## How This Relates to Post Optimization

The current multi-view scoring step does not optimize post scene parameters. It only evaluates which fixed pre Gaussians explain post poorly.

This is preferable as the first pass because a free post optimization could hide the removed object by moving or recoloring Gaussians. If we do post adaptation, it should be constrained:

- keep Gaussian count fixed,
- keep xyz/scale/rotation mostly frozen,
- update only presence/opacity or a small appearance correction,
- use the multi-view residual cluster as a 3D prior.

If we revisit post adaptation later, the likely next step is:

1. Use the multi-view residual cluster as a trusted 3D candidate set.
2. Run a constrained opacity/presence optimization where only cluster-near Gaussians are allowed to reduce opacity.
3. Export a removed-object PLY and optional inpainted/gated renders.

## Remaining Decisions

1. **Frame selection policy**

   Current selection uses 120 sharp frames per phase. We should decide whether this should remain fixed, be manually curated, or be selected by view coverage around the target foreground.

2. **Use of post depth evidence**

   The current post point cloud is from the whole single trajectory, so removed pre objects may still be present in the global semidense cloud. For depth-based free-space evidence, we need either:

   - post-only MPS/semidense reconstruction,
   - frame-filtered point cloud generation,
   - or a different depth estimator.

3. **Whether to run constrained post optimization**

   Current best result is score-based, not optimized. We need to decide whether final output should be:

   - direct scored Gaussian cluster,
   - constrained opacity/presence-gated Gaussian cluster,
   - or post-adapted model difference.

4. **Foreground prior**

   The current density prior is generic. The tomato/vine setting suggests foreground is a dense local object attached to vine structure. We still need to decide whether to explicitly crop to foreground, use hand/gaze/action hints, or keep it fully geometry/statistics based.

5. **Cluster filtering thresholds**

   Current filtering uses bbox diagonal and planarity ratio. These thresholds are reasonable but not yet validated across scenes.

## Ablations Needed

1. **Pixel mask gate vs Gaussian multi-view score**

   Compare:

   - RGB-only gate,
   - depth+RGB gate,
   - multi-view Gaussian score,
   - multi-view score followed by constrained opacity/presence optimization.

2. **Score components**

   Remove one component at a time:

   - residual magnitude,
   - per-view residual excess,
   - high residual ratio,
   - visibility count,
   - local density.

3. **Density prior**

   Test:

   - no density prior,
   - Gaussian density,
   - MPS point-cloud density,
   - combined Gaussian + point-cloud density.

4. **Cluster filters**

   Sweep:

   - `score_quantile`
   - `cluster_eps`
   - `cluster_min_samples`
   - `cluster_max_bbox_diag`
   - `cluster_max_planarity`

5. **Frame count and coverage**

   Test:

   - 40, 80, 120 post views,
   - uniform temporal selection vs manual view-balanced selection,
   - near-only vs mixed distance views.

6. **Visibility test**

   Current visibility checks Gaussian center projection, rasterized radius, alpha, and depth agreement. We should compare this with:

   - center-only projection,
   - rasterizer contribution-based visibility,
   - multi-sample per Gaussian footprint.

7. **Color correction**

   The current residual uses robust per-channel affine correction per frame. Ablate:

   - no correction,
   - affine correction,
   - masked affine correction,
   - histogram or illumination correction.

8. **Post depth/free-space**

   If a post-only depth source becomes available, compare:

   - RGB multi-view score only,
   - depth free-space only,
   - RGB + free-space combined score.
