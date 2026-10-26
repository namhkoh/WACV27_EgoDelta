# EgoDelta Context Summary for Next Agent

Last updated: 2026-05-19

## Repository State

- Repo: `/path/to/workspace/EgoDelta`
- Branch: `jy`, tracking `origin/jy`.
- Current worktree is intentionally dirty because original 2DGS core files were restored from git history after the latest simplification removed files that are needed to train pre-scene 2DGS models for new datasets.
- Restored/modified code has not been committed.

Uncommitted restored 2DGS files:

- `train.py`
- `render.py`
- `metrics.py`
- `view.py`
- `scene/__init__.py`
- `gaussian_renderer/network_gui.py`
- `lpipsPyTorch/`
- `utils/image_utils.py`
- `utils/mcube_utils.py`
- `utils/mesh_utils.py`
- `utils/render_utils.py`

Documentation updated:

- `README.md`
- `tools/README.md`
- `corl2026_latest_pipeline_runbook.md`

Validation already done:

- `git diff --check` passes.
- `python -m py_compile ...` passes for restored 2DGS files.
- `train.py --help`, `render.py --help`, `metrics.py --help`, `view.py --help` imported successfully before the whitespace cleanup.

## Latest Method in This Codebase

Current production pipeline:

1. Build aligned pre/post COLMAP-style datasets.
2. Train a pre-scene 2DGS model with `train.py`.
3. Score pre Gaussians with multiview post residuals.
4. Run constrained post-scene opacity-drop optimization on the fixed pre Gaussian set.
5. Conservatively prune small disconnected mask fragments.
6. For leaf tasks only, compute retained-plant connection paths and `backbone_gap_or_rise` interaction targets.
7. For tomato tasks, do not reuse leaf target logic. Use removed tomato mask first; a tomato-specific fruit-stem junction target rule is still needed.

Main wrapper:

```bash
python tools/run_corl2026_latest_pipeline.py \
  --task <task> \
  --data_root <pre_post_dataset_root> \
  --single_trajectory \
  --pre_model <pre_2dgs_model_dir> \
  --out_root <output_root> \
  --object_type leaf
```

For tomato, run score/optimize manually, then use `filter_tomato_fruit_mask.py`, `split_tomato_fruit_instances.py`, `extract_tomato_fruit_shell_targets.py`, and `visualize_component_points.py` as documented in the runbook.

## gs5 Dataset

Raw data:

- Dataset root: `/path/to/workspace/CoRL2026/data/gs5`
- VRS: `/path/to/workspace/CoRL2026/data/gs5/52620ee3-586c-48cf-9400-bcd0dee57e42.vrs`
- MPS: `/path/to/workspace/CoRL2026/data/gs5/mps_52620ee3-586c-48cf-9400-bcd0dee57e42_vrs`

Capture sequence:

```text
pre -> during deleaf -> after deleaf -> tomato harvest -> after harvest
```

Extracted pinhole frames:

- Root: `data/corl2026_2dgs_gs5/pinhole_stride2`
- Source frames: 2248
- Extracted frames: 1124
- Frames with pose: 1115
- Rectified frames: `data/corl2026_2dgs_gs5/pinhole_stride2/frames_rect`

Manual phase split used:

- `pre`: frames `000020-000720`, 351 frames
- `during_deleaf`: frames `000780-001200`, 211 frames, not used as post
- `after_deleaf`: frames `001260-001620`, 181 frames
- `tomato_pre`: frames `001260-001680`, 211 frames
- `tomato_removed`: frames `001800-002220`, 211 frames

Selection files:

- `data/corl2026_2dgs_gs5/selections/pre_frames.txt`
- `data/corl2026_2dgs_gs5/selections/during_deleaf_frames.txt`
- `data/corl2026_2dgs_gs5/selections/deleafed_frames.txt`
- `data/corl2026_2dgs_gs5/selections/tomato_pre_frames.txt`
- `data/corl2026_2dgs_gs5/selections/tomato_removed_frames.txt`

Pre/post pair datasets:

- Leaf pair root: `data/corl2026_2dgs_gs5/leaf_pair`
- Tomato pair root: `data/corl2026_2dgs_gs5/tomato_pair`
- Both are single-trajectory splits with identity alignment.
- Both were built with MPS semidense init points using `--point_stride 25`, producing 57,871 init points.

## 2DGS Training Results

Leaf pre-scene 2DGS:

- Model: `output/corl2026_2dgs_gs5/gs5_leaf_pre`
- Final PLY: `output/corl2026_2dgs_gs5/gs5_leaf_pre/point_cloud/iteration_30000/point_cloud.ply`
- Final Gaussian count: 252,145
- This model was already available and was reused.

Tomato pre-scene 2DGS:

- Model: `output/corl2026_2dgs_gs5/gs5_tomato_pre`
- Final PLY: `output/corl2026_2dgs_gs5/gs5_tomato_pre/point_cloud/iteration_30000/point_cloud.ply`
- Final Gaussian count: 232,647
- Trained in this restored repo with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/corl2026_2dgs_gs5/tomato_pair/gs5_tomato_pre \
  -m output/corl2026_2dgs_gs5/gs5_tomato_pre \
  -r 2 \
  --quiet
```

Note: GPU 0 later caused OOM during scoring because an old VLLM process was occupying nearly all memory. Use GPU 3 or another free GPU for subsequent pipeline runs.

## gs5 Leaf Results

Pipeline task:

- Task: `gs5_leaf`
- Data root: `data/corl2026_2dgs_gs5/leaf_pair`
- Pre model: `output/corl2026_2dgs_gs5/gs5_leaf_pre`
- Output root: `output/corl2026_2dgs_gs5`

Stage outputs:

- Scores: `output/corl2026_2dgs_gs5/gs5_leaf_scores`
- Optimized mask: `output/corl2026_2dgs_gs5/gs5_leaf_presence_losscolor`
- Pruned mask: `output/corl2026_2dgs_gs5/gs5_leaf_component_pruned`
- Component targets: `output/corl2026_2dgs_gs5/gs5_leaf_component_targets_validated_mask_endpoint`
- Visual checks: `output/corl2026_2dgs_gs5/gs5_leaf_component_targets_validated_mask_endpoint_points_pre_color`
- Alignment QC: `output/corl2026_2dgs_gs5/gs5_leaf_alignment_qc`

Key numbers:

- Residual scoring: 9 clusters, 3,470 core candidates, score threshold 0.693128.
- Optimization:
  - candidate Gaussians: 22,834
  - graph-expanded Gaussians: 18,966
  - optimized selected Gaussians: 15,424
  - post-opt clusters: 4
  - loss: 0.046014 -> 0.042421
- Component pruning:
  - kept/pruned mask Gaussians: 15,179
  - removed tiny fragments: 245

Leaf target status:

- Component target extractor: `tools/extract_leaf_component_targets.py --target_method validated_mask_endpoint`
- Accepted leaf/stem instances: 4
- Robot targets: 3
- Rejected weak tomato-adjacent components: 5
- Instance 0 target xyz: `[0.23410962522029877, -0.3009028732776642, -0.10100371390581131]`, confidence `0.5111`
- Instance 1 target xyz: `[-0.17603150010108948, -0.284546434879303, -0.08981171250343323]`, confidence `0.4079`
- Instance 2 target xyz: `[0.32081109285354614, -0.29433926939964294, -0.2228221744298935]`, confidence `0.2493`
- Instance 3 has no target because the best endpoint candidate was rejected as `ambiguous_endpoint_selected_ahead`.

Interpretation:

- The removed mask and interaction target are intentionally separate.
- Do not complete the removed mask to the main stem; use accepted component endpoints and retained green support for target selection.
- Tomato-adjacent false positives are rejected before target extraction.

Important visual files:

- `output/corl2026_2dgs_gs5/gs5_leaf_component_targets_validated_mask_endpoint_points_pre_color/points_contact_sheet.jpg`
- `output/corl2026_2dgs_gs5/gs5_leaf_presence_losscolor/projection_images/contact_sheet.jpg`
- `output/corl2026_2dgs_gs5/gs5_leaf_alignment_qc/000_frame_001260_alignment_overlay.jpg`

## gs5 Tomato Results

Pipeline task:

- Task: `gs5_tomato`
- Data root: `data/corl2026_2dgs_gs5/tomato_pair`
- Pre model: `output/corl2026_2dgs_gs5/gs5_tomato_pre`
- Output root: `output/corl2026_2dgs_gs5`

Manual tomato mask pipeline used:

1. Residual scoring with red appearance prior and no plant ROI:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/score_gaussian_multiview_residuals.py \
  --task gs5_tomato \
  --data_root data/corl2026_2dgs_gs5/tomato_pair \
  --single_trajectory \
  --pre_model output/corl2026_2dgs_gs5/gs5_tomato_pre \
  --model_path output/corl2026_2dgs_gs5/gs5_tomato_red_scores \
  --resolution 2 \
  --frame_stride 1 \
  --score_quantile 0.95 \
  --cluster_eps 0.06 \
  --cluster_min_samples 3 \
  --cluster_max_bbox_diag 0.8 \
  --cluster_max_planarity 200.0 \
  --plant_roi_mode none \
  --appearance_roi_mode red
```

2. Opacity optimization with larger local expansion:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/optimize_gaussian_presence_post.py \
  --task gs5_tomato \
  --data_root data/corl2026_2dgs_gs5/tomato_pair \
  --single_trajectory \
  --pre_model output/corl2026_2dgs_gs5/gs5_tomato_pre \
  --score_npz output/corl2026_2dgs_gs5/gs5_tomato_red_scores/gaussian_multiview_residual_scores.npz \
  --cluster_ids all \
  --out_dir output/corl2026_2dgs_gs5/gs5_tomato_red_presence_losscolor \
  --resolution 2 \
  --frame_stride 1 \
  --iterations 800 \
  --drop_sparsity_weight 0.0001 \
  --loss_color_correction \
  --candidate_expand_radius 0.12 \
  --select_drop_min 0.2 \
  --select_opacity_drop_min 0.02 \
  --max_selected_bbox_diag 0.8 \
  --max_selected_planarity 200.0 \
  --post_cluster_eps 0.06
```

3. Conservative component pruning:

```bash
python tools/prune_gaussian_mask_components.py \
  --presence_npz output/corl2026_2dgs_gs5/gs5_tomato_red_presence_losscolor/presence_optimization_scores.npz \
  --score_npz output/corl2026_2dgs_gs5/gs5_tomato_red_scores/gaussian_multiview_residual_scores.npz \
  --out_dir output/corl2026_2dgs_gs5/gs5_tomato_red_component_pruned \
  --selected_clusters all \
  --component_radius 0.018 \
  --min_component_size 120 \
  --min_component_ratio 0.08 \
  --min_median_score 0.25
```

Key numbers:

- Red-prior residual scoring: 1 cluster, 57 candidates, score threshold 0.823333.
- Optimization:
  - candidate core: 56
  - expanded candidate Gaussians: 4,753
  - selected Gaussians: 2,906
  - post-opt clusters: 1, noise 0
  - loss: 0.041000 -> 0.038891
  - bbox diagonal: 0.3607 m
- Component pruning:
  - kept/pruned mask Gaussians: 2,905
  - removed fragments: 1

Interpretation:

- Tomato mask visually captures the removed red fruit and nearby fruit-stem/leaf structure as one object.
- This is good for removed-object mask extraction.
- It is not yet a robot interaction target. A tomato-specific fruit-stem junction target rule should be added next.

Important visual files:

- `output/corl2026_2dgs_gs5/gs5_tomato_red_verification_post/verification_removed_outline_contact_sheet.jpg`
- `output/corl2026_2dgs_gs5/gs5_tomato_red_verification_post/verification_zoom_contact_sheet.jpg`
- `output/corl2026_2dgs_gs5/gs5_tomato_alignment_qc/000_frame_001800_alignment_overlay.jpg`

## Alignment and Quality Notes

- Both leaf and tomato are single-trajectory splits. Identity alignment is expected and was used.
- Alignment QC images were generated for both phases.
- No gross rigid misalignment was observed.
- Alignment metric CSVs exist:
  - `output/corl2026_2dgs_gs5/gs5_leaf_alignment_qc/alignment_overlay_metrics.csv`
  - `output/corl2026_2dgs_gs5/gs5_tomato_alignment_qc/alignment_overlay_metrics.csv`
- Common depth ratios are low/moderate because these are changed scenes with sparse projected MPS/depth support, not necessarily because rigid alignment is bad.
- Pre render quality is usable for the current residual/opacity pipeline.

## GPU / Runtime Notes

- GPU 0 caused OOM during leaf scoring because a long-running VLLM process was using about 167 GB.
- GPU 3 had enough free memory and was used successfully for scoring/optimization/visualization.
- If a future run OOMs, check `nvidia-smi` first and choose a less occupied GPU.

## Next Recommended Work

1. Commit the restored source files and documentation if the user approves.
2. For a new dataset:
   - Extract pinhole frames.
   - Split phases carefully, excluding during-action frames.
   - Train or reuse the pre-scene 2DGS model.
   - Run the leaf or tomato commands in `corl2026_latest_pipeline_runbook.md`.
3. Always regenerate the point-projection contact sheet and inspect it before reporting robot targets.
   - Build single-trajectory split with identity alignment.
   - Train pre 2DGS in this repo using restored `train.py`.
   - Run leaf wrapper for leaf tasks, and manual red-prior score/opt/prune for tomato until tomato target logic is implemented.
