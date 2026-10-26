# Aria Tomato Change Baseline

This repository implements a reproducible Python baseline for finding removed tomato-scene objects from pre-action and post-action Project Aria MPS semi-dense point clouds. It targets harvesting and deleafing trials by comparing geometry that was present before an action and missing after it.

The baseline is deliberately geometric:

1. Load Aria MPS semi-dense point clouds.
2. Filter uncertain and weakly observed points.
3. Align pre/post clouds in the pre-action MPS world frame, optionally refining with ICP on stable trellis/background boxes.
4. Compute pre-to-post nearest-neighbor distances.
5. Compute M3C2 distances with `py4dgeo` when available.
6. Combine NN and M3C2 evidence into removed-point scores.
7. Cluster removed points with DBSCAN.
8. Convert clusters into task-specific target estimates.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate aria-tomato-change
python -m pip install -e .
python -m pytest
python -m tomato_cutpoints.cli --help
```

If `open3d` installation fails through conda on your platform:

```bash
conda install -c conda-forge numpy scipy pandas scikit-learn pyarrow pyyaml typer rich tqdm pytest laspy matplotlib
python -m pip install open3d lazrs projectaria-tools==2.1.2 py4dgeo==1.1.0 numba
python -m pip install -e .
```

## Expected Inputs

The usual Project Aria MPS input layout is:

```text
trial_pre/
  slam/
    semidense_points.csv.gz
    semidense_observations.csv.gz
    closed_loop_trajectory.csv
trial_post/
  slam/
    semidense_points.csv.gz
    semidense_observations.csv.gz
    closed_loop_trajectory.csv
```

`semidense_points.csv.gz` is required. `semidense_observations.csv.gz` is optional but recommended because observation counts are a useful quality filter.

## Commands

Harvest:

```bash
python -m tomato_cutpoints.cli run \
  --task harvest \
  --pre-points /path/pre/slam/semidense_points.csv.gz \
  --post-points /path/post/slam/semidense_points.csv.gz \
  --pre-observations /path/pre/slam/semidense_observations.csv.gz \
  --post-observations /path/post/slam/semidense_observations.csv.gz \
  --out-dir outputs/harvest_trial_001 \
  --config configs/harvest.yaml
```

Deleafing with stable-region ICP:

```bash
python -m tomato_cutpoints.cli run \
  --task deleaf \
  --pre-points /path/pre/slam/semidense_points.csv.gz \
  --post-points /path/post/slam/semidense_points.csv.gz \
  --stable-roi-json rois/stable_trellis_boxes.json \
  --refine-icp \
  --out-dir outputs/deleaf_trial_001
```

Evaluate predictions:

```bash
python -m tomato_cutpoints.cli evaluate \
  --pred outputs/harvest_trial_001/targets.json \
  --gt annotations/harvest_trial_001_gt.json
```

Run a synthetic smoke test:

```bash
python -m tomato_cutpoints.cli synthetic-demo \
  --task harvest \
  --out-dir outputs/synthetic_harvest
```

Open diagnostic PLYs interactively:

```bash
python -m tomato_cutpoints.cli view --run-dir outputs/harvest_trial_001
```

## Output Schema

Every run writes:

```text
config_resolved.yaml
diagnostics.json
quality_report.json
alignment_diagnostics.json
alignment_transform_post_to_pre.txt
removed_points.parquet
clusters.csv
cluster_points.parquet
cluster_features.json
targets.json
distances_nn_pre_to_post.npy
distances_nn_post_to_pre.npy
m3c2_distances.npy            # when available
plots/
  nn_distance_hist.png
  m3c2_distance_hist.png      # when available
  cluster_size_hist.png
plys/
  pre_filtered.ply
  post_filtered_aligned.ply
  removed_candidates.ply            # all points that passed removed-point scoring
  clustered_removed_candidates.ply  # only DBSCAN-clustered removed points, colored by cluster
  target_candidates.ply             # cluster(s) selected for targets.json
  cut_points.ply                    # cut point crosses plus removal-direction rays
```

`targets.json` contains one or more ranked targets in the pre-action MPS world frame:

```json
{
  "task": "harvest",
  "coordinate_frame": "pre_action_mps_world",
  "method": "aria_mps_semidense_nn_m3c2_dbscan",
  "transforms": {"T_post_to_pre": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
  "targets": []
}
```

Set `task_priors.<task>.max_targets` above `1` to emit multiple cut points.
When `min_target_score` is present, only ranked clusters at or above that score
become targets.
For adjacent tomatoes that DBSCAN merges into one removed cluster, enable
`task_priors.harvest.split_merged` and tune the `split_*` thresholds in
`configs/harvest.yaml`.

## Tuning Thresholds

Start with `configs/default.yaml`. Increase `quality.min_observations` and tighten `quality.max_dist_std` if the cloud is noisy. Increase `nn.threshold_m` or `m3c2.threshold_m` if stable trellis/background points are being marked as removed. Increase `dbscan.eps_m` for sparse Aria clouds, and increase `dbscan.min_cluster_points` when isolated missing points dominate.

When stable trellis/background regions are known, provide `stable_boxes` in a ROI JSON file and enable `--refine-icp`. Stable boxes also allow automatic NN thresholding from the residual alignment noise.

## Known Limitations

- Aria semi-dense clouds are partial and sparse, so small or weakly textured objects can be under-sampled.
- Pre/post frames may not be globally aligned unless they share a common MPS world frame or a reliable transform/ICP refinement is used.
- M3C2 sign convention may not directly correspond to removal direction, so this baseline uses magnitude plus NN evidence.
- Geometric-only cut point inference is approximate without RGB or semantic stem segmentation.

## Future Work

- Use RGB/semantic cues to distinguish tomato, leaf, vine, and trellis.
- Add M3C2-EP when scan-position and covariance information are available.
- Integrate trial-level calibration metadata and robot frame transforms.
- Add active ROI proposal from camera trajectory and viewing frustums.
