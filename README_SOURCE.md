# EgoDelta

EgoDelta recovers 3D robot action targets from egocentric pre/post state changes using a pre-scene 2D Gaussian Splatting model.

This branch is organized around the latest CoRL2026 pipeline while keeping the 2DGS training/rendering entrypoints needed to build pre-scene models for new datasets.

## Start Here

- Latest operational runbook: [corl2026_latest_pipeline_runbook.md](corl2026_latest_pipeline_runbook.md)
- Method notes and verified results: [corl2026_change_detection_algorithm.md](corl2026_change_detection_algorithm.md)
- One-command runner: [tools/run_corl2026_latest_pipeline.py](tools/run_corl2026_latest_pipeline.py)
- Tool inventory: [tools/README.md](tools/README.md)
- Four-person image-mask annotation workflow: [annotation_tool/README.md](annotation_tool/README.md)
- Final mechanics-conditioned target extractor: [target_extraction/README.md](target_extraction/README.md)

## Latest Leaf Pipeline

```bash
python tools/run_corl2026_latest_pipeline.py \
  --task <task> \
  --data_root <pre_post_dataset_root> \
  --single_trajectory \
  --pre_model <pre_2dgs_model_dir> \
  --out_root <output_root> \
  --object_type leaf
```

Use `--dry_run` to print the stage commands without running them. Use `--stages score,optimize,prune` to run only part of the pipeline.

The current leaf/deleafing flow is:

1. Gaussian multiview residual scoring.
2. Constrained opacity-drop optimization on the fixed pre-scene Gaussian set.
3. Conservative component pruning for the removed-object mask.
4. Component-level leaf target extraction with `validated_mask_endpoint`.
5. Pre-scene point projection sanity check.

## Repository Layout

- `tools/`: pipeline entrypoints, dataset preparation, visualization, and validation utilities.
- `change_detection/`: shared alignment, projection, scoring, mask, and export helpers.
- `train.py`, `render.py`, `metrics.py`, `view.py`: upstream 2DGS entrypoints kept for pre-scene reconstruction and QC.
- `scene/`, `gaussian_renderer/`, `utils/`, `arguments/`, `lpipsPyTorch/`: 2DGS runtime needed to train, load, render, and evaluate pre-scene Gaussians.
- `baseline/`: baseline implementations kept for comparison work.
- `scripts/setup_env.sh`: environment setup.

## Data Assumptions

The latest pipeline expects:

- aligned pre/post COLMAP-style datasets,
- a trained pre-scene 2DGS model,
- verified pre/post alignment,
- post frames corresponding to the interaction phase.

For one-trajectory captures, use `--single_trajectory`.

## Environment

The project depends on the pinned Python/CUDA stack in `environment.yml` and the CUDA extensions in:

- `submodules/diff-surfel-rasterization`
- `submodules/simple-knn`

Setup:

```bash
./scripts/setup_env.sh
conda activate surfel_splatting
```

## Pre-Scene 2DGS Training

For a new dataset, train the pre-scene model before running EgoDelta:

```bash
python train.py \
  -s <pre_dataset_dir> \
  -m <pre_2dgs_model_dir> \
  -r 2 \
  --quiet
```

Then pass `<pre_2dgs_model_dir>` as `--pre_model` to `tools/run_corl2026_latest_pipeline.py`.
