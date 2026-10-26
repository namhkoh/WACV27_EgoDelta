# Scene-change baselines for WACV 2027

This directory provides one EgoDelta data/evaluation interface for three baselines:

| Method | Implementation used here | Output used for evaluation |
|---|---|---|
| 3DGS-CD | Pinned official repository | Move-out mask at the fixed pre-change benchmark camera |
| MV3DCD | Pinned official repository | Learned change channel rendered at the fixed benchmark camera |
| GS-DIFF | Our paper-faithful reimplementation | Symmetric primitive-change score rendered from both reconstructions |

The same interface also renders the final mechanics-conditioned **EgoDelta**
prediction. It regenerates the GT-free Stage-2 primitive-comparison cache from
the existing pre/post 2DGS models, runs the retained-side decoder for deleafing
or the removed-side decoder for harvesting, and projects each accepted removed
component into the fixed query cameras.

Every method writes the same layout:

```text
predictions/<method>/<scene_id>/
├── masks/<frame>.png       # binary mask
├── scores/<frame>.npy      # float32 score in [0, 1]
└── metadata.json           # provenance and method diagnostics
```

The frozen benchmark is the EgoMask manifest at
`annotation_runs/wacv2027_masks_12views/manifest.json`: 60 scene-intervention
units and 12 fixed pre-change views per unit (720 masks). Evaluation pools
the 12 views within each scene-intervention and then macro-averages the 60
independent units.

## Important interpretation

The GS-DIFF repository had no runnable implementation on 2026-08-15. The code
in `egoscene/gsdiff.py` follows Algorithm 1 and the appendix equations in
arXiv:2605.07203v2. We validate that implementation using two independently
trained models from the paper's official PGSR reconstruction backend on
PASLCD. For EgoDelta experiments, the same scorer consumes our already trained
2DGS surfels. It is therefore **our paper-faithful reimplementation**, not
official GS-DIFF; describe the EgoDelta variant as adapted to 2DGS.

There is also a scientific overlap to resolve before writing the paper:
EgoDelta's existing target-extraction Stage 2 already implements a GS-DIFF-like
primitive comparator. If the evaluated method output is the Stage 2 change map
itself, GS-DIFF is circular rather than an independent baseline. It is valid as
a “GS-DIFF core / before target decoder” ablation when the evaluated EgoDelta
system adds a distinct target-decoding stage. Keep this distinction explicit in
tables and text.

The learned official methods use 120 pre-change views and, by default, 10
trajectory-spanning post-change views. The adapted GS-DIFF consumes the two
existing reconstructions, each trained from 120 views. Report this input-budget
difference; do not imply that every method saw the same number of post views.

## One-time setup

From the EgoDelta repository root:

```bash
bash baseline/scene_change/scripts/setup_upstreams.sh
bash baseline/scene_change/scripts/setup_envs.sh common
bash baseline/scene_change/scripts/setup_envs.sh 3dgscd
bash baseline/scene_change/scripts/setup_envs.sh pgsr
```

The first command checks out exact upstream revisions recorded in
`upstreams.lock.json`. The common environment is shared by MV3DCD, GS-DIFF and
evaluation. 3DGS-CD has a separate Python/CUDA environment because its official
dependency versions are older.

## Validate implementations on paper data

Do this before running EgoDelta experiments. The data folders themselves are
large and are not committed; the commands below assume the official Mustard
and PASLCD downloads are present under the pinned upstream repositories.

```bash
# Official MV3DCD, exact 7k/10k/13k schedule on PASLCD.
bash baseline/scene_change/scripts/validate_mv3dcd_paper.sh Cantina Instance_1 1

# Official 3DGS-CD inference on the released Mustard checkpoint and inputs.
bash baseline/scene_change/scripts/validate_3dgscd_paper.sh 2
source baseline/scene_change/scripts/env_paths.sh
$EGOSCENE_COMMON_ENV/bin/python baseline/scene_change/scripts/score_3dgscd_paper.py \
  --scene .external/3DGS-CD/data/Mustard \
  --output baseline/scene_change/runs/paper_validation/3dgscd/Mustard

# GS-DIFF: split PASLCD without reference/post leakage, train two official
# PGSR reconstructions in the shared COLMAP frame, then score on GPU 2.
$EGOSCENE_COMMON_ENV/bin/python baseline/scene_change/scripts/prepare_gsdiff_paper.py \
  --source .external/MV3DCD/data/PASLCD/Cantina/Instance_1 \
  --output baseline/scene_change/runs/paper_validation/gsdiff/Cantina/Instance_1/data
bash baseline/scene_change/scripts/train_pgsr_paper.sh reference 0 30000
bash baseline/scene_change/scripts/train_pgsr_paper.sh inference 1 30000
bash baseline/scene_change/scripts/validate_gsdiff_paper.sh 2 30000
```

Current reproducibility checks on `Cantina/Instance_1` and Mustard:

- MV3DCD: mean IoU 0.5961 and mean F1 0.7429, versus 0.591/0.737 in its paper.
- 3DGS-CD: the current official code/released assets produce move-in IoU
  0.9340 and F1 0.9659 on Mustard. This is a strong functional reproduction,
  though below the paper's 0.9648/0.9821; retain that discrepancy in records.
- GS-DIFF: our paper-faithful implementation produces fixed-threshold mIoU
  0.3472 and F1 0.4928 on the single available `Cantina/Instance_1` run. The
  oracle diagnostic is mIoU 0.3688 and F1 0.5123 at threshold 0.6. The paper
  reports 0.644/0.758 only as an average over all 20 PASLCD instances and
  publishes neither per-instance results nor runnable code, so this is a
  functional paper-data validation rather than a numerical reproduction of
  the reported aggregate. Always label it as our paper-faithful
  reimplementation. The fixed 0.5 threshold is primary; the oracle number is
  a calibration diagnostic and is not reportable as a baseline result.

Run the audit and unit tests:

```bash
source baseline/scene_change/scripts/env_paths.sh
$EGOSCENE_COMMON_ENV/bin/python baseline/scene_change/scripts/doctor.py
$EGOSCENE_COMMON_ENV/bin/python -m pytest baseline/scene_change/tests -q
```

## Prepare one scene

Use a scene ID from the annotation manifest, for example
`egodelta_040bc88b3760_leaf`:

```bash
source baseline/scene_change/scripts/env_paths.sh
PY=$EGOSCENE_COMMON_ENV/bin/python
RUNS=baseline/scene_change/runs

$PY -m egoscene.cli prepare --baseline mv3dcd \
  --scene egodelta_040bc88b3760_leaf \
  --output-root $RUNS/data/mv3dcd --post-views 10

$PY -m egoscene.cli prepare --baseline 3dgscd \
  --scene egodelta_040bc88b3760_leaf \
  --output-root $RUNS/data/3dgscd --post-views 10
```

Preparation is deterministic and uses symlinks rather than copying RGB data.
Known Aria MPS/COLMAP poses are preserved; HLoc is not rerun.
For 3DGS-CD, two minimal tracked upstream patches fix robustness bugs without
changing its algorithm: one repairs the broken `refine_pose=False` branch so
the supplied benchmark MPS poses remain fixed and evaluation RGB is not used
in pose optimization; the other represents a view with no image-difference
contours as an empty mask tensor instead of crashing at `torch.stack`.

## Run one scene

GPU numbers below refer to physical GPUs on `compute-node-1`.

```bash
# GS-DIFF reimplementation, existing pre/post 2DGS models
source baseline/scene_change/scripts/env_paths.sh
CUDA_VISIBLE_DEVICES=2 $EGOSCENE_COMMON_ENV/bin/python -m egoscene.cli gsdiff \
  --scene egodelta_040bc88b3760_leaf \
  --output-root baseline/scene_change/runs/predictions/gsdiff

# Official MV3DCD schedule: 7k reference, 10k change, 13k augmentation
bash baseline/scene_change/scripts/run_mv3dcd_scene.sh \
  egodelta_040bc88b3760_leaf \
  baseline/scene_change/runs/data/mv3dcd \
  baseline/scene_change/runs/models/mv3dcd \
  baseline/scene_change/runs/predictions/mv3dcd 2

# Official 3DGS-CD: train pre-change Splatfacto, then detect/project move-out
bash baseline/scene_change/scripts/run_3dgscd_scene.sh \
  egodelta_040bc88b3760_leaf \
  baseline/scene_change/runs/data/3dgscd \
  baseline/scene_change/runs/models/3dgscd \
  baseline/scene_change/runs/predictions/3dgscd 2
```

The fixed-pose benchmark renderer follows the official final-mask path: it
renders the 13k change channel and gates it with alpha support from the frozen
7k reference reconstruction. The official 10k augmentation render does not
replace that reference alpha support.

For a quick code-path check only, GS-DIFF accepts `--max-primitives 500
--max-neighbors 64`. Never report results generated with those debug limits.

## Run all scenes

The batch launcher prepares data when needed and resumes completed scene
outputs:

```bash
bash baseline/scene_change/scripts/run_all_scenes.sh gsdiff 0
bash baseline/scene_change/scripts/run_all_scenes.sh mv3dcd 1
bash baseline/scene_change/scripts/run_all_scenes.sh 3dgscd 2

# EgoDelta on all 720 fixed benchmark views.
bash baseline/scene_change/scripts/run_egodelta_scenes.sh 0
```

The three commands are intended for three separate `tmux` windows. Verify GPU
ownership with `nvidia-smi` before starting them. The launcher does not kill or
replace any existing process.

The EgoDelta launcher is also resumable: a Stage-2 cache is retained after it
is written, and a scene is complete only when its prediction `metadata.json`
exists. It renders all fixed queries by default. Set `ANNOTATOR=annotator_2`
only for an explicitly scoped diagnostic subset. Scene-selected cache
generation does not read `target_extraction/gt.md`; the 3D point GT and
corrected image masks enter only during evaluation.

For independent round-robin shards, use `run_scene_shard.sh`. This is the
recommended way to occupy several GPUs without training the same scene twice:

```bash
# Four terminals or persistent tmux windows.
bash baseline/scene_change/scripts/run_scene_shard.sh 3dgscd 0 0 4
bash baseline/scene_change/scripts/run_scene_shard.sh 3dgscd 1 1 4
bash baseline/scene_change/scripts/run_scene_shard.sh 3dgscd 2 2 4
bash baseline/scene_change/scripts/run_scene_shard.sh 3dgscd 3 3 4
```

The 3DGS-CD runner assigns a separate PyTorch CUDA-extension cache to each GPU
shard. This avoids the upstream gsplat shared lock-file race when several
official trainings start simultaneously.

## Evaluate after annotation is frozen

Export the completed EgoMask annotations as a directory containing
`dataset.json` and its label maps. Then run each method independently:

```bash
source baseline/scene_change/scripts/env_paths.sh
$EGOSCENE_COMMON_ENV/bin/python -m egoscene.cli evaluate \
  --gt-export /path/to/frozen_egomask_export \
  --predictions-root baseline/scene_change/runs/predictions/gsdiff \
  --output-json baseline/scene_change/runs/eval/gsdiff.json \
  --output-csv baseline/scene_change/runs/eval/gsdiff_images.csv
```

The evaluator reports IoU, F1, precision and recall at image level and at the
primary scene-intervention level. Ground truth is used only here, never during
baseline preparation, training, threshold selection or mask generation.

While a batch is still running, add `--allow-incomplete-predictions` to score
only scene directories that contain completion metadata. The resulting JSON
reports explicit image and scene coverage; treat it as a progress diagnostic,
not a final benchmark result.

## Figure-3-style qualitative comparison

After all four prediction roots reach 720/720 coverage, create the qualitative
grid inspired by Figure 3 of arXiv:2412.03911v1:

```bash
source baseline/scene_change/scripts/env_paths.sh
$EGOSCENE_COMMON_ENV/bin/python baseline/scene_change/scripts/make_figure3.py \
  --gt-export annotation_runs/image_masks_gt_wacv2027_human_final_720_20260817 \
  --3dgscd-root baseline/scene_change/runs/predictions/3dgscd \
  --mv3dcd-root baseline/scene_change/runs/predictions/mv3dcd \
  --gsdiff-root baseline/scene_change/runs/predictions/gsdiff \
  --egodelta-root baseline/scene_change/runs/final_720_20260817/predictions/egodelta \
  --output-stem baseline/scene_change/runs/figures/final_720_20260817/figure3_comparison
```

The script uses a fixed seed to select five distinct deleafing interventions
and five distinct harvesting interventions, then one benchmark view from each.
Every task ID and input path is recorded beside the PNG/PDF in a JSON file, so
the qualitative panel is reproducible and not manually cherry-picked. Each
method cell shows the RGB overlay above its standalone binary prediction; the
last column shows the input RGB above the human ground truth.

## Reproducibility rules

- Keep the manifest and selected post-frame list fixed before full runs.
- Use threshold 0.5 for the primary comparison unless a validation-only tuning
  protocol is declared before looking at test masks.
- Store `metadata.json` with every result and archive `upstreams.lock.json`.
- Label GS-DIFF results as a paper-faithful reimplementation adapted to 2DGS.
- Preserve upstream license notices. MV3DCD is non-commercial research code;
  the GS-DIFF repository is CC BY-NC 4.0.
