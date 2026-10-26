# EGODELTA Diff Target Extraction

Primitive-level scene-difference scoring and 3D intervention-target recovery for
pre/post 2D Gaussian Splatting reconstructions.

## Package Layout

```text
egodelta_diff/
  data.py              2DGS IO and benchmark GT parsing
  primitives.py        primitive geometry, scoring, drift, frustum, and clustering
  contact_mass_extraction.py  crop-agnostic two-sided interface estimator
  contact_proposal.py  contact-side target proposal extraction
  targets.py           baseline proposals, selectors, and target metrics
  visualization.py     optional rendering and overlay helpers
  pipeline.py          end-to-end scene-difference pipeline
```

The top-level `scripts/` directory contains the release-facing commands:
`run_target_extraction.py`, `evaluate_targets.py`, and `scoring_ablation.py`.

## Contact-mass interface estimator

`extract_contact_mass_targets` consumes the continuous Stage-2 `delta` and
`omega` arrays and returns refined interface pairs `(g_ret, g_rem)`.  It has no
task, crop, or RGB input.  A downstream tool convention selects the retained or
removed endpoint:

```python
result = extract_contact_mass_targets(mu, normals, delta, omega, overlap)
cut_points = result.points("retained")  # cutter-side convention
grasp_points = result.points("removed") # removed-object convention
```

The confidence partition is `w_rem = delta` and `w_ret = omega - delta`.
Components are formed with DBSCAN; contact-field peaks are locally refined and
suppressed at the 3 cm tool/evaluation resolution.  Every emitted peak exposes
both local contact mass (`conf_c`) and parameter-free graph `prominence`.  The
latter uses the existing sigma-neighborhood contact graph and adds no estimator
scale.  No prominence threshold is hardcoded: operating points must be selected
on development data and evaluated on held-out scenes.  There is no
component-mass percentile filter.

## Environment

```bash
conda activate egodelta
cd target_extraction
python -m pip install -e .
```

The pipeline expects each 2DGS model directory to contain:

```text
<model>/
  cameras.json
  point_cloud/iteration_*/point_cloud.ply
```

## Run Target Extraction

```bash
python scripts/run_target_extraction.py \
  --pre-model /path/to/pre_model \
  --post-model /path/to/post_model \
  --out-dir outputs/example_scene
```

Batch over basename-matched pre/post model directories:

```bash
python scripts/run_target_extraction.py \
  --only-ready \
  --pre-root /path/to/pre_models \
  --post-root /path/to/post_models \
  --out-root outputs \
  --skip-existing
```

For a quick smoke test, add `--max-primitives 500 --disable-render --disable-overlay`.

## Outputs

- `removed_primitives.ply`: pre-action primitives classified as removed.
- `removed_clusters.json`: clustered removed-support components.
- `contact_target_proposals.json`: GT-free contact-point target proposals.
- `contact_target_proposals.ply`: proposal points for 3D inspection.
- `contact_proposal_eval.json`: proposal diagnostics when GT is available.
- `target_matches.json` and `selected_targets.ply`: GT target matches when enabled.
- `diagnostics.json`: stage statistics and configuration.
- `change_map_view_*.png` and overlays: optional visualization outputs.

## Evaluation And Ablations

```bash
python scripts/evaluate_targets.py \
  --outputs-root outputs \
  --gt gt.md \
  --mode contact \
  --write-proposals
```

Run primitive-scoring ablations:

```bash
python scripts/scoring_ablation.py run \
  --base-config configs/objectness_full.yaml \
  --variants full,geometry_only,without_confidence,without_view_overlap \
  --gt gt.md \
  --require-gt \
  --out-root outputs_scoring_ablation \
  --skip-existing
```

Summarize ablations with the paper target-selection policy:

```bash
python scripts/scoring_ablation.py summarize \
  --outputs-root outputs_scoring_ablation \
  --gt gt.md \
  --summary-json experiment_logs/scoring_ablation_summary.json \
  --per-scene-csv experiment_logs/scoring_ablation_per_scene.csv \
  --markdown experiment_logs/scoring_ablation_summary.md
```

## Verify

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall egodelta_diff scripts tests
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
```
