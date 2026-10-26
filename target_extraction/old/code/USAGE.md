# Target Extraction Usage

Use `egodelta_diff` for primitive-level scene-difference scoring and 3D target
recovery.

```bash
cd target_extraction
conda activate egodelta
python -m pip install -e .
```

## Use the two-sided estimator

The geometry-only Stage-3 API returns both endpoints of each detected
interface.  Select a side by mechanics, without passing a task or crop label:

```python
from egodelta_diff import extract_contact_mass_targets

result = extract_contact_mass_targets(mu, normals, delta, omega, overlap)
retained_targets = result.points("retained")
removed_targets = result.points("removed")
```

Use a configuration with `deployment_offset_m=0` for GT evaluation.

## Run A Pair

```bash
python scripts/run_target_extraction.py \
  --pre-model /path/to/pre_model \
  --post-model /path/to/post_model \
  --out-dir outputs/example_scene
```

For a smoke test, add `--max-primitives 500 --disable-render --disable-overlay`.

## Run Ready Pairs

```bash
python scripts/run_target_extraction.py \
  --only-ready \
  --pre-root /path/to/pre_models \
  --post-root /path/to/post_models \
  --out-root outputs \
  --skip-existing
```

## Evaluate Existing Outputs

```bash
python scripts/evaluate_targets.py \
  --outputs-root outputs \
  --gt gt.md \
  --mode contact \
  --write-proposals
```

## Run Scoring Ablations

```bash
python scripts/scoring_ablation.py run \
  --base-config configs/objectness_full.yaml \
  --variants full,geometry_only,without_confidence,without_view_overlap \
  --gt gt.md \
  --require-gt \
  --out-root outputs_scoring_ablation \
  --skip-existing
```

Optional variants:

```bash
--variants full,geometry_only,without_confidence,without_view_overlap,appearance_only,fixed_radius
```

Summarize the ablation:

```bash
python scripts/scoring_ablation.py summarize \
  --outputs-root outputs_scoring_ablation \
  --gt gt.md \
  --summary-json experiment_logs/scoring_ablation_summary.json \
  --per-scene-csv experiment_logs/scoring_ablation_per_scene.csv \
  --markdown experiment_logs/scoring_ablation_summary.md
```

To verify the `full` variant against a saved reference target JSON:

```bash
python scripts/scoring_ablation.py summarize \
  --outputs-root outputs_scoring_ablation \
  --gt gt.md \
  --summary-json experiment_logs/scoring_ablation_summary.json \
  --per-scene-csv experiment_logs/scoring_ablation_per_scene.csv \
  --markdown experiment_logs/scoring_ablation_summary.md \
  --reference-target-json /path/to/reference_targets.json \
  --check-reference-full
```

## Verify

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall egodelta_diff scripts tests
```
