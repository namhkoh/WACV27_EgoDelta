# Final Stage-3 reproduction manifest

Paper revision: `a15262713a00e48b2e7aca2c631f1dd8cac832c0`

Branch: `redacted/WACV`

Frozen Stage-2 cache set: `target_extraction/stage2_cache_phase1`
Aggregate cache checksum recorded by the final replay:
`b79e8683532ca1ef3c96f272389050093a69dcd2c0ac8556d2198adb464bc9db`.

All commands below start at the repository root. They read the frozen caches;
none reruns Stage 1 or Stage 2. The requested `corl` environment is not present
on this server; the reproducible local environment is `egodelta` (NumPy 1.26.4,
SciPy 1.15.3, scikit-learn 1.7.2, Matplotlib 3.10.9).

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate egodelta
export MPLCONFIGDIR=/tmp/egodelta_mpl
cd /home/user/CoRL/EgoDelta
```

## Final head-to-head table

Each command below regenerates the complete row (P/R/F1, median error, and
TP/FP/FN) rather than one isolated scalar.

**Ours, deleafing: 91.59 / 92.45 / 92.02, 1.060 cm, 98/9/8**

```bash
python target_extraction/scripts/run_deleafing.py \
  --cache-dir target_extraction/stage2_cache_phase1 \
  --gt target_extraction/gt.md \
  --output-dir /tmp/egodelta_final_deleafing \
  --verify-reference
```

**Ours, harvesting: 94.44 / 94.44 / 94.44, 1.395 cm, 51/3/3**

```bash
python target_extraction/scripts/run_harvesting.py \
  --cache-dir target_extraction/stage2_cache_phase1 \
  --gt target_extraction/gt.md \
  --output-dir /tmp/egodelta_final_harvesting \
  --verify-reference
```

**Legacy, identical cache: deleafing 90.91 / 84.91 / 87.80 at 1.182 cm;
harvesting 90.74 / 90.74 / 90.74 at 1.584 cm**

The packaging wrapper loads the archived proposal call graph and the frozen
cache, then writes identical-cache summaries:

```bash
python target_extraction/paper_assets/reproduce_legacy.py \
  --cache-dir target_extraction/stage2_cache_phase1 \
  --gt target_extraction/gt.md \
  --output-dir /tmp/egodelta_final_legacy \
  --verify-reference
```

Its expected deterministic totals are harvesting 49/5/5 and deleafing
90/9/16. The archived script also explains why the historical deleafing table
had one fewer prediction: a generated JSON was manually replaced before the
historical CSV was produced, not code drift or nondeterminism.

## Necessity and proximity tables

**All four cross-application rows, both proximity definitions, binned success,
and the diagnostic proximity-switch result**

```bash
python target_extraction/old/experiments/contact_mass_runs/decoder_cross_application/run_cross_application.py
```

This reproduces native F1 92.02/94.44%, cross-applied F1 0.18/11.76%, the
0.084/1.281 cm component-boundary medians, and the diagnostic proximity plots.

**Harvesting world-up versus retained-facing direction**

```bash
python target_extraction/old/experiments/contact_mass_runs/harvesting_geometry_peak_gate/run_direction_comparison.py
```

This reproduces the retained-facing degradation cited in `results.md`.

## Selector and gate evidence

**Boundary selector: one-sidedness, Gaussian distance, and product**

```bash
python target_extraction/old/experiments/contact_mass_runs/deleafing_peak_selection_e478073/run_final_boundary_ablation.py
```

**Deleafing normalized component-mass gate and all normalization forms**

```bash
python target_extraction/old/experiments/contact_mass_runs/deleafing_peak_selection_e478073/sweep_mc_normalizations_all30.py
```

**Harvesting component-gate sweep**

```bash
python target_extraction/old/experiments/contact_mass_runs/task_specific_filter_e478073/sweep_all30_sweet_spot.py
```

**Harvesting mode decomposition and `min_cluster_size` sweep**

```bash
python target_extraction/old/experiments/contact_mass_runs/harvesting_geometry_peak_gate/run_partition_completion.py
```

## Closed negative-result rows

These are deliberately archived diagnostics. Each command regenerates the row
or group of rows named in the comment.

```bash
# local drift
python target_extraction/old/experiments/contact_mass_runs/diagnostics_local_drift_167f9b1/run_local_drift.py

# nearest approach and oracle localization limits
python target_extraction/old/experiments/contact_mass_runs/oracle_masks_e478073/run_oracle.py

# multiplicative action fusion
python target_extraction/old/experiments/contact_mass_runs/oracle_action_ranking_e478073/run_action_ranking.py

# event segmentation
python target_extraction/old/experiments/contact_mass_runs/event_gate_e478073/run_event_gate.py

# revealed surface, reciprocal matching, component-feature products
python target_extraction/old/experiments/contact_mass_runs/feature_separability_e478073/extract_features.py
python target_extraction/old/experiments/contact_mass_runs/feature_separability_e478073/analyze_features.py

# prominence and drift-layer diagnostics
python target_extraction/old/experiments/contact_mass_runs/diagnostics_drift_prominence_167f9b1/run_prominence.py
python target_extraction/old/experiments/contact_mass_runs/diagnostics_drift_prominence_167f9b1/run_drift_diagnostics.py

# absence depth
python target_extraction/old/experiments/contact_mass_runs/absence_depth_all60/run_absence_depth.py

# scene-relative quantile gates
python target_extraction/old/experiments/contact_mass_runs/scene_relative_adequacy_gate_all60/run_audit.py

# footprint/voxel extent
python target_extraction/old/experiments/contact_mass_runs/physical_extent_gate_all60/run_audit.py

# mean delta alone
python target_extraction/old/experiments/contact_mass_runs/mean_delta_gate_all60/run_sweep.py

# post-side retained field
python target_extraction/old/experiments/contact_mass_runs/diagnostics_peak_reformulation/run_post_side_diagnostic.py
```

## Qualitative panels

The renderer selects five correctly matched targets at global matched-error
quantiles 0/25/50/75/100% and all final false-negative GT cases (8 deleafing,
3 harvesting). It colors retained primitives by `rho` for deleafing. For
harvesting, it colors the removed component by a visualization-only
delta-weighted local density while showing the actual frozen dense-mode peaks
and selected target.

```bash
python target_extraction/paper_assets/render_final_qualitatives.py \
  --cache-dir target_extraction/stage2_cache_phase1 \
  --gt target_extraction/gt.md \
  --final-runs target_extraction/final_runs \
  --output-dir target_extraction/paper_assets
```

Outputs:

- `paper_assets/deleafing_qualitative_panels.{png,pdf}`
- `paper_assets/harvesting_qualitative_panels.{png,pdf}`
- `paper_assets/qualitative_case_index.csv`
- `paper_assets/qualitative_manifest.json`

## Integrity checks

```bash
git show --stat --oneline a15262713a00e48b2e7aca2c631f1dd8cac832c0
git diff --check
python -m compileall -q target_extraction/egodelta_diff target_extraction/scripts target_extraction/paper_assets
python -m pytest -q target_extraction/tests
```

The reference checks in the two final runners use absolute tolerance `1e-12`
for floating metrics and exact integer equality for counts.
