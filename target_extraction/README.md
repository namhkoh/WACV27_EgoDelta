# EgoDelta mechanics-conditioned target extraction

This directory contains the selected geometry-only Stage-3 method. It reads the
frozen continuous Stage-2 cache and does not modify reconstruction or change
scoring.

## A-priori endpoint-side routing

The retargeting primitive already specifies which side of an intervention
interface the tool acts on. Stage 3 uses that pre-existing mechanical convention:

- `EndpointSide.RETAINED` for sever/cut primitives;
- `EndpointSide.REMOVED` for detach/grasp-payload primitives.

This is not a crop classifier or an empirically selected switch. No crop/task
label and no proximity threshold enter routing.

The two endpoint sides require different estimators for a principled reason:

- **Retained side.** The retained structure persists. Its target is
  distinguished from the rest of that structure by contact with removed
  material, so the decoder constructs a retained-side contact field, finds its
  modes, and selects the interface-boundary mode.
- **Removed side.** The removed object is present in the pre-action
  reconstruction, and the target is its own attachment feature. The decoder
  separates dense object modes, completes each mode to an exhaustive partition,
  and finds the locally refined gravity-facing surface point.

Both decoders share the continuous Stage-2 change evidence and standard DBSCAN
component proposal. Neither consumes color, a crop-specific shape rule, PCA
endpoint selection, or a consensus vote.

## Reproduction

Run the retained-side deleafing evaluation:

```bash
python scripts/run_deleafing.py \
  --cache-dir stage2_cache_phase1 \
  --gt gt.md \
  --output-dir final_runs/deleafing_all30 \
  --verify-reference
```

Run the removed-side harvesting evaluation:

```bash
python scripts/run_harvesting.py \
  --cache-dir stage2_cache_phase1 \
  --gt gt.md \
  --output-dir final_runs/harvesting_all30 \
  --verify-reference
```

The reported thresholds were selected in-sample over the same 30 scenes per
task. The world `+Z` harvesting direction encodes the physical prior that fruit
hangs from its support; it is not valid for horizontally trained trusses and
requires potted-plant confirmation.

Paper packaging:

- `results.md`: final tables, method framing, and negative results;
- `PARAMETER_AUDIT.md`: prediction-affecting constants and legacy call-graph audit;
- `REPRODUCTION.md`: one command for every final reported result;
- `paper_assets/`: qualitative panels and their deterministic case index.
