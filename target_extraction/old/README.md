# Archived target-extraction work

This directory preserves code and artifacts that are not part of the active
selected method:

- `code/`: the former multi-variant package, legacy pipeline, ablation scripts,
  configs, tests, and release documentation;
- `experiments/`: generated full-scale runs, diagnostic sweeps, legacy outputs,
  and smoke caches;
- `experimental_results.md`: the complete chronological experimental record,
  including negative results and harvesting applicability diagnostics.

Nothing below this directory is imported or invoked by the active package.
The active frozen Stage-2 cache remains at `../stage2_cache_phase1/` because it
is the input to the final reproducibility run.
