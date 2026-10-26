# WACV prospective robot-trial amendment

This amendment supersedes the 20-direct-trial and 20-policy-trial counts in
`deliverables_latest.md` for the new prospective cohort. It does not rewrite or
pool the historical pilot results.

## Final cohort

- Direct EGODELTA-target execution: 30 harvesting and 30 deleafing.
- Learned policy: 30 harvesting and 30 deleafing.
- Total: 120 physical rollouts.

The 120 rows are pre-registered before execution. Every failed plan, automatic
rejection, operator rejection, safety stop, physical failure, missing log, and
successful execution remains in the denominator.

The evaluation uses 30 scene/target slots per task. Direct and learned-policy
conditions are paired on each slot, use the same frozen pre-execution EGODELTA
target hash, and are randomized in order within the pair. The physical fixture
is reset between conditions.

The first collection batch is 30 rollouts: harvest scene slots 1-15, with one
direct and one policy run per slot. The remaining three 30-rollout batches cover
harvest slots 16-30 and both halves of the deleaf cohort.

These 120 trials are held-out evaluation. Their trajectories and outcomes must
not be used to retrain either policy until the complete evaluation and final
result-provenance manifest are frozen.
