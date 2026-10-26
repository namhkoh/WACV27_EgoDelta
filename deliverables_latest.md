# Latest WACV paper deliverables

This is the working checklist for freezing the remaining paper evidence. Robot
and runtime records are the immediate priority because they require the local
robot machine and original experiment logs.

## Global rules

- These are verification and packaging tasks, not tuning experiments.
- Keep evaluation data, reconstructions, caches, thresholds, gates, proposals,
  adapters, and matching rules frozen.
- Never remove failed, rejected, stopped, missing, or inconvenient trials.
- Never estimate missing values. Record `not_recorded` and the reason.
- Use the target available before execution, never a post-hoc correction.
- Compare coordinates only after expressing them in the same named frame.
- Preserve raw paths, commands, timestamps, commits, configurations, and hashes.
- Write to a new timestamped directory; never overwrite an existing export.

Recommended output directory:

```text
paper_deliverables_YYYYMMDD_HHMMSS/
```

## 1. Fresh robot-scene target-accuracy export — highest priority

Create `robot_target_accuracy.csv`, with one row for every recovered target used
in the 20 direct-execution trials.

Required columns:

```text
trial_id, scene_id, task, target_id,
predicted_target_x, predicted_target_y, predicted_target_z,
socket_gt_x, socket_gt_y, socket_gt_z,
coordinate_frame, euclidean_error_cm, within_3cm,
execution_attempted, execution_success, missing_data_reason
```

Requirements:

- Include every eligible target regardless of planning or execution outcome.
- Use the frozen prediction that existed before execution.
- Use the physically measured attachment-socket coordinate as ground truth.
- Apply only archived calibration transforms and record the final common frame.
- Preserve rows with missing prediction or socket data and explain the absence.
- Define error and `within_3cm` only when both coordinates are available.

Create `robot_target_accuracy_summary.json`, separately for harvesting,
deleafing, and overall, with:

- Eligible and evaluated target counts.
- Median target error in centimeters.
- Count and percentage within 3 cm.
- Missing prediction, missing socket, and missing-any counts.

Create `robot_target_coordinate_note.md` describing:

- How socket coordinates were measured.
- The source record for each measurement.
- Original prediction and socket frames and units.
- Every applied transform, its direction, calibration file, and timestamp.
- Whether the calibration existed before the evaluated execution.
- The exact Euclidean-error calculation.

Acceptance condition: all targets from all 20 direct trials are accounted for,
including failed and missing cases.

## 2. Complete robot-trial ledger

Create `robot_trial_ledger.csv`, with one row for every direct-execution and
policy-evaluation trial.

Required columns:

```text
trial_id, date, scene_id, task, evaluation_type, target_id,
plan_generated, automatic_gate_rejected,
operator_rejected_before_execution, execution_started,
operator_or_safety_stop, success, failure_category, failure_description,
included_in_denominator, raw_log_path, video_path
```

Use only `direct` or `policy` for `evaluation_type`. Use explicit Boolean values
when established by a record; use `not_recorded`, not `false`, when unknown.

Create `robot_trial_summary.json` answering:

- How many unique scenes were used for direct execution and policy evaluation.
- Whether rejected plans count among the reported ten trials per task.
- Counts of automatic rejections, pre-execution operator rejections, and stops
  after execution began.
- The causes of the one harvesting and three deleafing policy failures.
- Missing raw-log and video counts.
- Whether the ledger exactly reproduces direct 10/10 and 10/10, and policy 9/10
  harvesting and 7/10 deleafing.

Link every row to its raw log and video when available. Missing links must remain
explicit and must not cause the trial to be removed.

## 3. Detector-adaptation and policy-supervision manifest

Create `robot_supervision_manifest.json` containing:

- Manually verified detector targets by task, with target, image, scene, and raw
  annotation paths.
- Unique target, image, and scene counts by task.
- Adaptation-scene IDs, evaluation-scene IDs, and their explicit intersection.
- Policy demonstrations by task, with demonstration IDs and raw-data paths.
- Detector and policy configuration/model identifiers.
- A reason for every missing record.

Repository metadata currently reports two harvesting demonstrations (`AT1T1`,
`AT1T2`) and five deleafing demonstrations (`AD1`, `BD1`, `BD2`, `BD3`, `BD4`).
Confirm these against the final local experiment record rather than merely
copying the counts.

## 4. Paired tolerance-curve confidence intervals

Create `tolerance_paired_ci.csv` with columns:

```text
tolerance_cm, egodelta_f1, gsdiff_f1, delta_f1_points,
paired_ci_low, paired_ci_high, bootstrap_samples, seed
```

Protocol:

- Compare native EgoDelta and score-preserving GS-DIFF at 1–5 cm.
- Jointly sample the same 30 unique scene UUIDs for both methods.
- Keep both interventions and all predictions/targets from a sampled scene
  together.
- Use 10,000 paired bootstrap samples and record the seed.
- Recompute pooled TP, FP, FN, and F1 within every sample.
- Define `delta_f1_points = 100 * (F1_EgoDelta - F1_GS-DIFF)`.
- Preserve native EgoDelta's 161 predictions and the diagnostic's 160
  predictions; never force proposal counts to match.
- Keep F1@3 cm as the headline metric; 1–5 cm is sensitivity analysis.

If the paper makes a statistically supported full-curve claim, additionally
compute a simultaneous 95% confidence band using the bootstrap maximum absolute
deviation across the five tolerances.

## 5. Runtime verification — local logs required

Create `runtime_verified.csv`, with one row for each of the 60 interventions.

Required columns:

```text
scene_id, pre_reconstruction_s, post_reconstruction_s, pre_post_parallel,
stage2_s, stage3_s, observed_total_s, gpu, cpu, ram,
software_commit, source_log, missing_data_reason
```

Recover and verify where possible:

- Exact reconstruction GPU, CPU, RAM, software revision, and command.
- Whether pre/post reconstructions ran sequentially or concurrently.
- Whether each reconstruction log produced the final PLY used by evaluation.
- Stage-2 hardware, revision, cache ID, and runtime.
- Stage-3 hardware, revision, input-cache ID, and runtime.
- Actual observed end-to-end wall-clock time.

Create `runtime_verified_summary.json` with count, median, IQR, mean, and
standard deviation for every defensible stage. Create `runtime_provenance.md`
explaining how logs were tied to outputs and which fields remain missing.

Do not report the existing 931.28-second serial-equivalent sum as measured wall
time. If full provenance cannot be recovered, report only:

- Stage 2: median 364.35 s on NVIDIA L40S.
- Stage 3: median 0.75 s, CPU-only.

## 6. Final result-provenance freeze

Create `final_result_manifest.json`. For every manuscript result, record:

```text
manuscript_item, method_or_variant, source_commit, worktree_clean,
configuration_path, configuration_sha256,
cache_or_prediction_path, cache_or_prediction_sha256,
dataset_or_manifest_path, dataset_or_manifest_sha256,
evaluation_script_path, evaluation_script_sha256,
output_directory, protocol_notes
```

Keep these protocol distinctions explicit:

- Tables 1–2 use Stage-2 outputs and the 720-reference-mask protocol.
- Table 3 uses native, color-free Stage 3 and produces 161 predictions.
- Table 4 uses 160 frozen proposals.
- Score-preserving GS-DIFF remains a diagnostic, not an official result.
- Only Stage 3 is color-free; Stage 2 uses appearance evidence.

Add a new 60-file native Stage-2 integrity manifest. Restore the missing 60th
per-file hash entry by reading the existing cache without modifying any frozen
NPZ. Preserve the original 59-record manifest and document why the supplemental
manifest was created.

## 7. Normal-agreement diagnostic package

No new experiment is required. Package the verified result:

| Condition | F1@3 cm | Median error |
|---|---:|---:|
| Without normal agreement | 92.50% | 1.197 cm |
| Full selector | 92.50% | 1.165 cm |

Create `normal_agreement_diagnostic.csv` and
`normal_agreement_diagnostic.json` containing:

- Both aggregate rows.
- All 160 frozen proposal IDs and both selected coordinates.
- Correctness at 3 cm under both conditions.
- Selected-coordinate change count and correctness-transition counts.
- Untruncated deleafing error change.
- Commit, configuration, cache, evaluator, and bootstrap identifiers.
- Bootstrap seed, sampling unit, samples, point difference, and percentile CI.

Required conclusion: normal agreement slightly refines localization but is not
an F1@3 cm contributor on this frozen proposal set.

## Final validation checklist

- [ ] All direct-execution targets appear in the accuracy export.
- [ ] Every direct and policy trial has exactly one ledger row.
- [ ] Failed, rejected, stopped, and missing trials remain included.
- [ ] Ledger totals reproduce the four reported success fractions or explicitly
      identify the discrepancy.
- [ ] Prediction and socket coordinates use the same documented frame and units.
- [ ] Detector-adaptation and policy records trace to raw files.
- [ ] Adaptation/evaluation scene overlap is explicitly reported.
- [ ] Runtime covers all 60 interventions without presenting a sum as observed
      wall time.
- [ ] Bootstrap sampling units are scene UUIDs, not targets or views.
- [ ] Native, fixed-proposal, and diagnostic protocols remain separate.
- [ ] Every paper value maps to immutable paths and hashes.
- [ ] Missing values are marked, never estimated.
- [ ] A checksum file covers the final timestamped deliverable directory.
