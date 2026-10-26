# Controlled Robot Execution Evidence Audit

Generated: 2026-08-21T07:54:10.519048+00:00

## Evidence-backed findings

- The located final ledger contains 20 completed harvesting direct rows and 20 completed deleafing direct rows.
- The completed runs are scripted target-conditioned executions, not learned behavior-cloning rollouts. Their ledger records use `policy_dir=not_applicable`.
- The frozen target artifacts store manual RealSense/Open3D selections. They do not store the recovered EGODELTA prediction needed for a direct-target accuracy comparison.
- No located trial artifact stores physical attachment-socket ground truth. Therefore target-error N, median error, and the count within 3 cm are **unavailable**.
- The authoritative policy-action index confirms 2 harvesting demonstrations and 5 deleafing demonstrations.
- The raw manual front-view detector annotation file documented by the training code is absent. Annotation counts by task and overall are **unavailable**.
- No completed learned-policy rows are present in the located evaluation ledger. Policy scene counts, rejection/stop counts, and the five claimed failure identities are **unavailable**.

## Trial-accounting warning

The prospective amendment requires 30 rows per task and execution mode, with every rejection, stop, and failure retained. The current completed cohort binds successful retry artifacts to 40 direct rows. The attempt audit finds 32 run directories not included as denominator rows. Consequently, the available ledger does not reproduce a prospective 120-rollout evaluation.

## Section 4.6 fill status

| Requested value | Evidence status |
|---|---|
| Direct targets with socket GT | unavailable |
| Median direct-target error | unavailable |
| Direct targets within 3 cm | unavailable |
| Manually verified detector targets | unavailable |
| Unique policy-evaluation scenes | unavailable |
| Policy automatic rejections / operator rejections / stops | unavailable |
| Two harvesting and three deleafing policy failure identities | unavailable / unresolved |
| Training demonstrations | supported: 2 harvesting, 5 deleafing |

Do not replace the manuscript TBDs with aggregate claims until the missing trial-level records are recovered. The `direct_target_accuracy.csv` placeholders and `policy_failure_audit.csv` rows intentionally use `unavailable` rather than inferred values.

## Files

- `direct_target_accuracy.csv`: 60 claimed direct slots, including 40 artifact-backed rows and 20 explicit missing historical rows.
- `direct_target_accuracy_summary.json`: strict computed availability and metric summary.
- `trial_accounting_observed.csv`: the located final additional-20 ledger.
- `attempt_audit.csv`: every located direct-run directory, including retries excluded from the ledger.
- `trial_accounting_summary.json`: expected-versus-observed counts and denominator audit.
- `detector_supervision.csv` / `detector_supervision_summary.json`: annotation evidence and missing-source status.
- `policy_demonstrations.csv`: the seven action-library demonstrations.
- `policy_evaluation_scenes.json`: policy scene-count availability.
- `policy_failure_audit.csv`: five explicitly unresolved claimed policy failures.
- `source_log_paths.json`: source paths, availability, and hashes.
- `revision.json`: code/data revision used for this export.
- `checksums.sha256`: package checksums.
