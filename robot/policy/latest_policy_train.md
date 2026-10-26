# Latest Policy Training Pipeline

Last updated: 2026-05-29

## Objective

Train two separate goal-conditioned action-chunk policies from the curated action library:

- `deleaf`: right-arm cutting policy for clicked leaf/stem targets.
- `harvest`: left-arm tomato grasp/retract policy.

The immediate hardware goal is reliable execution, not a generic high-capacity policy. The current best strategy is hybrid:

1. Use non-GCSL BC only for the local action segment that benefits from demonstrations.
2. Use deterministic replay/planner tails for known safety-critical fixed behavior.
3. Snap clicked targets to known successful demo goals when evaluating on the current testbed.

## Active Data

Canonical data source:

```text
robot/policy/policy_action_library/
```

Active confirmed labels:

- Deleaf: `AD1`, `BD1`, `BD2`, `BD3`, `BD4`
- Harvest: `AT1T1`, `AT1T2`

Current index:

```text
robot/policy/policy_action_library/training_index.json
```

Regenerate the policy library indexes after adding/replacing actions:

```powershell
python robot\policy\organize_policy_library.py
```

## Policy Input

The policy consumes the GT 3D point clicked/collected during data collection, not the full point cloud:

```text
goal_robot_m
goal_robot_m - tcp_position_robot_m
q_arm_rad
dq_arm_rad_s
tcp_position_robot_m
```

Do not use `--gcsl` for the current hardware baseline. Older GCSL policies appended horizon features:

```text
gcsl_horizon_s
gcsl_horizon_fraction
gcsl_elapsed_fraction
```

For side-conditioned deleafing, the trainer can also append:

```text
deleaf_left_facing
deleaf_right_facing
deleaf_facing_sign
```

## Policy Output

The action is already relative:

```text
future_q_arm_rad - q_now
```

Each chunk row is a cumulative future offset from the current 7-DoF arm state. Runtime reconstructs:

```text
q_chunk[t] = q_now + delta_q_chunk[t]
```

The guards compute per-step joint increments from consecutive reconstructed `q_chunk` rows. This is not absolute joint prediction.

## Harvest: Current Fix

The previous harvest policies failed for three separate reasons:

- Full harvest trajectories included basket behavior, so the learned policy sometimes moved toward the basket.
- GCSL horizon features could make the model stop when the learned countdown expired.
- Goal jitter was larger than the separation between the two harvest targets, blurring AT1T1 and AT1T2 identity.

The current pipeline removes those failure modes:

1. Train harvest on grasp/retract only.
2. Close gripper deterministically near grasp.
3. Run a fixed known-good basket tail after learned grasp/retract.
4. Use certified successful hardware trajectories only.
5. Do not use GCSL or goal jitter for the current two-tomato dataset.

Relevant training flags:

```text
--harvest-stop-after retract
--trim-idle-prefix
--trim-preroll-s 0.0
--goal-jitter-count 0
--store-training-arrays
```

New rollout flags:

```text
--harvest-gripper-schedule
--harvest-auto-fixed-basket-tail
--harvest-fixed-basket-tail-start-after retract
```

### Train Harvest Grasp/Retract Policy

```powershell
python robot\policy\train_action_chunk_policy.py --primitive harvest --output-dir robot\policy\trained_policies\harvest_bc_success_trimmed_nojitter10hz_chunk30_seed11 --overwrite --sample-hz 10 --chunk-size 30 --pad-terminal-chunks --harvest-stop-after retract --trim-idle-prefix --trim-start-delta-rad 0.005 --trim-preroll-s 0.0 --goal-jitter-count 0 --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,256 --bc-epochs 8000 --bc-batch-size 512 --bc-learning-rate 0.0008 --bc-weight-decay 0.0 --seed 11 --print-progress --skip-loo-eval --store-training-arrays
```

Exact training-data eval:

```powershell
python robot\policy\eval_policy_on_training_data.py --policy-dir robot\policy\trained_policies\harvest_bc_success_trimmed_nojitter10hz_chunk30_seed11 --max-rows 128 --report-steps 15
```

### Run Harvest With Fixed Basket Tail

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\harvest_bc_success_trimmed_nojitter10hz_chunk30_seed11 --target-source realsense-open3d --snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02 --require-home --full-action --execute --harvest-gripper-schedule --harvest-auto-fixed-basket-tail --harvest-fixed-basket-tail-start-after retract --execute-prefix-mode endpoint --steps-to-execute 5 --minimum-time-s 0.30 --inter-step-sleep-s 0.0 --max-step-delta-rad 0.025 --clip-step-delta --clip-joint-limits --joint-limit-margin-rad 0.003 --min-clearance-m 0.012 --path-samples-per-step 1 --max-z-m 1.45 --full-action-duration-scale 1.0 --extra-recede-iterations 0 --harvest-gripper-close-distance-m 0.09 --harvest-gripper-close-phase 0.45 --harvest-gripper-phase-close-max-distance-m 0.12
```

Notes:

- `--execute-prefix-mode endpoint` commands one interpolated endpoint per receding prefix, which is faster and smoother than commanding every predicted row.
- `--harvest-auto-fixed-basket-tail` selects the fixed tail from the nearest AT1T1/AT1T2 demo label.
- Basket release is deterministic and tied to the fixed tail, not learned.
- `guarded_bc_rollout.py --full-action` now uses nearest-demo duration for non-GCSL policies too; previously it incorrectly used only `chunk_horizon_s`.

## Deleaf: Current Fix

The latest side-conditioned/GCSL deleaf model regressed. The side-conditioned setup is brittle because only one confirmed demo is left-facing (`BD1`) and four are right-facing (`AD1`, `BD2`, `BD3`, `BD4`). A wrong side label can visibly alter the first motion, and GCSL horizon features can cause premature stopping.

Current recommendation:

1. Prefer a safer non-side-conditioned, non-GCSL 10 Hz BC model.
2. Use nearest-demo snapping.
3. Execute only if the dry run reports `policy_goal_snapping.snapped=true` and guards pass.

### Train Safer Deleaf Model

```powershell
python robot\policy\train_action_chunk_policy.py --primitive deleaf --output-dir robot\policy\trained_policies\deleaf_bc_success_only_jitter10hz_chunk50_seed11 --overwrite --sample-hz 10 --chunk-size 50 --pad-terminal-chunks --no-trim-idle-prefix --goal-jitter-count 16 --goal-jitter-std-m 0.004 --goal-jitter-max-m 0.012 --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,256 --bc-epochs 8000 --bc-batch-size 1024 --bc-learning-rate 0.0008 --bc-weight-decay 0.0 --seed 11 --print-progress --skip-loo-eval --store-training-arrays
```

### Dry-Run Deleaf

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf_bc_success_only_jitter10hz_chunk50_seed11 --target-source realsense-open3d --snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02 --require-home --full-action --full-action-duration-scale 1.0 --extra-recede-iterations 0 --steps-to-execute 3 --guard-horizon-steps 10 --minimum-time-s 0.10 --inter-step-sleep-s 0.0 --max-step-delta-rad 0.017 --clip-step-delta --clip-joint-limits --max-z-m 1.35
```

### Execute Deleaf

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf_bc_success_only_jitter10hz_chunk50_seed11 --target-source realsense-open3d --snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02 --require-home --full-action --full-action-duration-scale 1.0 --extra-recede-iterations 0 --steps-to-execute 3 --guard-horizon-steps 10 --minimum-time-s 0.10 --inter-step-sleep-s 0.0 --max-step-delta-rad 0.017 --clip-step-delta --clip-joint-limits --max-z-m 1.35 --execute
```

Avoid `deleaf_gcsl_side10hz_full360_seed11` unless running a controlled ablation with an exact known label and clean dry-run guards.

## Frequency

Do not move to 20 Hz yet.

Reason:

- The source demonstrations are effectively 10 Hz.
- 20 Hz mostly interpolates the same data.
- It doubles the output size and increases overfitting risk with only 5 deleaf demos and 2 harvest demos.
- Runtime command latency and `minimum_time_s` currently fit 10 Hz better.

Use 10 Hz first. Improve the dataset/control split before increasing frequency.

## History Concatenation

History concatenation is not currently active.

It is feasible because trajectory logs contain ordered `q_arm_rad` and `tcp.position_robot_m`, but it is not the first fix. With only a few demonstrations, adding history greatly increases feature dimension and can overfit unless we add more corrective/off-policy data.

Recommended order:

1. Fix harvest by removing basket from learned policy.
2. Restore deleaf with safer non-side model and nearest-demo snapping.
3. Only then consider `--history-steps` if closed-loop behavior is still unstable.

## Runtime Guards

Current rollout safety controls:

```text
step_delta
joint_limits
self_clearance
workspace
target_progress for early full-action iterations
```

For faster/lax hardware tests, the current aggressive-but-usable values are:

```text
--max-step-delta-rad 0.035
--min-clearance-m 0.012
--path-samples-per-step 1
--joint-limit-margin-rad 0.003
--execute-prefix-mode endpoint
```

Use these primarily for harvest, where the arm path is constrained by known tomato/basket geometry. For deleafing around rods, keep stricter values unless the dry-run is clean.

## Current Known Failure Modes

- Full-tail harvest policies can jump to basket because basket is in the training target distribution.
- Side-conditioned deleaf can regress if the facing side is ambiguous or mislabeled.
- 3D point alone does not encode stem approach side.
- Candidate harvest demos are solver-passing but not all visually confirmed hardware successes.
- The learned policy does not output gripper state. Gripper open/close/release is deterministic in `guarded_bc_rollout.py`.

## Current Best Experimental Protocol

Harvest:

1. Train `harvest_bc_success_trimmed_nojitter10hz_chunk30_seed11`.
2. Run with `--harvest-gripper-schedule`.
3. Use `--harvest-auto-fixed-basket-tail`.
4. Verify gripper close before fixed tail executes.

Deleaf:

1. Train `deleaf_bc_success_only_jitter10hz_chunk50_seed11`.
2. Dry-run with nearest-demo snapping.
3. Execute only if snapping and guards are clean.

This pipeline prioritizes immediate hardware success over generalization.
