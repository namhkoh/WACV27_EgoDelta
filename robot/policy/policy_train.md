# Policy Training Notes

## Goal

Build a policy-ready action library for the gutter testbed that can train and evaluate a robot policy over symbolic deleafing and harvesting targets. The current strategy is to collect one visually confirmed atomic action per physical instance, then compose training/evaluation episodes from those labeled actions.

The target policy should consume robot state, a goal target, and scene context, then produce actions for the two-stage task order:

1. Deleaf one or more selected petioles.
2. Harvest one selected tomato and complete the basket drop.

## Current Testbed Labels

Current active saved actions:

- Deleafing: `AD1`, `BD1`, `BD2`, `BD3`, `BD4`
- Harvesting: `AT1T1`, `AT1T2`

Removed from the active setup:

- `AT1T3`

Each confirmed action is stored at:

```text
robot/policy/policy_action_library/<LABEL>/success_current/
```

## What We Built

Policy collection code now lives under `robot/policy/`.

- `collect_policy_episode.py`: CLI entry point.
- `episode_manager.py`: symbolic labels, action instances, live goal capture, fixed-plan replay, trajectory logging, and registration.
- `capture_robot_tcp_trajectory.py`: records TCP trajectory during live actions.
- `replay_arm_joint_trajectory.py`: replays existing trajectories when needed.
- `organize_policy_library.py`: normalizes the curated library and writes training indexes.

The curated library now has:

- 7 confirmed `success_current` actions.
- Per-action `episode.json` files.
- Pre-scene captures where available.
- Goal/target artifacts.
- Planner/action artifacts.
- Robot TCP trajectory artifacts.
- Execution artifacts for harvest actions.
- Candidate archives for both harvest and deleafing.

## Candidate Archives

Harvest candidates are stored as full solver-passing alternatives:

```text
robot/policy/policy_action_library/AT1T1/success_current/actions/AT1T1_instance_001_step_01_AT1T1/candidate_solves/
robot/policy/policy_action_library/AT1T2/success_current/actions/AT1T2_instance_001_step_01_AT1T2/candidate_solves/
```

Each harvest folder contains five candidate JSONs plus:

```text
top_solved_candidates.json
```

Deleafing candidates are planner summaries extracted from each confirmed action artifact:

```text
robot/policy/policy_action_library/<DELEAF_LABEL>/success_current/actions/<STEP_ID>/candidate_summaries/
```

Each deleaf folder contains:

- `deleaf_candidate_summaries.json`
- one JSON file per candidate summary

Important distinction: harvest alternatives are full candidate plans; deleaf alternatives are candidate summaries, not independently verified hardware executions.

## Training Indexes

The library organizer writes these files:

```text
robot/policy/policy_action_library/library_manifest.json
robot/policy/policy_action_library/training_index.json
robot/policy/policy_action_library/episodes_index.jsonl
robot/policy/policy_action_library/actions_index.jsonl
robot/policy/policy_action_library/artifact_integrity.json
robot/policy/policy_action_library/training_splits.json
```

Latest integrity result:

- `action_count`: 7
- `episode_count`: 7
- `artifact_count`: 63
- `missing_count`: 0

Regenerate indexes after adding or replacing any action:

```powershell
python robot\policy\organize_policy_library.py
```

## Conversation Summary

We initially planned 30 full end-to-end episodes from symbolic gutter configurations. During live collection, full-episode collection became inefficient because each physical cut or grasp required careful validation. We pivoted to an atomic action library: collect each leaf and tomato instance once, visually confirm it, and later compose policy episodes from those labeled primitives.

For deleafing, we validated side-aware cuts for `AD1`, `BD1`, `BD2`, `BD3`, and `BD4`. The main debugging issue was approach side and cut depth. BD4 was short because target extension was capped by the candidate list and entry-only contact advance did not move the actual cut target. The final successful BD4 run used target-mode contact advance and `target_extension_m = 0.085`.

The deleaf trainer can now include explicit stem-facing side features with `--include-deleaf-side-features`. Current label mapping:

- `AD1`, `BD2`, `BD3`, `BD4`: right-facing
- `BD1`: left-facing

These features are necessary because a single clicked 3D point does not encode which side of the stem/rod should be approached.

For harvesting, both active tomatoes moved during setup, so we re-solved `AT1T1` and `AT1T2`, confirmed full-tail hardware execution through `basket_place`, registered the successful solves, and replayed them through the policy wrapper to capture policy-ready trajectories and goals.

We then normalized the folder structure for training: `success_current` is the canonical source per label, harvest top-5 candidates are copied beside the confirmed action, deleaf candidate summaries are exported beside the confirmed action, and root-level indexes describe all trainable artifacts.

## Next Steps

## Two-Policy Training Setup

We now train two separate action-chunk policies:

- `deleaf`: right-arm policy trained from `AD1`, `BD1`, `BD2`, `BD3`, and `BD4`.
- `harvest`: left-arm policy trained from `AT1T1` and `AT1T2`.

The policy input is the GT 3D target point used during data collection plus the current robot state. The target is stored in the action library as `goal.point_robot_m` and copied into each training row as `goal_robot_m`; we are not feeding the full point cloud into the current policy.

```text
[goal_robot_m(3), goal_robot_m - tcp_position_robot_m(3), q_arm_rad(7), dq_arm_rad_s(7), tcp_position_robot_m(3)]
```

For GCSL-style training, pass `--gcsl`. This keeps the same GT clicked goal point and adds horizon-to-goal features:

```text
[gcsl_horizon_s, gcsl_horizon_fraction, gcsl_elapsed_fraction]
```

The policy output is an action chunk:

```text
10 future 7-DoF arm joint deltas = 70 values
```

At runtime each 7-DoF row is added to the current arm configuration to produce the next arm joint command. The trained artifact now stores a supervised MLP as the primary policy: plain BC for short chunks, or GCSL when `--gcsl` is enabled. Nearest-neighbor and ridge outputs are retained only as diagnostics/fallbacks; they are not the policy.

Arm velocity is recomputed from the captured active-arm joint sequence instead of slicing the full robot velocity vector. This avoids relying on any full-state joint ordering convention.

### Frequency and Chunk Parameters

The raw trajectory logger records at 10 Hz. Effective rates in the current library are consistently about 10 Hz, but the robot motions are slow and highly redundant. For overnight training and reliable replay we downsample to:

```text
sample_hz = 5.0
chunk_size = 10
chunk_horizon_s = 2.0
```

This gives enough temporal context to cover approach/contact transitions while avoiding overfitting to tiny frame-to-frame noise. The policy artifacts are written to:

```text
robot/policy/trained_policies/deleaf/
robot/policy/trained_policies/harvest/
```

Train both policies:

```powershell
python robot\policy\train_action_chunk_policy.py --primitive deleaf
python robot\policy\train_action_chunk_policy.py --primitive harvest
```

GCSL full-action training command for the current 10 Hz deleaf library:

```powershell
python robot\policy\train_action_chunk_policy.py --primitive deleaf --output-dir robot\policy\trained_policies\deleaf_gcsl_gpu10hz_full360_seed11 --sample-hz 10 --chunk-size 360 --gcsl --pad-terminal-chunks --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,512 --bc-epochs 8000 --bc-batch-size 1024 --bc-learning-rate 0.0007 --bc-weight-decay 0.0 --seed 11 --print-progress
```

### Robust Goal Conditioning

The first GCSL harvest rollout exposed a normalization failure: `AT1T1` and `AT1T2` were only a few millimeters apart, so the learned input std for `goal_robot_m.x/z` became tiny. A normal 1 mm RealSense click difference then became a very large normalized input shift and the policy jumped to the wrong phase of the action. The trainer now supports:

- `--goal-jitter-count`, `--goal-jitter-std-m`, `--goal-jitter-max-m`: augment each chunk with small GT target perturbations.
- `--goal-normalizer-min-std-m`: apply a minimum std floor to `goal_robot_m.*` and `goal_minus_tcp_robot_m.*`.
- `--include-harvest-candidate-solves`: opt-in conversion of archived harvest solver candidates into synthetic waypoint demonstrations.

Recommended robust GCSL training commands:

```powershell
python robot\policy\train_action_chunk_policy.py --primitive harvest --output-dir robot\policy\trained_policies\harvest_gcsl_robust10hz_full360_seed11 --sample-hz 10 --chunk-size 360 --gcsl --pad-terminal-chunks --goal-jitter-count 32 --goal-jitter-std-m 0.006 --goal-jitter-max-m 0.018 --goal-normalizer-min-std-m 0.02 --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,512 --bc-epochs 8000 --bc-batch-size 1024 --bc-learning-rate 0.0007 --bc-weight-decay 0.0 --seed 11 --print-progress

python robot\policy\train_action_chunk_policy.py --primitive deleaf --output-dir robot\policy\trained_policies\deleaf_gcsl_robust10hz_full360_seed11 --sample-hz 10 --chunk-size 360 --gcsl --pad-terminal-chunks --goal-jitter-count 8 --goal-jitter-std-m 0.004 --goal-jitter-max-m 0.012 --goal-normalizer-min-std-m 0.015 --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,512 --bc-epochs 8000 --bc-batch-size 1024 --bc-learning-rate 0.0007 --bc-weight-decay 0.0 --seed 11 --print-progress
```

Side-conditioned deleaf training command:

```powershell
python robot\policy\train_action_chunk_policy.py --primitive deleaf --output-dir robot\policy\trained_policies\deleaf_gcsl_side10hz_full360_seed11 --overwrite --sample-hz 10 --chunk-size 360 --gcsl --pad-terminal-chunks --include-deleaf-side-features --goal-jitter-count 8 --goal-jitter-std-m 0.004 --goal-jitter-max-m 0.012 --goal-normalizer-min-std-m 0.015 --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,512 --bc-epochs 8000 --bc-batch-size 1024 --bc-learning-rate 0.0007 --bc-weight-decay 0.0 --seed 11 --print-progress --skip-loo-eval
```

At rollout time, pass either `--deleaf-label BD3` or `--deleaf-facing right/left` when using a side-conditioned deleaf policy.

Candidate-solve harvest training is available, but keep it as a separate experiment because these plans passed solver gates but were not all visually confirmed on hardware:

```powershell
python robot\policy\train_action_chunk_policy.py --primitive harvest --output-dir robot\policy\trained_policies\harvest_gcsl_candidates10hz_full360_seed11 --sample-hz 10 --chunk-size 360 --gcsl --pad-terminal-chunks --include-harvest-candidate-solves --harvest-candidate-top-k 5 --goal-jitter-count 16 --goal-jitter-std-m 0.006 --goal-jitter-max-m 0.018 --goal-normalizer-min-std-m 0.02 --bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,512 --bc-epochs 8000 --bc-batch-size 1024 --bc-learning-rate 0.0007 --bc-weight-decay 0.0 --seed 11 --print-progress
```

For fixed tomato replay during evaluation, `guarded_bc_rollout.py` can snap a clicked goal to the nearest stored demo goal before prediction:

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\harvest_gcsl_robust10hz_full360_seed11 --target-source realsense-open3d --snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02 --require-home --full-action --execute --steps-to-execute 5 --minimum-time-s 0.10 --inter-step-sleep-s 0.0 --max-step-delta-rad 0.017 --clip-step-delta --clip-joint-limits --max-z-m 1.40 --full-action-duration-scale 1.15 --extra-recede-iterations 20
```

The BC model is intentionally small and dependency-free:

```text
model = MLP(features -> hidden layers -> flattened action chunk)
optimizer = Adam
loss = MSE on normalized action chunks
```

Latest trained artifacts:

- Deleaf: 5 demos, 849 chunks after idle-prefix trimming, 5 Hz, 10-step chunks, BC train first-step RMSE about 0.12 deg.
- Harvest: 2 demos, 771 chunks, 5 Hz, 10-step chunks, BC train first-step RMSE about 0.03 deg.

Training reconstruction with BC is sub-degree at the first step for both policies. Leave-one-label-out is a stricter stress test and currently shows BC first-step errors on the order of 0.2-0.5 deg for deleafing and about 0.1 deg for harvesting. This is enough for an initial BC policy artifact, but not enough evidence for broad unseen-target generalization; collect more confirmed instances before claiming that.

The trainer trims idle lead-in by default before chunking. This matters because the robot trajectory logger starts before point selection and planning; without trimming, deleafing demonstrations contain 15-27 s of near-home waiting and the BC model learns to output near-zero first chunks. The default trim keeps 1 s of preroll before the first `0.01 rad` displacement from the initial arm state.

Generate one predicted action chunk from a trained artifact:

```powershell
python robot\policy\predict_action_chunk_policy.py --policy-dir robot\policy\trained_policies\deleaf --goal-robot-m X,Y,Z --q-arm-rad Q0,Q1,Q2,Q3,Q4,Q5,Q6 --tcp-position-robot-m X,Y,Z
```

The default prediction mode is `--mode bc`. Use `--mode knn` or `--mode ridge` only for diagnostics and fallback comparison.

## Guarded BC Rollout

`robot/policy/guarded_bc_rollout.py` is the hardware-facing wrapper for the BC policy. It reads the live arm state, predicts one BC chunk, converts the cumulative predicted rows into consecutive per-step joint deltas, and blocks execution when any guard fails.

The default rollout is receding horizon:

```text
predict 10 steps at 5 Hz
guard the first 3 steps by default
execute only the first 3 steps when --execute is passed
re-read state and re-predict for the next prefix
```

Blocking guards currently include per-step joint delta, joint limits, TCP workspace, robot self-clearance along interpolated joint paths, and target-distance progress. The per-step delta cap defaults to `0.02 rad`, which is about `1.15 deg` per 0.2 s step. Full-chunk checking is available by passing `--guard-horizon-steps 10`, but the default matches the receding-horizon execution prefix.

Offline smoke check without robot FK:

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf --goal-robot-m X,Y,Z --q-arm-rad Q0,Q1,Q2,Q3,Q4,Q5,Q6 --tcp-position-robot-m X,Y,Z --skip-fk-guards
```

Live no-motion preflight with FK, workspace, joint-limit, and self-clearance guards:

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf --goal-robot-m X,Y,Z --require-home
```

To choose the target from the live RealSense RGB-D view instead of typing a point:

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf --target-source realsense-open3d --require-home
```

The Open3D picker uses the same interaction as the collection scripts:

```text
Shift + left click: pick one 3D point
Q or Esc: close after selecting
```

For a simpler RGB image click backed by local depth:

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf --target-source realsense-rgb --require-home
```

Only after `guards.ok=true`, execute the first receding-horizon prefix:

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf --goal-robot-m X,Y,Z --require-home --execute
```

For task success-rate evaluation, use full-action mode. It picks the target once, then repeatedly executes guarded prefixes while re-reading live state and re-predicting between prefixes. For GCSL policies, rollout now selects the nearest training demo by GT goal point and uses that demo's duration for the horizon feature and automatic full-action budget. This matters for harvesting because the harvest demonstrations are longer than the 36 s chunk horizon.

```powershell
python robot\policy\guarded_bc_rollout.py --policy-dir robot\policy\trained_policies\deleaf --target-source realsense-open3d --require-home --full-action --execute
```

The current defaults use 5-step receding prefixes, `0.10 s` minimum time, no artificial inter-step sleep, and an auto-sized full-action budget with a small duration margin. The first three receding prefixes keep the target-progress guard enabled with a 5 cm tolerance; this allows short posture/entry alignment motions before direct target approach. Later prefixes can move through cut/retract or basket behavior without being rejected for moving away from the target. Any guard or robot command failure stops the run and still writes the partial log.

Every rollout writes a JSON log under:

```text
robot/policy/rollout_logs/<primitive>/
```

These logs support four evaluation numbers for the paper: BC-only success, guarded BC success, guarded BC plus fallback success, and guard rejection rate.

## Immediate Next Steps

- Keep the action library as the source of truth: only visually confirmed actions should enter `success_current`.
- Add at least 2-3 more confirmed instances per task class before using the learned policy as anything more than a retrieval-style action chunker.
- For the deadline evaluation, deploy the BC policy only with hard safety gates, action clipping, home-restore wrappers, and no-motion checks. Keep KNN/fixed-action replay available as an emergency fallback.
- Treat candidate archives as auxiliary supervision or fallback analysis, not as confirmed demonstrations.
- Replace manually clicked targets with automatic EgoDelta/RealSense target proposals after the action-chunk baseline is stable.
