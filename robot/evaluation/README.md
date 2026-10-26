# Prospective Robot Evaluation

This package pre-registers and preserves the WACV physical-robot evaluation.
It does not execute robot commands.

## Registered design

- 30 direct EGODELTA-target harvest rollouts.
- 30 direct EGODELTA-target deleaf rollouts.
- 30 learned-policy harvest rollouts.
- 30 learned-policy deleaf rollouts.
- 120 total rollouts, all retained in the denominator.

The design is paired. Each task has 30 scene slots, and every scene slot is run
once with direct planning and once with the learned policy. Method order is
randomized within each pair. The fixture must be reset between paired runs, and
both runs must use the same frozen pre-execution target hash.

The four 30-rollout batches are:

1. Harvest scene slots 1-15, direct and policy.
2. Harvest scene slots 16-30, direct and policy.
3. Deleaf scene slots 1-15, direct and policy.
4. Deleaf scene slots 16-30, direct and policy.

These are evaluation trials, not new training demonstrations. Do not add their
trajectories or outcomes to the policy library until all 120 outcomes and the
final result manifest have been frozen.

## Initialize

```powershell
C:\path\to\user\miniconda3\envs\egodelta_robot\python.exe `
  C:\path\to\projects\EgoDelta\robot\evaluation\collect_robot_trials.py init `
  --output-dir C:\path\to\projects\EgoDelta\paper_deliverables_20260819_robot_trials\robot_trials `
  --trials-per-task 30 `
  --scenes-per-batch 15 `
  --seed 1493 `
  --harvest-policy-dir C:\path\to\projects\EgoDelta\robot\policy\trained_policies\harvest_bc_success_trimmed_nojitter10hz_chunk30_seed11_rerun_20260819 `
  --deleaf-policy-dir C:\path\to\projects\EgoDelta\robot\policy\trained_policies\deleaf_bc_success_only_jitter10hz_chunk50_seed11_rerun_20260819
```

## Freeze and bind a target

Freeze the planner-facing target JSON byte-for-byte and write a provenance
sidecar. `--point-key` is the dotted key containing the robot-frame target.

```powershell
python robot\evaluation\freeze_trial_target.py `
  --source-json PATH_TO_PREEXECUTION_TARGET.json `
  --output-json SESSION\targets\SCENE_TARGET.json `
  --task harvest `
  --scene-id SCENE_ID `
  --target-id TARGET_ID `
  --point-key target_robot_m
```

Bind that immutable target to the registered row:

```powershell
python robot\evaluation\collect_robot_trials.py bind `
  --session-dir SESSION `
  --trial-id wacv27_harvest_01_direct `
  --scene-id SCENE_ID `
  --target-id TARGET_ID `
  --target-artifact SESSION\targets\SCENE_TARGET.json
```

Bind the paired policy row to the same target. The command refuses a different
scene, target ID, or SHA-256.

## Prepare commands

```powershell
python robot\evaluation\prepare_trial_commands.py `
  --session-dir SESSION `
  --trial-id wacv27_harvest_01_direct
```

The generated JSON contains separate preflight and execution argument arrays.
It never launches subprocesses, and every execution stage is marked
`enabled=false` until operator review.

## Record an outcome

Use `annotate` exactly once per bound trial. Failed, rejected, stopped, and
missing trials are completed rows and remain in the denominator. A rerun is a
diagnostic attempt, not a replacement for the registered trial.

Use `next --batch-id batch_01` to display the next incomplete trial in the first
30-rollout batch.

## Automated harvest collection

Use `prepare_registered_scene.py` once after arranging each paired harvest
scene. It performs no robot motion. It captures RealSense, validates exactly one
named EgoDelta task frame, freezes the passed identity target, binds the direct
and policy rows to the same SHA-256, and creates both locked command manifests.

```powershell
C:\path\to\user\miniconda3\envs\egodelta_robot\python.exe `
  C:\path\to\projects\EgoDelta\robot\evaluation\prepare_registered_scene.py `
  --session-dir C:\path\to\projects\EgoDelta\paper_deliverables_20260819_robot_trials_additional20\robot_trials `
  --scene-slot 2 `
  --scene-id harvest_scene02 `
  --target-id tomato0 `
  --target-name tomato0 `
  --task-frame-json C:\path\to\projects\EgoDelta\robot\calibration\targets\a270c038\harvest_tomato0_task_frame_instance_split_jaw_minusx.json
```

Use `run_registered_trial.py` for each bound row. It verifies committed
controller sources, target and manifest hashes, the physically closed left
gripper, home state, and planner/policy gates. With `--execute`, the exact
trial ID must be repeated as confirmation. It preserves logs and artifacts,
asks for the observed physical outcome, and annotates the ledger exactly once.

```powershell
C:\path\to\user\miniconda3\envs\egodelta_robot\python.exe `
  C:\path\to\projects\EgoDelta\robot\evaluation\run_registered_trial.py `
  --session-dir C:\path\to\projects\EgoDelta\paper_deliverables_20260819_robot_trials_additional20\robot_trials `
  --trial-id wacv27_additional20_harvest_01_direct `
  --execute `
  --confirm-trial-id wacv27_additional20_harvest_01_direct
```

For a two-stage review, omit `--execute` first. Resume the immutable passing
preflight with `--reuse-preflight --execute --confirm-trial-id TRIAL_ID`.
The runner verifies all stored hashes and refuses to replan or silently retry.

The current automated executor covers harvest direct and policy rows. Deleaf
scene preparation and direct execution must be added and validated before the
deleaf registered batches begin.
