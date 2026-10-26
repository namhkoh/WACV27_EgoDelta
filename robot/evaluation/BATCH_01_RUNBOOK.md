# Batch 01 Runbook: 30 Harvest Rollouts

Batch 01 contains harvest scene slots 1-15. Each slot has one direct
EGODELTA-target execution and one learned-policy execution. Follow the order in
the exported batch CSV; do not reorder after seeing a target or preflight.

No trial may start until the gripper `/status` endpoint is reachable and both
arms pass a read-only TCP/home preflight.

## Per scene slot

1. Arrange and identify a fresh harvest scene. Record the scene ID and start
   video before target processing.
2. Run the frozen EGODELTA target pipeline before any robot action.
3. Copy the planner-facing target JSON with `freeze_trial_target.py`. Record the
   robot-frame point key and preserve the generated provenance sidecar.
4. Bind both registered rows for this scene slot to the same frozen target.
   The ledger rejects different target hashes, scene IDs, or target IDs.
5. Generate both command manifests with `prepare_trial_commands.py`.
6. Follow `within_scene_order` from `batch_01.csv`.
7. Reset the fixture to the same registered configuration before the paired
   condition. Record any reset failure rather than substituting a new scene.
8. For each condition, restore/check home, run the no-motion preflight, retain
   all gate outputs, then review the disabled execute command with the operator.
9. After execution or rejection, annotate the registered row exactly once.
   Never replace a failed row with a retry.
10. A troubleshooting rerun receives a separate diagnostic ID and is excluded
    from the registered 120-rollout result.

## Condition constraints

Direct execution uses the frozen target through the staged planner and all hard
IK, collision, target-geometry, and gripper gates.

Policy execution uses the same frozen `goal_robot_m`. The command manifest must
contain `--no-snap-goal-to-nearest-demo`; a live RealSense click or snapped demo
goal invalidates the registered condition.

The policy uses learned local motion plus the documented deterministic gripper
schedule and basket tail. Report it as a hybrid policy condition, not a purely
learned end-to-end controller.

## Harvest success

A success requires all of the following:

- The intended target fruit is enclosed and grasped.
- The fruit is detached without collision or safety/operator stop.
- The fruit remains held through retract.
- The fruit is released inside the designated basket.
- The planned action completes without human correction.

Approach-only, contact without detachment, dropped fruit, wrong fruit, release
outside the basket, operator correction, and safety stops are failures. An
automatic pre-execution gate rejection is also a completed failed trial and
remains in the denominator.

## Evidence to retain

- Frozen target JSON and provenance sidecar.
- Planner/preflight artifact, including rejected candidates.
- Execution artifact or explicit reason execution never started.
- Full raw console log.
- Trial video.
- Home and gripper status evidence.
- Explicit failure category and description.

Do not add batch trajectories or outcomes to the policy action library until
all 120 registered trials and the final result manifest are frozen.
