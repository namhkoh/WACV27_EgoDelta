# EgoDelta Robot Policy — Latest Implementation & Design Notes

_Last updated: 2026-05-29 • Robot: RB-Y1 (RBY1-A, A class, firmware v1.1) • Target venue: CoRL 2026_

This document describes the **current** learned-policy stack for the two greenhouse manipulation
tasks — **tomato harvesting** (left arm + gripper) and **deleafing** (right arm + knife) — built on top
of the deterministic EgoDelta IK layer. It covers the architecture, the trained policies, the data
representation and relabeling, the inverse-kinematics decode, the runtime executor, evaluation, the
debugging findings that shaped the design, and the exact deploy commands.

---

## 1. Objective

Learn two task-conditioned, low-level closed-loop policies that drive the RB-Y1 arm to a 3D target
produced by EgoDelta change detection (`goal_robot_m`), reusing the deterministic RB-Y1 kinematics
layer we already built (`robot/kinematics/`). The policy must **not** be a black-box IK; instead it
predicts **end-effector (TCP) poses** and our own bounded least-squares IK converts them to joints.

This is the **EEF-space DP/ACT-style pipeline (phase B)**:

```
   EgoDelta target (goal_robot_m)
            │
            ▼
   ┌─────────────────────┐   features (23-d)     ┌──────────────────────────┐
   │   Robot state        │ ───────────────────▶ │   Policy NN (BC MLP)      │
   │  q, dq, TCP (FK)      │                       │  action chunking          │
   └─────────────────────┘                       └────────────┬─────────────┘
                                                               │ chunk of absolute
                                                               │ future TCP poses
                                                               ▼
                                            ┌───────────────────────────────────┐
                                            │  IK decode (OUR solver)            │
                                            │  solve_tcp_pose_ik per step,        │
                                            │  warm-seeded, joint-limit/clearance │
                                            └────────────────┬──────────────────┘
                                                             │ q_chunk (7-DoF)
                                                             ▼
                                            ┌───────────────────────────────────┐
                                            │  Guarded executor                  │
                                            │  receding horizon + guards +        │
                                            │  RB-Y1 SDK JointPositionCommand     │
                                            └────────────────┬──────────────────┘
                                                             ▼
                                                   RB-Y1 onboard controller
```

No MoveIt / Pinocchio / KDL / external PID — only `rby1_sdk` + our `robot/kinematics/` layer.

---

## 2. The two trained policies (current, deployed)

Both are **EEF action-space** policies that route through our IK. (Joint-space variants also exist
but are *not* the chosen deployment — see §10.)

| Property | Harvest | Deleaf |
|---|---|---|
| Policy dir | `harvest_bc_eef_trimmed_nojitter10hz_chunk30_seed11` | `deleaf_bc_eef_trimon_jitter10hz_chunk50_seed11` |
| Primitive | `harvest` (left arm, gripper) | `deleaf` (right arm, knife) |
| `action_space` | `eef` | `eef` |
| Sample rate | 10 Hz | 10 Hz |
| Chunk size | 30 (3.0 s) | 50 (5.0 s) |
| Action count | 180 (30×6) | 300 (50×6) |
| Training method | `goal_conditioned_bc` | `goal_conditioned_bc` |
| Demos | 2 (AT1T1 87.3 s, AT1T2 66.3 s) | 5 (AD1, BD1–BD4 ~34.7 s each) |
| Training chunks | 1,537 | 29,529 |
| Idle-prefix trim | on | on |
| Demo goal(s) | `[0.570, 0.068, 0.882]` (both ≈identical) | per-leaf, e.g. AD1 `[0.692, 0.138, 1.243]` … BD4 `[0.627,-0.295,0.829]` |

---

## 3. Input representation (23 features, identical train ↔ deploy)

The feature vector is built identically in `train_action_chunk_policy.py` and at deploy in
`predict_action_chunk_policy.py` / `guarded_bc_rollout.py`, all in the **robot base frame**:

| Block | Dims | Meaning |
|---|---|---|
| `goal_robot_m` | 3 | EgoDelta target position (robot frame) |
| `goal_minus_tcp_robot_m` | 3 | vector from current TCP to goal |
| `q_arm_rad` | 7 | current arm joint angles |
| `dq_arm_rad_s` | 7 | current arm joint velocities |
| `tcp_position_robot_m` | 3 | current TCP position (from FK) |

Features are standardized with stored `bc_x_mean` / `bc_x_std` (same constants train and deploy).
This policy carries **no phase / GCSL / deleaf-side features** — it is a Markovian feedback policy on
state + goal.

---

## 4. Output representation & data relabeling

**EEF action (the design choice):** each chunk step is an **absolute future TCP pose in the robot
frame**:

```
[ pos_x, pos_y, pos_z (m),  aa_rx, aa_ry, aa_rz (rad) ]   # 6 numbers/step
```

- Rotation is an **axis-angle rotation vector** (radians; ‖v‖ = angle). Encoded in training via
  `axis_angle_from_rotation` and reconstructed at runtime via `rotation_from_axis_angle`
  (`robot/kinematics/geometry.py`). The two are exact inverses (round-trip error ≈ 1e-13°).
- The contract is stored in metadata: _"Flattened chunk of absolute future TCP poses in the robot
  frame … Decode each EEF pose to a joint command with the RB-Y1 IK layer at runtime (y_abs ==
  y_delta for this action space)."_ i.e. the EEF target is **absolute**, not a delta.

**Why absolute TCP poses (vs joint deltas):**
- **Retargetable** — a new `goal_robot_m` changes where the policy aims; IK re-solves to reach it.
- **Robust to joint drift** — the controller is corrected in task space each step (IK reaches the
  pose), unlike a joint-delta replay that compounds error.
- **Matches the paper's method** (EEF DP/ACT pipeline).

(The legacy joint policies relabel differently: `y_delta = future_q − q_now`, cumulative-from-now
joint deltas, applied as `q = q_now + Δ`. Kept for comparison only.)

---

## 5. IK decode layer (our solver)

`robot/policy/eef_action_utils.py :: decode_eef_chunk_to_joints` converts the predicted EEF chunk to a
joint chunk:

- For each step, calls **`robot/kinematics/ik_solver.py :: solve_tcp_pose_ik`** (bounded
  least-squares, SciPy TRF; finite-difference Jacobian), **warm-seeded** from the previous step's
  solution so the joint path stays continuous.
- Only the **active arm's 7 joints** are solved; base / torso / head / other arm are frozen at the
  locked home state (`locked_state`).
- `IKSolverConfig` defaults: `position_weight 20.0`, `orientation_weight 1.0`, `posture_weight 0.01`,
  `smoothness_weight 0.02`, `clearance_weight 2.0`, `clearance_margin_m 0.03`,
  `joint_limit_margin_rad 0.05`, `max_iterations 100`, `ftol/xtol/gtol 1e-6`.
- Per-step acceptance tolerances: **position ≤ 0.01 m, orientation ≤ 0.15 rad**.

The decode **enforces joint limits and self-clearance** through the solver, so it makes no assumption
about SDK twist conventions (no hand-rolled Jacobian on the runtime path).

---

## 6. Training

- **Algorithm:** goal-conditioned behavior cloning, MLP, **action chunking** (predict the next
  `chunk_size` steps at once).
- **Data:** teleop / IK-solved demonstrations, resampled to 10 Hz, **idle-prefix trimmed**
  (`trim_idle_prefix=on`) — critical: untrimmed demos created a large "stay still" basin that made
  policies barely move (deleaf no-trim eval was 17.4 mm vs 0.5 mm trimmed).
- **Targets:** EEF policies store absolute TCP poses (`y_abs == y_delta`); joint policies store
  cumulative joint deltas.
- **Normalization:** input standardized; arrays (`x`, `y_delta`, `y_abs`, `source_times_s`,
  `bc_x_mean/std`, `bc_y_mean/std`) saved with `--store-training-arrays` for exact-data eval.
- Entry point: `robot/policy/train_action_chunk_policy.py` (`--action-space {joint,eef}`).

---

## 7. Runtime executor — `robot/policy/guarded_bc_rollout.py`

**Receding horizon, no temporal ensembling.** Each iteration: read live state → build features →
`predict_chunk` → (EEF) IK-decode → clip/guard → execute a prefix → re-read → re-predict. (Grep
confirms zero ensembling/averaging code — there is no off-by-one ensembling risk.)

**`predict_chunk`** branches on `action_space`:
- `eef` → reshape to `(chunk,6)` → `decode_eef_chunk_to_joints` (requires `kinematics` +
  `locked_state`) → returns `(Δ, q_chunk, decode_summary)`.
- `joint` → `(Δ, q_now+Δ, None)`.

**Execution modes (`execute_prefix`):**
- `all_steps` — command every prefix row via a blocking `send_command(..., minimum_time_s).get()`.
  At `minimum_time_s = 0.1` this reproduces the **10 Hz demo cadence** (faithful).
- `endpoint` — command only `q_chunk[steps-1]` (one move/iteration). **Do not use for these
  policies** — with small `steps_to_execute` it runs many× the demo speed and the closed-loop drifts.

**Guards** (all on the decoded `q_chunk`): workspace bounds, self-clearance, joint limits, per-step
delta (`--clip-step-delta`, `--max-step-delta-rad`), joint-limit clip (`--clip-joint-limits`),
progress guard. A failed guard halts before the next prefix.

**NEW — `eef_ik_guard` (added 2026-05-29):** the IK decode summary (max position/orientation error,
`all_ik_ok`, `all_within_tolerance`) is now logged per iteration, and motion **halts with
`eef_ik_decode_failed`** if IK didn't converge or any step's TCP position error exceeds
`--eef-ik-max-position-error-m` (default 0.03 m). This closes the prior gap where unreachable EEF
poses were silently clipped and commanded.

**Harvest gripper schedule** (`maybe_apply_harvest_gripper_schedule`): phase/distance-driven
open→close→release keyed to `tcp_goal_distance` and `phase_fraction`:
- `pre_grasp_open` at start; `grasp_close` when `distance ≤ --harvest-gripper-close-distance-m`
  (**0.06 m ≈ the 7.3 cm jaw/tool offset**, *not* 0.02) or phase-backstop; `basket_release_open` at
  the fixed-tail `basket_place`.
- Gripper saturating command is `-8.0` (open) / close target; bridge at the gripper HTTP server.

**Fixed basket tail** (`--harvest-auto-fixed-basket-tail`): after the policy reaches grasp, a
**deterministic** IK-solved retract→basket sequence takes over (gated on `grasp_closed`), so the
drift-prone policy never has to learn the basket motion.

---

## 8. Evaluation (exact-training-data, `eval_policy_on_training_data.py`)

| Policy | Metric | Result |
|---|---|---|
| Harvest EEF | open-loop pose fidelity | ≈ 1.6 mm / 0.73° |
| Deleaf EEF (trim-on) | pos / rot RMSE, all 5 labels | 0.48–0.60 mm / 0.12–0.14° |
| Harvest joint (overfit check) | first-step joint RMSE | 0.008° (AT1T1/AT1T2) |
| Axis-angle round-trip | geodesic | ≈ 1e-13° |

The overfit check proves the offline pipeline (features → labels → predict) is internally
self-consistent: the policies reproduce the training actions essentially perfectly. **Failures are
execution/deploy-side, not the learned model.**

---

## 9. Debugging journey & root causes (what shaped the design)

The harvest grasp kept failing. A multi-agent verification cleared the usual suspects and found the
real causes:

**Four advisor suspects — verdicts:**
1. **Axis-angle convention** — *N/A* (deployed harvest was joint-space; EEF round-trips exactly).
2. **Reference-frame mismatch** — *consistent* (train ↔ deploy identical; overfit RMSE 0.008°).
3. **Temporal-ensembling indexing** — *N/A* (we don't use ensembling; pure receding horizon).
4. **Hz mismatch** — *confirmed & relevant* (a run used `endpoint` + `min_time 0.1` + `steps 5` =
   **50 Hz = 5× the demo**).

**Actual root causes of the failed harvest grasps (joint policy):**
- **Over-speed drift:** `endpoint` at 50 Hz → arm lagged, plateaued at TCP **X ≈ 0.475, ~10 cm short**
  of the demo grasp (X ≈ 0.58); never reached contact.
- **Gripper threshold wrong:** demo grasps at **40–55 mm** TCP-to-goal (≈ the 7.3 cm jaw offset), so
  `grasp_close` at 0.02 m can never fire. Correct ≈ **0.06 m**.
- **Rollout over-run:** `full_action_duration_scale 1.1` + `extra_recede 5` pushed the policy past the
  trimmed grasp into OOD (drifted to X = 0.234, below the 0.22 x-floor → workspace guard). The fixed
  basket tail never ran.

**Fixes adopted:** EEF + IK (reaches the target pose, robust to joint drift); `all_steps` +
`minimum_time 0.1` (faithful 10 Hz); gripper close at 0.06; stop near grasp + hand off to the fixed
basket tail; and the new `eef_ik_guard` safety halt.

Note: the demo goal **equals** the deploy goal `[0.570, 0.068, 0.882]` exactly — and the demo TCP X
never goes below 0.422, so the rollout's X = 0.234 was OOD drift, not demo replay.

---

## 10. Key design decisions & rationale

- **EEF poses + our IK over joint deltas** — retargetable, task-space corrected, paper-aligned.
- **Reuse the deterministic IK layer** (`solve_tcp_pose_ik`) — no parallel/black-box IK; joint limits
  and clearance enforced.
- **Warm-seeded per-step decode** — continuous joint path across the chunk.
- **`all_steps` @ 10 Hz** — match the demo cadence; never `endpoint` for these policies.
- **Idle-prefix trim** — removes the "stay still" basin.
- **Snap goal to nearest demo** (`--snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02`) — the
  2-demo policies are effectively memorized; snapping keeps the goal in-distribution.
- **Deterministic basket tail** — offload the drift-prone post-grasp motion to validated IK
  waypoints.
- **IK guard + decode observability** — never command an unvalidated/unreachable pose (consistent with
  the robot-safety rule after the gain=30 incident).

---

## 11. Deploy commands

Always **dry-run first** (no `--execute`, no motion — validates IK + guards), inspect the log
(`eef_ik_guard.ok = true`, `max_position_error_m ≤ 0.03`, `guards.ok = true`), then run live with the
**e-stop in hand**.

### Harvest (left arm + gripper)
```powershell
python robot\policy\guarded_bc_rollout.py `
  --policy-dir robot\policy\trained_policies\harvest_bc_eef_trimmed_nojitter10hz_chunk30_seed11 `
  --target-source realsense-open3d --snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02 `
  --require-home --full-action --full-action-duration-scale 0.8 --extra-recede-iterations 0 `
  --execute-prefix-mode all_steps --steps-to-execute 5 --guard-horizon-steps 10 `
  --minimum-time-s 0.10 --inter-step-sleep-s 0.0 `
  --max-step-delta-rad 0.025 --clip-step-delta --clip-joint-limits `
  --eef-ik-max-position-error-m 0.03 `
  --harvest-gripper-schedule --harvest-gripper-close-distance-m 0.06 `
  --harvest-gripper-close-phase 0.5 --harvest-gripper-phase-close-max-distance-m 0.08 `
  --harvest-auto-fixed-basket-tail `
  --gripper-url http://192.0.2.20:8765
# live: append  --execute   (gripper HTTP bridge must be running)
```

### Deleaf (right arm + knife)
```powershell
python robot\policy\guarded_bc_rollout.py `
  --policy-dir robot\policy\trained_policies\deleaf_bc_eef_trimon_jitter10hz_chunk50_seed11 `
  --target-source realsense-open3d --snap-goal-to-nearest-demo --snap-goal-max-distance-m 0.02 `
  --require-home --full-action --full-action-duration-scale 1.0 --extra-recede-iterations 0 `
  --execute-prefix-mode all_steps --steps-to-execute 3 --guard-horizon-steps 10 `
  --minimum-time-s 0.10 --inter-step-sleep-s 0.0 `
  --max-step-delta-rad 0.017 --clip-step-delta --clip-joint-limits --max-z-m 1.35 `
  --eef-ik-max-position-error-m 0.03
# live: append  --execute    (no gripper flags — the knife is passive)
```

---

## 12. Known limitations & next steps

- **2-demo memorization:** harvest is trained on 2 near-identical demos (goals ≈ 3 mm apart) → it only
  reliably grasps at the demo tomato location. True generalization needs demo diversity.
- **IK compute cost:** the decode currently solves **every** chunk step each iteration (harvest 30×,
  deleaf 50× ≈ 800 solver evals each) while only the prefix is executed → slow live runs. **Planned
  optimization:** decode only `max(steps_to_execute, guard_horizon_steps)` steps (≈3× harvest, ≈5×
  deleaf speedup). Small change to `eef_action_utils.py` + the call site.
- **Gripper distance uses live TCP at iteration start** (one-iteration lag) — handled by the 0.06 m
  threshold; could be tightened by evaluating against the predicted TCP.
- **Basket handoff tuning:** confirm `--full-action-duration-scale` ends the policy right at grasp so
  the fixed tail takes over cleanly (bump to 1.0 if it stops short).

---

## 13. File map

| File | Role |
|---|---|
| `robot/policy/train_action_chunk_policy.py` | BC action-chunk trainer (`--action-space joint\|eef`) |
| `robot/policy/predict_action_chunk_policy.py` | feature vector + `predict_bc/knn/ridge` |
| `robot/policy/eef_action_utils.py` | EEF→joint IK decode (`decode_eef_chunk_to_joints`, `decode_summary`) |
| `robot/policy/guarded_bc_rollout.py` | guarded receding-horizon executor (+ `eef_ik_guard`) |
| `robot/policy/eval_policy_on_training_data.py` | exact-training-data eval (joint & eef aware) |
| `robot/kinematics/ik_solver.py` | bounded least-squares IK (`solve_tcp_pose_ik`) |
| `robot/kinematics/geometry.py` | SO(3) log/exp (`axis_angle_from_rotation`, `rotation_from_axis_angle`) |
| `robot/kinematics/types.py` | `IKSolverConfig`, `IKResult`, locked-state types |
| `robot/kinematics/rby1_model.py` | RB-Y1 FK / Jacobian / limits via `rby1_sdk` |

---

_Safety: live motion is run by the operator (never auto-fired); always dry-run first, keep the e-stop
in hand, motions slow/clipped, and commit only after end-to-end confirmation on real hardware._
