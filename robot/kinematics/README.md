# Robot Kinematics And Retargeting

This package owns the robot-side kinematics, task-frame retargeting, and IK
validation code for 3D-mask-driven agricultural manipulation. It uses the
RB-Y1 SDK as the single source of truth for FK, Jacobians, joint limits, and
collision queries. It does not modify SDK submodules or duplicate the robot
model in another kinematics library.

## Method

The retargeting pipeline converts an EgoDelta 3D target into a robot action
through five explicit stages:

1. **Scene anchoring**
   The 3D mask target is transformed into the robot frame using the current
   RealSense-to-Aria and robot-to-RealSense calibration. Live depth and
   projection checks must pass before the target is considered commandable.

2. **Action prior extraction**
   Aria MPS hand tracking is segmented by action phase. For harvesting, the
   relevant phases are approach, pre-grasp, grasp, harvest, retract, and basket
   place. `hand_prior.py` maps frame windows to MPS timestamps, uses SLAM to
   express moving device-frame hand landmarks in the anchored Aria frame, then
   applies the calibrated Aria-to-robot transform. MPS provides a
   confidence-weighted prior for approach direction, pinch/jaw axis, harvest
   pull direction, waypoint timing, and motion style.

3. **Task-frame loading and generation**
   `task_frame.py` loads mask-derived harvest task-frame JSONs and validates
   target position, task axes, TCP/gripper orientation, aperture fields, quality
   flags, and the no-replay contract. The mask geometry and hand prior then
   produce or refine candidate task frames. The final grasp frame is not a
   hard-coded angle. It is selected by scoring candidate jaw axes and approach
   directions against hand evidence, mask geometry, robot reachability,
   clearance, and smoothness.

4. **IK solving**
   IK is solved in a fixed kinematic subspace for the first harvesting version:
   base, torso, head, and right arm are locked to home; only the left arm moves.
   The solver optimizes the calibrated TCP pose, not the bare EEF link pose.

5. **Execution gates**
   Plans are validated before motion: FK residual, joint limits, Jacobian
   sanity, path smoothness, robot self-clearance, non-target fruit clearance,
   TCP convention, and live RealSense consistency. Contact actions require
   stricter checks than non-contact pregrasp waypoints.

## Grasp Angle Selection

The grasp angle is determined by scoring candidate task frames, not by tuning a
single script parameter. The score combines:

- MPS thumb-index pinch axis stability during pre-grasp and grasp phases.
- Local 3D mask geometry: fruit center, visible surface, width, and depth.
- Approach and pull directions inferred from hand motion.
- IK feasibility, joint-limit margin, collision clearance, and path smoothness.

When MPS hand orientation is weak, the solver lowers its weight and falls back
to mask geometry plus robot feasibility. MPS still contributes phase timing and
trajectory style. This keeps the method target-agnostic and publishable: hand
tracking is used as a structured prior, not as an opaque replay trajectory.

## Package Layout

- `types.py`: small dataclasses shared across kinematics, retargeting, and IK.
- `rby1_model.py`: RB-Y1 SDK wrapper for FK, calibrated TCP poses, joint limits,
  and Jacobian validation.
- `run_outputs.py`: dated run-artifact path utilities.
- `phase_alignment.py`: timestamp-based MPS/video phase alignment from
  target-conditioned Aria-frame pinch proximity, with confidence diagnostics.
- `hand_prior.py`: MPS phase windows, SLAM-aligned hand landmarks, and
  confidence-weighted hand priors.
- `task_frame.py`: target-frame JSON adapter/validator and candidate task-frame
  generation from mask geometry and hand priors.
- `ik_solver.py`: bounded least-squares IK over the active arm joints.
- `validation.py`: reusable FK, Jacobian, calibration, and plan sanity checks.
- `planning/harvest_candidates.py`: target-agnostic harvest waypoint candidate
  families, including straight, elevated, lateral, combined, and soft
  orientation-ramp strategies. Candidate TCP orientations may roll around the
  task backoff axis, and candidate grasp anchors may be inserted from the mask
  surface target along `-backoff_axis` using task-frame depth geometry.
  Pre-grasp backoff is generated from the inserted grasp depth plus a swept
  target-surface clearance, with the configured pre-grasp distance used only as
  a lower bound. Palm-axis fruit-center correction is a candidate dimension, not
  a fixed per-target shift. Bounded over-insertion is represented as a
  dimensionless factor over the mask-derived nominal insertion depth, capped by
  `--max-grasp-insertion-factor`; the accepted side-grasp candidate must reach a
  configured fraction of that nominal depth so the jaws are planned around the
  fruit body rather than at the visible surface. This is a candidate policy, not
  a per-target coordinate adjustment.
- `apps/solve_task_frames.py`: no-motion single-waypoint IK smoke validation for
  validated task-frame JSONs and backoff-offset waypoints, with dated artifacts
  and hard post-solve gates.
- `apps/plan_harvest_staged.py`: no-motion staged harvest IK planner that ranks
  generated waypoint candidates and validates endpoint plus interpolated
  path-clearance gates. Candidate ranking includes an open-alignment gate for
  harvesting: the `pre_grasp` waypoint where the gripper opens must remain
  centered on the task jaw/palm frame, while lateral or elevated offsets may be
  used for approach routing. Grasp depth is also gated: inserted candidates use
  half the estimated fruit depth when available, fall back to the fruit-center
  projection along `-backoff_axis`, and are checked for pre-grasp surface
  clearance. When tail waypoints are enabled, the `harvest` waypoint combines
  the task pull direction with a configurable robot-frame downward detach
  component before retract. The planner solves and ranks the contact prefix
  first, freezes the selected prefix, then sweeps retract and basket-release
  tails so basket feasibility cannot silently alter the grasp geometry. The
  `pre_grasp` open waypoint has stricter target-frame alignment gates than
  generic non-contact waypoints. For round-fruit harvesting, the planner can
  evaluate `pre_grasp` and `grasp` with task-axis gates: the task jaw and
  backoff axes must align within bounded angular tolerances, while contact
  position tolerances scale from the estimated fruit width. This avoids
  over-constraining wrist roll with a rigid quaternion when the task only
  requires the gripper jaws and approach direction to be correct. When multiple
  task-frame targets are supplied, non-target fruit instances are modeled as
  inflated sphere obstacles;
  endpoint TCPs and dense joint-interpolated TCP paths must clear those
  obstacles. During open-gripper phases, clearance is checked against a
  TCP-frame oriented gripper box with separate local x/y/z half-extents instead
  of a uniform clearance sphere. Scene-avoidance approach offsets are derived
  from target-to-obstacle geometry, not target names. Post-grasp tail candidates
  are checked against a fixed-testbed TCP x envelope, and reverse-pull retracts
  are disabled unless explicitly requested. Use `--contact-frame-json` to load
  the calibrated TCP-frame offset from modeled TCP origin to physical gripper
  contact center; mask targets then solve for that contact center instead of the
  abstract TCP origin.
- `apps/execute_harvest_plan.py`: guarded live-motion executor for a validated
  staged plan artifact. It is dry-run by default and refuses artifacts whose
  selected candidate does not pass hard gates. Before live execution, it checks
  commanded torso, arm, and head joints against `robot/utils/homestate.json`.
  `--stop-after` can terminate a live validation at a named waypoint, such as
  `grasp`, before harvest/retract motion. If the artifact includes
  `basket_place`, the executor opens the gripper there by default to release the
  harvested target. When gripper control is enabled, it commands the calibrated
  full-open target after arriving at `pre_grasp`, verifies the open-side encoder
  position, and logs open/close gripper observations in the execution summary.
- `utils/pick_basket_point.py`: Open3D point picker for selecting the fixed
  basket location from the current aligned point cloud. When the matching
  calibration JSON is present, it prints the selected point as
  `--basket-anchor-robot-m x,y,z`. The harvest planner treats this as a basket
  anchor, then sweeps release poses above and inward from that anchor before
  accepting a path-checked `basket_place` waypoint.
- `utils/capture_live_realsense.py`: live RealSense RGB-D capture and
  robot-frame point probe. Use `--pixel-x/--pixel-y` for GUI-free point checks
  when OpenCV picking is unreliable.
- `utils/report_live_tcp_pose.py`: read-only live FK diagnostic that prints the
  calibrated EEF/TCP pose and can compare it against a selected plan waypoint.

Each executable Python file in this package must have one clear purpose and a
safe dry-run mode when it can affect hardware.

## Run Artifacts

Solver outputs are not written beside source files as loose JSON. Any solved
path, validation report, failed candidate, or execution log must use a dated run
directory:

```text
robot/kinematics/runs/MMDDYY/task_name/artifact_name.json
```

For example, a tomato-0 pregrasp solve on May 24, 2026 should write something
like:

```text
robot/kinematics/runs/052426/tomato0_pregrasp/a270c038_tomato0_ik_plan_seed_home_backoff_10cm.json
```

Use `run_outputs.py` to create these paths. Artifact names should encode the
dataset, target, task, and key parameters well enough to distinguish successful
and failed runs.

## Initial Safety Policy

For first harvesting validation:

- left arm only;
- base, torso, head, and right arm locked to `robot/utils/homestate.json`;
- gripper controlled separately from IK;
- gripper must be commanded to the calibrated full-open command after arriving
  at `pre_grasp`, then verified by encoder position before contact motion;
- the gripper-open waypoint must be aligned to the target jaw/palm frame, not
  laterally offset just because that improves arm clearance;
- the final grasp TCP must use a task-derived inserted grasp anchor rather than
  stopping at the mask surface/contact point when fruit depth geometry is
  available;
- pre-grasp distance must be derived from the inserted grasp depth and desired
  target-surface clearance, not manually tuned per target;
- side-grasp insertion must reach the configured fraction of the mask-derived
  nominal depth before a candidate can pass, preventing shallow surface-only
  grasps from ranking above deeper feasible contacts;
- palm-axis fruit-center correction must be evaluated as a candidate dimension,
  and `pre_grasp` must satisfy open-gripper alignment gates before any live
  execution; for round fruit, those gates should constrain task-relevant jaw and
  backoff axes rather than an arbitrary full wrist roll;
- final grasp anchors must satisfy a palm-axis fruit-center alignment gate, so a
  candidate cannot pass only because it preserves the original mask contact
  height when target geometry indicates a lower grasp plane;
- mask/task-frame targets should be interpreted as desired gripper contact-center
  targets; `--contact-frame-json` converts those targets into commanded TCP poses
  through `T_tcp_from_gripper_contact_center`;
- non-target fruit clearance must account for the open gripper at `pre_grasp`
  and `grasp`; TCP-only obstacle clearance is not sufficient for adjacent fruit;
  the current approximation is a TCP-frame oriented gripper box where local x is
  jaw width, local y is palm thickness, and local z is backoff depth;
- basket placement uses an explicit basket anchor or direct TCP pose and is
  path-checked like every other waypoint before live execution;
- basket/tail motion must remain inside the configured fixed-testbed forward
  reach envelope until trellis, rail, basket, and held-fruit geometry are modeled
  explicitly;
- no contact motion until TCP validation and live RealSense checks pass;
- no straight-backoff approach is assumed safe until IK clearance gates pass;
- no approach path may pass through another known fruit instance; non-target
  fruit clearance is checked separately from robot self-clearance;
- no plan-specific hot fixes in solver code.
- no basket-place motion until a basket anchor/release pose is explicitly
  supplied and the generated `basket_place` waypoint passes hard gates.
