# Robot Calibration Artifacts

This folder contains the calibration and state files needed to reuse the current
RB-Y1 tomato-harvest setup without recalibrating.

## Core

Use these first when the physical camera/testbed setup has not changed:

- `core/realsense_head2_mount_left_eef_apriltag_bundle.json`
- `core/T_robot_from_aria_featuredepth_hanger_trusted.json`
- `core/T_robot_from_aria_featuredepth_hanger_trusted_left_toolfit.json`
- `core/T_robot_from_aria_featuredepth_hanger_trusted_right_toolfit.json`

For left-arm tomato harvesting, the main toolfit file is:

- `core/T_robot_from_aria_featuredepth_hanger_trusted_left_toolfit.json`

## Current Session Alignment

These are specific to the latest `a270c038` current-view alignment:

- `session/a270c038/T_robot_from_aria_selected_current_best_target_toolfit.json`
- `session/a270c038/selected_current_alignment_decision_best_target.json`

Reuse them only if the robot, camera, testbed, and current RealSense view are
effectively unchanged.

## Home And Ready State

- `home/harvest_left_ground_truth_20260523_204547_trial3_home_state.json`
- `home/rby1_ready_posture.json`

Use the home-state JSON to restore the robot without recalibrating.

## Gripper

- `gripper/left_gripper_live_aperture_calibration_20260524.json`

Known live left-gripper encoder values:

```text
open_position   = 1.784
closed_position = 10.945
```

## Target Frames

These are target/task-frame files for the current tomato scene:

- `targets/a270c038/harvest_tomato0_task_frame_instance_split_jaw_minusx.json`
- `targets/a270c038/harvest_tomato1_task_frame_instance_split_jaw_minusx.json`

Reuse them only if the tomato scene and EgoDelta masks have not changed.
