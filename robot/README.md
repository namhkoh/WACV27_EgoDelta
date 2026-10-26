# Robot Package

This directory is a clean handoff package for the final robot work. It keeps the
minimum reusable calibration/state artifacts separate from the larger experimental
robotics codebase.

## Layout

- `calibration/`: reusable calibration, alignment, home-state, gripper, and
  target-frame JSON artifacts.
- `server/`: gripper HTTP bridge/server code and matching client utility.

The files here are copied from the working harvest/debug session. They are meant
to avoid repeating calibration when the physical setup is unchanged.

## Recalibration Rule

Do not recalibrate if the robot base, RealSense head mount, testbed, basket, and
gripper hardware have not moved. Reuse the JSON files in `calibration/` and only
refresh live FK/state as needed.

Recalibrate or re-align if any of these changed:

- RealSense camera mount or head bracket;
- robot base/testbed relative pose;
- basket position;
- gripper encoder behavior;
- tomato scene or EgoDelta masks;
- current RealSense/Aria alignment evidence.
