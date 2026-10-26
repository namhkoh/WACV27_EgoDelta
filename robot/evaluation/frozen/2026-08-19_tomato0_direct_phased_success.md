# Frozen Tomato0 Direct Harvest Pipeline

This freeze records the physically successful tomato0 harvest from 2026-08-19.
The machine-readable provenance is in
`20260819_tomato0_direct_phased_success.json`.

## Classification

- Direct deterministic controller, not a learned-policy rollout.
- `controller_execution_mode=deterministic_tail_only`.
- The BC wrapper performed zero learned motion steps.
- The left arm was the only commanded robot joint group.

## Successful Configuration

- RealSense RGB target pixel: `[248, 408]`.
- Robot target: `[0.5708606633, 0.1177140471, 0.8934811079]` m.
- Fixed-tail template: `AT1T1`.
- Phases: approach, orientation ramp, pre-grasp/open, grasp/close,
  harvest, retract, basket-place/release.
- Extra insertion: `0.02 m`.
- Added grasp height: `0.0 m`. The former `+0.025 m` setting was too high.
- Waypoint minimum time: `4.0 s`.
- Left-arm velocity/acceleration limits: `0.15 rad/s`, `0.08 rad/s^2`.
- Gripper targets came from live status at `192.0.2.21:8765`.

## Scope

Pixel `[248, 408]` is frozen evidence for this exact testbed and camera pose. It
must not be reused after the tomato, testbed, head, or RealSense pose moves.
Wider trials must bind each fresh target through validated instance identity.

The successful run may be counted only in the direct-controller condition. It
must not be presented as learned-policy success.

After the run, all 22 checked robot joints were restored to
`robot/utils/homestate.json` within `0.088 deg`, and the left gripper was closed.
