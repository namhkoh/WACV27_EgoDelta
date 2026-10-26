# Robot Server

This folder contains the gripper HTTP bridge code used during the harvest
experiments.

## Files

- `rby1_gripper_http_bridge.py`: HTTP bridge/server for gripper commands and
  status.
- `rby1_gripper_http_client.py`: small client utility for status/open/close/grasp
  commands.

The expected bridge URL used in the experiments was:

```text
http://192.0.2.20:8765
```

The server owns the gripper bus while it is running. Do not run another gripper
bridge or teleop process that owns the same bus at the same time.
