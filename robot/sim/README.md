# RBY1-A Isaac Sim

This directory contains Isaac Sim utilities for debugging RBY1-A motion against simple robot-frame targets.

## Sphere Scene

`rby1_sphere_scene.py` creates a minimal scene with:

- RBY1-A imported from `submodules/rby1-sdk/models/rby1a/urdf/model.urdf`
- fixed robot base
- gravity and a collision ground plane
- dome/key lighting
- a visible static target sphere in robot coordinates
- robot joints initialized from `robot/utils/homestate.json` when joint names match

Run from the repo root:

```powershell
D:\isaac-sim\python.bat robot\sim\rby1_sphere_scene.py
```

Run with the Isaac MCP extension enabled:

```powershell
D:\isaac-sim\python.bat robot\sim\rby1_sphere_scene.py --enable-mcp
```

Run with a solved arm pose:

```powershell
D:\isaac-sim\python.bat robot\sim\rby1_sphere_scene.py --enable-mcp --solution-json D:\tmp\egodelta_isaac_sphere_ik_solution.json
```

When `--solution-json` is used, the scene applies `robot\utils\homestate.json`
to every matching joint first, then applies only `left_arm_*` or `right_arm_*`
joints from the solution. Non-arm joints in a solution are rejected.

To move the target sphere:

```powershell
D:\isaac-sim\python.bat robot\sim\rby1_sphere_scene.py --sphere-position-m 0.58,0.03,0.90
```

## Boundary

The Isaac code should load and visualize the robot, target sphere, and later solved waypoint artifacts. It should not reimplement IK. The source of truth for FK, IK, task frames, and action retargeting remains `robot/kinematics`.
