# Zero-shot openpi (π0.5) deployment on RB-Y1-a

Deploys `pi05_droid` (Physical Intelligence openpi) as a **zero-shot** policy on the
RB-Y1-a, **dry-run first**. This is a research integration, not a working task policy:
`pi05_droid` was fine-tuned on a Franka (DROID) and outputs **Franka joint velocities**,
so on RB-Y1 its raw output is out-of-distribution. We bridge the embodiment gap by
retargeting through Cartesian space and routing the result through the existing
RB-Y1 IK + safety guards. **Expect erratic motion zero-shot; the clamps/guards are the
safety net, not the policy.**

## Architecture

```
 WSL2 (Ubuntu-22.04, GPU)                         Windows (egodelta_robot env)
 ┌─────────────────────────┐   websocket :8000   ┌──────────────────────────────┐
 │ openpi policy server     │  <--------------->  │ openpi_rollout.py (client)    │
 │ pi05_droid (PyTorch,cu128)│  obs / action chunk │  cameras → obs → retarget → IK│
 └─────────────────────────┘                      │  → guards → command_arm        │
                                                   └──────────────────────────────┘
                                                       │ rby1_sdk (Wi-Fi 192.0.2.10)
                                                       ▼  RB-Y1-a  +  gripper HTTP bridge
```

Why split: openpi's PyTorch path needs torch 2.7.1+cu128 (Blackwell) and Unix tooling →
runs in WSL2. `rby1_sdk` + RealSense run on Windows. They talk over the WSL2↔Windows
localhost bridge. (The default JAX path is broken on the RTX 5090 — ptxas too old,
openpi #419 — so we use the PyTorch path.)

## One-time server setup (WSL2)

Scripts live in `C:/path/to/projects/openpi_deploy_scripts/` (run with `bash <script>` inside WSL2;
they `sed` CRLF→LF first). They assume openpi is cloned at `~/openpi` with `uv sync` done.

1. `01_torch_patch.sh` — force torch to **2.7.1+cu128** (sm_120) and patch transformers.
   *Always* launch openpi with **`uv run --no-sync`**; plain `uv run` reverts torch to the
   cu126 lockfile pin and breaks the GPU.
2. `02_download_convert.sh` — download `pi05_droid` (JAX) and convert to PyTorch at
   `~/openpi_ckpts/pi05_droid_pytorch` (copies `assets/droid` norm_stats for un-normalization).
3. `03_serve.sh` — launch the policy server on `:8000`.
   First inference triggers a multi-minute `torch.compile` (max-autotune), then is fast.

## Cameras

`config/realsense_cameras.json` pins serials: base = D455 (scene/exterior), two D405 = wrist
(EEF) cams. `pi05_droid` uses **base + one wrist** (the active arm's). RGB only — depth unused.
Discover serials: `python -m robot.policy.openpi.multicam_capture --list`.
**TODO before motion:** confirm which D405 serial is the left vs right arm.

## Running the client (Windows, `egodelta_robot` env, repo root)

```bash
# 0) Server + cameras smoke test (no robot): capture -> infer -> print raw (15,8) chunk
python -m robot.policy.openpi.openpi_rollout --no-ik --prompt "pick up the object"

# 1) Full DRY-RUN (robot connected, STATE-ONLY, no motion): retarget -> IK -> guards -> print
python -m robot.policy.openpi.openpi_rollout --arm right --prompt "cut the leaf petiole"

# 2) Guarded motion (only after dry-run looks sane + serials/scale confirmed)
python -m robot.policy.openpi.openpi_rollout --arm right --prompt "..." --execute   # confirms each waypoint
```

Key flags: `--host/--port` (server), `--arm {left,right}`, `--wrist-serial`, `--steps-to-execute`,
`--loop` (receding horizon), and the safety knobs `--joint-vel-scale` (conservative; tune before
motion), `--max-trans-m`, `--max-rot-rad`, `--minimum-time-s`, `--velocity-limit-rad-s`. `--execute`
is the master motion gate (off = dry-run); `--no-confirm` skips the per-waypoint prompt.

## Action mapping (the retarget)

`pi05_droid` → `(15, 8)` chunk = 7 Franka joint velocities + 1 gripper, ~15 Hz receding-horizon.
`franka_retarget.py` evaluates a Franka Panda Jacobian at the joint angles we fed the model,
turns the joint velocities into an end-effector twist, integrates one step, clamps it, and
produces an absolute target TCP pose for `solve_tcp_pose_ik`. This makes the *task-space intent*
embodiment-agnostic. It is a heuristic — see the assumptions block in `franka_retarget.py`
(base-frame alignment, unknown velocity scale). The gripper dim drives the HTTP bridge only
when `--gripper` is passed (opt-in, and only under `--execute`): hysteresis (open > 0.6,
close < 0.4, else hold) avoids chatter near 0.5, and the bridge is commanded via position-mode
`/jog` (open -8.0; close 4.8 right / 10.945 left) only on an open<->close state change. Without
`--gripper` the gripper dim is decoded and printed (`grip=OPEN/CLOSE`) but NOT actuated.

## Safety

Dry-run by default. `--execute` requires every guard to pass (workspace box, IK convergence,
per-step joint delta, joint limits, self-clearance), uses slow `minimum_time` + low velocity/accel
caps, and confirms each waypoint. Per project policy, nothing here is committed until validated
end-to-end on hardware.
