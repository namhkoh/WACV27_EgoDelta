# MolmoBot RB-Y1 manipulation deployment

Deploys allenai/MolmoBot's **native RB-Y1** checkpoints on our RB-Y1-a. Unlike the
openpi/Franka path, MolmoBot was trained (in MolmoSpaces sim) directly on RB-Y1, so it
outputs **RB-Y1 joint-space actions** — per-arm joint-position *deltas* + absolute
gripper + absolute torso. No Franka retarget, no IK: we command the joints directly.

## Honest expectation
MolmoBot's real-world results are reported on **Franka/DROID only**; its **RB-Y1 numbers
are simulation-only** (sim pick-and-place ~22.5%) and the authors did not validate RB-Y1
on real hardware. So this is a sim→real attempt on RB-Y1 — structurally the right model
(it knows the embodiment), but not a guaranteed success. Treat it as a strong baseline to
*evaluate*, with the safety harness doing the safety, not the policy.

## Architecture
```
 WSL2 (GPU)                              Windows (egodelta_robot)
 MolmoBot policy server  <-- websocket -->  molmobot_rollout.py
 (Molmo2-4B + flow head,                    cameras + qpos -> obs dict -> infer
  torch cu128, :8000)                       -> joint deltas -> command both arms
```
The MolmoBot WebSocket server is a fork of openpi's, so our existing `openpi_client`
transport talks to it directly — only the obs/action *schema* differs.

## 1. Server (WSL2) — one-time install + serve
- Install: `bash /mnt/d/research/openpi_deploy_scripts/molmo_01_install.sh`
  (clones into `~/MolmoBot`, `uv sync --extra eval`, then forces torch **cu128** for the
  5090; verified `torch 2.7.1+cu128`, cap (12,0)). Always launch with `uv run --no-sync`.
- Serve pick-and-place: `bash /mnt/d/research/openpi_deploy_scripts/molmo_02_serve.sh`
  → `serve_molmobot_rby1_multitask.py --task_type pick_pnp --hf_repo allenai/MolmoBot-RBY1Multitask --port 8000`
  (first run downloads the checkpoint). Wait for `Serving pick_pnp policy on ws://0.0.0.0:8000`.
  - Door/articulated variant: `--task_type door_plus_open` (also needs point prompts +
    a conditioning image in the obs — not wired in our client yet; use pick_pnp first).

## 2. Client (Windows, egodelta_robot, repo root)
```powershell
# (a) Smoke: cameras + infer + print the action dict (no robot)
python -m robot.policy.molmobot.molmobot_rollout --no-robot --task "pick up the red mug"

# (b) DRY-RUN with robot (state-only, no motion): qpos -> infer -> clamp -> joint targets -> guards
python -m robot.policy.molmobot.molmobot_rollout --arms right --task "pick up the red mug" --read-gripper-state

# (c) Guarded motion, single arm first (de-risk joint mapping), per-waypoint confirm:
python -m robot.policy.molmobot.molmobot_rollout --arms right --task "..." --execute
# then dual-arm + gripper, once joint mapping is validated:
python -m robot.policy.molmobot.molmobot_rollout --arms both --task "..." --execute --gripper --read-gripper-state --loop --iterations 30
```

## Obs / action contract (pick_pnp)
- **Obs**: `head_camera` (D455, server fisheye-warps it), `wrist_camera_l` (D405 LEFT_WRIST_CAMERA_SERIAL),
  `wrist_camera_r` (D405 RIGHT_WRIST_CAMERA_SERIAL) — native RGB uint8; `qpos`{base(3, we send 0),
  left_arm(7), left_gripper(1 norm), right_arm(7), right_gripper(1 norm), torso(6 raw)};
  `task` string; `reset` on first tick.
- **Action**: `left_arm`/`right_arm`/`base` = joint-position **deltas**; `left_gripper`/
  `right_gripper` = absolute (binary ±100 → open/close); `torso` = absolute height.
- We **ignore the base action** (never drive the mobile base) and **pin the torso to home**
  (`--drive-torso` not implemented — a 1-D height→6-joint mapping is TBD).

## ⚠️ #1 risk to validate first: joint ordering
MolmoBot's sim joint order (`molmo_spaces.RBY1MConfig`, external) may not match our
`RBY1Kinematics` order. **Before trusting `--execute`**, validate joint-by-joint in dry-run:
watch the printed `action.<arm> Δ(deg)` per joint and confirm, on a single small commanded
move, that the joint that physically moves is the one you expect. Also confirm the served
checkpoint actually un-normalizes (delta scale sane, not tiny/huge). A permuted or
sign-flipped joint will drive the wrong way — the per-step clamp (`--max-step-delta-rad`)
and self-clearance guard are the safety net.

## Safety
Dry-run by default; `--execute` is the master gate; per-waypoint `--confirm` unless
`--no-confirm`; slow `minimum_time` + low velocity/accel caps; hard per-joint delta clamp;
self-clearance + joint-limit clamp. E-stop in hand for any `--execute`. Nothing committed
to git until validated end-to-end on hardware.
