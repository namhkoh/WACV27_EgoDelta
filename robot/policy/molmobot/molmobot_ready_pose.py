"""Move the RB-Y1 to the MolmoBot READY pose (episode start) — slow + confirmed.

MolmoBot RBY1 policies were trained from episodes initialized at molmo_spaces
``RBY1Config.init_qpos``: both arms at [0.5, 0, 0, -2.3, 0, -0.5, 0] rad (+/- small
noise), TORSO UPRIGHT (all zeros), head (pan 0, tilt 0.6 rad down). Poses far from
that start distribution make the policy emit ~zero actions (hardware-verified).

The canonical READY is resolvable two ways:
  * sim defaults (above) with --torso/--head/--head-tilt-rad selections, or
  * a USER-SAVED snapshot (``--save-current`` -> ready_state.json), which then
    overrides everything here AND the rollout's start-pose gate. Delete the file
    (or re-save) to change it; --head-tilt-rad still overrides the head.

Dry-run by default; ``--execute`` is the motion gate; exit code 2 = motion did not
finish Ok. E-stop in hand.

    python -m robot.policy.molmobot.molmobot_ready_pose                    # print plan
    python -m robot.policy.molmobot.molmobot_ready_pose --execute          # move (confirm prompt)
    python -m robot.policy.molmobot.molmobot_ready_pose --save-current     # snapshot live pose as READY
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rby1_sdk as rby

from robot.kinematics.utils.realsense_click_move_arm import (
    DEFAULT_ADDRESS,
    DEFAULT_HOME,
    command_positions,
    connect_motion_robot,
)

# molmo_spaces RBY1Config.init_qpos (sim episode start), both arms.
READY_ARM_RAD = np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0], dtype=float)
READY_HEAD_RAD = np.array([0.0, 0.6], dtype=float)  # pan 0, tilt 0.6 rad down (training view)

# User-saved canonical READY (via --save-current). When present it overrides the sim
# defaults everywhere: this script's targets AND the rollout's start-pose gate.
READY_STATE_PATH = Path(__file__).resolve().parent / "ready_state.json"


def load_ready_state() -> dict[str, np.ndarray] | None:
    """The saved canonical READY state, or None if never saved."""
    if READY_STATE_PATH.exists():
        d = json.loads(READY_STATE_PATH.read_text(encoding="utf-8"))
        return {k: np.asarray(v, dtype=float) for k, v in d.items()}
    return None


def _joint_cmd(position: np.ndarray, *, minimum_time_s: float, velocity_limit: float,
               acceleration_limit: float) -> "rby.JointPositionCommandBuilder":
    pos = np.asarray(position, dtype=float).reshape(-1)
    return (
        rby.JointPositionCommandBuilder()
        .set_minimum_time(float(minimum_time_s))
        .set_position(pos)
        .set_velocity_limit(np.full(pos.shape[0], float(velocity_limit), dtype=float))
        .set_acceleration_limit(np.full(pos.shape[0], float(acceleration_limit), dtype=float))
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--model", default="a")
    p.add_argument("--home", type=Path, default=DEFAULT_HOME)
    p.add_argument("--minimum-time-s", type=float, default=15.0)
    p.add_argument("--velocity-limit-rad-s", type=float, default=0.10)
    p.add_argument("--acceleration-limit-rad-s2", type=float, default=0.10)
    p.add_argument("--head", choices=("keep", "ready"), default="keep",
                   help="keep = hold the CURRENT head pose (default, per user preference); "
                        "ready = command the sim training viewpoint (pan 0, tilt 0.6 down).")
    p.add_argument("--head-tilt-rad", type=float, default=None,
                   help="Explicit head tilt (rad, +down; URDF limit 1.57). Overrides --head. "
                        "The D455's ~42deg vFOV needs ~0.8-0.9 to frame a near table from the "
                        "upright torso (sim's 139deg GoPro saw it at 0.6).")
    p.add_argument("--torso", choices=("keep", "home", "ready"), default="keep",
                   help="keep = hold the CURRENT torso; home = homestate torso (greenhouse "
                        "crouch [0,45,-90,45,0,0] deg); ready = SIM INIT torso (all zeros, "
                        "standing upright -- the pick_pnp training posture). The policy was "
                        "trained standing: a crouched torso shifts the whole reach frame.")
    p.add_argument("--save-current", action="store_true",
                   help="Snapshot the LIVE pose (arms+torso+head) into ready_state.json as the "
                        "canonical READY, then exit (no motion). Future runs + the rollout's "
                        "start gate use it instead of the sim defaults.")
    p.add_argument("--execute", action="store_true", help="MASTER MOTION GATE. Off = print plan only.")
    p.add_argument("--no-confirm", action="store_true")
    args = p.parse_args()

    if args.save_current:
        robot = rby.create_robot(args.address, args.model)
        if not robot.connect():
            raise RuntimeError(f"failed to connect to RB-Y1 at {args.address}")
        cur = np.asarray(robot.get_state().position, dtype=float)
        m = robot.model()
        names = list(m.robot_joint_names)
        head_idx = [i for i, n in enumerate(names) if n.startswith("head_")]
        state = {
            "left_arm": cur[[int(i) for i in m.left_arm_idx]].tolist(),
            "right_arm": cur[[int(i) for i in m.right_arm_idx]].tolist(),
            "torso": cur[[int(i) for i in m.torso_idx]].tolist(),
            "head": cur[head_idx].tolist(),
        }
        READY_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"saved canonical READY to {READY_STATE_PATH}:")
        for k, v in state.items():
            print(f"  {k} (deg): {np.round(np.degrees(np.asarray(v)), 1).tolist()}")
        return 0

    saved = load_ready_state()
    if saved is not None and (args.torso != "keep" or args.head != "keep"):
        print("[warn] ready_state.json exists -> --torso/--head selections are IGNORED "
              "(all targets come from the saved state). Delete "
              f"{READY_STATE_PATH} or re-run --save-current to change it. "
              "(--head-tilt-rad still overrides the head.)")
    left_ready = saved["left_arm"] if saved else READY_ARM_RAD
    right_ready = saved["right_arm"] if saved else READY_ARM_RAD

    home = json.loads(Path(args.home).read_text(encoding="utf-8"))
    torso = saved["torso"] if saved else command_positions(home, "torso")
    head = saved["head"].copy() if saved else READY_HEAD_RAD.copy()

    print("===== MolmoBot READY pose " + ("(USER-SAVED ready_state.json)" if saved else "(sim episode-start)") + " =====")
    print(f"  left  arm (deg): {np.round(np.degrees(left_ready), 1).tolist()}")
    print(f"  right arm (deg): {np.round(np.degrees(right_ready), 1).tolist()}")
    print(f"  torso: {'saved' if saved else args.torso}   head: {'saved' if saved else args.head}")
    print(f"  minimum_time={args.minimum_time_s}s  vel<={args.velocity_limit_rad_s} rad/s  "
          f"acc<={args.acceleration_limit_rad_s2} rad/s^2")

    if not args.execute:
        print("\nDRY-RUN (no --execute): nothing sent. Re-run with --execute to move.")
        return 0

    robot = connect_motion_robot(address=args.address, model=args.model, power=".*", servo=".*")
    cur = np.asarray(robot.get_state().position, dtype=float)
    model = robot.model()
    if saved is None and args.head == "keep":
        # adopt the CURRENT physical head pose (user aims it at the table; we hold it)
        names = list(model.robot_joint_names)
        head_idx = [i for i, n in enumerate(names) if n.startswith("head_")]
        head = cur[head_idx].astype(float)
        print(f"  head (kept from live): pan {np.degrees(head[0]):+.1f} deg, tilt {np.degrees(head[1]):+.1f} deg")
    if args.head_tilt_rad is not None:
        head = np.array([0.0, float(np.clip(args.head_tilt_rad, -0.35, 1.57))], dtype=float)
        print(f"  head OVERRIDE: pan 0.0, tilt {np.degrees(head[1]):+.1f} deg down")
    if saved is None and args.torso == "ready":
        torso = np.zeros(6, dtype=float)  # sim init_qpos: standing upright
        cur_torso = cur[[int(i) for i in model.torso_idx]]
        print(f"  torso: READY (all zeros, upright). current(deg)={np.round(np.degrees(cur_torso),1).tolist()}"
              f" -> target zeros; max|delta|={np.degrees(np.max(np.abs(cur_torso))):.1f} deg -- CLEAR THE AREA")
    elif saved is None and args.torso == "keep":
        torso = cur[[int(i) for i in model.torso_idx]].astype(float)
        print(f"  torso (kept from live): {np.round(np.degrees(torso),1).tolist()} deg")
    for arm_name, idx, tgt in (("left", model.left_arm_idx, left_ready), ("right", model.right_arm_idx, right_ready)):
        cur_arm = cur[[int(i) for i in idx]]
        print(f"  {arm_name} arm current(deg): {np.round(np.degrees(cur_arm), 1).tolist()}")
        print(f"  {arm_name} arm  target(deg): {np.round(np.degrees(tgt), 1).tolist()}  "
              f"max|delta|={np.degrees(np.max(np.abs(cur_arm - tgt))):.1f} deg")

    if not args.no_confirm:
        ans = input("\nMove BOTH arms + head to READY now? Watch the arms; e-stop in hand. [y/N] ").strip().lower()
        if ans != "y":
            print("aborted.")
            return 1

    kw = dict(minimum_time_s=args.minimum_time_s, velocity_limit=args.velocity_limit_rad_s,
              acceleration_limit=args.acceleration_limit_rad_s2)
    body = (
        rby.BodyComponentBasedCommandBuilder()
        .set_torso_command(_joint_cmd(torso, **kw))
        .set_right_arm_command(_joint_cmd(right_ready, **kw))
        .set_left_arm_command(_joint_cmd(left_ready, **kw))
    )
    cmd = rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(body)
        .set_head_command(_joint_cmd(head, **kw))
    )
    print("[motion] sending READY pose command ...")
    rv = robot.send_command(cmd, 10).get()  # NOTE: 10 is the SDK command PRIORITY, not a timeout
    finish_ok = rv.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
    print(f"[motion] finish code: {rv.finish_code}" + ("" if finish_ok else "  <-- NOT Ok: READY pose NOT reached"))
    return 0 if finish_ok else 2


if __name__ == "__main__":
    sys.exit(main())
