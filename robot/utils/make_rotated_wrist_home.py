"""Write a new RB-Y1 home-state JSON with the wrist-roll joints (arm_6) rotated.

Used to re-orient the wrist (D405) cameras toward a more DROID-like alignment. Applies
a per-arm DELTA (in degrees) to the wrist-roll joint (right_arm_6 / left_arm_6) of an
existing home JSON and writes a NEW file (never overwrites the input).

Only joint positions are edited; the FK transforms (T_base_link) in the file become
STALE. After you command the robot to this posture and confirm the cameras look right,
re-capture a fully-consistent home with capture_home_state.py.

Direction ("inwards / clockwise") depends on the physical camera/flange mounting and the
per-arm joint sign convention, which differ between the mirrored arms -- so COMMAND it
slowly and check the live camera view, then flip a sign if a wrist rotated the wrong way.

Typical workflow:
  # 1) make the rotated-wrist home (signs are a starting guess -- verify in step 4)
  python -m robot.utils.make_rotated_wrist_home --output robot/utils/homestate_wrists90.json \
      --right-wrist-roll-deg -90 --left-wrist-roll-deg 90
  # 2) preview the move (no motion)
  python -m robot.utils.restore_home_state --home robot/utils/homestate_wrists90.json --scope arms
  # 3) execute slowly (e-stop in hand; wrist roll only -> small swept volume)
  python -m robot.utils.restore_home_state --home robot/utils/homestate_wrists90.json --scope arms --execute --minimum-time 8
  # 4) verify the wrist cams are aligned; if a wrist went the wrong way, flip its sign and redo 1-3
  python -m robot.policy.openpi.multicam_view
  # 5) finalize a clean home (correct FK) once happy:
  python -m robot.utils.capture_home_state --output robot/utils/homestate_wrists90.json
  # then pass --home robot/utils/homestate_wrists90.json to the rollout / restore from now on.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_INPUT = Path(__file__).with_name("homestate.json")
WRIST_ROLL_LOCAL_INDEX = 6  # arm_6 is the 7th (last) arm joint = wrist roll


def main() -> None:
    p = argparse.ArgumentParser(description="Write a home JSON with rotated wrist-roll (arm_6) joints.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--right-wrist-roll-deg", type=float, default=0.0, help="Degrees added to right_arm_6.")
    p.add_argument("--left-wrist-roll-deg", type=float, default=0.0, help="Degrees added to left_arm_6.")
    args = p.parse_args()

    if args.output.resolve() == args.input.resolve():
        raise SystemExit("refusing to overwrite the input home; choose a different --output")

    home = json.loads(args.input.read_text(encoding="utf-8"))
    joint_names = home["joint_names"]

    def apply(side: str, delta_deg: float) -> None:
        if abs(delta_deg) < 1e-9:
            print(f"{side}: no change")
            return
        delta = math.radians(delta_deg)
        gname = f"{side}_arm_6"
        gidx = joint_names.index(gname)
        arm = home["arms"][side]
        old = float(arm["positions_rad"][WRIST_ROLL_LOCAL_INDEX])
        new = old + delta
        arm["positions_rad"][WRIST_ROLL_LOCAL_INDEX] = new
        if "positions_deg" in arm:
            arm["positions_deg"][WRIST_ROLL_LOCAL_INDEX] = math.degrees(new)
        if "target_positions_rad" in arm and len(arm["target_positions_rad"]) > WRIST_ROLL_LOCAL_INDEX:
            arm["target_positions_rad"][WRIST_ROLL_LOCAL_INDEX] = new
        aj = home["all_joints"]
        aj["positions_rad"][gidx] = new
        if "positions_deg" in aj:
            aj["positions_deg"][gidx] = math.degrees(new)
        print(f"{gname}: {old:+.4f} -> {new:+.4f} rad  ({math.degrees(old):+.1f} -> {math.degrees(new):+.1f} deg)")

    apply("right", args.right_wrist_roll_deg)
    apply("left", args.left_wrist_roll_deg)

    home.setdefault("notes", []).append(
        f"wrist-roll edited: right {args.right_wrist_roll_deg:+g}deg, left {args.left_wrist_roll_deg:+g}deg. "
        "FK transforms (T_base_link) are STALE -- re-capture with capture_home_state.py after commanding."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(home, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
