from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby


DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_HOME = Path(__file__).with_name("homestate.json")
DEFAULT_LEFT_CLOSE_POSITION = 10.945
DEFAULT_RIGHT_CLOSE_POSITION = 4.8
RESTORE_SCOPES = ("all", "body_head", "left_arm", "right_arm", "arms")
RIGHT_FROM_LEFT_MIRROR_SIGNS = np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)


def _positions(home: dict[str, Any], group: str, side: str | None = None) -> np.ndarray:
    if side is None:
        values = home[group]["positions_rad"]
    else:
        values = home[group][side]["positions_rad"]
    return np.asarray(values, dtype=float)


def mirror_left_arm_to_right(home: dict[str, Any]) -> np.ndarray:
    """Return the RB-Y1 right-arm joint posture mirrored from the saved left arm."""
    left = _positions(home, "arms", "left").reshape(-1)
    if left.shape != RIGHT_FROM_LEFT_MIRROR_SIGNS.shape:
        raise ValueError(f"expected 7 left-arm joints, got shape {left.shape}")
    if not np.all(np.isfinite(left)):
        raise ValueError("left-arm home posture contains non-finite values")
    return left * RIGHT_FROM_LEFT_MIRROR_SIGNS


def _connect_for_motion(address: str, model: str, power: str, servo: str) -> Any:
    robot = rby.create_robot(address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")
    if not robot.is_power_on(power) and not robot.power_on(power):
        raise RuntimeError(f"failed to power on {power}")
    if not robot.is_servo_on(servo) and not robot.servo_on(servo):
        raise RuntimeError(f"failed to servo on {servo}")

    state = robot.get_control_manager_state().state
    if state in (rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault):
        if not robot.reset_fault_control_manager():
            raise RuntimeError("failed to reset control manager fault")
    if not robot.enable_control_manager():
        raise RuntimeError("failed to enable control manager")
    return robot


def _body_head_command(
    home: dict[str, Any],
    minimum_time: float,
    scope: str = "all",
    *,
    mirror_left_to_right: bool = False,
) -> Any:
    if scope not in RESTORE_SCOPES:
        raise ValueError(f"unsupported restore scope: {scope}")

    body = rby.BodyComponentBasedCommandBuilder()
    if scope in ("all", "body_head"):
        body = body.set_torso_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(float(minimum_time))
            .set_position(_positions(home, "torso"))
        )
    if scope in ("all", "right_arm", "arms"):
        right_positions = (
            mirror_left_arm_to_right(home)
            if mirror_left_to_right
            else _positions(home, "arms", "right")
        )
        body = body.set_right_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(float(minimum_time))
            .set_position(right_positions)
        )
    if scope in ("all", "left_arm", "arms"):
        body = body.set_left_arm_command(
            rby.JointPositionCommandBuilder()
            .set_minimum_time(float(minimum_time))
            .set_position(_positions(home, "arms", "left"))
        )

    command = rby.ComponentBasedCommandBuilder().set_body_command(body)
    if scope in ("all", "body_head"):
        head = (
            rby.JointPositionCommandBuilder()
            .set_minimum_time(float(minimum_time))
            .set_position(_positions(home, "head"))
        )
        command = command.set_head_command(head)
    return rby.RobotCommandBuilder().set_command(command)


def _gripper_arms_for_scope(scope: str) -> tuple[str, ...]:
    if scope == "left_arm":
        return ("left",)
    if scope == "right_arm":
        return ("right",)
    if scope in ("all", "arms"):
        return ("left", "right")
    return ()


def _restore_gripper(url: str, arm: str, position: float, current_limit: float, hold_s: float, timeout_s: float) -> None:
    query = urllib.parse.urlencode(
        {
            "arm": arm,
            "position": float(position),
            "current_limit": float(current_limit),
            "hold_s": float(hold_s),
        }
    )
    with urllib.request.urlopen(f"{url.rstrip('/')}/jog?{query}", timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok", False):
        raise RuntimeError(f"{arm} gripper restore failed: {payload}")


def _close_position_for_arm(arm: str, left_close_position: float, right_close_position: float) -> float:
    if arm == "left":
        return float(left_close_position)
    if arm == "right":
        return float(right_close_position)
    raise ValueError(f"unsupported gripper arm: {arm}")


def restore_home_state(
    home_path: Path,
    address: str = DEFAULT_ADDRESS,
    model: str = "a",
    power: str = ".*",
    servo: str = ".*",
    minimum_time: float = 6.0,
    timeout_s: float = 15.0,
    scope: str = "all",
    restore_grippers: bool = False,
    gripper_restore_mode: str = "saved",
    gripper_current_limit: float = 2.0,
    gripper_hold_s: float = 2.0,
    left_close_position: float = DEFAULT_LEFT_CLOSE_POSITION,
    right_close_position: float = DEFAULT_RIGHT_CLOSE_POSITION,
    mirror_left_to_right: bool = False,
) -> None:
    home = json.loads(home_path.read_text(encoding="utf-8"))
    robot = _connect_for_motion(address, model, power, servo)
    feedback = robot.send_command(
        _body_head_command(
            home,
            minimum_time,
            scope,
            mirror_left_to_right=mirror_left_to_right,
        ),
        int(max(float(timeout_s), float(minimum_time) + 5.0)),
    ).get()
    if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
        raise RuntimeError(f"home restore failed: {feedback.finish_code}")

    if restore_grippers:
        if gripper_restore_mode not in {"saved", "close"}:
            raise ValueError(f"unsupported gripper restore mode: {gripper_restore_mode}")
        grippers = home.get("grippers", {})
        url = str(grippers.get("url", "")).rstrip("/")
        if not url:
            raise RuntimeError("home state does not contain gripper url")
        for arm in _gripper_arms_for_scope(scope):
            if gripper_restore_mode == "close":
                _restore_gripper(
                    url,
                    arm,
                    _close_position_for_arm(arm, left_close_position, right_close_position),
                    gripper_current_limit,
                    gripper_hold_s,
                    timeout_s,
                )
                continue
            arm_state = grippers.get(arm)
            if arm_state is None or arm_state.get("position") is None:
                raise RuntimeError(f"home state does not contain {arm} gripper position")
            _restore_gripper(
                url,
                arm,
                float(arm_state["position"]),
                gripper_current_limit,
                gripper_hold_s,
                timeout_s,
            )


def _print_dry_run(
    home: dict[str, Any],
    home_path: Path,
    scope: str,
    *,
    mirror_left_to_right: bool = False,
) -> None:
    print(f"home: {home_path}")
    print(f"schema: {home.get('schema')} version={home.get('version')}")
    print(f"scope: {scope}")
    if scope in ("all", "body_head"):
        print(f"torso_rad: {home['torso']['positions_rad']}")
        print(f"head_rad: {home['head']['positions_rad']}")
    if scope in ("all", "right_arm", "arms"):
        if mirror_left_to_right:
            print(f"right_arm_rad_mirrored_from_left: {mirror_left_arm_to_right(home).tolist()}")
        else:
            print(f"right_arm_rad: {home['arms']['right']['positions_rad']}")
    if scope in ("all", "left_arm", "arms"):
        print(f"left_arm_rad: {home['arms']['left']['positions_rad']}")
    grippers = home.get("grippers", {})
    if scope in ("all", "left_arm", "arms"):
        print(f"left_gripper: {grippers.get('left')}")
    if scope in ("all", "right_arm", "arms"):
        print(f"right_gripper: {grippers.get('right')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore RB-Y1 to the captured robot/utils/homestate.json posture.")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--power", default=".*")
    parser.add_argument("--servo", default=".*")
    parser.add_argument("--minimum-time", type=float, default=6.0)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--scope", choices=RESTORE_SCOPES, default="all")
    parser.add_argument(
        "--mirror-left-to-right",
        action="store_true",
        help="For scopes containing the right arm, command the kinematic mirror of the saved left-arm posture.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually command the robot. Without this, only prints targets.")
    parser.add_argument("--restore-grippers", action="store_true", help="Also restore saved gripper encoder positions.")
    parser.add_argument(
        "--gripper-restore-mode",
        choices=("saved", "close"),
        default="saved",
        help="With --restore-grippers, restore saved gripper positions or force-close the grippers after homing.",
    )
    parser.add_argument("--close-grippers", action="store_true", help="Alias for --restore-grippers --gripper-restore-mode close.")
    parser.add_argument("--gripper-current-limit", type=float, default=2.0)
    parser.add_argument("--gripper-hold-s", type=float, default=2.0)
    parser.add_argument("--left-close-position", type=float, default=DEFAULT_LEFT_CLOSE_POSITION)
    parser.add_argument("--right-close-position", type=float, default=DEFAULT_RIGHT_CLOSE_POSITION)
    args = parser.parse_args()
    if args.close_grippers:
        args.restore_grippers = True
        args.gripper_restore_mode = "close"
    if args.mirror_left_to_right and args.scope not in ("all", "right_arm", "arms"):
        parser.error("--mirror-left-to-right requires a scope containing the right arm")

    home = json.loads(args.home.read_text(encoding="utf-8"))
    if not args.execute:
        _print_dry_run(
            home,
            args.home,
            args.scope,
            mirror_left_to_right=args.mirror_left_to_right,
        )
        print("dry_run: pass --execute to command the robot")
        return 0

    restore_home_state(
        home_path=args.home,
        address=args.address,
        model=args.model,
        power=args.power,
        servo=args.servo,
        minimum_time=args.minimum_time,
        timeout_s=args.timeout_s,
        scope=args.scope,
        restore_grippers=args.restore_grippers,
        gripper_restore_mode=args.gripper_restore_mode,
        gripper_current_limit=args.gripper_current_limit,
        gripper_hold_s=args.gripper_hold_s,
        left_close_position=args.left_close_position,
        right_close_position=args.right_close_position,
        mirror_left_to_right=args.mirror_left_to_right,
    )
    print("home restore complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
