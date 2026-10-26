from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _request_json(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=float(timeout_s)) as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"ok": False, "error": body}
        return int(exc.code), payload


def _motor_position(status: dict[str, Any], arm: str) -> float | None:
    target_id = 1 if arm == "left" else 0
    for state in status.get("motor_states", []):
        if int(state.get("id", -1)) == target_id:
            return float(state["position"])
    return None


def _motor_position_from_payload(payload: dict[str, Any], arm: str) -> float | None:
    target_id = 1 if arm == "left" else 0
    for state in payload.get("motor_states", []):
        if int(state.get("id", -1)) == target_id and state.get("position") is not None:
            return float(state["position"])
    return None


def _parse_positions(text: str | None) -> list[float]:
    if text is None:
        return []
    values: list[float] = []
    for part in text.replace(",", " ").split():
        values.append(float(part))
    return values


def _closing_direction(start: float, close_position: float) -> float:
    return 1.0 if float(close_position) >= float(start) else -1.0


def _moved_far_enough_toward_target(
    *,
    after: float | None,
    target: float,
    direction: float,
    tolerance: float,
) -> bool:
    if after is None:
        return False
    return direction * (float(target) - float(after)) <= float(tolerance)


def _target_not_reached_due_to_resistance(
    *,
    after: float | None,
    target: float,
    direction: float,
    tolerance: float,
) -> bool:
    if after is None:
        return False
    return direction * (float(target) - float(after)) > float(tolerance)


def _closed_past_contact_gate(
    *,
    after: float | None,
    direction: float,
    min_contact_position: float | None,
    tolerance: float,
) -> bool:
    if min_contact_position is None:
        return True
    if after is None:
        return False
    return direction * (float(after) - float(min_contact_position)) >= -float(tolerance)


def _accepted_position(
    *,
    action: str,
    after: float | None,
    target: float,
    open_threshold: float,
    closed_threshold: float,
    position_tolerance: float,
) -> bool:
    return (
        after is not None
        and (
            (action == "position" and abs(after - target) <= float(position_tolerance))
            or (action == "close" and after >= float(closed_threshold))
            or (action == "open" and after <= float(open_threshold))
        )
    )


def _wait_for_accepted_position(
    *,
    base_url: str,
    arm: str,
    action: str,
    target: float,
    open_threshold: float,
    closed_threshold: float,
    position_tolerance: float,
    timeout_s: float,
    poll_s: float,
) -> float | None:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last_position: float | None = None
    while time.monotonic() <= deadline:
        last_position = _motor_position(_status(base_url, timeout_s=max(1.0, float(poll_s) + 0.5)), arm)
        if _accepted_position(
            action=action,
            after=last_position,
            target=target,
            open_threshold=open_threshold,
            closed_threshold=closed_threshold,
            position_tolerance=position_tolerance,
        ):
            return last_position
        time.sleep(max(0.05, float(poll_s)))
    return last_position


def _generate_grasp_targets(
    *,
    start: float,
    close_position: float,
    grasp_steps: list[float],
    step_delta: float,
) -> list[float]:
    direction = _closing_direction(start, close_position)
    if grasp_steps:
        filtered = [step for step in grasp_steps if direction * (step - start) > 1e-6]
        if not filtered or abs(filtered[-1] - close_position) > 1e-6:
            filtered.append(float(close_position))
        return filtered
    if step_delta <= 0.0:
        raise ValueError("--grasp-step-delta must be positive")
    targets: list[float] = []
    target = float(start)
    while direction * (float(close_position) - target) > 1e-6:
        target = target + direction * float(step_delta)
        if direction * (target - float(close_position)) > 0.0:
            target = float(close_position)
        targets.append(float(target))
    return targets


def _status(base_url: str, timeout_s: float) -> dict[str, Any]:
    code, payload = _request_json(f"{base_url}/status", timeout_s)
    if code != 200 or not payload.get("ok", False):
        raise RuntimeError(f"status failed: code={code} payload={payload}")
    return payload


def _jog(
    *,
    base_url: str,
    arm: str,
    position: float,
    current_limit: float,
    hold_s: float,
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "arm": arm,
            "position": float(position),
            "current_limit": float(current_limit),
            "hold_s": float(hold_s),
        }
    )
    return _request_json(f"{base_url}/jog?{query}", timeout_s)


def _grasp_until_contact(
    *,
    base_url: str,
    arm: str,
    start_position: float,
    close_position: float,
    current_limit: float,
    hold_s: float,
    timeout_s: float,
    grasp_steps: list[float],
    grasp_step_delta: float,
    contact_position_tolerance: float,
    require_contact: bool,
    min_contact_position: float | None = None,
) -> dict[str, Any]:
    direction = _closing_direction(start_position, close_position)
    targets = _generate_grasp_targets(
        start=start_position,
        close_position=close_position,
        grasp_steps=grasp_steps,
        step_delta=grasp_step_delta,
    )
    steps: list[dict[str, Any]] = []
    before = float(start_position)
    for target in targets:
        code, payload = _jog(
            base_url=base_url,
            arm=arm,
            position=float(target),
            current_limit=float(current_limit),
            hold_s=float(hold_s),
            timeout_s=float(timeout_s),
        )
        after = _motor_position_from_payload(payload, arm)
        if after is None:
            try:
                after = _motor_position(_status(base_url, timeout_s), arm)
            except Exception:
                after = None
        reached = _moved_far_enough_toward_target(
            after=after,
            target=float(target),
            direction=direction,
            tolerance=float(contact_position_tolerance),
        )
        contact = _target_not_reached_due_to_resistance(
            after=after,
            target=float(target),
            direction=direction,
            tolerance=float(contact_position_tolerance),
        )
        contact_position_allowed = _closed_past_contact_gate(
            after=after,
            direction=direction,
            min_contact_position=min_contact_position,
            tolerance=float(contact_position_tolerance),
        )
        step = {
            "target": float(target),
            "position_before": float(before),
            "position_after": after,
            "bridge_code": int(code),
            "bridge_ok": bool(payload.get("ok", False)),
            "target_reached": bool(reached),
            "contact_detected": bool(contact),
            "contact_position_allowed": bool(contact_position_allowed),
        }
        steps.append(step)
        if contact:
            if not contact_position_allowed:
                return {
                    "ok": False,
                    "action": "grasp",
                    "arm": arm,
                    "reason": "contact_before_min_contact_position",
                    "contact_detected": True,
                    "final_position": after,
                    "contact_target": float(target),
                    "min_contact_position": None if min_contact_position is None else float(min_contact_position),
                    "contact_position_tolerance": float(contact_position_tolerance),
                    "steps": steps,
                }
            return {
                "ok": True,
                "action": "grasp",
                "arm": arm,
                "contact_detected": True,
                "final_position": after,
                "contact_target": float(target),
                "min_contact_position": None if min_contact_position is None else float(min_contact_position),
                "contact_position_tolerance": float(contact_position_tolerance),
                "steps": steps,
            }
        if code != 200 or not payload.get("ok", False):
            return {
                "ok": False,
                "action": "grasp",
                "arm": arm,
                "reason": "bridge_command_failed_before_contact",
                "bridge_code": int(code),
                "bridge_payload": payload,
                "final_position": after,
                "steps": steps,
            }
        if after is None:
            return {
                "ok": False,
                "action": "grasp",
                "arm": arm,
                "reason": "missing_motor_position_after_grasp_step",
                "steps": steps,
            }
        before = float(after)

    result = {
        "ok": not bool(require_contact),
        "action": "grasp",
        "arm": arm,
        "contact_detected": False,
        "reason": "no_contact_before_close_limit",
        "final_position": before,
        "close_position": float(close_position),
        "min_contact_position": None if min_contact_position is None else float(min_contact_position),
        "contact_position_tolerance": float(contact_position_tolerance),
        "steps": steps,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust HTTP client for the RB-Y1 gripper bridge.")
    parser.add_argument("--url", default="http://192.0.2.20:8765")
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--action", choices=["status", "open", "close", "position", "grasp"], required=True)
    parser.add_argument("--position", type=float, help="Absolute gripper encoder target for --action position.")
    parser.add_argument("--open-position", type=float, default=-8.0)
    parser.add_argument("--close-position", type=float, default=4.8)
    parser.add_argument("--open-current-limit", type=float, default=3.0)
    parser.add_argument("--close-current-limit", type=float, default=2.5)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--post-command-wait-s", type=float, default=5.0)
    parser.add_argument("--post-command-poll-s", type=float, default=0.20)
    parser.add_argument("--closed-threshold", type=float, default=4.45)
    parser.add_argument("--open-threshold", type=float, default=-4.35)
    parser.add_argument(
        "--verify-after-command",
        action="store_true",
        help="Re-read/inspect motor position after a successful command and fail if the requested state was not reached.",
    )
    parser.add_argument("--position-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--grasp-steps",
        help="Optional whitespace/comma separated close targets for --action grasp. Stops early when contact is detected.",
    )
    parser.add_argument("--grasp-step-delta", type=float, default=0.6)
    parser.add_argument(
        "--contact-position-tolerance",
        type=float,
        default=0.25,
        help=(
            "For --action grasp, treat a low-current close step as contact when "
            "the motor stops this far before the commanded target."
        ),
    )
    parser.add_argument(
        "--min-contact-position",
        type=float,
        help=(
            "For --action grasp, reject a resistance/contact event that occurs before this encoder "
            "position along the closing direction. This prevents accepting early mechanism friction "
            "as a tomato grasp."
        ),
    )
    parser.add_argument(
        "--require-contact",
        action="store_true",
        help="For --action grasp, fail if no contact/resistance is observed before --close-position.",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    if args.action == "status":
        payload = _status(base_url, args.timeout_s)
        print(json.dumps(payload, sort_keys=True))
        return

    status = _status(base_url, args.timeout_s)
    before = _motor_position(status, args.arm)
    if before is None:
        raise RuntimeError(f"could not read {args.arm} gripper motor position from status")

    if args.action == "grasp":
        result = _grasp_until_contact(
            base_url=base_url,
            arm=args.arm,
            start_position=float(before),
            close_position=float(args.close_position),
            current_limit=float(args.close_current_limit),
            hold_s=float(args.hold_s),
            timeout_s=float(args.timeout_s),
            grasp_steps=_parse_positions(args.grasp_steps),
            grasp_step_delta=float(args.grasp_step_delta),
            contact_position_tolerance=float(args.contact_position_tolerance),
            require_contact=bool(args.require_contact),
            min_contact_position=args.min_contact_position,
        )
        print(json.dumps(result, sort_keys=True))
        if not result.get("ok", False):
            sys.exit(1)
        return

    if args.action == "position":
        if args.position is None:
            raise ValueError("--position is required for --action position")
        target = float(args.position)
        current_limit = float(args.close_current_limit)
    elif args.action == "close":
        target = float(args.close_position)
        current_limit = float(args.close_current_limit)
        if before >= float(args.closed_threshold):
            print(
                json.dumps(
                    {
                        "ok": True,
                        "action": "close",
                        "arm": args.arm,
                        "skipped": True,
                        "reason": "already_closed",
                        "position": before,
                        "threshold": float(args.closed_threshold),
                    },
                    sort_keys=True,
                )
            )
            return
    else:
        target = float(args.open_position)
        current_limit = float(args.open_current_limit)
        if before <= float(args.open_threshold):
            print(
                json.dumps(
                    {
                        "ok": True,
                        "action": "open",
                        "arm": args.arm,
                        "skipped": True,
                        "reason": "already_open",
                        "position": before,
                        "threshold": float(args.open_threshold),
                    },
                    sort_keys=True,
                )
            )
            return

    code, payload = _jog(
        base_url=base_url,
        arm=args.arm,
        position=target,
        current_limit=current_limit,
        hold_s=float(args.hold_s),
        timeout_s=float(args.timeout_s),
    )
    if code == 200 and payload.get("ok", False):
        if bool(args.verify_after_command):
            after = _motor_position_from_payload(payload, args.arm)
            if after is None:
                after = _motor_position(_status(base_url, args.timeout_s), args.arm)
            accepted = _accepted_position(
                action=args.action,
                after=after,
                target=target,
                open_threshold=float(args.open_threshold),
                closed_threshold=float(args.closed_threshold),
                position_tolerance=float(args.position_tolerance),
            )
            if not accepted and float(args.post_command_wait_s) > 0.0:
                after = _wait_for_accepted_position(
                    base_url=base_url,
                    arm=args.arm,
                    action=args.action,
                    target=target,
                    open_threshold=float(args.open_threshold),
                    closed_threshold=float(args.closed_threshold),
                    position_tolerance=float(args.position_tolerance),
                    timeout_s=float(args.post_command_wait_s),
                    poll_s=float(args.post_command_poll_s),
                )
                accepted = _accepted_position(
                    action=args.action,
                    after=after,
                    target=target,
                    open_threshold=float(args.open_threshold),
                    closed_threshold=float(args.closed_threshold),
                    position_tolerance=float(args.position_tolerance),
                )
            if not accepted:
                result = {
                    "ok": False,
                    "action": args.action,
                    "arm": args.arm,
                    "reason": "post_command_position_not_reached",
                    "target": target,
                    "position_after": after,
                    "open_threshold": float(args.open_threshold),
                    "closed_threshold": float(args.closed_threshold),
                    "position_tolerance": float(args.position_tolerance),
                    "bridge_payload": payload,
                }
                print(json.dumps(result, sort_keys=True))
                sys.exit(1)
        print(json.dumps(payload, sort_keys=True))
        return

    # The bridge can return 500 when the motor is already at a mechanical stop.
    # Re-read status and accept only if the requested open/close state is reached.
    after_status = _status(base_url, args.timeout_s)
    after = _motor_position(after_status, args.arm)
    accepted = (
        after is not None
        and (
            (args.action == "position" and abs(after - target) <= 0.35)
            or (args.action == "close" and after >= float(args.closed_threshold))
            or (args.action == "open" and after <= float(args.open_threshold))
        )
    )
    result = {
        "ok": bool(accepted),
        "action": args.action,
        "arm": args.arm,
        "bridge_code": code,
        "bridge_payload": payload,
        "position_after": after,
    }
    print(json.dumps(result, sort_keys=True))
    if not accepted:
        sys.exit(1)


if __name__ == "__main__":
    main()
