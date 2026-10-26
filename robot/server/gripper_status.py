from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence


DEFAULT_GRIPPER_URL = "http://192.0.2.21:8765"
MAX_POLL_HZ = 10.0
_ARM_BY_ID = {0: "right", 1: "left"}
_OPEN_BOUND_BY_ARM = {"right": "max", "left": "min"}
_ENCODER_TURN_RAD = 2.0 * math.pi
_REBASE_ENDPOINT_TOLERANCE_RAD = 0.5


class GripperStatusError(ValueError):
    pass


@dataclass(frozen=True)
class GripperReading:
    arm: str
    motor_id: int
    position: float
    current_a: float
    openness: float
    openness_unclamped: float
    min_q: float
    max_q: float
    within_calibration: bool


@dataclass(frozen=True)
class GripperCommandTargets:
    arm: str
    motor_id: int
    open_position: float
    close_position: float
    encoder_span: float
    calibration_schema: str
    raw_open_position: float
    raw_close_position: float
    encoder_turn_offset: int
    calibration_endpoint_distance: float | None


def _number_list(value: Any, *, key: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GripperStatusError(f"{key} must be a numeric sequence")
    result = [float(item) for item in value]
    if len(result) < 2 or not all(math.isfinite(item) for item in result):
        raise GripperStatusError(f"{key} must contain two finite motor bounds")
    return result


def calibration_bounds(payload: dict[str, Any]) -> tuple[list[float], list[float], str]:
    has_teleop = "gripper_min_q" in payload or "gripper_max_q" in payload
    if has_teleop:
        if "gripper_min_q" not in payload or "gripper_max_q" not in payload:
            raise GripperStatusError("teleop status must provide both gripper_min_q and gripper_max_q")
        lo_key, hi_key, schema = "gripper_min_q", "gripper_max_q", "teleop"
    else:
        if "min_q" not in payload or "max_q" not in payload:
            raise GripperStatusError(
                "status is missing gripper calibration bounds: expected "
                "gripper_min_q/gripper_max_q or min_q/max_q"
            )
        lo_key, hi_key, schema = "min_q", "max_q", "bridge"

    lo = _number_list(payload[lo_key], key=lo_key)
    hi = _number_list(payload[hi_key], key=hi_key)
    if len(lo) != len(hi):
        raise GripperStatusError("gripper calibration bound arrays have different lengths")
    for motor_id, (lower, upper) in enumerate(zip(lo, hi)):
        if upper <= lower:
            raise GripperStatusError(
                f"motor {motor_id} calibration is invalid: max_q={upper} <= min_q={lower}"
            )
    return lo, hi, schema


def _motor_position(payload: dict[str, Any], motor_id: int) -> float | None:
    states = payload.get("motor_states")
    if not isinstance(states, list):
        return None
    positions = [
        float(state["position"])
        for state in states
        if isinstance(state, dict) and int(state.get("id", -1)) == motor_id
    ]
    if not positions:
        return None
    if len(positions) != 1 or not math.isfinite(positions[0]):
        raise GripperStatusError(f"motor {motor_id} must have one finite position")
    return positions[0]


def _encoder_turn_offset(
    lower: float,
    upper: float,
    position: float | None,
) -> tuple[int, float | None]:
    """Rebase a multi-turn calibration after the encoder origin wraps on startup."""
    if position is None:
        return 0, None

    raw_endpoint_distance = min(abs(position - lower), abs(position - upper))
    approximate_turn = int(round((position - lower) / _ENCODER_TURN_RAD))
    candidates = range(approximate_turn - 2, approximate_turn + 3)
    best_turn = min(
        candidates,
        key=lambda turn: (
            min(
                abs(position - (lower + turn * _ENCODER_TURN_RAD)),
                abs(position - (upper + turn * _ENCODER_TURN_RAD)),
            ),
            abs(turn),
        ),
    )
    shifted_distance = min(
        abs(position - (lower + best_turn * _ENCODER_TURN_RAD)),
        abs(position - (upper + best_turn * _ENCODER_TURN_RAD)),
    )
    if (
        best_turn != 0
        and shifted_distance <= _REBASE_ENDPOINT_TOLERANCE_RAD
        and shifted_distance + 1e-9 < raw_endpoint_distance
    ):
        return best_turn, shifted_distance
    return 0, raw_endpoint_distance


def command_targets(payload: dict[str, Any], *, arm: str) -> GripperCommandTargets:
    normalized_arm = str(arm).lower()
    motor_by_arm = {name: motor_id for motor_id, name in _ARM_BY_ID.items()}
    if normalized_arm not in motor_by_arm:
        raise GripperStatusError(f"unknown gripper arm: {arm!r}")
    lo, hi, schema = calibration_bounds(payload)
    motor_id = motor_by_arm[normalized_arm]
    if motor_id >= len(lo):
        raise GripperStatusError(f"calibration bounds are missing motor id {motor_id}")
    raw_lower = float(lo[motor_id])
    raw_upper = float(hi[motor_id])
    current_position = _motor_position(payload, motor_id)
    turn_offset, endpoint_distance = _encoder_turn_offset(
        raw_lower,
        raw_upper,
        current_position,
    )
    position_offset = turn_offset * _ENCODER_TURN_RAD
    lower = raw_lower + position_offset
    upper = raw_upper + position_offset
    if _OPEN_BOUND_BY_ARM[normalized_arm] == "min":
        raw_open_position = raw_lower
        raw_close_position = raw_upper
        open_position = lower
        close_position = upper
    else:
        raw_open_position = raw_upper
        raw_close_position = raw_lower
        open_position = upper
        close_position = lower
    return GripperCommandTargets(
        arm=normalized_arm,
        motor_id=motor_id,
        # min_q/max_q are numeric travel bounds, not semantic aperture states.
        # The mirrored grippers have opposite encoder directions.
        open_position=open_position,
        close_position=close_position,
        encoder_span=float(hi[motor_id] - lo[motor_id]),
        calibration_schema=schema,
        raw_open_position=raw_open_position,
        raw_close_position=raw_close_position,
        encoder_turn_offset=turn_offset,
        calibration_endpoint_distance=endpoint_distance,
    )


def normalized_readings(payload: dict[str, Any]) -> dict[str, GripperReading]:
    if not payload.get("ok", False):
        raise GripperStatusError(f"gripper status is not ok: {payload.get('error', payload)!r}")
    lo, hi, _schema = calibration_bounds(payload)
    states = payload.get("motor_states")
    if not isinstance(states, list):
        raise GripperStatusError("motor_states must be a list")

    readings: dict[str, GripperReading] = {}
    for state in states:
        if not isinstance(state, dict):
            raise GripperStatusError("each motor state must be an object")
        motor_id = int(state.get("id", -1))
        if motor_id not in _ARM_BY_ID:
            continue
        if motor_id >= len(lo):
            raise GripperStatusError(f"calibration bounds are missing motor id {motor_id}")
        position = float(state["position"])
        current = float(state.get("current", 0.0))
        if not math.isfinite(position) or not math.isfinite(current):
            raise GripperStatusError(f"motor {motor_id} has a non-finite state")
        arm = _ARM_BY_ID[motor_id]
        targets = command_targets(payload, arm=arm)
        raw_openness = (position - targets.close_position) / (
            targets.open_position - targets.close_position
        )
        if arm in readings:
            raise GripperStatusError(f"duplicate motor state for {arm} gripper")
        readings[arm] = GripperReading(
            arm=arm,
            motor_id=motor_id,
            position=position,
            current_a=current,
            openness=min(1.0, max(0.0, raw_openness)),
            openness_unclamped=raw_openness,
            min_q=lo[motor_id],
            max_q=hi[motor_id],
            within_calibration=0.0 <= raw_openness <= 1.0,
        )
    if not readings:
        raise GripperStatusError("status contains no recognized gripper motor states")
    return readings


def normalized_status(payload: dict[str, Any]) -> dict[str, Any]:
    _lo, _hi, schema = calibration_bounds(payload)
    readings = normalized_readings(payload)
    result: dict[str, Any] = {
        "ok": True,
        "calibration_schema": schema,
        "grippers": {arm: asdict(reading) for arm, reading in readings.items()},
    }
    if "tcp" in payload:
        result["tcp"] = payload["tcp"]
    return result


def fetch_status(base_url: str = DEFAULT_GRIPPER_URL, *, timeout_s: float = 1.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/status", timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise GripperStatusError("status response must be a JSON object")
    return payload


def poll_status(
    base_url: str = DEFAULT_GRIPPER_URL,
    *,
    hz: float = MAX_POLL_HZ,
    timeout_s: float = 1.0,
) -> Iterator[dict[str, Any]]:
    rate = float(hz)
    if not 0.0 < rate <= MAX_POLL_HZ:
        raise ValueError(f"poll rate must be in (0, {MAX_POLL_HZ:g}] Hz")
    period_s = 1.0 / rate
    while True:
        started = time.monotonic()
        yield normalized_status(fetch_status(base_url, timeout_s=timeout_s))
        remaining = period_s - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


def _format_human(payload: dict[str, Any]) -> str:
    parts = []
    for arm in ("right", "left"):
        reading = payload["grippers"].get(arm)
        if reading is None:
            continue
        parts.append(
            f"{arm} open={reading['openness']:.3f} "
            f"pos={reading['position']:.3f} cur={reading['current_a']:.3f}A"
        )
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only normalized RB-Y1 gripper status listener.")
    parser.add_argument("--url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument("--hz", type=float, default=MAX_POLL_HZ)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print normalized JSON instead of a compact status line.")
    args = parser.parse_args()
    if not 0.0 < args.hz <= MAX_POLL_HZ:
        parser.error(f"--hz must be in (0, {MAX_POLL_HZ:g}]")
    if args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")

    while True:
        started = time.monotonic()
        try:
            payload = normalized_status(fetch_status(args.url, timeout_s=args.timeout_s))
            print(json.dumps(payload, sort_keys=True) if args.json else _format_human(payload), flush=True)
        except Exception as exc:
            print(f"gripper status error: {exc}", file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        remaining = (1.0 / args.hz) - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
