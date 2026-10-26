from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import rby1_sdk as rby


def _arms(value: str) -> list[str]:
    if value == "both":
        return ["left", "right"]
    return [value]


def _arm_to_ids(arm: str) -> list[int]:
    key = arm.lower().strip()
    if key == "right":
        return [0]
    if key == "left":
        return [1]
    if key == "both":
        return [0, 1]
    if key in {"0", "id0"}:
        return [0]
    if key in {"1", "id1"}:
        return [1]
    raise ValueError(f"unsupported arm {arm!r}; expected left, right, both, 0, or 1")


def _float_arg(params: dict[str, list[str]], name: str, default: float) -> float:
    values = params.get(name)
    if not values:
        return float(default)
    return float(values[-1])


def _optional_float_arg(params: dict[str, list[str]], name: str) -> float | None:
    values = params.get(name)
    if not values:
        return None
    return float(values[-1])


def _enable_tool_flange_power(
    *,
    robot_address: str,
    model: str,
    flange_arm: str,
    flange_voltage: int,
    power: str,
    settle_s: float,
) -> None:
    robot = rby.create_robot(robot_address, model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {robot_address}")
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            raise RuntimeError(f"failed to power on {power}")
        logging.info("Powered on %s", power)
    for arm in _arms(flange_arm):
        if not robot.set_tool_flange_output_voltage(arm, int(flange_voltage)):
            raise RuntimeError(f"failed to set {arm} tool flange output voltage to {flange_voltage}V")
        logging.info("Set %s tool flange output voltage to %dV", arm, flange_voltage)
    if float(settle_s) > 0.0:
        logging.info("Waiting %.2fs for gripper bus to settle after flange power", settle_s)
        time.sleep(float(settle_s))


class GripperBridge:
    def __init__(
        self,
        *,
        device_name: str,
        baud_rate: int,
        torque_limit: float,
        command_period_s: float,
        robot_address: str | None,
        flange_arm: str,
        flange_voltage: int | None,
        ping_retries: int,
        ping_retry_s: float,
    ) -> None:
        self._device_name = device_name
        self._baud_rate = int(baud_rate)
        self._torque_limit = float(torque_limit)
        self._command_period_s = float(command_period_s)
        self._robot_address = robot_address
        self._flange_arm = flange_arm
        self._flange_voltage = flange_voltage
        self._ping_retries = max(1, int(ping_retries))
        self._ping_retry_s = max(0.0, float(ping_retry_s))
        self._bus = rby.DynamixelBus(device_name)

    def initialize(self) -> None:
        logging.info("Opening gripper bus %s", self._device_name)
        self._bus.open_port()
        self._bus.set_baud_rate(self._baud_rate)
        self._bus.set_torque_constant([1, 1])
        missing = [0, 1]
        for attempt in range(1, self._ping_retries + 1):
            missing = []
            for dev_id in [0, 1]:
                if not self._bus.ping(dev_id):
                    missing.append(dev_id)
            if not missing:
                break
            if attempt < self._ping_retries:
                logging.warning(
                    "Gripper ping attempt %d/%d missing ids=%s; retrying in %.2fs",
                    attempt,
                    self._ping_retries,
                    missing,
                    self._ping_retry_s,
                )
                time.sleep(self._ping_retry_s)
        if missing:
            raise RuntimeError(f"failed to detect gripper Dynamixel id(s): {missing}")
        self._bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self._bus.group_sync_write_operating_mode(
            [(dev_id, rby.DynamixelBus.CurrentControlMode) for dev_id in [0, 1]]
        )
        self._bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        logging.info("Gripper bridge ready")

    def _motor_states(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        states = self._bus.get_motor_states([0, 1])
        modes = dict(self._bus.group_fast_sync_read_operating_mode([0, 1], False) or [])
        if states is None:
            return payload
        for dev_id, state in states:
            payload.append(
                {
                    "id": int(dev_id),
                    "operating_mode": int(modes.get(int(dev_id), -1)),
                    "torque_enable": bool(state.torque_enable),
                    "position": float(state.position),
                    "velocity": float(state.velocity),
                    "current": float(state.current),
                    "torque": float(state.torque),
                    "temperature": int(state.temperature),
                }
            )
        return payload

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "device": self._device_name,
            "robot_address": self._robot_address,
            "flange_arm": self._flange_arm,
            "flange_voltage": self._flange_voltage,
            "torque_limit": self._torque_limit,
            "motor_states": self._motor_states(),
        }

    def command(
        self,
        *,
        arm: str,
        action: str,
        torque_limit: float | None,
        open_sign: float,
        close_sign: float,
        hold_s: float,
    ) -> dict[str, Any]:
        action_key = action.lower().strip()
        if action_key not in {"open", "close", "stop"}:
            raise ValueError(f"unsupported action {action!r}")
        dev_ids = _arm_to_ids(arm)
        limit = abs(float(self._torque_limit if torque_limit is None else torque_limit))
        sign = open_sign if action_key == "open" else close_sign
        if action_key == "stop":
            sign = 0.0
        torque = (1.0 if float(sign) >= 0.0 else -1.0) * limit
        if action_key == "stop":
            torque = 0.0
        deadline = time.perf_counter() + max(0.0, float(hold_s))
        logging.info("gripper %s arm=%s ids=%s torque=%.4f hold=%.3fs", action_key, arm, dev_ids, torque, hold_s)
        while True:
            self._bus.group_sync_write_send_torque([(dev_id, torque) for dev_id in dev_ids])
            if time.perf_counter() >= deadline:
                break
            time.sleep(max(0.02, self._command_period_s))
        if action_key in {"open", "stop"}:
            self._bus.group_sync_write_send_torque([(dev_id, 0.0) for dev_id in dev_ids])
        return {
            "ok": True,
            "action": action_key,
            "arm": arm,
            "device_ids": dev_ids,
            "torque": torque,
            "hold_s": float(hold_s),
            "motor_states": self._motor_states(),
        }

    def jog_position(
        self,
        *,
        arm: str,
        delta: float,
        position: float | None,
        current_limit: float,
        hold_s: float,
    ) -> dict[str, Any]:
        dev_ids = _arm_to_ids(arm)
        before = self._motor_states()
        encoders = dict(self._bus.group_fast_sync_read_encoder(dev_ids) or [])
        missing = [dev_id for dev_id in dev_ids if dev_id not in encoders]
        if missing:
            raise RuntimeError(f"failed to read encoder(s) for id(s): {missing}")

        self._bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in dev_ids])
        self._bus.group_sync_write_operating_mode(
            [(dev_id, rby.DynamixelBus.CurrentBasedPositionControlMode) for dev_id in dev_ids]
        )
        self._bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in dev_ids])
        self._bus.group_sync_write_send_torque([(dev_id, abs(float(current_limit))) for dev_id in dev_ids])

        targets = []
        for dev_id in dev_ids:
            target = float(position) if position is not None else float(encoders[dev_id]) + float(delta)
            targets.append((dev_id, target))
        deadline = time.perf_counter() + max(0.0, float(hold_s))
        logging.info(
            "gripper jog arm=%s ids=%s targets=%s current_limit=%.4f hold=%.3fs",
            arm,
            dev_ids,
            targets,
            current_limit,
            hold_s,
        )
        while True:
            self._bus.group_sync_write_send_position(targets)
            if time.perf_counter() >= deadline:
                break
            time.sleep(max(0.02, self._command_period_s))
        return {
            "ok": True,
            "action": "jog",
            "arm": arm,
            "device_ids": dev_ids,
            "delta": float(delta),
            "position": position,
            "targets": [{"id": int(dev_id), "position": float(target)} for dev_id, target in targets],
            "current_limit": float(current_limit),
            "hold_s": float(hold_s),
            "before": before,
            "motor_states": self._motor_states(),
        }


class Handler(BaseHTTPRequestHandler):
    bridge: GripperBridge

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        action = parsed.path.strip("/") or "status"
        params = parse_qs(parsed.query)
        try:
            if action == "status":
                payload = self.bridge.status()
            elif action in {"open", "close", "stop"}:
                payload = self.bridge.command(
                    arm=params.get("arm", ["left"])[-1],
                    action=action,
                    torque_limit=_float_arg(params, "torque_limit", self.bridge._torque_limit),
                    open_sign=_float_arg(params, "open_sign", 1.0),
                    close_sign=_float_arg(params, "close_sign", -1.0),
                    hold_s=_float_arg(params, "hold_s", 0.35),
                )
            elif action == "jog":
                payload = self.bridge.jog_position(
                    arm=params.get("arm", ["left"])[-1],
                    delta=_float_arg(params, "delta", 0.1),
                    position=_optional_float_arg(params, "position"),
                    current_limit=_float_arg(params, "current_limit", self.bridge._torque_limit),
                    hold_s=_float_arg(params, "hold_s", 0.5),
                )
            else:
                raise ValueError(f"unknown endpoint {parsed.path!r}")
            self._write_json(200, payload)
        except Exception as exc:
            logging.exception("request failed")
            self._write_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format % args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot-side HTTP bridge for RB-Y1 gripper torque commands.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default=rby.upc.GripperDeviceName)
    parser.add_argument("--baud-rate", type=int, default=2_000_000)
    parser.add_argument("--torque-limit", type=float, default=5.0)
    parser.add_argument("--command-period-s", type=float, default=0.05)
    parser.add_argument(
        "--robot-address",
        help="If set, connect to RB-Y1 and enable tool-flange voltage before opening the gripper bus.",
    )
    parser.add_argument("--model", default="a")
    parser.add_argument("--power", default="48v")
    parser.add_argument("--flange-arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--flange-voltage", type=int, choices=[0, 12, 24], default=12)
    parser.add_argument(
        "--startup-settle-s",
        type=float,
        default=3.0,
        help="Seconds to wait after enabling tool-flange power before opening the gripper bus.",
    )
    parser.add_argument(
        "--ping-retries",
        type=int,
        default=4,
        help="Number of Dynamixel ping attempts during bridge startup.",
    )
    parser.add_argument(
        "--ping-retry-s",
        type=float,
        default=1.0,
        help="Seconds to wait between startup ping retries.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.robot_address:
        _enable_tool_flange_power(
            robot_address=args.robot_address,
            model=args.model,
            flange_arm=args.flange_arm,
            flange_voltage=args.flange_voltage,
            power=args.power,
            settle_s=args.startup_settle_s,
        )
    bridge = GripperBridge(
        device_name=args.device,
        baud_rate=args.baud_rate,
        torque_limit=args.torque_limit,
        command_period_s=args.command_period_s,
        robot_address=args.robot_address,
        flange_arm=args.flange_arm,
        flange_voltage=args.flange_voltage if args.robot_address else None,
        ping_retries=args.ping_retries,
        ping_retry_s=args.ping_retry_s,
    )
    bridge.initialize()
    Handler.bridge = bridge
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logging.info("Serving gripper bridge at http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
