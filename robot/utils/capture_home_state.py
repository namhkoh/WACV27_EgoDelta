from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby


DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_GRIPPER_URL = "http://192.0.2.20:8765"
DEFAULT_OUTPUT = Path("robot/calibration/home/rby1_ready_posture.json")
LINKS = [
    "base",
    "link_torso_5",
    "ee_right",
    "ee_left",
    "link_head_0",
    "link_head_1",
    "link_head_2",
]


def _as_floats(value: Any) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1).tolist()]


def _matrix(value: Any) -> list[list[float]]:
    return [[float(x) for x in row] for row in np.asarray(value, dtype=float).tolist()]


def _degrees(values: list[float]) -> list[float]:
    return [float(math.degrees(x)) for x in values]


def _group(names: list[str], indices: list[int], state: Any) -> dict[str, Any]:
    positions = [float(state.position[i]) for i in indices]
    velocities = [float(state.velocity[i]) for i in indices]
    targets = [float(state.target_position[i]) for i in indices]
    return {
        "joint_indices": indices,
        "joint_names": [names[i] for i in indices],
        "positions_rad": positions,
        "positions_deg": _degrees(positions),
        "velocities_rad_s": velocities,
        "target_positions_rad": targets,
        "target_positions_deg": _degrees(targets),
    }


def _tool_flange_state(value: Any) -> dict[str, Any]:
    return {
        "gyro": _as_floats(value.gyro),
        "acceleration": _as_floats(value.acceleration),
        "switch_A": bool(value.switch_A),
        "output_voltage": float(value.output_voltage),
        "digital_input_A": bool(value.digital_input_A),
        "digital_input_B": bool(value.digital_input_B),
        "digital_output_A": bool(value.digital_output_A),
        "digital_output_B": bool(value.digital_output_B),
    }


def _ft_sensor(value: Any) -> dict[str, Any]:
    return {
        "force": _as_floats(value.force),
        "torque": _as_floats(value.torque),
    }


def _fetch_gripper_status(url: str, timeout_s: float) -> dict[str, Any]:
    status_url = url.rstrip("/") + "/status"
    try:
        with urllib.request.urlopen(status_url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read gripper bridge status at {status_url}: {exc}") from exc
    if not payload.get("ok", False):
        raise RuntimeError(f"gripper bridge status returned not-ok payload: {payload}")
    payload["url"] = url.rstrip("/")
    states = payload.get("motor_states", [])
    by_id = {int(item["id"]): item for item in states if "id" in item}
    payload["right"] = by_id.get(0)
    payload["left"] = by_id.get(1)
    return payload


def capture_home_state(address: str, model_name: str, gripper_url: str, timeout_s: float) -> dict[str, Any]:
    robot = rby.create_robot(address, model_name)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {address}")

    model = robot.model()
    state = robot.get_state()
    joint_names = list(model.robot_joint_names)
    q = _as_floats(state.position)

    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(LINKS, joint_names)
    dyn_state.set_q(np.asarray(q, dtype=float))
    dyn_robot.compute_forward_kinematics(dyn_state)
    transforms = {
        link: _matrix(dyn_robot.compute_transformation(dyn_state, 0, index))
        for index, link in enumerate(LINKS)
        if index != 0
    }

    mobility = _group(joint_names, list(model.mobility_idx), state)
    torso = _group(joint_names, list(model.torso_idx), state)
    right_arm = _group(joint_names, list(model.right_arm_idx), state)
    left_arm = _group(joint_names, list(model.left_arm_idx), state)
    head = _group(joint_names, list(model.head_idx), state)
    body = _group(joint_names, list(model.body_idx), state)

    return {
        "schema": "rby1_home_state",
        "version": 2,
        "source": "live_robot_state_capture",
        "captured_at_unix_s": time.time(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "address": address,
        "model": model_name,
        "robot_model_name": str(model.model_name),
        "robot_dof": int(model.robot_dof),
        "joint_names": joint_names,
        "restore_policy": {
            "base": "fixed",
            "torso": "restore_to_home_then_lock",
            "head": "restore_to_home_then_lock",
            "arms": "restore_to_home_between_trials",
            "grippers": "restore_to_home_between_trials",
        },
        "all_joints": {
            "positions_rad": q,
            "positions_deg": _degrees(q),
            "velocities_rad_s": _as_floats(state.velocity),
            "target_positions_rad": _as_floats(state.target_position),
            "target_positions_deg": _degrees(_as_floats(state.target_position)),
            "current": _as_floats(state.current),
            "torque": _as_floats(state.torque),
            "temperature": [float(x) for x in np.asarray(state.temperature).reshape(-1).tolist()],
        },
        "mobility": mobility,
        "body": body,
        "torso": {
            **torso,
            "link_name": "link_torso_5",
            "T_base_link": transforms["link_torso_5"],
        },
        "arms": {
            "right": {
                **right_arm,
                "link_name": "ee_right",
                "T_base_link": transforms["ee_right"],
            },
            "left": {
                **left_arm,
                "link_name": "ee_left",
                "T_base_link": transforms["ee_left"],
            },
        },
        "head": {
            **head,
            "T_base_links": {
                name: transforms[name]
                for name in ("link_head_0", "link_head_1", "link_head_2")
            },
        },
        "kinematics": {
            "T_base_links": transforms,
            "center_of_mass_base": _as_floats(state.center_of_mass),
            "odometry": _matrix(state.odometry),
        },
        "tool_flange": {
            "right": _tool_flange_state(state.tool_flange_right),
            "left": _tool_flange_state(state.tool_flange_left),
        },
        "force_torque": {
            "right": _ft_sensor(state.ft_sensor_right),
            "left": _ft_sensor(state.ft_sensor_left),
        },
        "grippers": _fetch_gripper_status(gripper_url, timeout_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the live RB-Y1 home state.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    home_state = capture_home_state(args.address, args.model, args.gripper_url, args.timeout_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(home_state, indent=2) + "\n", encoding="utf-8")
    print(f"captured home state: {args.output}")


if __name__ == "__main__":
    main()
