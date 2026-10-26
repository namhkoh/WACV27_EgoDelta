from __future__ import annotations

import pytest

from robot.server.gripper_status import (
    GripperStatusError,
    command_targets,
    normalized_status,
    poll_status,
)


def _motor(motor_id: int, position: float, current: float = 0.0) -> dict:
    return {"id": motor_id, "position": position, "current": current}


def test_normalizes_standalone_bridge_schema() -> None:
    result = normalized_status(
        {
            "ok": True,
            "min_q": [-4.0, -5.0],
            "max_q": [6.0, 5.0],
            "motor_states": [_motor(0, 6.0), _motor(1, 0.0, 0.2)],
        }
    )

    assert result["calibration_schema"] == "bridge"
    assert result["grippers"]["right"]["openness"] == pytest.approx(1.0)
    assert result["grippers"]["left"]["openness"] == pytest.approx(0.5)
    assert result["grippers"]["left"]["current_a"] == pytest.approx(0.2)


def test_normalizes_teleop_schema_and_preserves_tcp() -> None:
    tcp = {"ee_left": {"position": [0.1, 0.2, 0.3]}}
    result = normalized_status(
        {
            "ok": True,
            "gripper_min_q": [1.0, 2.0],
            "gripper_max_q": [3.0, 6.0],
            "motor_states": [_motor(0, 1.0), _motor(1, 6.0)],
            "tcp": tcp,
        }
    )

    assert result["calibration_schema"] == "teleop"
    assert result["grippers"]["right"]["openness"] == pytest.approx(0.0)
    assert result["grippers"]["left"]["openness"] == pytest.approx(0.0)
    assert result["tcp"] == tcp


def test_clamps_small_encoder_overshoot_but_reports_it() -> None:
    result = normalized_status(
        {
            "ok": True,
            "min_q": [0.0, 0.0],
            "max_q": [1.0, 1.0],
            "motor_states": [_motor(0, 1.01)],
        }
    )
    right = result["grippers"]["right"]
    assert right["openness"] == pytest.approx(1.0)
    assert right["openness_unclamped"] == pytest.approx(1.01)
    assert right["within_calibration"] is False


def test_rejects_incomplete_or_invalid_calibration() -> None:
    with pytest.raises(GripperStatusError, match="both"):
        normalized_status(
            {"ok": True, "gripper_min_q": [0.0, 0.0], "motor_states": [_motor(0, 0.0)]}
        )
    with pytest.raises(GripperStatusError, match="invalid"):
        normalized_status(
            {
                "ok": True,
                "min_q": [0.0, 1.0],
                "max_q": [0.0, 2.0],
                "motor_states": [_motor(0, 0.0)],
            }
        )


def test_poll_rate_is_capped_at_ten_hz() -> None:
    with pytest.raises(ValueError, match="10"):
        next(poll_status("http://unused", hz=10.01))

def test_command_targets_follow_live_calibration_bounds() -> None:
    bridge = {
        "ok": True,
        "min_q": [-4.1724, -4.50377],
        "max_q": [5.78588, 4.57097],
        "motor_states": [_motor(0, 5.7), _motor(1, 4.5)],
    }
    targets = command_targets(bridge, arm="left")

    assert targets.motor_id == 1
    assert targets.open_position == pytest.approx(-4.50377)
    assert targets.close_position == pytest.approx(4.57097)
    assert targets.encoder_span == pytest.approx(9.07474)
    assert targets.calibration_schema == "bridge"


def test_command_targets_respect_mirrored_encoder_directions() -> None:
    bridge = {
        "ok": True,
        "min_q": [-4.1724, -4.50377],
        "max_q": [5.78588, 4.57097],
        "motor_states": [_motor(0, 5.7), _motor(1, -4.4)],
    }

    right = command_targets(bridge, arm="right")
    left = command_targets(bridge, arm="left")

    assert right.open_position == pytest.approx(5.78588)
    assert right.close_position == pytest.approx(-4.1724)
    assert left.open_position == pytest.approx(-4.50377)
    assert left.close_position == pytest.approx(4.57097)


def test_command_targets_rebase_full_interval_after_encoder_wrap() -> None:
    bridge = {
        "ok": True,
        "min_q": [-4.1724, -4.50376665625],
        "max_q": [5.78588, 4.570970527343751],
        "motor_states": [_motor(0, 5.7), _motor(1, 1.785553265625)],
    }

    targets = command_targets(bridge, arm="left")

    assert targets.encoder_turn_offset == 1
    assert targets.open_position == pytest.approx(-4.50376665625 + 2.0 * 3.141592653589793)
    assert targets.close_position == pytest.approx(4.570970527343751 + 2.0 * 3.141592653589793)
    assert targets.encoder_span == pytest.approx(9.07473718359375)


def test_command_targets_keep_raw_interval_away_from_wrapped_endpoint() -> None:
    bridge = {
        "ok": True,
        "min_q": [-4.0, -5.0],
        "max_q": [6.0, 5.0],
        "motor_states": [_motor(0, 6.0), _motor(1, 0.0)],
    }

    targets = command_targets(bridge, arm="left")

    assert targets.encoder_turn_offset == 0
    assert targets.open_position == pytest.approx(-5.0)
    assert targets.close_position == pytest.approx(5.0)
