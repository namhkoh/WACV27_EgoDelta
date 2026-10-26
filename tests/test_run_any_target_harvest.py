from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot.kinematics.apps import run_any_target_harvest as runner
from robot.kinematics.apps.run_any_target_harvest import selected_target


def test_selected_target_accepts_centered_rgbd_fruit(tmp_path) -> None:
    artifact = tmp_path / "target.json"
    artifact.write_text(
        json.dumps(
            {
                "selected_point": {
                    "point_robot_m": [0.55, 0.10, 0.90],
                    "selection_method": "manual_click_red_instance_center",
                    "estimated_fruit_radius_m": 0.031,
                }
            }
        ),
        encoding="utf-8",
    )

    point, payload = selected_target(artifact)

    assert point == [0.55, 0.10, 0.90]
    assert payload["selected_point"]["estimated_fruit_radius_m"] == 0.031


def test_selected_target_rejects_generic_click(tmp_path) -> None:
    artifact = tmp_path / "target.json"
    artifact.write_text(
        json.dumps(
            {
                "selected_point": {
                    "point_robot_m": [0.55, 0.10, 0.90],
                    "selection_method": "raw_click",
                    "estimated_fruit_radius_m": 0.031,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="centered red-fruit"):
        selected_target(artifact)


def test_selected_target_requires_metric_radius(tmp_path) -> None:
    artifact = tmp_path / "target.json"
    artifact.write_text(
        json.dumps(
            {
                "selected_point": {
                    "point_robot_m": [0.55, 0.10, 0.90],
                    "selection_method": "manual_click_red_instance_center",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="fruit-radius"):
        selected_target(artifact)


def test_point_picker_python_falls_back_to_sibling_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    robot_python = tmp_path / "envs" / "egodelta_robot" / "python.exe"
    picker_python = tmp_path / "envs" / "surfel_splatting" / "python.exe"
    robot_python.parent.mkdir(parents=True)
    picker_python.parent.mkdir(parents=True)
    robot_python.touch()
    picker_python.touch()

    class Probe:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    monkeypatch.setattr(runner.sys, "executable", str(robot_python))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **kwargs: Probe(0 if Path(argv[0]) == picker_python else 1),
    )

    assert runner.resolve_point_picker_python(None) == picker_python.resolve()
