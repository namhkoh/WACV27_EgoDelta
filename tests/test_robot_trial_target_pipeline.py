from __future__ import annotations

from pathlib import Path

from robot.evaluation.freeze_trial_target import dotted_get, point3
from robot.evaluation.prepare_trial_commands import direct_commands, policy_commands


def policy_row(task: str) -> dict[str, str]:
    return {
        "trial_id": f"wacv27_{task}_01_policy",
        "batch_id": "batch_01",
        "task": task,
        "policy_dir": "D:/policy",
    }


def test_dotted_target_point_is_validated() -> None:
    payload = {"goal": {"point_robot_m": [0.5, -0.2, 1.1]}}
    assert point3(dotted_get(payload, "goal.point_robot_m")) == [0.5, -0.2, 1.1]


def test_policy_manifest_has_separate_locked_execute_stage(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    commands = policy_commands(policy_row("harvest"), target, [0.5, 0.1, 0.9], tmp_path)
    preflight = commands["preflight"]
    execute = commands["execute"]

    assert preflight["commands_robot"] is False
    assert "--execute" not in preflight["argv"]
    assert "--no-snap-goal-to-nearest-demo" in preflight["argv"]
    assert execute["enabled"] is False
    assert execute["commands_robot"] is True
    assert "--execute" in execute["argv"]
    assert commands["phase_plan_and_preflight"]["commands_robot"] is False
    assert (
        commands["phase_plan_and_preflight"]["argv"][
            commands["phase_plan_and_preflight"]["argv"].index("--target-artifact") + 1
        ]
        == str(target)
    )


def test_direct_manifest_does_not_execute_during_preparation(tmp_path: Path) -> None:
    row = {
        "trial_id": "wacv27_deleaf_01_direct",
        "batch_id": "batch_03",
        "task": "deleaf",
    }
    commands = direct_commands(row, tmp_path / "target.json", [0.6, -0.2, 1.0], tmp_path)

    assert commands["plan_and_preflight"]["commands_robot"] is False
    assert "--execute" not in commands["plan_and_preflight"]["argv"]
    assert commands["execute"]["enabled"] is False
    assert "--execute" in commands["execute"]["argv"]

def test_direct_deleaf_manifest_locks_historical_mps_ik_configuration(tmp_path: Path) -> None:
    row = {
        "trial_id": "wacv27_deleaf_01_direct",
        "batch_id": "batch_03",
        "task": "deleaf",
    }
    point = [0.612345678, -0.123456789, 1.101010101]

    commands = direct_commands(row, tmp_path / "target.json", point, tmp_path)
    argv = commands["plan_and_preflight"]["argv"]

    assert argv[argv.index("--selected-point-robot-m") + 1] == "0.612345678,-0.123456789,1.101010101"
    assert argv[argv.index("--waypoint-policy") + 1] == "teleop_offset"
    assert argv[argv.index("--teleop-prior-source-match") + 1] == "AD1"
    assert argv[argv.index("--target-extension-values-m") + 1] == "0"
    assert argv[argv.index("--approach-stroke-standoff-m") + 1] == "0.002"
    assert argv[argv.index("--pre-cut-standoff-m") + 1] == "0.08"
    assert argv[argv.index("--pre-cut-time-s") + 1] == "0.5"
    assert argv[argv.index("--cut-time-s") + 1] == "0.04"
    assert argv[argv.index("--right-velocity-limit-rad-s") + 1] == "4.0"
    assert argv[argv.index("--right-acceleration-limit-rad-s2") + 1] == "18.0"
    assert "--output-json" in argv
    assert "--execute" not in argv

def test_harvest_identity_target_keeps_task_frame_and_supplies_live_identity(tmp_path: Path) -> None:
    task_frame = tmp_path / "tomato0_task_frame.json"
    task_frame.write_text("{}\n", encoding="utf-8")
    frozen_identity = tmp_path / "tomato0_identity.json"
    frozen_identity.write_text("{}\n", encoding="utf-8")
    row = {
        "trial_id": "wacv27_harvest_01_direct",
        "batch_id": "batch_01",
        "task": "harvest",
    }

    commands = direct_commands(
        row,
        frozen_identity,
        [0.57, 0.08, 0.89],
        tmp_path,
        target_provenance={
            "schema": "egodelta_frozen_identity_target_provenance_v1",
            "task_frame_json": str(task_frame),
        },
    )
    argv = commands["plan_and_preflight"]["argv"]

    assert argv[1].endswith("plan_target_conditioned_harvest.py")
    assert argv[argv.index("--goal-robot-m") + 1] == "0.570000000,0.080000000,0.890000000"
    assert argv[argv.index("--target-artifact") + 1] == str(frozen_identity)
    assert commands["structural_task_frame"]["path"] == str(task_frame.resolve())
    execute = commands["execute"]["argv"]
    assert execute[execute.index("--minimum-time-s") + 1] == "4.0"
    assert execute[execute.index("--velocity-limit-rad-s") + 1] == "0.15"
    assert execute[execute.index("--acceleration-limit-rad-s2") + 1] == "0.08"
