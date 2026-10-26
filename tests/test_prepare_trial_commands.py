from __future__ import annotations

from pathlib import Path

from robot.evaluation.prepare_trial_commands import policy_commands


def _option(argv: list[str], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def test_harvest_policy_uses_smooth_approach_and_named_phase_handoff(tmp_path: Path) -> None:
    row = {
        "trial_id": "harvest-policy-001",
        "batch_id": "batch_01",
        "task": "harvest",
        "policy_dir": "policy",
    }

    target = tmp_path / "target.json"
    commands = policy_commands(row, target, [0.57, 0.05, 0.87], tmp_path)
    argv = commands["execute"]["argv"]

    assert _option(argv, "--steps-to-execute") == "30"
    assert _option(argv, "--guard-horizon-steps") == "30"
    assert _option(argv, "--minimum-time-s") == "3.0"
    assert _option(argv, "--policy-velocity-limit-rad-s") == "0.15"
    assert _option(argv, "--policy-acceleration-limit-rad-s2") == "0.08"
    assert _option(argv, "--harvest-policy-handoff-plan-margin-m") == "0.05"
    assert "--harvest-policy-handoff-distance-m" not in argv
    assert _option(argv, "--harvest-fixed-basket-tail-start-after") == "begin"
    assert _option(argv, "--harvest-fixed-basket-tail-plan") == "<PHASE_PLANNER_OUTPUT_JSON>"
    assert _option(argv, "--harvest-fixed-basket-tail-minimum-time-s") == "4.0"
    assert _option(argv, "--harvest-tail-velocity-limit-rad-s") == "0.15"
    assert _option(argv, "--harvest-tail-acceleration-limit-rad-s2") == "0.08"
    assert "--no-harvest-fixed-basket-tail-require-closed" in argv
    assert "--no-harvest-grasp-retarget" in argv
    assert "--harvest-grasp-retarget" not in argv
    planner = commands["phase_plan_and_preflight"]["argv"]
    assert Path(_option(planner, "--target-artifact")) == target
    assert _option(planner, "--goal-robot-m") == "0.570000000,0.050000000,0.870000000"


def test_harvest_preflight_and_execution_share_the_same_controller_profile(tmp_path: Path) -> None:
    row = {
        "trial_id": "harvest-policy-001",
        "batch_id": "batch_01",
        "task": "harvest",
        "policy_dir": "policy",
    }

    commands = policy_commands(
        row,
        tmp_path / "target.json",
        [0.57, 0.05, 0.87],
        tmp_path,
    )
    preflight = commands["preflight"]["argv"]
    execute = commands["execute"]["argv"]

    assert preflight[:-2] == execute[:-3]
    assert preflight[-2] == "--output-json"
    assert execute[-3] == "--output-json"
    assert execute[-1] == "--execute"
