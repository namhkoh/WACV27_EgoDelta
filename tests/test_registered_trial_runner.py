from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot.evaluation import collect_robot_trials as ledger
from robot.evaluation import run_registered_trial as runner
from robot.evaluation.prepare_registered_scene import (
    find_validation_artifact,
    paired_scene_rows,
    verify_resumable_validation,
)


def _policies() -> dict[str, dict[str, str]]:
    return {
        "harvest": {"path": "harvest-policy", "manifest_sha256": "h" * 64},
        "deleaf": {"path": "deleaf-policy", "manifest_sha256": "d" * 64},
    }


def _bound_direct_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, str], Path]:
    monkeypatch.setattr(ledger, "git_commit", lambda: "c" * 40)
    session = tmp_path / "session"
    session.mkdir()
    rows = ledger.build_schedule(2, 5, _policies(), scenes_per_batch=1)
    _, ledger_path, _ = ledger.session_paths(session)
    ledger.write_ledger(ledger_path, rows)
    ledger.refresh_summary(session)
    row = next(
        item
        for item in rows
        if item["task"] == "harvest"
        and item["scene_slot"] == "1"
        and item["evaluation_type"] == "direct"
    )
    target = tmp_path / "target.json"
    target.write_text('{"target_robot_m": [0.5, 0.1, 0.9]}', encoding="utf-8")
    ledger.bind(
        SimpleNamespace(
            session_dir=session,
            trial_id=row["trial_id"],
            scene_id="scene-001",
            target_id="tomato0",
            target_artifact=target,
            date="2026-08-19",
        )
    )
    updated = ledger.find_trial(ledger.read_ledger(ledger_path), row["trial_id"])
    return session, updated, target


def test_replace_option_is_deterministic_and_rejects_duplicates() -> None:
    assert runner.replace_option(["cmd"], "--x", "1") == ["cmd", "--x", "1"]
    assert runner.replace_option(["cmd", "--x", "0"], "--x", "1") == ["cmd", "--x", "1"]
    with pytest.raises(ValueError, match="duplicate"):
        runner.replace_option(["cmd", "--x", "0", "--x", "1"], "--x", "2")


def test_direct_execution_command_uses_frozen_plan_and_live_gripper_targets(tmp_path: Path) -> None:
    row = {
        "trial_id": "trial-direct",
        "task": "harvest",
        "evaluation_type": "direct",
    }
    manifest = {
        "commands": {
            "execute": {
                "argv": [
                    "python",
                    "execute_harvest_plan.py",
                    "<PLANNER_OUTPUT_JSON>",
                    "--use-gripper",
                    "--output-json",
                    "old.json",
                    "--execute",
                ]
            }
        }
    }
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")

    argv, output = runner.execution_argv(
        row,
        manifest,
        tmp_path,
        "http://bridge:8765",
        plan,
    )

    assert str(plan) in argv
    assert "<PLANNER_OUTPUT_JSON>" not in argv
    assert runner.option_value(argv, "--gripper-url") == "http://bridge:8765"
    assert runner.option_value(argv, "--gripper-target-source") == "live_status"
    assert runner.option_value(argv, "--output-json") == str(output)
    assert "--execute" in argv


def test_policy_execution_command_uses_hash_locked_phase_plan(tmp_path: Path) -> None:
    row = {
        "trial_id": "trial-policy",
        "task": "harvest",
        "evaluation_type": "policy",
    }
    manifest = {
        "commands": {
            "execute": {
                "argv": [
                    "python",
                    "guarded_bc_rollout.py",
                    "--harvest-fixed-basket-tail-plan",
                    "<PHASE_PLANNER_OUTPUT_JSON>",
                    "--output-json",
                    "old.json",
                    "--execute",
                ]
            }
        }
    }
    plan = tmp_path / "phase-plan.json"
    plan.write_text("{}\n", encoding="utf-8")

    argv, output = runner.execution_argv(
        row,
        manifest,
        tmp_path,
        "http://bridge:8765",
        plan,
    )

    assert runner.option_value(argv, "--harvest-fixed-basket-tail-plan") == str(plan)
    assert "<PHASE_PLANNER_OUTPUT_JSON>" not in argv
    assert runner.option_value(argv, "--output-json") == str(output)


def test_policy_phase_preflight_returns_strict_planner_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_plan(argv: object, _log: Path) -> runner.ProcessResult:
        args = list(argv)
        root = Path(runner.option_value(args, "--runs-root"))
        root.mkdir(parents=True, exist_ok=True)
        (root / "phase-plan.json").write_text(
            json.dumps({"hard_gate": {"ok": True}, "candidates": [{}]}),
            encoding="utf-8",
        )
        return runner.ProcessResult(argv=tuple(args), returncode=0, output="ok")

    monkeypatch.setattr(runner, "run_logged", write_plan)
    manifest = {
        "commands": {
            "phase_plan_and_preflight": {
                "argv": ["python", "plan.py", "--run-label", "trial"]
            }
        }
    }

    path, payload = runner.policy_harvest_phase_preflight(
        manifest,
        tmp_path,
        tmp_path / "console.log",
        1,
    )

    assert path.name == "phase-plan.json"
    assert payload["hard_gate"]["ok"] is True


def test_policy_guard_before_motion_is_automatic_rejection() -> None:
    category, automatic, execution_started, description = runner.classify_execution_failure(
        {"evaluation_type": "policy"},
        2,
        {"execution_error": "guard_failed", "executed": False},
    )

    assert category == "automatic_gate_rejection"
    assert automatic is True
    assert execution_started is False
    assert "before motion" in description


def test_harvest_policy_contract_requires_learned_approach_and_complete_phase_tail() -> None:
    row = {"evaluation_type": "policy", "task": "harvest"}
    payload = {
        "controller_execution_mode": "hybrid_policy_and_fixed_tail",
        "policy_motion_executed": True,
        "policy_handoff": {"triggered": True, "source": "phase_plan_approach"},
        "fixed_basket_tail": {
            "ok": True,
            "live_transition_hard_gate": {"ok": True},
            "start_after": "begin",
            "waypoints": [
                {"name": name}
                for name in (
                    "approach",
                    "orient_soft_03",
                    "orient_soft_07",
                    "pre_grasp",
                    "grasp",
                    "harvest",
                    "retract",
                    "basket_place",
                )
            ],
            "gripper_events": [
                {"action": "pre_grasp_open"},
                {"action": "grasp_close"},
                {"action": "basket_release_open"},
            ],
        },
    }

    assert runner.validate_policy_execution_contract(row, payload) is None

    payload["controller_execution_mode"] = "deterministic_tail_only"
    error = runner.validate_policy_execution_contract(row, payload)
    assert error is not None
    assert "learned approach" in error


def test_non_harvest_policy_has_no_hybrid_contract() -> None:
    assert (
        runner.validate_policy_execution_contract(
            {"evaluation_type": "policy", "task": "deleaf"},
            {"controller_execution_mode": "policy_only"},
        )
        is None
    )


def test_connectivity_failure_detects_sdk_connection_errors() -> None:
    result = runner.ProcessResult(
        argv=("python", "planner.py"),
        returncode=1,
        output="RuntimeError: failed to connect to RB-Y1 at 192.0.2.10:50051",
    )

    assert runner.connectivity_failure(result) is True


def test_registered_home_is_task_specific() -> None:
    harvest = runner.default_home_path({"task": "harvest"})
    deleaf = runner.default_home_path({"task": "deleaf"})

    assert harvest.name == "homestate.json"
    assert deleaf.name == "deleaf_home_right_tucked.json"


def test_direct_deleaf_preflight_requires_expected_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "deleaf_preflight.json"

    def write_passing(_argv: object, _log: Path, *, attempts: int) -> runner.ProcessResult:
        assert attempts == 3
        output.write_text(
            json.dumps(
                {
                    "method": "deleaf_click_cut_mps_guided_ik",
                    "hard_gate": {"ok": True},
                    "home_check": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return runner.ProcessResult(argv=("python",), returncode=0, output="ok")

    monkeypatch.setattr(runner, "run_read_only_with_connectivity_retry", write_passing)
    manifest = {
        "commands": {
            "plan_and_preflight": {
                "argv": ["python", "deleaf_click_cut.py", "--output-json", str(output)]
            }
        }
    }

    path, payload = runner.direct_deleaf_preflight(manifest, tmp_path, tmp_path / "log.txt", 3)

    assert path == output
    assert payload["hard_gate"]["ok"] is True

def test_read_only_connectivity_retry_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fail_connectivity(_argv: object, _log_path: Path) -> runner.ProcessResult:
        calls.append(1)
        return runner.ProcessResult(
            argv=("python", "planner.py"),
            returncode=1,
            output="failed to connect to RB-Y1",
        )

    monkeypatch.setattr(runner, "run_logged", fail_connectivity)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(runner.InfrastructureUnavailable, match="after 3 read-only attempts"):
        runner.run_read_only_with_connectivity_retry(
            ("python", "planner.py"),
            tmp_path / "console.log",
            attempts=3,
        )

    assert len(calls) == 3

def test_preflight_state_is_hash_verified(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"point": [1, 2, 3]}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text('{"ok": true}\n', encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text('{"hard_gate": {"ok": true}}\n', encoding="utf-8")
    state = tmp_path / "state.json"
    row = {"trial_id": "trial-001"}
    controller = {"files": {"controller.py": "abc"}}

    runner.write_preflight_state(
        state,
        row=row,
        target=target,
        manifest_path=manifest,
        controller=controller,
        gripper={"ok": True},
        preflight_artifact=preflight,
        plan_path=plan,
    )
    _, loaded_plan, loaded_preflight = runner.load_preflight_state(
        state,
        row=row,
        target=target,
        manifest_path=manifest,
        controller=controller,
    )

    assert loaded_plan == plan.resolve()
    assert loaded_preflight == preflight.resolve()

    plan.write_text('{"hard_gate": {"ok": false}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="planner artifact hash"):
        runner.load_preflight_state(
            state,
            row=row,
            target=target,
            manifest_path=manifest,
            controller=controller,
        )


def test_success_annotation_completes_once_and_refuses_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, row, _target = _bound_direct_trial(tmp_path, monkeypatch)
    raw_log = tmp_path / "console.log"
    raw_log.write_text("complete\n", encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text('{"ok": true}\n', encoding="utf-8")

    runner.annotate(
        session,
        row,
        plan_generated=True,
        automatic_gate_rejected=False,
        operator_rejected=False,
        execution_started=True,
        operator_stop=False,
        success=True,
        failure_category="none",
        failure_description="",
        raw_log=raw_log,
        video=None,
        execution_artifact=execution,
        started_at_utc="2026-08-19T00:00:00+00:00",
    )

    _, ledger_path, _ = ledger.session_paths(session)
    completed = ledger.find_trial(ledger.read_ledger(ledger_path), row["trial_id"])
    assert completed["trial_status"] == "completed"
    assert completed["success"] == "true"

    with pytest.raises(ValueError, match="must be bound"):
        runner.annotate(
            session,
            completed,
            plan_generated=True,
            automatic_gate_rejected=False,
            operator_rejected=False,
            execution_started=True,
            operator_stop=False,
            success=True,
            failure_category="none",
            failure_description="",
            raw_log=raw_log,
            video=None,
            execution_artifact=execution,
            started_at_utc="2026-08-19T00:00:00+00:00",
        )

def test_scene_preparation_selects_exact_registered_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ledger, "git_commit", lambda: "c" * 40)
    session = tmp_path / "session-pair"
    session.mkdir()
    rows = ledger.build_schedule(2, 5, _policies(), scenes_per_batch=1)
    _, ledger_path, _ = ledger.session_paths(session)
    ledger.write_ledger(ledger_path, rows)

    pair = paired_scene_rows(session, task="harvest", scene_slot=1)

    assert len(pair) == 2
    assert {row["evaluation_type"] for row in pair} == {"direct", "policy"}
    assert [row["within_scene_order"] for row in pair] == ["1", "2"]

def test_scene_preparation_selects_canonical_validation_artifact(tmp_path: Path) -> None:
    label = "scene01_tomato0"
    capture = tmp_path / "live_realsense_capture" / f"a270c038_{label}.json"
    canonical = tmp_path / "target_identity_validation" / f"a270c038_{label}.json"
    payload = {
        "method": "live_realsense_task_frame_identity_validation",
        "ok": True,
    }
    for path in (capture, canonical):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    selected, _ = find_validation_artifact(tmp_path, run_label=label)

    assert selected == canonical.resolve()


def test_resumable_validation_checks_task_frame_provenance(tmp_path: Path) -> None:
    task_frame = tmp_path / "task_frame.json"
    task_frame.write_text("{}", encoding="utf-8")
    validation = {
        "ok": True,
        "dataset_name": "a270c038",
        "run_label": "scene01_tomato0",
        "task_frames": [{"source_path": str(task_frame)}],
    }

    verify_resumable_validation(
        validation,
        dataset_name="a270c038",
        run_label="scene01_tomato0",
        task_frame=task_frame.resolve(),
    )

    validation["task_frames"][0]["source_path"] = str(tmp_path / "wrong.json")
    with pytest.raises(ValueError, match="task-frame mismatch"):
        verify_resumable_validation(
            validation,
            dataset_name="a270c038",
            run_label="scene01_tomato0",
            task_frame=task_frame.resolve(),
        )
