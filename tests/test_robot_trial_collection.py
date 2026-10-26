from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot.evaluation import collect_robot_trials as trials


def policies() -> dict[str, dict[str, str]]:
    return {
        "harvest": {"path": "harvest-policy", "manifest_sha256": "h" * 64},
        "deleaf": {"path": "deleaf-policy", "manifest_sha256": "d" * 64},
    }


def test_schedule_registers_120_paired_trials_in_four_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trials, "git_commit", lambda: "c" * 40)
    rows = trials.build_schedule(30, 1493, policies(), scenes_per_batch=15)

    assert len(rows) == 120
    assert len({row["trial_id"] for row in rows}) == 120
    assert Counter(row["batch_id"] for row in rows) == {
        "batch_01": 30,
        "batch_02": 30,
        "batch_03": 30,
        "batch_04": 30,
    }
    assert all(row["included_in_denominator"] == "true" for row in rows)
    for task in trials.TASKS:
        for scene_slot in range(1, 31):
            pair = [row for row in rows if row["task"] == task and int(row["scene_slot"]) == scene_slot]
            assert {row["evaluation_type"] for row in pair} == {"direct", "policy"}
            assert {row["within_scene_order"] for row in pair} == {"1", "2"}


def test_bind_rejects_different_target_for_paired_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trials, "git_commit", lambda: "c" * 40)
    session = tmp_path / "session"
    session.mkdir()
    rows = trials.build_schedule(2, 5, policies(), scenes_per_batch=1)
    _, ledger_path, _ = trials.session_paths(session)
    trials.write_ledger(ledger_path, rows)
    trials.refresh_summary(session)
    pair = [row for row in rows if row["task"] == "harvest" and row["scene_slot"] == "1"]
    target_a = tmp_path / "target_a.json"
    target_b = tmp_path / "target_b.json"
    target_a.write_text('{"target_robot_m": [1, 2, 3]}', encoding="utf-8")
    target_b.write_text('{"target_robot_m": [4, 5, 6]}', encoding="utf-8")

    trials.bind(
        SimpleNamespace(
            session_dir=session,
            trial_id=pair[0]["trial_id"],
            scene_id="scene-001",
            target_id="target-001",
            target_artifact=target_a,
            date="2026-08-19",
        )
    )
    with pytest.raises(ValueError, match="same frozen target artifact"):
        trials.bind(
            SimpleNamespace(
                session_dir=session,
                trial_id=pair[1]["trial_id"],
                scene_id="scene-001",
                target_id="target-001",
                target_artifact=target_b,
                date="2026-08-19",
            )
        )


def test_ledger_round_trip_uses_fixed_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trials, "git_commit", lambda: "c" * 40)
    path = tmp_path / "ledger.csv"
    rows = trials.build_schedule(2, 7, policies(), scenes_per_batch=1)
    trials.write_ledger(path, rows)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == trials.LEDGER_FIELDS
        assert len(list(reader)) == 8
