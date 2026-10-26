from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
UNKNOWN = "not_recorded"
TASKS = ("harvest", "deleaf")
METHODS = ("direct", "policy")
BOOL_VALUES = ("true", "false", UNKNOWN)
FAILURES = (
    "none",
    "missing_target",
    "target_transform_failure",
    "planning_failure",
    "automatic_gate_rejection",
    "operator_rejection",
    "execution_command_failure",
    "physical_contact_failure",
    "gripper_failure",
    "safety_stop",
    "human_intervention",
    "scene_reset_failure",
    "other",
    UNKNOWN,
)
LEDGER_FIELDS = (
    "trial_id",
    "batch_id",
    "schedule_index",
    "scene_slot",
    "within_scene_order",
    "date",
    "scene_id",
    "task",
    "evaluation_type",
    "target_id",
    "target_artifact_path",
    "target_artifact_sha256",
    "plan_generated",
    "automatic_gate_rejected",
    "operator_rejected_before_execution",
    "execution_started",
    "operator_or_safety_stop",
    "success",
    "failure_category",
    "failure_description",
    "included_in_denominator",
    "raw_log_path",
    "video_path",
    "execution_artifact_path",
    "execution_artifact_sha256",
    "policy_dir",
    "policy_manifest_sha256",
    "source_commit",
    "trial_status",
    "started_at_utc",
    "completed_at_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory contains no policy artifacts: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_ledger(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
            raise ValueError("ledger columns do not match the registered protocol")
        return list(reader)


def session_paths(session_dir: Path) -> tuple[Path, Path, Path]:
    return (
        session_dir / "trial_protocol.json",
        session_dir / "robot_trial_ledger.csv",
        session_dir / "robot_trial_summary.json",
    )


def policy_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"policy directory does not exist: {resolved}")
    return {"path": str(resolved), "manifest_sha256": sha256_directory(resolved)}


def batch_for(task: str, scene_slot: int, scenes_per_batch: int) -> str:
    task_offset = 0 if task == "harvest" else 2
    local_batch = (scene_slot - 1) // scenes_per_batch + 1
    return f"batch_{task_offset + local_batch:02d}"


def build_schedule(
    trials_per_task: int,
    seed: int,
    policies: dict[str, dict[str, str]],
    scenes_per_batch: int = 15,
) -> list[dict[str, str]]:
    if trials_per_task < 1 or scenes_per_batch < 1:
        raise ValueError("trial and batch sizes must be positive")
    if trials_per_task % scenes_per_batch:
        raise ValueError("trials_per_task must be divisible by scenes_per_batch")
    rng = random.Random(seed)
    commit = git_commit()
    rows: list[dict[str, str]] = []
    schedule_index = 0
    for task in TASKS:
        for scene_slot in range(1, trials_per_task + 1):
            methods = list(METHODS)
            rng.shuffle(methods)
            for within_scene_order, method in enumerate(methods, start=1):
                schedule_index += 1
                policy = policies[task] if method == "policy" else {
                    "path": "not_applicable",
                    "manifest_sha256": "not_applicable",
                }
                rows.append(
                    {
                        "trial_id": f"wacv27_{task}_{scene_slot:02d}_{method}",
                        "batch_id": batch_for(task, scene_slot, scenes_per_batch),
                        "schedule_index": str(schedule_index),
                        "scene_slot": str(scene_slot),
                        "within_scene_order": str(within_scene_order),
                        "date": UNKNOWN,
                        "scene_id": UNKNOWN,
                        "task": task,
                        "evaluation_type": method,
                        "target_id": UNKNOWN,
                        "target_artifact_path": UNKNOWN,
                        "target_artifact_sha256": UNKNOWN,
                        "plan_generated": UNKNOWN,
                        "automatic_gate_rejected": UNKNOWN,
                        "operator_rejected_before_execution": UNKNOWN,
                        "execution_started": UNKNOWN,
                        "operator_or_safety_stop": UNKNOWN,
                        "success": UNKNOWN,
                        "failure_category": UNKNOWN,
                        "failure_description": "",
                        "included_in_denominator": "true",
                        "raw_log_path": UNKNOWN,
                        "video_path": UNKNOWN,
                        "execution_artifact_path": UNKNOWN,
                        "execution_artifact_sha256": UNKNOWN,
                        "policy_dir": policy["path"],
                        "policy_manifest_sha256": policy["manifest_sha256"],
                        "source_commit": commit,
                        "trial_status": "scheduled",
                        "started_at_utc": UNKNOWN,
                        "completed_at_utc": UNKNOWN,
                    }
                )
    rows.sort(key=lambda row: (row["batch_id"], int(row["scene_slot"]), int(row["within_scene_order"])))
    for index, row in enumerate(rows, start=1):
        row["schedule_index"] = str(index)
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_condition: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_batch: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[(row["task"], row["evaluation_type"])].append(row)
        by_batch[row["batch_id"]].append(row)
        by_pair[(row["task"], row["scene_slot"])].append(row)
    mismatches: list[dict[str, str]] = []
    for (task, slot), pair in sorted(by_pair.items()):
        hashes = {row["target_artifact_sha256"] for row in pair if row["target_artifact_sha256"] != UNKNOWN}
        scenes = {row["scene_id"] for row in pair if row["scene_id"] != UNKNOWN}
        targets = {row["target_id"] for row in pair if row["target_id"] != UNKNOWN}
        if len(pair) != 2 or len(hashes) > 1 or len(scenes) > 1 or len(targets) > 1:
            mismatches.append({"task": task, "scene_slot": slot})

    def counts(items: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "scheduled": len(items),
            "status_counts": dict(Counter(row["trial_status"] for row in items)),
            "success_counts": dict(Counter(row["success"] for row in items)),
        }

    return {
        "schema": "egodelta_robot_trial_summary_v1",
        "updated_at_utc": utc_now(),
        "scheduled_rollout_count": len(rows),
        "completed_rollout_count": sum(row["trial_status"] == "completed" for row in rows),
        "included_in_denominator_count": sum(row["included_in_denominator"] == "true" for row in rows),
        "conditions": {f"{task}_{method}": counts(items) for (task, method), items in sorted(by_condition.items())},
        "batches": {batch: counts(items) for batch, items in sorted(by_batch.items())},
        "paired_target_mismatch_count": len(mismatches),
        "paired_target_mismatches": mismatches,
    }


def refresh_summary(session_dir: Path) -> None:
    _, ledger_path, summary_path = session_paths(session_dir)
    write_json(summary_path, summarize(read_ledger(ledger_path)))


def initialize(args: argparse.Namespace) -> None:
    session_dir = args.output_dir.resolve()
    protocol_path, ledger_path, _ = session_paths(session_dir)
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(f"refusing to initialize non-empty directory: {session_dir}")
    policies = {
        "harvest": policy_record(args.harvest_policy_dir),
        "deleaf": policy_record(args.deleaf_policy_dir),
    }
    rows = build_schedule(args.trials_per_task, args.seed, policies, args.scenes_per_batch)
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        protocol_path,
        {
            "schema": "egodelta_prospective_robot_trial_protocol_v1",
            "created_at_utc": utc_now(),
            "source_commit": git_commit(),
            "random_seed": args.seed,
            "trials_per_task_per_condition": args.trials_per_task,
            "scenes_per_batch": args.scenes_per_batch,
            "scheduled_rollout_count": len(rows),
            "first_batch": "batch_01: harvest scene slots 1-15, paired direct/policy, 30 rollouts",
            "paired_design": (
                "Each scene slot is evaluated once with direct execution and once with the learned policy. "
                "Method order is randomized within each pair and the fixture is reset between conditions."
            ),
            "target_rule": (
                "Both paired trials use the same pre-execution EgoDelta target artifact and SHA-256. "
                "Policy goal snapping is disabled."
            ),
            "evaluation_only": (
                "Do not add these outcomes or trajectories to policy training until the complete held-out "
                "evaluation and result manifest are frozen."
            ),
            "denominator_rule": "Every registered row remains in the denominator, including rejections and stops.",
            "policies": policies,
            "ledger": str(ledger_path),
        },
    )
    write_ledger(ledger_path, rows)
    refresh_summary(session_dir)
    print(json.dumps({"session_dir": str(session_dir), "scheduled_rollout_count": len(rows)}, indent=2))


def find_trial(rows: list[dict[str, str]], trial_id: str) -> dict[str, str]:
    found = [row for row in rows if row["trial_id"] == trial_id]
    if len(found) != 1:
        raise ValueError(f"expected one row for {trial_id!r}, found {len(found)}")
    return found[0]


def bind(args: argparse.Namespace) -> None:
    session_dir = args.session_dir.resolve()
    _, ledger_path, _ = session_paths(session_dir)
    rows = read_ledger(ledger_path)
    row = find_trial(rows, args.trial_id)
    if row["trial_status"] != "scheduled":
        raise ValueError(f"trial is already {row['trial_status']}; target binding is immutable")
    target = args.target_artifact.resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    target_hash = sha256_file(target)
    pair = [
        item for item in rows
        if item["task"] == row["task"]
        and item["scene_slot"] == row["scene_slot"]
        and item["trial_id"] != row["trial_id"]
    ]
    if len(pair) != 1:
        raise ValueError("registered trial does not have exactly one paired condition")
    paired = pair[0]
    if paired["target_artifact_sha256"] not in (UNKNOWN, target_hash):
        raise ValueError("paired direct/policy trials must use the same frozen target artifact")
    if paired["scene_id"] not in (UNKNOWN, args.scene_id) or paired["target_id"] not in (UNKNOWN, args.target_id):
        raise ValueError("paired direct/policy trials must use the same scene_id and target_id")
    row.update(
        date=args.date or datetime.now().date().isoformat(),
        scene_id=args.scene_id,
        target_id=args.target_id,
        target_artifact_path=str(target),
        target_artifact_sha256=target_hash,
        trial_status="bound",
    )
    write_ledger(ledger_path, rows)
    refresh_summary(session_dir)
    print(json.dumps(row, indent=2, sort_keys=True))


def annotate(args: argparse.Namespace) -> None:
    session_dir = args.session_dir.resolve()
    _, ledger_path, _ = session_paths(session_dir)
    rows = read_ledger(ledger_path)
    row = find_trial(rows, args.trial_id)
    if row["trial_status"] != "bound":
        raise ValueError(f"trial must be bound before annotation; status={row['trial_status']}")
    if args.success == "true" and args.failure_category not in ("none", UNKNOWN):
        raise ValueError("successful trial cannot have a failure category")
    if args.success == "false" and args.failure_category in ("none", UNKNOWN):
        raise ValueError("failed trial requires an explicit failure category")
    artifact = args.execution_artifact.resolve() if args.execution_artifact else None
    if artifact is not None and not artifact.is_file():
        raise FileNotFoundError(artifact)
    row.update(
        plan_generated=args.plan_generated,
        automatic_gate_rejected=args.automatic_gate_rejected,
        operator_rejected_before_execution=args.operator_rejected_before_execution,
        execution_started=args.execution_started,
        operator_or_safety_stop=args.operator_or_safety_stop,
        success=args.success,
        failure_category=args.failure_category,
        failure_description=args.failure_description,
        raw_log_path=str(args.raw_log.resolve()) if args.raw_log else UNKNOWN,
        video_path=str(args.video.resolve()) if args.video else UNKNOWN,
        execution_artifact_path=str(artifact) if artifact else UNKNOWN,
        execution_artifact_sha256=sha256_file(artifact) if artifact else UNKNOWN,
        started_at_utc=args.started_at_utc or UNKNOWN,
        completed_at_utc=args.completed_at_utc or utc_now(),
        trial_status="completed",
    )
    write_ledger(ledger_path, rows)
    refresh_summary(session_dir)
    print(json.dumps(row, indent=2, sort_keys=True))


def show_next(args: argparse.Namespace) -> None:
    _, ledger_path, _ = session_paths(args.session_dir.resolve())
    rows = read_ledger(ledger_path)
    candidates = [
        row for row in rows
        if row["trial_status"] != "completed" and (args.batch_id is None or row["batch_id"] == args.batch_id)
    ]
    candidates.sort(key=lambda row: int(row["schedule_index"]))
    print(json.dumps({"complete": not candidates, "next_trial": candidates[0] if candidates else None}, indent=2))


def add_bool(parser: argparse.ArgumentParser, flag: str) -> None:
    parser.add_argument(flag, choices=BOOL_VALUES, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintain a pre-registered, failure-preserving robot trial ledger.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output-dir", type=Path, required=True)
    init.add_argument("--trials-per-task", type=int, default=30)
    init.add_argument("--scenes-per-batch", type=int, default=15)
    init.add_argument("--seed", type=int, default=1493)
    init.add_argument("--harvest-policy-dir", type=Path, required=True)
    init.add_argument("--deleaf-policy-dir", type=Path, required=True)
    init.set_defaults(func=initialize)

    bind_parser = sub.add_parser("bind")
    bind_parser.add_argument("--session-dir", type=Path, required=True)
    bind_parser.add_argument("--trial-id", required=True)
    bind_parser.add_argument("--scene-id", required=True)
    bind_parser.add_argument("--target-id", required=True)
    bind_parser.add_argument("--target-artifact", type=Path, required=True)
    bind_parser.add_argument("--date")
    bind_parser.set_defaults(func=bind)

    note = sub.add_parser("annotate")
    note.add_argument("--session-dir", type=Path, required=True)
    note.add_argument("--trial-id", required=True)
    add_bool(note, "--plan-generated")
    add_bool(note, "--automatic-gate-rejected")
    add_bool(note, "--operator-rejected-before-execution")
    add_bool(note, "--execution-started")
    add_bool(note, "--operator-or-safety-stop")
    add_bool(note, "--success")
    note.add_argument("--failure-category", choices=FAILURES, required=True)
    note.add_argument("--failure-description", default="")
    note.add_argument("--raw-log", type=Path)
    note.add_argument("--video", type=Path)
    note.add_argument("--execution-artifact", type=Path)
    note.add_argument("--started-at-utc")
    note.add_argument("--completed-at-utc")
    note.set_defaults(func=annotate)

    next_parser = sub.add_parser("next")
    next_parser.add_argument("--session-dir", type=Path, required=True)
    next_parser.add_argument("--batch-id")
    next_parser.set_defaults(func=show_next)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
