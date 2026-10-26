from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot.evaluation.collect_robot_trials import (
    build_schedule,
    git_commit,
    policy_record,
    refresh_summary,
    session_paths,
    utc_now,
    write_json,
    write_ledger,
)


def initialize(args: argparse.Namespace) -> dict[str, object]:
    session_dir = args.output_dir.resolve()
    protocol_path, ledger_path, _summary_path = session_paths(session_dir)
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(f"refusing to initialize non-empty directory: {session_dir}")

    policies = {
        "harvest": policy_record(args.harvest_policy_dir),
        "deleaf": policy_record(args.deleaf_policy_dir),
    }
    rows = build_schedule(
        args.additional_trials_per_task_per_condition,
        args.seed,
        policies,
        args.scenes_per_batch,
    )
    for row in rows:
        row["trial_id"] = row["trial_id"].replace("wacv27_", "wacv27_additional20_", 1)

    additional = int(args.additional_trials_per_task_per_condition)
    historical = int(args.historical_trials_per_task_per_condition)
    conditions = 4  # harvest/deleaf x direct/policy
    protocol = {
        "schema": "egodelta_prospective_additional_robot_trial_protocol_v2",
        "cohort_id": args.cohort_id,
        "created_at_utc": utc_now(),
        "source_commit": git_commit(),
        "random_seed": int(args.seed),
        "historical_trials_per_task_per_condition": historical,
        "additional_trials_per_task_per_condition": additional,
        "planned_final_trials_per_task_per_condition": historical + additional,
        "additional_rollout_count": additional * conditions,
        "planned_final_rollout_count": (historical + additional) * conditions,
        "scenes_per_batch": int(args.scenes_per_batch),
        "first_batch": (
            f"batch_01: harvest scene slots 1-{args.scenes_per_batch}, paired direct/policy, "
            f"{2 * args.scenes_per_batch} additional rollouts"
        ),
        "paired_design": (
            "Each new scene slot is evaluated once with direct execution and once with the learned policy. "
            "Method order is randomized within each pair, the fixture is reset between conditions, and both "
            "conditions use the same frozen target hash."
        ),
        "historical_cohort_rule": (
            "The prior 10 trials per task per condition remain a separately identified historical cohort. "
            "Report new-cohort and historical-cohort results separately; report a combined 30-trial result "
            "only after confirming matching task definitions, success criteria, and condition implementations."
        ),
        "target_rule": (
            "Both paired trials use the same pre-execution EgoDelta target artifact and SHA-256. "
            "Policy goal snapping is disabled."
        ),
        "evaluation_only": (
            "Do not add these outcomes or trajectories to policy training until the complete additional "
            "held-out evaluation and result manifest are frozen."
        ),
        "denominator_rule": "Every registered additional-cohort row remains in the denominator.",
        "policies": policies,
        "ledger": str(ledger_path),
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(protocol_path, protocol)
    write_ledger(ledger_path, rows)
    refresh_summary(session_dir)
    return {
        "session_dir": str(session_dir),
        "additional_rollout_count": len(rows),
        "planned_final_rollout_count": protocol["planned_final_rollout_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Register the prospective additional-20 robot cohort.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort-id", default="wacv27_additional20_20260819")
    parser.add_argument("--additional-trials-per-task-per-condition", type=int, default=20)
    parser.add_argument("--historical-trials-per-task-per-condition", type=int, default=10)
    parser.add_argument("--scenes-per-batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1493)
    parser.add_argument("--harvest-policy-dir", type=Path, required=True)
    parser.add_argument("--deleaf-policy-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.additional_trials_per_task_per_condition <= 0:
        parser.error("--additional-trials-per-task-per-condition must be positive")
    if args.historical_trials_per_task_per_condition < 0:
        parser.error("--historical-trials-per-task-per-condition cannot be negative")
    if args.scenes_per_batch <= 0:
        parser.error("--scenes-per-batch must be positive")
    if args.additional_trials_per_task_per_condition % args.scenes_per_batch:
        parser.error("additional trial count must be divisible by --scenes-per-batch")
    print(json.dumps(initialize(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
