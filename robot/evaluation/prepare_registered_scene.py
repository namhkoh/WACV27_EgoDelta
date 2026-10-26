from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot.evaluation import collect_robot_trials as ledger
from robot.evaluation.run_registered_trial import PYTHON, run_logged


def paired_scene_rows(
    session_dir: Path,
    *,
    task: str,
    scene_slot: int,
) -> list[dict[str, str]]:
    _, ledger_path, _ = ledger.session_paths(session_dir)
    rows = [
        row
        for row in ledger.read_ledger(ledger_path)
        if row["task"] == task and int(row["scene_slot"]) == int(scene_slot)
    ]
    if len(rows) != 2 or {row["evaluation_type"] for row in rows} != {"direct", "policy"}:
        raise ValueError(f"scene slot must contain one direct and one policy row; found {len(rows)}")
    if any(row["trial_status"] != "scheduled" for row in rows):
        statuses = {row["trial_id"]: row["trial_status"] for row in rows}
        raise ValueError(f"scene rows must both be scheduled before target preparation: {statuses}")
    return sorted(rows, key=lambda row: int(row["within_scene_order"]))


def find_validation_artifact(root: Path, *, run_label: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in root.rglob(f"*{run_label}*.json"):
        if "target_identity_validation" not in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("method") == "live_realsense_task_frame_identity_validation":
            matches.append((path.resolve(), payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one target-identity artifact under {root}, found {len(matches)}")
    return matches[0]


def verify_resumable_validation(
    validation: dict[str, Any],
    *,
    dataset_name: str,
    run_label: str,
    task_frame: Path,
) -> None:
    if not validation.get("ok", False):
        raise ValueError("stored target-identity validation did not pass")
    if validation.get("dataset_name") != dataset_name:
        raise ValueError("stored target-identity validation dataset mismatch")
    if validation.get("run_label") != run_label:
        raise ValueError("stored target-identity validation run-label mismatch")
    task_frames = validation.get("task_frames")
    if not isinstance(task_frames, list) or len(task_frames) != 1:
        raise ValueError("stored target-identity validation must contain exactly one task frame")
    source = Path(str(task_frames[0].get("source_path", ""))).resolve()
    if source != task_frame:
        raise ValueError("stored target-identity validation task-frame mismatch")


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and bind one paired registered harvest scene without robot motion."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--scene-slot", type=int, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--task-frame-json", type=Path, required=True)
    parser.add_argument("--dataset-name", default="a270c038")
    parser.add_argument("--serial", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse and verify a passed canonical validation capture from an interrupted preparation.",
    )
    args = parser.parse_args()

    if args.scene_slot < 1:
        parser.error("--scene-slot must be positive")
    session_dir = args.session_dir.resolve()
    task_frame = args.task_frame_json.resolve()
    if not task_frame.is_file():
        parser.error(f"--task-frame-json does not exist: {task_frame}")
    rows = paired_scene_rows(session_dir, task="harvest", scene_slot=args.scene_slot)

    prep_dir = session_dir / "scene_preparation" / args.scene_id
    if prep_dir.exists() and any(prep_dir.rglob("*")) and not args.resume:
        raise FileExistsError(f"scene preparation directory is not empty: {prep_dir}")
    prep_dir.mkdir(parents=True, exist_ok=True)
    log_path = prep_dir / "console.log"
    capture_root = prep_dir / "captures"
    run_label = f"{args.scene_id}_{args.target_name}"

    validator = [
        str(PYTHON),
        str(REPO_ROOT / "robot" / "kinematics" / "apps" / "validate_live_target_identity.py"),
        "--dataset-name",
        args.dataset_name,
        "--task-frame-jsons",
        str(task_frame),
        "--runs-root",
        str(capture_root),
        "--run-label",
        run_label,
    ]
    if args.serial:
        validator.extend(("--serial", args.serial))
    result_returncode = 0
    if not args.resume:
        result_returncode = run_logged(validator, log_path).returncode
    validation_path, validation = find_validation_artifact(capture_root, run_label=run_label)
    if args.resume:
        verify_resumable_validation(
            validation,
            dataset_name=args.dataset_name,
            run_label=run_label,
            task_frame=task_frame,
        )
    if result_returncode != 0 or not validation.get("ok", False):
        summary = {
            "schema": "egodelta_registered_scene_preparation_v1",
            "created_from_commit": git_commit(),
            "scene_id": args.scene_id,
            "scene_slot": args.scene_slot,
            "target_name": args.target_name,
            "validation_artifact": str(validation_path),
            "validation_ok": False,
            "bound": False,
        }
        ledger.write_json(prep_dir / "scene_preparation.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    frozen_target = (
        session_dir
        / "targets"
        / f"{args.scene_id}_{args.target_id}_identity.json"
    )
    freeze = run_logged(
        (
            str(PYTHON),
            str(REPO_ROOT / "robot" / "evaluation" / "freeze_identity_target.py"),
            "--source-json",
            str(validation_path),
            "--output-json",
            str(frozen_target),
            "--task",
            "harvest",
            "--scene-id",
            args.scene_id,
            "--target-id",
            args.target_id,
            "--target-name",
            args.target_name,
        ),
        log_path,
    )
    if freeze.returncode != 0 or not frozen_target.is_file():
        raise RuntimeError("failed to freeze the passed identity target")

    for row in rows:
        ledger.bind(
            argparse.Namespace(
                session_dir=session_dir,
                trial_id=row["trial_id"],
                scene_id=args.scene_id,
                target_id=args.target_id,
                target_artifact=frozen_target,
                date=None,
            )
        )
    for row in rows:
        command = run_logged(
            (
                str(PYTHON),
                str(REPO_ROOT / "robot" / "evaluation" / "prepare_trial_commands.py"),
                "--session-dir",
                str(session_dir),
                "--trial-id",
                row["trial_id"],
            ),
            log_path,
        )
        if command.returncode != 0:
            raise RuntimeError(f"failed to prepare command manifest for {row['trial_id']}")

    summary = {
        "schema": "egodelta_registered_scene_preparation_v1",
        "created_from_commit": git_commit(),
        "scene_id": args.scene_id,
        "scene_slot": args.scene_slot,
        "target_id": args.target_id,
        "target_name": args.target_name,
        "task_frame_json": str(task_frame),
        "task_frame_sha256": ledger.sha256_file(task_frame),
        "validation_artifact": str(validation_path),
        "validation_sha256": ledger.sha256_file(validation_path),
        "frozen_target": str(frozen_target),
        "frozen_target_sha256": ledger.sha256_file(frozen_target),
        "bound_trial_ids": [row["trial_id"] for row in rows],
        "validation_ok": True,
        "bound": True,
        "robot_motion": False,
    }
    ledger.write_json(prep_dir / "scene_preparation.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())