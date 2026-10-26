from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/path/to/user/miniconda3/envs/egodelta_robot/python.exe")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trial(session_dir: Path, trial_id: str) -> dict[str, str]:
    ledger = session_dir / "robot_trial_ledger.csv"
    with ledger.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["trial_id"] == trial_id]
    if len(rows) != 1:
        raise ValueError(f"expected one ledger row for {trial_id!r}, found {len(rows)}")
    row = rows[0]
    if row["trial_status"] != "bound":
        raise ValueError(f"trial {trial_id} must have a frozen target before command preparation")
    return row


def load_target(row: dict[str, str]) -> tuple[Path, list[float], Path, dict[str, Any]]:
    target = Path(row["target_artifact_path"])
    provenance = target.with_suffix(target.suffix + ".provenance.json")
    if not target.is_file() or not provenance.is_file():
        raise FileNotFoundError("frozen target JSON and provenance sidecar are both required")
    if sha256_file(target) != row["target_artifact_sha256"]:
        raise ValueError("frozen target hash no longer matches the registered ledger")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    if record.get("frozen_sha256") != row["target_artifact_sha256"]:
        raise ValueError("target provenance hash does not match the registered ledger")
    if record.get("scene_id") != row["scene_id"] or record.get("target_id") != row["target_id"]:
        raise ValueError("target provenance scene/target identity does not match the ledger")
    point = [float(value) for value in record["point_robot_m"]]
    return target, point, provenance, record


def harvest_planner_argv(
    *,
    trial_id: str,
    target: Path,
    point: list[float],
) -> list[str]:
    return [
        str(PYTHON),
        str(REPO_ROOT / "robot" / "kinematics" / "apps" / "plan_target_conditioned_harvest.py"),
        "--goal-robot-m", ",".join(f"{value:.9f}" for value in point),
        "--target-artifact", str(target),
        "--extra-insertion-m", "0.0",
        "--height-offset-m", "0.0",
        "--minimum-clearance-m", "0.012",
        "--path-samples-per-segment", "20",
        "--run-label", trial_id,
    ]


def policy_commands(
    row: dict[str, str],
    target: Path,
    point: list[float],
    session_dir: Path,
) -> dict[str, Any]:
    trial_id = row["trial_id"]
    output_dir = session_dir / "execution_artifacts" / row["batch_id"]
    base = [
        str(PYTHON),
        str(REPO_ROOT / "robot" / "policy" / "guarded_bc_rollout.py"),
        "--policy-dir", row["policy_dir"],
        "--target-source", "goal",
        "--goal-robot-m", ",".join(f"{value:.9f}" for value in point),
        "--no-snap-goal-to-nearest-demo",
        "--require-home",
        "--full-action",
        "--full-action-duration-scale", "1.0",
        "--extra-recede-iterations", "0",
        "--clip-step-delta",
        "--clip-joint-limits",
        "--inter-step-sleep-s", "0.0",
        "--run-label", trial_id,
    ]
    if row["task"] == "harvest":
        base += [
            "--harvest-gripper-schedule",
            "--harvest-fixed-basket-tail-plan", "<PHASE_PLANNER_OUTPUT_JSON>",
            "--harvest-policy-handoff-plan-margin-m", "0.05",
            "--harvest-fixed-basket-tail-start-after", "begin",
            "--no-harvest-fixed-basket-tail-require-closed",
            "--no-harvest-grasp-retarget",
            "--harvest-fixed-basket-tail-minimum-time-s", "4.0",
            "--harvest-tail-velocity-limit-rad-s", "0.15",
            "--harvest-tail-acceleration-limit-rad-s2", "0.08",
            "--execute-prefix-mode", "endpoint",
            "--steps-to-execute", "30",
            "--guard-horizon-steps", "30",
            "--minimum-time-s", "3.0",
            "--policy-velocity-limit-rad-s", "0.15",
            "--policy-acceleration-limit-rad-s2", "0.08",
            "--max-step-delta-rad", "0.025",
            "--joint-limit-margin-rad", "0.003",
            "--min-clearance-m", "0.012",
            "--path-samples-per-step", "3",
            "--max-z-m", "1.45",
            "--harvest-gripper-close-distance-m", "0.09",
            "--harvest-gripper-close-phase", "0.45",
            "--harvest-gripper-phase-close-max-distance-m", "0.12",
        ]
    else:
        base += [
            "--steps-to-execute", "3",
            "--guard-horizon-steps", "10",
            "--minimum-time-s", "0.10",
            "--max-step-delta-rad", "0.017",
            "--max-z-m", "1.35",
        ]
    preflight = base + ["--output-json", str(output_dir / f"{trial_id}_preflight.json")]
    execute = base + ["--output-json", str(output_dir / f"{trial_id}_execution.json"), "--execute"]
    commands = {
        "preflight": {"enabled": True, "commands_robot": False, "argv": preflight},
        "execute": {"enabled": False, "commands_robot": True, "argv": execute},
    }
    if row["task"] == "harvest":
        commands["phase_plan_and_preflight"] = {
            "enabled": True,
            "commands_robot": False,
            "argv": harvest_planner_argv(
                trial_id=trial_id,
                target=target,
                point=point,
            ),
        }
    return commands


def direct_commands(
    row: dict[str, str],
    target: Path,
    point: list[float],
    session_dir: Path,
    *,
    target_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trial_id = row["trial_id"]
    output_dir = session_dir / "execution_artifacts" / row["batch_id"]
    if row["task"] == "harvest":
        structural_input: dict[str, str] | None = None
        if target_provenance and target_provenance.get("schema") == "egodelta_frozen_identity_target_provenance_v1":
            planner_target = Path(str(target_provenance["task_frame_json"])).resolve()
            if not planner_target.is_file():
                raise FileNotFoundError(f"identity target task frame does not exist: {planner_target}")
            task_frame_hash = sha256_file(planner_target)
            expected_hash = target_provenance.get("task_frame_sha256")
            if expected_hash is not None and task_frame_hash != expected_hash:
                raise ValueError("identity target task-frame hash no longer matches its frozen provenance")
            structural_input = {
                "path": str(planner_target),
                "sha256": task_frame_hash,
            }
        planner = harvest_planner_argv(
            trial_id=trial_id,
            target=target,
            point=point,
        )
        execution_template = [
            str(PYTHON),
            str(REPO_ROOT / "robot" / "kinematics" / "apps" / "execute_harvest_plan.py"),
            "<PLANNER_OUTPUT_JSON>",
            "--use-gripper",
            "--minimum-time-s", "4.0",
            "--velocity-limit-rad-s", "0.15",
            "--acceleration-limit-rad-s2", "0.08",
            "--output-json", str(output_dir / f"{trial_id}_execution.json"),
            "--execute",
        ]
        commands: dict[str, Any] = {
            "plan_and_preflight": {"enabled": True, "commands_robot": False, "argv": planner},
            "execute": {
                "enabled": False,
                "commands_robot": True,
                "requires": "Replace <PLANNER_OUTPUT_JSON> with the passing full-tail planner artifact.",
                "argv": execution_template,
            },
        }
        if structural_input is not None:
            commands["structural_task_frame"] = structural_input
        return commands
    base = [
        str(PYTHON),
        str(REPO_ROOT / "robot" / "kinematics" / "apps" / "deleaf_click_cut.py"),
        "--deleaf-config", str(REPO_ROOT / "robot" / "kinematics" / "configs" / "deleaf_right_knife_default.json"),
        "--home", str(REPO_ROOT / "robot" / "utils" / "deleaf_home_right_tucked.json"),
        "--manifest", str(REPO_ROOT / "robot" / "kinematics" / "runs" / "deleaf_0d058720_6cuts_manifest.json"),
        "--alignment", str(
            REPO_ROOT
            / "robot"
            / "calibration"
            / "session"
            / "0d058720"
            / "T_robot_from_aria_manual_world_direct.json"
        ),
        "--hand-tracking",
        "C:/path/to/user/aria/handtrack/mps_0d058720-9302-4563-8764-3a30da628444_vrs/"
        "hand_tracking/hand_tracking_results.csv",
        "--slam-trajectory",
        "C:/path/to/user/aria/handtrack/mps_0d058720-9302-4563-8764-3a30da628444_vrs/"
        "slam/closed_loop_trajectory.csv",
        "--waypoint-policy", "teleop_offset",
        "--teleop-prior-json", str(
            REPO_ROOT
            / "robot"
            / "kinematics"
            / "runs"
            / "060326"
            / "deleaf_teleop_prior"
            / "a270c038_AD1_teleop_prior.json"
        ),
        "--teleop-prior-source-match", "AD1",
        "--toolfit", str(
            REPO_ROOT
            / "robot"
            / "calibration"
            / "tooltips"
            / "a270c038_right_knife_tip_edge_repivot_060226.json"
        ),
        "--selected-point-robot-m", ",".join(f"{value:.9f}" for value in point),
        "--target-extension-values-m", "0",
        "--approach-stroke-standoff-m", "0.002",
        "--pre-cut-standoff-m", "0.08",
        "--pre-cut-time-s", "0.5",
        "--cut-time-s", "0.04",
        "--right-velocity-limit-rad-s", "4.0",
        "--right-acceleration-limit-rad-s2", "18.0",
        "--run-label", trial_id,
        "--execute-until", "return_transit",
        "--output-json", str(output_dir / f"{trial_id}_preflight.json"),
    ]
    return {
        "plan_and_preflight": {"enabled": True, "commands_robot": False, "argv": base},
        "execute": {
            "enabled": False,
            "commands_robot": True,
            "argv": base + ["--no-execute-key-confirm", "--execute"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, but never execute, one registered robot trial command set.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    args = parser.parse_args()
    session_dir = args.session_dir.resolve()
    row = read_trial(session_dir, args.trial_id)
    target, point, provenance, target_provenance = load_target(row)
    commands = (
        policy_commands(row, target, point, session_dir)
        if row["evaluation_type"] == "policy"
        else direct_commands(
            row,
            target,
            point,
            session_dir,
            target_provenance=target_provenance,
        )
    )
    payload = {
        "schema": "egodelta_robot_trial_commands_v1",
        "trial": row,
        "target_provenance": str(provenance),
        "commands": commands,
        "execution_locked": True,
        "note": "This manifest does not execute subprocesses. Execute stages stay disabled until operator review.",
    }
    output = session_dir / "command_manifests" / row["batch_id"] / f"{row['trial_id']}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite command manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"command_manifest": str(output), "execution_locked": True}, indent=2))


if __name__ == "__main__":
    main()
