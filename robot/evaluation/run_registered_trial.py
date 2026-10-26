from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot.evaluation import collect_robot_trials as ledger
from robot.server.gripper_status import (
    DEFAULT_GRIPPER_URL,
    command_targets,
    fetch_status,
    normalized_status,
)


PYTHON = Path("C:/path/to/user/miniconda3/envs/egodelta_robot/python.exe")
OUTCOMES = (
    "success",
    "physical_contact_failure",
    "gripper_failure",
    "safety_stop",
    "human_intervention",
    "execution_command_failure",
    "other",
)


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    output: str


class InfrastructureUnavailable(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return ledger.sha256_file(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def command_manifest_path(session_dir: Path, row: dict[str, str]) -> Path:
    return session_dir / "command_manifests" / row["batch_id"] / f"{row['trial_id']}.json"


def run_directory(session_dir: Path, row: dict[str, str]) -> Path:
    return session_dir / "trial_runs" / row["batch_id"] / row["trial_id"]


def read_trial(session_dir: Path, trial_id: str | None, batch_id: str | None) -> dict[str, str]:
    _, ledger_path, _ = ledger.session_paths(session_dir)
    rows = ledger.read_ledger(ledger_path)
    if trial_id is not None:
        row = ledger.find_trial(rows, trial_id)
    else:
        candidates = [
            item
            for item in rows
            if item["trial_status"] != "completed"
            and (batch_id is None or item["batch_id"] == batch_id)
        ]
        candidates.sort(key=lambda item: int(item["schedule_index"]))
        if not candidates:
            raise RuntimeError("no incomplete registered trial remains")
        row = candidates[0]
    if row["trial_status"] != "bound":
        raise RuntimeError(
            f"next trial {row['trial_id']} is {row['trial_status']}; prepare and bind its scene target first"
        )
    return row


def replace_option(argv: Sequence[str], flag: str, value: str) -> list[str]:
    result = list(argv)
    indices = [index for index, token in enumerate(result) if token == flag]
    if len(indices) > 1:
        raise ValueError(f"command contains duplicate option {flag}")
    if indices:
        index = indices[0]
        if index + 1 >= len(result):
            raise ValueError(f"command option has no value: {flag}")
        result[index + 1] = value
    else:
        result.extend((flag, value))
    return result


def remove_flag(argv: Sequence[str], flag: str) -> list[str]:
    result = list(argv)
    if result.count(flag) > 1:
        raise ValueError(f"command contains duplicate flag {flag}")
    return [token for token in result if token != flag]


def replace_token(argv: Sequence[str], old: str, new: str) -> list[str]:
    result = list(argv)
    if result.count(old) != 1:
        raise ValueError(f"expected exactly one {old!r} placeholder")
    return [new if token == old else token for token in result]


def option_value(argv: Sequence[str], flag: str) -> str:
    result = list(argv)
    if result.count(flag) != 1:
        raise ValueError(f"expected exactly one {flag} option")
    index = result.index(flag)
    if index + 1 >= len(result):
        raise ValueError(f"command option has no value: {flag}")
    return result[index + 1]


def run_logged(argv: Sequence[str], log_path: Path) -> ProcessResult:
    command = [str(token) for token in argv]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as log:
        header = f"\n[{utc_now()}] $ {subprocess.list2cmdline(command)}\n"
        print(header.rstrip(), flush=True)
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
        returncode = int(process.wait())
        trailer = f"[returncode={returncode}]\n"
        print(trailer.rstrip(), flush=True)
        log.write(trailer)
    return ProcessResult(tuple(command), returncode, "".join(lines))


def connectivity_failure(result: ProcessResult) -> bool:
    text = result.output.lower()
    return (
        "failed to connect to rb-y1" in text
        or "failed to connect to rb_y1" in text
        or "connection refused" in text
        or "failed to connect to all addresses" in text
    )


def run_read_only_with_connectivity_retry(
    argv: Sequence[str],
    log_path: Path,
    *,
    attempts: int,
) -> ProcessResult:
    if attempts < 1:
        raise ValueError("connectivity attempts must be positive")
    result: ProcessResult | None = None
    for attempt in range(1, attempts + 1):
        result = run_logged(argv, log_path)
        if result.returncode == 0 or not connectivity_failure(result):
            return result
        if attempt < attempts:
            message = f"read-only connectivity retry {attempt}/{attempts}\n"
            print(message.rstrip(), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message)
            time.sleep(float(attempt))
    assert result is not None
    raise InfrastructureUnavailable(
        f"RB-Y1 SDK remained unreachable after {attempts} read-only attempts"
    )


def ensure_command_manifest(
    session_dir: Path,
    row: dict[str, str],
    log_path: Path,
) -> tuple[Path, dict[str, Any]]:
    path = command_manifest_path(session_dir, row)
    if not path.is_file():
        result = run_logged(
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
        if result.returncode != 0 or not path.is_file():
            raise RuntimeError("failed to generate the locked command manifest")
    payload = read_json(path)
    trial = payload.get("trial")
    if not isinstance(trial, dict) or trial.get("trial_id") != row["trial_id"]:
        raise ValueError("command manifest trial identity does not match the ledger")
    if trial.get("target_artifact_sha256") != row["target_artifact_sha256"]:
        raise ValueError("command manifest target hash does not match the ledger")
    if not payload.get("execution_locked", False):
        raise ValueError("command manifest must remain execution_locked")
    return path, payload


def verify_target(row: dict[str, str]) -> Path:
    target = Path(row["target_artifact_path"]).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if sha256_file(target) != row["target_artifact_sha256"]:
        raise ValueError("registered target artifact hash mismatch")
    return target


def critical_controller_paths(row: dict[str, str]) -> tuple[Path, ...]:
    common = (
        REPO_ROOT / "robot" / "evaluation" / "run_registered_trial.py",
        REPO_ROOT / "robot" / "evaluation" / "collect_robot_trials.py",
        REPO_ROOT / "robot" / "evaluation" / "prepare_trial_commands.py",
        REPO_ROOT / "robot" / "server" / "gripper_status.py",
        REPO_ROOT / "robot" / "kinematics" / "utils" / "realsense_click_move_arm.py",
    )
    if row["evaluation_type"] == "direct" and row["task"] == "harvest":
        return common + (
            REPO_ROOT / "robot" / "kinematics" / "harvest_primitive.py",
            REPO_ROOT / "robot" / "kinematics" / "ik_solver.py",
            REPO_ROOT / "robot" / "kinematics" / "apps" / "plan_target_conditioned_harvest.py",
            REPO_ROOT / "robot" / "kinematics" / "apps" / "execute_harvest_plan.py",
            REPO_ROOT / "robot" / "calibration" / "tooltips" / "a270c038_left_gripper_jaw_center_01.json",
            REPO_ROOT
            / "robot"
            / "policy"
            / "policy_action_library"
            / "AT1T1"
            / "success_current"
            / "actions"
            / "AT1T1_instance_001_step_01_AT1T1"
            / "planner"
            / "a270c038_AT1T1_success_current.json",
        )
    if row["evaluation_type"] == "direct" and row["task"] == "deleaf":
        return common + (
            REPO_ROOT / "robot" / "kinematics" / "apps" / "deleaf_click_cut.py",
            REPO_ROOT / "robot" / "kinematics" / "utils" / "realsense_click_move_arm.py",
            REPO_ROOT / "robot" / "kinematics" / "configs" / "deleaf_right_knife_default.json",
            REPO_ROOT / "robot" / "kinematics" / "runs" / "deleaf_0d058720_6cuts_manifest.json",
            REPO_ROOT
            / "robot"
            / "kinematics"
            / "runs"
            / "060326"
            / "deleaf_teleop_prior"
            / "a270c038_AD1_teleop_prior.json",
            REPO_ROOT
            / "robot"
            / "calibration"
            / "session"
            / "0d058720"
            / "T_robot_from_aria_manual_world_direct.json",
            REPO_ROOT
            / "robot"
            / "calibration"
            / "tooltips"
            / "a270c038_right_knife_tip_edge_repivot_060226.json",
        )
    if row["evaluation_type"] == "policy":
        paths = common + (REPO_ROOT / "robot" / "policy" / "guarded_bc_rollout.py",)
        if row["task"] == "harvest":
            paths += (
                REPO_ROOT / "robot" / "kinematics" / "harvest_primitive.py",
                REPO_ROOT / "robot" / "kinematics" / "ik_solver.py",
                REPO_ROOT / "robot" / "kinematics" / "apps" / "plan_target_conditioned_harvest.py",
                REPO_ROOT
                / "robot"
                / "policy"
                / "policy_action_library"
                / "AT1T1"
                / "success_current"
                / "actions"
                / "AT1T1_instance_001_step_01_AT1T1"
                / "planner"
                / "a270c038_AT1T1_success_current.json",
            )
        return paths
    return common


def controller_provenance(row: dict[str, str], *, allow_dirty: bool) -> dict[str, Any]:
    paths = critical_controller_paths(row)
    relative = [str(path.relative_to(REPO_ROOT)) for path in paths]
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *relative],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status and not allow_dirty:
        raise RuntimeError(
            "critical controller files are not committed; commit them before a registered run:\n"
            + status
        )
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "registered_source_commit": row["source_commit"],
        "critical_status": status or "clean",
        "files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in paths
        },
    }


def default_home_path(row: dict[str, str]) -> Path:
    if row["task"] == "deleaf":
        return REPO_ROOT / "robot" / "utils" / "deleaf_home_right_tucked.json"
    return REPO_ROOT / "robot" / "utils" / "homestate.json"

def robot_home_readiness(
    *,
    address: str,
    model: str,
    home_path: Path,
    tolerance_deg: float,
    attempts: int,
) -> dict[str, Any]:
    import numpy as np
    import rby1_sdk as rby

    from robot.kinematics.utils.realsense_click_move_arm import validate_home_state

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            robot = rby.create_robot(address, model)
            if not robot.connect():
                raise InfrastructureUnavailable(f"failed to connect to RB-Y1 at {address}")
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
                continue
            raise InfrastructureUnavailable(
                f"RB-Y1 readiness connection failed after {attempts} attempts: {exc}"
            ) from exc

        home = read_json(home_path)
        report = validate_home_state(
            robot,
            home,
            tolerance_rad=float(np.deg2rad(tolerance_deg)),
        )
        return {
            "ok": True,
            "address": address,
            "model": model,
            "home": str(home_path),
            "home_tolerance_deg": float(tolerance_deg),
            "attempt": attempt,
            "home_check": report,
        }
    raise InfrastructureUnavailable(str(last_error or "RB-Y1 readiness failed"))


def gripper_preflight(base_url: str, *, max_start_openness: float) -> dict[str, Any]:
    raw = fetch_status(base_url, timeout_s=2.0)
    status = normalized_status(raw)
    left = status["grippers"].get("left")
    if left is None:
        raise RuntimeError("live gripper status has no left motor")
    if not left["within_calibration"]:
        raise RuntimeError(f"left gripper encoder is outside calibration: {left}")
    if float(left["openness"]) > float(max_start_openness):
        raise RuntimeError(
            "left gripper must be physically closed at trial start: "
            f"openness={left['openness']:.4f} limit={max_start_openness:.4f}"
        )
    targets = command_targets(raw, arm="left")
    return {
        "url": base_url.rstrip("/"),
        "normalized_status": status,
        "left_command_targets": {
            "open_position": targets.open_position,
            "close_position": targets.close_position,
            "encoder_span": targets.encoder_span,
            "calibration_schema": targets.calibration_schema,
        },
    }


def planner_artifact(root: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in root.rglob("*.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload.get("hard_gate"), dict) and "candidates" in payload:
            matches.append((path.resolve(), payload))
    if len(matches) != 1:
        raise RuntimeError(f"expected one harvest planner artifact under {root}, found {len(matches)}")
    return matches[0]


def direct_preflight(
    manifest: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    gripper_url: str,
    connectivity_attempts: int,
) -> tuple[Path, dict[str, Any], Path]:
    planner_parent = run_dir / "planner"
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    planner_root: Path | None = None
    planner_argv_base = list(manifest["commands"]["plan_and_preflight"]["argv"])
    for attempt in range(1, connectivity_attempts + 1):
        attempt_root = planner_parent / f"{invocation}_attempt_{attempt:02d}"
        planner_argv = replace_option(planner_argv_base, "--runs-root", str(attempt_root))
        result = run_logged(planner_argv, log_path)
        if result.returncode == 0:
            planner_root = attempt_root
            break
        if not connectivity_failure(result):
            raise RuntimeError(f"harvest planner process failed with return code {result.returncode}")
        if attempt < connectivity_attempts:
            message = f"read-only connectivity retry {attempt}/{connectivity_attempts}\n"
            print(message.rstrip(), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message)
            time.sleep(float(attempt))
    if planner_root is None:
        raise InfrastructureUnavailable(
            f"RB-Y1 SDK remained unreachable after {connectivity_attempts} read-only attempts"
        )
    plan_path, plan = planner_artifact(planner_root)
    if not plan.get("hard_gate", {}).get("ok", False):
        return plan_path, plan, Path()

    home_output = run_dir / "home_preflight.json"
    execute_argv = list(manifest["commands"]["execute"]["argv"])
    execute_argv = replace_token(execute_argv, "<PLANNER_OUTPUT_JSON>", str(plan_path))
    execute_argv = remove_flag(execute_argv, "--execute")
    execute_argv = replace_option(execute_argv, "--output-json", str(home_output))
    execute_argv = replace_option(execute_argv, "--gripper-url", gripper_url)
    execute_argv = replace_option(execute_argv, "--gripper-target-source", "live_status")
    execute_argv.append("--check-home-only")
    result = run_read_only_with_connectivity_retry(
        execute_argv,
        log_path,
        attempts=connectivity_attempts,
    )
    if result.returncode != 0 or not home_output.is_file():
        raise RuntimeError("direct home preflight failed")
    return plan_path, plan, home_output


def policy_preflight(
    manifest: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    gripper_url: str,
    connectivity_attempts: int,
) -> tuple[Path, dict[str, Any]]:
    output = run_dir / "policy_preflight.json"
    argv = list(manifest["commands"]["preflight"]["argv"])
    argv = replace_option(argv, "--output-json", str(output))
    argv = replace_option(argv, "--gripper-url", gripper_url)
    argv = replace_option(argv, "--gripper-target-source", "live_status")
    result = run_read_only_with_connectivity_retry(
        argv,
        log_path,
        attempts=connectivity_attempts,
    )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError("policy preflight process failed")
    payload = read_json(output)
    if not payload.get("guards", {}).get("ok", False):
        return output, payload
    home = payload.get("home_check") or payload.get("locked_home_check")
    if not isinstance(home, dict) or not home.get("ok", False):
        raise RuntimeError("policy preflight did not produce a passing home check")
    return output, payload


def policy_harvest_phase_preflight(
    manifest: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    connectivity_attempts: int,
) -> tuple[Path, dict[str, Any]]:
    planner_parent = run_dir / "policy_phase_planner"
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    planner_root: Path | None = None
    planner_argv_base = list(manifest["commands"]["phase_plan_and_preflight"]["argv"])
    for attempt in range(1, connectivity_attempts + 1):
        attempt_root = planner_parent / f"{invocation}_attempt_{attempt:02d}"
        planner_argv = replace_option(planner_argv_base, "--runs-root", str(attempt_root))
        result = run_logged(planner_argv, log_path)
        if result.returncode == 0:
            planner_root = attempt_root
            break
        if not connectivity_failure(result):
            raise RuntimeError(
                f"harvest policy phase planner failed with return code {result.returncode}"
            )
        if attempt < connectivity_attempts:
            message = f"read-only connectivity retry {attempt}/{connectivity_attempts}\n"
            print(message.rstrip(), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message)
            time.sleep(float(attempt))
    if planner_root is None:
        raise InfrastructureUnavailable(
            f"RB-Y1 SDK remained unreachable after {connectivity_attempts} read-only attempts"
        )
    return planner_artifact(planner_root)


def direct_deleaf_preflight(
    manifest: dict[str, Any],
    run_dir: Path,
    log_path: Path,
    connectivity_attempts: int,
) -> tuple[Path, dict[str, Any]]:
    output = run_dir / "deleaf_preflight.json"
    argv = list(manifest["commands"]["plan_and_preflight"]["argv"])
    argv = replace_option(argv, "--output-json", str(output))
    result = run_read_only_with_connectivity_retry(argv, log_path, attempts=connectivity_attempts)
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError("deleaf planner preflight failed")
    payload = read_json(output)
    if payload.get("method") != "deleaf_click_cut_mps_guided_ik":
        raise RuntimeError("deleaf preflight produced an unexpected artifact type")
    home = payload.get("home_check")
    if not isinstance(home, dict) or not home.get("ok", False):
        raise RuntimeError("deleaf preflight did not produce a passing home check")
    return output, payload

def execution_argv(
    row: dict[str, str],
    manifest: dict[str, Any],
    run_dir: Path,
    gripper_url: str,
    plan_path: Path | None,
) -> tuple[list[str], Path]:
    output = run_dir / "execution.json"
    argv = list(manifest["commands"]["execute"]["argv"])
    if row["evaluation_type"] == "direct" and row["task"] == "harvest":
        if plan_path is None:
            raise ValueError("direct harvest execution requires a planner artifact")
        argv = replace_token(argv, "<PLANNER_OUTPUT_JSON>", str(plan_path))
    if row["evaluation_type"] == "policy" and row["task"] == "harvest":
        if plan_path is None:
            raise ValueError("harvest policy execution requires a phase planner artifact")
        argv = replace_token(argv, "<PHASE_PLANNER_OUTPUT_JSON>", str(plan_path))
    argv = replace_option(argv, "--output-json", str(output))
    if row["task"] == "harvest":
        argv = replace_option(argv, "--gripper-url", gripper_url)
        argv = replace_option(argv, "--gripper-target-source", "live_status")
    if "--execute" not in argv:
        argv.append("--execute")
    return argv, output


def annotate(
    session_dir: Path,
    row: dict[str, str],
    *,
    plan_generated: bool,
    automatic_gate_rejected: bool,
    operator_rejected: bool,
    execution_started: bool,
    operator_stop: bool,
    success: bool,
    failure_category: str,
    failure_description: str,
    raw_log: Path,
    video: Path | None,
    execution_artifact: Path | None,
    started_at_utc: str,
) -> None:
    ledger.annotate(
        SimpleNamespace(
            session_dir=session_dir,
            trial_id=row["trial_id"],
            plan_generated=str(bool(plan_generated)).lower(),
            automatic_gate_rejected=str(bool(automatic_gate_rejected)).lower(),
            operator_rejected_before_execution=str(bool(operator_rejected)).lower(),
            execution_started=str(bool(execution_started)).lower(),
            operator_or_safety_stop=str(bool(operator_stop)).lower(),
            success=str(bool(success)).lower(),
            failure_category=failure_category,
            failure_description=failure_description,
            raw_log=raw_log,
            video=video,
            execution_artifact=execution_artifact,
            started_at_utc=started_at_utc,
            completed_at_utc=utc_now(),
        )
    )


def prompt_outcome() -> str:
    print("\nPhysical outcome is required; controller return code alone is insufficient.")
    print("Choices: " + ", ".join(OUTCOMES))
    while True:
        value = input("outcome> ").strip().lower()
        if value in OUTCOMES:
            return value
        print("Enter one of the listed outcome names.")


def classify_execution_failure(
    row: dict[str, str],
    returncode: int,
    payload: dict[str, Any] | None,
) -> tuple[str, bool, bool, str]:
    executed = bool(payload and payload.get("executed"))
    execution_error = None if payload is None else payload.get("execution_error")
    if row["evaluation_type"] == "policy" and execution_error == "guard_failed" and not executed:
        return "automatic_gate_rejection", True, False, "policy runtime guard rejected before motion"
    if execution_error == "gripper_failed" or returncode == 3:
        return "gripper_failure", False, executed, "gripper command or verification failed"
    if execution_error in {
        "fixed_tail_failed",
        "fixed_tail_skipped_gripper_not_closed",
        "fixed_tail_skipped_handoff_not_triggered",
    }:
        return "execution_command_failure", False, True, str(execution_error)
    return "execution_command_failure", False, True, f"controller returned {returncode}"


def validate_policy_execution_contract(
    row: dict[str, str],
    payload: dict[str, Any] | None,
    expected_plan_path: Path | None = None,
) -> str | None:
    """Require both learned approach and the complete contact-critical phase tail."""
    if row.get("evaluation_type") != "policy" or row.get("task") != "harvest":
        return None
    if not isinstance(payload, dict):
        return "harvest policy execution did not write a readable artifact"
    if payload.get("controller_execution_mode") != "hybrid_policy_and_fixed_tail":
        return (
            "harvest policy must execute learned approach and deterministic phase tail; "
            f"observed mode={payload.get('controller_execution_mode')!r}"
        )
    if not payload.get("policy_motion_executed", False):
        return "harvest policy trial executed no learned-policy motion"
    handoff = payload.get("policy_handoff", {})
    if not handoff.get("triggered", False):
        return "harvest policy did not reach the pre-contact phase handoff"
    if handoff.get("source") != "phase_plan_approach":
        return "harvest policy handoff was not derived from the solved approach phase"
    tail = payload.get("fixed_basket_tail")
    if not isinstance(tail, dict) or not tail.get("ok", False):
        return "harvest deterministic phase tail did not complete successfully"
    if not tail.get("live_transition_hard_gate", {}).get("ok", False):
        return "harvest live handoff-to-approach path did not pass its hard gate"
    if expected_plan_path is not None:
        observed_plan = Path(str(tail.get("plan", ""))).resolve()
        if observed_plan != expected_plan_path.resolve():
            return (
                "harvest execution used a different phase plan than preflight: "
                f"expected={expected_plan_path.resolve()}, observed={observed_plan}"
            )
    if tail.get("start_after") != "begin":
        return f"harvest phase tail started after {tail.get('start_after')!r}, expected 'begin'"
    expected = [
        "approach",
        "orient_soft_03",
        "orient_soft_07",
        "pre_grasp",
        "grasp",
        "harvest",
        "retract",
        "basket_place",
    ]
    observed = [str(item.get("name")) for item in tail.get("waypoints", [])]
    if observed != expected:
        return f"harvest phase sequence mismatch: expected={expected}, observed={observed}"
    expected_gripper = ["pre_grasp_open", "grasp_close", "basket_release_open"]
    observed_gripper = [str(item.get("action")) for item in tail.get("gripper_events", [])]
    if observed_gripper != expected_gripper:
        return (
            "harvest phase gripper sequence mismatch: "
            f"expected={expected_gripper}, observed={observed_gripper}"
        )
    return None


def preflight_state_path(run_dir: Path) -> Path:
    return run_dir / "preflight_state.json"


def write_preflight_state(
    path: Path,
    *,
    row: dict[str, str],
    target: Path,
    manifest_path: Path,
    controller: dict[str, Any],
    gripper: dict[str, Any],
    preflight_artifact: Path,
    plan_path: Path | None,
) -> None:
    payload = {
        "schema": "egodelta_registered_trial_preflight_v1",
        "created_at_utc": utc_now(),
        "trial_id": row["trial_id"],
        "target_artifact": str(target),
        "target_sha256": sha256_file(target),
        "command_manifest": str(manifest_path),
        "command_manifest_sha256": sha256_file(manifest_path),
        "controller_provenance": controller,
        "gripper_preflight": gripper,
        "preflight_artifact": str(preflight_artifact),
        "preflight_artifact_sha256": sha256_file(preflight_artifact),
        "plan_artifact": None if plan_path is None else str(plan_path),
        "plan_artifact_sha256": None if plan_path is None else sha256_file(plan_path),
        "ok": True,
    }
    ledger.write_json(path, payload)


def load_preflight_state(
    path: Path,
    *,
    row: dict[str, str],
    target: Path,
    manifest_path: Path,
    controller: dict[str, Any],
) -> tuple[dict[str, Any], Path | None, Path]:
    payload = read_json(path)
    if payload.get("schema") != "egodelta_registered_trial_preflight_v1" or not payload.get("ok", False):
        raise ValueError("stored preflight state is not a passing registered preflight")
    if payload.get("trial_id") != row["trial_id"]:
        raise ValueError("stored preflight trial identity mismatch")
    if payload.get("target_sha256") != sha256_file(target):
        raise ValueError("stored preflight target hash mismatch")
    if payload.get("command_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("stored preflight command manifest hash mismatch")
    previous_files = payload.get("controller_provenance", {}).get("files")
    if previous_files != controller.get("files"):
        raise ValueError("controller source hashes changed after the stored preflight")
    preflight_artifact = Path(str(payload["preflight_artifact"])).resolve()
    if not preflight_artifact.is_file():
        raise FileNotFoundError(preflight_artifact)
    if payload.get("preflight_artifact_sha256") != sha256_file(preflight_artifact):
        raise ValueError("stored preflight artifact hash mismatch")
    plan_value = payload.get("plan_artifact")
    plan_path = None if plan_value is None else Path(str(plan_value)).resolve()
    if plan_path is not None:
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        if payload.get("plan_artifact_sha256") != sha256_file(plan_path):
            raise ValueError("stored planner artifact hash mismatch")
    return payload, plan_path, preflight_artifact


def write_gate_rejection(run_dir: Path, *, row: dict[str, str], error: Exception) -> Path:
    path = run_dir / "automatic_gate_rejection.json"
    ledger.write_json(
        path,
        {
            "schema": "egodelta_registered_gate_rejection_v1",
            "created_at_utc": utc_now(),
            "trial_id": row["trial_id"],
            "error_type": type(error).__name__,
            "error": str(error),
            "automatic_gate_rejected": True,
            "execution_started": False,
        },
    )
    return path


def write_readiness(
    run_dir: Path,
    *,
    row: dict[str, str],
    ok: bool,
    robot: dict[str, Any] | None = None,
    gripper: dict[str, Any] | None = None,
    error: Exception | None = None,
    stage: str = "startup",
) -> Path:
    path = run_dir / "readiness.json"
    ledger.write_json(
        path,
        {
            "schema": "egodelta_registered_trial_readiness_v1",
            "created_at_utc": utc_now(),
            "trial_id": row["trial_id"],
            "stage": stage,
            "ok": bool(ok),
            "registered_trial_started": False if not ok else None,
            "ledger_row_consumed": False if not ok else None,
            "robot": robot,
            "gripper": gripper,
            "error_type": None if error is None else type(error).__name__,
            "error": None if error is None else str(error),
        },
    )
    return path

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one failure-preserving registered robot trial with automated evidence capture."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--trial-id")
    parser.add_argument("--batch-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm-trial-id",
        help="Required with --execute and must exactly match the selected registered trial ID.",
    )
    parser.add_argument(
        "--reuse-preflight",
        action="store_true",
        help="Reuse a hash-verified passing preflight state instead of creating a second registered solve.",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--outcome", choices=("prompt", *OUTCOMES), default="prompt")
    parser.add_argument("--failure-description", default="")
    parser.add_argument("--address", default="192.0.2.10:50051")
    parser.add_argument("--model", default="a")
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Override the task-specific registered home state.",
    )
    parser.add_argument("--home-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--connectivity-attempts", type=int, default=3)
    parser.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument("--max-start-openness", type=float, default=0.10)
    parser.add_argument("--allow-dirty-controller", action="store_true")
    args = parser.parse_args()

    if args.max_start_openness < 0.0 or args.max_start_openness > 1.0:
        parser.error("--max-start-openness must be in [0, 1]")
    if args.home_tolerance_deg <= 0.0:
        parser.error("--home-tolerance-deg must be positive")
    if args.connectivity_attempts < 1:
        parser.error("--connectivity-attempts must be positive")
    if args.execute and args.confirm_trial_id is None:
        parser.error("--execute requires --confirm-trial-id")
    if not args.execute and args.confirm_trial_id is not None:
        parser.error("--confirm-trial-id is only valid with --execute")
    if args.video is not None and not args.video.is_file():
        parser.error(f"--video does not exist: {args.video}")

    session_dir = args.session_dir.resolve()
    row = read_trial(session_dir, args.trial_id, args.batch_id)
    if args.execute and args.confirm_trial_id != row["trial_id"]:
        parser.error("--confirm-trial-id must exactly match the selected registered trial")
    target = verify_target(row)
    run_dir = run_directory(session_dir, row)
    state_path = preflight_state_path(run_dir)
    if state_path.exists() and not args.reuse_preflight:
        raise RuntimeError(
            f"preflight state already exists for this row: {state_path}. "
            "Use --reuse-preflight to consume the hash-verified state; do not replan."
        )
    if args.reuse_preflight and not state_path.is_file():
        raise RuntimeError(f"--reuse-preflight requested but no state exists: {state_path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    manifest_path, manifest = ensure_command_manifest(session_dir, row, log_path)
    controller = controller_provenance(row, allow_dirty=args.allow_dirty_controller)
    home_path = args.home.resolve() if args.home is not None else default_home_path(row).resolve()

    robot_readiness: dict[str, Any] | None = None
    gripper: dict[str, Any] | None = None
    try:
        robot_readiness = robot_home_readiness(
            address=args.address,
            model=args.model,
            home_path=home_path,
            tolerance_deg=args.home_tolerance_deg,
            attempts=args.connectivity_attempts,
        )
        gripper = (
            gripper_preflight(
                args.gripper_url,
                max_start_openness=args.max_start_openness,
            )
            if row["task"] == "harvest"
            else {"not_applicable": True, "reason": "deleaf uses a passive right-arm knife"}
        )
        write_readiness(
            run_dir,
            row=row,
            ok=True,
            robot=robot_readiness,
            gripper=gripper,
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] non-counting readiness failure: {type(exc).__name__}: {exc}\n")
        readiness_path = write_readiness(
            run_dir,
            row=row,
            ok=False,
            robot=robot_readiness,
            gripper=gripper,
            error=exc,
        )
        print(
            json.dumps(
                {
                    "trial_id": row["trial_id"],
                    "readiness_ok": False,
                    "registered_trial_started": False,
                    "ledger_row_consumed": False,
                    "readiness_artifact": str(readiness_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        return 10

    assert gripper is not None
    started_at = utc_now()
    plan_path: Path | None = None
    plan_generated = row["evaluation_type"] == "direct"
    if args.reuse_preflight:
        _, plan_path, preflight_artifact = load_preflight_state(
            state_path,
            row=row,
            target=target,
            manifest_path=manifest_path,
            controller=controller,
        )
        plan_generated = plan_path is not None
        ledger.write_json(run_dir / "gripper_pre_execution.json", gripper)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "trial_id": row["trial_id"],
                        "preflight_ok": True,
                        "preflight_reused": True,
                        "robot_motion": False,
                    },
                    indent=2,
                )
            )
            return 0
    else:
        try:
            ledger.write_json(run_dir / "gripper_preflight.json", gripper)

            if row["evaluation_type"] == "direct" and row["task"] == "harvest":
                plan_path, plan, home_artifact = direct_preflight(
                    manifest,
                    run_dir,
                    log_path,
                    args.gripper_url,
                    args.connectivity_attempts,
                )
                if not plan.get("hard_gate", {}).get("ok", False):
                    annotate(
                        session_dir,
                        row,
                        plan_generated=True,
                        automatic_gate_rejected=True,
                        operator_rejected=False,
                        execution_started=False,
                        operator_stop=False,
                        success=False,
                        failure_category="automatic_gate_rejection",
                        failure_description="direct harvest planner hard gate rejected all candidates",
                        raw_log=log_path,
                        video=args.video,
                        execution_artifact=plan_path,
                        started_at_utc=started_at,
                    )
                    return 2
                preflight_artifact = home_artifact
            elif row["evaluation_type"] == "direct" and row["task"] == "deleaf":
                plan_path, plan = direct_deleaf_preflight(
                    manifest,
                    run_dir,
                    log_path,
                    args.connectivity_attempts,
                )
                preflight_artifact = plan_path
                if not plan.get("hard_gate", {}).get("ok", False):
                    annotate(
                        session_dir,
                        row,
                        plan_generated=True,
                        automatic_gate_rejected=True,
                        operator_rejected=False,
                        execution_started=False,
                        operator_stop=False,
                        success=False,
                        failure_category="automatic_gate_rejection",
                        failure_description="direct deleaf planner hard gate rejected the action",
                        raw_log=log_path,
                        video=args.video,
                        execution_artifact=plan_path,
                        started_at_utc=started_at,
                    )
                    return 2
            elif row["evaluation_type"] == "policy":
                preflight_artifact, policy = policy_preflight(
                    manifest,
                    run_dir,
                    log_path,
                    args.gripper_url,
                    args.connectivity_attempts,
                )
                if not policy.get("guards", {}).get("ok", False):
                    annotate(
                        session_dir,
                        row,
                        plan_generated=False,
                        automatic_gate_rejected=True,
                        operator_rejected=False,
                        execution_started=False,
                        operator_stop=False,
                        success=False,
                        failure_category="automatic_gate_rejection",
                        failure_description="policy dry-run guard rejected the predicted action",
                        raw_log=log_path,
                        video=args.video,
                        execution_artifact=preflight_artifact,
                        started_at_utc=started_at,
                    )
                    return 2
                if row["task"] == "harvest":
                    plan_path, phase_plan = policy_harvest_phase_preflight(
                        manifest,
                        run_dir,
                        log_path,
                        args.connectivity_attempts,
                    )
                    plan_generated = True
                    if not phase_plan.get("hard_gate", {}).get("ok", False):
                        annotate(
                            session_dir,
                            row,
                            plan_generated=True,
                            automatic_gate_rejected=True,
                            operator_rejected=False,
                            execution_started=False,
                            operator_stop=False,
                            success=False,
                            failure_category="automatic_gate_rejection",
                            failure_description=(
                                "harvest policy phase planner hard gate rejected the target"
                            ),
                            raw_log=log_path,
                            video=args.video,
                            execution_artifact=plan_path,
                            started_at_utc=started_at,
                        )
                        return 2
            else:
                raise NotImplementedError(
                    f"registered automation does not support task={row['task']} "
                    f"evaluation_type={row['evaluation_type']}"
                )
        except InfrastructureUnavailable as exc:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"[{utc_now()}] non-counting preflight infrastructure failure: "
                    f"{type(exc).__name__}: {exc}\n"
                )
            readiness_path = write_readiness(
                run_dir,
                row=row,
                ok=False,
                robot=robot_readiness,
                gripper=gripper,
                error=exc,
                stage="registered_preflight",
            )
            print(
                json.dumps(
                    {
                        "trial_id": row["trial_id"],
                        "readiness_ok": False,
                        "registered_trial_started": False,
                        "ledger_row_consumed": False,
                        "readiness_artifact": str(readiness_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
            )
            return 10
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utc_now()}] automatic gate rejection: {type(exc).__name__}: {exc}\n")
            rejection = write_gate_rejection(run_dir, row=row, error=exc)
            annotate(
                session_dir,
                row,
                plan_generated=bool(plan_path),
                automatic_gate_rejected=True,
                operator_rejected=False,
                execution_started=False,
                operator_stop=False,
                success=False,
                failure_category="automatic_gate_rejection",
                failure_description=f"{type(exc).__name__}: {exc}",
                raw_log=log_path,
                video=args.video,
                execution_artifact=rejection,
                started_at_utc=started_at,
            )
            return 2

        write_preflight_state(
            state_path,
            row=row,
            target=target,
            manifest_path=manifest_path,
            controller=controller,
            gripper=gripper,
            preflight_artifact=preflight_artifact,
            plan_path=plan_path,
        )
        if not args.execute:
            print(
                json.dumps(
                    {
                        "trial_id": row["trial_id"],
                        "preflight_ok": True,
                        "preflight_state": str(state_path),
                        "robot_motion": False,
                    },
                    indent=2,
                )
            )
            return 0

    argv, execution_output = execution_argv(
        row,
        manifest,
        run_dir,
        args.gripper_url,
        plan_path,
    )
    result = run_logged(argv, log_path)
    payload = read_json(execution_output) if execution_output.is_file() else None
    if result.returncode != 0:
        category, automatic, execution_started, description = classify_execution_failure(
            row,
            result.returncode,
            payload,
        )
        annotate(
            session_dir,
            row,
            plan_generated=plan_generated,
            automatic_gate_rejected=automatic,
            operator_rejected=False,
            execution_started=execution_started,
            operator_stop=category == "safety_stop",
            success=False,
            failure_category=category,
            failure_description=description,
            raw_log=log_path,
            video=args.video,
            execution_artifact=execution_output if execution_output.is_file() else preflight_artifact,
            started_at_utc=started_at,
        )
        return result.returncode

    contract_error = validate_policy_execution_contract(
        row,
        payload,
        expected_plan_path=plan_path,
    )
    if contract_error is not None:
        annotate(
            session_dir,
            row,
            plan_generated=plan_generated,
            automatic_gate_rejected=False,
            operator_rejected=False,
            execution_started=True,
            operator_stop=False,
            success=False,
            failure_category="execution_command_failure",
            failure_description=contract_error,
            raw_log=log_path,
            video=args.video,
            execution_artifact=(
                execution_output if execution_output.is_file() else preflight_artifact
            ),
            started_at_utc=started_at,
        )
        return 6

    outcome = prompt_outcome() if args.outcome == "prompt" else args.outcome
    success = outcome == "success"
    description = args.failure_description
    if not success and not description:
        description = input("failure description> ").strip()
    annotate(
        session_dir,
        row,
        plan_generated=plan_generated,
        automatic_gate_rejected=False,
        operator_rejected=False,
        execution_started=True,
        operator_stop=outcome in {"safety_stop", "human_intervention"},
        success=success,
        failure_category="none" if success else outcome,
        failure_description=description,
        raw_log=log_path,
        video=args.video,
        execution_artifact=execution_output if execution_output.is_file() else preflight_artifact,
        started_at_utc=started_at,
    )
    return 0 if success else 5


if __name__ == "__main__":
    raise SystemExit(main())
