from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = (
    REPO_ROOT
    / "paper_deliverables_20260820_robot_trials_final_additional20"
    / "robot_trials"
)
DEFAULT_PREREGISTERED_SESSION = (
    REPO_ROOT / "paper_deliverables_20260819_robot_trials" / "robot_trials"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "paper" / "evidence" / "controlled_robot_execution_20260821"
)
UNAVAILABLE = "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an evidence-first audit for the controlled robot execution section."
    )
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION)
    parser.add_argument(
        "--preregistered-session-dir", type=Path, default=DEFAULT_PREREGISTERED_SESSION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def path_or_unavailable(value: str | Path | None) -> str:
    if value is None or str(value).strip() in {"", "not_recorded"}:
        return UNAVAILABLE
    return str(value)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def xyz(payload: dict[str, Any], candidates: Iterable[tuple[str, ...]]) -> list[float] | None:
    for keys in candidates:
        value = nested(payload, *keys)
        if (
            isinstance(value, list)
            and len(value) == 3
            and all(isinstance(item, (int, float)) for item in value)
        ):
            return [float(item) for item in value]
    return None


def split_xyz(value: list[float] | None) -> tuple[Any, Any, Any]:
    if value is None:
        return UNAVAILABLE, UNAVAILABLE, UNAVAILABLE
    return value[0], value[1], value[2]


def manual_target_source(payload: dict[str, Any]) -> str:
    method = str(
        nested(payload, "selected_point", "selection_method")
        or nested(payload, "selected_point", "selection_source")
        or nested(payload, "picked_point", "selection_source")
        or ""
    ).lower()
    if any(token in method for token in ("manual", "opencv_pick", "open3d", "external")):
        return "manual_realsense_selection"
    return UNAVAILABLE


def build_accuracy_rows(
    completed_direct: list[dict[str, str]], expected_per_task: int = 30
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in completed_direct:
        by_task[row["task"]].append(row)

    for task in ("harvest", "deleaf"):
        task_rows = sorted(by_task[task], key=lambda item: item["trial_id"])
        for evidence_index, ledger_row in enumerate(task_rows, start=1):
            target_path = Path(ledger_row["target_artifact_path"])
            target_payload = read_json(target_path) if target_path.exists() else {}
            selected = xyz(
                target_payload,
                (
                    ("selected_point", "point_robot_m"),
                    ("picked_point", "point_robot_m"),
                ),
            )
            predicted = xyz(
                target_payload,
                (
                    ("predicted_xyz",),
                    ("predicted_target_robot_m",),
                    ("egodelta_target_robot_m",),
                ),
            )
            socket = xyz(
                target_payload,
                (
                    ("socket_gt_xyz",),
                    ("socket_gt_robot_m",),
                    ("socket_ground_truth_robot_m",),
                ),
            )
            reasons: list[str] = []
            source = manual_target_source(target_payload)
            if predicted is None:
                reasons.append(
                    "No frozen EGODELTA prediction is stored in the target artifact; "
                    f"target source is {source}."
                )
            if socket is None:
                reasons.append("No physical socket ground-truth coordinate is stored.")
            error_cm: float | str = UNAVAILABLE
            within_3cm: bool | str = UNAVAILABLE
            if predicted is not None and socket is not None:
                error_cm = 100.0 * math.dist(predicted, socket)
                within_3cm = error_cm <= 3.0
            pred_x, pred_y, pred_z = split_xyz(predicted)
            socket_x, socket_y, socket_z = split_xyz(socket)
            selected_x, selected_y, selected_z = split_xyz(selected)
            rows.append(
                {
                    "evidence_row_id": f"recorded_{task}_{evidence_index:02d}",
                    "trial_id": ledger_row["trial_id"],
                    "task": task,
                    "scene_id": ledger_row["scene_id"],
                    "predicted_x": pred_x,
                    "predicted_y": pred_y,
                    "predicted_z": pred_z,
                    "socket_gt_x": socket_x,
                    "socket_gt_y": socket_y,
                    "socket_gt_z": socket_z,
                    "error_cm": error_cm,
                    "within_3cm": within_3cm,
                    "selected_target_x": selected_x,
                    "selected_target_y": selected_y,
                    "selected_target_z": selected_z,
                    "coordinate_frame": "robot_base" if selected is not None else UNAVAILABLE,
                    "target_source": source,
                    "execution_success": ledger_row["success"],
                    "target_artifact": str(target_path),
                    "execution_artifact": path_or_unavailable(
                        ledger_row.get("execution_artifact_path")
                    ),
                    "source_log": path_or_unavailable(ledger_row.get("raw_log_path")),
                    "missing_data_reason": " ".join(reasons) if reasons else "",
                }
            )

        missing_count = max(0, expected_per_task - len(task_rows))
        for missing_index in range(1, missing_count + 1):
            rows.append(
                {
                    "evidence_row_id": f"missing_historical_{task}_{missing_index:02d}",
                    "trial_id": UNAVAILABLE,
                    "task": task,
                    "scene_id": UNAVAILABLE,
                    "predicted_x": UNAVAILABLE,
                    "predicted_y": UNAVAILABLE,
                    "predicted_z": UNAVAILABLE,
                    "socket_gt_x": UNAVAILABLE,
                    "socket_gt_y": UNAVAILABLE,
                    "socket_gt_z": UNAVAILABLE,
                    "error_cm": UNAVAILABLE,
                    "within_3cm": UNAVAILABLE,
                    "selected_target_x": UNAVAILABLE,
                    "selected_target_y": UNAVAILABLE,
                    "selected_target_z": UNAVAILABLE,
                    "coordinate_frame": UNAVAILABLE,
                    "target_source": UNAVAILABLE,
                    "execution_success": UNAVAILABLE,
                    "target_artifact": UNAVAILABLE,
                    "execution_artifact": UNAVAILABLE,
                    "source_log": UNAVAILABLE,
                    "missing_data_reason": (
                        "The claimed historical direct trial has no trial-level ID, frozen "
                        "prediction, socket GT, log, or video in the located authoritative ledgers."
                    ),
                }
            )
    return rows


def accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [
            float(row["error_cm"])
            for row in task_rows
            if isinstance(row["error_cm"], (int, float))
        ]
        return {
            "expected_claimed_trial_count": len(task_rows),
            "artifact_backed_trial_count": sum(
                row["trial_id"] != UNAVAILABLE for row in task_rows
            ),
            "evaluated_with_prediction_and_socket_gt": len(errors),
            "median_error_cm": statistics.median(errors) if errors else UNAVAILABLE,
            "within_3cm_count": sum(error <= 3.0 for error in errors) if errors else UNAVAILABLE,
            "within_3cm_percent": (
                100.0 * sum(error <= 3.0 for error in errors) / len(errors)
                if errors
                else UNAVAILABLE
            ),
            "missing_prediction_count": sum(
                row["predicted_x"] == UNAVAILABLE for row in task_rows
            ),
            "missing_socket_gt_count": sum(
                row["socket_gt_x"] == UNAVAILABLE for row in task_rows
            ),
            "manual_target_selection_count": sum(
                row["target_source"] == "manual_realsense_selection" for row in task_rows
            ),
        }

    return {
        "harvest": summarize([row for row in rows if row["task"] == "harvest"]),
        "deleaf": summarize([row for row in rows if row["task"] == "deleaf"]),
        "overall": summarize(rows),
        "manuscript_fill_status": "unavailable",
        "reason": (
            "No artifact-backed row contains both a frozen EGODELTA prediction and a "
            "physical attachment-socket ground-truth coordinate."
        ),
    }


def build_attempt_rows(
    run_root: Path, observed_ledger: list[dict[str, str]]
) -> list[dict[str, Any]]:
    bound_execution_to_trial = {
        str(Path(row["execution_artifact_path"]).resolve()).lower(): row["trial_id"]
        for row in observed_ledger
        if row.get("execution_artifact_path") not in {None, "", "not_recorded"}
    }
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^wacv27_additional20_(harvest|deleaf)_(\d+)_direct(?:_retry(\d+))?$")
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        match = pattern.match(run_dir.name)
        if not match:
            continue
        task, trial_number, retry_number = match.groups()
        summary_path = run_dir / "run_summary.json"
        execution_path = run_dir / "execution.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        execution = read_json(execution_path) if execution_path.exists() else {}
        hard_gate = nested(execution, "hard_gate", "ok")
        motion_executed = nested(execution, "motion", "executed")
        summary_executed = summary.get("executed")
        executed = motion_executed if isinstance(motion_executed, bool) else summary_executed
        ledger_trial_id = bound_execution_to_trial.get(str(execution_path.resolve()).lower())
        error = summary.get("error") or execution.get("error") or execution.get("execution_error")
        if hard_gate is False and executed is not True:
            automatic_gate_rejected: bool | str = True
        elif hard_gate is True:
            automatic_gate_rejected = False
        else:
            automatic_gate_rejected = UNAVAILABLE
        rows.append(
            {
                "run_label": run_dir.name,
                "task": task,
                "nominal_trial_number": int(trial_number),
                "retry_number": int(retry_number) if retry_number is not None else 0,
                "summary_exists": summary_path.exists(),
                "execution_exists": execution_path.exists(),
                "summary_ok": summary.get("ok", UNAVAILABLE),
                "hard_gate_ok": hard_gate if isinstance(hard_gate, bool) else UNAVAILABLE,
                "automatic_gate_rejected": automatic_gate_rejected,
                "motion_executed": executed if isinstance(executed, bool) else UNAVAILABLE,
                "ledger_trial_id": ledger_trial_id or UNAVAILABLE,
                "included_in_reported_denominator": ledger_trial_id is not None,
                "error": str(error) if error else "",
                "run_summary": str(summary_path) if summary_path.exists() else UNAVAILABLE,
                "execution_artifact": str(execution_path) if execution_path.exists() else UNAVAILABLE,
                "video": str(run_dir / "trial.mp4") if (run_dir / "trial.mp4").exists() else UNAVAILABLE,
            }
        )
    return rows


def count_values(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(field, "") for row in rows).items()))


def trial_accounting_summary(
    observed: list[dict[str, str]], preregistered: list[dict[str, str]], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for task in ("harvest", "deleaf"):
        for evaluation_type in ("direct", "policy"):
            key = f"{task}_{evaluation_type}"
            rows = [
                row
                for row in observed
                if row["task"] == task and row["evaluation_type"] == evaluation_type
            ]
            completed = [row for row in rows if row["trial_status"] == "completed"]
            conditions[key] = {
                "expected_count": 30,
                "observed_ledger_rows": len(rows),
                "completed_artifact_backed_count": len(completed),
                "success_counts": count_values(completed, "success"),
                "automatic_gate_rejection_counts": count_values(
                    completed, "automatic_gate_rejected"
                ),
                "operator_pre_execution_rejection_counts": count_values(
                    completed, "operator_rejected_before_execution"
                ),
                "operator_or_safety_stop_counts": count_values(
                    completed, "operator_or_safety_stop"
                ),
                "unique_recorded_scene_ids": len(
                    {
                        row["scene_id"]
                        for row in completed
                        if row["scene_id"] not in {"", "not_recorded"}
                    }
                ),
            }

    return {
        "protocol_expected": {
            "harvest_direct": 30,
            "deleaf_direct": 30,
            "harvest_policy": 30,
            "deleaf_policy": 30,
            "total": 120,
        },
        "preregistered_ledger_rows": len(preregistered),
        "observed_additional20_ledger_rows": len(observed),
        "conditions": conditions,
        "attempt_directory_audit": {
            "attempt_count": len(attempts),
            "included_in_ledger_count": sum(
                bool(row["included_in_reported_denominator"]) for row in attempts
            ),
            "excluded_attempt_count": sum(
                not bool(row["included_in_reported_denominator"]) for row in attempts
            ),
            "automatic_gate_rejected_count": sum(
                row["automatic_gate_rejected"] is True for row in attempts
            ),
            "note": (
                "The final ledger binds successful retry artifacts to nominal rows. Failed or "
                "abandoned attempt directories are not denominator rows, contrary to the frozen "
                "prospective amendment's no-replacement rule."
            ),
        },
        "learned_policy_verification": {
            "completed_policy_rows": sum(
                row["trial_status"] == "completed" and row["evaluation_type"] == "policy"
                for row in observed
            ),
            "status": "unsupported",
            "reason": (
                "The completed rows use run_any_target_harvest.py or "
                "run_any_target_deleaf.py. No completed row identifies a learned policy model; "
                "policy_dir is not_applicable."
            ),
        },
        "denominator_reproduces_manuscript_claim": False,
    }


def build_demo_rows(training_index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in training_index.get("actions", []):
        for action in episode.get("actions", []):
            rows.append(
                {
                    "demonstration_id": episode.get("label", UNAVAILABLE),
                    "task": action.get("primitive", UNAVAILABLE),
                    "step_id": action.get("step_id", UNAVAILABLE),
                    "outcome": action.get("outcome", UNAVAILABLE),
                    "source_kind": "real_execution",
                    "trajectory_path": path_or_unavailable(
                        nested(action, "artifacts", "trajectory")
                    ),
                    "goal_source_artifact": path_or_unavailable(
                        nested(action, "goal", "source_artifact")
                    ),
                    "episode_json": path_or_unavailable(episode.get("episode_json")),
                }
            )
    return rows


def build_failure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, count in (("harvest", 2), ("deleaf", 3)):
        for index in range(1, count + 1):
            rows.append(
                {
                    "failure_index": f"{task}_{index}",
                    "task": task,
                    "trial_id": UNAVAILABLE,
                    "scene_id": UNAVAILABLE,
                    "observed_failure": UNAVAILABLE,
                    "likely_pipeline_stage": "unresolved",
                    "source_log": UNAVAILABLE,
                    "missing_data_reason": (
                        "The requested aggregate failure count is stated in the manuscript, but no "
                        "authoritative 30-trial policy ledger maps this failure to a trial, scene, "
                        "observation, or log."
                    ),
                }
            )
    return rows


def detector_summary() -> dict[str, Any]:
    documented_labels = Path("/tmp/egodelta/front/labels.jsonl")
    return {
        "annotation_unit": "manually verified 2D target point on a front-view image",
        "is_image_count": False,
        "is_scene_count": False,
        "counts": {
            "harvest": UNAVAILABLE,
            "deleaf": UNAVAILABLE,
            "overall": UNAVAILABLE,
        },
        "raw_annotation_path_documented_by_code": str(documented_labels),
        "raw_annotation_available_on_this_machine": documented_labels.exists(),
        "status": "unavailable",
        "reason": (
            "The documented labels.jsonl and companion front-view manifests are absent from "
            "this machine and no checked-in annotation manifest preserves their row counts."
        ),
        "source_code": [
            str(REPO_ROOT / "tools" / "build_manual_front_yolo_dataset_032.py"),
            str(REPO_ROOT / "tools" / "evaluate_manual_front_yolo_032.py"),
            str(REPO_ROOT / "policy_train.tex"),
        ],
    }


def source_manifest(session: Path, preregistered_session: Path) -> dict[str, Any]:
    paths = {
        "observed_trial_ledger": session / "robot_trial_ledger.csv",
        "observed_trial_summary": session / "robot_trial_summary.json",
        "preregistered_trial_ledger": preregistered_session / "robot_trial_ledger.csv",
        "preregistered_trial_summary": preregistered_session / "robot_trial_summary.json",
        "prospective_protocol": REPO_ROOT / "deliverables_robot_trials_20260819.md",
        "latest_deliverables_spec": REPO_ROOT / "deliverables_latest.md",
        "policy_training_index": REPO_ROOT
        / "robot"
        / "policy"
        / "policy_action_library"
        / "training_index.json",
        "harvest_policy_metadata": REPO_ROOT
        / "robot"
        / "policy"
        / "trained_policies"
        / "harvest_bc_success_trimmed_nojitter10hz_chunk30_seed11_rerun_20260819"
        / "policy_metadata.json",
        "deleaf_policy_metadata": REPO_ROOT
        / "robot"
        / "policy"
        / "trained_policies"
        / "deleaf_bc_success_only_jitter10hz_chunk50_seed11_rerun_20260819"
        / "policy_metadata.json",
        "historical_harvest_rollout_log_dir": REPO_ROOT
        / "robot"
        / "policy"
        / "rollout_logs"
        / "harvest",
        "historical_deleaf_rollout_log_dir": REPO_ROOT
        / "robot"
        / "policy"
        / "rollout_logs"
        / "deleaf",
    }
    records = {}
    for name, path in paths.items():
        records[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.is_file() else UNAVAILABLE,
        }
    records["historical_rollout_log_caveat"] = {
        "status": "not_an_evaluation_ledger",
        "reason": (
            "These logs mix dry runs, guard failures, commissioning, and multiple model variants; "
            "they contain no authoritative physical success labels or evaluation scene IDs."
        ),
    }
    return records


def write_readme(
    path: Path,
    accuracy: dict[str, Any],
    accounting: dict[str, Any],
    demo_rows: list[dict[str, Any]],
) -> None:
    harvest_demos = sum(row["task"] == "harvest" for row in demo_rows)
    deleaf_demos = sum(row["task"] == "deleaf" for row in demo_rows)
    text = f"""# Controlled Robot Execution Evidence Audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Evidence-backed findings

- The located final ledger contains 20 completed harvesting direct rows and 20 completed deleafing direct rows.
- The completed runs are scripted target-conditioned executions, not learned behavior-cloning rollouts. Their ledger records use `policy_dir=not_applicable`.
- The frozen target artifacts store manual RealSense/Open3D selections. They do not store the recovered EGODELTA prediction needed for a direct-target accuracy comparison.
- No located trial artifact stores physical attachment-socket ground truth. Therefore target-error N, median error, and the count within 3 cm are **unavailable**.
- The authoritative policy-action index confirms {harvest_demos} harvesting demonstrations and {deleaf_demos} deleafing demonstrations.
- The raw manual front-view detector annotation file documented by the training code is absent. Annotation counts by task and overall are **unavailable**.
- No completed learned-policy rows are present in the located evaluation ledger. Policy scene counts, rejection/stop counts, and the five claimed failure identities are **unavailable**.

## Trial-accounting warning

The prospective amendment requires 30 rows per task and execution mode, with every rejection, stop, and failure retained. The current completed cohort binds successful retry artifacts to 40 direct rows. The attempt audit finds {accounting['attempt_directory_audit']['excluded_attempt_count']} run directories not included as denominator rows. Consequently, the available ledger does not reproduce a prospective 120-rollout evaluation.

## Section 4.6 fill status

| Requested value | Evidence status |
|---|---|
| Direct targets with socket GT | unavailable |
| Median direct-target error | unavailable |
| Direct targets within 3 cm | unavailable |
| Manually verified detector targets | unavailable |
| Unique policy-evaluation scenes | unavailable |
| Policy automatic rejections / operator rejections / stops | unavailable |
| Two harvesting and three deleafing policy failure identities | unavailable / unresolved |
| Training demonstrations | supported: {harvest_demos} harvesting, {deleaf_demos} deleafing |

Do not replace the manuscript TBDs with aggregate claims until the missing trial-level records are recovered. The `direct_target_accuracy.csv` placeholders and `policy_failure_audit.csv` rows intentionally use `unavailable` rather than inferred values.

## Files

- `direct_target_accuracy.csv`: 60 claimed direct slots, including 40 artifact-backed rows and 20 explicit missing historical rows.
- `direct_target_accuracy_summary.json`: strict computed availability and metric summary.
- `trial_accounting_observed.csv`: the located final additional-20 ledger.
- `attempt_audit.csv`: every located direct-run directory, including retries excluded from the ledger.
- `trial_accounting_summary.json`: expected-versus-observed counts and denominator audit.
- `detector_supervision.csv` / `detector_supervision_summary.json`: annotation evidence and missing-source status.
- `policy_demonstrations.csv`: the seven action-library demonstrations.
- `policy_evaluation_scenes.json`: policy scene-count availability.
- `policy_failure_audit.csv`: five explicitly unresolved claimed policy failures.
- `source_log_paths.json`: source paths, availability, and hashes.
- `revision.json`: code/data revision used for this export.
- `checksums.sha256`: package checksums.
"""
    path.write_text(text, encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{sha256(path)}  {path.name}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty evidence directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_ledger_path = args.session_dir / "robot_trial_ledger.csv"
    preregistered_ledger_path = args.preregistered_session_dir / "robot_trial_ledger.csv"
    observed = read_csv(observed_ledger_path)
    preregistered = read_csv(preregistered_ledger_path)
    completed_direct = [
        row
        for row in observed
        if row["evaluation_type"] == "direct" and row["trial_status"] == "completed"
    ]

    accuracy_rows = build_accuracy_rows(completed_direct)
    accuracy_fields = [
        "evidence_row_id",
        "trial_id",
        "task",
        "scene_id",
        "predicted_x",
        "predicted_y",
        "predicted_z",
        "socket_gt_x",
        "socket_gt_y",
        "socket_gt_z",
        "error_cm",
        "within_3cm",
        "selected_target_x",
        "selected_target_y",
        "selected_target_z",
        "coordinate_frame",
        "target_source",
        "execution_success",
        "target_artifact",
        "execution_artifact",
        "source_log",
        "missing_data_reason",
    ]
    write_csv(output_dir / "direct_target_accuracy.csv", accuracy_rows, accuracy_fields)
    accuracy_payload = accuracy_summary(accuracy_rows)
    write_json(output_dir / "direct_target_accuracy_summary.json", accuracy_payload)

    run_root = args.session_dir / "manual_target_selection"
    attempt_rows = build_attempt_rows(run_root, observed)
    attempt_fields = [
        "run_label",
        "task",
        "nominal_trial_number",
        "retry_number",
        "summary_exists",
        "execution_exists",
        "summary_ok",
        "hard_gate_ok",
        "automatic_gate_rejected",
        "motion_executed",
        "ledger_trial_id",
        "included_in_reported_denominator",
        "error",
        "run_summary",
        "execution_artifact",
        "video",
    ]
    write_csv(output_dir / "attempt_audit.csv", attempt_rows, attempt_fields)
    write_csv(
        output_dir / "trial_accounting_observed.csv",
        observed,
        list(observed[0].keys()),
    )
    accounting = trial_accounting_summary(observed, preregistered, attempt_rows)
    write_json(output_dir / "trial_accounting_summary.json", accounting)

    detector_fields = [
        "annotation_id",
        "task",
        "target_point",
        "image_path",
        "scene_id",
        "raw_annotation_path",
        "status",
        "missing_data_reason",
    ]
    write_csv(output_dir / "detector_supervision.csv", [], detector_fields)
    write_json(output_dir / "detector_supervision_summary.json", detector_summary())

    training_index_path = (
        REPO_ROOT
        / "robot"
        / "policy"
        / "policy_action_library"
        / "training_index.json"
    )
    demo_rows = build_demo_rows(read_json(training_index_path))
    demo_fields = [
        "demonstration_id",
        "task",
        "step_id",
        "outcome",
        "source_kind",
        "trajectory_path",
        "goal_source_artifact",
        "episode_json",
    ]
    write_csv(output_dir / "policy_demonstrations.csv", demo_rows, demo_fields)

    completed_policy = [
        row
        for row in observed
        if row["evaluation_type"] == "policy" and row["trial_status"] == "completed"
    ]
    write_json(
        output_dir / "policy_evaluation_scenes.json",
        {
            "harvest": {
                "completed_policy_rows": sum(row["task"] == "harvest" for row in completed_policy),
                "unique_scene_count": UNAVAILABLE,
                "scene_ids": [],
            },
            "deleaf": {
                "completed_policy_rows": sum(row["task"] == "deleaf" for row in completed_policy),
                "unique_scene_count": UNAVAILABLE,
                "scene_ids": [],
            },
            "overall_unique_scene_count": UNAVAILABLE,
            "reason": "No completed learned-policy rows exist in the located evaluation ledger.",
        },
    )

    failure_rows = build_failure_rows()
    failure_fields = [
        "failure_index",
        "task",
        "trial_id",
        "scene_id",
        "observed_failure",
        "likely_pipeline_stage",
        "source_log",
        "missing_data_reason",
    ]
    write_csv(output_dir / "policy_failure_audit.csv", failure_rows, failure_fields)

    write_json(
        output_dir / "section_4_6_fill_values.json",
        {
            "direct_execution_targets_with_socket_gt": UNAVAILABLE,
            "direct_median_target_error_cm": UNAVAILABLE,
            "direct_within_3cm_count": UNAVAILABLE,
            "direct_within_3cm_denominator": UNAVAILABLE,
            "direct_within_3cm_percent": UNAVAILABLE,
            "manual_detector_targets_harvest": UNAVAILABLE,
            "manual_detector_targets_deleaf": UNAVAILABLE,
            "manual_detector_targets_overall": UNAVAILABLE,
            "policy_evaluation_unique_scenes_harvest": UNAVAILABLE,
            "policy_evaluation_unique_scenes_deleaf": UNAVAILABLE,
            "policy_evaluation_unique_scenes_overall": UNAVAILABLE,
            "policy_automatic_plan_rejections": UNAVAILABLE,
            "policy_operator_pre_execution_rejections": UNAVAILABLE,
            "policy_operator_or_safety_stops": UNAVAILABLE,
            "failure_causes": "unresolved",
            "training_demonstrations_harvest": sum(
                row["task"] == "harvest" for row in demo_rows
            ),
            "training_demonstrations_deleaf": sum(
                row["task"] == "deleaf" for row in demo_rows
            ),
            "direct_additional_artifact_backed_successes_harvest": 20,
            "direct_additional_artifact_backed_successes_deleaf": 20,
            "learned_policy_completed_rows_harvest": 0,
            "learned_policy_completed_rows_deleaf": 0,
        },
    )

    write_json(
        output_dir / "source_log_paths.json",
        source_manifest(args.session_dir, args.preregistered_session_dir),
    )
    write_json(
        output_dir / "revision.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "branch": git_output("branch", "--show-current"),
            "head_commit": git_output("rev-parse", "HEAD"),
            "worktree_clean_before_export": git_output("status", "--short") == "",
            "exporter": str(Path(__file__).resolve()),
            "observed_ledger_source_commits": sorted(
                {
                    row["source_commit"]
                    for row in observed
                    if row.get("source_commit") not in {None, "", "not_recorded"}
                }
            ),
        },
    )
    write_readme(output_dir / "README.md", accuracy_payload, accounting, demo_rows)
    write_checksums(output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
