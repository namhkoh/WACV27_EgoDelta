from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUNS_ROOT = (
    REPO_ROOT
    / "paper_deliverables_20260820_robot_trials_final_additional20"
    / "robot_trials"
    / "manual_target_selection"
)
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "deleaf_home_right_tucked.json"
DEFAULT_CONFIG = REPO_ROOT / "robot" / "kinematics" / "configs" / "deleaf_right_knife_default.json"
DEFAULT_MANIFEST = REPO_ROOT / "robot" / "kinematics" / "runs" / "deleaf_0d058720_6cuts_manifest.json"
DEFAULT_ALIGNMENT = (
    REPO_ROOT
    / "robot"
    / "calibration"
    / "session"
    / "0d058720"
    / "T_robot_from_aria_manual_world_direct.json"
)
DEFAULT_TOOLFIT = (
    REPO_ROOT
    / "robot"
    / "calibration"
    / "tooltips"
    / "a270c038_right_knife_tip_edge_repivot_060226.json"
)
DEFAULT_MPS_ROOT = Path(
    r"C:\path\to\user\aria\handtrack\mps_0d058720-9302-4563-8764-3a30da628444_vrs"
)
IITP_MOTION_SOURCE = "redacted/iitp_demo@08f1556"
IITP_RIGHT_VELOCITY_LIMIT_RAD_S = 0.4
IITP_RIGHT_ACCELERATION_LIMIT_RAD_S2 = 0.2
IITP_PRE_CUT_TIME_S = 0.5
IITP_CUT_TIME_S = 0.04
IITP_CUT_THROUGH_M = 0.015
IITP_CUT_VELOCITY_LIMIT_RAD_S = 4.0
IITP_CUT_ACCELERATION_LIMIT_RAD_S2 = 18.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_checked(argv: Sequence[str | Path], *, log_path: Path | None = None) -> None:
    command = [str(value) for value in argv]
    print("$ " + subprocess.list2cmdline(command), flush=True)
    if log_path is None:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="") as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("failed to capture deleaf subprocess output")
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            completed = subprocess.CompletedProcess(command, process.wait())
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {subprocess.list2cmdline(command)}")


def selection_posture_command(args: argparse.Namespace) -> list[str | Path]:
    return [
        sys.executable,
        REPO_ROOT / "robot" / "utils" / "restore_home_state.py",
        "--home", args.home,
        "--execute",
        "--scope", "all",
        "--mirror-left-to-right",
        "--minimum-time", str(args.home_minimum_time_s),
    ]


def deleaf_command(args: argparse.Namespace, *, output_json: Path, video_path: Path) -> list[str | Path]:
    return [
        sys.executable,
        REPO_ROOT / "robot" / "kinematics" / "apps" / "deleaf_click_cut.py",
        "--deleaf-config", args.deleaf_config,
        "--dataset-name", args.dataset_name,
        "--home", args.home,
        "--post-selection-home", args.home,
        "--post-selection-home-minimum-time-s", str(args.post_selection_home_minimum_time_s),
        "--manifest", args.manifest,
        "--alignment", args.alignment,
        "--hand-tracking", args.hand_tracking,
        "--slam-trajectory", args.slam_trajectory,
        # Preserve the current calibration and gates while restoring the validated IITP
        # per-target MPS direction selection and conservative right-arm execution limits.
        "--mps-auto-select-action",
        "--require-selected-point-cut-anchor",
        "--toolfit", args.toolfit,
        "--right-velocity-limit-rad-s", str(IITP_RIGHT_VELOCITY_LIMIT_RAD_S),
        "--right-acceleration-limit-rad-s2", str(IITP_RIGHT_ACCELERATION_LIMIT_RAD_S2),
        "--pre-cut-time-s", str(IITP_PRE_CUT_TIME_S),
        "--cut-time-s", str(IITP_CUT_TIME_S),
        "--cut-through-m", str(IITP_CUT_THROUGH_M),
        "--cut-velocity-limit-rad-s", str(IITP_CUT_VELOCITY_LIMIT_RAD_S),
        "--cut-acceleration-limit-rad-s2", str(IITP_CUT_ACCELERATION_LIMIT_RAD_S2),
        "--execute-until", "return_transit",
        "--run-label", args.run_label,
        "--runs-root", args.runs_root,
        "--output-json", output_json,
        "--trial-video-output", video_path,
        "--no-execute-key-confirm",
        "--execute",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select any visible petiole and execute the frozen right-arm deleaf primitive."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--confirm-run-label")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--dataset-name", default="a270c038")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--deleaf-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument(
        "--hand-tracking",
        type=Path,
        default=DEFAULT_MPS_ROOT / "hand_tracking" / "hand_tracking_results.csv",
    )
    parser.add_argument(
        "--slam-trajectory",
        type=Path,
        default=DEFAULT_MPS_ROOT / "slam" / "closed_loop_trajectory.csv",
    )
    parser.add_argument("--home-minimum-time-s", type=float, default=15.0)
    parser.add_argument("--post-selection-home-minimum-time-s", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-restore-home", action="store_true")
    args = parser.parse_args()

    if args.execute and args.confirm_run_label != args.run_label:
        parser.error("--execute requires --confirm-run-label to exactly match --run-label")
    required = (
        args.home,
        args.deleaf_config,
        args.manifest,
        args.alignment,
        args.toolfit,
        args.hand_tracking,
        args.slam_trajectory,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("required deleaf input does not exist: " + ", ".join(map(str, missing)))

    run_dir = args.runs_root / args.run_label
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty trial directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    execution_path = run_dir / "execution.json"
    execution_log = run_dir / "execution.log"
    video_path = run_dir / "trial.mp4"
    target_path = run_dir / "selected_target.json"
    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "schema": "egodelta_any_target_deleaf_run_v1",
        "created_at_utc": utc_now(),
        "run_label": args.run_label,
        "controller": "mps_auto_selected_downward_constrained_right_knife_ik",
        "execute_requested": bool(args.execute),
        "right_arm_only": True,
        "motion_contract": {
            "source": IITP_MOTION_SOURCE,
            "mps_auto_select_action": True,
            "stroke_axis_policy": "downward_constrained_from_deleaf_config",
            "orientation_search": "adaptive_azimuth_from_deleaf_config",
            "phase_policy": "target_relative_transit_entry_approach_pre_cut_cut_lift_retract_return",
            "selected_target_contract": "clicked_3d_point_is_nominal_cut_anchor",
            "soft_foliage_blade_clearance": "report_only",
            "rigid_support_clearance": "mandatory_calibrated_stake_gate",
            "right_velocity_limit_rad_s": IITP_RIGHT_VELOCITY_LIMIT_RAD_S,
            "right_acceleration_limit_rad_s2": IITP_RIGHT_ACCELERATION_LIMIT_RAD_S2,
            "pre_cut_time_s": IITP_PRE_CUT_TIME_S,
            "cut_time_s": IITP_CUT_TIME_S,
            "cut_through_m": IITP_CUT_THROUGH_M,
            "cut_velocity_limit_rad_s": IITP_CUT_VELOCITY_LIMIT_RAD_S,
            "cut_acceleration_limit_rad_s2": IITP_CUT_ACCELERATION_LIMIT_RAD_S2,
            "fast_limits_apply_only_to": "cut",
            "selection_posture": "right_arm_kinematic_mirror_of_saved_left_arm",
            "post_selection_transition": "right_arm_only_to_saved_deleaf_execution_home",
        },
        "artifacts": {},
    }
    try:
        if args.execute and not args.no_restore_home:
            run_checked(selection_posture_command(args))
        if not args.execute:
            raise RuntimeError("this trial wrapper requires --execute after operator review")
        run_checked(
            deleaf_command(args, output_json=execution_path, video_path=video_path),
            log_path=execution_log,
        )
        payload = json.loads(execution_path.read_text(encoding="utf-8"))
        if not payload.get("hard_gate", {}).get("ok", False):
            raise RuntimeError("deleaf IK or clearance hard gate failed")
        if not payload.get("execute", False) or not payload.get("motion", {}).get("executed", False):
            raise RuntimeError("deleaf motion did not execute completely")
        if not payload.get("trial_video", {}).get("ok", False) or not video_path.is_file():
            raise RuntimeError("deleaf trial video was not finalized")
        selected = payload.get("selected_point")
        if not isinstance(selected, dict) or "point_robot_m" not in selected:
            raise RuntimeError("deleaf execution did not preserve the selected cut point")
        write_json(
            target_path,
            {
                "schema": "egodelta_selected_deleaf_target_v1",
                "run_label": args.run_label,
                "selected_point": selected,
                "capture_artifacts": payload.get("capture_artifacts"),
                "transform_json": payload.get("transform_json"),
                "source_execution": str(execution_path),
            },
        )
        summary["artifacts"] = {
            "execution": str(execution_path),
            "execution_log": str(execution_log),
            "target": str(target_path),
            "video": str(video_path),
            "video_metadata": payload["trial_video"]["metadata"],
        }
        summary["selected_point_robot_m"] = selected["point_robot_m"]
        summary["mps_action_name"] = payload.get("mps_guidance", {}).get("action_name")
        summary["candidate_count"] = len(payload.get("candidate_summaries", []))
        summary["executed"] = True
        summary["ok"] = True
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = str(exc)
        raise
    finally:
        summary["updated_at_utc"] = utc_now()
        write_json(summary_path, summary)
        print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
