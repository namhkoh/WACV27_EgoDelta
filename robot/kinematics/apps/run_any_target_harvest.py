from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from robot.server.gripper_status import command_targets


DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "robot"
    / "policy"
    / "policy_action_library"
    / "AT1T1"
    / "success_current"
    / "actions"
    / "AT1T1_instance_001_step_01_AT1T1"
    / "planner"
    / "a270c038_AT1T1_success_current.json"
)
DEFAULT_RUNS_ROOT = REPO_ROOT / "robot" / "kinematics" / "runs"
DEFAULT_GRIPPER_URL = "http://192.0.2.21:8765"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_checked(argv: Sequence[str], *, log_path: Path | None = None) -> None:
    normalized = [str(value) for value in argv]
    print("$ " + subprocess.list2cmdline(normalized), flush=True)
    if log_path is None:
        returncode = subprocess.run(normalized, cwd=REPO_ROOT, check=False).returncode
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="") as log_file:
            process = subprocess.Popen(
                normalized,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("failed to capture child-process output")
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}: {subprocess.list2cmdline(normalized)}"
        )


def resolve_point_picker_python(explicit: Path | None) -> Path:
    sibling_environment = (
        Path(sys.executable).resolve().parent.parent / "surfel_splatting" / Path(sys.executable).name
    )
    candidates = [Path(explicit)] if explicit is not None else [Path(sys.executable), sibling_environment]
    attempted: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if str(resolved) in attempted:
            continue
        attempted.append(str(resolved))
        if not resolved.is_file():
            continue
        probe = subprocess.run(
            [str(resolved), "-c", "import open3d"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return resolved
    raise RuntimeError(
        "Open3D point picking requires a Python interpreter with the open3d package; "
        f"checked {attempted}. Pass --point-picker-python explicitly."
    )


def selected_target(path: Path) -> tuple[list[float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_point")
    if not isinstance(selected, dict):
        raise RuntimeError(f"target artifact has no selected_point: {path}")
    point = selected.get("point_robot_m")
    if not isinstance(point, list) or len(point) != 3:
        raise RuntimeError(f"target artifact has no three-vector selected_point.point_robot_m: {path}")
    values = [float(value) for value in point]
    if selected.get("selection_method") != "manual_click_red_instance_center":
        raise RuntimeError(
            "harvest target must be a centered red-fruit RGB-D selection; "
            f"got {selected.get('selection_method')!r}"
        )
    if selected.get("estimated_fruit_radius_m") is None:
        raise RuntimeError("harvest target has no metric fruit-radius estimate")
    return values, payload


def _status(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(f"{url.rstrip('/')}/status", timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not payload.get("ok", False):
        raise RuntimeError(f"gripper status failed: {payload}")
    return payload


def _left_position(status: dict[str, Any]) -> float:
    states = status.get("motor_states")
    if not isinstance(states, list):
        raise RuntimeError("gripper status has no motor_states")
    for state in states:
        if isinstance(state, dict) and int(state.get("id", -1)) == 1:
            return float(state["position"])
    raise RuntimeError("gripper status has no left motor")


def close_left_gripper(url: str, *, timeout_s: float = 10.0, tolerance: float = 0.25) -> dict[str, Any]:
    before_status = _status(url, timeout_s)
    targets = command_targets(before_status, arm="left")
    open_position = targets.open_position
    close_position = targets.close_position
    before = _left_position(before_status)
    if abs(before - close_position) <= float(tolerance):
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_closed",
            "position": before,
            "open_position": open_position,
            "close_position": close_position,
            "encoder_turn_offset": targets.encoder_turn_offset,
        }
    query = urllib.parse.urlencode(
        {
            "arm": "left",
            "position": close_position,
            "current_limit": 2.5,
            "hold_s": 2.0,
        }
    )
    with urllib.request.urlopen(f"{url.rstrip('/')}/jog?{query}", timeout=float(timeout_s)) as response:
        jog = json.loads(response.read().decode("utf-8"))
    if not isinstance(jog, dict) or not jog.get("ok", False):
        raise RuntimeError(f"left gripper close failed: {jog}")
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        status = _status(url, min(1.0, timeout_s))
        position = _left_position(status)
        if abs(position - close_position) <= float(tolerance):
            return {
                "ok": True,
                "skipped": False,
                "position_before": before,
                "position_after": position,
                "open_position": open_position,
                "close_position": close_position,
                "encoder_turn_offset": targets.encoder_turn_offset,
            }
        time.sleep(0.1)
    raise RuntimeError(f"left gripper did not settle closed at {close_position:.6f}")


def wait_for_file(path: Path, process: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"video recorder exited before becoming ready: {process.returncode}")
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for video recorder ready file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select any visible tomato and run one shared target-conditioned harvest primitive. "
            "No tomato identity or tomato-specific motion template is selected."
        )
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--dataset-name", default="a270c038")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--target-artifact", type=Path, default=None)
    parser.add_argument(
        "--point-picker-python",
        type=Path,
        default=None,
        help="Python interpreter containing open3d. Auto-detects the sibling surfel_splatting environment.",
    )
    parser.add_argument("--template-plan", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--gripper-url", default=DEFAULT_GRIPPER_URL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-run-label", default=None)
    parser.add_argument("--no-restore-home", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--home-minimum-time-s", type=float, default=15.0)
    parser.add_argument("--minimum-time-s", type=float, default=4.0)
    parser.add_argument("--velocity-limit-rad-s", type=float, default=0.15)
    parser.add_argument("--acceleration-limit-rad-s2", type=float, default=0.08)
    parser.add_argument("--execution-timeout-s", type=float, default=20.0)
    parser.add_argument("--arm-settle-timeout-s", type=float, default=8.0)
    parser.add_argument("--gripper-settle-timeout-s", type=float, default=8.0)
    args = parser.parse_args()

    if args.execute and args.confirm_run_label != args.run_label:
        parser.error("--execute requires --confirm-run-label to exactly match --run-label")
    for required in (args.home, args.template_plan):
        if not required.is_file():
            parser.error(f"required file does not exist: {required}")
    if args.target_artifact is not None and not args.target_artifact.is_file():
        parser.error(f"--target-artifact does not exist: {args.target_artifact}")

    run_dir = args.runs_root / args.run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    plan_path = run_dir / "harvest_plan.json"
    preflight_path = run_dir / "preflight.json"
    execution_path = run_dir / "execution.json"
    execution_log_path = run_dir / "execution.log"
    video_path = run_dir / "trial.mp4"
    video_metadata = run_dir / "trial_video.json"
    video_ready = run_dir / "trial_video.ready.json"
    video_stop = run_dir / "trial_video.stop"

    summary: dict[str, Any] = {
        "schema": "egodelta_any_target_harvest_run_v1",
        "created_at_utc": utc_now(),
        "run_label": args.run_label,
        "controller": "shared_target_conditioned_ik_primitive",
        "tomato_identity_conditioning": False,
        "template_plan": str(args.template_plan),
        "execute_requested": bool(args.execute),
        "artifacts": {},
    }
    recorder: subprocess.Popen[Any] | None = None
    try:
        if args.execute and not args.no_restore_home:
            run_checked(
                [
                    sys.executable,
                    REPO_ROOT / "robot" / "utils" / "restore_home_state.py",
                    "--home",
                    args.home,
                    "--execute",
                    "--scope",
                    "all",
                    "--minimum-time",
                    str(args.home_minimum_time_s),
                ]
            )
        if args.execute:
            summary["home_gripper_close"] = close_left_gripper(args.gripper_url)

        target_artifact = args.target_artifact
        if target_artifact is None:
            point_picker_python = resolve_point_picker_python(args.point_picker_python)
            summary["point_picker_python"] = str(point_picker_python)
            target_run_label = f"{args.run_label}_target"
            run_checked(
                [
                    sys.executable,
                    REPO_ROOT / "robot" / "kinematics" / "utils" / "capture_live_realsense.py",
                    "--dataset-name",
                    args.dataset_name,
                    "--run-label",
                    target_run_label,
                    "--runs-root",
                    args.runs_root,
                ]
            )
            from robot.kinematics.run_outputs import artifact_path

            capture_npz = artifact_path(
                "live_realsense_capture",
                f"{args.dataset_name}_{target_run_label}",
                suffix=".npz",
                runs_root=args.runs_root,
            )
            target_artifact = run_dir / "selected_target.json"
            run_checked(
                [
                    point_picker_python,
                    REPO_ROOT / "robot" / "kinematics" / "utils" / "pick_realsense_capture_point.py",
                    capture_npz,
                    "--dataset-name",
                    args.dataset_name,
                    "--snap-red-fruit-center",
                    "--output-json",
                    target_artifact,
                ]
            )
        goal, target_payload = selected_target(target_artifact)
        summary["target"] = {
            "artifact": str(target_artifact),
            "point_robot_m": goal,
            "selection_method": target_payload["selected_point"]["selection_method"],
            "estimated_fruit_radius_m": target_payload["selected_point"]["estimated_fruit_radius_m"],
        }
        summary["artifacts"]["target"] = str(target_artifact)

        goal_arg = ",".join(f"{value:.9f}" for value in goal)
        run_checked(
            [
                sys.executable,
                REPO_ROOT / "robot" / "kinematics" / "apps" / "plan_target_conditioned_harvest.py",
                "--goal-robot-m",
                goal_arg,
                "--target-artifact",
                target_artifact,
                "--template-plan",
                args.template_plan,
                "--run-label",
                args.run_label,
                "--runs-root",
                args.runs_root,
                "--output-json",
                plan_path,
            ]
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not plan.get("hard_gate", {}).get("ok", False):
            raise RuntimeError(f"target-conditioned harvest plan failed hard gates: {plan_path}")
        summary["artifacts"]["plan"] = str(plan_path)
        summary["plan"] = {
            "hard_gate_ok": True,
            "candidate_count": len(plan.get("candidates", [])),
            "waypoint_names": [
                str(item.get("name")) for item in plan.get("best_candidate", {}).get("waypoints", [])
            ],
        }

        if args.execute:
            executor_base = [
                sys.executable,
                REPO_ROOT / "robot" / "kinematics" / "apps" / "execute_harvest_plan.py",
                plan_path,
                "--use-gripper",
                "--gripper-url",
                args.gripper_url,
                "--gripper-target-source",
                "live_status",
                "--minimum-time-s",
                str(args.minimum_time_s),
                "--velocity-limit-rad-s",
                str(args.velocity_limit_rad_s),
                "--acceleration-limit-rad-s2",
                str(args.acceleration_limit_rad_s2),
                "--timeout-s",
                str(args.execution_timeout_s),
                "--arm-settle-timeout-s",
                str(args.arm_settle_timeout_s),
                "--gripper-post-command-wait-s",
                str(args.gripper_settle_timeout_s),
            ]
            run_checked([*executor_base, "--check-home-only", "--output-json", preflight_path])
            summary["artifacts"]["preflight"] = str(preflight_path)

            if not args.no_video:
                video_ready.unlink(missing_ok=True)
                video_stop.unlink(missing_ok=True)
                recorder = subprocess.Popen(
                    [
                        sys.executable,
                        REPO_ROOT / "robot" / "evaluation" / "record_realsense_video.py",
                        "--output",
                        video_path,
                        "--metadata",
                        video_metadata,
                        "--ready-file",
                        video_ready,
                        "--stop-file",
                        video_stop,
                        "--max-duration-s",
                        "180",
                    ],
                    cwd=REPO_ROOT,
                )
                wait_for_file(video_ready, recorder, 15.0)
            summary["artifacts"]["execution_log"] = str(execution_log_path)
            run_checked(
                [*executor_base, "--execute", "--output-json", execution_path],
                log_path=execution_log_path,
            )
            summary["artifacts"]["execution"] = str(execution_path)
            summary["executed"] = True
        else:
            summary["executed"] = False
        summary["ok"] = True
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = str(exc)
        raise
    finally:
        if recorder is not None:
            video_stop.touch()
            try:
                recorder.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder.wait(timeout=5.0)
            if video_path.exists():
                summary["artifacts"]["video"] = str(video_path)
            if video_metadata.exists():
                summary["artifacts"]["video_metadata"] = str(video_metadata)
        summary["updated_at_utc"] = utc_now()
        write_json(summary_path, summary)
        print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
