from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.run_outputs import RUNS_ROOT, run_directory
from robot.kinematics.utils.detect_realsense_apriltags import (
    DEFAULT_ADDRESS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    TASK_NAME,
    default_head_mount_json,
    run_detection,
)


REPO_ROOT = Path(_REPO_ROOT)
LEFT_BOARD_JSON = REPO_ROOT / "data" / "archive" / "testbed_052126" / "apriltag_36h11_eef_2x2_ids50_53_25mm.json"
RIGHT_BOARD_JSON = (
    REPO_ROOT
    / "data"
    / "archive"
    / "testbed_052126"
    / "apriltag_36h11_right_deleaf_eef_2x2_ids60_63_25mm.json"
)


def default_board_json(arm: str) -> Path:
    return LEFT_BOARD_JSON if arm == "left" else RIGHT_BOARD_JSON


def next_index(dataset_name: str, label_prefix: str, runs_root: Path) -> int:
    run_dir = run_directory(TASK_NAME, runs_root=runs_root)
    if not run_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(dataset_name)}_{re.escape(label_prefix)}_(\d+)\.json$")
    indices = []
    for path in run_dir.glob(f"{dataset_name}_{label_prefix}_*.json"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def capture_args(args: argparse.Namespace, run_label: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        run_label=run_label,
        runs_root=args.runs_root,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        serial=args.serial,
        transform_json=args.transform_json or default_head_mount_json(),
        board_json=[args.board_json],
        single_tag_size_m=None,
        single_tag_id=None,
        include_live_fk=True,
        address=args.address,
        model=args.model,
        arm=args.arm,
        toolfit_path=args.toolfit_path,
    )


def selected_board(payload: dict[str, Any], board_json: Path) -> dict[str, Any] | None:
    for board in payload.get("boards", []):
        if Path(board["board_json"]).name == board_json.name:
            return board
    return None


def validate_capture(payload: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    board = selected_board(payload, args.board_json)
    if board is None:
        return False, "target board was not detected"
    if len(board.get("used_tag_ids", [])) < args.min_tags:
        return False, f"only {len(board.get('used_tag_ids', []))} tags detected"
    if float(board["reprojection_rms_px"]) > args.max_rms_px:
        return False, f"reprojection RMS {board['reprojection_rms_px']:.3f}px exceeds {args.max_rms_px:.3f}px"
    if "T_eef_from_board" not in board:
        return False, "live FK diagnostic is missing T_eef_from_board"
    return True, "accepted"


def print_capture_summary(payload: dict[str, Any], args: argparse.Namespace) -> bool:
    board = selected_board(payload, args.board_json)
    ok, reason = validate_capture(payload, args)
    print(f"saved: {payload['artifacts']['json']}")
    if board is None:
        print(f"rejected: {reason}")
        return False
    print(f"tags: {board['used_tag_ids']}  rms_px: {board['reprojection_rms_px']:.3f}")
    if "board_origin_eef_m" in board:
        origin = ", ".join(f"{value:.6f}" for value in board["board_origin_eef_m"])
        print(f"board_origin_eef_m: [{origin}]")
    print(reason)
    return ok


def run_series(args: argparse.Namespace) -> int:
    accepted = 0
    index = args.start_index or next_index(args.dataset_name, args.label_prefix, args.runs_root)
    print("Move the arm to a static pose with the EEF AprilTag board visible.")
    print("Press Enter to capture. Type q then Enter to stop.")
    while args.count is None or accepted < args.count:
        command = input(f"[{args.label_prefix}_{index:02d}] capture? ").strip().lower()
        if command in {"q", "quit", "exit"}:
            break
        run_label = f"{args.label_prefix}_{index:02d}"
        index += 1
        try:
            payload = run_detection(capture_args(args, run_label))
        except Exception as exc:
            print(f"capture failed: {exc}")
            continue
        if print_capture_summary(payload, args):
            accepted += 1
    print(f"accepted_samples: {accepted}")
    return 0 if accepted > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture an Enter-driven EEF AprilTag/FK pose series.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--board-json", type=Path, default=None)
    parser.add_argument("--toolfit-path", type=Path, default=None)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--label-prefix", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--min-tags", type=int, default=4)
    parser.add_argument("--max-rms-px", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--serial", default=None)
    args = parser.parse_args()
    if args.board_json is None:
        args.board_json = default_board_json(args.arm)
    if args.label_prefix is None:
        args.label_prefix = f"{args.arm}_eef_contact_calib_sample"
    return args


def main() -> int:
    return run_series(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
