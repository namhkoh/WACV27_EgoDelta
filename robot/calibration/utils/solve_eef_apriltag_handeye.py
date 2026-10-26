from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.geometry import orientation_residual
from robot.kinematics.run_outputs import safe_name


REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_RUNS_ROOT = REPO_ROOT / "robot" / "calibration" / "runs"
DEFAULT_INPUT_GLOB = (
    REPO_ROOT
    / "robot"
    / "kinematics"
    / "runs"
    / "052626"
    / "realsense_apriltag_detection"
    / "a270c038_left_eef_headmount_crosspose_*.json"
)
TASK_NAME = "eef_apriltag_handeye_fit"


def transform_from_params(params: np.ndarray, offset: int) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(params[offset : offset + 3]).as_matrix()
    transform[:3, 3] = params[offset + 3 : offset + 6]
    return transform


def params_from_transform(transform: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]])


def load_matrix(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(4, 4)


def selected_board(payload: dict[str, Any], board_name_contains: str) -> dict[str, Any] | None:
    matches = [
        board
        for board in payload.get("boards", [])
        if board_name_contains in str(board.get("board_name", "")) and "T_realsense_from_board" in board
    ]
    return matches[0] if matches else None


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    for path_text in sorted(glob.glob(str(args.input_glob))):
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        board = selected_board(payload, args.board_name_contains)
        if board is None:
            continue
        if len(board.get("used_tag_ids", [])) < args.min_tags:
            continue
        if float(board["reprojection_rms_px"]) > args.max_rms_px:
            continue
        source = payload.get("live_fk", {}).get("source_links", {}).get(args.source_link)
        if source is None:
            continue
        samples.append(
            {
                "path": str(path),
                "used_tag_ids": board["used_tag_ids"],
                "reprojection_rms_px": float(board["reprojection_rms_px"]),
                "T_robot_from_source": load_matrix(source["T_robot_from_link"]),
                "T_robot_from_eef": load_matrix(payload["live_fk"]["T_robot_from_eef"]),
                "T_realsense_from_board": load_matrix(board["T_realsense_from_board"]),
            }
        )
    if len(samples) < args.min_samples:
        raise RuntimeError(f"only {len(samples)} usable samples; need at least {args.min_samples}")
    return samples


def residuals(params: np.ndarray, samples: list[dict[str, Any]], rotation_weight: float) -> np.ndarray:
    T_source_from_realsense = transform_from_params(params, 0)
    T_eef_from_board = transform_from_params(params, 6)
    values = []
    for sample in samples:
        left = sample["T_robot_from_source"] @ T_source_from_realsense @ sample["T_realsense_from_board"]
        right = sample["T_robot_from_eef"] @ T_eef_from_board
        error = np.linalg.inv(right) @ left
        values.extend(error[:3, 3])
        values.extend(rotation_weight * orientation_residual(error[:3, :3], np.eye(3)))
    return np.asarray(values, dtype=float)


def fit(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    seed_source_from_realsense = load_matrix(json.loads(args.seed_mount_json.read_text(encoding="utf-8"))["T_source_link_from_realsense"])
    first = samples[0]
    seed_eef_from_board = (
        np.linalg.inv(first["T_robot_from_eef"])
        @ first["T_robot_from_source"]
        @ seed_source_from_realsense
        @ first["T_realsense_from_board"]
    )
    seed = np.concatenate([params_from_transform(seed_source_from_realsense), params_from_transform(seed_eef_from_board)])
    result = least_squares(
        residuals,
        seed,
        args=(samples, args.rotation_weight),
        loss="soft_l1",
        f_scale=args.robust_f_scale,
        max_nfev=args.max_nfev,
    )
    T_source_from_realsense = transform_from_params(result.x, 0)
    T_eef_from_board = transform_from_params(result.x, 6)
    reports = sample_reports(samples, T_source_from_realsense, T_eef_from_board)
    return {
        "ok": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "T_source_link_from_realsense": T_source_from_realsense.tolist(),
        "T_eef_from_board": T_eef_from_board.tolist(),
        "source_from_realsense_translation_m": T_source_from_realsense[:3, 3].tolist(),
        "eef_from_board_translation_m": T_eef_from_board[:3, 3].tolist(),
        "summary": summarize_reports(reports),
        "samples": reports,
    }


def sample_reports(
    samples: list[dict[str, Any]],
    T_source_from_realsense: np.ndarray,
    T_eef_from_board: np.ndarray,
) -> list[dict[str, Any]]:
    reports = []
    for sample in samples:
        left = sample["T_robot_from_source"] @ T_source_from_realsense @ sample["T_realsense_from_board"]
        right = sample["T_robot_from_eef"] @ T_eef_from_board
        error = np.linalg.inv(right) @ left
        reports.append(
            {
                "path": sample["path"],
                "used_tag_ids": sample["used_tag_ids"],
                "reprojection_rms_px": sample["reprojection_rms_px"],
                "translation_error_m": float(np.linalg.norm(error[:3, 3])),
                "rotation_error_rad": float(np.linalg.norm(orientation_residual(error[:3, :3], np.eye(3)))),
            }
        )
    return reports


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, float]:
    translation = np.asarray([item["translation_error_m"] for item in reports], dtype=float)
    rotation = np.asarray([item["rotation_error_rad"] for item in reports], dtype=float)
    return {
        "sample_count": int(len(reports)),
        "translation_rms_m": float(np.sqrt(np.mean(translation * translation))),
        "translation_max_m": float(np.max(translation)),
        "rotation_rms_rad": float(np.sqrt(np.mean(rotation * rotation))),
        "rotation_max_rad": float(np.max(rotation)),
    }


def output_path(args: argparse.Namespace) -> Path:
    date_code = datetime.now().strftime("%m%d%y")
    run_label = safe_name(args.run_label)
    return args.runs_root / date_code / TASK_NAME / f"{run_label}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit RealSense/EFF-board hand-eye calibration from captured AprilTag samples.")
    parser.add_argument("--input-glob", type=Path, default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--seed-mount-json", type=Path, default=REPO_ROOT / "robot" / "calibration" / "core" / "realsense_head2_mount_left_eef_apriltag_bundle.json")
    parser.add_argument("--board-name-contains", default="ids50_53")
    parser.add_argument("--source-link", default="link_head_2")
    parser.add_argument("--min-tags", type=int, default=4)
    parser.add_argument("--max-rms-px", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--rotation-weight", type=float, default=0.05)
    parser.add_argument("--robust-f-scale", type=float, default=0.01)
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-label", default="left_eef_headmount_crosspose_fit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(args)
    payload = {
        "method": "fit_source_realsense_and_eef_board_from_eef_apriltag_samples",
        "input_glob": str(args.input_glob),
        "seed_mount_json": str(args.seed_mount_json),
        "board_name_contains": args.board_name_contains,
        **fit(samples, args),
    }
    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(path), "summary": payload["summary"], "ok": payload["ok"]}, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
