from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rby1_sdk as rby

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR
    ]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.utils.capture_live_realsense import capture_realsense


REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_DATASET = "a270c038"
DEFAULT_ADDRESS = "192.0.2.10:50051"
DEFAULT_MODEL = "a"
DEFAULT_REALSENSE_SOURCE_LINK = "link_head_2"
TASK_NAME = "realsense_apriltag_detection"
DEFAULT_BOARD_JSONS = (
    REPO_ROOT / "data" / "archive" / "indoor_v1" / "apriltag_36h11_left_eef_ids10_15_1in_gap_half.json",
    REPO_ROOT / "data" / "archive" / "indoor_v1" / "apriltag_36h11_right_eef_ids30_35_1in_gap_half.json",
    REPO_ROOT / "data" / "archive" / "indoor_v1" / "apriltag_36h11_board_measured_2in_1in.json",
    REPO_ROOT / "data" / "archive" / "indoor_v1" / "apriltag_36h11_board_side_ids20_25_A4_fullscale_2in_1in.json",
)


def default_transform_json(dataset_name: str) -> Path:
    return (
        REPO_ROOT
        / "robot"
        / "calibration"
        / "session"
        / dataset_name
        / "T_robot_from_aria_selected_current_best_target_toolfit.json"
    )


def default_head_mount_json() -> Path:
    return REPO_ROOT / "robot" / "calibration" / "core" / "realsense_head2_mount_left_eef_apriltag_bundle.json"


def run_detection(args: argparse.Namespace) -> dict[str, Any]:
    capture = capture_realsense(
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        serial=args.serial,
    )
    live_fk = _read_live_fk(args) if args.include_live_fk else None
    color_bgr = capture["color_bgr"]
    intrinsics = capture["intrinsics"]
    board_specs = [_load_board(path) for path in args.board_json]
    detections = _detect_markers(color_bgr)
    transform = _load_transform(args.transform_json, live_fk=live_fk)
    board_poses = [
        pose
        for board in board_specs
        for pose in [_solve_board_pose(board, detections, intrinsics, transform)]
        if pose is not None
    ]
    if live_fk is not None:
        _add_live_fk_board_diagnostics(board_poses, live_fk)
    single_tag_poses = _solve_single_tag_poses(
        detections=detections,
        intrinsics=intrinsics,
        T_robot_from_realsense=transform,
        tag_size_m=args.single_tag_size_m,
        tag_ids=args.single_tag_id,
    )

    label = args.run_label or datetime.now().strftime("apriltag_%H%M%S")
    artifact_base = f"{args.dataset_name}_{label}"
    json_path = artifact_path(TASK_NAME, artifact_base, suffix=".json", runs_root=args.runs_root)
    jpg_path = artifact_path(TASK_NAME, artifact_base, suffix=".jpg", runs_root=args.runs_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = _draw_overlay(color_bgr, detections, board_poses)
    cv2.imwrite(str(jpg_path), overlay)

    payload = {
        "method": "live_realsense_apriltag_pose_detection",
        "dataset_name": args.dataset_name,
        "run_label": label,
        "ok": bool(board_poses or single_tag_poses),
        "artifacts": {"json": str(json_path), "overlay_jpg": str(jpg_path)},
        "transform_json": None if args.transform_json is None else str(args.transform_json),
        "intrinsics": intrinsics,
        "frame_metadata": capture["frame_metadata"],
        "live_fk": live_fk,
        "detected_tag_ids": sorted(detections.keys()),
        "detected_tags": [_tag_summary(tag_id, corners) for tag_id, corners in sorted(detections.items())],
        "boards": board_poses,
        "single_tags": single_tag_poses,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _read_live_fk(args: argparse.Namespace) -> dict[str, Any]:
    toolfit_path = args.toolfit_path or default_transform_json(args.dataset_name)
    robot = rby.create_robot(args.address, args.model)
    if not robot.connect():
        raise RuntimeError(f"failed to connect to RB-Y1 at {args.address}")
    q_live = np.asarray(robot.get_state().position, dtype=float)
    kinematics = RBY1Kinematics.from_sdk_model(
        robot.get_dynamics(),
        robot.model(),
        arm=args.arm,
        toolfit_path=toolfit_path,
    )
    T_robot_from_eef = kinematics.eef_pose(q_live)
    source_links = _source_link_poses(kinematics, q_live)
    return {
        "address": args.address,
        "model": args.model,
        "arm": args.arm,
        "eef_link": kinematics.active_arm.eef_link_name,
        "toolfit_path": str(toolfit_path),
        "T_robot_from_eef": T_robot_from_eef.tolist(),
        "eef_origin_robot_m": T_robot_from_eef[:3, 3].tolist(),
        "source_links": source_links,
    }


def _source_link_poses(kinematics: RBY1Kinematics, q_live: np.ndarray) -> dict[str, Any]:
    links = ("base", DEFAULT_REALSENSE_SOURCE_LINK)
    state = kinematics.make_state(q_live, links=links)
    transform = np.asarray(kinematics.dyn_robot.compute_transformation(state, 0, 1), dtype=np.float64)
    return {
        DEFAULT_REALSENSE_SOURCE_LINK: {
            "T_robot_from_link": transform.tolist(),
            "link_origin_robot_m": transform[:3, 3].tolist(),
        }
    }


def _add_live_fk_board_diagnostics(board_poses: list[dict[str, Any]], live_fk: dict[str, Any]) -> None:
    T_robot_from_eef = np.asarray(live_fk["T_robot_from_eef"], dtype=np.float64).reshape(4, 4)
    T_eef_from_robot = np.linalg.inv(T_robot_from_eef)
    for pose in board_poses:
        matrix = pose.get("T_robot_from_board")
        if matrix is None:
            continue
        T_robot_from_board = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        T_eef_from_board = T_eef_from_robot @ T_robot_from_board
        pose["T_eef_from_board"] = T_eef_from_board.tolist()
        pose["board_origin_eef_m"] = T_eef_from_board[:3, 3].tolist()


def _load_board(path: Path) -> dict[str, Any]:
    board = json.loads(path.read_text(encoding="utf-8"))
    if board.get("opencv_dictionary") != "DICT_APRILTAG_36h11":
        raise ValueError(f"unsupported board dictionary in {path}: {board.get('opencv_dictionary')}")
    tags = {}
    for tag in board["tags"]:
        tags[int(tag["id"])] = np.asarray(tag["corners_board_m"], dtype=np.float64).reshape(4, 3)
    return {
        "path": str(path),
        "name": path.stem,
        "tag_size_m": float(board["tag_size_m"]),
        "tag_ids": sorted(tags.keys()),
        "tags": tags,
    }


def _detect_markers(color_bgr: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(tag_id): np.asarray(corner, dtype=np.float64).reshape(4, 2)
        for tag_id, corner in zip(ids.reshape(-1), corners)
    }


def _load_transform(path: Path | None, *, live_fk: dict[str, Any] | None = None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("T_robot_from_realsense")
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    matrix = payload.get("T_source_link_from_realsense")
    source_link = payload.get("source_link")
    if matrix is None or source_link is None or live_fk is None:
        return None
    source = live_fk.get("source_links", {}).get(str(source_link))
    if source is None:
        return None
    T_robot_from_source = np.asarray(source["T_robot_from_link"], dtype=np.float64).reshape(4, 4)
    T_source_from_realsense = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return T_robot_from_source @ T_source_from_realsense


def _solve_board_pose(
    board: dict[str, Any],
    detections: dict[int, np.ndarray],
    intrinsics: dict[str, Any],
    T_robot_from_realsense: np.ndarray | None,
) -> dict[str, Any] | None:
    object_points = []
    image_points = []
    used_ids = []
    for tag_id in board["tag_ids"]:
        corners = detections.get(tag_id)
        if corners is None:
            continue
        object_points.append(board["tags"][tag_id])
        image_points.append(corners)
        used_ids.append(tag_id)
    if not object_points:
        return None

    obj = np.concatenate(object_points, axis=0).astype(np.float64)
    img = np.concatenate(image_points, axis=0).astype(np.float64)
    camera_matrix, dist_coeffs = _camera_calibration(intrinsics)
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    T_realsense_from_board = np.eye(4, dtype=np.float64)
    T_realsense_from_board[:3, :3] = rotation
    T_realsense_from_board[:3, 3] = tvec.reshape(3)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = img - projected.reshape(-1, 2)
    reprojection_rms_px = float(np.sqrt(np.mean(np.sum(reprojection_error * reprojection_error, axis=1))))

    result = {
        "board_name": board["name"],
        "board_json": board["path"],
        "used_tag_ids": used_ids,
        "used_corner_count": int(len(img)),
        "reprojection_rms_px": reprojection_rms_px,
        "T_realsense_from_board": T_realsense_from_board.tolist(),
        "board_origin_realsense_m": T_realsense_from_board[:3, 3].tolist(),
    }
    if T_robot_from_realsense is not None:
        T_robot_from_board = T_robot_from_realsense @ T_realsense_from_board
        result["T_robot_from_board"] = T_robot_from_board.tolist()
        result["board_origin_robot_m"] = T_robot_from_board[:3, 3].tolist()
    return result


def _solve_single_tag_poses(
    detections: dict[int, np.ndarray],
    intrinsics: dict[str, Any],
    T_robot_from_realsense: np.ndarray | None,
    tag_size_m: float | None,
    tag_ids: list[int] | None,
) -> list[dict[str, Any]]:
    if tag_size_m is None:
        return []
    selected_ids = sorted(detections.keys()) if tag_ids is None else sorted(set(tag_ids))
    half = float(tag_size_m) * 0.5
    object_points = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )
    camera_matrix, dist_coeffs = _camera_calibration(intrinsics)
    poses = []
    for tag_id in selected_ids:
        corners = detections.get(tag_id)
        if corners is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        T_realsense_from_tag = np.eye(4, dtype=np.float64)
        T_realsense_from_tag[:3, :3] = rotation
        T_realsense_from_tag[:3, 3] = tvec.reshape(3)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = corners - projected.reshape(-1, 2)
        result = {
            "tag_id": int(tag_id),
            "tag_size_m": float(tag_size_m),
            "reprojection_rms_px": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
            "T_realsense_from_tag": T_realsense_from_tag.tolist(),
            "tag_origin_realsense_m": T_realsense_from_tag[:3, 3].tolist(),
        }
        if T_robot_from_realsense is not None:
            T_robot_from_tag = T_robot_from_realsense @ T_realsense_from_tag
            result["T_robot_from_tag"] = T_robot_from_tag.tolist()
            result["tag_origin_robot_m"] = T_robot_from_tag[:3, 3].tolist()
        poses.append(result)
    return poses


def _camera_calibration(intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    return camera_matrix, dist_coeffs


def _tag_summary(tag_id: int, corners: np.ndarray) -> dict[str, Any]:
    center = np.mean(corners, axis=0)
    return {
        "tag_id": int(tag_id),
        "center_pixel_uv": center.tolist(),
        "corners_pixel_uv": corners.tolist(),
    }


def _draw_overlay(
    color_bgr: np.ndarray,
    detections: dict[int, np.ndarray],
    board_poses: list[dict[str, Any]],
) -> np.ndarray:
    overlay = color_bgr.copy()
    if detections:
        corners = [corners.reshape(1, 4, 2).astype(np.float32) for corners in detections.values()]
        ids = np.asarray(sorted(detections.keys()), dtype=np.int32).reshape(-1, 1)
        ordered = [detections[int(tag_id)].reshape(1, 4, 2).astype(np.float32) for tag_id in ids.reshape(-1)]
        cv2.aruco.drawDetectedMarkers(overlay, ordered, ids)
    y = 28
    for pose in board_poses:
        text = f"{pose['board_name']}: ids={pose['used_tag_ids']} rms={pose['reprojection_rms_px']:.2f}px"
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect 36h11 AprilTags in a live RealSense frame.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--transform-json", type=Path, default=None)
    parser.add_argument("--board-json", type=Path, action="append", default=None)
    parser.add_argument("--single-tag-size-m", type=float, default=None)
    parser.add_argument("--single-tag-id", type=int, action="append", default=None)
    parser.add_argument("--include-live-fk", action="store_true")
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--toolfit-path", type=Path, default=None)
    args = parser.parse_args()
    if args.transform_json is None:
        args.transform_json = default_transform_json(args.dataset_name)
    if args.board_json is None:
        args.board_json = list(DEFAULT_BOARD_JSONS)
    return args


def main() -> int:
    payload = run_detection(parse_args())
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
