from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR
    ]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot.kinematics.types import Array, Pose, TaskFrameCandidate


DEFAULT_TARGET_DIR = Path("robot/calibration/targets/a270c038")
DEFAULT_TARGET_PATTERN = "harvest_*_task_frame_*.json"

ROTATION_TOL = 1e-5
AXIS_TOL = 1e-5
CONTACT_POINT_TOL_M = 0.02


class TaskFrameValidationError(ValueError):
    pass


@dataclass(frozen=True)
class HarvestTaskFrame:
    target_name: str
    source_path: Path
    target_index: int
    target_position_robot_m: Array
    fruit_center_robot_m: Array
    rotation_robot_from_task: Array
    rotation_robot_from_tcp: Array
    jaw_axis_robot: Array
    backoff_axis_robot: Array
    pull_axis_robot: Array
    palm_axis_robot: Array
    contact_points_robot_m: Array | None
    contact_midpoint_robot_m: Array | None
    gripper_jaw_axis_local: str
    gripper_backoff_axis_local: str
    recommended_pregrasp_aperture_m: float | None
    recommended_close_aperture_m: float | None
    estimated_grasp_width_m: float | None
    estimated_grasp_depth_m: float | None
    hand_frame_weak: bool
    quality: dict[str, Any]
    no_replay_contract: dict[str, Any]

    @property
    def task_pose(self) -> Pose:
        return Pose(_transform(self.rotation_robot_from_task, self.target_position_robot_m))

    @property
    def tcp_grasp_pose(self) -> Pose:
        return Pose(_transform(self.rotation_robot_from_tcp, self.target_position_robot_m))

    def as_task_candidate(self, *, score: float = 1.0) -> TaskFrameCandidate:
        return TaskFrameCandidate(
            target_position_robot_m=self.target_position_robot_m,
            rotation_robot_from_task=self.rotation_robot_from_task,
            score=float(score),
            source=str(self.source_path),
            diagnostics={
                "target_name": self.target_name,
                "target_index": self.target_index,
                "hand_frame_weak": self.hand_frame_weak,
            },
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "source_path": str(self.source_path),
            "target_index": self.target_index,
            "target_position_robot_m": self.target_position_robot_m.tolist(),
            "fruit_center_robot_m": self.fruit_center_robot_m.tolist(),
            "jaw_axis_robot": self.jaw_axis_robot.tolist(),
            "backoff_axis_robot": self.backoff_axis_robot.tolist(),
            "pull_axis_robot": self.pull_axis_robot.tolist(),
            "palm_axis_robot": self.palm_axis_robot.tolist(),
            "contact_points_robot_m": None
            if self.contact_points_robot_m is None
            else self.contact_points_robot_m.tolist(),
            "contact_midpoint_robot_m": None
            if self.contact_midpoint_robot_m is None
            else self.contact_midpoint_robot_m.tolist(),
            "gripper_jaw_axis_local": self.gripper_jaw_axis_local,
            "gripper_backoff_axis_local": self.gripper_backoff_axis_local,
            "recommended_pregrasp_aperture_m": self.recommended_pregrasp_aperture_m,
            "recommended_close_aperture_m": self.recommended_close_aperture_m,
            "estimated_grasp_width_m": self.estimated_grasp_width_m,
            "estimated_grasp_depth_m": self.estimated_grasp_depth_m,
            "hand_frame_weak": self.hand_frame_weak,
            "no_replay_ok": bool(self.no_replay_contract.get("ok", False)),
        }


def load_harvest_task_frame(path: Path) -> HarvestTaskFrame:
    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    quality = _required_dict(data, "quality")
    axes = _required_dict(data, "task_axes_robot")
    no_replay_contract = _required_dict(data, "no_replay_contract")
    target_record = data.get("target_record", {})
    if target_record is not None and not isinstance(target_record, dict):
        raise TaskFrameValidationError("target_record must be an object when present")

    target_position = _vector(data, "target_robot_m")
    fruit_center = _vector(data, "fruit_center_robot_m")
    rotation_task = _rotation_matrix(data, "R_robot_from_task")
    rotation_tcp = _rotation_from_quat_wxyz(_vector4(data, "grasp_orientation_quat_wxyz"))

    backoff_axis = _unit(_vector(axes, "plus_x_backoff_axis_robot"), "plus_x_backoff_axis_robot")
    jaw_axis = _unit(_vector(axes, "plus_y_jaw_axis_robot"), "plus_y_jaw_axis_robot")
    palm_axis = _unit(_vector(axes, "plus_z_palm_axis_robot"), "plus_z_palm_axis_robot")
    pull_axis = _unit(_vector(axes, "harvest_pull_axis_robot"), "harvest_pull_axis_robot")

    top_level_jaw = _unit(_vector(data, "gripper_jaw_axis_robot"), "gripper_jaw_axis_robot")
    top_level_backoff = _unit(_vector(data, "gripper_backoff_axis_robot"), "gripper_backoff_axis_robot")
    gripper_jaw_axis_local = str(data["gripper_jaw_axis_local"])
    gripper_backoff_axis_local = str(data["gripper_backoff_axis_local"])

    _assert_axis_close(rotation_task[:, 0], backoff_axis, "task +x backoff")
    _assert_axis_close(rotation_task[:, 1], jaw_axis, "task +y jaw")
    _assert_axis_close(rotation_task[:, 2], palm_axis, "task +z palm")
    _assert_axis_close(top_level_jaw, jaw_axis, "top-level jaw axis")
    _assert_axis_close(top_level_backoff, backoff_axis, "top-level backoff axis")
    _assert_axis_close(_local_axis(rotation_tcp, gripper_jaw_axis_local), jaw_axis, "TCP local jaw axis")
    _assert_axis_close(_local_axis(rotation_tcp, gripper_backoff_axis_local), backoff_axis, "TCP local backoff axis")

    contact_points = _optional_contact_points(target_record)
    contact_midpoint = _optional_contact_midpoint(target_record)
    if contact_points is not None:
        computed_midpoint = np.mean(contact_points, axis=0)
        if contact_midpoint is not None and float(np.linalg.norm(computed_midpoint - contact_midpoint)) > CONTACT_POINT_TOL_M:
            raise TaskFrameValidationError("contact_midpoint_robot_m does not match contact_points_robot_m")
        if float(np.linalg.norm(computed_midpoint - target_position)) > CONTACT_POINT_TOL_M:
            raise TaskFrameValidationError("contact midpoint does not match target_robot_m")
        contact_delta = contact_points[1] - contact_points[0]
        if abs(float(np.dot(_unit(contact_delta, "contact point delta"), jaw_axis))) < 1.0 - AXIS_TOL:
            raise TaskFrameValidationError("contact point delta is not aligned with jaw axis")
        if contact_midpoint is None:
            contact_midpoint = computed_midpoint

    if not bool(no_replay_contract.get("ok", False)):
        raise TaskFrameValidationError("no_replay_contract.ok must be true")
    if bool(no_replay_contract.get("recorded_trajectory_samples_used_for_robot_motion", True)):
        raise TaskFrameValidationError("task frame must not encode a recorded robot trajectory")

    target_index = int(data["target_index"])
    target_name = str(data.get("target_name") or data.get("target_label") or _target_name_from_path(source_path, target_index))
    return HarvestTaskFrame(
        target_name=target_name,
        source_path=source_path,
        target_index=target_index,
        target_position_robot_m=target_position,
        fruit_center_robot_m=fruit_center,
        rotation_robot_from_task=rotation_task,
        rotation_robot_from_tcp=rotation_tcp,
        jaw_axis_robot=jaw_axis,
        backoff_axis_robot=backoff_axis,
        pull_axis_robot=pull_axis,
        palm_axis_robot=palm_axis,
        contact_points_robot_m=contact_points,
        contact_midpoint_robot_m=contact_midpoint,
        gripper_jaw_axis_local=gripper_jaw_axis_local,
        gripper_backoff_axis_local=gripper_backoff_axis_local,
        recommended_pregrasp_aperture_m=_optional_float(quality, "recommended_pregrasp_aperture_m"),
        recommended_close_aperture_m=_optional_float(quality, "recommended_close_aperture_m"),
        estimated_grasp_width_m=_optional_float(quality, "estimated_grasp_width_m"),
        estimated_grasp_depth_m=_optional_float(quality, "estimated_grasp_depth_m"),
        hand_frame_weak=bool(quality.get("hand_frame_weak", False)),
        quality=dict(quality),
        no_replay_contract=dict(no_replay_contract),
    )


def discover_harvest_task_frame_paths(
    target_dir: Path = DEFAULT_TARGET_DIR,
    pattern: str = DEFAULT_TARGET_PATTERN,
) -> tuple[Path, ...]:
    paths = tuple(sorted(Path(target_dir).glob(pattern)))
    if not paths:
        raise TaskFrameValidationError(f"no task-frame JSON files found in {target_dir} matching {pattern!r}")
    return paths


def load_harvest_task_frames(
    paths: tuple[Path, ...] | None = None,
    *,
    target_dir: Path = DEFAULT_TARGET_DIR,
    pattern: str = DEFAULT_TARGET_PATTERN,
) -> tuple[HarvestTaskFrame, ...]:
    selected_paths = paths if paths is not None else discover_harvest_task_frame_paths(target_dir, pattern)
    return tuple(load_harvest_task_frame(path) for path in selected_paths)


def _required_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TaskFrameValidationError(f"{key} must be an object")
    return value


def _vector(data: dict[str, Any], key: str) -> Array:
    try:
        value = np.asarray(data[key], dtype=float).reshape(3)
    except KeyError as exc:
        raise TaskFrameValidationError(f"missing required key: {key}") from exc
    except ValueError as exc:
        raise TaskFrameValidationError(f"{key} must be a length-3 vector") from exc
    if not np.all(np.isfinite(value)):
        raise TaskFrameValidationError(f"{key} contains non-finite values")
    return value


def _vector4(data: dict[str, Any], key: str) -> Array:
    try:
        value = np.asarray(data[key], dtype=float).reshape(4)
    except KeyError as exc:
        raise TaskFrameValidationError(f"missing required key: {key}") from exc
    except ValueError as exc:
        raise TaskFrameValidationError(f"{key} must be a length-4 vector") from exc
    if not np.all(np.isfinite(value)):
        raise TaskFrameValidationError(f"{key} contains non-finite values")
    return value


def _rotation_matrix(data: dict[str, Any], key: str) -> Array:
    try:
        rotation = np.asarray(data[key], dtype=float).reshape(3, 3)
    except KeyError as exc:
        raise TaskFrameValidationError(f"missing required key: {key}") from exc
    except ValueError as exc:
        raise TaskFrameValidationError(f"{key} must be a 3x3 matrix") from exc
    _assert_rotation(rotation, key)
    return rotation


def _rotation_from_quat_wxyz(quat_wxyz: Array) -> Array:
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise TaskFrameValidationError("grasp_orientation_quat_wxyz norm is zero")
    w, x, y, z = q / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
            [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
            [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=float,
    )
    _assert_rotation(rotation, "grasp_orientation_quat_wxyz")
    return rotation


def _assert_rotation(rotation: Array, name: str) -> None:
    if rotation.shape != (3, 3):
        raise TaskFrameValidationError(f"{name} must be 3x3")
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > ROTATION_TOL or abs(determinant - 1.0) > ROTATION_TOL:
        raise TaskFrameValidationError(
            f"{name} is not a proper rotation: orthogonality_error={orthogonality_error}, det={determinant}"
        )


def _local_axis(rotation_robot_from_tcp: Array, local_axis: str) -> Array:
    sign_name = local_axis.lower().strip()
    mapping = {
        "plus-x": (1.0, 0),
        "minus-x": (-1.0, 0),
        "plus-y": (1.0, 1),
        "minus-y": (-1.0, 1),
        "plus-z": (1.0, 2),
        "minus-z": (-1.0, 2),
    }
    if sign_name not in mapping:
        raise TaskFrameValidationError(f"unsupported local axis {local_axis!r}")
    sign, index = mapping[sign_name]
    return sign * rotation_robot_from_tcp[:, index]


def _unit(vector: Array, name: str) -> Array:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise TaskFrameValidationError(f"{name} must be non-zero")
    return value / norm


def _assert_axis_close(actual: Array, expected: Array, name: str) -> None:
    dot = float(np.dot(_unit(actual, name), _unit(expected, name)))
    if dot < 1.0 - AXIS_TOL:
        raise TaskFrameValidationError(f"{name} mismatch: dot={dot}")


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    return float(data[key])


def _optional_contact_points(data: dict[str, Any]) -> Array | None:
    if "contact_points_robot_m" not in data or data["contact_points_robot_m"] is None:
        return None
    try:
        points = np.asarray(data["contact_points_robot_m"], dtype=float).reshape(2, 3)
    except ValueError as exc:
        raise TaskFrameValidationError("contact_points_robot_m must be a 2x3 matrix") from exc
    if not np.all(np.isfinite(points)):
        raise TaskFrameValidationError("contact_points_robot_m contains non-finite values")
    return points


def _optional_contact_midpoint(data: dict[str, Any]) -> Array | None:
    if "contact_midpoint_robot_m" not in data or data["contact_midpoint_robot_m"] is None:
        return None
    return _vector(data, "contact_midpoint_robot_m")


def _target_name_from_path(path: Path, target_index: int) -> str:
    match = re.search(r"(tomato\d+)", path.stem)
    if match:
        return match.group(1)
    return f"target{target_index}"


def _transform(rotation: Array, translation: Array) -> Array:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate harvest task-frame JSON files.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--pattern", default=DEFAULT_TARGET_PATTERN)
    args = parser.parse_args()

    paths = tuple(args.paths) if args.paths else discover_harvest_task_frame_paths(args.target_dir, args.pattern)
    frames = tuple(load_harvest_task_frame(path) for path in paths)
    print(json.dumps([frame.to_summary() for frame in frames], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
