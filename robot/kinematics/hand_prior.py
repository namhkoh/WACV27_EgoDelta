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
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from robot.kinematics.types import Array, HandPrior


HandSide = Literal["left", "right"]
HandSideSelection = Literal["auto", "left", "right"]

DEFAULT_PHASE_PATH = Path("C:/path/to/projects/EgoDelta/data/policy_training/harvest_ht_frames.txt")
DEFAULT_FRAME_INDEX = Path("C:/path/to/projects/EgoDelta/data/policy_training/a270c038/pinhole_view/frames_rect/frame_index.csv")
DEFAULT_HAND_TRACKING = Path(
    "C:/path/to/user/aria/policy_training/"
    "mps_a270c038-8f08-4adb-98cc-18541bfb2165_vrs/hand_tracking/hand_tracking_results.csv"
)
DEFAULT_SLAM_TRAJECTORY = Path(
    "C:/path/to/user/aria/policy_training/"
    "mps_a270c038-8f08-4adb-98cc-18541bfb2165_vrs/slam/closed_loop_trajectory.csv"
)
DEFAULT_ALIGNMENT = Path("robot/calibration/session/a270c038/T_robot_from_aria_selected_current_best_target_toolfit.json")

PHASE_NAMES = ("approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place")
THUMB_TIP = 4
INDEX_TIP = 8


@dataclass(frozen=True)
class PhaseWindow:
    name: str
    start_frame_idx: int
    end_frame_idx: int


@dataclass(frozen=True)
class TimeWindow:
    name: str
    start_us: float
    end_us: float


@dataclass(frozen=True)
class DevicePoseSample:
    timestamp_us: float
    T_world_from_device: Array


@dataclass(frozen=True)
class HandTrackingSample:
    timestamp_us: float
    left_confidence: float
    right_confidence: float
    row: dict[str, str]


@dataclass(frozen=True)
class HandPriorExtraction:
    target_name: str
    hand_side: HandSide
    prior: HandPrior
    phase_windows: tuple[PhaseWindow, ...]
    time_windows: tuple[TimeWindow, ...]
    diagnostics: dict[str, float | int | str | bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "hand_side": self.hand_side,
            "prior": {
                "approach_axis_robot": self.prior.approach_axis_robot.tolist(),
                "jaw_axis_robot": self.prior.jaw_axis_robot.tolist(),
                "pull_axis_robot": self.prior.pull_axis_robot.tolist(),
                "confidence": self.prior.confidence,
                "source": self.prior.source,
            },
            "phase_windows": [window.__dict__ for window in self.phase_windows],
            "time_windows": [window.__dict__ for window in self.time_windows],
            "diagnostics": dict(self.diagnostics),
        }


class PoseLookup:
    def __init__(self, samples: list[DevicePoseSample]) -> None:
        if not samples:
            raise ValueError("SLAM trajectory has no pose samples")
        self.samples = tuple(sorted(samples, key=lambda sample: sample.timestamp_us))
        self.timestamps_us = np.asarray([sample.timestamp_us for sample in self.samples], dtype=float)

    def nearest(self, timestamp_us: float) -> DevicePoseSample:
        index = int(np.searchsorted(self.timestamps_us, float(timestamp_us)))
        if index <= 0:
            return self.samples[0]
        if index >= len(self.samples):
            return self.samples[-1]
        before = self.samples[index - 1]
        after = self.samples[index]
        if abs(before.timestamp_us - timestamp_us) <= abs(after.timestamp_us - timestamp_us):
            return before
        return after


@dataclass(frozen=True)
class RobotFrameMapper:
    T_robot_from_anchor: Array
    T_anchor_from_world: Array
    pose_lookup: PoseLookup

    def point_to_robot(self, timestamp_us: float, point_device_m: Array) -> Array:
        pose = self.pose_lookup.nearest(timestamp_us)
        point_anchor = self.T_anchor_from_world @ pose.T_world_from_device @ _homogeneous(point_device_m)
        return (self.T_robot_from_anchor @ point_anchor)[:3]

    def vector_to_robot(self, timestamp_us: float, vector_device: Array) -> Array:
        pose = self.pose_lookup.nearest(timestamp_us)
        rotation = self.T_robot_from_anchor[:3, :3] @ self.T_anchor_from_world[:3, :3] @ pose.T_world_from_device[:3, :3]
        return rotation @ np.asarray(vector_device, dtype=float).reshape(3)


def load_phase_windows(path: Path) -> dict[str, tuple[PhaseWindow, ...]]:
    target_name: str | None = None
    phase_name: str | None = None
    parsed: dict[str, dict[str, dict[str, int]]] = {}

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lower()
        if not line:
            continue
        target_match = re.fullmatch(r"tomato\s+(\d+)", line)
        if target_match:
            target_name = f"tomato{target_match.group(1)}"
            parsed[target_name] = {}
            phase_name = None
            continue
        if line in {"approach", "pre_grasp", "grasp", "harvest", "retract", "basket place"}:
            if target_name is None:
                raise ValueError(f"phase {line!r} appears before a tomato target")
            phase_name = line.replace(" ", "_")
            parsed[target_name][phase_name] = {}
            continue
        frame_match = re.search(r"(start|end):.*frame_(\d+)\.jpg", line)
        if frame_match:
            if target_name is None or phase_name is None:
                raise ValueError(f"frame boundary appears before target and phase: {raw_line!r}")
            parsed[target_name][phase_name][frame_match.group(1)] = int(frame_match.group(2))

    windows: dict[str, tuple[PhaseWindow, ...]] = {}
    for target, phases in parsed.items():
        target_windows: list[PhaseWindow] = []
        for phase in PHASE_NAMES:
            if phase not in phases:
                raise ValueError(f"{target} missing phase {phase}")
            bounds = phases[phase]
            if "start" not in bounds or "end" not in bounds:
                raise ValueError(f"{target} {phase} missing start or end frame")
            if bounds["start"] > bounds["end"]:
                raise ValueError(f"{target} {phase} start frame is after end frame")
            target_windows.append(
                PhaseWindow(
                    name=phase,
                    start_frame_idx=int(bounds["start"]),
                    end_frame_idx=int(bounds["end"]),
                )
            )
        windows[target] = tuple(target_windows)
    return windows


def load_frame_timestamps(path: Path) -> dict[int, float]:
    timestamps: dict[int, float] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamps[int(row["frame_idx"])] = int(row["timestamp_ns"]) / 1000.0
    return timestamps


def load_hand_tracking(path: Path) -> list[HandTrackingSample]:
    samples: list[HandTrackingSample] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                HandTrackingSample(
                    timestamp_us=float(row["tracking_timestamp_us"]),
                    left_confidence=float(row["left_tracking_confidence"]),
                    right_confidence=float(row["right_tracking_confidence"]),
                    row=row,
                )
            )
    return samples


def load_slam_trajectory(path: Path) -> PoseLookup:
    samples: list[DevicePoseSample] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = _rotation_from_xyzw(
                np.asarray(
                    [
                        float(row["qx_world_device"]),
                        float(row["qy_world_device"]),
                        float(row["qz_world_device"]),
                        float(row["qw_world_device"]),
                    ],
                    dtype=float,
                )
            )
            transform[:3, 3] = [
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ]
            samples.append(DevicePoseSample(timestamp_us=float(row["tracking_timestamp_us"]), T_world_from_device=transform))
    return PoseLookup(samples)


def phase_time_windows(phase_windows: tuple[PhaseWindow, ...], frame_timestamps_us: dict[int, float]) -> tuple[TimeWindow, ...]:
    time_windows: list[TimeWindow] = []
    for window in phase_windows:
        try:
            start_us = frame_timestamps_us[window.start_frame_idx]
            end_us = frame_timestamps_us[window.end_frame_idx]
        except KeyError as exc:
            raise ValueError(f"missing timestamp for frame {exc.args[0]}") from exc
        if start_us > end_us:
            raise ValueError(f"{window.name} start timestamp is after end timestamp")
        time_windows.append(TimeWindow(window.name, float(start_us), float(end_us)))
    return tuple(time_windows)


def build_robot_frame_mapper(alignment_path: Path, frame_timestamps_us: dict[int, float] | None, pose_lookup: PoseLookup) -> RobotFrameMapper:
    alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
    T_robot_from_aria = np.asarray(alignment["T_robot_from_aria"], dtype=float)
    if T_robot_from_aria.shape != (4, 4):
        raise ValueError("T_robot_from_aria must be 4x4")
    if str(alignment.get("mode", "")) == "world_direct":
        # T_robot_from_aria maps Aria SLAM world straight to robot; the per-sample device
        # pose (T_world_from_device) handles device->world, so the anchor is the identity
        # and no frame index / frame-timestamp map is needed.
        return RobotFrameMapper(
            T_robot_from_anchor=T_robot_from_aria,
            T_anchor_from_world=np.eye(4, dtype=float),
            pose_lookup=pose_lookup,
        )
    if frame_timestamps_us is None:
        raise ValueError("anchor-frame alignment requires frame_timestamps_us (frame index)")
    anchor_frame_idx = int(alignment["frame_idx"])
    if anchor_frame_idx not in frame_timestamps_us:
        raise ValueError(f"alignment anchor frame {anchor_frame_idx} is missing from frame timestamp map")
    anchor_pose = pose_lookup.nearest(frame_timestamps_us[anchor_frame_idx])
    return RobotFrameMapper(
        T_robot_from_anchor=T_robot_from_aria,
        T_anchor_from_world=np.linalg.inv(anchor_pose.T_world_from_device),
        pose_lookup=pose_lookup,
    )


def extract_hand_prior(
    *,
    target_name: str,
    phase_path: Path = DEFAULT_PHASE_PATH,
    frame_index_path: Path = DEFAULT_FRAME_INDEX,
    hand_tracking_path: Path = DEFAULT_HAND_TRACKING,
    slam_trajectory_path: Path = DEFAULT_SLAM_TRAJECTORY,
    alignment_path: Path = DEFAULT_ALIGNMENT,
    hand_side: HandSideSelection = "auto",
    min_confidence: float = 0.2,
) -> HandPriorExtraction:
    phase_windows_by_target = load_phase_windows(phase_path)
    if target_name not in phase_windows_by_target:
        raise ValueError(f"target {target_name!r} is not present in {phase_path}")

    phase_windows = phase_windows_by_target[target_name]
    frame_timestamps_us = load_frame_timestamps(frame_index_path)
    time_windows = phase_time_windows(phase_windows, frame_timestamps_us)
    samples = load_hand_tracking(hand_tracking_path)
    pose_lookup = load_slam_trajectory(slam_trajectory_path)
    mapper = build_robot_frame_mapper(alignment_path, frame_timestamps_us, pose_lookup)

    side_scores = _hand_side_scores(samples, time_windows, mapper, min_confidence)
    selected_side = _select_hand_side(hand_side, side_scores)
    approach_samples = _phase_samples(samples, time_windows, ("approach", "pre_grasp"), selected_side, min_confidence)
    jaw_samples = _phase_samples(samples, time_windows, ("pre_grasp", "grasp"), selected_side, min_confidence)
    harvest_samples = _phase_samples(samples, time_windows, ("harvest",), selected_side, min_confidence)

    approach_axis, approach_confidence, approach_motion_m = _motion_axis(approach_samples, selected_side, mapper)
    jaw_axis, jaw_confidence, pinch_width_m = _jaw_axis(jaw_samples, selected_side, mapper)
    pull_axis, pull_confidence, pull_motion_m = _motion_axis(harvest_samples, selected_side, mapper)

    confidence = float(np.clip(np.mean([approach_confidence, jaw_confidence, pull_confidence]), 0.0, 1.0))
    source = f"MPS hand_tracking={Path(hand_tracking_path).name}; slam={Path(slam_trajectory_path).name}; side={selected_side}"
    prior = HandPrior(
        approach_axis_robot=approach_axis,
        jaw_axis_robot=jaw_axis,
        pull_axis_robot=pull_axis,
        confidence=confidence,
        source=source,
    )
    diagnostics: dict[str, float | int | str | bool] = {
        "hand_side_requested": hand_side,
        "hand_side_selected": selected_side,
        "left_hand_score": side_scores["left"],
        "right_hand_score": side_scores["right"],
        "approach_samples": len(approach_samples),
        "jaw_samples": len(jaw_samples),
        "harvest_samples": len(harvest_samples),
        "approach_confidence": approach_confidence,
        "jaw_confidence": jaw_confidence,
        "pull_confidence": pull_confidence,
        "approach_motion_m": approach_motion_m,
        "pull_motion_m": pull_motion_m,
        "median_pinch_width_m": pinch_width_m,
    }
    return HandPriorExtraction(
        target_name=target_name,
        hand_side=selected_side,
        prior=prior,
        phase_windows=phase_windows,
        time_windows=time_windows,
        diagnostics=diagnostics,
    )


def _select_hand_side(
    requested: HandSideSelection,
    side_scores: dict[HandSide, float],
) -> HandSide:
    if requested in ("left", "right"):
        return requested
    return "left" if side_scores["left"] >= side_scores["right"] else "right"


def _hand_side_scores(
    samples: list[HandTrackingSample],
    time_windows: tuple[TimeWindow, ...],
    mapper: RobotFrameMapper,
    min_confidence: float,
) -> dict[HandSide, float]:
    scores: dict[HandSide, float] = {}
    for side in ("left", "right"):
        approach = _phase_samples(samples, time_windows, ("approach", "pre_grasp"), side, min_confidence)
        jaw = _phase_samples(samples, time_windows, ("pre_grasp", "grasp"), side, min_confidence)
        harvest = _phase_samples(samples, time_windows, ("harvest",), side, min_confidence)
        _, approach_confidence, _ = _motion_axis(approach, side, mapper)
        _, jaw_confidence, _ = _jaw_axis(jaw, side, mapper)
        _, pull_confidence, _ = _motion_axis(harvest, side, mapper)
        scores[side] = float(np.mean([approach_confidence, jaw_confidence, pull_confidence]))
    return scores


def _phase_samples(
    samples: list[HandTrackingSample],
    time_windows: tuple[TimeWindow, ...],
    phase_names: tuple[str, ...],
    side: HandSide,
    min_confidence: float,
) -> list[HandTrackingSample]:
    windows = [window for window in time_windows if window.name in set(phase_names)]
    selected: list[HandTrackingSample] = []
    for sample in samples:
        if _confidence(sample, side) < min_confidence:
            continue
        if any(window.start_us <= sample.timestamp_us <= window.end_us for window in windows):
            selected.append(sample)
    return selected


def _motion_axis(samples: list[HandTrackingSample], side: HandSide, mapper: RobotFrameMapper) -> tuple[Array, float, float]:
    if len(samples) < 2:
        return np.asarray([1.0, 0.0, 0.0], dtype=float), 0.0, 0.0
    ordered = sorted(samples, key=lambda sample: sample.timestamp_us)
    points = np.vstack([mapper.point_to_robot(sample.timestamp_us, _pinch_center_device(sample.row, side)) for sample in ordered])
    split = max(1, len(points) // 3)
    start = np.median(points[:split], axis=0)
    end = np.median(points[-split:], axis=0)
    vector = end - start
    motion_m = float(np.linalg.norm(vector))
    confidence = _motion_confidence(motion_m, len(samples))
    if motion_m <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0], dtype=float), 0.0, 0.0
    return vector / motion_m, confidence, motion_m


def _jaw_axis(samples: list[HandTrackingSample], side: HandSide, mapper: RobotFrameMapper) -> tuple[Array, float, float]:
    vectors: list[Array] = []
    widths: list[float] = []
    for sample in samples:
        thumb = _landmark_device(sample.row, side, THUMB_TIP)
        index = _landmark_device(sample.row, side, INDEX_TIP)
        axis_device = index - thumb
        width = float(np.linalg.norm(axis_device))
        if width <= 1e-9:
            continue
        vectors.append(_normalize(mapper.vector_to_robot(sample.timestamp_us, axis_device)))
        widths.append(width)
    if not vectors:
        return np.asarray([0.0, 1.0, 0.0], dtype=float), 0.0, 0.0
    vector_stack = np.vstack(vectors)
    scatter = vector_stack.T @ vector_stack / float(len(vectors))
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    mean_axis = _normalize(eigenvectors[:, int(np.argmax(eigenvalues))])
    if float(np.dot(mean_axis, vector_stack[0])) < 0.0:
        mean_axis = -mean_axis
    concentration = float(np.max(eigenvalues))
    confidence = float(np.clip(concentration * min(1.0, len(vectors) / 10.0), 0.0, 1.0))
    return mean_axis, confidence, float(np.median(widths))


def _motion_confidence(motion_m: float, sample_count: int) -> float:
    motion_score = min(1.0, motion_m / 0.05)
    sample_score = min(1.0, sample_count / 10.0)
    return float(np.clip(math.sqrt(motion_score * sample_score), 0.0, 1.0))


def _confidence(sample: HandTrackingSample, side: HandSide) -> float:
    return sample.left_confidence if side == "left" else sample.right_confidence


def _pinch_center_device(row: dict[str, str], side: HandSide) -> Array:
    return 0.5 * (_landmark_device(row, side, THUMB_TIP) + _landmark_device(row, side, INDEX_TIP))


def _landmark_device(row: dict[str, str], side: HandSide, index: int) -> Array:
    return np.asarray(
        [
            float(row[f"tx_{side}_landmark_{index}_device"]),
            float(row[f"ty_{side}_landmark_{index}_device"]),
            float(row[f"tz_{side}_landmark_{index}_device"]),
        ],
        dtype=float,
    )


def _homogeneous(point: Array) -> Array:
    return np.r_[np.asarray(point, dtype=float).reshape(3), 1.0]


def _rotation_from_xyzw(quaternion: Array) -> Array:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
            [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
            [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=float,
    )


def _normalize(vector: Array) -> Array:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return value / norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract an MPS hand prior for a harvest target.")
    parser.add_argument("target_name", help="Target name from the phase file, for example tomato0 or tomato1.")
    parser.add_argument("--phase-path", type=Path, default=DEFAULT_PHASE_PATH)
    parser.add_argument("--frame-index", type=Path, default=DEFAULT_FRAME_INDEX)
    parser.add_argument("--hand-tracking", type=Path, default=DEFAULT_HAND_TRACKING)
    parser.add_argument("--slam-trajectory", type=Path, default=DEFAULT_SLAM_TRAJECTORY)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--hand-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--min-confidence", type=float, default=0.2)
    args = parser.parse_args()

    extraction = extract_hand_prior(
        target_name=args.target_name,
        phase_path=args.phase_path,
        frame_index_path=args.frame_index,
        hand_tracking_path=args.hand_tracking,
        slam_trajectory_path=args.slam_trajectory,
        alignment_path=args.alignment,
        hand_side=args.hand_side,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(extraction.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
