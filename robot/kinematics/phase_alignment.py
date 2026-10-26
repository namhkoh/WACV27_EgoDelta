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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robot.kinematics.hand_prior import DEFAULT_FRAME_INDEX
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.task_frame import (
    DEFAULT_TARGET_DIR,
    DEFAULT_TARGET_PATTERN,
    discover_harvest_task_frame_paths,
    load_harvest_task_frame,
)


HandSide = Literal["left", "right"]

DEFAULT_HANDS_FULL = Path("C:/path/to/projects/EgoDelta/data/policy_training/a270c038/hands_full_aria.json")
DEFAULT_PHASE_OFFSETS_S = {
    "approach": (-2.5, -0.6),
    "pre_grasp": (-0.6, -0.1),
    "grasp": (-0.1, 0.8),
    "harvest": (0.8, 2.5),
    "retract": (2.5, 3.4),
    "basket_place": (3.4, 7.9),
}


@dataclass(frozen=True)
class FrameTimestampIndex:
    frame_indices: np.ndarray
    timestamps_us: np.ndarray

    def nearest_frame(self, timestamp_us: float) -> int:
        index = int(np.searchsorted(self.timestamps_us, float(timestamp_us)))
        if index <= 0:
            return int(self.frame_indices[0])
        if index >= len(self.timestamps_us):
            return int(self.frame_indices[-1])
        before = index - 1
        after = index
        if abs(self.timestamps_us[before] - timestamp_us) <= abs(self.timestamps_us[after] - timestamp_us):
            return int(self.frame_indices[before])
        return int(self.frame_indices[after])

    def clamp_timestamp(self, timestamp_us: float) -> float:
        return float(np.clip(float(timestamp_us), self.timestamps_us[0], self.timestamps_us[-1]))


@dataclass(frozen=True)
class AriaHandSample:
    timestamp_us: float
    hand: HandSide
    tracking_confidence: float
    pinch_aria_m: np.ndarray
    pinch_aperture_m: float


@dataclass(frozen=True)
class PhaseWindowProposal:
    name: str
    start_us: float
    end_us: float
    start_frame_idx: int
    end_frame_idx: int
    sample_count: int
    mean_tracking_confidence: float | None
    median_pinch_aperture_m: float | None


@dataclass(frozen=True)
class HarvestPhaseAlignment:
    target_name: str
    target_path: Path
    hand_side: HandSide
    contact_timestamp_us: float
    contact_frame_idx: int
    contact_source: str
    windows: tuple[PhaseWindowProposal, ...]
    confidence: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "target_path": str(self.target_path),
            "hand_side": self.hand_side,
            "contact_timestamp_us": self.contact_timestamp_us,
            "contact_frame_idx": self.contact_frame_idx,
            "contact_source": self.contact_source,
            "windows": [window.__dict__ for window in self.windows],
            "confidence": self.confidence,
            "diagnostics": dict(self.diagnostics),
        }


def load_frame_timestamp_index(path: Path) -> FrameTimestampIndex:
    import csv

    frame_indices: list[int] = []
    timestamps_us: list[float] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_indices.append(int(row["frame_idx"]))
            timestamps_us.append(int(row["timestamp_ns"]) / 1000.0)
    if not frame_indices:
        raise ValueError(f"no frame timestamps in {path}")
    order = np.argsort(np.asarray(timestamps_us, dtype=float))
    return FrameTimestampIndex(
        frame_indices=np.asarray(frame_indices, dtype=int)[order],
        timestamps_us=np.asarray(timestamps_us, dtype=float)[order],
    )


def align_harvest_phases(
    *,
    task_frame_paths: tuple[Path, ...],
    frame_index_path: Path = DEFAULT_FRAME_INDEX,
    hands_full_path: Path = DEFAULT_HANDS_FULL,
    phase_offsets_s: dict[str, tuple[float, float]] | None = None,
    min_confidence: float = 0.5,
    runs_root: Path = RUNS_ROOT,
    run_label: str = "timestamp_contact",
    dataset_name: str = "dataset",
    write_artifacts: bool = True,
) -> tuple[HarvestPhaseAlignment, ...]:
    frame_index = load_frame_timestamp_index(frame_index_path)
    offsets = phase_offsets_s or DEFAULT_PHASE_OFFSETS_S
    alignments: list[HarvestPhaseAlignment] = []
    for path in task_frame_paths:
        alignment = align_harvest_phases_for_task_frame(
            path,
            frame_index=frame_index,
            hands_full_path=hands_full_path,
            phase_offsets_s=offsets,
            min_confidence=min_confidence,
        )
        if write_artifacts:
            output = artifact_path(
                "phase_alignment",
                f"{dataset_name}_{alignment.target_name}_{run_label}",
                runs_root=runs_root,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = alignment.to_dict()
            payload["dataset_name"] = dataset_name
            payload["artifact_path"] = str(output)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        alignments.append(alignment)
    return tuple(alignments)


def align_harvest_phases_for_task_frame(
    path: Path,
    *,
    frame_index: FrameTimestampIndex,
    hands_full_path: Path,
    phase_offsets_s: dict[str, tuple[float, float]],
    min_confidence: float,
) -> HarvestPhaseAlignment:
    frame = load_harvest_task_frame(path)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    target_aria_m = np.asarray(raw["target_aria_m"], dtype=float).reshape(3)
    hand_json_path = _hand_json_path(raw, hands_full_path)
    hand_samples = load_aria_hand_samples(hand_json_path)
    anchor = _detect_contact_anchor(hand_samples, target_aria_m, min_confidence)
    hand_contact = raw.get("hand_contact")
    contact_timestamp_us = anchor["timestamp_us"]
    hand_side = anchor["hand_side"]
    contact_frame_idx = frame_index.nearest_frame(contact_timestamp_us)
    windows = tuple(
        _window_from_offsets(
            name=name,
            offset_s=offset_s,
            contact_timestamp_us=contact_timestamp_us,
            frame_index=frame_index,
            samples=hand_samples,
            hand_side=hand_side,
        )
        for name, offset_s in phase_offsets_s.items()
    )
    confidence = _alignment_confidence(windows, contact_timestamp_us, frame_index)
    diagnostics = {
        "method": "target_aria_pinch_proximity_plus_action_phase_offsets",
        "phase_offsets_s": {key: list(value) for key, value in phase_offsets_s.items()},
        "frame_index_path": str(DEFAULT_FRAME_INDEX),
        "hand_json_path": str(hand_json_path),
        "target_index": frame.target_index,
        "target_aria_m": target_aria_m.tolist(),
        "detected_contact_distance_to_target_m": anchor["distance_to_target_m"],
        "detected_left_score_m": anchor["left_score_m"],
        "detected_right_score_m": anchor["right_score_m"],
    }
    if isinstance(hand_contact, dict) and "timestamp_us" in hand_contact:
        diagnostics["embedded_hand_contact_timestamp_delta_us"] = contact_timestamp_us - float(hand_contact["timestamp_us"])
        diagnostics["embedded_hand_contact_pinch_distance_to_target_m"] = hand_contact.get("pinch_distance_to_target_m")
        diagnostics["embedded_hand_contact_pinch_distance_to_mask_surface_m"] = hand_contact.get("pinch_distance_to_mask_surface_m")
    return HarvestPhaseAlignment(
        target_name=frame.target_name,
        target_path=Path(path),
        hand_side=hand_side,
        contact_timestamp_us=contact_timestamp_us,
        contact_frame_idx=contact_frame_idx,
        contact_source="target_aria_pinch_proximity",
        windows=windows,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def _window_from_offsets(
    *,
    name: str,
    offset_s: tuple[float, float],
    contact_timestamp_us: float,
    frame_index: FrameTimestampIndex,
    samples: list[AriaHandSample],
    hand_side: HandSide,
) -> PhaseWindowProposal:
    start_us = frame_index.clamp_timestamp(contact_timestamp_us + 1_000_000.0 * float(offset_s[0]))
    end_us = frame_index.clamp_timestamp(contact_timestamp_us + 1_000_000.0 * float(offset_s[1]))
    selected = [sample for sample in samples if start_us <= sample.timestamp_us <= end_us and _confidence(sample, hand_side) > 0.0]
    confidences = [_confidence(sample, hand_side) for sample in selected]
    apertures = [_pinch_aperture_m(sample, hand_side) for sample in selected]
    return PhaseWindowProposal(
        name=name,
        start_us=start_us,
        end_us=end_us,
        start_frame_idx=frame_index.nearest_frame(start_us),
        end_frame_idx=frame_index.nearest_frame(end_us),
        sample_count=len(selected),
        mean_tracking_confidence=float(np.mean(confidences)) if confidences else None,
        median_pinch_aperture_m=float(np.median(apertures)) if apertures else None,
    )


def _alignment_confidence(windows: tuple[PhaseWindowProposal, ...], contact_timestamp_us: float, frame_index: FrameTimestampIndex) -> float:
    sample_score = min(1.0, min((window.sample_count for window in windows), default=0) / 4.0)
    in_range = float(frame_index.timestamps_us[0] <= contact_timestamp_us <= frame_index.timestamps_us[-1])
    confidence_values = [
        window.mean_tracking_confidence
        for window in windows
        if window.mean_tracking_confidence is not None
    ]
    tracking_score = float(np.mean(confidence_values)) if confidence_values else 0.0
    return float(np.clip(sample_score * in_range * tracking_score, 0.0, 1.0))


def load_aria_hand_samples(path: Path) -> list[AriaHandSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples: list[AriaHandSample] = []
    for item in data.get("samples", []):
        hand = _hand_side(item.get("hand"))
        thumb = np.asarray(item["thumb_tip_aria_m"], dtype=float).reshape(3)
        index = np.asarray(item["index_tip_aria_m"], dtype=float).reshape(3)
        samples.append(
            AriaHandSample(
                timestamp_us=float(item["timestamp_us"]),
                hand=hand,
                tracking_confidence=float(item["tracking_confidence"]),
                pinch_aria_m=np.asarray(item["pinch_aria_m"], dtype=float).reshape(3),
                pinch_aperture_m=float(np.linalg.norm(index - thumb)),
            )
        )
    if not samples:
        raise ValueError(f"no Aria hand samples in {path}")
    return samples


def _detect_contact_anchor(samples: list[AriaHandSample], target_aria_m: np.ndarray, min_confidence: float) -> dict[str, Any]:
    side_results: dict[HandSide, tuple[float, AriaHandSample]] = {}
    for side in ("left", "right"):
        side_samples = [
            sample
            for sample in samples
            if sample.hand == side and sample.tracking_confidence >= min_confidence
        ]
        if not side_samples:
            continue
        distances = np.asarray([np.linalg.norm(sample.pinch_aria_m - target_aria_m) for sample in side_samples], dtype=float)
        best_index = int(np.argmin(distances))
        side_results[side] = (float(distances[best_index]), side_samples[best_index])
    if not side_results:
        raise ValueError("no confident hand samples for contact alignment")
    hand_side = min(side_results, key=lambda side: side_results[side][0])
    distance, sample = side_results[hand_side]
    return {
        "hand_side": hand_side,
        "timestamp_us": sample.timestamp_us,
        "distance_to_target_m": distance,
        "left_score_m": side_results.get("left", (None, None))[0],
        "right_score_m": side_results.get("right", (None, None))[0],
    }


def _hand_json_path(task_frame_json: dict[str, Any], fallback_path: Path) -> Path:
    value = task_frame_json.get("hand_json")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
    return Path(fallback_path)


def _hand_side(value: Any) -> HandSide:
    text = str(value).lower().strip()
    if text not in ("left", "right"):
        raise ValueError(f"unsupported hand side {value!r}")
    return text  # type: ignore[return-value]


def _confidence(sample: AriaHandSample, hand_side: HandSide) -> float:
    return sample.tracking_confidence if sample.hand == hand_side else -1.0


def _pinch_aperture_m(sample: AriaHandSample, hand_side: HandSide) -> float:
    if sample.hand != hand_side:
        return float("nan")
    return sample.pinch_aperture_m


def main() -> int:
    parser = argparse.ArgumentParser(description="Align harvest action phases to video frames using MPS timestamps.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--pattern", default=DEFAULT_TARGET_PATTERN)
    parser.add_argument("--frame-index", type=Path, default=DEFAULT_FRAME_INDEX)
    parser.add_argument("--hands-full", type=Path, default=DEFAULT_HANDS_FULL)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--run-label", default="timestamp_contact")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--no-write-artifacts", action="store_true")
    args = parser.parse_args()

    paths = tuple(args.paths) if args.paths else discover_harvest_task_frame_paths(args.target_dir, args.pattern)
    alignments = align_harvest_phases(
        task_frame_paths=paths,
        frame_index_path=args.frame_index,
        hands_full_path=args.hands_full,
        min_confidence=args.min_confidence,
        runs_root=args.runs_root,
        run_label=args.run_label,
        dataset_name=args.dataset_name or _dataset_name(args.target_dir, paths),
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps([alignment.to_dict() for alignment in alignments], indent=2))
    return 0


def _dataset_name(target_dir: Path, paths: tuple[Path, ...]) -> str:
    if paths:
        parents = {Path(path).parent.name for path in paths}
        if len(parents) == 1:
            return next(iter(parents))
    return Path(target_dir).name


if __name__ == "__main__":
    raise SystemExit(main())
