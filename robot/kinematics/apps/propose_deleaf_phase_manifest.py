"""Velocity-assisted proposer for the deleaf phase manifest (timestamp-based).

Given a recording's MPS hand tracking + SLAM trajectory, this auto-proposes per-cut
approach/cut/retract windows from the hand's speed profile (the hand decelerates/
dwells at a cut and accelerates on transit between stems). It emits a
`window_units: "timestamps_us"` phase manifest that
deleaf_click_cut.extract_deleaf_mps_guidance consumes, plus a verification CSV so a
human can confirm the cut count/timing.

Segmentation is frame-invariant (speed only), so it needs NEITHER egocentric frames
NOR the robot alignment -- it runs from hand_tracking + SLAM alone, in world frame.
If a robot alignment (+ its frame index) is supplied, the reported cut anchors are in
robot frame; otherwise they are world frame and flagged as such. The actual robot-frame
guidance axes/anchors are (re)computed by the extractor at use time once the alignment
exists.

Reuses load_hand_tracking / load_slam_trajectory / build_robot_frame_mapper /
load_frame_timestamps (hand_prior) and hand_point_device / confidence
(deleaf_click_cut). No robot connection, no motion. Output is a PROPOSAL to verify.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robot.kinematics.hand_prior import (
    build_robot_frame_mapper,
    load_frame_timestamps,
    load_hand_tracking,
    load_slam_trajectory,
)
from robot.kinematics.apps.deleaf_click_cut import confidence, hand_point_device


def _dominant_side(samples, t0_us, t1_us, min_conf) -> str:
    scores = {
        side: sum(
            1 for s in samples
            if t0_us <= s.timestamp_us <= t1_us and confidence(s, side) >= min_conf
        )
        for side in ("left", "right")
    }
    return "left" if scores["left"] >= scores["right"] else "right"


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _point_fn(args: argparse.Namespace, poses) -> tuple[Callable[[float, np.ndarray], np.ndarray], str, dict | None]:
    """Return (timestamp, device_point) -> point. Robot frame if alignment+frame-index, else world."""
    if args.alignment is not None and args.frame_index is not None:
        frame_ts = load_frame_timestamps(args.frame_index)
        mapper = build_robot_frame_mapper(args.alignment, frame_ts, poses)
        return (lambda ts, dp: mapper.point_to_robot(ts, dp)), "robot", frame_ts
    if args.frame_index is not None:
        frame_ts = load_frame_timestamps(args.frame_index)
    else:
        frame_ts = None

    def world(ts: float, dp: np.ndarray) -> np.ndarray:
        pose = poses.nearest(ts)
        return (pose.T_world_from_device @ np.r_[np.asarray(dp, float).reshape(3), 1.0])[:3]

    return world, "world", frame_ts


def propose(args: argparse.Namespace) -> dict[str, Any]:
    poses = load_slam_trajectory(args.slam_trajectory)
    samples = load_hand_tracking(args.hand_tracking)
    to_point, frame_label, frame_ts = _point_fn(args, poses)

    all_us = sorted(s.timestamp_us for s in samples)
    lo_us, hi_us = all_us[0], all_us[-1]
    if args.time_range_s is not None:
        lo_us = all_us[0] + args.time_range_s[0] * 1e6
        hi_us = all_us[0] + args.time_range_s[1] * 1e6
    elif args.frame_range is not None and frame_ts is not None:
        lo_us = frame_ts.get(int(args.frame_range[0]), lo_us)
        hi_us = frame_ts.get(int(args.frame_range[1]), hi_us)

    side = args.hand_side if args.hand_side in ("left", "right") else _dominant_side(samples, lo_us, hi_us, args.min_confidence)

    confident = [
        s for s in samples
        if lo_us <= s.timestamp_us <= hi_us and confidence(s, side) >= args.min_confidence
    ]
    confident.sort(key=lambda s: s.timestamp_us)
    if len(confident) < 5:
        raise RuntimeError(f"only {len(confident)} confident {side}-hand samples in range; cannot segment")

    ts_us = np.asarray([s.timestamp_us for s in confident], dtype=float)
    pos = np.vstack([to_point(s.timestamp_us, hand_point_device(s.row, side, args.hand_point)) for s in confident])
    t0 = ts_us[0]

    dt_s = np.diff(ts_us) / 1e6
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    seg_speed = np.divide(dp, dt_s, out=np.zeros_like(dp), where=dt_s > 1e-6)
    speed = np.r_[seg_speed, seg_speed[-1] if seg_speed.size else 0.0]
    speed = _smooth(speed, args.smooth_window)
    rate = len(confident) / max((ts_us[-1] - ts_us[0]) / 1e6, 1e-6)

    speed_low = float(args.speed_low_mps) if args.speed_low_mps is not None else float(np.percentile(speed, args.auto_low_percentile))

    # Low-speed dwells: extend a run ONLY while speed stays low; merge across short gaps.
    low = speed < speed_low
    runs: list[tuple[int, int]] = []
    i, n = 0, len(low)
    while i < n:
        if low[i]:
            j = i
            while j + 1 < n and low[j + 1]:
                j += 1
            if (ts_us[j] - ts_us[i]) / 1e6 >= args.min_dwell_s:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and (ts_us[a] - ts_us[merged[-1][1]]) / 1e6 < args.min_gap_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    lookback_us = args.approach_lookback_s * 1e6
    lookahead_us = args.retract_lookahead_s * 1e6
    cut_times = [(ts_us[a], ts_us[b]) for a, b in merged]

    actions: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    for k, ((a, b), (c_s, c_e)) in enumerate(zip(merged, cut_times), start=1):
        prev_c_e = cut_times[k - 2][1] if k >= 2 else lo_us
        next_c_s = cut_times[k][0] if k < len(cut_times) else hi_us
        approach_start = max(c_s - lookback_us, 0.5 * (prev_c_e + c_s))
        retract_end = min(c_e + lookahead_us, 0.5 * (c_e + next_c_s))
        name = f"deleafing_{k:02d}"
        actions.append({
            "name": name,
            "approach": [float(approach_start), float(c_s)],
            "cut": [float(c_s), float(c_e)],
            "retract": [float(c_e), float(retract_end)],
        })
        anchor = np.median(pos[a : b + 1], axis=0)
        debug.append({
            "name": name,
            "hand_side": side,
            "cut_anchor_m": anchor.round(4).tolist(),
            "cut_anchor_frame": frame_label,
            "cut_window_s": [round((c_s - t0) / 1e6, 2), round((c_e - t0) / 1e6, 2)],
            "cut_samples": int(b - a + 1),
            "cut_mean_speed_mps": float(round(speed[a : b + 1].mean(), 4)),
        })

    manifest = {
        "version": "0.2",
        "dataset": args.dataset,
        "window_units": "timestamps_us",
        "generator": "propose_deleaf_phase_manifest (velocity-assisted; VERIFY before use)",
        "mps_hand_tracking_csv": str(args.hand_tracking),
        "slam_trajectory_csv": str(args.slam_trajectory),
        "alignment": None if args.alignment is None else str(args.alignment),
        "anchor_frame": frame_label,
        "segmentation": {
            "hand_side": side,
            "hand_point": args.hand_point,
            "speed_low_mps": speed_low,
            "min_dwell_s": args.min_dwell_s,
            "min_gap_s": args.min_gap_s,
            "approach_lookback_s": args.approach_lookback_s,
            "retract_lookahead_s": args.retract_lookahead_s,
            "smooth_window": args.smooth_window,
            "sample_rate_hz": round(rate, 1),
            "confident_samples": len(confident),
        },
        "actions": actions,
        "proposal_debug": debug,
        "notes": [
            "AUTO-PROPOSED windows (timestamps_us) from the hand-speed profile -- verify",
            "the cut count/timing before use; anchors are in the 'anchor_frame' frame.",
        ],
    }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.output_csv is not None:
        label = ["none"] * len(confident)
        for k, (a, b) in enumerate(merged, start=1):
            for idx in range(a, b + 1):
                label[idx] = f"cut:{k}"
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", "elapsed_s", "timestamp_us", "speed_mps", "label"])
            for idx in range(len(confident)):
                writer.writerow([idx, round((ts_us[idx] - t0) / 1e6, 3), int(ts_us[idx]), round(float(speed[idx]), 4), label[idx]])

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-tracking", type=Path, required=True)
    parser.add_argument("--slam-trajectory", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, default=None, help="Optional robot alignment; if given with --frame-index, anchors are robot-frame.")
    parser.add_argument("--frame-index", type=Path, default=None, help="Optional frame_index.csv (only needed for robot-frame anchors or --frame-range).")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--hand-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--hand-point", choices=("wrist", "pinch"), default="wrist")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--speed-low-mps", type=float, default=None, help="Absolute low-speed threshold; default auto = percentile of the speed profile.")
    parser.add_argument("--auto-low-percentile", type=float, default=35.0)
    parser.add_argument("--min-dwell-s", type=float, default=0.25)
    parser.add_argument("--min-gap-s", type=float, default=0.6)
    parser.add_argument("--approach-lookback-s", type=float, default=1.5)
    parser.add_argument("--retract-lookahead-s", type=float, default=1.5)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--time-range-s", type=float, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--frame-range", type=int, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    manifest = propose(args)
    print(f"proposed {len(manifest['actions'])} cut(s) [anchor_frame={manifest['anchor_frame']}] -> {args.output_manifest}")
    for entry in manifest["proposal_debug"]:
        print(f"  {entry['name']}: side={entry['hand_side']} cut_window={entry['cut_window_s']}s "
              f"samples={entry['cut_samples']} mean_speed={entry['cut_mean_speed_mps']} anchor={entry['cut_anchor_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
