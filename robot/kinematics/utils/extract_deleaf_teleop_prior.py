from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robot.kinematics.geometry import normalize
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path


REPO_ROOT = Path(_REPO_ROOT)


def load_samples(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if len(samples) < 4:
        raise ValueError(f"teleop prior needs at least 4 samples: {path}")
    positions = np.asarray([sample["tcp"]["position_robot_m"] for sample in samples], dtype=float)
    q_arm = np.asarray([sample["q_arm_rad"] for sample in samples], dtype=float)
    return data, positions, q_arm


def extract_indices(positions: np.ndarray, *, pre_samples: int, approach_samples: int, retract_samples: int) -> dict[str, int]:
    downward_step = np.diff(positions[:, 2])
    cut_step_index = int(np.argmin(downward_step))
    pre_cut = max(0, cut_step_index - int(pre_samples))
    cut = min(len(positions) - 1, cut_step_index + 1)
    approach = max(0, pre_cut - int(approach_samples))
    retract = min(len(positions) - 1, cut + int(retract_samples))
    return {
        "approach": approach,
        "pre_cut": pre_cut,
        "cut": cut,
        "retract": retract,
        "cut_step": cut_step_index,
    }


def extract_indices_explicit(data: dict[str, Any]) -> dict[str, int]:
    """Phase indices from the per-sample 'phase' tags of a --phase-tagging capture.

    The END of each phase is that phase's representative waypoint (the pose the
    operator settled on before tapping Enter to advance).
    """
    samples = data.get("samples", [])
    phases = [s.get("phase") for s in samples]
    if not any(phases):
        raise ValueError("--use-explicit-phases: capture has no per-sample 'phase' tags (record with --phase-tagging)")
    indices: dict[str, int] = {}
    for name in ("approach", "pre_cut", "cut", "retract"):
        idxs = [i for i, p in enumerate(phases) if p == name]
        if not idxs:
            raise ValueError(f"--use-explicit-phases: no samples tagged '{name}'")
        indices[name] = idxs[-1]
    indices["cut_step"] = indices["cut"]
    return indices


def _joint_distance_resample(q_arm: np.ndarray, start: int, end: int, pin: int, max_hop_rad: float, max_points: int) -> list[int]:
    """Pick sample indices in [start,end] spaced by JOINT distance so each consecutive hop's largest
    single-joint move is <= max_hop_rad (kills the big per-waypoint jumps that come from index-uniform
    resampling of a fast human motion). The `pin` index (the exact cut sample) is always included.
    """
    if end <= start:
        return [start]
    qs = np.asarray(q_arm[start : end + 1], dtype=float)
    # cumulative arc length measured as the per-hop MAX single-joint delta
    arc = np.concatenate([[0.0], np.cumsum(np.max(np.abs(np.diff(qs, axis=0)), axis=1))])
    total = float(arc[-1])
    if total <= 1e-9:
        return [start, end]
    n = int(np.clip(int(np.ceil(total / max(max_hop_rad, 1e-3))) + 1, 2, int(max_points)))
    chosen = {int(start + int(np.argmin(np.abs(arc - t)))) for t in np.linspace(0.0, total, n)}
    chosen.add(int(start))
    chosen.add(int(end))
    chosen.add(int(pin))
    return sorted(chosen)


def build_prior(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data, positions, q_arm = load_samples(path)
    if getattr(args, "use_explicit_phases", False):
        indices = extract_indices_explicit(data)
    else:
        indices = extract_indices(
            positions,
            pre_samples=args.pre_samples,
            approach_samples=args.approach_samples,
            retract_samples=args.retract_samples,
        )
    cut_anchor = positions[indices["cut"]]
    approach_axis = normalize(cut_anchor - positions[indices["approach"]])
    cut_axis = normalize(positions[indices["cut"]] - positions[indices["pre_cut"]])
    retract_axis = normalize(positions[indices["retract"]] - positions[indices["cut"]])
    waypoints = {}
    for name in ("approach", "pre_cut", "cut", "retract"):
        index = indices[name]
        rotation = np.asarray(data["samples"][index]["tcp"]["rotation_robot_from_tcp"], dtype=float).reshape(3, 3)
        waypoints[name] = {
            "sample_index": int(index),
            "elapsed_s": float(data["samples"][index]["elapsed_s"]),
            "tcp_position_robot_m": positions[index].astype(float).tolist(),
            "rotation_robot_from_tcp": rotation.astype(float).tolist(),
            "tcp_minus_y_axis_robot": (-rotation[:, 1]).astype(float).tolist(),
            "tcp_plus_z_axis_robot": rotation[:, 2].astype(float).tolist(),
            "offset_from_cut_anchor_m": (positions[index] - cut_anchor).astype(float).tolist(),
            "q_arm_rad": q_arm[index].astype(float).tolist(),
        }
    # DENSE CUT-ACTION PATH (the operator's hybrid: IK plans the approach to pre_cut, then the solver
    # FOLLOWS the human's recorded pre_cut->cut->retract draw). Span is pre_cut..retract ONLY (NOT the
    # high approach arc, which re-anchored would blow the z-ceiling). Resampled by joint distance so
    # hops are bounded; the exact cut sample is pinned so the demonstrated through-cut is reproduced.
    cut_path: list[dict[str, Any]] = []
    if getattr(args, "use_explicit_phases", False):
        sel = _joint_distance_resample(
            q_arm,
            start=int(indices["pre_cut"]),
            end=int(indices["retract"]),
            pin=int(indices["cut"]),
            max_hop_rad=float(getattr(args, "follow_cut_max_hop_rad", 0.15)),
            max_points=int(getattr(args, "follow_cut_max_points", 40)),
        )
        for index in sel:
            rotation = np.asarray(data["samples"][index]["tcp"]["rotation_robot_from_tcp"], dtype=float).reshape(3, 3)
            cut_path.append({
                "sample_index": int(index),
                "phase": data["samples"][index].get("phase"),
                "elapsed_s": float(data["samples"][index]["elapsed_s"]),
                "is_cut": bool(index == int(indices["cut"])),
                "tcp_position_robot_m": positions[index].astype(float).tolist(),
                "rotation_robot_from_tcp": rotation.astype(float).tolist(),
                "offset_from_cut_anchor_m": (positions[index] - cut_anchor).astype(float).tolist(),
                "q_arm_rad": q_arm[index].astype(float).tolist(),
            })
    return {
        "method": "deleaf_teleop_prior_from_downward_cut_segment",
        "source": str(path),
        "dataset_name": data.get("dataset_name"),
        "arm": data.get("arm"),
        "sample_count": int(data.get("sample_count", len(data.get("samples", [])))),
        "stopped_by": data.get("stopped_by"),
        "indices": indices,
        "cut_anchor_robot_m": cut_anchor.astype(float).tolist(),
        "approach_axis_robot": approach_axis.astype(float).tolist(),
        "cut_axis_robot": cut_axis.astype(float).tolist(),
        "retract_axis_robot": retract_axis.astype(float).tolist(),
        "waypoints": waypoints,
        "cut_path": cut_path,
        "cut_path_meta": {
            "span": [int(indices["pre_cut"]), int(indices["retract"])],
            "point_count": len(cut_path),
            "pinned_cut_sample": int(indices["cut"]),
            "max_hop_rad": float(getattr(args, "follow_cut_max_hop_rad", 0.15)),
            "resample": "joint_distance_pre_cut_to_retract",
        } if cut_path else None,
        "diagnostics": {
            "cut_step_delta_m": (positions[indices["cut"]] - positions[indices["cut_step"]]).astype(float).tolist(),
            "cut_step_distance_m": float(np.linalg.norm(positions[indices["cut"]] - positions[indices["cut_step"]])),
            "tcp_min_robot_m": positions.min(axis=0).astype(float).tolist(),
            "tcp_max_robot_m": positions.max(axis=0).astype(float).tolist(),
            "pre_samples": int(args.pre_samples),
            "approach_samples": int(args.approach_samples),
            "retract_samples": int(args.retract_samples),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract reviewable deleafing waypoint priors from teleop TCP captures.")
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--dataset-name", default="a270c038")
    parser.add_argument("--run-label", default="deleaf_teleop_prior")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--pre-samples", type=int, default=4)
    parser.add_argument("--approach-samples", type=int, default=20)
    parser.add_argument("--retract-samples", type=int, default=20)
    parser.add_argument(
        "--use-explicit-phases",
        action="store_true",
        help="Use per-sample phase tags from a --phase-tagging teleop capture (end of each phase = its waypoint) "
        "instead of the dz-argmin heuristic.",
    )
    parser.add_argument(
        "--follow-cut-max-hop-rad",
        type=float,
        default=0.15,
        help="Dense cut-path (pre_cut->retract) is resampled so each hop's largest single-joint move is <= this (rad).",
    )
    parser.add_argument("--follow-cut-max-points", type=int, default=40, help="Cap on dense cut-path point count.")
    args = parser.parse_args()
    if not args.use_explicit_phases and min(args.pre_samples, args.approach_samples, args.retract_samples) < 1:
        parser.error("sample offsets must be positive")
    priors = [build_prior(path, args) for path in args.captures]
    payload = {
        "method": "deleaf_teleop_prior_collection",
        "dataset_name": args.dataset_name,
        "prior_count": len(priors),
        "priors": priors,
    }
    output = artifact_path("deleaf_teleop_prior", f"{args.dataset_name}_{args.run_label}", runs_root=args.runs_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "prior_count": len(priors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
