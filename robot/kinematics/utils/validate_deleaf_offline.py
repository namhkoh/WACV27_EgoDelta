"""Offline (no-robot, no-motion) validation of the deleaf HYBRID plan for one cut label.

Calls deleaf_click_cut.py in --urdf offline mode (reference RB-Y1-a fw v1.1 URDF, proven to match
the live kinematics to <0.1mm on BD3) for a small grid of candidate cut points around the human's
recorded pre_cut tip, parses each run's hard_gate + stake clearance, and reports the best
gate-passing point (max stake clearance, tie-break smallest offset from the human petiole point).

This is an ORCHESTRATION wrapper, not a parallel solver: it only invokes the real pipeline and
reads its artifacts. Offline gates (cloud-off) exactly match the live collect (which is also
cloud-off via --selected-point-robot-m), and the live plan+gate before motion stays authoritative.

Usage:  python robot/kinematics/utils/validate_deleaf_offline.py --cut BD2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
APP = REPO / "robot" / "kinematics" / "apps" / "deleaf_click_cut.py"
URDF = REPO / "submodules" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model_v1.1.urdf"
PRIOR_DIR = REPO / "robot" / "kinematics" / "runs" / "060326" / "deleaf_teleop_prior"
ART_DIR = REPO / "robot" / "kinematics" / "runs" / "060326" / "deleaf_click_cut"

TUCKED = "robot/utils/deleaf_home_right_tucked.json"
HOMESTATE = "robot/utils/homestate.json"

# label -> (home, human pre_cut tip [x,y,z])
CUTS = {
    "AD1": (TUCKED, [0.659389, 0.128191, 1.270352]),
    "AD2": (TUCKED, [0.655384, 0.197540, 1.217248]),
    "BD1": (TUCKED, [0.628868, -0.244594, 1.246320]),
    "BD2": (TUCKED, [0.617366, -0.313959, 1.190105]),
    "BD3": (HOMESTATE, [0.604792, -0.315785, 1.003022]),  # the verified clearing click
    "BD4": (HOMESTATE, [0.619918, -0.285677, 0.876705]),
}

# tight local grid (m) around the petiole point; bias toward outboard/lower which cleared BD3.
DX = [0.0, -0.02]
DY = [0.0, -0.01, -0.02, 0.01]
DZ = [0.0, -0.02, -0.04]
STAKE_MIN = 0.015


def run_offline(label: str, home: str, point: np.ndarray, tag: str) -> dict:
    run_label = f"{label}_off_{tag}"
    cmd = [
        PY, str(APP),
        "--deleaf-config", "robot/kinematics/configs/deleaf_right_knife_default.json",
        "--dataset-name", "a270c038",
        "--home", home,
        "--toolfit", "robot/calibration/tooltips/a270c038_right_knife_tip_edge_repivot_060226.json",
        "--stakes-json", "robot/calibration/session/a270c038/stakes_robot.json",
        "--transform-json", "robot/calibration/session/a270c038/T_robot_from_aria_selected_current_best_target_toolfit.json",
        "--waypoint-policy", "teleop_offset",
        "--teleop-prior-json", str(PRIOR_DIR / f"a270c038_{label}_teleop_prior.json"),
        "--teleop-follow", "--teleop-follow-cut", "--pointcloud-gate-approach-only",
        "--mps-auto-select-action", "--safe-entry-standoff-m", "0",
        "--blade-clearance-radius-m", "0.02", "--stake-min-clearance-m", str(STAKE_MIN),
        "--urdf", str(URDF),
        "--selected-point-robot-m", "%.6f,%.6f,%.6f" % tuple(point),
        "--run-label", run_label,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    art = ART_DIR / f"a270c038_{run_label}.json"
    if not art.exists():
        return {"tag": tag, "ok": False, "error": (proc.stderr or proc.stdout or "no artifact")[-300:]}
    a = json.loads(art.read_text(encoding="utf-8"))
    sc = a.get("selected_candidate") or {}
    stk = (sc.get("stake_clearance") or {})
    wps = a.get("waypoints", [])
    return {
        "tag": tag,
        "ok": bool((a.get("hard_gate") or {}).get("ok")),
        "stake_mm": round(float(stk.get("minimum_distance_m", -9)) * 1000, 1),
        "worst_stake": stk.get("worst_stake"),
        "worst_segment": stk.get("worst_segment"),
        "n_waypoints": len(wps),
        "final_err_m": sc.get("final_position_error_m"),
        "all_wp_ok": all(w.get("ok") for w in wps) if wps else False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True, choices=sorted(CUTS))
    args = ap.parse_args()
    home, center = CUTS[args.cut]
    center = np.asarray(center, dtype=float)

    results = []
    # 1) test the human petiole point first
    r0 = run_offline(args.cut, home, center, "c00")
    r0["offset_mm"] = [0.0, 0.0, 0.0]
    results.append(r0)
    best = r0 if r0.get("ok") else None

    # 2) only sweep if the center failed
    if best is None:
        i = 0
        for dx in DX:
            for dy in DY:
                for dz in DZ:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    i += 1
                    pt = center + np.array([dx, dy, dz])
                    r = run_offline(args.cut, home, pt, f"g{i:02d}")
                    r["offset_mm"] = [round(dx * 1000, 1), round(dy * 1000, 1), round(dz * 1000, 1)]
                    results.append(r)
                    if r.get("ok"):
                        # prefer max stake clearance; among passers keep the best
                        if best is None or r["stake_mm"] > best["stake_mm"]:
                            best = dict(r)

    passers = [r for r in results if r.get("ok")]
    summary = {
        "cut": args.cut,
        "home": home,
        "human_point": [round(v, 6) for v in center.tolist()],
        "solved": best is not None,
        "best": best,
        "best_point": None if best is None else [round(v, 6) for v in (center + np.array(best["offset_mm"]) / 1000.0).tolist()],
        "n_points_tried": len(results),
        "n_passers": len(passers),
        "closest_miss_stake_mm": None if passers else max((r.get("stake_mm", -99) for r in results), default=None),
        "all_results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
