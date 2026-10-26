from __future__ import annotations

"""Offline ablation: Full EgoDelta robot actions (WITH the Aria MPS human-motion prior)
vs the SAME actions MINUS the MPS prior.

The question: does conditioning the robot action on the demonstrated human motion (the Aria
MPS hand-tracking prior) produce action axes/orientation closer to the human than the
prior-free default? We score BOTH arms' produced action against the independent human MPS
demonstration and report the per-task, per-action, per-arm deviation.

Two tasks (run harvest first, then deleaf):

  HARVEST -- grasp axes (no robot, no IK needed).
    WITH  = MPS hand prior, extract_hand_prior() approach/jaw/pull axes in robot frame.
    MINUS = the generic robot-front fallback the harvest primitive uses when no prior is
            available (episode_manager.harvest_primitive_prior_from_task_frames fallback:
            backoff[-1,0,0], jaw[0,1,0], palm[0,0,-1], pull=norm[-0.70,-0.70,-0.10]).
    Human reference = the MPS hand-prior axes. WITH reproduces them (~0 by construction --
    that IS applying the prior); MINUS is the measured deviation.

  DELEAF -- cut waypoints. Both arms share the SAME demonstrated cut anchor, SAME IK solver,
    SAME URDF kinematics; the ONLY difference is the stroke/orientation SOURCE.
    WITH  = mps_template + MPS stroke axis (stroke_axis_policy, default 'mps') + mps_axes.
    MINUS = vertical straight-down cut (--vertical-cut) with --force-approach-axis-robot
            1,0,0 so NO MPS component leaks into the blade heading. Proven MPS-off via the
            diagnostics stroke_axis_source=='vertical_cut_world_down' /
            task_orientation_source=='vertical_straight_down' (NOT the always-written
            mps_guidance block).
    Human reference = DeleafMpsGuidance axes (cut_axis_robot, approach_axis_robot) and the
    human TCP rotation deleaf_rotation(approach, cut).

Fully offline: harvest needs no robot; deleaf uses RBY1Kinematics.from_urdf (reference
RB-Y1-a fw v1.1 URDF, proven to match live FK to <0.1mm). This module NEVER commands the
robot and NEVER opens a camera.

Usage:
  python robot/kinematics/apps/egodelta_mps_ablation.py --task both
  python robot/kinematics/apps/egodelta_mps_ablation.py --task deleaf --with-stroke-policy downward_constrained
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from robot.kinematics.apps.ablate_deleaf_ik_vs_mps import _planner_parser  # noqa: E402
from robot.kinematics.apps.deleaf_click_cut import (  # noqa: E402
    build_deleaf_waypoints,
    extract_deleaf_mps_guidance,
    load_deleaf_tool_axes,
)
from robot.kinematics.clearance import nearest_self_clearances  # noqa: E402
from robot.kinematics.geometry import normalize, orientation_residual  # noqa: E402
from robot.kinematics.hand_prior import extract_hand_prior  # noqa: E402
from robot.kinematics.ik_solver import _axis_error_rad, solve_tcp_task_axis_ik  # noqa: E402
from robot.kinematics.rby1_model import RBY1Kinematics  # noqa: E402
from robot.kinematics.types import IKSolverConfig  # noqa: E402

# --------------------------------------------------------------------------------------
# Recording-specific paths (a270c038). All offline.
# --------------------------------------------------------------------------------------
URDF = REPO / "submodules" / "rby1-sdk" / "models" / "rby1a" / "urdf" / "model_v1.1.urdf"
ALIGNMENT = REPO / "robot" / "calibration" / "session" / "a270c038" / "T_robot_from_aria_world_direct.json"
DELEAF_TOOLFIT = REPO / "robot" / "calibration" / "tooltips" / "a270c038_right_knife_tip_edge_repivot_060226.json"
DELEAF_HOME = REPO / "robot" / "utils" / "deleaf_home_right_tucked.json"
DELEAF_MANIFEST = REPO / "data" / "policy_training" / "a270c038" / "selections_deleafing" / "phase_manifest.json"
SLAM = Path(
    "C:/path/to/user/aria/policy_training/"
    "mps_a270c038-8f08-4adb-98cc-18541bfb2165_vrs/slam/closed_loop_trajectory.csv"
)
DEFAULT_OUTPUT = REPO / "robot" / "kinematics" / "runs" / "egodelta_mps_ablation" / "egodelta_mps_ablation.json"

# Harvest MINUS-MPS = generic robot-front fallback grasp frame (no MPS prior available).
# Mirrors episode_manager.harvest_primitive_prior_from_task_frames generic_robot_front_fallback.
FALLBACK_BACKOFF = np.array([-1.0, 0.0, 0.0])
FALLBACK_JAW = np.array([0.0, 1.0, 0.0])
FALLBACK_PALM = np.array([0.0, 0.0, -1.0])
FALLBACK_PULL = normalize(np.array([-0.70, -0.70, -0.10]))

HARVEST_TARGETS = ("tomato0", "tomato1")
DELEAF_ACTIONS = ("deleafing_01", "deleafing_02", "deleafing_03")

# Reachable workspace ceiling (m) for the cut POINT flag. The solve itself uses a relaxed
# box so the high transit/safe-entry waypoints do not early-break the warm-started chain.
WORKSPACE_CEILING_Z_M = 1.25
SOLVE_MAX_Z_M = 1.50


def _u(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def angle_deg(u: np.ndarray, v: np.ndarray, *, bidirectional: bool = False) -> float:
    a = float(np.degrees(np.arccos(np.clip(float(_u(u) @ _u(v)), -1.0, 1.0))))
    if bidirectional and a > 90.0:
        a = 180.0 - a
    return a


# --------------------------------------------------------------------------------------
# HARVEST: grasp-axis comparison (offline, no robot, no IK).
# --------------------------------------------------------------------------------------
def harvest_ablation(*, hand_side: str = "left", arms: tuple[str, ...] = ("WITH", "MINUS")) -> list[dict]:
    rows: list[dict] = []
    for target in HARVEST_TARGETS:
        extraction = extract_hand_prior(target_name=target, alignment_path=ALIGNMENT, hand_side=hand_side)
        prior = extraction.prior
        human_approach = _u(prior.approach_axis_robot)
        human_jaw = _u(prior.jaw_axis_robot)
        human_pull = _u(prior.pull_axis_robot)

        # MINUS-MPS robot-front fallback axes, mapped to the same names as the human prior.
        minus_approach = -_u(FALLBACK_BACKOFF)  # backoff is the reverse of approach
        minus_jaw = _u(FALLBACK_JAW)
        minus_pull = _u(FALLBACK_PULL)

        for arm, app, jaw, pull, source in (
            ("WITH", human_approach, human_jaw, human_pull, "mps_hand_prior"),
            ("MINUS", minus_approach, minus_jaw, minus_pull, "generic_robot_front_fallback"),
        ):
            if arm not in arms:
                continue
            rows.append(
                {
                    "task": "harvest",
                    "action": target,
                    "arm": arm,
                    "approach_axis_err_deg": angle_deg(app, human_approach),
                    "jaw_axis_err_deg": angle_deg(jaw, human_jaw, bidirectional=True),
                    "pull_axis_err_deg": angle_deg(pull, human_pull),
                    "prior_source": source,
                    "hand_side": extraction.hand_side,
                    "confidence": float(prior.confidence),
                    "approach_axis_robot": app.tolist(),
                    "jaw_axis_robot": jaw.tolist(),
                    "pull_axis_robot": pull.tolist(),
                    "human_approach_axis_robot": human_approach.tolist(),
                    "human_jaw_axis_robot": human_jaw.tolist(),
                    "human_pull_axis_robot": human_pull.tolist(),
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# DELEAF: cut-waypoint comparison (offline via URDF FK/IK).
# --------------------------------------------------------------------------------------
def _deleaf_planner_args(**overrides) -> argparse.Namespace:
    pa = _planner_parser().parse_args([])
    pa.manifest = DELEAF_MANIFEST
    pa.alignment = ALIGNMENT
    pa.hand_side = "right"
    pa.hand_point = "pinch"
    pa.toolfit = DELEAF_TOOLFIT
    pa.home = DELEAF_HOME
    pa.target_extension_values_m = "0"  # plural form -- the singular -m is silently ignored
    for key, value in overrides.items():
        setattr(pa, key, value)
    return pa


def _solve_one_arm(
    *, arm: str, overrides: dict, guidance, cut_point, home_tcp, tool_axes, kinematics, locked_state
) -> dict:
    """Build one deleaf arm's waypoints + diagnostics, and solve the CUT pose in isolation.

    The cut pose is solved on its own (seeded from the tucked home, identical for both arms)
    so the reachability comparison is fair and free of the warm-started chain's early-break.
    """
    pa = _deleaf_planner_args(max_z_m=SOLVE_MAX_Z_M, **overrides)
    config = IKSolverConfig(
        position_weight=float(pa.position_weight),
        orientation_weight=float(pa.orientation_weight),
        posture_weight=float(pa.posture_weight),
        smoothness_weight=float(pa.smoothness_weight),
        clearance_weight=float(pa.clearance_weight),
        clearance_margin_m=float(pa.clearance_margin_m),
        clearance_pair_count=int(pa.clearance_pair_count),
        max_iterations=int(pa.max_iterations),
    )
    waypoints, diag = build_deleaf_waypoints(
        cut_point_robot_m=cut_point,
        home_tcp_robot_m=home_tcp,
        guidance=guidance,
        teleop_prior=None,
        tool_axes=tool_axes,
        args=pa,
    )
    cut_wp = next(w for w in waypoints if w.name == "cut")
    local_axes = diag.get("tool_axis_local", {})
    primary_local = np.asarray(local_axes.get("primary_axis_local", [0.0, -1.0, 0.0]), dtype=float)
    secondary_local = np.asarray(local_axes.get("secondary_axis_local", [0.0, 0.0, 1.0]), dtype=float)
    cut_targets = diag.get("axis_targets_by_name", {}).get("cut", {})
    jaw_axis_robot = np.asarray(cut_targets.get("primary_axis_robot", diag["constrained_cut_axis_robot"]), dtype=float)
    backoff_axis_robot = np.asarray(
        cut_targets.get("secondary_axis_robot", -np.asarray(diag["approach_axis_robot"], dtype=float)), dtype=float
    )
    home_q = np.asarray(locked_state.full_q_rad, dtype=float)[list(kinematics.active_arm.joint_indices)]
    result = solve_tcp_task_axis_ik(
        kinematics,
        locked_state,
        cut_wp.target_pose,
        jaw_axis_robot=jaw_axis_robot,
        backoff_axis_robot=backoff_axis_robot,
        gripper_jaw_axis_local=primary_local,
        gripper_backoff_axis_local=secondary_local,
        seed_q_arm_rad=home_q,
        reference_q_arm_rad=home_q,
        config=config,
    )
    q_full = kinematics.q_full_from_arm(locked_state, result.q_arm_rad)
    clearance = nearest_self_clearances(
        kinematics, q_full, pair_count=int(pa.clearance_pair_count), margin_m=float(pa.min_clearance_m)
    )
    cut_ik_ok = bool(
        result.ok
        and result.position_error_m <= float(pa.position_tolerance_m)
        and result.orientation_error_rad <= float(pa.orientation_tolerance_rad)
        and clearance.ok
    )
    return {
        "arm": arm,
        "rotation_robot_from_tcp": np.asarray(diag["rotation_robot_from_tcp"], dtype=float).reshape(3, 3),
        "planned_cut_axis": np.asarray(diag["stroke_axis_robot"], dtype=float),
        "planned_approach_axis": np.asarray(diag["approach_axis_robot"], dtype=float),
        "stroke_axis_source": str(diag["stroke_axis_source"]),
        "task_orientation_source": str(diag["task_orientation_source"]),
        "discarded_mps_cut_axis": [round(float(x), 4) for x in diag["raw_mps_cut_axis_robot"]],
        "cut_ik_ok": cut_ik_ok,
        "cut_ik_position_error_m": float(result.position_error_m),
        "cut_ik_orientation_error_rad": float(result.orientation_error_rad),
        "cut_self_clearance_m": clearance.minimum_distance_m,
    }


def deleaf_ablation(
    *, with_stroke_policy: str = "mps", hand_side: str = "right", arms: tuple[str, ...] = ("WITH", "MINUS")
) -> list[dict]:
    tool_axes = load_deleaf_tool_axes(DELEAF_TOOLFIT)
    kinematics = RBY1Kinematics.from_urdf(
        urdf_path=URDF, model_name="a", arm="right", toolfit_path=DELEAF_TOOLFIT
    )
    locked_state = kinematics.locked_state_from_home(DELEAF_HOME)
    home_tcp = kinematics.tcp_pose(locked_state.full_q_rad)[:3, 3]

    arm_overrides = {
        "WITH": dict(
            waypoint_policy="mps_template",
            stroke_axis_policy=with_stroke_policy,
            task_orientation_policy="mps_axes",
            vertical_cut=False,
            force_approach_axis_robot=None,
        ),
        "MINUS": dict(
            waypoint_policy="mps_template",
            vertical_cut=True,
            force_approach_axis_robot="1,0,0",
            task_orientation_policy="mps_axes",
        ),
    }

    rows: list[dict] = []
    for action in DELEAF_ACTIONS:
        guidance = extract_deleaf_mps_guidance(
            manifest_path=DELEAF_MANIFEST,
            action_name=action,
            frame_index_path=None,
            hand_tracking_path=None,
            slam_trajectory_path=SLAM,
            alignment_path=ALIGNMENT,
            hand_side=hand_side,
            hand_point="pinch",
            min_confidence=0.2,
            prefer_downward_cut_axis=True,
        )
        if guidance.cut_anchor_robot_m is None:
            rows.append({"task": "deleaf", "action": action, "arm": "WITH", "error": "no_cut_anchor"})
            continue
        cut_point = np.asarray(guidance.cut_anchor_robot_m, dtype=float).reshape(3)
        human_cut = _u(guidance.cut_axis_robot)
        human_approach = _u(guidance.approach_axis_robot)
        human_tilt_deg = angle_deg(human_cut, [0, 0, -1])
        anchor_reachable = bool(cut_point[2] <= WORKSPACE_CEILING_Z_M)

        # Solve both arms first; the WITH rotation IS the MPS-faithful (human) TCP orientation,
        # so the MINUS orientation deviation is measured against it (no sign-convention noise).
        solved_by_arm = {
            arm: _solve_one_arm(
                arm=arm,
                overrides=ov,
                guidance=guidance,
                cut_point=cut_point,
                home_tcp=home_tcp,
                tool_axes=tool_axes,
                kinematics=kinematics,
                locked_state=locked_state,
            )
            for arm, ov in arm_overrides.items()
        }
        rotation_human = solved_by_arm["WITH"]["rotation_robot_from_tcp"]

        for arm in ("WITH", "MINUS"):
            if arm not in arms:
                continue
            s = solved_by_arm[arm]
            orient_dev = float(
                np.degrees(np.linalg.norm(orientation_residual(s["rotation_robot_from_tcp"], rotation_human)))
            )
            rows.append(
                {
                    "task": "deleaf",
                    "action": action,
                    "arm": arm,
                    "cut_axis_err_deg": float(np.degrees(_axis_error_rad(s["planned_cut_axis"], human_cut))),
                    "heading_axis_err_deg": angle_deg(s["planned_approach_axis"], human_approach, bidirectional=True),
                    "orient_dev_from_human_deg": orient_dev,
                    "human_cut_tilt_off_vertical_deg": human_tilt_deg,
                    "stroke_axis_source": s["stroke_axis_source"],
                    "task_orientation_source": s["task_orientation_source"],
                    "discarded_mps_cut_axis": s["discarded_mps_cut_axis"],
                    "anchor_z_m": float(cut_point[2]),
                    "anchor_reachable": anchor_reachable,
                    "cut_ik_ok": s["cut_ik_ok"],
                    "cut_ik_position_error_m": s["cut_ik_position_error_m"],
                    "cut_ik_orientation_error_rad": s["cut_ik_orientation_error_rad"],
                    "cut_self_clearance_m": s["cut_self_clearance_m"],
                    "confidence": float(guidance.confidence),
                    "cut_anchor_robot_m": cut_point.tolist(),
                    "planned_cut_axis_robot": _u(s["planned_cut_axis"]).tolist(),
                    "planned_heading_axis_robot": _u(s["planned_approach_axis"]).tolist(),
                    "human_cut_axis_robot": human_cut.tolist(),
                    "human_heading_axis_robot": human_approach.tolist(),
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------
def print_harvest_table(rows: list[dict]) -> None:
    print("\n================ HARVEST: action-axis deviation from human MPS demonstration ================")
    print(f"{'action':<9} {'arm':<6} {'approach°':>10} {'jaw°':>8} {'pull°':>8}  source")
    for r in rows:
        print(
            f"{r['action']:<9} {r['arm']:<6} {r['approach_axis_err_deg']:>10.1f} "
            f"{r['jaw_axis_err_deg']:>8.1f} {r['pull_axis_err_deg']:>8.1f}  {r['prior_source']}"
        )
    print("  (WITH reproduces the human axes by construction = applying the prior; MINUS is the prior-free default.)")


def print_deleaf_table(rows: list[dict]) -> None:
    print("\n================ DELEAF: cut deviation from human MPS demonstration ================")
    print(
        f"{'action':<13} {'arm':<6} {'cutAx°':>7} {'heading°':>9} {'orientDev°':>11} "
        f"{'humanTilt°':>11} {'anchorZ':>8} {'reach':>6} {'cutIK':>6} {'ikPosErr':>9}  stroke_source"
    )
    for r in rows:
        if "error" in r:
            print(f"{r['action']:<13} {r['arm']:<6}  ERROR: {r['error']}")
            continue
        pe = r["cut_ik_position_error_m"]
        pe_s = f"{pe*1000:>7.1f}mm" if pe is not None else "     n/a"
        print(
            f"{r['action']:<13} {r['arm']:<6} {r['cut_axis_err_deg']:>7.1f} "
            f"{r['heading_axis_err_deg']:>9.1f} {r['orient_dev_from_human_deg']:>11.1f} "
            f"{r['human_cut_tilt_off_vertical_deg']:>11.1f} {r['anchor_z_m']:>8.3f} "
            f"{('yes' if r['anchor_reachable'] else 'NO'):>6} {('yes' if r['cut_ik_ok'] else 'no'):>6} "
            f"{pe_s:>9}  {r['stroke_axis_source']}"
        )
    print("  cutAx° = planned blade-descent vs human cut axis; heading° = blade horizontal heading vs human;")
    print("  orientDev° = full TCP-orientation deviation from the MPS-faithful (WITH) frame.")
    print("  WITH≈0 = the prior reproduces the human cut; MINUS = the prior-free vertical default's deviation.")
    print("  MINUS proven MPS-off via stroke_source=='vertical_cut_world_down' (the discarded MPS axis is logged).")


def main() -> int:
    parser = argparse.ArgumentParser(description="EgoDelta MPS-prior ablation (offline, no robot).")
    parser.add_argument("--task", choices=("harvest", "deleaf", "both"), default="both")
    parser.add_argument(
        "--with-stroke-policy",
        choices=("mps", "downward_constrained"),
        default="mps",
        help="WITH-MPS deleaf stroke axis: 'mps' = full demonstrated tilt; "
        "'downward_constrained' = deployed 8deg safety clamp toward the MPS axis.",
    )
    parser.add_argument("--harvest-hand-side", default="left", choices=("auto", "left", "right"))
    parser.add_argument("--deleaf-hand-side", default="right", choices=("auto", "left", "right"))
    parser.add_argument(
        "--arm",
        choices=("with", "minus", "both"),
        default="both",
        help="Run only the WITH-MPS configuration, only the MINUS (no-prior) configuration, or both.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arms = {"with": ("WITH",), "minus": ("MINUS",), "both": ("WITH", "MINUS")}[args.arm]

    payload: dict = {
        "method": "egodelta_mps_ablation",
        "alignment": str(ALIGNMENT),
        "with_stroke_policy": args.with_stroke_policy,
        "harvest_hand_side": args.harvest_hand_side,
        "deleaf_hand_side": args.deleaf_hand_side,
        "arm": args.arm,
    }
    if args.task in ("harvest", "both"):
        harvest_rows = harvest_ablation(hand_side=args.harvest_hand_side, arms=arms)
        payload["harvest"] = harvest_rows
        print_harvest_table(harvest_rows)
    if args.task in ("deleaf", "both"):
        deleaf_rows = deleaf_ablation(
            with_stroke_policy=args.with_stroke_policy, hand_side=args.deleaf_hand_side, arms=arms
        )
        payload["deleaf"] = deleaf_rows
        print_deleaf_table(deleaf_rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
