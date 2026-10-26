from __future__ import annotations

import os
import sys

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

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from robot.kinematics.clearance import nearest_self_clearances
from robot.kinematics.ik_solver import solve_tcp_pose_ik, solve_tcp_task_axis_ik
from robot.kinematics.planning.harvest_candidates import (
    HarvestPlanCandidate,
    HarvestWaypointTarget,
    append_harvest_tail,
    generate_harvest_plan_candidates,
    grasp_depth_policy_summary,
)
from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.run_outputs import RUNS_ROOT, artifact_path
from robot.kinematics.task_frame import (
    DEFAULT_TARGET_PATTERN,
    HarvestTaskFrame,
    discover_harvest_task_frame_paths,
    load_harvest_task_frame,
)
from robot.kinematics.types import Array, IKSolverConfig, LockedRobotState


DEFAULT_ADDRESS = "192.0.2.10:50051"
REPO_ROOT = Path(_REPO_ROOT)
DEFAULT_TARGET_DIR = REPO_ROOT / "robot" / "calibration" / "targets" / "a270c038"
DEFAULT_HOME = REPO_ROOT / "robot" / "utils" / "homestate.json"
DEFAULT_TOOLFIT = (
    REPO_ROOT
    / "robot"
    / "calibration"
    / "session"
    / "a270c038"
    / "T_robot_from_aria_selected_current_best_target_toolfit.json"
)
GRASP_DEPTH_RAMP_MIN_STAGE_FRACTION = 0.8


def _progress(message: str) -> None:
    print(f"[harvest-plan] {message}", flush=True)


@dataclass(frozen=True)
class BasketPoseCandidate:
    name: str
    pose: np.ndarray
    diagnostics: dict[str, float | str | bool | None]

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("basket candidate pose must be 4x4")
        object.__setattr__(self, "pose", pose)


@dataclass(frozen=True)
class SceneObstacle:
    name: str
    center_m: np.ndarray
    base_radius_m: float
    margin_m: float
    source_path: str
    target_index: int

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float).reshape(3)
        if self.base_radius_m <= 0.0:
            raise ValueError("scene obstacle base radius must be positive")
        if self.margin_m < 0.0:
            raise ValueError("scene obstacle margin must be non-negative")
        object.__setattr__(self, "center_m", center)

    @property
    def inflated_radius_m(self) -> float:
        return float(self.base_radius_m + self.margin_m)

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "center_m": self.center_m.tolist(),
            "base_radius_m": float(self.base_radius_m),
            "margin_m": float(self.margin_m),
            "inflated_radius_m": self.inflated_radius_m,
            "source_path": self.source_path,
            "target_index": int(self.target_index),
        }


@dataclass(frozen=True)
class GraspJointPrior:
    q_arm_rad: np.ndarray
    weight: float
    source_path: str
    sample_count: int

    def __post_init__(self) -> None:
        q_arm = np.asarray(self.q_arm_rad, dtype=float).reshape(-1)
        if q_arm.size != 7:
            raise ValueError("grasp joint prior must contain 7 active arm joints")
        if not np.all(np.isfinite(q_arm)):
            raise ValueError("grasp joint prior must contain finite values")
        if self.weight <= 0.0:
            raise ValueError("grasp joint prior weight must be positive")
        if self.sample_count <= 0:
            raise ValueError("grasp joint prior sample count must be positive")
        object.__setattr__(self, "q_arm_rad", q_arm)

    def to_summary(self) -> dict[str, Any]:
        return {
            "q_arm_rad": self.q_arm_rad.tolist(),
            "weight": float(self.weight),
            "source_path": self.source_path,
            "sample_count": int(self.sample_count),
        }


def plan_staged_harvest(
    *,
    paths: tuple[Path, ...],
    address: str,
    model: str,
    arm: str,
    home_path: Path,
    toolfit_path: Path,
    base_config: IKSolverConfig,
    approach_orientation_weight: float,
    full_orientation_weight: float,
    approach_backoff_m: float,
    pregrasp_backoff_m: float,
    pregrasp_surface_clearances_m: tuple[float, ...],
    include_tail: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_distance_m: float,
    retract_up_m: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    basket_pose_candidates: tuple[BasketPoseCandidate, ...],
    basket_orientation_weight: float,
    basket_orientation_required: bool,
    candidate_set: str,
    elevated_offsets_m: tuple[float, ...],
    lateral_offsets_m: tuple[float, ...],
    tcp_rolls_rad: tuple[float, ...],
    grasp_insertion_factors: tuple[float, ...],
    palm_correction_factors: tuple[float, ...],
    max_grasp_insertion_factor: float,
    surface_target_robot_m: np.ndarray | None,
    identity_validation_json: Path | None,
    contact_offset_tcp_m: np.ndarray | None,
    include_combined_offsets: bool,
    include_soft_orientation_ramp: bool,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    open_alignment_tolerance_m: float | None,
    open_position_tolerance_m: float,
    open_orientation_tolerance_rad: float,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    grasp_palm_plausibility_margin_m: float,
    grasp_palm_alignment_tolerance_m: float | None,
    min_pregrasp_surface_clearance_m: float,
    min_grasp_insertion_fraction: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_depth_ramp_factors: tuple[float, ...],
    scene_obstacle_margin_m: float,
    open_gripper_half_extents_tcp_m: np.ndarray,
    use_scene_obstacle_clearance: bool,
    max_tail_tcp_x_m: float | None,
    allow_pull_reverse_tail: bool,
    tail_prefix_count: int,
    path_samples: int,
    grasp_joint_prior: GraspJointPrior | None,
    runs_root: Path,
    run_label: str,
    dataset_name: str,
) -> tuple[dict[str, Any], ...]:
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    if path_samples < 1:
        raise ValueError("path_samples must be at least 1")
    if basket_pose_candidates and not include_tail:
        raise ValueError("basket pose candidates require include_tail so basket_place can be planned")
    if harvest_down_m < 0.0:
        raise ValueError("harvest_down_m must be non-negative")
    if tail_orientation_weight is not None and tail_orientation_weight < 0.0:
        raise ValueError("tail_orientation_weight must be non-negative when provided")
    if min_pregrasp_surface_clearance_m < 0.0:
        raise ValueError("min_pregrasp_surface_clearance_m must be non-negative")
    if tail_prefix_count < 1:
        raise ValueError("tail_prefix_count must be at least 1")
    if open_orientation_tolerance_rad <= 0.0:
        raise ValueError("open_orientation_tolerance_rad must be positive")
    if open_position_tolerance_m <= 0.0:
        raise ValueError("open_position_tolerance_m must be positive")
    if pregrasp_task_axis_position_tolerance_m is not None and pregrasp_task_axis_position_tolerance_m <= 0.0:
        raise ValueError("pregrasp_task_axis_position_tolerance_m must be positive when provided")
    if pregrasp_task_jaw_axis_tolerance_rad <= 0.0:
        raise ValueError("pregrasp_task_jaw_axis_tolerance_rad must be positive")
    if pregrasp_task_backoff_axis_tolerance_rad <= 0.0:
        raise ValueError("pregrasp_task_backoff_axis_tolerance_rad must be positive")
    if grasp_palm_plausibility_margin_m < 0.0:
        raise ValueError("grasp_palm_plausibility_margin_m must be non-negative")
    if grasp_palm_alignment_tolerance_m is not None and grasp_palm_alignment_tolerance_m <= 0.0:
        raise ValueError("grasp_palm_alignment_tolerance_m must be positive when provided")
    if not 0.0 <= min_grasp_insertion_fraction <= 1.0:
        raise ValueError("min_grasp_insertion_fraction must be in [0, 1]")
    if grasp_task_axis_position_tolerance_m is not None and grasp_task_axis_position_tolerance_m <= 0.0:
        raise ValueError("grasp_task_axis_position_tolerance_m must be positive when provided")
    if grasp_task_jaw_axis_tolerance_rad <= 0.0:
        raise ValueError("grasp_task_jaw_axis_tolerance_rad must be positive")
    if grasp_task_backoff_axis_tolerance_rad <= 0.0:
        raise ValueError("grasp_task_backoff_axis_tolerance_rad must be positive")
    grasp_depth_ramp_factors = _validated_grasp_depth_ramp_factors(grasp_depth_ramp_factors)
    if max_grasp_insertion_factor < 0.0:
        raise ValueError("max_grasp_insertion_factor must be non-negative")
    if scene_obstacle_margin_m < 0.0:
        raise ValueError("scene_obstacle_margin_m must be non-negative")
    open_gripper_half_extents_tcp_m = np.asarray(open_gripper_half_extents_tcp_m, dtype=float).reshape(3)
    if np.any(open_gripper_half_extents_tcp_m < 0.0):
        raise ValueError("open_gripper_half_extents_tcp_m values must be non-negative")

    _progress(f"connecting to robot model address={address} arm={arm}")
    started_s = time.monotonic()
    kinematics = RBY1Kinematics.from_live_robot(
        address=address,
        model_name=model,
        arm=arm,
        toolfit_path=toolfit_path,
    )
    _progress(f"robot model loaded in {time.monotonic() - started_s:.1f}s")
    locked_state = kinematics.locked_state_from_home(home_path)
    frames = tuple(load_harvest_task_frame(path) for path in paths)
    _progress(f"loaded {len(frames)} harvest target frame(s)")
    identity_targets = _load_identity_surface_targets(identity_validation_json)
    payloads: list[dict[str, Any]] = []
    for frame in frames:
        frame_started_s = time.monotonic()
        _progress(f"planning target={frame.target_name}")
        frame_surface_target = _surface_target_for_frame(
            frame,
            explicit_surface_target_robot_m=surface_target_robot_m,
            identity_targets=identity_targets,
        )
        if frame_surface_target is not None:
            _progress(
                " surface_target_robot_m="
                + ",".join(f"{value:.4f}" for value in np.asarray(frame_surface_target, dtype=float).reshape(3))
            )
        scene_obstacles = _scene_obstacles_for_target(
            frame,
            frames,
            margin_m=scene_obstacle_margin_m,
            enabled=use_scene_obstacle_clearance,
        )
        _progress(f" scene_obstacles={len(scene_obstacles)}")
        payload = _plan_one_target(
            kinematics=kinematics,
            locked_state=locked_state,
            frame=frame,
            scene_obstacles=scene_obstacles,
            address=address,
            model=model,
            arm=arm,
            home_path=home_path,
            toolfit_path=toolfit_path,
            dataset_name=dataset_name,
            base_config=base_config,
            approach_orientation_weight=approach_orientation_weight,
            full_orientation_weight=full_orientation_weight,
            approach_backoff_m=approach_backoff_m,
            pregrasp_backoff_m=pregrasp_backoff_m,
            pregrasp_surface_clearances_m=pregrasp_surface_clearances_m,
            include_tail=include_tail,
            harvest_distance_m=harvest_distance_m,
            harvest_down_m=harvest_down_m,
            retract_distance_m=retract_distance_m,
            retract_up_m=retract_up_m,
            tail_orientation_weight=tail_orientation_weight,
            tail_orientation_required=tail_orientation_required,
            basket_pose_candidates=basket_pose_candidates,
            basket_orientation_weight=basket_orientation_weight,
            basket_orientation_required=basket_orientation_required,
            candidate_set=candidate_set,
            elevated_offsets_m=elevated_offsets_m,
            lateral_offsets_m=lateral_offsets_m,
            tcp_rolls_rad=tcp_rolls_rad,
            grasp_insertion_factors=grasp_insertion_factors,
            palm_correction_factors=palm_correction_factors,
            max_grasp_insertion_factor=max_grasp_insertion_factor,
            surface_target_robot_m=frame_surface_target,
            contact_offset_tcp_m=contact_offset_tcp_m,
            include_combined_offsets=include_combined_offsets,
            include_soft_orientation_ramp=include_soft_orientation_ramp,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            min_clearance_m=min_clearance_m,
            open_alignment_tolerance_m=open_alignment_tolerance_m,
            open_position_tolerance_m=open_position_tolerance_m,
            open_orientation_tolerance_rad=open_orientation_tolerance_rad,
            use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
            pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
            pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
            grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
            grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
            min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
            min_grasp_insertion_fraction=min_grasp_insertion_fraction,
            use_task_axis_grasp_gate=use_task_axis_grasp_gate,
            grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
            grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
            grasp_depth_ramp_factors=grasp_depth_ramp_factors,
            open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
            max_tail_tcp_x_m=max_tail_tcp_x_m,
            allow_pull_reverse_tail=allow_pull_reverse_tail,
            tail_prefix_count=tail_prefix_count,
            path_samples=path_samples,
            grasp_joint_prior=grasp_joint_prior,
        )
        artifact = artifact_path(
            "staged_harvest_ik_plan",
            f"{dataset_name}_{frame.target_name}_{run_label}",
            runs_root=runs_root,
        )
        payload["artifact_path"] = str(artifact)
        if identity_targets is not None:
            payload["identity_validation"] = _identity_summary_for_frame(frame, identity_targets)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _progress(
            f"wrote artifact={artifact} target={frame.target_name} "
            f"ok={payload['hard_gate']['ok']} elapsed={time.monotonic() - frame_started_s:.1f}s"
        )
        payloads.append(payload)
    return tuple(payloads)


def _plan_one_target(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    scene_obstacles: tuple[SceneObstacle, ...],
    address: str,
    model: str,
    arm: str,
    home_path: Path,
    toolfit_path: Path,
    dataset_name: str,
    base_config: IKSolverConfig,
    approach_orientation_weight: float,
    full_orientation_weight: float,
    approach_backoff_m: float,
    pregrasp_backoff_m: float,
    pregrasp_surface_clearances_m: tuple[float, ...],
    include_tail: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_distance_m: float,
    retract_up_m: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    basket_pose_candidates: tuple[BasketPoseCandidate, ...],
    basket_orientation_weight: float,
    basket_orientation_required: bool,
    candidate_set: str,
    elevated_offsets_m: tuple[float, ...],
    lateral_offsets_m: tuple[float, ...],
    tcp_rolls_rad: tuple[float, ...],
    grasp_insertion_factors: tuple[float, ...],
    palm_correction_factors: tuple[float, ...],
    max_grasp_insertion_factor: float,
    surface_target_robot_m: np.ndarray | None,
    contact_offset_tcp_m: np.ndarray | None,
    include_combined_offsets: bool,
    include_soft_orientation_ramp: bool,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    open_alignment_tolerance_m: float | None,
    open_position_tolerance_m: float,
    open_orientation_tolerance_rad: float,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    grasp_palm_plausibility_margin_m: float,
    grasp_palm_alignment_tolerance_m: float | None,
    min_pregrasp_surface_clearance_m: float,
    min_grasp_insertion_fraction: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_depth_ramp_factors: tuple[float, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    max_tail_tcp_x_m: float | None,
    allow_pull_reverse_tail: bool,
    tail_prefix_count: int,
    path_samples: int,
    grasp_joint_prior: GraspJointPrior | None,
) -> dict[str, Any]:
    target_started_s = time.monotonic()
    _progress(f" generating prefix candidates for target={frame.target_name}")
    prefix_candidates = _candidate_specs(
        frame=frame,
        scene_avoidance_offsets_m=_scene_avoidance_offsets(frame, scene_obstacles),
        approach_orientation_weight=approach_orientation_weight,
        full_orientation_weight=full_orientation_weight,
        approach_backoff_m=approach_backoff_m,
        pregrasp_backoff_m=pregrasp_backoff_m,
        pregrasp_surface_clearances_m=pregrasp_surface_clearances_m,
        include_tail=False,
        harvest_distance_m=harvest_distance_m,
        harvest_down_m=harvest_down_m,
        retract_distance_m=retract_distance_m,
        retract_up_m=retract_up_m,
        tail_orientation_weight=tail_orientation_weight,
        tail_orientation_required=tail_orientation_required,
        basket_pose_candidates=(),
        basket_orientation_weight=basket_orientation_weight,
        basket_orientation_required=basket_orientation_required,
        candidate_set=candidate_set,
        elevated_offsets_m=elevated_offsets_m,
        lateral_offsets_m=lateral_offsets_m,
        tcp_rolls_rad=tcp_rolls_rad,
        grasp_insertion_factors=grasp_insertion_factors,
        palm_correction_factors=palm_correction_factors,
        max_grasp_insertion_factor=max_grasp_insertion_factor,
        surface_target_robot_m=surface_target_robot_m,
        contact_offset_tcp_m=contact_offset_tcp_m,
        include_combined_offsets=include_combined_offsets,
        include_soft_orientation_ramp=include_soft_orientation_ramp,
    )
    _progress(f" solving prefix candidates: count={len(prefix_candidates)}")
    prefix_payloads = _solve_candidate_payloads(
        kinematics=kinematics,
        locked_state=locked_state,
        frame=frame,
        candidates=prefix_candidates,
        base_config=base_config,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
        min_clearance_m=min_clearance_m,
        open_alignment_tolerance_m=open_alignment_tolerance_m,
        open_position_tolerance_m=open_position_tolerance_m,
        open_orientation_tolerance_rad=open_orientation_tolerance_rad,
        use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
        pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
        pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
        pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
        grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
        grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
        min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
        min_grasp_insertion_fraction=min_grasp_insertion_fraction,
        use_task_axis_grasp_gate=use_task_axis_grasp_gate,
        grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
        grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
        grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
        grasp_depth_ramp_factors=grasp_depth_ramp_factors,
        open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
        max_tail_tcp_x_m=None,
        scene_obstacles=scene_obstacles,
        path_samples=path_samples,
        grasp_joint_prior=grasp_joint_prior,
    )
    ranked_prefix_candidates = _rank_candidates(prefix_payloads)
    best_prefix_candidate = ranked_prefix_candidates[0] if ranked_prefix_candidates else None
    passing_prefix_candidates = [
        candidate for candidate in ranked_prefix_candidates if candidate["hard_gate"]["ok"]
    ][:tail_prefix_count]
    _progress(
        f" prefix solved: total={len(prefix_payloads)} passing={len(passing_prefix_candidates)} "
        f"best={None if best_prefix_candidate is None else best_prefix_candidate['name']} "
        f"elapsed={time.monotonic() - target_started_s:.1f}s"
    )
    ranked_candidates = ranked_prefix_candidates
    if include_tail:
        tail_payloads_all: list[dict[str, Any]] = []
        prefix_candidate_by_name = {candidate.name: candidate for candidate in prefix_candidates}
        _progress(f" expanding tail candidates from {len(passing_prefix_candidates)} prefix candidate(s)")
        for prefix_index, prefix_payload in enumerate(passing_prefix_candidates, start=1):
            frozen_prefix = prefix_candidate_by_name[prefix_payload["name"]]
            solved_prefix = _candidate_with_solved_prefix_anchor(
                frozen_prefix,
                prefix_payload["waypoints"],
                frame,
            )
            tail_candidates = _tail_candidate_specs(
                prefix_candidate=solved_prefix,
                frame=frame,
                full_orientation_weight=full_orientation_weight,
                harvest_distance_m=harvest_distance_m,
                harvest_down_m=harvest_down_m,
                retract_distance_m=retract_distance_m,
                retract_up_m=retract_up_m,
                tail_orientation_weight=tail_orientation_weight,
                tail_orientation_required=tail_orientation_required,
                basket_pose_candidates=basket_pose_candidates,
                basket_orientation_weight=basket_orientation_weight,
                basket_orientation_required=basket_orientation_required,
                allow_pull_reverse_tail=allow_pull_reverse_tail,
            )
            _progress(
                f" solving tail candidates for prefix {prefix_index}/{len(passing_prefix_candidates)} "
                f"name={prefix_payload['name']} count={len(tail_candidates)}"
            )
            tail_payloads = _solve_candidate_payloads(
                kinematics=kinematics,
                locked_state=locked_state,
                frame=frame,
                candidates=tail_candidates,
                base_config=base_config,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                min_clearance_m=min_clearance_m,
                open_alignment_tolerance_m=open_alignment_tolerance_m,
                open_position_tolerance_m=open_position_tolerance_m,
                open_orientation_tolerance_rad=open_orientation_tolerance_rad,
                use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
                pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
                pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
                pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
                grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
                grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
                min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
                min_grasp_insertion_fraction=min_grasp_insertion_fraction,
                use_task_axis_grasp_gate=use_task_axis_grasp_gate,
                grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
                grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
                grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
                grasp_depth_ramp_factors=grasp_depth_ramp_factors,
                open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
                max_tail_tcp_x_m=max_tail_tcp_x_m,
                scene_obstacles=scene_obstacles,
                path_samples=path_samples,
                frozen_prefix_waypoints=prefix_payload["waypoints"],
                grasp_joint_prior=grasp_joint_prior,
            )
            tail_payloads_all.extend(tail_payloads)
        ranked_candidates = _rank_candidates(tail_payloads_all)
        _progress(
            f" tail solved: total={len(tail_payloads_all)} "
            f"elapsed={time.monotonic() - target_started_s:.1f}s"
        )
    best_candidate = ranked_candidates[0] if ranked_candidates else None
    plan_ok = bool(best_candidate and best_candidate["hard_gate"]["ok"])
    return {
        "schema": "rby1_staged_harvest_candidate_ik_plan",
        "version": 2,
        "dataset_name": dataset_name,
        "target": frame.to_summary(),
        "robot": {
            "address": address,
            "model": model,
            "arm": arm,
            "home_path": str(home_path),
            "toolfit_path": str(toolfit_path),
        },
        "planner": {
            "candidate_set": candidate_set,
            "approach_backoff_m": approach_backoff_m,
            "pregrasp_backoff_lower_bound_m": pregrasp_backoff_m,
            "pregrasp_surface_clearances_m": list(pregrasp_surface_clearances_m),
            "include_tail": include_tail,
            "tail_prefix_count": int(tail_prefix_count),
            "harvest_distance_m": harvest_distance_m,
            "harvest_down_m": harvest_down_m,
            "retract_distance_m": retract_distance_m,
            "retract_up_m": retract_up_m,
            "tail_orientation_weight": tail_orientation_weight,
            "tail_orientation_required": tail_orientation_required,
            "basket_pose_candidates": [
                {
                    "name": candidate.name,
                    "pose": candidate.pose.tolist(),
                    "diagnostics": candidate.diagnostics,
                }
                for candidate in basket_pose_candidates
            ],
            "basket_orientation_weight": basket_orientation_weight,
            "basket_orientation_required": basket_orientation_required,
            "approach_orientation_weight": approach_orientation_weight,
            "full_orientation_weight": full_orientation_weight,
            "elevated_offsets_m": list(elevated_offsets_m),
            "lateral_offsets_m": list(lateral_offsets_m),
            "tcp_rolls_rad": list(tcp_rolls_rad),
            "grasp_depth_policy": grasp_depth_policy_summary(
                frame,
                grasp_insertion_factors,
                palm_correction_factors,
                max_grasp_insertion_factor,
                surface_target_robot_m,
            ),
            "surface_target_override_robot_m": None
            if surface_target_robot_m is None
            else surface_target_robot_m.tolist(),
            "contact_offset_tcp_m": None if contact_offset_tcp_m is None else contact_offset_tcp_m.tolist(),
            "include_combined_offsets": include_combined_offsets,
            "include_soft_orientation_ramp": include_soft_orientation_ramp,
            "path_samples_per_segment": path_samples,
            "grasp_joint_prior": None if grasp_joint_prior is None else grasp_joint_prior.to_summary(),
            "open_alignment_tolerance_m": _open_alignment_tolerance_m(frame, open_alignment_tolerance_m),
            "open_position_tolerance_m": float(open_position_tolerance_m),
            "open_orientation_tolerance_rad": float(open_orientation_tolerance_rad),
            "task_axis_pregrasp_gate": {
                "enabled": bool(use_task_axis_pregrasp_gate),
                "position_tolerance_m": _pregrasp_task_axis_position_tolerance_m(
                    frame,
                    pregrasp_task_axis_position_tolerance_m,
                ),
                "jaw_axis_tolerance_rad": float(pregrasp_task_jaw_axis_tolerance_rad),
                "backoff_axis_tolerance_rad": float(pregrasp_task_backoff_axis_tolerance_rad),
            },
            "grasp_palm_plausibility_margin_m": float(grasp_palm_plausibility_margin_m),
            "grasp_palm_alignment_tolerance_m": _grasp_palm_alignment_tolerance_m(
                frame,
                grasp_palm_alignment_tolerance_m,
            ),
            "min_pregrasp_surface_clearance_m": float(min_pregrasp_surface_clearance_m),
            "min_grasp_insertion_fraction": float(min_grasp_insertion_fraction),
            "task_axis_grasp_gate": {
                "enabled": bool(use_task_axis_grasp_gate),
                "position_tolerance_m": _grasp_task_axis_position_tolerance_m(
                    frame,
                    grasp_task_axis_position_tolerance_m,
                ),
                "jaw_axis_tolerance_rad": float(grasp_task_jaw_axis_tolerance_rad),
                "backoff_axis_tolerance_rad": float(grasp_task_backoff_axis_tolerance_rad),
            },
            "grasp_depth_ramp_factors": list(grasp_depth_ramp_factors),
            "open_gripper_half_extents_tcp_m": open_gripper_half_extents_tcp_m.tolist(),
            "max_tail_tcp_x_m": None if max_tail_tcp_x_m is None else float(max_tail_tcp_x_m),
            "allow_pull_reverse_tail": bool(allow_pull_reverse_tail),
            "scene_obstacle_clearance": {
                "enabled": bool(scene_obstacles),
                "obstacles": [obstacle.to_summary() for obstacle in scene_obstacles],
            },
        },
        "best_candidate_name": None if best_candidate is None else best_candidate["name"],
        "best_candidate": best_candidate,
        "candidates": ranked_candidates,
        "prefix_phase": {
            "frozen_for_tail": bool(include_tail),
            "tail_prefix_count": int(tail_prefix_count),
            "tail_prefix_candidate_names": [candidate["name"] for candidate in passing_prefix_candidates],
            "best_candidate_name": None
            if best_prefix_candidate is None
            else best_prefix_candidate["name"],
            "best_ok": False
            if best_prefix_candidate is None
            else best_prefix_candidate["hard_gate"]["ok"],
            "best_candidate": best_prefix_candidate,
            "candidates": ranked_prefix_candidates,
        },
        "hard_gate": {
            "ok": plan_ok,
            "best_endpoint_ok": False if best_candidate is None else best_candidate["hard_gate"]["endpoint_ok"],
            "best_path_clearance_ok": False if best_candidate is None else best_candidate["hard_gate"]["path_clearance_ok"],
            "best_joint_delta_ok": False if best_candidate is None else best_candidate["hard_gate"]["joint_delta_ok"],
            "best_open_alignment_ok": False if best_candidate is None else best_candidate["hard_gate"]["open_alignment_ok"],
            "best_open_position_ok": False if best_candidate is None else best_candidate["hard_gate"]["open_position_ok"],
            "best_open_orientation_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["open_orientation_ok"],
            "best_grasp_palm_plausibility_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["grasp_palm_plausibility_ok"],
            "best_grasp_palm_alignment_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["grasp_palm_alignment_ok"],
            "best_grasp_depth_ok": False if best_candidate is None else best_candidate["hard_gate"]["grasp_depth_ok"],
            "best_task_axis_grasp_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["task_axis_grasp_ok"],
            "best_pregrasp_surface_clearance_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["pregrasp_surface_clearance_ok"],
            "best_scene_obstacle_clearance_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["scene_obstacle_clearance_ok"],
            "best_tail_forward_reach_ok": False
            if best_candidate is None
            else best_candidate["hard_gate"]["tail_forward_reach_ok"],
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "min_clearance_m": min_clearance_m,
            "open_alignment_tolerance_m": _open_alignment_tolerance_m(frame, open_alignment_tolerance_m),
            "open_position_tolerance_m": float(open_position_tolerance_m),
            "open_orientation_tolerance_rad": float(open_orientation_tolerance_rad),
            "task_axis_pregrasp_gate": {
                "enabled": bool(use_task_axis_pregrasp_gate),
                "position_tolerance_m": _pregrasp_task_axis_position_tolerance_m(
                    frame,
                    pregrasp_task_axis_position_tolerance_m,
                ),
                "jaw_axis_tolerance_rad": float(pregrasp_task_jaw_axis_tolerance_rad),
                "backoff_axis_tolerance_rad": float(pregrasp_task_backoff_axis_tolerance_rad),
            },
            "grasp_palm_plausibility_margin_m": float(grasp_palm_plausibility_margin_m),
            "grasp_palm_alignment_tolerance_m": _grasp_palm_alignment_tolerance_m(
                frame,
                grasp_palm_alignment_tolerance_m,
            ),
            "min_pregrasp_surface_clearance_m": float(min_pregrasp_surface_clearance_m),
            "min_grasp_insertion_fraction": float(min_grasp_insertion_fraction),
            "task_axis_grasp_gate": {
                "enabled": bool(use_task_axis_grasp_gate),
                "position_tolerance_m": _grasp_task_axis_position_tolerance_m(
                    frame,
                    grasp_task_axis_position_tolerance_m,
                ),
                "jaw_axis_tolerance_rad": float(grasp_task_jaw_axis_tolerance_rad),
                "backoff_axis_tolerance_rad": float(grasp_task_backoff_axis_tolerance_rad),
            },
            "open_gripper_half_extents_tcp_m": open_gripper_half_extents_tcp_m.tolist(),
            "max_tail_tcp_x_m": None if max_tail_tcp_x_m is None else float(max_tail_tcp_x_m),
        },
    }


def _solve_candidate(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    candidate: HarvestPlanCandidate,
    base_config: IKSolverConfig,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    open_alignment_tolerance_m: float | None,
    open_position_tolerance_m: float,
    open_orientation_tolerance_rad: float,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    grasp_palm_plausibility_margin_m: float,
    grasp_palm_alignment_tolerance_m: float | None,
    min_pregrasp_surface_clearance_m: float,
    min_grasp_insertion_fraction: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_depth_ramp_factors: tuple[float, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    max_tail_tcp_x_m: float | None,
    scene_obstacles: tuple[SceneObstacle, ...],
    path_samples: int,
    grasp_joint_prior: GraspJointPrior | None,
    frozen_prefix_waypoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    waypoint_payloads = list(frozen_prefix_waypoints or [])
    early_stop_reason: str | None = None
    previous_q_arm: np.ndarray | None = None
    previous_name = "home"
    start_index = 0
    if waypoint_payloads:
        if len(waypoint_payloads) >= len(candidate.waypoints):
            raise ValueError("frozen prefix must leave at least one waypoint to solve")
        for index, frozen_waypoint in enumerate(waypoint_payloads):
            if frozen_waypoint["name"] != candidate.waypoints[index].name:
                raise ValueError("frozen prefix does not match candidate waypoint order")
        last_frozen = waypoint_payloads[-1]
        if not last_frozen["ik_result"]["ok"]:
            raise ValueError("frozen prefix cannot end in a failed IK result")
        previous_q_arm = np.asarray(last_frozen["ik_result"]["q_arm_rad"], dtype=float).reshape(-1)
        previous_name = last_frozen["name"]
        start_index = len(waypoint_payloads)

    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    for waypoint in candidate.waypoints[start_index:]:
        if waypoint.name == "grasp" and grasp_joint_prior is not None:
            seed_attempts: list[tuple[np.ndarray | None, str]] = [
                (grasp_joint_prior.q_arm_rad, "grasp_joint_prior"),
                (previous_q_arm, previous_name),
            ]
        else:
            seed_attempts = [(previous_q_arm, previous_name)]
        if waypoint.orientation_required and previous_q_arm is not None:
            seed_attempts.append((np.asarray(home_q_arm, dtype=float), "home_retry"))
        if waypoint.name == "grasp" and len(grasp_depth_ramp_factors) > 1:
            waypoint_payload = _solve_grasp_depth_ramp(
                kinematics=kinematics,
                locked_state=locked_state,
                frame=frame,
                waypoint=waypoint,
                base_config=base_config,
                seed_attempts=tuple(seed_attempts),
                reference_q_arm=previous_q_arm,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                min_clearance_m=min_clearance_m,
                scene_obstacles=scene_obstacles,
                open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
                use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
                pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
                pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
                pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
                use_task_axis_grasp_gate=use_task_axis_grasp_gate,
                grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
                grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
                grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
                grasp_joint_prior=grasp_joint_prior,
                ramp_factors=grasp_depth_ramp_factors,
            )
        else:
            waypoint_payload = _solve_waypoint_with_retries(
                kinematics=kinematics,
                locked_state=locked_state,
                frame=frame,
                waypoint=waypoint,
                base_config=base_config,
                seed_attempts=tuple(seed_attempts),
                reference_q_arm=previous_q_arm,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                min_clearance_m=min_clearance_m,
                scene_obstacles=scene_obstacles,
                open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
                use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
                pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
                pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
                pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
                use_task_axis_grasp_gate=use_task_axis_grasp_gate,
                grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
                grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
                grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
                grasp_joint_prior=grasp_joint_prior,
            )
        waypoint_payloads.append(waypoint_payload)
        if waypoint_payload["ik_result"]["ok"]:
            previous_q_arm = np.asarray(waypoint_payload["ik_result"]["q_arm_rad"], dtype=float)
            previous_name = waypoint.name
        early_stop_reason = _candidate_early_stop_reason(
            frame=frame,
            waypoint_payloads=waypoint_payloads,
            candidate_diagnostics=candidate.diagnostics,
            latest_waypoint=waypoint_payload,
            open_alignment_tolerance_m=open_alignment_tolerance_m,
            open_position_tolerance_m=open_position_tolerance_m,
            open_orientation_tolerance_rad=open_orientation_tolerance_rad,
            use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
            pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
            pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
            grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
            grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
            min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
            min_grasp_insertion_fraction=min_grasp_insertion_fraction,
            use_task_axis_grasp_gate=use_task_axis_grasp_gate,
            grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
            grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
        )
        if early_stop_reason is not None:
            break

    path_clearance = _path_clearance(
        kinematics=kinematics,
        locked_state=locked_state,
        waypoint_payloads=waypoint_payloads,
        min_clearance_m=min_clearance_m,
        pair_count=base_config.clearance_pair_count,
        path_samples=path_samples,
    )
    scene_obstacle_clearance = _scene_obstacle_path_clearance(
        kinematics=kinematics,
        locked_state=locked_state,
        waypoint_payloads=waypoint_payloads,
        scene_obstacles=scene_obstacles,
        open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
        path_samples=path_samples,
    )
    tail_forward_reach = _tail_forward_reach_report(
        kinematics=kinematics,
        locked_state=locked_state,
        waypoint_payloads=waypoint_payloads,
        max_tail_tcp_x_m=max_tail_tcp_x_m,
        path_samples=path_samples,
    )
    joint_motion = _joint_motion_report(kinematics, locked_state, waypoint_payloads)
    manipulation_quality = _manipulation_quality(
        frame,
        waypoint_payloads,
        candidate.diagnostics,
        open_alignment_tolerance_m=open_alignment_tolerance_m,
        open_position_tolerance_m=open_position_tolerance_m,
        open_orientation_tolerance_rad=open_orientation_tolerance_rad,
        use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
        pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
        pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
        pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
        grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
        grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
        min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
        min_grasp_insertion_fraction=min_grasp_insertion_fraction,
        use_task_axis_grasp_gate=use_task_axis_grasp_gate,
        grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
        grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
        grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
    )
    _apply_effective_endpoint_gates(waypoint_payloads, manipulation_quality)
    endpoint_ok = all(item["endpoint_gate"]["effective_ok"] for item in waypoint_payloads)
    candidate_ok = bool(
        endpoint_ok
        and path_clearance["ok"]
        and scene_obstacle_clearance["ok"]
        and not joint_motion["any_abs_delta_gt_pi"]
        and manipulation_quality["open_alignment_ok"]
        and manipulation_quality["open_position_ok"]
        and manipulation_quality["open_orientation_ok"]
        and manipulation_quality["grasp_palm_plausibility_ok"]
        and manipulation_quality["grasp_palm_alignment_ok"]
        and manipulation_quality["grasp_depth_ok"]
        and manipulation_quality["task_axis_grasp_ok"]
        and manipulation_quality["pregrasp_surface_clearance_ok"]
        and tail_forward_reach["ok"]
    )
    return {
        "name": candidate.name,
        "family": candidate.family,
        "diagnostics": candidate.diagnostics,
        "waypoints": waypoint_payloads,
        "early_stop_reason": early_stop_reason,
        "path_clearance": path_clearance,
        "scene_obstacle_clearance": scene_obstacle_clearance,
        "tail_forward_reach": tail_forward_reach,
        "joint_motion": joint_motion,
        "manipulation_quality": manipulation_quality,
        "hard_gate": {
            "ok": candidate_ok,
            "endpoint_ok": endpoint_ok,
            "path_clearance_ok": path_clearance["ok"],
            "scene_obstacle_clearance_ok": scene_obstacle_clearance["ok"],
            "joint_delta_ok": not joint_motion["any_abs_delta_gt_pi"],
            "open_alignment_ok": manipulation_quality["open_alignment_ok"],
            "open_position_ok": manipulation_quality["open_position_ok"],
            "open_orientation_ok": manipulation_quality["open_orientation_ok"],
            "pregrasp_task_axis_gate": manipulation_quality["pregrasp_task_axis_gate"],
            "grasp_palm_plausibility_ok": manipulation_quality["grasp_palm_plausibility_ok"],
            "grasp_palm_alignment_ok": manipulation_quality["grasp_palm_alignment_ok"],
            "grasp_depth_ok": manipulation_quality["grasp_depth_ok"],
            "task_axis_grasp_ok": manipulation_quality["task_axis_grasp_ok"],
            "pregrasp_surface_clearance_ok": manipulation_quality["pregrasp_surface_clearance_ok"],
            "tail_forward_reach_ok": tail_forward_reach["ok"],
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "min_clearance_m": min_clearance_m,
            "open_alignment_tolerance_m": manipulation_quality["open_alignment_tolerance_m"],
            "open_position_tolerance_m": manipulation_quality["open_position_tolerance_m"],
            "open_orientation_tolerance_rad": manipulation_quality["open_orientation_tolerance_rad"],
            "grasp_palm_plausibility_margin_m": manipulation_quality["grasp_palm_plausibility_margin_m"],
            "grasp_palm_alignment_tolerance_m": manipulation_quality["grasp_palm_alignment_tolerance_m"],
            "min_pregrasp_surface_clearance_m": manipulation_quality["min_pregrasp_surface_clearance_m"],
            "min_grasp_insertion_fraction": manipulation_quality["min_grasp_insertion_fraction"],
            "task_axis_grasp_gate": manipulation_quality["task_axis_grasp_gate"],
            "max_tail_tcp_x_m": tail_forward_reach["max_tail_tcp_x_m"],
        },
    }


def _solve_grasp_depth_ramp(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    waypoint: HarvestWaypointTarget,
    base_config: IKSolverConfig,
    seed_attempts: tuple[tuple[np.ndarray | None, str], ...],
    reference_q_arm: np.ndarray | None,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    scene_obstacles: tuple[SceneObstacle, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_joint_prior: GraspJointPrior | None,
    ramp_factors: tuple[float, ...],
) -> dict[str, Any]:
    stage_summaries: list[dict[str, Any]] = []
    current_seed_attempts = seed_attempts
    current_reference = reference_q_arm
    selected: dict[str, Any] | None = None
    for factor in ramp_factors:
        stage_waypoint = _scaled_grasp_depth_waypoint(waypoint, frame, factor)
        selected = _solve_waypoint_with_retries(
            kinematics=kinematics,
            locked_state=locked_state,
            frame=frame,
            waypoint=stage_waypoint,
            base_config=base_config,
            seed_attempts=current_seed_attempts,
            reference_q_arm=current_reference,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            min_clearance_m=min_clearance_m,
            scene_obstacles=scene_obstacles,
            open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
            use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
            pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
            pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
            use_task_axis_grasp_gate=use_task_axis_grasp_gate,
            grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
            grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
            grasp_joint_prior=grasp_joint_prior,
        )
        depth_report = _grasp_depth_stage_report(selected, frame)
        selected["depth_ramp_fraction"] = float(factor)
        selected["depth_ramp_depth_report"] = depth_report
        stage_summaries.append(_grasp_depth_ramp_stage_summary(selected))
        if not selected["ik_result"]["ok"] or not depth_report["depth_ok"]:
            break
        q_arm = np.asarray(selected["ik_result"]["q_arm_rad"], dtype=float)
        current_seed_attempts = ((q_arm, f"grasp_depth_{int(round(100.0 * float(factor)))}"),)
        current_reference = q_arm
    if selected is None:
        raise RuntimeError("grasp depth ramp requires at least one stage")
    selected["depth_ramp"] = {
        "enabled": True,
        "factors": [float(value) for value in ramp_factors],
        "min_stage_achieved_fraction": GRASP_DEPTH_RAMP_MIN_STAGE_FRACTION,
        "stages": stage_summaries,
    }
    return selected


def _scaled_grasp_depth_waypoint(
    waypoint: HarvestWaypointTarget,
    frame: HarvestTaskFrame,
    depth_fraction: float,
) -> HarvestWaypointTarget:
    fraction = float(depth_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("grasp depth ramp factors must be in (0, 1]")
    if abs(fraction - 1.0) <= 1e-9:
        return waypoint
    backoff_axis = _unit(frame.backoff_axis_robot)
    surface = np.asarray(waypoint.surface_target_robot_m, dtype=float).reshape(3)
    final_contact = np.asarray(waypoint.target_contact_position_robot_m, dtype=float).reshape(3)
    contact_offset = np.asarray(waypoint.contact_offset_tcp_m, dtype=float).reshape(3)
    target_delta = final_contact - surface
    final_insertion_m = max(0.0, -float(np.dot(target_delta, backoff_axis)))
    cross_axis_delta = target_delta + final_insertion_m * backoff_axis
    contact = surface + cross_axis_delta - fraction * final_insertion_m * backoff_axis
    pose = np.asarray(waypoint.target_pose, dtype=float).reshape(4, 4).copy()
    pose[:3, 3] = contact - pose[:3, :3] @ contact_offset
    grasp_anchor = np.asarray(waypoint.grasp_anchor_robot_m, dtype=float).reshape(3)
    return replace(
        waypoint,
        target_pose=pose,
        target_contact_position_robot_m=contact,
        offset_robot_m=contact - surface,
        surface_backoff_component_m=float(np.dot(contact - surface, backoff_axis)),
        grasp_anchor_backoff_component_m=float(np.dot(contact - grasp_anchor, backoff_axis)),
    )


def _grasp_depth_ramp_stage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload["endpoint_gate"]
    depth = payload["depth_ramp_depth_report"]
    return {
        "fraction": float(payload["depth_ramp_fraction"]),
        "seed_source": payload["seed_source"],
        "solver_mode": payload["solver_mode"],
        "ik_ok": bool(payload["ik_result"]["ok"]),
        "depth_ok": bool(depth["depth_ok"]),
        "target_insertion_m": depth["target_insertion_m"],
        "achieved_insertion_m": depth["achieved_insertion_m"],
        "achieved_fraction": depth["achieved_fraction"],
        "min_stage_achieved_fraction": depth["min_stage_achieved_fraction"],
        "position_error_m": float(payload["ik_result"]["position_error_m"]),
        "orientation_error_rad": float(payload["ik_result"]["orientation_error_rad"]),
        "self_clearance_ok": bool(gate["self_clearance_ok"]),
        "scene_obstacle_clearance_ok": bool(gate["scene_obstacle_clearance_ok"]),
        "minimum_clearance_m": gate["minimum_clearance_m"],
        "scene_obstacle_minimum_clearance_m": gate["scene_obstacle_minimum_clearance_m"],
    }


def _grasp_depth_stage_report(payload: dict[str, Any], frame: HarvestTaskFrame) -> dict[str, Any]:
    backoff_axis = _unit(frame.backoff_axis_robot)
    surface = np.asarray(payload["surface_target_robot_m"], dtype=float).reshape(3)
    target_contact = np.asarray(payload["target_contact_position_robot_m"], dtype=float).reshape(3)
    achieved_contact = _waypoint_final_contact_position(payload)
    target_insertion = max(0.0, -float(np.dot(target_contact - surface, backoff_axis)))
    achieved_insertion = max(0.0, -float(np.dot(achieved_contact - surface, backoff_axis)))
    achieved_fraction = 1.0 if target_insertion <= 1e-9 else achieved_insertion / target_insertion
    return {
        "target_insertion_m": float(target_insertion),
        "achieved_insertion_m": float(achieved_insertion),
        "achieved_fraction": float(achieved_fraction),
        "min_stage_achieved_fraction": GRASP_DEPTH_RAMP_MIN_STAGE_FRACTION,
        "depth_ok": bool(achieved_fraction >= GRASP_DEPTH_RAMP_MIN_STAGE_FRACTION),
    }


def _solve_waypoint_with_retries(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    waypoint: Any,
    base_config: IKSolverConfig,
    seed_attempts: tuple[tuple[np.ndarray | None, str], ...],
    reference_q_arm: np.ndarray | None,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    scene_obstacles: tuple[SceneObstacle, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_joint_prior: GraspJointPrior | None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    seen: set[tuple[int, ...] | None] = set()
    last_payload: dict[str, Any] | None = None
    for seed_q, seed_source in seed_attempts:
        if seed_source.endswith("_retry") and last_payload is not None:
            if not _retryable_orientation_failure(last_payload):
                continue
        key = None if seed_q is None else tuple(int(round(float(value) * 1e6)) for value in seed_q)
        if key in seen:
            continue
        seen.add(key)
        attempt_reference_q = seed_q if seed_source.endswith("_retry") and seed_q is not None else reference_q_arm
        payload = _solve_waypoint_once(
            kinematics=kinematics,
            locked_state=locked_state,
            frame=frame,
            waypoint=waypoint,
            base_config=base_config,
            seed_q_arm=seed_q,
            seed_source=seed_source,
            reference_q_arm=attempt_reference_q,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            min_clearance_m=min_clearance_m,
            scene_obstacles=scene_obstacles,
            open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
            use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
            pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
            pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
            use_task_axis_grasp_gate=use_task_axis_grasp_gate,
            grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
            grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
            grasp_joint_prior=grasp_joint_prior,
        )
        attempts.append(payload)
        last_payload = payload
        task_axis_gate = payload.get("task_axis_attempt_gate", {"enabled": False, "ok": True})
        if payload["endpoint_gate"]["ok"] and (not task_axis_gate.get("enabled", False) or task_axis_gate["ok"]):
            break
    if not attempts:
        raise RuntimeError("at least one waypoint seed attempt is required")
    selected = min(attempts, key=_waypoint_attempt_sort_key)
    selected["seed_attempts"] = [
        {
            "seed_source": attempt["seed_source"],
            "endpoint_ok": attempt["endpoint_gate"]["ok"],
            "position_error_m": attempt["ik_result"]["position_error_m"],
            "orientation_error_rad": attempt["ik_result"]["orientation_error_rad"],
            "minimum_clearance_m": attempt["endpoint_gate"]["minimum_clearance_m"],
            "scene_obstacle_minimum_clearance_m": attempt["endpoint_gate"]["scene_obstacle_minimum_clearance_m"],
            "task_axis_ok": attempt.get("task_axis_attempt_gate", {}).get("ok"),
            "task_axis_jaw_error_rad": attempt.get("task_axis_attempt_gate", {}).get("jaw_axis_error_rad"),
            "task_axis_backoff_error_rad": attempt.get("task_axis_attempt_gate", {}).get("backoff_axis_error_rad"),
        }
        for attempt in attempts
    ]
    return selected


def _candidate_early_stop_reason(
    *,
    frame: HarvestTaskFrame,
    waypoint_payloads: list[dict[str, Any]],
    candidate_diagnostics: dict[str, Any],
    latest_waypoint: dict[str, Any],
    open_alignment_tolerance_m: float | None,
    open_position_tolerance_m: float,
    open_orientation_tolerance_rad: float,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    grasp_palm_plausibility_margin_m: float,
    grasp_palm_alignment_tolerance_m: float | None,
    min_pregrasp_surface_clearance_m: float,
    min_grasp_insertion_fraction: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
) -> str | None:
    gate = latest_waypoint["endpoint_gate"]
    if not latest_waypoint["ik_result"]["ok"]:
        return f"{latest_waypoint['name']}:ik_failed"
    if not gate["self_clearance_ok"]:
        return f"{latest_waypoint['name']}:self_clearance_failed"
    if not gate["scene_obstacle_clearance_ok"]:
        return f"{latest_waypoint['name']}:scene_clearance_failed"
    if latest_waypoint["name"] == "pre_grasp" and use_task_axis_pregrasp_gate:
        task_gate = _task_axis_waypoint_gate(
            frame=frame,
            waypoint=latest_waypoint,
            enabled=True,
            position_tolerance_m=_pregrasp_task_axis_position_tolerance_m(
                frame,
                pregrasp_task_axis_position_tolerance_m,
            ),
            jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
        )
        return None if task_gate["ok"] else "pre_grasp:task_axis_failed"
    if latest_waypoint["name"] == "grasp":
        quality = _manipulation_quality(
            frame,
            waypoint_payloads,
            candidate_diagnostics,
            open_alignment_tolerance_m=open_alignment_tolerance_m,
            open_position_tolerance_m=open_position_tolerance_m,
            open_orientation_tolerance_rad=open_orientation_tolerance_rad,
            use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
            pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
            pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
            grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
            grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
            min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
            min_grasp_insertion_fraction=min_grasp_insertion_fraction,
            use_task_axis_grasp_gate=use_task_axis_grasp_gate,
            grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
            grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
        )
        required_ok = (
            quality["open_alignment_ok"]
            and quality["open_position_ok"]
            and quality["open_orientation_ok"]
            and quality["grasp_palm_plausibility_ok"]
            and quality["grasp_palm_alignment_ok"]
            and quality["grasp_depth_ok"]
            and quality["task_axis_grasp_ok"]
            and quality["pregrasp_surface_clearance_ok"]
        )
        return None if required_ok else "grasp:manipulation_quality_failed"
    return None if gate["ok"] else f"{latest_waypoint['name']}:endpoint_failed"


def _solve_waypoint_once(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    waypoint: Any,
    base_config: IKSolverConfig,
    seed_q_arm: np.ndarray | None,
    seed_source: str,
    reference_q_arm: np.ndarray | None,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    scene_obstacles: tuple[SceneObstacle, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_joint_prior: GraspJointPrior | None,
) -> dict[str, Any]:
    waypoint_config = replace(base_config, orientation_weight=waypoint.orientation_weight)
    waypoint_reference_q_arm = reference_q_arm
    active_grasp_prior = waypoint.name == "grasp" and grasp_joint_prior is not None
    if active_grasp_prior:
        waypoint_config = replace(waypoint_config, smoothness_weight=grasp_joint_prior.weight)
        waypoint_reference_q_arm = grasp_joint_prior.q_arm_rad
    use_task_axis_ik = bool(
        (waypoint.name == "pre_grasp" and use_task_axis_pregrasp_gate)
        or (waypoint.name == "grasp" and use_task_axis_grasp_gate)
    )
    if use_task_axis_ik:
        solver_mode = "task_axis"
        result = solve_tcp_task_axis_ik(
            kinematics,
            locked_state,
            waypoint.target_pose,
            jaw_axis_robot=frame.jaw_axis_robot,
            backoff_axis_robot=frame.backoff_axis_robot,
            gripper_jaw_axis_local=frame.gripper_jaw_axis_local,
            gripper_backoff_axis_local=frame.gripper_backoff_axis_local,
            seed_q_arm_rad=seed_q_arm,
            reference_q_arm_rad=waypoint_reference_q_arm,
            config=waypoint_config,
        )
    else:
        solver_mode = "full_pose"
        result = solve_tcp_pose_ik(
            kinematics,
            locked_state,
            waypoint.target_pose,
            seed_q_arm_rad=seed_q_arm,
            reference_q_arm_rad=waypoint_reference_q_arm,
            config=waypoint_config,
        )
    q_full = kinematics.q_full_from_arm(locked_state, result.q_arm_rad)
    final_pose = kinematics.tcp_pose(q_full)
    clearance = nearest_self_clearances(
        kinematics,
        q_full,
        pair_count=waypoint_config.clearance_pair_count,
        margin_m=min_clearance_m,
    )
    scene_tool_half_extents_tcp_m = _waypoint_scene_tool_half_extents(
        waypoint.name,
        open_gripper_half_extents_tcp_m,
    )
    scene_clearance = _scene_obstacle_clearance(
        final_pose,
        scene_obstacles,
        tool_half_extents_tcp_m=scene_tool_half_extents_tcp_m,
    )
    orientation_ok = True if not waypoint.orientation_required else result.orientation_error_rad <= orientation_tolerance_rad
    position_ok = result.position_error_m <= position_tolerance_m
    endpoint_ok = bool(
        result.ok
        and position_ok
        and orientation_ok
        and clearance.ok
        and scene_clearance["ok"]
    )
    payload = {
        "name": waypoint.name,
        "backoff_m": waypoint.backoff_m,
        "backoff_reference": waypoint.backoff_reference,
        "orientation_mode": waypoint.orientation_mode,
        "orientation_required": waypoint.orientation_required,
        "offset_robot_m": waypoint.offset_robot_m.tolist(),
        "target_contact_position_robot_m": waypoint.target_contact_position_robot_m.tolist(),
        "contact_offset_tcp_m": waypoint.contact_offset_tcp_m.tolist(),
        "surface_target_robot_m": waypoint.surface_target_robot_m.tolist(),
        "grasp_anchor_robot_m": waypoint.grasp_anchor_robot_m.tolist(),
        "surface_backoff_component_m": waypoint.surface_backoff_component_m,
        "grasp_anchor_backoff_component_m": waypoint.grasp_anchor_backoff_component_m,
        "seed_source": seed_source,
        "solver_mode": solver_mode,
        "active_grasp_joint_prior": None if not active_grasp_prior else grasp_joint_prior.to_summary(),
        "solver_config": asdict(waypoint_config),
        "target_tcp_pose": waypoint.target_pose.tolist(),
        "final_tcp_pose": final_pose.tolist(),
        "ik_result": {
            "ok": result.ok,
            "q_arm_rad": result.q_arm_rad.tolist(),
            "residual_norm": result.residual_norm,
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "message": result.message,
            "iterations": result.iterations,
        },
        "endpoint_gate": {
            "ok": endpoint_ok,
            "position_tolerance_m": position_tolerance_m,
            "position_ok": position_ok,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "orientation_ok": orientation_ok,
            "min_clearance_m": min_clearance_m,
            "self_clearance_ok": clearance.ok,
            "minimum_clearance_m": clearance.minimum_distance_m,
            "scene_obstacle_clearance_ok": scene_clearance["ok"],
            "scene_obstacle_minimum_clearance_m": scene_clearance["minimum_clearance_m"],
            "scene_tool_half_extents_tcp_m": None
            if scene_tool_half_extents_tcp_m is None
            else scene_tool_half_extents_tcp_m.tolist(),
        },
        "scene_obstacle_clearance": scene_clearance,
        "nearest_clearance_pairs": [asdict(pair) for pair in clearance.nearest_pairs],
    }
    payload["task_axis_attempt_gate"] = _task_axis_attempt_gate_for_waypoint(
        frame=frame,
        waypoint=payload,
        waypoint_name=waypoint.name,
        use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
        pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
        pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
        pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
        use_task_axis_grasp_gate=use_task_axis_grasp_gate,
        grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
        grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
        grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
    )
    return payload


def _retryable_orientation_failure(payload: dict[str, Any]) -> bool:
    gate = payload["endpoint_gate"]
    return bool(
        payload["orientation_required"]
        and gate["position_ok"]
        and gate["self_clearance_ok"]
        and gate["scene_obstacle_clearance_ok"]
        and not gate["orientation_ok"]
    )


def _task_axis_attempt_gate_for_waypoint(
    *,
    frame: HarvestTaskFrame,
    waypoint: dict[str, Any],
    waypoint_name: str,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
) -> dict[str, Any]:
    if waypoint_name == "pre_grasp" and use_task_axis_pregrasp_gate:
        return _task_axis_waypoint_gate(
            frame=frame,
            waypoint=waypoint,
            enabled=True,
            position_tolerance_m=_pregrasp_task_axis_position_tolerance_m(
                frame,
                pregrasp_task_axis_position_tolerance_m,
            ),
            jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
        )
    if waypoint_name == "grasp" and use_task_axis_grasp_gate:
        return _task_axis_waypoint_gate(
            frame=frame,
            waypoint=waypoint,
            enabled=True,
            position_tolerance_m=_grasp_task_axis_position_tolerance_m(
                frame,
                grasp_task_axis_position_tolerance_m,
            ),
            jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
        )
    return {
        "enabled": False,
        "ok": True,
        "position_ok": True,
        "jaw_axis_ok": True,
        "backoff_axis_ok": True,
        "position_error_m": 0.0,
        "jaw_axis_error_rad": 0.0,
        "backoff_axis_error_rad": 0.0,
    }


def _waypoint_attempt_sort_key(payload: dict[str, Any]) -> tuple[float, ...]:
    self_clearance = payload["endpoint_gate"]["minimum_clearance_m"]
    scene_clearance = payload["endpoint_gate"]["scene_obstacle_minimum_clearance_m"]
    if self_clearance is None:
        self_clearance = -1e9
    if scene_clearance is None:
        scene_clearance = 1e9
    task_axis_gate = payload.get("task_axis_attempt_gate", {"ok": True})
    return (
        -float(bool(payload["endpoint_gate"]["ok"])),
        -float(bool(task_axis_gate.get("ok", True))),
        float(task_axis_gate.get("jaw_axis_error_rad", 0.0)),
        float(task_axis_gate.get("backoff_axis_error_rad", 0.0)),
        float(task_axis_gate.get("position_error_m", 0.0)),
        float(payload["ik_result"]["position_error_m"]),
        float(payload["ik_result"]["orientation_error_rad"]),
        -float(scene_clearance),
        -float(self_clearance),
    )


def _solve_candidate_payloads(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    frame: HarvestTaskFrame,
    candidates: tuple[HarvestPlanCandidate, ...],
    base_config: IKSolverConfig,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    min_clearance_m: float,
    open_alignment_tolerance_m: float | None,
    open_position_tolerance_m: float,
    open_orientation_tolerance_rad: float,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    grasp_palm_plausibility_margin_m: float,
    grasp_palm_alignment_tolerance_m: float | None,
    min_pregrasp_surface_clearance_m: float,
    min_grasp_insertion_fraction: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
    grasp_depth_ramp_factors: tuple[float, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    max_tail_tcp_x_m: float | None,
    scene_obstacles: tuple[SceneObstacle, ...],
    path_samples: int,
    grasp_joint_prior: GraspJointPrior | None,
    frozen_prefix_waypoints: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    total = len(candidates)
    if total == 0:
        _progress("  no candidates to solve")
        return []
    payloads: list[dict[str, Any]] = []
    batch_started_s = time.monotonic()
    for index, candidate in enumerate(candidates, start=1):
        candidate_started_s = time.monotonic()
        _progress(
            f"  candidate {index}/{total}: {candidate.name} "
            f"family={candidate.family} waypoints={len(candidate.waypoints)}"
        )
        payload = _solve_candidate(
            kinematics=kinematics,
            locked_state=locked_state,
            frame=frame,
            candidate=candidate,
            base_config=base_config,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            min_clearance_m=min_clearance_m,
            open_alignment_tolerance_m=open_alignment_tolerance_m,
            open_position_tolerance_m=open_position_tolerance_m,
            open_orientation_tolerance_rad=open_orientation_tolerance_rad,
            use_task_axis_pregrasp_gate=use_task_axis_pregrasp_gate,
            pregrasp_task_axis_position_tolerance_m=pregrasp_task_axis_position_tolerance_m,
            pregrasp_task_jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
            pregrasp_task_backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
            grasp_palm_plausibility_margin_m=grasp_palm_plausibility_margin_m,
            grasp_palm_alignment_tolerance_m=grasp_palm_alignment_tolerance_m,
            min_pregrasp_surface_clearance_m=min_pregrasp_surface_clearance_m,
            min_grasp_insertion_fraction=min_grasp_insertion_fraction,
            use_task_axis_grasp_gate=use_task_axis_grasp_gate,
            grasp_task_axis_position_tolerance_m=grasp_task_axis_position_tolerance_m,
            grasp_task_jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
            grasp_task_backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
            grasp_depth_ramp_factors=grasp_depth_ramp_factors,
            open_gripper_half_extents_tcp_m=open_gripper_half_extents_tcp_m,
            max_tail_tcp_x_m=max_tail_tcp_x_m,
            scene_obstacles=scene_obstacles,
            path_samples=path_samples,
            grasp_joint_prior=grasp_joint_prior,
            frozen_prefix_waypoints=frozen_prefix_waypoints,
        )
        payloads.append(payload)
        ok = payload["hard_gate"]["ok"]
        endpoint_ok = payload["hard_gate"]["endpoint_ok"]
        best_waypoint = payload["waypoints"][-1] if payload["waypoints"] else {}
        ik = best_waypoint.get("ik_result", {})
        _progress(
            f"  candidate {index}/{total} done: ok={ok} endpoint_ok={endpoint_ok} "
            f"last={best_waypoint.get('name')} ik_ok={ik.get('ok')} "
            f"pos_err={ik.get('position_error_m')} elapsed={time.monotonic() - candidate_started_s:.1f}s"
        )
    _progress(f" solved {len(payloads)}/{total} candidates in {time.monotonic() - batch_started_s:.1f}s")
    return payloads


def _candidate_specs(
    *,
    frame: HarvestTaskFrame,
    scene_avoidance_offsets_m: tuple[np.ndarray, ...],
    approach_orientation_weight: float,
    full_orientation_weight: float,
    approach_backoff_m: float,
    pregrasp_backoff_m: float,
    pregrasp_surface_clearances_m: tuple[float, ...],
    include_tail: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_distance_m: float,
    retract_up_m: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    basket_pose_candidates: tuple[BasketPoseCandidate, ...],
    basket_orientation_weight: float,
    basket_orientation_required: bool,
    candidate_set: str,
    elevated_offsets_m: tuple[float, ...],
    lateral_offsets_m: tuple[float, ...],
    tcp_rolls_rad: tuple[float, ...],
    grasp_insertion_factors: tuple[float, ...],
    palm_correction_factors: tuple[float, ...],
    max_grasp_insertion_factor: float,
    surface_target_robot_m: np.ndarray | None,
    contact_offset_tcp_m: np.ndarray | None,
    include_combined_offsets: bool,
    include_soft_orientation_ramp: bool,
) -> tuple[HarvestPlanCandidate, ...]:
    if candidate_set not in ("single", "basic"):
        raise ValueError("candidate_set must be 'single' or 'basic'")
    generation_kwargs = {
        "approach_backoff_m": approach_backoff_m,
        "pregrasp_backoff_m": pregrasp_backoff_m,
        "pregrasp_surface_clearances_m": pregrasp_surface_clearances_m,
        "approach_orientation_weight": approach_orientation_weight,
        "full_orientation_weight": full_orientation_weight,
        "include_tail": include_tail,
        "harvest_distance_m": harvest_distance_m,
        "harvest_down_m": harvest_down_m,
        "retract_distance_m": retract_distance_m,
        "retract_up_m": retract_up_m,
        "tail_orientation_weight": tail_orientation_weight,
        "tail_orientation_required": tail_orientation_required,
        "basket_orientation_weight": basket_orientation_weight,
        "basket_orientation_required": basket_orientation_required,
        "elevated_offsets_m": () if candidate_set == "single" else elevated_offsets_m,
        "lateral_offsets_m": () if candidate_set == "single" else lateral_offsets_m,
        "tcp_rolls_rad": tcp_rolls_rad,
        "grasp_insertion_factors": grasp_insertion_factors,
        "palm_correction_factors": palm_correction_factors,
        "max_grasp_insertion_factor": max_grasp_insertion_factor,
        "scene_avoidance_offsets_m": scene_avoidance_offsets_m,
        "surface_target_robot_m": surface_target_robot_m,
        "contact_offset_tcp_m": contact_offset_tcp_m,
        "include_combined_offsets": False if candidate_set == "single" else include_combined_offsets,
        "include_soft_orientation_ramp": False if candidate_set == "single" else include_soft_orientation_ramp,
    }
    if not basket_pose_candidates:
        return generate_harvest_plan_candidates(frame, basket_pose=None, **generation_kwargs)

    candidates: list[HarvestPlanCandidate] = []
    for basket_candidate in basket_pose_candidates:
        generated = generate_harvest_plan_candidates(frame, basket_pose=basket_candidate.pose, **generation_kwargs)
        candidates.extend(_with_basket_candidate(candidate, basket_candidate) for candidate in generated)
    return tuple(candidates)


def _candidate_with_solved_prefix_anchor(
    candidate: HarvestPlanCandidate,
    waypoint_payloads: list[dict[str, Any]],
    frame: HarvestTaskFrame,
) -> HarvestPlanCandidate:
    """Anchor generated tail waypoints to the prefix pose the robot will execute."""

    solved_by_name = {
        payload["name"]: payload
        for payload in waypoint_payloads
        if payload["name"] in {"approach", "grasp"}
    }
    if "approach" not in solved_by_name or "grasp" not in solved_by_name:
        raise ValueError("solved prefix must contain approach and grasp waypoints before tail planning")
    backoff_axis = _unit(frame.backoff_axis_robot)
    anchored_waypoints = tuple(
        _waypoint_with_solved_pose(waypoint, solved_by_name[waypoint.name], backoff_axis)
        if waypoint.name in solved_by_name
        else waypoint
        for waypoint in candidate.waypoints
    )
    diagnostics = dict(candidate.diagnostics)
    diagnostics["tail_anchor_source"] = "solved_prefix"
    return HarvestPlanCandidate(
        name=candidate.name,
        family=candidate.family,
        waypoints=anchored_waypoints,
        diagnostics=diagnostics,
    )


def _waypoint_with_solved_pose(
    waypoint: HarvestWaypointTarget,
    solved_payload: dict[str, Any],
    backoff_axis: np.ndarray,
) -> HarvestWaypointTarget:
    pose = np.asarray(solved_payload["final_tcp_pose"], dtype=float).reshape(4, 4)
    contact_offset = np.asarray(waypoint.contact_offset_tcp_m, dtype=float).reshape(3)
    contact_position = pose[:3, 3] + pose[:3, :3] @ contact_offset
    surface_target = np.asarray(waypoint.surface_target_robot_m, dtype=float).reshape(3)
    grasp_anchor = np.asarray(waypoint.grasp_anchor_robot_m, dtype=float).reshape(3)
    return HarvestWaypointTarget(
        name=waypoint.name,
        backoff_m=waypoint.backoff_m,
        backoff_reference=waypoint.backoff_reference,
        orientation_mode=waypoint.orientation_mode,
        orientation_weight=waypoint.orientation_weight,
        orientation_required=waypoint.orientation_required,
        target_pose=pose,
        target_contact_position_robot_m=contact_position,
        contact_offset_tcp_m=contact_offset,
        offset_robot_m=contact_position - surface_target,
        surface_target_robot_m=surface_target,
        grasp_anchor_robot_m=grasp_anchor,
        surface_backoff_component_m=float(np.dot(contact_position - surface_target, backoff_axis)),
        grasp_anchor_backoff_component_m=float(np.dot(contact_position - grasp_anchor, backoff_axis)),
    )


def _tail_candidate_specs(
    *,
    prefix_candidate: HarvestPlanCandidate,
    frame: HarvestTaskFrame,
    full_orientation_weight: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_distance_m: float,
    retract_up_m: float,
    basket_pose_candidates: tuple[BasketPoseCandidate, ...],
    basket_orientation_weight: float,
    basket_orientation_required: bool,
    allow_pull_reverse_tail: bool,
) -> tuple[HarvestPlanCandidate, ...]:
    candidates: list[HarvestPlanCandidate] = []
    basket_options: tuple[BasketPoseCandidate | None, ...]
    basket_options = basket_pose_candidates if basket_pose_candidates else (None,)
    for retract_mode, distance_m, up_m in _tail_retract_specs(
        retract_distance_m,
        retract_up_m,
        allow_pull_reverse_tail=allow_pull_reverse_tail,
    ):
        for basket_candidate in basket_options:
            tailed = append_harvest_tail(
                prefix_candidate,
                frame,
                full_orientation_weight=full_orientation_weight,
                tail_orientation_weight=tail_orientation_weight,
                tail_orientation_required=tail_orientation_required,
                harvest_distance_m=harvest_distance_m,
                harvest_down_m=harvest_down_m,
                retract_mode=retract_mode,
                retract_distance_m=distance_m,
                retract_up_m=up_m,
                basket_pose=None if basket_candidate is None else basket_candidate.pose,
                basket_orientation_weight=basket_orientation_weight,
                basket_orientation_required=basket_orientation_required,
            )
            candidates.append(tailed if basket_candidate is None else _with_basket_candidate(tailed, basket_candidate))
    return tuple(candidates)


def _tail_retract_specs(
    retract_distance_m: float,
    retract_up_m: float,
    *,
    allow_pull_reverse_tail: bool,
) -> tuple[tuple[str, float, float], ...]:
    if retract_distance_m < 0.0:
        raise ValueError("retract_distance_m must be non-negative")
    if retract_up_m < 0.0:
        raise ValueError("retract_up_m must be non-negative")
    distance = float(retract_distance_m)
    up = float(retract_up_m)
    specs = [
        ("prefix_approach", distance, up),
        ("backoff", distance, up),
        ("backoff", 1.5 * distance, up),
        ("backoff", distance, 2.0 * up),
        ("lift_then_backoff", distance, up),
        ("lift_then_backoff", 1.5 * distance, 2.0 * up),
    ]
    if allow_pull_reverse_tail:
        specs.append(("pull_reverse", distance, up))
    return tuple(specs)


def _with_basket_candidate(
    candidate: HarvestPlanCandidate,
    basket_candidate: BasketPoseCandidate,
) -> HarvestPlanCandidate:
    diagnostics = dict(candidate.diagnostics)
    diagnostics.update(basket_candidate.diagnostics)
    return replace(
        candidate,
        name=f"{candidate.name}_basket_{basket_candidate.name}",
        diagnostics=diagnostics,
    )


def _scene_obstacles_for_target(
    target: HarvestTaskFrame,
    frames: tuple[HarvestTaskFrame, ...],
    *,
    margin_m: float,
    enabled: bool,
) -> tuple[SceneObstacle, ...]:
    if not enabled:
        return ()
    obstacles: list[SceneObstacle] = []
    for frame in frames:
        if frame.target_name == target.target_name and frame.target_index == target.target_index:
            continue
        obstacles.append(_fruit_scene_obstacle(frame, margin_m=margin_m))
    return tuple(obstacles)


def _load_identity_surface_targets(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_source: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for assignment in payload.get("assignments", []):
        if not bool(assignment.get("ok", False)):
            continue
        live_target = (assignment.get("best_match") or {}).get("live_depth_target")
        if live_target is None:
            continue
        record = {
            "identity_validation_json": str(path),
            "target_name": assignment["target_name"],
            "target_index": int(assignment["target_index"]),
            "source_path": assignment["source_path"],
            "surface_target_robot_m": np.asarray(live_target["point_robot_m"], dtype=float).reshape(3),
            "live_depth_target": live_target,
            "assigned_detection_id": assignment.get("assigned_detection_id"),
            "assignment_margin_m": assignment.get("assignment_margin_m"),
            "live_target_separation_m": assignment.get("live_target_separation_m"),
        }
        by_source[str(Path(assignment["source_path"]).resolve()).lower()] = record
        by_name[str(assignment["target_name"])] = record
    if not by_source:
        raise RuntimeError(f"identity validation has no passed live-depth targets: {path}")
    return {
        "path": str(path),
        "ok": bool(payload.get("ok", False)),
        "by_source": by_source,
        "by_name": by_name,
    }


def _surface_target_for_frame(
    frame: HarvestTaskFrame,
    *,
    explicit_surface_target_robot_m: np.ndarray | None,
    identity_targets: dict[str, Any] | None,
) -> np.ndarray | None:
    if explicit_surface_target_robot_m is not None:
        return explicit_surface_target_robot_m
    if identity_targets is None:
        return None
    record = _identity_record_for_frame(frame, identity_targets)
    if record is None:
        raise RuntimeError(f"identity validation has no live-depth target for {frame.target_name}")
    return np.asarray(record["surface_target_robot_m"], dtype=float).reshape(3)


def _identity_record_for_frame(frame: HarvestTaskFrame, identity_targets: dict[str, Any]) -> dict[str, Any] | None:
    source_key = str(Path(frame.source_path).resolve()).lower()
    record = identity_targets["by_source"].get(source_key)
    if record is not None:
        return record
    return identity_targets["by_name"].get(frame.target_name)


def _identity_summary_for_frame(frame: HarvestTaskFrame, identity_targets: dict[str, Any]) -> dict[str, Any] | None:
    record = _identity_record_for_frame(frame, identity_targets)
    if record is None:
        return None
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in record.items()
    }


def _fruit_scene_obstacle(frame: HarvestTaskFrame, *, margin_m: float) -> SceneObstacle:
    dimensions: list[float] = []
    if frame.estimated_grasp_width_m is not None and frame.estimated_grasp_width_m > 0.0:
        dimensions.append(float(frame.estimated_grasp_width_m))
    if frame.estimated_grasp_depth_m is not None and frame.estimated_grasp_depth_m > 0.0:
        dimensions.append(float(frame.estimated_grasp_depth_m))
    if frame.contact_points_robot_m is not None:
        contact_points = np.asarray(frame.contact_points_robot_m, dtype=float).reshape(2, 3)
        dimensions.append(float(np.linalg.norm(contact_points[1] - contact_points[0])))
    base_radius_m = 0.5 * max(dimensions) if dimensions else 0.035
    return SceneObstacle(
        name=frame.target_name,
        center_m=np.asarray(frame.fruit_center_robot_m, dtype=float).reshape(3),
        base_radius_m=base_radius_m,
        margin_m=margin_m,
        source_path=str(frame.source_path),
        target_index=frame.target_index,
    )


def _scene_avoidance_offsets(
    frame: HarvestTaskFrame,
    scene_obstacles: tuple[SceneObstacle, ...],
) -> tuple[np.ndarray, ...]:
    if not scene_obstacles:
        return ()
    target = np.asarray(frame.target_position_robot_m, dtype=float).reshape(3)
    backoff_axis = _unit(frame.backoff_axis_robot)
    offsets: list[np.ndarray] = []
    for obstacle in scene_obstacles:
        away = target - obstacle.center_m
        avoid_direction = away - float(np.dot(away, backoff_axis)) * backoff_axis
        norm = float(np.linalg.norm(avoid_direction))
        if norm <= 1e-9:
            continue
        avoid_direction = avoid_direction / norm
        for distance_m in (0.06, 0.09, 0.12):
            offsets.append(avoid_direction * distance_m)
    return _unique_offsets(offsets)


def _unique_offsets(offsets: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    unique: list[np.ndarray] = []
    keys: set[tuple[int, int, int]] = set()
    for offset in offsets:
        key = tuple(int(round(float(value) * 1000.0)) for value in offset)
        if key in keys:
            continue
        keys.add(key)
        unique.append(np.asarray(offset, dtype=float).reshape(3))
    return tuple(unique)


def _path_clearance(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    waypoint_payloads: list[dict[str, Any]],
    min_clearance_m: float,
    pair_count: int,
    path_samples: int,
) -> dict[str, Any]:
    if not all(item["ik_result"]["ok"] for item in waypoint_payloads):
        return {
            "ok": False,
            "skipped": True,
            "reason": "one or more waypoint IK solves failed",
            "minimum_clearance_m": None,
            "segments": [],
        }

    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    q_by_name = {"home": np.asarray(home_q_arm, dtype=float)}
    for item in waypoint_payloads:
        q_by_name[item["name"]] = np.asarray(item["ik_result"]["q_arm_rad"], dtype=float)

    segments: list[dict[str, Any]] = []
    names = ["home"] + [item["name"] for item in waypoint_payloads]
    for start_name, end_name in zip(names[:-1], names[1:]):
        segment = _segment_clearance(
            kinematics=kinematics,
            locked_state=locked_state,
            start_name=start_name,
            end_name=end_name,
            start_q_arm=q_by_name[start_name],
            end_q_arm=q_by_name[end_name],
            min_clearance_m=min_clearance_m,
            pair_count=pair_count,
            path_samples=path_samples,
        )
        segments.append(segment)

    minimums = [
        segment["minimum_clearance_m"]
        for segment in segments
        if segment["minimum_clearance_m"] is not None
    ]
    minimum = min(minimums) if minimums else None
    return {
        "ok": all(segment["ok"] for segment in segments),
        "skipped": False,
        "path_samples_per_segment": path_samples,
        "minimum_clearance_m": minimum,
        "segments": segments,
    }


def _scene_obstacle_path_clearance(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    waypoint_payloads: list[dict[str, Any]],
    scene_obstacles: tuple[SceneObstacle, ...],
    open_gripper_half_extents_tcp_m: np.ndarray,
    path_samples: int,
) -> dict[str, Any]:
    if not scene_obstacles:
        return {
            "ok": True,
            "enabled": False,
            "skipped": False,
            "minimum_clearance_m": None,
            "segments": [],
            "obstacles": [],
        }
    if not all(item["ik_result"]["ok"] for item in waypoint_payloads):
        return {
            "ok": False,
            "enabled": True,
            "skipped": True,
            "reason": "one or more waypoint IK solves failed",
            "minimum_clearance_m": None,
            "segments": [],
            "obstacles": [obstacle.to_summary() for obstacle in scene_obstacles],
        }

    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    q_by_name = {"home": np.asarray(home_q_arm, dtype=float)}
    for item in waypoint_payloads:
        q_by_name[item["name"]] = np.asarray(item["ik_result"]["q_arm_rad"], dtype=float)

    segments: list[dict[str, Any]] = []
    names = ["home"] + [item["name"] for item in waypoint_payloads]
    for start_name, end_name in zip(names[:-1], names[1:]):
        segment = _segment_scene_obstacle_clearance(
            kinematics=kinematics,
            locked_state=locked_state,
            start_name=start_name,
            end_name=end_name,
            start_q_arm=q_by_name[start_name],
            end_q_arm=q_by_name[end_name],
            scene_obstacles=scene_obstacles,
            tool_half_extents_tcp_m=_segment_scene_tool_half_extents(
                start_name,
                end_name,
                open_gripper_half_extents_tcp_m,
            ),
            path_samples=path_samples,
        )
        segments.append(segment)

    minimums = [
        segment["minimum_clearance_m"]
        for segment in segments
        if segment["minimum_clearance_m"] is not None
    ]
    minimum = min(minimums) if minimums else None
    return {
        "ok": all(segment["ok"] for segment in segments),
        "enabled": True,
        "skipped": False,
        "path_samples_per_segment": path_samples,
        "minimum_clearance_m": minimum,
        "segments": segments,
        "obstacles": [obstacle.to_summary() for obstacle in scene_obstacles],
    }


def _segment_scene_obstacle_clearance(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    start_name: str,
    end_name: str,
    start_q_arm: Array,
    end_q_arm: Array,
    scene_obstacles: tuple[SceneObstacle, ...],
    tool_half_extents_tcp_m: np.ndarray | None,
    path_samples: int,
) -> dict[str, Any]:
    worst: dict[str, Any] | None = None
    worst_index = 0
    worst_alpha = 0.0
    for index in range(path_samples + 1):
        alpha = float(index) / float(path_samples)
        q_arm = (1.0 - alpha) * start_q_arm + alpha * end_q_arm
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        tcp_pose = kinematics.tcp_pose(q_full)
        clearance = _scene_obstacle_clearance(
            tcp_pose,
            scene_obstacles,
            tool_half_extents_tcp_m=tool_half_extents_tcp_m,
        )
        if worst is None or _clearance_value(clearance) < _clearance_value(worst):
            worst = clearance
            worst_index = index
            worst_alpha = alpha
    minimum = None if worst is None else worst["minimum_clearance_m"]
    return {
        "from": start_name,
        "to": end_name,
        "ok": True if worst is None else bool(worst["ok"]),
        "minimum_clearance_m": minimum,
        "worst_sample_index": worst_index,
        "worst_alpha": worst_alpha,
        "scene_tool_half_extents_tcp_m": None
        if tool_half_extents_tcp_m is None
        else tool_half_extents_tcp_m.tolist(),
        "nearest_obstacle_at_worst_sample": None if worst is None else worst["nearest_obstacle"],
    }


def _scene_obstacle_clearance(
    tcp_pose_robot: Array,
    scene_obstacles: tuple[SceneObstacle, ...],
    *,
    tool_half_extents_tcp_m: np.ndarray | None = None,
) -> dict[str, Any]:
    tool_half_extents = _tool_half_extents(tool_half_extents_tcp_m)
    if not scene_obstacles:
        return {
            "ok": True,
            "minimum_clearance_m": None,
            "nearest_obstacle": None,
            "tool_half_extents_tcp_m": None if tool_half_extents is None else tool_half_extents.tolist(),
        }
    tcp_pose = np.asarray(tcp_pose_robot, dtype=float).reshape(4, 4)
    position = tcp_pose[:3, 3]
    rotation = tcp_pose[:3, :3]
    nearest: dict[str, Any] | None = None
    min_clearance: float | None = None
    for obstacle in scene_obstacles:
        delta_robot = obstacle.center_m - position
        distance_to_center_m = float(np.linalg.norm(delta_robot))
        distance_to_tool_m, closest_tool_point_tcp_m = _distance_to_tool_envelope(
            rotation.T @ delta_robot,
            tool_half_extents,
        )
        clearance_m = distance_to_tool_m - obstacle.inflated_radius_m
        if min_clearance is None or clearance_m < min_clearance:
            min_clearance = clearance_m
            nearest = {
                "name": obstacle.name,
                "center_m": obstacle.center_m.tolist(),
                "base_radius_m": float(obstacle.base_radius_m),
                "margin_m": float(obstacle.margin_m),
                "inflated_radius_m": obstacle.inflated_radius_m,
                "tool_model": "tcp_point" if tool_half_extents is None else "tcp_oriented_box",
                "tool_half_extents_tcp_m": None if tool_half_extents is None else tool_half_extents.tolist(),
                "closest_tool_point_tcp_m": closest_tool_point_tcp_m.tolist(),
                "distance_to_center_m": distance_to_center_m,
                "distance_to_tool_m": distance_to_tool_m,
                "clearance_m": clearance_m,
                "source_path": obstacle.source_path,
                "target_index": int(obstacle.target_index),
            }
    return {
        "ok": bool(min_clearance is None or min_clearance >= 0.0),
        "minimum_clearance_m": min_clearance,
        "nearest_obstacle": nearest,
        "tool_half_extents_tcp_m": None if tool_half_extents is None else tool_half_extents.tolist(),
    }


def _clearance_value(clearance: dict[str, Any]) -> float:
    value = clearance["minimum_clearance_m"]
    return float("inf") if value is None else float(value)


def _waypoint_scene_tool_half_extents(
    name: str,
    open_gripper_half_extents_tcp_m: Array,
) -> np.ndarray | None:
    if name not in ("pre_grasp", "grasp"):
        return None
    return _tool_half_extents(open_gripper_half_extents_tcp_m)


def _segment_scene_tool_half_extents(
    start_name: str,
    end_name: str,
    open_gripper_half_extents_tcp_m: Array,
) -> np.ndarray | None:
    if start_name != "pre_grasp" or end_name != "grasp":
        return None
    return _tool_half_extents(open_gripper_half_extents_tcp_m)


def _tool_half_extents(value: Array | None) -> np.ndarray | None:
    if value is None:
        return None
    extents = np.asarray(value, dtype=float).reshape(3)
    if np.any(extents < 0.0):
        raise ValueError("tool half extents must be non-negative")
    if float(np.linalg.norm(extents)) <= 1e-12:
        return None
    return extents


def _distance_to_tool_envelope(
    obstacle_center_tcp_m: Array,
    tool_half_extents_tcp_m: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    center = np.asarray(obstacle_center_tcp_m, dtype=float).reshape(3)
    if tool_half_extents_tcp_m is None:
        return float(np.linalg.norm(center)), np.zeros(3, dtype=float)
    closest = np.clip(center, -tool_half_extents_tcp_m, tool_half_extents_tcp_m)
    return float(np.linalg.norm(center - closest)), closest


def _tail_forward_reach_report(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    waypoint_payloads: list[dict[str, Any]],
    max_tail_tcp_x_m: float | None,
    path_samples: int,
) -> dict[str, Any]:
    if max_tail_tcp_x_m is None:
        return {
            "ok": True,
            "enabled": False,
            "max_tail_tcp_x_m": None,
            "observed_max_tcp_x_m": None,
            "segments": [],
        }
    if not all(item["ik_result"]["ok"] for item in waypoint_payloads):
        return {
            "ok": False,
            "enabled": True,
            "skipped": True,
            "reason": "one or more waypoint IK solves failed",
            "max_tail_tcp_x_m": float(max_tail_tcp_x_m),
            "observed_max_tcp_x_m": None,
            "segments": [],
        }

    names = ["home"] + [item["name"] for item in waypoint_payloads]
    if "grasp" not in names:
        return {
            "ok": True,
            "enabled": True,
            "skipped": False,
            "reason": "no_grasp_waypoint",
            "max_tail_tcp_x_m": float(max_tail_tcp_x_m),
            "observed_max_tcp_x_m": None,
            "segments": [],
        }

    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    q_by_name = {"home": np.asarray(home_q_arm, dtype=float)}
    for item in waypoint_payloads:
        q_by_name[item["name"]] = np.asarray(item["ik_result"]["q_arm_rad"], dtype=float)

    grasp_index = names.index("grasp")
    segments: list[dict[str, Any]] = []
    observed_max: float | None = None
    for index, (start_name, end_name) in enumerate(zip(names[:-1], names[1:])):
        if index < grasp_index:
            continue
        segment = _segment_tail_forward_reach(
            kinematics=kinematics,
            locked_state=locked_state,
            start_name=start_name,
            end_name=end_name,
            start_q_arm=q_by_name[start_name],
            end_q_arm=q_by_name[end_name],
            max_tail_tcp_x_m=float(max_tail_tcp_x_m),
            path_samples=path_samples,
        )
        segments.append(segment)
        value = segment["observed_max_tcp_x_m"]
        if value is not None:
            observed_max = float(value) if observed_max is None else max(observed_max, float(value))

    return {
        "ok": all(segment["ok"] for segment in segments),
        "enabled": True,
        "skipped": False,
        "max_tail_tcp_x_m": float(max_tail_tcp_x_m),
        "observed_max_tcp_x_m": observed_max,
        "segments": segments,
    }


def _segment_tail_forward_reach(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    start_name: str,
    end_name: str,
    start_q_arm: Array,
    end_q_arm: Array,
    max_tail_tcp_x_m: float,
    path_samples: int,
) -> dict[str, Any]:
    observed_max: float | None = None
    worst_index = 0
    worst_alpha = 0.0
    for index in range(path_samples + 1):
        alpha = float(index) / float(path_samples)
        q_arm = (1.0 - alpha) * start_q_arm + alpha * end_q_arm
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        tcp_x_m = float(kinematics.tcp_pose(q_full)[0, 3])
        if observed_max is None or tcp_x_m > observed_max:
            observed_max = tcp_x_m
            worst_index = index
            worst_alpha = alpha
    return {
        "from": start_name,
        "to": end_name,
        "ok": observed_max is None or observed_max <= max_tail_tcp_x_m,
        "max_tail_tcp_x_m": float(max_tail_tcp_x_m),
        "observed_max_tcp_x_m": observed_max,
        "worst_sample_index": worst_index,
        "worst_alpha": worst_alpha,
    }


def _segment_clearance(
    *,
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    start_name: str,
    end_name: str,
    start_q_arm: Array,
    end_q_arm: Array,
    min_clearance_m: float,
    pair_count: int,
    path_samples: int,
) -> dict[str, Any]:
    worst_distance: float | None = None
    worst_index = 0
    worst_alpha = 0.0
    worst_pair: dict[str, Any] | None = None
    for index in range(path_samples + 1):
        alpha = float(index) / float(path_samples)
        q_arm = (1.0 - alpha) * start_q_arm + alpha * end_q_arm
        q_full = kinematics.q_full_from_arm(locked_state, q_arm)
        clearance = nearest_self_clearances(
            kinematics,
            q_full,
            pair_count=pair_count,
            margin_m=min_clearance_m,
        )
        distance = clearance.minimum_distance_m
        if distance is None:
            continue
        if worst_distance is None or distance < worst_distance:
            worst_distance = distance
            worst_index = index
            worst_alpha = alpha
            worst_pair = asdict(clearance.nearest_pairs[0]) if clearance.nearest_pairs else None

    return {
        "from": start_name,
        "to": end_name,
        "ok": worst_distance is None or worst_distance >= min_clearance_m,
        "minimum_clearance_m": worst_distance,
        "worst_sample_index": worst_index,
        "worst_alpha": worst_alpha,
        "nearest_pair_at_worst_sample": worst_pair,
    }


def _rank_candidates(candidate_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(candidate_payloads, key=_candidate_sort_key)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["score"] = _candidate_score(candidate)
    return ranked


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    hard_gate = candidate["hard_gate"]
    path_clearance = candidate["path_clearance"]
    scene_obstacle_clearance = candidate["scene_obstacle_clearance"]
    joint_motion = candidate["joint_motion"]
    manipulation_quality = candidate["manipulation_quality"]
    endpoint_clearances = [
        waypoint["endpoint_gate"]["minimum_clearance_m"]
        for waypoint in candidate["waypoints"]
        if waypoint["endpoint_gate"]["minimum_clearance_m"] is not None
    ]
    min_endpoint_clearance = min(endpoint_clearances) if endpoint_clearances else -1e9
    path_min = path_clearance["minimum_clearance_m"]
    if path_min is None:
        path_min = -1e9
    scene_path_min = scene_obstacle_clearance["minimum_clearance_m"]
    if scene_path_min is None:
        scene_path_min = 1e9
    max_position_error = max(waypoint["ik_result"]["position_error_m"] for waypoint in candidate["waypoints"])
    required_orientation_errors = [
        waypoint["ik_result"]["orientation_error_rad"]
        for waypoint in candidate["waypoints"]
        if waypoint["orientation_required"]
    ]
    max_required_orientation_error = max(required_orientation_errors, default=0.0)
    actual_grasp_insertion_m = float(manipulation_quality.get("actual_grasp_insertion_m", 0.0))
    task_axis_grasp_position_error_m = float(manipulation_quality["task_axis_grasp_position_error_m"])
    task_axis_grasp_jaw_error_rad = float(manipulation_quality["task_axis_grasp_jaw_axis_error_rad"])
    task_axis_grasp_backoff_error_rad = float(manipulation_quality["task_axis_grasp_backoff_axis_error_rad"])
    return (
        -float(bool(hard_gate["ok"])),
        -float(bool(hard_gate["endpoint_ok"])),
        -float(bool(hard_gate["joint_delta_ok"])),
        -float(bool(hard_gate["open_alignment_ok"])),
        -float(bool(hard_gate["open_position_ok"])),
        -float(bool(hard_gate["open_orientation_ok"])),
        -float(bool(hard_gate["grasp_palm_plausibility_ok"])),
        -float(bool(hard_gate["grasp_palm_alignment_ok"])),
        -float(bool(hard_gate["grasp_depth_ok"])),
        -float(bool(hard_gate["task_axis_grasp_ok"])),
        -float(bool(hard_gate["pregrasp_surface_clearance_ok"])),
        -float(bool(hard_gate["scene_obstacle_clearance_ok"])),
        -float(bool(hard_gate["tail_forward_reach_ok"])),
        -float(bool(hard_gate["path_clearance_ok"])),
        -actual_grasp_insertion_m,
        task_axis_grasp_backoff_error_rad,
        task_axis_grasp_jaw_error_rad,
        task_axis_grasp_position_error_m,
        -float(scene_path_min),
        abs(float(manipulation_quality["grasp_palm_axis_offset_m"])),
        float(manipulation_quality["open_position_error_m"]),
        float(max_position_error),
        float(manipulation_quality["open_orientation_error_rad"]),
        float(max_required_orientation_error),
        float(manipulation_quality["grasp_centering_error_m"]),
        float(manipulation_quality["open_cross_axis_offset_norm_m"]),
        -float(path_min),
        -float(min_endpoint_clearance),
        abs(float(manipulation_quality.get("palm_axis_correction_m", 1.0e9))),
        float(joint_motion["max_abs_delta_rad"]),
    )


def _candidate_score(candidate: dict[str, Any]) -> float:
    path_min = candidate["path_clearance"]["minimum_clearance_m"]
    scene_path_min = candidate["scene_obstacle_clearance"]["minimum_clearance_m"]
    quality = candidate["manipulation_quality"]
    open_offset = quality["open_cross_axis_offset_norm_m"]
    clearance_term = -1.0 if path_min is None else float(path_min)
    scene_clearance_term = 0.0 if scene_path_min is None else float(scene_path_min)
    alignment_term = max(0.0, 1.0 - float(open_offset) / max(1e-9, quality["open_alignment_tolerance_m"]))
    nominal_insertion = max(1e-9, float(quality["nominal_grasp_insertion_m"]))
    depth_term = max(0.0, 1.0 - float(quality["grasp_centering_error_m"]) / nominal_insertion)
    task_axis_position_tolerance = max(1e-9, float(quality["task_axis_grasp_gate"]["position_tolerance_m"]))
    task_axis_term = max(
        0.0,
        1.0 - float(quality["task_axis_grasp_position_error_m"]) / task_axis_position_tolerance,
    )
    hard_bonus = 1.0 if candidate["hard_gate"]["ok"] else 0.0
    endpoint_bonus = 0.25 if candidate["hard_gate"]["endpoint_ok"] else 0.0
    path_bonus = 0.25 if candidate["hard_gate"]["path_clearance_ok"] else 0.0
    return (
        hard_bonus
        + endpoint_bonus
        + path_bonus
        + 0.1 * alignment_term
        + 0.1 * depth_term
        + 0.1 * task_axis_term
        + clearance_term
        + scene_clearance_term
    )


def _manipulation_quality(
    frame: HarvestTaskFrame,
    waypoint_payloads: list[dict[str, Any]],
    candidate_diagnostics: dict[str, Any],
    *,
    open_alignment_tolerance_m: float | None,
    open_position_tolerance_m: float,
    open_orientation_tolerance_rad: float,
    use_task_axis_pregrasp_gate: bool,
    pregrasp_task_axis_position_tolerance_m: float | None,
    pregrasp_task_jaw_axis_tolerance_rad: float,
    pregrasp_task_backoff_axis_tolerance_rad: float,
    grasp_palm_plausibility_margin_m: float,
    grasp_palm_alignment_tolerance_m: float | None,
    min_pregrasp_surface_clearance_m: float,
    min_grasp_insertion_fraction: float,
    use_task_axis_grasp_gate: bool,
    grasp_task_axis_position_tolerance_m: float | None,
    grasp_task_jaw_axis_tolerance_rad: float,
    grasp_task_backoff_axis_tolerance_rad: float,
) -> dict[str, Any]:
    if not 0.0 <= min_grasp_insertion_fraction <= 1.0:
        raise ValueError("min_grasp_insertion_fraction must be in [0, 1]")
    tolerance = _open_alignment_tolerance_m(frame, open_alignment_tolerance_m)
    palm_alignment_tolerance = _grasp_palm_alignment_tolerance_m(frame, grasp_palm_alignment_tolerance_m)
    pregrasp_task_axis_position_tolerance = _pregrasp_task_axis_position_tolerance_m(
        frame,
        pregrasp_task_axis_position_tolerance_m,
    )
    grasp_task_axis_position_tolerance = _grasp_task_axis_position_tolerance_m(
        frame,
        grasp_task_axis_position_tolerance_m,
    )
    missing_pregrasp_task_axis_gate = _missing_task_axis_gate(
        enabled=use_task_axis_pregrasp_gate,
        position_tolerance_m=pregrasp_task_axis_position_tolerance,
        jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
        backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
        reason="missing_pre_grasp_waypoint",
    )
    missing_grasp_task_axis_gate = _missing_task_axis_gate(
        enabled=use_task_axis_grasp_gate,
        position_tolerance_m=grasp_task_axis_position_tolerance,
        jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
        backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
        reason="missing_grasp_waypoint",
    )
    open_waypoint = next((item for item in waypoint_payloads if item["name"] == "pre_grasp"), None)
    grasp_waypoint = next((item for item in waypoint_payloads if item["name"] == "grasp"), None)
    if open_waypoint is None:
        return {
            "open_waypoint": "pre_grasp",
            "open_alignment_ok": False,
            "open_position_ok": False,
            "open_orientation_ok": False,
            "open_full_position_ok": False,
            "open_full_orientation_ok": False,
            "pregrasp_task_axis_gate": missing_pregrasp_task_axis_gate,
            "grasp_palm_plausibility_ok": False,
            "grasp_palm_alignment_ok": False,
            "grasp_depth_ok": False,
            "task_axis_grasp_ok": False,
            "task_axis_grasp_gate": missing_grasp_task_axis_gate,
            "pregrasp_surface_clearance_ok": False,
            "open_alignment_tolerance_m": tolerance,
            "open_position_tolerance_m": float(open_position_tolerance_m),
            "open_orientation_tolerance_rad": float(open_orientation_tolerance_rad),
            "grasp_palm_plausibility_margin_m": float(grasp_palm_plausibility_margin_m),
            "grasp_palm_alignment_tolerance_m": palm_alignment_tolerance,
            "min_pregrasp_surface_clearance_m": float(min_pregrasp_surface_clearance_m),
            "min_grasp_insertion_fraction": float(min_grasp_insertion_fraction),
            "open_cross_axis_offset_norm_m": 1.0e9,
            "open_position_error_m": 1.0e9,
            "open_orientation_error_rad": 1.0e9,
            "pregrasp_task_axis_position_error_m": 1.0e9,
            "pregrasp_task_axis_jaw_axis_error_rad": 1.0e9,
            "pregrasp_task_axis_backoff_axis_error_rad": 1.0e9,
            "grasp_palm_axis_offset_m": 1.0e9,
            "task_axis_grasp_position_error_m": 1.0e9,
            "task_axis_grasp_jaw_axis_error_rad": 1.0e9,
            "task_axis_grasp_backoff_axis_error_rad": 1.0e9,
            "grasp_centering_error_m": 1.0e9,
            "nominal_grasp_insertion_m": 0.0,
            "reason": "missing_pre_grasp_waypoint",
        }
    if grasp_waypoint is None:
        return {
            "open_waypoint": "pre_grasp",
            "open_alignment_ok": False,
            "open_position_ok": False,
            "open_orientation_ok": False,
            "open_full_position_ok": False,
            "open_full_orientation_ok": False,
            "pregrasp_task_axis_gate": missing_pregrasp_task_axis_gate,
            "grasp_palm_plausibility_ok": False,
            "grasp_palm_alignment_ok": False,
            "grasp_depth_ok": False,
            "task_axis_grasp_ok": False,
            "task_axis_grasp_gate": missing_grasp_task_axis_gate,
            "pregrasp_surface_clearance_ok": False,
            "open_alignment_tolerance_m": tolerance,
            "open_position_tolerance_m": float(open_position_tolerance_m),
            "open_orientation_tolerance_rad": float(open_orientation_tolerance_rad),
            "grasp_palm_plausibility_margin_m": float(grasp_palm_plausibility_margin_m),
            "grasp_palm_alignment_tolerance_m": palm_alignment_tolerance,
            "min_pregrasp_surface_clearance_m": float(min_pregrasp_surface_clearance_m),
            "min_grasp_insertion_fraction": float(min_grasp_insertion_fraction),
            "open_cross_axis_offset_norm_m": 1.0e9,
            "open_position_error_m": 1.0e9,
            "open_orientation_error_rad": 1.0e9,
            "pregrasp_task_axis_position_error_m": 1.0e9,
            "pregrasp_task_axis_jaw_axis_error_rad": 1.0e9,
            "pregrasp_task_axis_backoff_axis_error_rad": 1.0e9,
            "grasp_palm_axis_offset_m": 1.0e9,
            "task_axis_grasp_position_error_m": 1.0e9,
            "task_axis_grasp_jaw_axis_error_rad": 1.0e9,
            "task_axis_grasp_backoff_axis_error_rad": 1.0e9,
            "grasp_centering_error_m": 1.0e9,
            "nominal_grasp_insertion_m": 0.0,
            "reason": "missing_grasp_waypoint",
        }
    open_position = _waypoint_contact_position(open_waypoint)
    grasp_anchor = np.asarray(open_waypoint["grasp_anchor_robot_m"], dtype=float).reshape(3)
    surface_target = np.asarray(grasp_waypoint["surface_target_robot_m"], dtype=float).reshape(3)
    fruit_center = np.asarray(frame.fruit_center_robot_m, dtype=float).reshape(3)
    delta_anchor = open_position - grasp_anchor
    backoff_axis = _unit(frame.backoff_axis_robot)
    jaw_axis = _unit(frame.jaw_axis_robot)
    palm_axis = _unit(frame.palm_axis_robot)
    backoff_component_m = float(np.dot(delta_anchor, backoff_axis))
    expected_backoff_m = float(open_waypoint["backoff_m"])
    expected_open_center = grasp_anchor + expected_backoff_m * backoff_axis
    delta_open_expected = open_position - expected_open_center
    jaw_component_m = float(np.dot(delta_open_expected, jaw_axis))
    palm_component_m = float(np.dot(delta_open_expected, palm_axis))
    cross_axis_offset_m = float(np.linalg.norm([jaw_component_m, palm_component_m]))
    backoff_error_m = abs(backoff_component_m - expected_backoff_m)
    pregrasp_surface_clearance_m = float(open_waypoint["surface_backoff_component_m"])
    pregrasp_surface_clearance_ok = pregrasp_surface_clearance_m >= float(min_pregrasp_surface_clearance_m)
    open_position_error_m = float(open_waypoint["ik_result"]["position_error_m"])
    open_full_position_ok = open_position_error_m <= float(open_position_tolerance_m)
    open_orientation_error_rad = float(open_waypoint["ik_result"]["orientation_error_rad"])
    open_full_orientation_ok = open_orientation_error_rad <= float(open_orientation_tolerance_rad)
    pregrasp_task_axis_gate = _task_axis_waypoint_gate(
        frame=frame,
        waypoint=open_waypoint,
        enabled=use_task_axis_pregrasp_gate,
        position_tolerance_m=pregrasp_task_axis_position_tolerance,
        jaw_axis_tolerance_rad=pregrasp_task_jaw_axis_tolerance_rad,
        backoff_axis_tolerance_rad=pregrasp_task_backoff_axis_tolerance_rad,
    )
    open_position_ok = (
        pregrasp_task_axis_gate["position_ok"]
        if use_task_axis_pregrasp_gate
        else open_full_position_ok
    )
    open_orientation_ok = (
        pregrasp_task_axis_gate["jaw_axis_ok"] and pregrasp_task_axis_gate["backoff_axis_ok"]
        if use_task_axis_pregrasp_gate
        else open_full_orientation_ok
    )
    grasp_palm_axis_offset_m = float(np.dot(grasp_anchor - fruit_center, palm_axis))
    palm_axis_correction_m = float(candidate_diagnostics.get("palm_axis_correction_m", 0.0))
    palm_axis_correction_cap_m = candidate_diagnostics.get("palm_axis_correction_cap_m")
    if palm_axis_correction_cap_m is None:
        palm_axis_correction_cap_m = abs(float(candidate_diagnostics.get("center_palm_projection_m", 0.0)))
    palm_axis_correction_cap_m = float(palm_axis_correction_cap_m)
    palm_plausibility_limit_m = palm_axis_correction_cap_m + float(grasp_palm_plausibility_margin_m)
    grasp_palm_plausibility_ok = abs(palm_axis_correction_m) <= palm_plausibility_limit_m + 1e-9
    grasp_palm_alignment_ok = abs(grasp_palm_axis_offset_m) <= palm_alignment_tolerance + 1e-9
    declared_insertion_m = float(candidate_diagnostics.get("grasp_insertion_m", 0.0))
    target_insertion_m = max(0.0, -float(grasp_waypoint["surface_backoff_component_m"]))
    actual_grasp_contact = _waypoint_final_contact_position(grasp_waypoint)
    actual_insertion_m = max(0.0, -float(np.dot(actual_grasp_contact - surface_target, backoff_axis)))
    nominal_insertion_m = float(candidate_diagnostics.get("nominal_grasp_insertion_m", 0.0))
    min_required_insertion_m = float(min_grasp_insertion_fraction) * nominal_insertion_m
    grasp_depth_ok = actual_insertion_m + 1e-9 >= min_required_insertion_m
    task_axis_gate = _task_axis_waypoint_gate(
        frame=frame,
        waypoint=grasp_waypoint,
        enabled=use_task_axis_grasp_gate,
        position_tolerance_m=grasp_task_axis_position_tolerance,
        jaw_axis_tolerance_rad=grasp_task_jaw_axis_tolerance_rad,
        backoff_axis_tolerance_rad=grasp_task_backoff_axis_tolerance_rad,
    )
    return {
        "open_waypoint": "pre_grasp",
        "open_alignment_ok": bool(cross_axis_offset_m <= tolerance and backoff_error_m <= tolerance),
        "open_position_ok": bool(open_position_ok),
        "open_orientation_ok": bool(open_orientation_ok),
        "open_full_position_ok": bool(open_full_position_ok),
        "open_full_orientation_ok": bool(open_full_orientation_ok),
        "pregrasp_task_axis_gate": pregrasp_task_axis_gate,
        "grasp_palm_plausibility_ok": bool(grasp_palm_plausibility_ok),
        "grasp_palm_alignment_ok": bool(grasp_palm_alignment_ok),
        "grasp_depth_ok": bool(grasp_depth_ok),
        "task_axis_grasp_ok": bool(task_axis_gate["ok"]),
        "task_axis_grasp_gate": task_axis_gate,
        "pregrasp_surface_clearance_ok": bool(pregrasp_surface_clearance_ok),
        "open_alignment_tolerance_m": tolerance,
        "open_position_tolerance_m": float(open_position_tolerance_m),
        "open_orientation_tolerance_rad": float(open_orientation_tolerance_rad),
        "grasp_palm_plausibility_margin_m": float(grasp_palm_plausibility_margin_m),
        "grasp_palm_alignment_tolerance_m": palm_alignment_tolerance,
        "min_pregrasp_surface_clearance_m": float(min_pregrasp_surface_clearance_m),
        "min_grasp_insertion_fraction": float(min_grasp_insertion_fraction),
        "open_backoff_component_m": backoff_component_m,
        "open_expected_backoff_m": expected_backoff_m,
        "open_backoff_error_m": backoff_error_m,
        "open_jaw_offset_m": jaw_component_m,
        "open_palm_offset_m": palm_component_m,
        "open_cross_axis_offset_norm_m": cross_axis_offset_m,
        "open_position_error_m": open_position_error_m,
        "open_orientation_error_rad": open_orientation_error_rad,
        "pregrasp_task_axis_position_error_m": pregrasp_task_axis_gate["position_error_m"],
        "pregrasp_task_axis_jaw_axis_error_rad": pregrasp_task_axis_gate["jaw_axis_error_rad"],
        "pregrasp_task_axis_backoff_axis_error_rad": pregrasp_task_axis_gate["backoff_axis_error_rad"],
        "grasp_palm_axis_offset_m": grasp_palm_axis_offset_m,
        "task_axis_grasp_position_error_m": task_axis_gate["position_error_m"],
        "task_axis_grasp_jaw_axis_error_rad": task_axis_gate["jaw_axis_error_rad"],
        "task_axis_grasp_backoff_axis_error_rad": task_axis_gate["backoff_axis_error_rad"],
        "palm_axis_correction_m": palm_axis_correction_m,
        "palm_axis_correction_cap_m": palm_axis_correction_cap_m,
        "palm_axis_correction_plausibility_limit_m": palm_plausibility_limit_m,
        "pregrasp_surface_clearance_m": pregrasp_surface_clearance_m,
        "grasp_insertion_m": target_insertion_m,
        "actual_grasp_insertion_m": actual_insertion_m,
        "declared_grasp_insertion_m": declared_insertion_m,
        "nominal_grasp_insertion_m": nominal_insertion_m,
        "min_required_grasp_insertion_m": min_required_insertion_m,
        "grasp_centering_error_m": abs(nominal_insertion_m - actual_insertion_m),
    }


def _apply_effective_endpoint_gates(
    waypoint_payloads: list[dict[str, Any]],
    manipulation_quality: dict[str, Any],
) -> None:
    for waypoint in waypoint_payloads:
        gate = waypoint["endpoint_gate"]
        gate["effective_mode"] = "full_pose"
        gate["effective_ok"] = bool(gate["ok"])
    task_axis_gate = manipulation_quality.get("task_axis_grasp_gate", {})
    pregrasp_task_axis_gate = manipulation_quality.get("pregrasp_task_axis_gate", {})
    if pregrasp_task_axis_gate.get("enabled", False):
        pregrasp_waypoint = next((item for item in waypoint_payloads if item["name"] == "pre_grasp"), None)
        if pregrasp_waypoint is not None:
            gate = pregrasp_waypoint["endpoint_gate"]
            gate["effective_mode"] = "task_axis_pregrasp"
            gate["task_axis_pregrasp_ok"] = bool(
                manipulation_quality["open_position_ok"] and manipulation_quality["open_orientation_ok"]
            )
            gate["effective_ok"] = bool(
                pregrasp_waypoint["ik_result"]["ok"]
                and gate["self_clearance_ok"]
                and gate["scene_obstacle_clearance_ok"]
                and manipulation_quality["open_position_ok"]
                and manipulation_quality["open_orientation_ok"]
            )
    if not task_axis_gate.get("enabled", False):
        return
    grasp_waypoint = next((item for item in waypoint_payloads if item["name"] == "grasp"), None)
    if grasp_waypoint is None:
        return
    gate = grasp_waypoint["endpoint_gate"]
    gate["effective_mode"] = "task_axis_grasp"
    gate["task_axis_grasp_ok"] = bool(manipulation_quality["task_axis_grasp_ok"])
    gate["effective_ok"] = bool(
        grasp_waypoint["ik_result"]["ok"]
        and gate["self_clearance_ok"]
        and gate["scene_obstacle_clearance_ok"]
        and manipulation_quality["task_axis_grasp_ok"]
    )


def _missing_task_axis_gate(
    *,
    enabled: bool,
    position_tolerance_m: float,
    jaw_axis_tolerance_rad: float,
    backoff_axis_tolerance_rad: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "ok": False,
        "position_ok": False,
        "jaw_axis_ok": False,
        "backoff_axis_ok": False,
        "position_tolerance_m": float(position_tolerance_m),
        "jaw_axis_tolerance_rad": float(jaw_axis_tolerance_rad),
        "backoff_axis_tolerance_rad": float(backoff_axis_tolerance_rad),
        "position_error_m": 1.0e9,
        "jaw_axis_error_rad": 1.0e9,
        "backoff_axis_error_rad": 1.0e9,
        "reason": reason,
    }


def _task_axis_waypoint_gate(
    *,
    frame: HarvestTaskFrame,
    waypoint: dict[str, Any],
    enabled: bool,
    position_tolerance_m: float,
    jaw_axis_tolerance_rad: float,
    backoff_axis_tolerance_rad: float,
) -> dict[str, Any]:
    final_pose = np.asarray(waypoint["final_tcp_pose"], dtype=float).reshape(4, 4)
    rotation = final_pose[:3, :3]
    actual_contact = _waypoint_final_contact_position(waypoint)
    target_contact = _waypoint_contact_position(waypoint)
    position_error_m = float(np.linalg.norm(actual_contact - target_contact))
    jaw_axis_error_rad = _axis_error_rad(
        _tcp_local_axis(rotation, frame.gripper_jaw_axis_local),
        frame.jaw_axis_robot,
    )
    backoff_axis_error_rad = _axis_error_rad(
        _tcp_local_axis(rotation, frame.gripper_backoff_axis_local),
        frame.backoff_axis_robot,
    )
    position_ok = position_error_m <= float(position_tolerance_m)
    jaw_axis_ok = jaw_axis_error_rad <= float(jaw_axis_tolerance_rad)
    backoff_axis_ok = backoff_axis_error_rad <= float(backoff_axis_tolerance_rad)
    axis_gate_ok = bool(position_ok and jaw_axis_ok and backoff_axis_ok)
    return {
        "enabled": bool(enabled),
        "ok": bool(axis_gate_ok if enabled else True),
        "position_ok": bool(position_ok),
        "jaw_axis_ok": bool(jaw_axis_ok),
        "backoff_axis_ok": bool(backoff_axis_ok),
        "position_tolerance_m": float(position_tolerance_m),
        "jaw_axis_tolerance_rad": float(jaw_axis_tolerance_rad),
        "backoff_axis_tolerance_rad": float(backoff_axis_tolerance_rad),
        "position_error_m": position_error_m,
        "jaw_axis_error_rad": jaw_axis_error_rad,
        "backoff_axis_error_rad": backoff_axis_error_rad,
        "actual_contact_position_robot_m": actual_contact.tolist(),
        "target_contact_position_robot_m": target_contact.tolist(),
        "tcp_local_jaw_axis_robot": _tcp_local_axis(rotation, frame.gripper_jaw_axis_local).tolist(),
        "tcp_local_backoff_axis_robot": _tcp_local_axis(rotation, frame.gripper_backoff_axis_local).tolist(),
        "target_jaw_axis_robot": _unit(frame.jaw_axis_robot).tolist(),
        "target_backoff_axis_robot": _unit(frame.backoff_axis_robot).tolist(),
    }


def _waypoint_contact_position(waypoint: dict[str, Any]) -> np.ndarray:
    if "target_contact_position_robot_m" in waypoint:
        return np.asarray(waypoint["target_contact_position_robot_m"], dtype=float).reshape(3)
    return np.asarray(waypoint["target_tcp_pose"], dtype=float).reshape(4, 4)[:3, 3]


def _waypoint_final_contact_position(waypoint: dict[str, Any]) -> np.ndarray:
    final_pose = np.asarray(waypoint["final_tcp_pose"], dtype=float).reshape(4, 4)
    offset_tcp = np.asarray(waypoint["contact_offset_tcp_m"], dtype=float).reshape(3)
    return final_pose[:3, 3] + final_pose[:3, :3] @ offset_tcp


def _open_alignment_tolerance_m(frame: HarvestTaskFrame, override_m: float | None) -> float:
    if override_m is not None:
        if override_m <= 0.0:
            raise ValueError("open_alignment_tolerance_m must be positive when provided")
        return float(override_m)
    width = frame.estimated_grasp_width_m
    if width is None or width <= 0.0:
        return 0.012
    return float(min(0.02, max(0.008, 0.2 * float(width))))


def _grasp_palm_alignment_tolerance_m(frame: HarvestTaskFrame, override_m: float | None) -> float:
    if override_m is not None:
        if override_m <= 0.0:
            raise ValueError("grasp_palm_alignment_tolerance_m must be positive when provided")
        return float(override_m)
    width = frame.estimated_grasp_width_m
    if width is None or width <= 0.0:
        return 0.006
    return float(min(0.008, max(0.004, 0.1 * float(width))))


def _grasp_task_axis_position_tolerance_m(frame: HarvestTaskFrame, override_m: float | None) -> float:
    if override_m is not None:
        if override_m <= 0.0:
            raise ValueError("grasp_task_axis_position_tolerance_m must be positive when provided")
        return float(override_m)
    width = frame.estimated_grasp_width_m
    if width is None or width <= 0.0:
        return 0.025
    return float(min(0.04, max(0.018, 0.5 * float(width))))


def _pregrasp_task_axis_position_tolerance_m(frame: HarvestTaskFrame, override_m: float | None) -> float:
    if override_m is not None:
        if override_m <= 0.0:
            raise ValueError("pregrasp_task_axis_position_tolerance_m must be positive when provided")
        return float(override_m)
    width = frame.estimated_grasp_width_m
    if width is None or width <= 0.0:
        return 0.006
    return float(min(0.010, max(0.004, 0.12 * float(width))))


def _tcp_local_axis(rotation_robot_from_tcp: Array, local_axis: str) -> np.ndarray:
    axis_name = local_axis.lower().strip()
    mapping = {
        "plus-x": (1.0, 0),
        "minus-x": (-1.0, 0),
        "plus-y": (1.0, 1),
        "minus-y": (-1.0, 1),
        "plus-z": (1.0, 2),
        "minus-z": (-1.0, 2),
    }
    if axis_name not in mapping:
        raise ValueError(f"unsupported TCP local axis {local_axis!r}")
    sign, index = mapping[axis_name]
    rotation = np.asarray(rotation_robot_from_tcp, dtype=float).reshape(3, 3)
    return sign * rotation[:, index]


def _axis_error_rad(actual: Array, expected: Array) -> float:
    dot = float(np.dot(_unit(actual), _unit(expected)))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def _joint_motion_report(
    kinematics: RBY1Kinematics,
    locked_state: LockedRobotState,
    waypoint_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if not all(item["ik_result"]["ok"] for item in waypoint_payloads):
        return {
            "skipped": True,
            "reason": "one or more waypoint IK solves failed",
            "max_abs_delta_rad": 0.0,
            "any_abs_delta_gt_pi": False,
            "segments": [],
        }

    home_q_arm = locked_state.full_q_rad[list(kinematics.active_arm.joint_indices)]
    previous_name = "home"
    previous_q = np.asarray(home_q_arm, dtype=float)
    max_abs_delta = 0.0
    any_abs_delta_gt_pi = False
    segments: list[dict[str, Any]] = []
    for waypoint in waypoint_payloads:
        current_q = np.asarray(waypoint["ik_result"]["q_arm_rad"], dtype=float)
        delta = current_q - previous_q
        abs_delta = np.abs(delta)
        segment_max = float(np.max(abs_delta))
        segment_gt_pi = bool(np.any(abs_delta > np.pi))
        max_abs_delta = max(max_abs_delta, segment_max)
        any_abs_delta_gt_pi = any_abs_delta_gt_pi or segment_gt_pi
        segments.append(
            {
                "from": previous_name,
                "to": waypoint["name"],
                "max_abs_delta_rad": segment_max,
                "any_abs_delta_gt_pi": segment_gt_pi,
                "delta_rad": delta.tolist(),
            }
        )
        previous_name = waypoint["name"]
        previous_q = current_q
    return {
        "skipped": False,
        "max_abs_delta_rad": max_abs_delta,
        "any_abs_delta_gt_pi": any_abs_delta_gt_pi,
        "segments": segments,
    }


def _paths_from_args(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.paths:
        return tuple(args.paths)
    return discover_harvest_task_frame_paths(args.target_dir, args.pattern)


def _parse_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset_name:
        return args.dataset_name
    return Path(args.target_dir).name


def _run_label(args: argparse.Namespace) -> str:
    if args.run_label:
        return args.run_label
    approach_cm = int(round(args.approach_backoff_m * 100.0))
    pregrasp_floor_cm = int(round(args.pregrasp_backoff_m * 100.0))
    return (
        f"{args.candidate_set}_approach{approach_cm:02d}cm_posonly_"
        f"pregrasp_floor{pregrasp_floor_cm:02d}cm_fullpose_depthsweep"
    )


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_degree_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(np.deg2rad(degrees)) for degrees in _parse_float_tuple(value))


def _parse_vector3(value: str | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    values = _parse_float_tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated numbers")
    return np.asarray(values, dtype=float)


def _validated_grasp_depth_ramp_factors(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return (1.0,)
    factors = sorted({float(value) for value in values})
    if any(value <= 0.0 or value > 1.0 for value in factors):
        raise ValueError("grasp depth ramp factors must be in (0, 1]")
    if abs(factors[-1] - 1.0) > 1e-9:
        factors.append(1.0)
    return tuple(factors)


def _grasp_joint_prior_from_args(args: argparse.Namespace) -> GraspJointPrior | None:
    if args.grasp_prior_capture_json is None:
        return None
    if args.grasp_prior_weight <= 0.0:
        raise ValueError("--grasp-prior-weight must be positive when --grasp-prior-capture-json is used")
    capture_path = Path(args.grasp_prior_capture_json)
    payload = json.loads(capture_path.read_text(encoding="utf-8"))
    if payload.get("arm") not in (None, args.arm):
        raise ValueError(f"grasp prior arm {payload.get('arm')!r} does not match requested arm {args.arm!r}")
    samples = payload.get("samples", [])
    q_values = [np.asarray(sample["q_arm_rad"], dtype=float).reshape(7) for sample in samples if "q_arm_rad" in sample]
    if not q_values:
        raise ValueError(f"grasp prior capture has no q_arm_rad samples: {capture_path}")
    return GraspJointPrior(
        q_arm_rad=np.median(np.vstack(q_values), axis=0),
        weight=float(args.grasp_prior_weight),
        source_path=str(capture_path),
        sample_count=len(q_values),
    )


def _validated_vector3(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _contact_offset_from_args(args: argparse.Namespace) -> np.ndarray | None:
    raw_offset = _parse_vector3(args.contact_offset_tcp_m, name="contact offset")
    if args.contact_frame_json is None:
        return raw_offset
    if raw_offset is not None:
        raise ValueError("use either --contact-offset-tcp-m or --contact-frame-json, not both")

    with args.contact_frame_json.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "contact_offset_tcp_m" in payload:
        return _validated_vector3(payload["contact_offset_tcp_m"], name="contact_offset_tcp_m")
    if "T_tcp_from_gripper_contact_center" in payload:
        transform = np.asarray(payload["T_tcp_from_gripper_contact_center"], dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("T_tcp_from_gripper_contact_center must be a 4x4 transform")
        return _validated_vector3(transform[:3, 3], name="T_tcp_from_gripper_contact_center translation")
    raise ValueError("contact frame JSON must contain contact_offset_tcp_m or T_tcp_from_gripper_contact_center")


def _unit(vector: Array) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("axis must be non-zero")
    return value / norm


def _basket_pose_candidates_from_args(args: argparse.Namespace) -> tuple[BasketPoseCandidate, ...]:
    direct_position = _parse_vector3(args.basket_position_robot_m, name="basket position")
    anchor_position = _parse_vector3(args.basket_anchor_robot_m, name="basket anchor")
    if direct_position is not None and anchor_position is not None:
        raise ValueError("use either --basket-position-robot-m or --basket-anchor-robot-m, not both")
    if direct_position is None and anchor_position is None:
        return ()

    rpy_deg = _parse_vector3(args.basket_rpy_deg, name="basket rpy")
    if rpy_deg is None:
        rotation = np.eye(3, dtype=float)
    else:
        rotation = _rpy_rotation(np.deg2rad(rpy_deg))
    if direct_position is not None:
        return (
            _basket_pose_candidate(
                name="direct_tcp",
                position=direct_position,
                rotation=rotation,
                diagnostics={
                    "basket_policy": "direct_tcp",
                    "basket_drop_height_m": 0.0,
                    "basket_inset_m": 0.0,
                },
            ),
        )

    heights = _nonnegative_tuple(args.basket_drop_heights_m, name="basket drop heights")
    insets = _nonnegative_tuple(args.basket_insets_m, name="basket insets")
    if not heights:
        raise ValueError("basket drop heights must contain at least one value")
    if not insets:
        raise ValueError("basket insets must contain at least one value")

    anchor = np.asarray(anchor_position, dtype=float).reshape(3)
    inset_axis = _horizontal_toward_robot(anchor)
    candidates: list[BasketPoseCandidate] = []
    for height_m in heights:
        for inset_m in insets:
            position = anchor + height_m * np.asarray([0.0, 0.0, 1.0], dtype=float) + inset_m * inset_axis
            candidates.append(
                _basket_pose_candidate(
                    name=f"anchor_h{_basket_cm_label(height_m)}_in{_basket_cm_label(inset_m)}",
                    position=position,
                    rotation=rotation,
                    diagnostics={
                        "basket_policy": "anchor_height_inset_sweep",
                        "basket_anchor_x_m": float(anchor[0]),
                        "basket_anchor_y_m": float(anchor[1]),
                        "basket_anchor_z_m": float(anchor[2]),
                        "basket_drop_height_m": float(height_m),
                        "basket_inset_m": float(inset_m),
                    },
                )
            )
    return tuple(candidates)


def _basket_pose_candidate(
    *,
    name: str,
    position: Array,
    rotation: Array,
    diagnostics: dict[str, float | str | bool | None],
) -> BasketPoseCandidate:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    pose[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return BasketPoseCandidate(name=name, pose=pose, diagnostics=diagnostics)


def _horizontal_toward_robot(anchor_robot_m: Array) -> np.ndarray:
    anchor = np.asarray(anchor_robot_m, dtype=float).reshape(3)
    direction = np.asarray([-anchor[0], -anchor[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return np.zeros(3, dtype=float)
    return direction / norm


def _nonnegative_tuple(value: str, *, name: str) -> tuple[float, ...]:
    values = _parse_float_tuple(value)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must be non-negative")
    return values


def _basket_cm_label(value_m: float) -> str:
    return f"{int(round(100.0 * float(value_m))):02d}cm"


def _rpy_rotation(rpy_rad: Array) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=float).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def main() -> int:
    parser = argparse.ArgumentParser(description="No-motion staged harvest IK planner for validated task frames.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--pattern", default=DEFAULT_TARGET_PATTERN)
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--model", default="a")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--toolfit", type=Path, default=DEFAULT_TOOLFIT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--approach-backoff-m", type=float, default=0.10)
    parser.add_argument(
        "--pregrasp-backoff-m",
        type=float,
        default=0.05,
        help="Lower bound for target-derived pre-grasp backoff; actual candidates also use insertion depth plus surface clearance.",
    )
    parser.add_argument("--pregrasp-surface-clearances-m", default="0.02,0.03")
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument(
        "--tail-prefix-count",
        type=int,
        default=3,
        help="Number of passing prefix candidates expanded into full tail candidates.",
    )
    parser.add_argument("--harvest-distance-m", type=float, default=0.05)
    parser.add_argument("--harvest-down-m", type=float, default=0.025)
    parser.add_argument("--retract-distance-m", type=float, default=0.12)
    parser.add_argument("--retract-up-m", type=float, default=0.04)
    parser.add_argument(
        "--relax-tail-orientation",
        action="store_true",
        help="For post-grasp harvest/retract, prefer but do not require the frozen grasp orientation.",
    )
    parser.add_argument(
        "--tail-orientation-weight",
        type=float,
        default=None,
        help="Optional orientation residual weight for relaxed post-grasp harvest/retract waypoints.",
    )
    parser.add_argument("--basket-position-robot-m", default=None, help="Comma-separated x,y,z direct basket TCP target.")
    parser.add_argument("--basket-anchor-robot-m", default=None, help="Comma-separated x,y,z basket anchor point.")
    parser.add_argument("--basket-drop-heights-m", default="0.12,0.16,0.20,0.24")
    parser.add_argument("--basket-insets-m", default="0.00,0.08,0.12,0.16,0.20")
    parser.add_argument("--basket-rpy-deg", default=None, help="Optional comma-separated roll,pitch,yaw for basket TCP pose.")
    parser.add_argument("--basket-orientation-weight", type=float, default=0.0)
    parser.add_argument("--basket-require-orientation", action="store_true")
    parser.add_argument("--candidate-set", choices=("single", "basic"), default="basic")
    parser.add_argument("--elevated-offsets-m", default="0.05,0.08")
    parser.add_argument("--lateral-offsets-m", default="0.04,0.06,0.08,0.10,0.12,-0.04,-0.06,-0.08")
    parser.add_argument("--tcp-rolls-deg", default="0,180")
    parser.add_argument("--grasp-insertion-factors", default="0.5,0.75,1.0,1.25")
    parser.add_argument(
        "--max-grasp-insertion-factor",
        type=float,
        default=1.0,
        help="Maximum allowed grasp insertion as a factor of nominal insertion; defaults to the mask-derived half-depth.",
    )
    parser.add_argument("--palm-correction-factors", default="0.0,0.5,1.0")
    parser.add_argument(
        "--surface-target-robot-m",
        default=None,
        help="Optional live corrected surface target as comma-separated robot-frame x,y,z.",
    )
    parser.add_argument(
        "--identity-validation-json",
        type=Path,
        default=None,
        help="Output JSON from validate_live_target_identity.py; supplies per-target live RealSense depth targets.",
    )
    parser.add_argument(
        "--contact-offset-tcp-m",
        default=None,
        help="Optional TCP-frame vector from TCP origin to modeled contact point, as x,y,z meters.",
    )
    parser.add_argument(
        "--contact-frame-json",
        type=Path,
        default=None,
        help="Optional gripper contact-frame calibration JSON containing contact_offset_tcp_m.",
    )
    parser.add_argument("--no-combined-offsets", action="store_true")
    parser.add_argument("--no-soft-orientation-ramp", action="store_true")
    parser.add_argument("--approach-orientation-weight", type=float, default=0.0)
    parser.add_argument("--full-orientation-weight", type=float, default=1.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.15)
    parser.add_argument("--min-clearance-m", type=float, default=0.03)
    parser.add_argument(
        "--open-alignment-tolerance-m",
        type=float,
        default=None,
        help=(
            "Optional cross-axis tolerance for the pre_grasp gripper-open waypoint. "
            "Default scales from estimated target grasp width."
        ),
    )
    parser.add_argument("--open-position-tolerance-m", type=float, default=0.002)
    parser.add_argument("--open-orientation-tolerance-rad", type=float, default=0.05)
    parser.add_argument(
        "--task-axis-pregrasp-gate",
        action="store_true",
        help="Evaluate pre_grasp open alignment with task-frame jaw/backoff axes instead of a rigid quaternion gate.",
    )
    parser.add_argument(
        "--pregrasp-task-axis-position-tolerance-m",
        type=float,
        default=None,
        help="Optional pre_grasp contact-position tolerance for the task-axis gate. Default scales from target width.",
    )
    parser.add_argument("--pregrasp-task-jaw-axis-tolerance-rad", type=float, default=0.30)
    parser.add_argument("--pregrasp-task-backoff-axis-tolerance-rad", type=float, default=0.30)
    parser.add_argument("--grasp-palm-plausibility-margin-m", type=float, default=0.002)
    parser.add_argument(
        "--grasp-palm-alignment-tolerance-m",
        type=float,
        default=None,
        help="Optional grasp-plane tolerance along the palm axis. Default scales from estimated target grasp width.",
    )
    parser.add_argument("--min-pregrasp-surface-clearance-m", type=float, default=0.02)
    parser.add_argument("--min-grasp-insertion-fraction", type=float, default=0.75)
    parser.add_argument(
        "--task-axis-grasp-gate",
        action="store_true",
        help="Evaluate the final grasp with task-frame jaw/backoff axes instead of a rigid quaternion gate.",
    )
    parser.add_argument(
        "--grasp-task-axis-position-tolerance-m",
        type=float,
        default=None,
        help="Optional final grasp contact-position tolerance for the task-axis gate. Default scales from target width.",
    )
    parser.add_argument("--grasp-task-jaw-axis-tolerance-rad", type=float, default=0.35)
    parser.add_argument("--grasp-task-backoff-axis-tolerance-rad", type=float, default=0.48)
    parser.add_argument(
        "--grasp-depth-ramp-factors",
        default="1.0",
        help="Comma-separated internal final-grasp depth fractions. Values must be in (0,1]; 1.0 is appended.",
    )
    parser.add_argument(
        "--grasp-prior-capture-json",
        type=Path,
        default=None,
        help="Optional teleop capture JSON supplying a median grasp joint prior.",
    )
    parser.add_argument(
        "--grasp-prior-weight",
        type=float,
        default=0.2,
        help="Smoothness/reference weight applied only to the final grasp when a grasp prior is supplied.",
    )
    parser.add_argument(
        "--scene-obstacle-margin-m",
        type=float,
        default=0.025,
        help="Inflation margin around non-target fruit obstacles for TCP path clearance.",
    )
    parser.add_argument(
        "--open-gripper-half-extents-tcp-m",
        default="0.05,0.02,0.02",
        help=(
            "Open-gripper TCP-frame half-extents x,y,z in meters for non-target fruit clearance. "
            "By convention local x is jaw width, local y is palm thickness, and local z is backoff depth."
        ),
    )
    parser.add_argument(
        "--disable-scene-obstacle-clearance",
        action="store_true",
        help="Disable non-target fruit obstacle clearance checks.",
    )
    parser.add_argument(
        "--max-tail-tcp-x-m",
        type=float,
        default=0.68,
        help="Maximum allowed TCP x after grasp; use a negative value to disable this fixed-testbed safety gate.",
    )
    parser.add_argument(
        "--allow-pull-reverse-tail",
        action="store_true",
        help="Include reverse-pull retract candidates. Disabled by default until environment obstacles are modeled.",
    )
    parser.add_argument("--path-samples", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--position-weight", type=float, default=20.0)
    parser.add_argument("--posture-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--clearance-weight", type=float, default=2.0)
    parser.add_argument("--clearance-margin-m", type=float, default=0.03)
    parser.add_argument("--clearance-pair-count", type=int, default=10)
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()
    if args.surface_target_robot_m is not None and args.identity_validation_json is not None:
        parser.error("--surface-target-robot-m cannot be combined with --identity-validation-json")
    basket_pose_candidates = _basket_pose_candidates_from_args(args)

    base_config = IKSolverConfig(
        position_weight=args.position_weight,
        orientation_weight=args.full_orientation_weight,
        posture_weight=args.posture_weight,
        smoothness_weight=args.smoothness_weight,
        clearance_weight=args.clearance_weight,
        clearance_margin_m=args.clearance_margin_m,
        clearance_pair_count=args.clearance_pair_count,
        max_iterations=args.max_iterations,
    )
    results = plan_staged_harvest(
        paths=_paths_from_args(args),
        address=args.address,
        model=args.model,
        arm=args.arm,
        home_path=args.home,
        toolfit_path=args.toolfit,
        base_config=base_config,
        approach_orientation_weight=args.approach_orientation_weight,
        full_orientation_weight=args.full_orientation_weight,
        approach_backoff_m=args.approach_backoff_m,
        pregrasp_backoff_m=args.pregrasp_backoff_m,
        pregrasp_surface_clearances_m=_parse_float_tuple(args.pregrasp_surface_clearances_m),
        include_tail=args.include_tail,
        harvest_distance_m=args.harvest_distance_m,
        harvest_down_m=args.harvest_down_m,
        retract_distance_m=args.retract_distance_m,
        retract_up_m=args.retract_up_m,
        tail_orientation_weight=args.tail_orientation_weight,
        tail_orientation_required=not args.relax_tail_orientation,
        basket_pose_candidates=basket_pose_candidates,
        basket_orientation_weight=args.basket_orientation_weight,
        basket_orientation_required=args.basket_require_orientation,
        candidate_set=args.candidate_set,
        elevated_offsets_m=_parse_float_tuple(args.elevated_offsets_m),
        lateral_offsets_m=_parse_float_tuple(args.lateral_offsets_m),
        tcp_rolls_rad=_parse_degree_tuple(args.tcp_rolls_deg),
        grasp_insertion_factors=_parse_float_tuple(args.grasp_insertion_factors),
        palm_correction_factors=_parse_float_tuple(args.palm_correction_factors),
        max_grasp_insertion_factor=args.max_grasp_insertion_factor,
        surface_target_robot_m=_parse_vector3(args.surface_target_robot_m, name="surface target"),
        identity_validation_json=args.identity_validation_json,
        contact_offset_tcp_m=_contact_offset_from_args(args),
        include_combined_offsets=not args.no_combined_offsets,
        include_soft_orientation_ramp=not args.no_soft_orientation_ramp,
        position_tolerance_m=args.position_tolerance_m,
        orientation_tolerance_rad=args.orientation_tolerance_rad,
        min_clearance_m=args.min_clearance_m,
        open_alignment_tolerance_m=args.open_alignment_tolerance_m,
        open_position_tolerance_m=args.open_position_tolerance_m,
        open_orientation_tolerance_rad=args.open_orientation_tolerance_rad,
        use_task_axis_pregrasp_gate=args.task_axis_pregrasp_gate,
        pregrasp_task_axis_position_tolerance_m=args.pregrasp_task_axis_position_tolerance_m,
        pregrasp_task_jaw_axis_tolerance_rad=args.pregrasp_task_jaw_axis_tolerance_rad,
        pregrasp_task_backoff_axis_tolerance_rad=args.pregrasp_task_backoff_axis_tolerance_rad,
        grasp_palm_plausibility_margin_m=args.grasp_palm_plausibility_margin_m,
        grasp_palm_alignment_tolerance_m=args.grasp_palm_alignment_tolerance_m,
        min_pregrasp_surface_clearance_m=args.min_pregrasp_surface_clearance_m,
        min_grasp_insertion_fraction=args.min_grasp_insertion_fraction,
        use_task_axis_grasp_gate=args.task_axis_grasp_gate,
        grasp_task_axis_position_tolerance_m=args.grasp_task_axis_position_tolerance_m,
        grasp_task_jaw_axis_tolerance_rad=args.grasp_task_jaw_axis_tolerance_rad,
        grasp_task_backoff_axis_tolerance_rad=args.grasp_task_backoff_axis_tolerance_rad,
        grasp_depth_ramp_factors=_parse_float_tuple(args.grasp_depth_ramp_factors),
        scene_obstacle_margin_m=args.scene_obstacle_margin_m,
        open_gripper_half_extents_tcp_m=_parse_vector3(
            args.open_gripper_half_extents_tcp_m,
            name="open gripper half extents",
        ),
        use_scene_obstacle_clearance=not args.disable_scene_obstacle_clearance,
        max_tail_tcp_x_m=None if args.max_tail_tcp_x_m < 0.0 else args.max_tail_tcp_x_m,
        allow_pull_reverse_tail=args.allow_pull_reverse_tail,
        tail_prefix_count=args.tail_prefix_count,
        path_samples=args.path_samples,
        grasp_joint_prior=_grasp_joint_prior_from_args(args),
        runs_root=args.runs_root,
        run_label=_run_label(args),
        dataset_name=_parse_dataset_name(args),
    )
    for payload in results:
        target_name = payload["target"]["target_name"]
        print(
            target_name,
            "hard_gate_ok=",
            payload["hard_gate"]["ok"],
            "best_candidate=",
            payload["best_candidate_name"],
            "best_endpoint_ok=",
            payload["hard_gate"]["best_endpoint_ok"],
            "best_path_clearance_ok=",
            payload["hard_gate"]["best_path_clearance_ok"],
            "best_open_alignment_ok=",
            payload["hard_gate"]["best_open_alignment_ok"],
            "best_open_position_ok=",
            payload["hard_gate"]["best_open_position_ok"],
            "best_open_orientation_ok=",
            payload["hard_gate"]["best_open_orientation_ok"],
            "best_grasp_palm_plausibility_ok=",
            payload["hard_gate"]["best_grasp_palm_plausibility_ok"],
            "best_grasp_palm_alignment_ok=",
            payload["hard_gate"]["best_grasp_palm_alignment_ok"],
            "best_grasp_depth_ok=",
            payload["hard_gate"]["best_grasp_depth_ok"],
            "best_task_axis_grasp_ok=",
            payload["hard_gate"]["best_task_axis_grasp_ok"],
            "best_pregrasp_surface_clearance_ok=",
            payload["hard_gate"]["best_pregrasp_surface_clearance_ok"],
            "best_scene_obstacle_clearance_ok=",
            payload["hard_gate"]["best_scene_obstacle_clearance_ok"],
            "best_tail_forward_reach_ok=",
            payload["hard_gate"]["best_tail_forward_reach_ok"],
            "artifact=",
            payload["artifact_path"],
        )
        prefix_phase = payload.get("prefix_phase")
        if prefix_phase and prefix_phase["frozen_for_tail"]:
            print(
                " prefix",
                "best_candidate=",
                prefix_phase["best_candidate_name"],
                "ok=",
                prefix_phase["best_ok"],
            )
        for candidate in payload["candidates"][:5]:
            quality = candidate["manipulation_quality"]
            print(
                " candidate",
                candidate["rank"],
                candidate["name"],
                "family=",
                candidate["family"],
                "ok=",
                candidate["hard_gate"]["ok"],
                "endpoint_ok=",
                candidate["hard_gate"]["endpoint_ok"],
                "path_ok=",
                candidate["hard_gate"]["path_clearance_ok"],
                "scene_ok=",
                candidate["hard_gate"]["scene_obstacle_clearance_ok"],
                "open_alignment_ok=",
                candidate["hard_gate"]["open_alignment_ok"],
                "open_position_ok=",
                candidate["hard_gate"]["open_position_ok"],
                "open_orientation_ok=",
                candidate["hard_gate"]["open_orientation_ok"],
                "grasp_palm_plausibility_ok=",
                candidate["hard_gate"]["grasp_palm_plausibility_ok"],
                "grasp_palm_alignment_ok=",
                candidate["hard_gate"]["grasp_palm_alignment_ok"],
                "open_cross_axis_offset_m=",
                quality.get("open_cross_axis_offset_norm_m"),
                "open_position_error_m=",
                quality.get("open_position_error_m"),
                "open_orientation_error_rad=",
                quality.get("open_orientation_error_rad"),
                "grasp_palm_axis_offset_m=",
                quality.get("grasp_palm_axis_offset_m"),
                "palm_correction_factor=",
                candidate["diagnostics"].get("palm_correction_factor"),
                "palm_axis_correction_m=",
                quality.get("palm_axis_correction_m"),
                "grasp_insertion_m=",
                quality.get("grasp_insertion_m"),
                "grasp_depth_ok=",
                candidate["hard_gate"]["grasp_depth_ok"],
                "task_axis_grasp_ok=",
                candidate["hard_gate"]["task_axis_grasp_ok"],
                "task_axis_position_error_m=",
                quality.get("task_axis_grasp_position_error_m"),
                "task_axis_jaw_error_rad=",
                quality.get("task_axis_grasp_jaw_axis_error_rad"),
                "task_axis_backoff_error_rad=",
                quality.get("task_axis_grasp_backoff_axis_error_rad"),
                "pregrasp_surface_clearance_m=",
                quality.get("pregrasp_surface_clearance_m"),
                "path_min_clearance_m=",
                candidate["path_clearance"]["minimum_clearance_m"],
                "scene_min_clearance_m=",
                candidate["scene_obstacle_clearance"]["minimum_clearance_m"],
                "tail_forward_reach_ok=",
                candidate["hard_gate"]["tail_forward_reach_ok"],
            )
        best = payload["best_candidate"]
        if best is None:
            continue
        for waypoint in best["waypoints"]:
            print(
                " ",
                waypoint["name"],
                "mode=",
                waypoint["orientation_mode"],
                "endpoint_ok=",
                waypoint["endpoint_gate"]["ok"],
                "effective_endpoint_ok=",
                waypoint["endpoint_gate"].get("effective_ok", waypoint["endpoint_gate"]["ok"]),
                "effective_mode=",
                waypoint["endpoint_gate"].get("effective_mode", "full_pose"),
                "position_error_m=",
                waypoint["ik_result"]["position_error_m"],
                "orientation_error_rad=",
                waypoint["ik_result"]["orientation_error_rad"],
                "min_clearance_m=",
                waypoint["endpoint_gate"]["minimum_clearance_m"],
                "scene_clearance_m=",
                waypoint["endpoint_gate"]["scene_obstacle_minimum_clearance_m"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
