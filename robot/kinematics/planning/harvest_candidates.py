from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from robot.kinematics.task_frame import HarvestTaskFrame
from robot.kinematics.types import Array


WORLD_UP_ROBOT = np.asarray([0.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class HarvestWaypointTarget:
    name: str
    backoff_m: float
    backoff_reference: str
    orientation_mode: str
    orientation_weight: float
    orientation_required: bool
    target_pose: Array
    target_contact_position_robot_m: Array
    contact_offset_tcp_m: Array
    offset_robot_m: Array
    surface_target_robot_m: Array
    grasp_anchor_robot_m: Array
    surface_backoff_component_m: float
    grasp_anchor_backoff_component_m: float

    def __post_init__(self) -> None:
        pose = np.asarray(self.target_pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("target_pose must be 4x4")
        object.__setattr__(self, "target_pose", pose)
        object.__setattr__(
            self,
            "target_contact_position_robot_m",
            np.asarray(self.target_contact_position_robot_m, dtype=float).reshape(3),
        )
        object.__setattr__(
            self,
            "contact_offset_tcp_m",
            np.asarray(self.contact_offset_tcp_m, dtype=float).reshape(3),
        )
        object.__setattr__(self, "offset_robot_m", np.asarray(self.offset_robot_m, dtype=float).reshape(3))
        object.__setattr__(
            self,
            "surface_target_robot_m",
            np.asarray(self.surface_target_robot_m, dtype=float).reshape(3),
        )
        object.__setattr__(
            self,
            "grasp_anchor_robot_m",
            np.asarray(self.grasp_anchor_robot_m, dtype=float).reshape(3),
        )


@dataclass(frozen=True)
class HarvestPlanCandidate:
    name: str
    family: str
    waypoints: tuple[HarvestWaypointTarget, ...]
    diagnostics: dict[str, float | str | bool | None | list[float]]


def append_harvest_tail(
    candidate: HarvestPlanCandidate,
    frame: HarvestTaskFrame,
    *,
    full_orientation_weight: float,
    tail_orientation_weight: float | None = None,
    tail_orientation_required: bool = True,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_mode: str = "prefix_approach",
    retract_distance_m: float = 0.12,
    retract_up_m: float = 0.04,
    basket_pose: Array | None = None,
    basket_orientation_weight: float = 0.0,
    basket_orientation_required: bool = False,
) -> HarvestPlanCandidate:
    """Append harvest/retract/basket waypoints without changing the grasp prefix."""

    waypoint_names = {waypoint.name for waypoint in candidate.waypoints}
    if "harvest" in waypoint_names or "retract" in waypoint_names or "basket_place" in waypoint_names:
        raise ValueError("candidate already contains harvest tail waypoints")
    approach = _required_waypoint(candidate, "approach")
    grasp = _required_waypoint(candidate, "grasp")
    tail = _tail_waypoints_from_prefix(
        frame=frame,
        approach=approach,
        grasp=grasp,
        full_orientation_weight=full_orientation_weight,
        tail_orientation_weight=tail_orientation_weight,
        tail_orientation_required=tail_orientation_required,
        harvest_distance_m=harvest_distance_m,
        harvest_down_m=harvest_down_m,
        retract_mode=retract_mode,
        retract_distance_m=retract_distance_m,
        retract_up_m=retract_up_m,
        basket_pose=basket_pose,
        basket_orientation_weight=basket_orientation_weight,
        basket_orientation_required=basket_orientation_required,
    )
    diagnostics = dict(candidate.diagnostics)
    diagnostics["tail_uses_frozen_prefix"] = True
    diagnostics["tail_retract_mode"] = retract_mode
    diagnostics["tail_retract_distance_m"] = float(retract_distance_m)
    diagnostics["tail_retract_up_m"] = float(retract_up_m)
    return HarvestPlanCandidate(
        name=f"{candidate.name}_tail_{_tail_label(retract_mode, retract_distance_m, retract_up_m)}",
        family=candidate.family,
        waypoints=tuple(candidate.waypoints) + tuple(tail),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class GraspInsertionCandidate:
    factor: float
    palm_correction_factor: float
    insertion_m: float
    nominal_insertion_m: float
    nominal_source: str
    center_projection_insertion_m: float
    half_depth_cap_m: float | None
    insertion_cap_m: float | None
    insertion_clipped: bool
    center_palm_projection_m: float
    palm_axis_correction_m: float
    palm_axis_correction_cap_m: float | None
    palm_axis_correction_clipped: bool
    grasp_anchor_robot_m: Array
    surface_target_robot_m: Array

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grasp_anchor_robot_m",
            np.asarray(self.grasp_anchor_robot_m, dtype=float).reshape(3),
        )
        object.__setattr__(
            self,
            "surface_target_robot_m",
            np.asarray(self.surface_target_robot_m, dtype=float).reshape(3),
        )


def grasp_depth_policy_summary(
    frame: HarvestTaskFrame,
    grasp_insertion_factors: Iterable[float],
    palm_correction_factors: Iterable[float] = (0.0, 0.5, 1.0),
    max_grasp_insertion_factor: float = 1.0,
    surface_target_robot_m: Array | None = None,
) -> dict[str, float | list[float] | str | None]:
    candidates = grasp_insertion_candidates(
        frame,
        grasp_insertion_factors,
        palm_correction_factors,
        max_grasp_insertion_factor=max_grasp_insertion_factor,
        surface_target_robot_m=surface_target_robot_m,
    )
    if not candidates:
        raise ValueError("at least one grasp insertion candidate is required")
    first = candidates[0]
    return {
        "policy": "depth_insertion_with_bounded_fruit_center_palm_axis_correction",
        "surface_target_source": "override" if surface_target_robot_m is not None else "task_frame",
        "surface_target_robot_m": first.surface_target_robot_m.tolist(),
        "center_projection_insertion_m": first.center_projection_insertion_m,
        "half_depth_cap_m": first.half_depth_cap_m,
        "insertion_cap_m": first.insertion_cap_m,
        "max_grasp_insertion_factor": float(max_grasp_insertion_factor),
        "center_palm_projection_m": first.center_palm_projection_m,
        "palm_axis_correction_cap_m": first.palm_axis_correction_cap_m,
        "nominal_grasp_insertion_m": first.nominal_insertion_m,
        "nominal_source": first.nominal_source,
        "insertion_factors": sorted({candidate.factor for candidate in candidates}),
        "palm_correction_factors": sorted({candidate.palm_correction_factor for candidate in candidates}),
        "candidate_insertions_m": [candidate.insertion_m for candidate in candidates],
        "candidate_palm_axis_corrections_m": [candidate.palm_axis_correction_m for candidate in candidates],
    }


def grasp_insertion_candidates(
    frame: HarvestTaskFrame,
    grasp_insertion_factors: Iterable[float],
    palm_correction_factors: Iterable[float] = (0.0, 0.5, 1.0),
    *,
    max_grasp_insertion_factor: float = 1.0,
    surface_target_robot_m: Array | None = None,
) -> tuple[GraspInsertionCandidate, ...]:
    factors = tuple(float(value) for value in grasp_insertion_factors)
    palm_factors = tuple(float(value) for value in palm_correction_factors)
    if not factors:
        raise ValueError("grasp_insertion_factors must contain at least one value")
    if not palm_factors:
        raise ValueError("palm_correction_factors must contain at least one value")
    if any(value < 0.0 for value in factors):
        raise ValueError("grasp insertion factors must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in palm_factors):
        raise ValueError("palm correction factors must be in [0, 1]")
    if max_grasp_insertion_factor < 0.0:
        raise ValueError("max_grasp_insertion_factor must be non-negative")

    backoff_axis = _unit(frame.backoff_axis_robot)
    palm_axis = _unit(frame.palm_axis_robot)
    if surface_target_robot_m is None:
        surface_target = np.asarray(frame.target_position_robot_m, dtype=float).reshape(3)
    else:
        surface_target = np.asarray(surface_target_robot_m, dtype=float).reshape(3)
    fruit_center = np.asarray(frame.fruit_center_robot_m, dtype=float).reshape(3)
    center_projection = max(0.0, float(np.dot(fruit_center - surface_target, -backoff_axis)))
    center_palm_projection = float(np.dot(fruit_center - surface_target, palm_axis))
    half_depth_cap = None
    if frame.estimated_grasp_depth_m is not None and frame.estimated_grasp_depth_m > 0.0:
        half_depth_cap = 0.5 * float(frame.estimated_grasp_depth_m)
    max_palm_axis_correction = center_palm_projection
    if half_depth_cap is not None:
        max_palm_axis_correction = float(np.clip(max_palm_axis_correction, -half_depth_cap, half_depth_cap))

    if half_depth_cap is not None:
        nominal = half_depth_cap
        nominal_source = "half_estimated_grasp_depth"
    else:
        nominal = center_projection
        nominal_source = "fruit_center_projection_fallback"

    if half_depth_cap is not None:
        nominal = min(nominal, half_depth_cap)
    insertion_cap = None
    if half_depth_cap is not None:
        insertion_cap = min(float(frame.estimated_grasp_depth_m), float(max_grasp_insertion_factor) * half_depth_cap)
    elif nominal > 0.0:
        insertion_cap = float(max_grasp_insertion_factor) * float(nominal)

    candidates: list[GraspInsertionCandidate] = []
    seen_candidates: set[tuple[float, float]] = set()
    for factor in factors:
        raw_insertion = factor * nominal
        insertion = raw_insertion
        if insertion_cap is not None:
            insertion = min(insertion, insertion_cap)
        insertion = max(0.0, insertion)
        for palm_factor in palm_factors:
            palm_axis_correction = float(palm_factor) * max_palm_axis_correction
            candidate_key = (round(float(insertion), 6), round(float(palm_axis_correction), 6))
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            candidates.append(
                GraspInsertionCandidate(
                    factor=float(factor),
                    palm_correction_factor=float(palm_factor),
                    insertion_m=float(insertion),
                    nominal_insertion_m=float(nominal),
                    nominal_source=nominal_source,
                    center_projection_insertion_m=float(center_projection),
                    half_depth_cap_m=half_depth_cap,
                    insertion_cap_m=insertion_cap,
                    insertion_clipped=abs(float(insertion) - float(raw_insertion)) > 1e-9,
                    center_palm_projection_m=float(center_palm_projection),
                    palm_axis_correction_m=float(palm_axis_correction),
                    palm_axis_correction_cap_m=half_depth_cap,
                    palm_axis_correction_clipped=(
                        abs(float(max_palm_axis_correction) - float(center_palm_projection)) > 1e-9
                    ),
                    grasp_anchor_robot_m=(
                        surface_target
                        - float(insertion) * backoff_axis
                        + float(palm_axis_correction) * palm_axis
                    ),
                    surface_target_robot_m=surface_target,
                )
            )
    return tuple(candidates)


def generate_harvest_plan_candidates(
    frame: HarvestTaskFrame,
    *,
    approach_backoff_m: float,
    pregrasp_backoff_m: float,
    approach_orientation_weight: float,
    full_orientation_weight: float,
    tail_orientation_weight: float | None = None,
    tail_orientation_required: bool = True,
    pregrasp_surface_clearances_m: Iterable[float] = (0.02, 0.03),
    include_tail: bool = False,
    harvest_distance_m: float = 0.05,
    harvest_down_m: float = 0.025,
    retract_distance_m: float = 0.12,
    retract_up_m: float = 0.04,
    basket_pose: Array | None = None,
    basket_orientation_weight: float | None = None,
    basket_orientation_required: bool = False,
    elevated_offsets_m: Iterable[float] = (0.05, 0.08),
    lateral_offsets_m: Iterable[float] = (0.04, 0.06, 0.08, 0.10, 0.12, -0.04, -0.06, -0.08),
    tcp_rolls_rad: Iterable[float] = (0.0, np.pi),
    grasp_insertion_factors: Iterable[float] = (0.5, 0.75, 1.0, 1.25),
    palm_correction_factors: Iterable[float] = (0.0, 0.5, 1.0),
    max_grasp_insertion_factor: float = 1.0,
    scene_avoidance_offsets_m: Iterable[Array] = (),
    surface_target_robot_m: Array | None = None,
    contact_offset_tcp_m: Array | None = None,
    include_combined_offsets: bool = True,
    include_soft_orientation_ramp: bool = True,
) -> tuple[HarvestPlanCandidate, ...]:
    """Generate target-agnostic staged harvest IK candidates.

    Candidates vary waypoint geometry and orientation policy using task-frame
    axes. They do not branch on target names, dataset IDs, or tomato-specific
    constants.
    """

    tcp_rolls = tuple(float(value) for value in tcp_rolls_rad)
    if harvest_down_m < 0.0:
        raise ValueError("harvest_down_m must be non-negative")
    insertion_candidates = grasp_insertion_candidates(
        frame,
        grasp_insertion_factors,
        palm_correction_factors,
        max_grasp_insertion_factor=max_grasp_insertion_factor,
        surface_target_robot_m=surface_target_robot_m,
    )
    basket_pose_array = None if basket_pose is None else np.asarray(basket_pose, dtype=float).reshape(4, 4)
    contact_offset = _contact_offset(contact_offset_tcp_m)
    scene_avoidance_offsets = _offset_vectors(scene_avoidance_offsets_m)
    basket_weight = (
        float(full_orientation_weight)
        if basket_orientation_weight is None
        else float(basket_orientation_weight)
    )
    pregrasp_clearances = _pregrasp_surface_clearances(pregrasp_surface_clearances_m)
    candidates: list[HarvestPlanCandidate] = []
    for insertion in insertion_candidates:
        for pregrasp_m in _pregrasp_backoff_options(
            insertion,
            lower_bound_m=pregrasp_backoff_m,
            surface_clearances_m=pregrasp_clearances,
        ):
            for tcp_roll_rad in tcp_rolls:
                candidates.append(
                    _candidate(
                        frame,
                        name=_name_with_depth_pregrasp_and_roll("straight", insertion, pregrasp_m, tcp_roll_rad),
                        family="straight",
                        offsets=(np.zeros(3), np.zeros(3)),
                        approach_backoff_m=approach_backoff_m,
                        pregrasp_backoff_m=pregrasp_m,
                        approach_orientation_weight=approach_orientation_weight,
                        full_orientation_weight=full_orientation_weight,
                        tail_orientation_weight=tail_orientation_weight,
                        tail_orientation_required=tail_orientation_required,
                        include_tail=include_tail,
                        harvest_distance_m=harvest_distance_m,
                        harvest_down_m=harvest_down_m,
                        retract_distance_m=retract_distance_m,
                        retract_up_m=retract_up_m,
                        basket_pose=basket_pose_array,
                        basket_orientation_weight=basket_weight,
                        basket_orientation_required=basket_orientation_required,
                        tcp_roll_rad=tcp_roll_rad,
                        insertion=insertion,
                        contact_offset_tcp_m=contact_offset,
                        diagnostics={
                            "vertical_offset_m": 0.0,
                            "lateral_offset_m": 0.0,
                            "offset_mode": "none",
                            "pregrasp_backoff_m": float(pregrasp_m),
                            "pregrasp_surface_clearance_m": float(pregrasp_m - insertion.insertion_m),
                            "tcp_roll_rad": tcp_roll_rad,
                            "contact_offset_tcp_m": contact_offset.tolist(),
                        },
                    )
                )

    vertical_offsets = tuple(_positive_values(elevated_offsets_m))
    lateral_offsets = tuple(float(value) for value in lateral_offsets_m if abs(float(value)) > 1e-9)
    lateral_axis = _unit(frame.jaw_axis_robot)

    for insertion in insertion_candidates:
        for pregrasp_m in _pregrasp_backoff_options(
            insertion,
            lower_bound_m=pregrasp_backoff_m,
            surface_clearances_m=pregrasp_clearances,
        ):
            for vertical_m in vertical_offsets:
                offset = WORLD_UP_ROBOT * vertical_m
                for suffix, offsets, offset_mode in _offset_variants(offset):
                    for tcp_roll_rad in tcp_rolls:
                        candidates.append(
                            _candidate(
                                frame,
                                name=_name_with_depth_pregrasp_and_roll(
                                    f"elevated_{_cm_label(vertical_m)}_{suffix}",
                                    insertion,
                                    pregrasp_m,
                                    tcp_roll_rad,
                                ),
                                family="elevated",
                                offsets=offsets,
                                approach_backoff_m=approach_backoff_m,
                                pregrasp_backoff_m=pregrasp_m,
                                approach_orientation_weight=approach_orientation_weight,
                                full_orientation_weight=full_orientation_weight,
                                tail_orientation_weight=tail_orientation_weight,
                                tail_orientation_required=tail_orientation_required,
                                include_tail=include_tail,
                                harvest_distance_m=harvest_distance_m,
                                harvest_down_m=harvest_down_m,
                                retract_distance_m=retract_distance_m,
                                retract_up_m=retract_up_m,
                                basket_pose=basket_pose_array,
                                basket_orientation_weight=basket_weight,
                                basket_orientation_required=basket_orientation_required,
                                tcp_roll_rad=tcp_roll_rad,
                                insertion=insertion,
                                contact_offset_tcp_m=contact_offset,
                                diagnostics={
                                    "vertical_offset_m": vertical_m,
                                    "lateral_offset_m": 0.0,
                                    "offset_mode": offset_mode,
                                    "pregrasp_backoff_m": float(pregrasp_m),
                                    "pregrasp_surface_clearance_m": float(pregrasp_m - insertion.insertion_m),
                                    "tcp_roll_rad": tcp_roll_rad,
                                    "contact_offset_tcp_m": contact_offset.tolist(),
                                },
                            )
                        )

    for insertion in insertion_candidates:
        for pregrasp_m in _pregrasp_backoff_options(
            insertion,
            lower_bound_m=pregrasp_backoff_m,
            surface_clearances_m=pregrasp_clearances,
        ):
            for lateral_m in lateral_offsets:
                offset = lateral_axis * lateral_m
                for suffix, offsets, offset_mode in _offset_variants(offset):
                    for tcp_roll_rad in tcp_rolls:
                        candidates.append(
                            _candidate(
                                frame,
                                name=_name_with_depth_pregrasp_and_roll(
                                    f"lateral_{_signed_cm_label(lateral_m)}_{suffix}",
                                    insertion,
                                    pregrasp_m,
                                    tcp_roll_rad,
                                ),
                                family="lateral",
                                offsets=offsets,
                                approach_backoff_m=approach_backoff_m,
                                pregrasp_backoff_m=pregrasp_m,
                                approach_orientation_weight=approach_orientation_weight,
                                full_orientation_weight=full_orientation_weight,
                                tail_orientation_weight=tail_orientation_weight,
                                tail_orientation_required=tail_orientation_required,
                                include_tail=include_tail,
                                harvest_distance_m=harvest_distance_m,
                                harvest_down_m=harvest_down_m,
                                retract_distance_m=retract_distance_m,
                                retract_up_m=retract_up_m,
                                basket_pose=basket_pose_array,
                                basket_orientation_weight=basket_weight,
                                basket_orientation_required=basket_orientation_required,
                                tcp_roll_rad=tcp_roll_rad,
                                insertion=insertion,
                                contact_offset_tcp_m=contact_offset,
                                diagnostics={
                                    "vertical_offset_m": 0.0,
                                    "lateral_offset_m": lateral_m,
                                    "offset_mode": offset_mode,
                                    "pregrasp_backoff_m": float(pregrasp_m),
                                    "pregrasp_surface_clearance_m": float(pregrasp_m - insertion.insertion_m),
                                    "tcp_roll_rad": tcp_roll_rad,
                                    "contact_offset_tcp_m": contact_offset.tolist(),
                                },
                            )
                        )

    if include_combined_offsets and vertical_offsets and lateral_offsets:
        vertical_m = vertical_offsets[0]
        for insertion in insertion_candidates:
            for pregrasp_m in _pregrasp_backoff_options(
                insertion,
                lower_bound_m=pregrasp_backoff_m,
                surface_clearances_m=pregrasp_clearances,
            ):
                for lateral_m in lateral_offsets:
                    offset = WORLD_UP_ROBOT * vertical_m + lateral_axis * lateral_m
                    for suffix, offsets, offset_mode in _offset_variants(offset):
                        for tcp_roll_rad in tcp_rolls:
                            candidates.append(
                                _candidate(
                                    frame,
                                    name=_name_with_depth_pregrasp_and_roll(
                                        (
                                            f"elevated_{_cm_label(vertical_m)}_"
                                            f"lateral_{_signed_cm_label(lateral_m)}_{suffix}"
                                        ),
                                        insertion,
                                        pregrasp_m,
                                        tcp_roll_rad,
                                    ),
                                    family="elevated_lateral",
                                    offsets=offsets,
                                    approach_backoff_m=approach_backoff_m,
                                    pregrasp_backoff_m=pregrasp_m,
                                    approach_orientation_weight=approach_orientation_weight,
                                    full_orientation_weight=full_orientation_weight,
                                    tail_orientation_weight=tail_orientation_weight,
                                    tail_orientation_required=tail_orientation_required,
                                    include_tail=include_tail,
                                    harvest_distance_m=harvest_distance_m,
                                    harvest_down_m=harvest_down_m,
                                    retract_distance_m=retract_distance_m,
                                    retract_up_m=retract_up_m,
                                    basket_pose=basket_pose_array,
                                    basket_orientation_weight=basket_weight,
                                    basket_orientation_required=basket_orientation_required,
                                    tcp_roll_rad=tcp_roll_rad,
                                    insertion=insertion,
                                    contact_offset_tcp_m=contact_offset,
                                    diagnostics={
                                        "vertical_offset_m": vertical_m,
                                        "lateral_offset_m": lateral_m,
                                        "offset_mode": offset_mode,
                                        "pregrasp_backoff_m": float(pregrasp_m),
                                        "pregrasp_surface_clearance_m": float(pregrasp_m - insertion.insertion_m),
                                        "tcp_roll_rad": tcp_roll_rad,
                                        "contact_offset_tcp_m": contact_offset.tolist(),
                                    },
                                )
                            )

    zero_offset = np.zeros(3, dtype=float)
    for offset_index, scene_offset in enumerate(scene_avoidance_offsets, start=1):
        offset_norm_m = float(np.linalg.norm(scene_offset))
        if offset_norm_m <= 1e-9:
            continue
        for insertion in insertion_candidates:
            for pregrasp_m in _pregrasp_backoff_options(
                insertion,
                lower_bound_m=pregrasp_backoff_m,
                surface_clearances_m=pregrasp_clearances,
            ):
                for tcp_roll_rad in tcp_rolls:
                    candidates.append(
                        _candidate(
                            frame,
                            name=_name_with_depth_pregrasp_and_roll(
                                f"scene_avoid_{offset_index}_{_cm_label(offset_norm_m)}",
                                insertion,
                                pregrasp_m,
                                tcp_roll_rad,
                            ),
                            family="scene_avoidance",
                            offsets=(scene_offset, zero_offset),
                            approach_backoff_m=approach_backoff_m,
                            pregrasp_backoff_m=pregrasp_m,
                            approach_orientation_weight=approach_orientation_weight,
                            full_orientation_weight=full_orientation_weight,
                            tail_orientation_weight=tail_orientation_weight,
                            tail_orientation_required=tail_orientation_required,
                            include_tail=include_tail,
                            harvest_distance_m=harvest_distance_m,
                            harvest_down_m=harvest_down_m,
                            retract_distance_m=retract_distance_m,
                            retract_up_m=retract_up_m,
                            basket_pose=basket_pose_array,
                            basket_orientation_weight=basket_weight,
                            basket_orientation_required=basket_orientation_required,
                            tcp_roll_rad=tcp_roll_rad,
                            insertion=insertion,
                            contact_offset_tcp_m=contact_offset,
                            diagnostics={
                                "vertical_offset_m": float(scene_offset[2]),
                                "lateral_offset_m": float(np.dot(scene_offset, lateral_axis)),
                                "offset_mode": "scene_avoidance_approach_only",
                                "scene_avoidance_offset_index": float(offset_index),
                                "scene_avoidance_offset_norm_m": offset_norm_m,
                                "scene_avoidance_offset_robot_m": scene_offset.tolist(),
                                "pregrasp_backoff_m": float(pregrasp_m),
                                "pregrasp_surface_clearance_m": float(pregrasp_m - insertion.insertion_m),
                                "tcp_roll_rad": tcp_roll_rad,
                                "contact_offset_tcp_m": contact_offset.tolist(),
                            },
                        )
                    )

    if include_soft_orientation_ramp:
        for insertion in insertion_candidates:
            for pregrasp_m in _pregrasp_backoff_options(
                insertion,
                lower_bound_m=pregrasp_backoff_m,
                surface_clearances_m=pregrasp_clearances,
            ):
                for tcp_roll_rad in tcp_rolls:
                    candidates.append(
                        _soft_orientation_ramp_candidate(
                            frame,
                            approach_backoff_m=approach_backoff_m,
                            pregrasp_backoff_m=pregrasp_m,
                            approach_orientation_weight=approach_orientation_weight,
                            full_orientation_weight=full_orientation_weight,
                            tail_orientation_weight=tail_orientation_weight,
                            tail_orientation_required=tail_orientation_required,
                            include_tail=include_tail,
                            harvest_distance_m=harvest_distance_m,
                            harvest_down_m=harvest_down_m,
                            retract_distance_m=retract_distance_m,
                            retract_up_m=retract_up_m,
                            basket_pose=basket_pose_array,
                            basket_orientation_weight=basket_weight,
                            basket_orientation_required=basket_orientation_required,
                            tcp_roll_rad=tcp_roll_rad,
                            insertion=insertion,
                            contact_offset_tcp_m=contact_offset,
                        )
                    )
                for offset_index, scene_offset in enumerate(scene_avoidance_offsets, start=1):
                    offset_norm_m = float(np.linalg.norm(scene_offset))
                    if offset_norm_m <= 1e-9:
                        continue
                    candidates.append(
                        _soft_orientation_ramp_candidate(
                            frame,
                            approach_backoff_m=approach_backoff_m,
                            pregrasp_backoff_m=pregrasp_m,
                            approach_orientation_weight=approach_orientation_weight,
                            full_orientation_weight=full_orientation_weight,
                            tail_orientation_weight=tail_orientation_weight,
                            tail_orientation_required=tail_orientation_required,
                            include_tail=include_tail,
                            harvest_distance_m=harvest_distance_m,
                            harvest_down_m=harvest_down_m,
                            retract_distance_m=retract_distance_m,
                            retract_up_m=retract_up_m,
                            basket_pose=basket_pose_array,
                            basket_orientation_weight=basket_weight,
                            basket_orientation_required=basket_orientation_required,
                            tcp_roll_rad=tcp_roll_rad,
                            insertion=insertion,
                            contact_offset_tcp_m=contact_offset,
                            approach_offset_robot_m=scene_offset,
                            name_prefix=f"scene_avoid_{offset_index}_{_cm_label(offset_norm_m)}_soft_orientation_ramp",
                            family="scene_avoidance_ramp",
                            diagnostics={
                                "offset_mode": "scene_avoidance_ramp",
                                "scene_avoidance_offset_index": float(offset_index),
                                "scene_avoidance_offset_norm_m": offset_norm_m,
                                "scene_avoidance_offset_robot_m": scene_offset.tolist(),
                            },
                        )
                    )

    return tuple(candidates)


def _candidate(
    frame: HarvestTaskFrame,
    *,
    name: str,
    family: str,
    offsets: tuple[Array, Array],
    approach_backoff_m: float,
    pregrasp_backoff_m: float,
    approach_orientation_weight: float,
    full_orientation_weight: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    include_tail: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_distance_m: float,
    retract_up_m: float,
    basket_pose: Array | None,
    basket_orientation_weight: float,
    basket_orientation_required: bool,
    tcp_roll_rad: float,
    insertion: GraspInsertionCandidate,
    contact_offset_tcp_m: Array,
    diagnostics: dict[str, float | str | bool | None | list[float]],
) -> HarvestPlanCandidate:
    approach_offset, pregrasp_offset = offsets
    waypoints = [
        _waypoint(
            frame,
            name="approach",
            backoff_m=approach_backoff_m,
            offset_robot_m=approach_offset,
            orientation_mode="position_only",
            orientation_weight=approach_orientation_weight,
            orientation_required=False,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
        _waypoint(
            frame,
            name="pre_grasp",
            backoff_m=pregrasp_backoff_m,
            offset_robot_m=pregrasp_offset,
            orientation_mode="full_pose",
            orientation_weight=full_orientation_weight,
            orientation_required=True,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
        _waypoint(
            frame,
            name="grasp",
            backoff_m=0.0,
            offset_robot_m=np.zeros(3),
            orientation_mode="full_pose",
            orientation_weight=full_orientation_weight,
            orientation_required=True,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
    ]
    if include_tail:
        waypoints.extend(
            _tail_waypoints_from_prefix(
                frame=frame,
                approach=waypoints[0],
                grasp=waypoints[2],
                full_orientation_weight=full_orientation_weight,
                tail_orientation_weight=tail_orientation_weight,
                tail_orientation_required=tail_orientation_required,
                harvest_distance_m=harvest_distance_m,
                harvest_down_m=harvest_down_m,
                retract_mode="backoff",
                retract_distance_m=retract_distance_m,
                retract_up_m=retract_up_m,
                basket_pose=basket_pose,
                basket_orientation_weight=basket_orientation_weight,
                basket_orientation_required=basket_orientation_required,
            )
        )
    candidate_diagnostics = dict(diagnostics)
    candidate_diagnostics.update(_insertion_diagnostics(insertion))
    return HarvestPlanCandidate(
        name=name,
        family=family,
        waypoints=tuple(waypoints),
        diagnostics=candidate_diagnostics,
    )


def _soft_orientation_ramp_candidate(
    frame: HarvestTaskFrame,
    *,
    approach_backoff_m: float,
    pregrasp_backoff_m: float,
    approach_orientation_weight: float,
    full_orientation_weight: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    include_tail: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_distance_m: float,
    retract_up_m: float,
    basket_pose: Array | None,
    basket_orientation_weight: float,
    basket_orientation_required: bool,
    tcp_roll_rad: float,
    insertion: GraspInsertionCandidate,
    contact_offset_tcp_m: Array,
    approach_offset_robot_m: Array | None = None,
    name_prefix: str = "soft_orientation_ramp",
    family: str = "orientation_ramp",
    diagnostics: dict[str, float | str | bool | None | list[float]] | None = None,
) -> HarvestPlanCandidate:
    mid_backoff_m = 0.5 * (float(approach_backoff_m) + float(pregrasp_backoff_m))
    approach_offset = (
        np.zeros(3)
        if approach_offset_robot_m is None
        else np.asarray(approach_offset_robot_m, dtype=float).reshape(3)
    )
    mid_offset = 0.5 * approach_offset
    waypoints = [
        _waypoint(
            frame,
            name="approach",
            backoff_m=approach_backoff_m,
            offset_robot_m=approach_offset,
            orientation_mode="position_only",
            orientation_weight=approach_orientation_weight,
            orientation_required=False,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
        _waypoint(
            frame,
            name="orient_soft_03",
            backoff_m=approach_backoff_m,
            offset_robot_m=approach_offset,
            orientation_mode="soft_full_pose",
            orientation_weight=0.3 * full_orientation_weight,
            orientation_required=False,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
        _waypoint(
            frame,
            name="orient_soft_07",
            backoff_m=mid_backoff_m,
            offset_robot_m=mid_offset,
            orientation_mode="soft_full_pose",
            orientation_weight=0.7 * full_orientation_weight,
            orientation_required=False,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
        _waypoint(
            frame,
            name="pre_grasp",
            backoff_m=pregrasp_backoff_m,
            offset_robot_m=np.zeros(3),
            orientation_mode="full_pose",
            orientation_weight=full_orientation_weight,
            orientation_required=True,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
        _waypoint(
            frame,
            name="grasp",
            backoff_m=0.0,
            offset_robot_m=np.zeros(3),
            orientation_mode="full_pose",
            orientation_weight=full_orientation_weight,
            orientation_required=True,
            tcp_roll_rad=tcp_roll_rad,
            insertion=insertion,
            contact_offset_tcp_m=contact_offset_tcp_m,
        ),
    ]
    if include_tail:
        waypoints.extend(
            _tail_waypoints_from_prefix(
                frame=frame,
                approach=waypoints[0],
                grasp=waypoints[-1],
                full_orientation_weight=full_orientation_weight,
                tail_orientation_weight=tail_orientation_weight,
                tail_orientation_required=tail_orientation_required,
                harvest_distance_m=harvest_distance_m,
                harvest_down_m=harvest_down_m,
                retract_mode="backoff",
                retract_distance_m=retract_distance_m,
                retract_up_m=retract_up_m,
                basket_pose=basket_pose,
                basket_orientation_weight=basket_orientation_weight,
                basket_orientation_required=basket_orientation_required,
            )
        )
    candidate_diagnostics = {
        "vertical_offset_m": float(approach_offset[2]),
        "lateral_offset_m": float(np.dot(approach_offset, _unit(frame.jaw_axis_robot))),
        "offset_mode": "none" if float(np.linalg.norm(approach_offset)) <= 1e-9 else "approach_ramp",
        "pregrasp_backoff_m": float(pregrasp_backoff_m),
        "pregrasp_surface_clearance_m": float(pregrasp_backoff_m - insertion.insertion_m),
        "tcp_roll_rad": tcp_roll_rad,
        "contact_offset_tcp_m": _contact_offset(contact_offset_tcp_m).tolist(),
    }
    if diagnostics:
        candidate_diagnostics.update(diagnostics)
    candidate_diagnostics.update(_insertion_diagnostics(insertion))
    return HarvestPlanCandidate(
        name=_name_with_depth_pregrasp_and_roll(name_prefix, insertion, pregrasp_backoff_m, tcp_roll_rad),
        family=family,
        waypoints=tuple(waypoints),
        diagnostics=candidate_diagnostics,
    )


def _waypoint(
    frame: HarvestTaskFrame,
    *,
    name: str,
    backoff_m: float,
    offset_robot_m: Array,
    orientation_mode: str,
    orientation_weight: float,
    orientation_required: bool,
    tcp_roll_rad: float,
    insertion: GraspInsertionCandidate,
    contact_offset_tcp_m: Array,
) -> HarvestWaypointTarget:
    offset = np.asarray(offset_robot_m, dtype=float).reshape(3)
    contact_offset = _contact_offset(contact_offset_tcp_m)
    backoff_axis = _unit(frame.backoff_axis_robot)
    surface_target = np.asarray(insertion.surface_target_robot_m, dtype=float).reshape(3)
    grasp_anchor = np.asarray(insertion.grasp_anchor_robot_m, dtype=float).reshape(3)
    pose = frame.tcp_grasp_pose.T.copy()
    pose[:3, :3] = pose[:3, :3] @ _rot_z(tcp_roll_rad)
    contact_position = grasp_anchor + float(backoff_m) * backoff_axis + offset
    pose[:3, 3] = contact_position - pose[:3, :3] @ contact_offset
    surface_backoff = float(np.dot(contact_position - surface_target, backoff_axis))
    anchor_backoff = float(np.dot(contact_position - grasp_anchor, backoff_axis))
    return HarvestWaypointTarget(
        name=name,
        backoff_m=float(backoff_m),
        backoff_reference="inserted_grasp_anchor",
        orientation_mode=orientation_mode,
        orientation_weight=float(orientation_weight),
        orientation_required=bool(orientation_required),
        target_pose=pose,
        target_contact_position_robot_m=contact_position,
        contact_offset_tcp_m=contact_offset,
        offset_robot_m=contact_position - surface_target,
        surface_target_robot_m=surface_target,
        grasp_anchor_robot_m=grasp_anchor,
        surface_backoff_component_m=surface_backoff,
        grasp_anchor_backoff_component_m=anchor_backoff,
    )


def _tail_waypoints_from_prefix(
    *,
    frame: HarvestTaskFrame,
    approach: HarvestWaypointTarget,
    grasp: HarvestWaypointTarget,
    full_orientation_weight: float,
    tail_orientation_weight: float | None,
    tail_orientation_required: bool,
    harvest_distance_m: float,
    harvest_down_m: float,
    retract_mode: str,
    retract_distance_m: float,
    retract_up_m: float,
    basket_pose: Array | None,
    basket_orientation_weight: float,
    basket_orientation_required: bool,
) -> list[HarvestWaypointTarget]:
    if retract_distance_m < 0.0:
        raise ValueError("retract_distance_m must be non-negative")
    if retract_up_m < 0.0:
        raise ValueError("retract_up_m must be non-negative")
    contact_offset = _contact_offset(grasp.contact_offset_tcp_m)
    approach_pose = np.asarray(approach.target_pose, dtype=float).reshape(4, 4)
    approach_contact_position = np.asarray(approach.target_contact_position_robot_m, dtype=float).reshape(3)
    grasp_pose = np.asarray(grasp.target_pose, dtype=float).reshape(4, 4)
    grasp_contact_position = np.asarray(grasp.target_contact_position_robot_m, dtype=float).reshape(3)
    harvest_contact_position = (
        grasp_contact_position
        + float(harvest_distance_m) * _unit(frame.pull_axis_robot)
        - float(harvest_down_m) * WORLD_UP_ROBOT
    )
    surface_target = np.asarray(grasp.surface_target_robot_m, dtype=float).reshape(3)
    grasp_anchor = np.asarray(grasp.grasp_anchor_robot_m, dtype=float).reshape(3)
    backoff_axis = _unit(frame.backoff_axis_robot)
    pull_axis = _unit(frame.pull_axis_robot)
    carry_orientation_weight = (
        float(full_orientation_weight)
        if tail_orientation_weight is None
        else float(tail_orientation_weight)
    )
    carry_orientation_required = bool(tail_orientation_required)
    carry_orientation_mode = "full_pose" if carry_orientation_required else "carry_soft_pose"
    tail = [
        _tail_waypoint(
            name="harvest",
            backoff_reference="frozen_prefix_grasp",
            orientation_mode=carry_orientation_mode,
            orientation_weight=carry_orientation_weight,
            orientation_required=carry_orientation_required,
            reference_pose=grasp_pose,
            contact_position=harvest_contact_position,
            contact_offset_tcp_m=contact_offset,
            surface_target_robot_m=surface_target,
            grasp_anchor_robot_m=grasp_anchor,
            backoff_axis_robot=backoff_axis,
        ),
    ]
    if retract_mode == "prefix_approach":
        tail.append(
            _tail_waypoint(
                name="retract",
                backoff_reference="frozen_prefix_approach",
                orientation_mode=carry_orientation_mode,
                orientation_weight=carry_orientation_weight,
                orientation_required=carry_orientation_required,
                reference_pose=approach_pose,
                contact_position=approach_contact_position,
                contact_offset_tcp_m=contact_offset,
                surface_target_robot_m=surface_target,
                grasp_anchor_robot_m=grasp_anchor,
                backoff_axis_robot=backoff_axis,
            )
        )
    elif retract_mode == "backoff":
        tail.append(
            _tail_waypoint(
                name="retract",
                backoff_reference="harvest_backoff_up",
                orientation_mode=carry_orientation_mode,
                orientation_weight=carry_orientation_weight,
                orientation_required=carry_orientation_required,
                reference_pose=grasp_pose,
                contact_position=harvest_contact_position
                + float(retract_distance_m) * backoff_axis
                + float(retract_up_m) * WORLD_UP_ROBOT,
                contact_offset_tcp_m=contact_offset,
                surface_target_robot_m=surface_target,
                grasp_anchor_robot_m=grasp_anchor,
                backoff_axis_robot=backoff_axis,
            )
        )
    elif retract_mode == "lift_then_backoff":
        tail.append(
            _tail_waypoint(
                name="retract_lift",
                backoff_reference="harvest_lift",
                orientation_mode=carry_orientation_mode,
                orientation_weight=carry_orientation_weight,
                orientation_required=carry_orientation_required,
                reference_pose=grasp_pose,
                contact_position=harvest_contact_position + float(retract_up_m) * WORLD_UP_ROBOT,
                contact_offset_tcp_m=contact_offset,
                surface_target_robot_m=surface_target,
                grasp_anchor_robot_m=grasp_anchor,
                backoff_axis_robot=backoff_axis,
            )
        )
        tail.append(
            _tail_waypoint(
                name="retract",
                backoff_reference="harvest_lift_then_backoff",
                orientation_mode=carry_orientation_mode,
                orientation_weight=carry_orientation_weight,
                orientation_required=carry_orientation_required,
                reference_pose=grasp_pose,
                contact_position=harvest_contact_position
                + float(retract_distance_m) * backoff_axis
                + float(retract_up_m) * WORLD_UP_ROBOT,
                contact_offset_tcp_m=contact_offset,
                surface_target_robot_m=surface_target,
                grasp_anchor_robot_m=grasp_anchor,
                backoff_axis_robot=backoff_axis,
            )
        )
    elif retract_mode == "pull_reverse":
        tail.append(
            _tail_waypoint(
                name="retract",
                backoff_reference="harvest_reverse_pull_up",
                orientation_mode=carry_orientation_mode,
                orientation_weight=carry_orientation_weight,
                orientation_required=carry_orientation_required,
                reference_pose=grasp_pose,
                contact_position=harvest_contact_position
                - float(retract_distance_m) * pull_axis
                + float(retract_up_m) * WORLD_UP_ROBOT,
                contact_offset_tcp_m=contact_offset,
                surface_target_robot_m=surface_target,
                grasp_anchor_robot_m=grasp_anchor,
                backoff_axis_robot=backoff_axis,
            )
        )
    else:
        raise ValueError(f"unsupported retract_mode: {retract_mode}")
    if basket_pose is not None:
        basket_pose_array = np.asarray(basket_pose, dtype=float).reshape(4, 4)
        basket_contact_position = basket_pose_array[:3, 3] + basket_pose_array[:3, :3] @ contact_offset
        tail.append(
            _tail_waypoint(
                name="basket_place",
                backoff_reference="explicit_basket_pose",
                orientation_mode="basket_release",
                orientation_weight=basket_orientation_weight,
                orientation_required=basket_orientation_required,
                reference_pose=basket_pose_array,
                contact_position=basket_contact_position,
                contact_offset_tcp_m=contact_offset,
                surface_target_robot_m=surface_target,
                grasp_anchor_robot_m=grasp_anchor,
                backoff_axis_robot=backoff_axis,
            )
        )
    return tail


def _tail_waypoint(
    *,
    name: str,
    backoff_reference: str,
    orientation_mode: str,
    orientation_weight: float,
    orientation_required: bool,
    reference_pose: Array,
    contact_position: Array,
    contact_offset_tcp_m: Array,
    surface_target_robot_m: Array,
    grasp_anchor_robot_m: Array,
    backoff_axis_robot: Array,
) -> HarvestWaypointTarget:
    pose = np.asarray(reference_pose, dtype=float).reshape(4, 4).copy()
    contact_offset = _contact_offset(contact_offset_tcp_m)
    contact = np.asarray(contact_position, dtype=float).reshape(3)
    surface_target = np.asarray(surface_target_robot_m, dtype=float).reshape(3)
    grasp_anchor = np.asarray(grasp_anchor_robot_m, dtype=float).reshape(3)
    backoff_axis = _unit(backoff_axis_robot)
    pose[:3, 3] = contact - pose[:3, :3] @ contact_offset
    return HarvestWaypointTarget(
        name=name,
        backoff_m=0.0,
        backoff_reference=backoff_reference,
        orientation_mode=orientation_mode,
        orientation_weight=float(orientation_weight),
        orientation_required=bool(orientation_required),
        target_pose=pose,
        target_contact_position_robot_m=contact,
        contact_offset_tcp_m=contact_offset,
        offset_robot_m=contact - surface_target,
        surface_target_robot_m=surface_target,
        grasp_anchor_robot_m=grasp_anchor,
        surface_backoff_component_m=float(np.dot(contact - surface_target, backoff_axis)),
        grasp_anchor_backoff_component_m=float(np.dot(contact - grasp_anchor, backoff_axis)),
    )


def _required_waypoint(candidate: HarvestPlanCandidate, name: str) -> HarvestWaypointTarget:
    for waypoint in candidate.waypoints:
        if waypoint.name == name:
            return waypoint
    raise ValueError(f"candidate is missing required waypoint {name!r}")


def _positive_values(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values if float(value) > 1e-9)


def _pregrasp_surface_clearances(values: Iterable[float]) -> tuple[float, ...]:
    clearances = tuple(sorted({float(value) for value in values}))
    if not clearances:
        raise ValueError("pregrasp_surface_clearances_m must contain at least one value")
    if any(value < 0.0 for value in clearances):
        raise ValueError("pregrasp surface clearances must be non-negative")
    return clearances


def _pregrasp_backoff_options(
    insertion: GraspInsertionCandidate,
    *,
    lower_bound_m: float,
    surface_clearances_m: Iterable[float],
) -> tuple[float, ...]:
    if lower_bound_m <= 0.0:
        raise ValueError("pregrasp_backoff_m must be positive")
    options = {
        max(float(lower_bound_m), float(insertion.insertion_m) + float(clearance_m))
        for clearance_m in surface_clearances_m
    }
    return tuple(sorted(options))


def _contact_offset(value: Array | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    return np.asarray(value, dtype=float).reshape(3)


def _offset_vectors(values: Iterable[Array]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value, dtype=float).reshape(3) for value in values)


def _insertion_diagnostics(insertion: GraspInsertionCandidate) -> dict[str, float | bool | None]:
    return {
        "grasp_insertion_factor": float(insertion.factor),
        "palm_correction_factor": float(insertion.palm_correction_factor),
        "grasp_insertion_m": float(insertion.insertion_m),
        "nominal_grasp_insertion_m": float(insertion.nominal_insertion_m),
        "nominal_grasp_insertion_source": insertion.nominal_source,
        "center_projection_insertion_m": float(insertion.center_projection_insertion_m),
        "half_depth_cap_m": None if insertion.half_depth_cap_m is None else float(insertion.half_depth_cap_m),
        "insertion_cap_m": None if insertion.insertion_cap_m is None else float(insertion.insertion_cap_m),
        "insertion_clipped": bool(insertion.insertion_clipped),
        "center_palm_projection_m": float(insertion.center_palm_projection_m),
        "palm_axis_correction_m": float(insertion.palm_axis_correction_m),
        "palm_axis_correction_cap_m": None
        if insertion.palm_axis_correction_cap_m is None
        else float(insertion.palm_axis_correction_cap_m),
        "palm_axis_correction_clipped": bool(insertion.palm_axis_correction_clipped),
        "grasp_centering_error_m": abs(float(insertion.nominal_insertion_m) - float(insertion.insertion_m)),
    }


def _offset_variants(offset: Array) -> tuple[tuple[str, tuple[np.ndarray, np.ndarray], str], ...]:
    value = np.asarray(offset, dtype=float).reshape(3)
    zero = np.zeros(3, dtype=float)
    return (
        ("pregrasp_only", (zero, value), "pregrasp_only"),
        ("approach_only", (value, zero), "approach_only"),
        ("both", (value, value), "both"),
    )


def _unit(vector: Array) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("axis must be non-zero")
    return value / norm


def _rot_z(angle_rad: float) -> np.ndarray:
    cos_angle = float(np.cos(angle_rad))
    sin_angle = float(np.sin(angle_rad))
    return np.asarray(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _name_with_roll(base_name: str, tcp_roll_rad: float) -> str:
    if abs(float(tcp_roll_rad)) <= 1e-9:
        return base_name
    return f"{base_name}_tcp_roll_{_roll_label(tcp_roll_rad)}"


def _name_with_depth_and_roll(
    base_name: str,
    insertion: GraspInsertionCandidate,
    tcp_roll_rad: float,
) -> str:
    with_depth = (
        f"{base_name}_insert_f{_factor_label(insertion.factor)}_{_mm_label(insertion.insertion_m)}"
        f"_palm_f{_factor_label(insertion.palm_correction_factor)}"
        f"_{_mm_label(abs(insertion.palm_axis_correction_m))}"
    )
    return _name_with_roll(with_depth, tcp_roll_rad)


def _name_with_depth_pregrasp_and_roll(
    base_name: str,
    insertion: GraspInsertionCandidate,
    pregrasp_backoff_m: float,
    tcp_roll_rad: float,
) -> str:
    with_pregrasp = f"{base_name}_pregrasp_{_cm_label(pregrasp_backoff_m)}"
    return _name_with_depth_and_roll(with_pregrasp, insertion, tcp_roll_rad)


def _factor_label(value: float) -> str:
    return f"{int(round(float(value) * 100.0)):03d}"


def _mm_label(distance_m: float) -> str:
    return f"{int(round(float(distance_m) * 1000.0)):03d}mm"


def _roll_label(tcp_roll_rad: float) -> str:
    degrees = int(round(np.rad2deg(float(tcp_roll_rad)))) % 360
    return f"{degrees:03d}deg"


def _cm_label(distance_m: float) -> str:
    return f"{int(round(float(distance_m) * 100.0)):02d}cm"


def _signed_cm_label(distance_m: float) -> str:
    sign = "plus" if float(distance_m) >= 0.0 else "minus"
    return f"{sign}{_cm_label(abs(float(distance_m)))}"


def _tail_label(retract_mode: str, retract_distance_m: float, retract_up_m: float) -> str:
    return f"{retract_mode}_d{_cm_label(retract_distance_m)}_u{_cm_label(retract_up_m)}"
