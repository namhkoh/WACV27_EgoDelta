from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot.kinematics.rby1_model import RBY1Kinematics
from robot.kinematics.types import Array


@dataclass(frozen=True)
class ClearancePair:
    link1: str
    link2: str
    distance_m: float
    position1_robot_m: tuple[float, float, float]
    position2_robot_m: tuple[float, float, float]


@dataclass(frozen=True)
class ClearanceReport:
    nearest_pairs: tuple[ClearancePair, ...]
    minimum_distance_m: float | None
    margin_m: float

    @property
    def ok(self) -> bool:
        return self.minimum_distance_m is None or self.minimum_distance_m >= self.margin_m


def nearest_self_clearances(
    kinematics: RBY1Kinematics,
    q_full_rad: Array,
    *,
    pair_count: int = 10,
    margin_m: float = 0.03,
) -> ClearanceReport:
    links = tuple(str(name) for name in kinematics.dyn_robot.get_link_names())
    state = kinematics.make_state(q_full_rad, links=links)
    raw_pairs = kinematics.dyn_robot.detect_collisions_or_nearest_links(state, int(pair_count))
    pairs = tuple(
        ClearancePair(
            link1=str(item.link1),
            link2=str(item.link2),
            distance_m=float(item.distance),
            position1_robot_m=tuple(float(x) for x in np.asarray(item.position1, dtype=float).reshape(3)),
            position2_robot_m=tuple(float(x) for x in np.asarray(item.position2, dtype=float).reshape(3)),
        )
        for item in raw_pairs
    )
    minimum = min((pair.distance_m for pair in pairs), default=None)
    return ClearanceReport(
        nearest_pairs=pairs,
        minimum_distance_m=minimum,
        margin_m=float(margin_m),
    )


def clearance_residuals(
    kinematics: RBY1Kinematics,
    q_full_rad: Array,
    *,
    margin_m: float,
    pair_count: int,
) -> Array:
    report = nearest_self_clearances(
        kinematics,
        q_full_rad,
        pair_count=pair_count,
        margin_m=margin_m,
    )
    return np.asarray(
        [max(0.0, float(margin_m) - pair.distance_m) for pair in report.nearest_pairs],
        dtype=float,
    )
