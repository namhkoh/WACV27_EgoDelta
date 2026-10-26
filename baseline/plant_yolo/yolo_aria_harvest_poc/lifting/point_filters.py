"""Point filtering helpers used by generic mask lifting."""

from __future__ import annotations

import numpy as np


def front_surface_filter(
    points_world: np.ndarray,
    depths: np.ndarray,
    *,
    depth_band_m: float,
    front_quantile: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points_world) == 0:
        return points_world.reshape(0, 3), np.zeros((0,), dtype=bool)
    z_front = float(np.quantile(depths, front_quantile))
    keep = np.asarray(depths <= z_front + depth_band_m, dtype=bool)
    return np.asarray(points_world, dtype=float).reshape(-1, 3)[keep], keep
