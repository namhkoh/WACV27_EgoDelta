"""Mechanics-conditioned geometry-only Stage-3 target extraction."""

from .contact_mass_extraction import (
    ExtractionConfig,
    ExtractionResult,
    Target,
    extract_targets,
)
from .data import find_scene_targets, parse_gt_markdown
from .evaluation import aggregate_evaluations, evaluate_points
from .mechanics import EndpointSide, extract_for_endpoint_side
from .removed_surface_extraction import (
    RemovedSurfaceConfig,
    RemovedSurfaceResult,
    RemovedSurfaceTarget,
    extract_removed_surface_targets,
)
from .stage2_cache import Stage2Cache, cache_digest, load_stage2_cache

__all__ = [
    "ExtractionConfig",
    "ExtractionResult",
    "EndpointSide",
    "RemovedSurfaceConfig",
    "RemovedSurfaceResult",
    "RemovedSurfaceTarget",
    "Stage2Cache",
    "Target",
    "aggregate_evaluations",
    "cache_digest",
    "evaluate_points",
    "extract_targets",
    "extract_for_endpoint_side",
    "extract_removed_surface_targets",
    "find_scene_targets",
    "load_stage2_cache",
    "parse_gt_markdown",
]
