"""EGODELTA primitive-level scene-difference and target-recovery tools."""

from __future__ import annotations

from .data import Camera, ScenePair, discover_ready_pairs, load_2dgs_ply, load_cameras
from .primitives import GaussianSet2D
from .contact_mass_extraction import (
    ContactMassComponent,
    ContactMassConfig,
    ContactMassGroup,
    ContactMassResult,
    extract_contact_mass_targets,
)

__all__ = [
    "Camera",
    "ContactMassComponent",
    "ContactMassConfig",
    "ContactMassGroup",
    "ContactMassResult",
    "GaussianSet2D",
    "EgoDeltaDiffPipeline",
    "ScenePair",
    "discover_ready_pairs",
    "extract_contact_mass_targets",
    "load_2dgs_ply",
    "load_cameras",
]


def __getattr__(name: str):
    if name == "EgoDeltaDiffPipeline":
        from .pipeline import EgoDeltaDiffPipeline

        globals()["EgoDeltaDiffPipeline"] = EgoDeltaDiffPipeline
        return EgoDeltaDiffPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
