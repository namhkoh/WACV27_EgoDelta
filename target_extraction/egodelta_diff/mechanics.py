"""A-priori mechanics-conditioned routing for Stage-3 endpoint decoding."""

from __future__ import annotations

from enum import Enum

from .contact_mass_extraction import ExtractionConfig, extract_targets
from .removed_surface_extraction import (
    RemovedSurfaceConfig,
    extract_removed_surface_targets,
)


class EndpointSide(str, Enum):
    """Side of the intervention interface acted on by the tool."""

    RETAINED = "retained"
    REMOVED = "removed"


def extract_for_endpoint_side(
    endpoint_side: EndpointSide,
    *,
    mu,
    normals,
    delta,
    omega,
    overlap,
    source_indices=None,
    retained_config: ExtractionConfig | None = None,
    removed_config: RemovedSurfaceConfig | None = None,
):
    """Decode the interface endpoint selected by known tool mechanics.

    Routing is explicit and a priori: sever/cut primitives request the retained
    endpoint; detach/grasp primitives request the removed endpoint.  Neither
    decoder receives or infers a crop/task label, and proximity is not used to
    select the decoder.
    """

    side = EndpointSide(endpoint_side)
    if side is EndpointSide.RETAINED:
        return extract_targets(
            mu,
            normals,
            delta,
            omega,
            overlap,
            source_indices,
            retained_config,
        )
    return extract_removed_surface_targets(
        mu,
        delta,
        overlap,
        source_indices,
        removed_config,
    )
