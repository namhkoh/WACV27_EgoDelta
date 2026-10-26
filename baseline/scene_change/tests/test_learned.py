from inspect import signature

from egoscene.learned import render_mv3dcd_queries


def test_mv3dcd_uses_reference_iteration_for_alpha_support():
    default = signature(render_mv3dcd_queries).parameters["alpha_iteration"].default
    assert default == 7_000
