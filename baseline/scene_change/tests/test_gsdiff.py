import torch

from egoscene.gsdiff import (
    GaussianSet,
    aggregate_scores,
    directional_color_bandwidth,
    nearest_neighbor_drift,
    observability_confidence,
    primitive_scores,
    quaternion_to_matrix,
    symmetric_drift,
)


def make_gaussians(points: torch.Tensor) -> GaussianSet:
    count = len(points)
    quaternion = torch.zeros((count, 4))
    quaternion[:, 0] = 1
    rotation = quaternion_to_matrix(quaternion)
    return GaussianSet(
        means=points.float(),
        quaternions=quaternion,
        rotations=rotation,
        scales=torch.ones((count, 3)) * 0.01,
        normals=torch.tensor([[0.0, 0.0, 1.0]]).repeat(count, 1),
        opacity=torch.ones(count) * 0.9,
        dc_color=torch.zeros((count, 3)),
        source_indices=torch.arange(count),
        is_surfel=False,
    )


def test_quaternion_identity():
    matrix = quaternion_to_matrix(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(matrix[0], torch.eye(3))


def test_symmetric_drift_is_zero_for_identical_sets():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    normal2, tangent2 = symmetric_drift(make_gaussians(points), make_gaussians(points))
    assert normal2 == 0.0
    assert tangent2 == 0.0


def test_confidence_and_aggregate_are_bounded():
    fim = torch.stack((torch.eye(3), torch.eye(3) * 2, torch.eye(3) * 4, torch.eye(3) * 8))
    confidence = observability_confidence(fim)
    assert torch.all(confidence[1:] >= confidence[:-1])
    score = aggregate_scores(torch.ones(4), torch.ones(4), fim)
    assert torch.all((score >= 0) & (score <= 1))


def test_unobserved_primitive_has_near_zero_confidence():
    fim = torch.stack((torch.zeros((3, 3)), torch.eye(3), torch.eye(3) * 2, torch.eye(3) * 4))
    confidence = observability_confidence(fim)
    assert confidence[0] < 1.0e-6
    assert torch.all(confidence[1:] > confidence[0])


def test_normal_drift_uses_absolute_upper_quartile_before_squaring():
    points = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    shifted = points + torch.tensor([[0.0, 0.0, -0.4], [0.0, 0.0, -0.3], [0.0, 0.0, -0.2], [0.0, 0.0, 1.0]])
    normal2, tangent2 = nearest_neighbor_drift(make_gaussians(points), make_gaussians(shifted))
    assert abs(normal2 - 0.3025) < 1.0e-6
    assert tangent2 == 0.0


def test_color_bandwidth_is_median_of_weighted_squared_differences():
    points_a = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    # With identity covariance in both scenes, this offset gives a geometric
    # kernel weight of 0.01 for the middle pair.
    offset = (-4.0 * torch.log(torch.tensor(0.01))).sqrt().item()
    points_b = torch.tensor([[0.0, 0.0, 0.0], [100.0 + offset, 0.0, 0.0], [200.0, 0.0, 0.0]])
    a = make_gaussians(points_a)
    b = make_gaussians(points_b)
    b.dc_color[:, 0] = torch.tensor([0.0, 1.0, 2.0])
    covariance = torch.eye(3).repeat(3, 1, 1)
    bandwidth2 = directional_color_bandwidth(a, b, covariance, covariance)
    # Eq. 9: median([1*0^2, 0.01*1^2, 1*2^2]) = 0.01.
    assert abs(bandwidth2 - 0.01) < 1.0e-5


def test_primitive_kernels_separate_geometry_and_appearance():
    at_origin = make_gaussians(torch.tensor([[0.0, 0.0, 0.0]]))
    displaced = make_gaussians(torch.tensor([[1.0, 0.0, 0.0]]))
    covariance_origin = at_origin.covariance
    covariance_displaced = displaced.covariance
    geometry, appearance = primitive_scores(
        at_origin, displaced, covariance_origin, covariance_displaced, color_bandwidth2=0.01
    )
    assert geometry.item() == 1.0
    assert appearance.item() == 1.0

    recolored = make_gaussians(torch.tensor([[0.0, 0.0, 0.0]]))
    recolored.dc_color[0, 0] = 1.0
    geometry, appearance = primitive_scores(
        at_origin, recolored, covariance_origin, recolored.covariance, color_bandwidth2=0.01
    )
    assert geometry.item() < 1.0e-6
    assert appearance.item() > 0.99
