from __future__ import annotations

import cv2
import json
import numpy as np
import pytest

from robot.kinematics.harvest_target_selection import (
    refine_red_fruit_center_surface,
    validate_target_artifact_goal,
)


def _touching_fruit_capture() -> dict[str, np.ndarray]:
    height, width = 120, 160
    color = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(color, (55, 60), 22, (0, 0, 255), thickness=-1)
    cv2.circle(color, (91, 68), 23, (0, 0, 255), thickness=-1)
    y, x = np.indices((height, width))
    uv = np.column_stack((x.ravel(), y.ravel())).astype(float)
    depth = np.full((height, width), 0.6, dtype=float)
    points = np.column_stack((x.ravel() * 0.001, y.ravel() * 0.001, depth.ravel()))
    return {"color_bgr": color, "uv": uv, "points": points}


def test_click_selects_and_centers_one_of_touching_fruit() -> None:
    result = refine_red_fruit_center_surface(_touching_fruit_capture(), [92, 86])

    assert result["point_semantics"] == "selected_fruit_center_visible_surface"
    assert result["detected_instance_peak_count"] == 2
    assert np.linalg.norm(np.asarray(result["pixel_xy"]) - np.asarray([91, 68])) < 4.0
    assert result["raw_to_center_pixel_distance_px"] > 10.0
    assert result["depth_support_points"] >= 20


def test_selected_instance_reports_metric_radius_from_rgbd() -> None:
    result = refine_red_fruit_center_surface(
        _touching_fruit_capture(),
        [92, 86],
        intrinsics={"fx": 600.0, "fy": 600.0},
    )

    expected = result["selected_peak_radius_px"] * 0.6 / 600.0
    assert result["estimated_fruit_radius_m"] == pytest.approx(expected)
    assert result["estimated_fruit_diameter_m"] == pytest.approx(2.0 * expected)
    assert result["metric_radius_method"] == (
        "distance_transform_inscribed_radius_times_depth_over_focal"
    )


def test_click_far_from_fruit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="nearest red fruit"):
        refine_red_fruit_center_surface(_touching_fruit_capture(), [150, 5])


def test_planner_goal_must_match_centered_target_artifact(tmp_path) -> None:
    artifact = tmp_path / "target.json"
    artifact.write_text(
        json.dumps(
            {
                "selected_point": {
                    "point_robot_m": [0.56, 0.04, 0.86],
                    "pixel_xy": [294, 428],
                    "raw_click_pixel_xy": [290, 438],
                    "point_semantics": "selected_fruit_center_visible_surface",
                    "selection_method": "manual_click_red_instance_center",
                    "estimated_fruit_radius_m": 0.033,
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = validate_target_artifact_goal(
        artifact,
        np.asarray([0.5605, 0.04, 0.86]),
        tolerance_m=0.002,
    )
    assert metadata["artifact_goal_checked"]
    assert metadata["point_semantics"] == "selected_fruit_center_visible_surface"
    assert metadata["estimated_fruit_radius_m"] == pytest.approx(0.033)

    with pytest.raises(RuntimeError, match="disagrees"):
        validate_target_artifact_goal(
            artifact,
            np.asarray([0.55, 0.04, 0.86]),
            tolerance_m=0.002,
        )
