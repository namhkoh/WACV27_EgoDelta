from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from robot.kinematics.apps import deleaf_click_cut
from robot.kinematics.apps import run_any_target_deleaf as runner
from robot.utils.restore_home_state import RIGHT_FROM_LEFT_MIRROR_SIGNS, mirror_left_arm_to_right


def test_deleaf_command_preserves_proven_execution_contract(tmp_path: Path) -> None:
    args = argparse.Namespace(
        deleaf_config=tmp_path / "config.json",
        dataset_name="a270c038",
        home=tmp_path / "home.json",
        manifest=tmp_path / "manifest.json",
        alignment=tmp_path / "alignment.json",
        hand_tracking=tmp_path / "hands.csv",
        slam_trajectory=tmp_path / "trajectory.csv",
        toolfit=tmp_path / "toolfit.json",
        run_label="deleaf_01",
        runs_root=tmp_path / "runs",
        post_selection_home_minimum_time_s=10.0,
    )

    command = [str(value) for value in runner.deleaf_command(
        args,
        output_json=tmp_path / "execution.json",
        video_path=tmp_path / "trial.mp4",
    )]

    assert "--selected-point-robot-m" not in command
    assert command[command.index("--post-selection-home") + 1] == str(args.home)
    assert command[command.index("--post-selection-home-minimum-time-s") + 1] == "10.0"
    assert "--mps-auto-select-action" in command
    assert "--require-selected-point-cut-anchor" in command
    assert "--waypoint-policy" not in command
    assert "--teleop-prior-json" not in command
    assert "--target-extension-values-m" not in command
    assert "--approach-stroke-standoff-m" not in command
    assert "--pre-cut-standoff-m" not in command
    assert command[command.index("--pre-cut-time-s") + 1] == "0.5"
    assert command[command.index("--cut-time-s") + 1] == "0.04"
    assert command[command.index("--cut-through-m") + 1] == "0.015"
    assert command[command.index("--right-velocity-limit-rad-s") + 1] == "0.4"
    assert command[command.index("--right-acceleration-limit-rad-s2") + 1] == "0.2"
    assert command[command.index("--cut-velocity-limit-rad-s") + 1] == "4.0"
    assert command[command.index("--cut-acceleration-limit-rad-s2") + 1] == "18.0"
    assert command[command.index("--execute-until") + 1] == "return_transit"
    assert "--trial-video-output" in command
    assert "--no-execute-key-confirm" in command
    assert "--execute" in command

    config = runner.DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert '"approach_azimuth_sweep_deg": "0,-30,-60,-90,30,60,90"' in config
    assert '"adaptive_approach_azimuth_top_k": 3' in config
    assert '"pointcloud_contact_advance_values_m": "0.07"' in config
    assert '"pointcloud_candidate_angles_deg": "-35,0,35"' in config
    assert '"pointcloud_blade_min_clearance_m": 0.0' in config
    assert '"target_extension_m": 0.0' in config
    assert '"target_extension_values_m": "0"' in config
    assert '"teleop_prior_library_dir": "robot/kinematics/runs/060326/deleaf_teleop_prior"' in config
    assert '"successful_demo_library": "robot/policy/policy_action_library"' in config
    assert '"successful_demo_seed_count": 5' in config
    assert '"cut_through_m": 0.015' in config
    assert '"pre_cut_time_s": 0.5' in config
    assert '"cut_time_s": 0.04' in config
    assert '"cut_velocity_limit_rad_s": 4.0' in config
    assert '"cut_acceleration_limit_rad_s2": 18.0' in config


def test_selection_posture_mirrors_left_arm_then_restores_right_only(tmp_path: Path) -> None:
    args = argparse.Namespace(
        home=tmp_path / "home.json",
        home_minimum_time_s=15.0,
    )
    command = [str(value) for value in runner.selection_posture_command(args)]
    assert command[command.index("--scope") + 1] == "all"
    assert "--mirror-left-to-right" in command
    assert command[command.index("--minimum-time") + 1] == "15.0"

    post_args = argparse.Namespace(
        post_selection_home=args.home,
        post_selection_home_minimum_time_s=10.0,
    )
    post_command = deleaf_click_cut.post_selection_home_command(post_args)
    assert post_command is not None
    assert post_command[post_command.index("--scope") + 1] == "right_arm"
    assert "--mirror-left-to-right" not in post_command
    assert post_command[post_command.index("--minimum-time") + 1] == "10.0"


def test_rb_y1_left_to_right_mirror_joint_convention() -> None:
    left = [0.2, 0.3, -0.4, -1.2, 0.5, 0.8, -0.6]
    home = {"arms": {"left": {"positions_rad": left}}}
    mirrored = mirror_left_arm_to_right(home)
    np.testing.assert_allclose(
        mirrored,
        np.asarray([0.2, -0.3, 0.4, -1.2, -0.5, 0.8, 0.6]),
        atol=0.0,
    )
    np.testing.assert_array_equal(
        RIGHT_FROM_LEFT_MIRROR_SIGNS,
        np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
    )


def test_fast_iitp_limits_apply_only_to_cut_stroke() -> None:
    args = argparse.Namespace(
        right_velocity_limit_rad_s=0.4,
        right_acceleration_limit_rad_s2=0.2,
        cut_velocity_limit_rad_s=4.0,
        cut_acceleration_limit_rad_s2=18.0,
    )
    time_by_name = {"approach": 6.0, "pre_cut": 0.5, "cut": 0.04, "retract": 6.0}

    for name in ("approach", "pre_cut", "retract"):
        profile = deleaf_click_cut.deleaf_phase_motion_profile(
            name,
            args,
            time_by_name=time_by_name,
            path_hop_time_s=0.8,
        )
        assert profile["velocity_limit_rad_s"] == 0.4
        assert profile["acceleration_limit_rad_s2"] == 0.2

    cut = deleaf_click_cut.deleaf_phase_motion_profile(
        "cut",
        args,
        time_by_name=time_by_name,
        path_hop_time_s=0.8,
    )
    assert cut == {
        "minimum_time_s": 0.04,
        "velocity_limit_rad_s": 4.0,
        "acceleration_limit_rad_s2": 18.0,
    }


def test_teleop_offset_azimuth_sweep_rotates_orientation_without_moving_waypoints() -> None:
    identity = np.eye(3).tolist()
    prior = {
        "source": "unit_test",
        "indices": {},
        "cut_anchor_robot_m": [0.5, 0.0, 1.0],
        "waypoints": {
            "approach": {
                "offset_from_cut_anchor_m": [0.0, 0.0, 0.10],
                "rotation_robot_from_tcp": identity,
            },
            "pre_cut": {
                "offset_from_cut_anchor_m": [0.0, 0.0, 0.05],
                "rotation_robot_from_tcp": identity,
            },
            "cut": {
                "offset_from_cut_anchor_m": [0.0, 0.0, 0.0],
                "rotation_robot_from_tcp": identity,
            },
            "retract": {
                "offset_from_cut_anchor_m": [-0.10, 0.0, 0.10],
                "rotation_robot_from_tcp": identity,
            },
        },
    }
    tool_axes = {
        "mode": "unit_test",
        "source": "unit_test",
        "primary_axis_local": np.asarray([1.0, 0.0, 0.0]),
        "secondary_axis_local": np.asarray([0.0, 1.0, 0.0]),
    }

    def build(yaw_deg: float):
        args = argparse.Namespace(
            teleop_follow=False,
            teleop_follow_cut=False,
            orientation_yaw_deg=yaw_deg,
            contact_advance_m=0.0,
            contact_advance_axis_robot=None,
            contact_advance_mode="target",
            pre_cut_standoff_m=None,
            target_extension_m=0.0,
            approach_stroke_standoff_m=None,
            max_z_m=1.35,
            safe_entry_axis_robot=None,
            safe_entry_standoff_m=0.0,
            safe_entry_lift_m=0.04,
            transit_lift_m=0.0,
            cut_through_m=0.0,
            post_cut_lift_m=0.0,
            return_transit_lift_m=0.0,
            use_teleop_joint_seeds=False,
        )
        return deleaf_click_cut.build_teleop_prior_waypoints(
            cut_point_robot_m=np.asarray([0.5, 0.0, 1.0]),
            home_tcp_robot_m=np.asarray([0.3, 0.0, 1.0]),
            prior=prior,
            tool_axes=tool_axes,
            args=args,
        )

    base_waypoints, base_diagnostics = build(0.0)
    yaw_waypoints, yaw_diagnostics = build(90.0)
    expected_yaw = deleaf_click_cut.yaw_about_world_z(90.0)

    assert [waypoint.name for waypoint in yaw_waypoints] == [waypoint.name for waypoint in base_waypoints]
    for base, yawed in zip(base_waypoints, yaw_waypoints, strict=True):
        np.testing.assert_allclose(yawed.target_pose[:3, 3], base.target_pose[:3, 3], atol=1e-12)
        np.testing.assert_allclose(yawed.target_pose[:3, :3], expected_yaw @ base.target_pose[:3, :3], atol=1e-12)
    assert base_diagnostics["orientation_yaw_deg"] == 0.0
    assert yaw_diagnostics["orientation_yaw_deg"] == 90.0
    assert yaw_diagnostics["task_orientation_source"].endswith("+yaw_+90deg")


def test_high_cut_shortens_only_extra_approach_standoff() -> None:
    pre_cut = np.asarray([0.65, 0.16, 1.317168017080177])
    stroke = np.asarray([-0.13865868991486072, 0.011954904513841786, -0.9902680687415704])

    effective, diagnostic = deleaf_click_cut.limit_approach_standoff_to_workspace(
        pre_cut_position_robot_m=pre_cut,
        stroke_axis_robot=stroke,
        requested_standoff_m=0.04,
        max_z_m=1.35,
    )

    assert diagnostic["applied"] is True
    assert diagnostic["reason"] == "workspace_ceiling"
    assert 0.027 < effective < 0.029
    np.testing.assert_allclose(pre_cut - stroke * effective, [0.6539, 0.1597, 1.345], atol=5e-4)
    assert diagnostic["effective_approach_z_m"] <= 1.345 + 1e-9

    unchanged, unchanged_diagnostic = deleaf_click_cut.limit_approach_standoff_to_workspace(
        pre_cut_position_robot_m=np.asarray([0.65, 0.16, 1.20]),
        stroke_axis_robot=stroke,
        requested_standoff_m=0.04,
        max_z_m=1.35,
    )
    assert unchanged == 0.04
    assert unchanged_diagnostic["applied"] is False
    assert unchanged_diagnostic["reason"] == "within_workspace"


def test_selected_target_anchor_contract_fails_closed_on_extension() -> None:
    good = argparse.Namespace(
        target_extension_m=0.0,
        target_extension_values_m="0",
        require_selected_point_cut_anchor=True,
    )
    diagnostic = deleaf_click_cut.selected_target_anchor_contract(good)
    assert diagnostic["ok"] is True
    assert diagnostic["maximum_abs_extension_m"] == 0.0

    bad = argparse.Namespace(
        target_extension_m=0.07,
        target_extension_values_m="0,0.07",
        require_selected_point_cut_anchor=True,
    )
    try:
        deleaf_click_cut.selected_target_anchor_contract(bad)
    except ValueError as exc:
        assert "cut-anchor contract violated" in str(exc)
    else:
        raise AssertionError("nonzero selected-target extension must fail closed")


def test_only_blade_engagement_phases_require_full_orientation() -> None:
    phase_names = (
        "transit_lift",
        "safe_entry",
        "approach",
        "pre_cut",
        "cut",
        "post_cut_lift",
        "retract_lift",
        "retract",
        "return_transit",
    )
    required = [
        name for name in phase_names if deleaf_click_cut.deleaf_phase_requires_orientation(name)
    ]
    assert required == ["pre_cut", "cut"]
    assert "orientation_required=deleaf_phase_requires_orientation(name)" in inspect.getsource(
        deleaf_click_cut.build_mps_template_waypoints
    )
    assert "orientation_required=deleaf_phase_requires_orientation(name)" in inspect.getsource(
        deleaf_click_cut.build_teleop_prior_waypoints
    )


def test_successful_demo_library_is_ranked_seed_only(tmp_path: Path) -> None:
    def write_demo(label: str, point: list[float], seed_value: float) -> None:
        path = (
            tmp_path
            / label
            / "success_current"
            / "actions"
            / f"{label}_action"
            / "action"
            / "referenced"
            / "deleaf_click_cut"
            / "plan.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "dataset_name": "a270c038",
                    "hard_gate": {"ok": True},
                    "motion": {"executed": True},
                    "selected_point": {"point_robot_m": point},
                    "waypoints": [
                        {
                            "name": "approach",
                            "ok": True,
                            "q_arm_rad": [seed_value] * 7,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    write_demo("far", [0.2, 0.0, 1.0], 1.0)
    write_demo("near", [0.6, 0.2, 1.2], 2.0)
    library = deleaf_click_cut.load_successful_demo_joint_seed_library(
        tmp_path,
        cut_point_robot_m=np.asarray([0.61, 0.19, 1.2]),
        dataset_name="a270c038",
        max_count=2,
    )

    assert library["enabled"] is True
    assert library["uses_joint_replay"] is False
    assert library["role"] == "target_conditioned_ik_multistart_seed_only"
    assert [candidate["label"] for candidate in library["candidates"]] == ["near", "far"]
    assert library["candidates"][0]["q_seed_by_name"]["approach"] == [2.0] * 7


def test_teleop_prior_library_selects_nearest_target_without_replay(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(
        json.dumps(
            {
                "priors": [
                    {
                        "source": "base_demo.json",
                        "cut_anchor_robot_m": [0.4, 0.0, 1.0],
                        "waypoints": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    (library_dir / "a270c038_AD1_teleop_prior.json").write_text(
        json.dumps(
            {
                "source": "successful_ad1_demo.json",
                "cut_anchor_robot_m": [0.7, 0.2, 1.2],
                "waypoints": {},
            }
        ),
        encoding="utf-8",
    )
    selected = deleaf_click_cut.load_teleop_prior(
        base_path,
        "deleafing_01",
        None,
        library_dir=library_dir,
        cut_point_robot_m=np.asarray([0.69, 0.19, 1.2]),
    )

    assert selected is not None
    assert selected["source"] == "successful_ad1_demo.json"
    assert selected["selection"]["policy"] == "nearest_target_conditioned_successful_prior"
    assert selected["selection"]["candidate_count"] == 2
