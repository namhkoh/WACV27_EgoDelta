from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
else:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.kinematics.run_outputs import run_directory


DATASET_SCHEMA_VERSION = 1
DEFAULT_DATASET_NAME = "a270c038"
DEFAULT_TESTBED_ID = "gutter_v1"
TASK_NAME = "policy_training_episodes"
POLICY_RUNS_ROOT = Path(__file__).resolve().parent / "runs"
POLICY_DATASET_ROOT = Path(__file__).resolve().parent / "policy_training_dataset"
POLICY_EPISODES_ROOT = Path(__file__).resolve().parent / "policy_training_episodes"
POLICY_ACTION_LIBRARY_ROOT = Path(__file__).resolve().parent / "policy_action_library"
HARVEST_SOLVE_LIBRARY_ROOT = Path(__file__).resolve().parent / "harvest_solve_library"

DEFAULT_DELEAF_TARGETS = ("AD1", "AD2", "AD3", "BD1", "BD2", "BD3")
DEFAULT_PHYSICAL_LEAF_COUNT = 4
HARVEST_TARGETS = ("AT1T1", "AT1T2", "AT1T3")
DEFAULT_PHYSICAL_TOMATO_COUNT = 1
DEFAULT_HARVEST_BASKET_ANCHOR_ROBOT_M = "-0.155109,0.907240,0.388685"
DEFAULT_HARVEST_TOOLFIT = Path(r"D:\tmp\egodelta_isaac_left_jaw_center_toolfit.json")
DEFAULT_HARVEST_CONTACT_FRAME = (
    Path(_REPO_ROOT) / "robot" / "calibration" / "gripper" / "left_gripper_contact_frame_visual_estimate_20260525.json"
)
DEFAULT_HARVEST_TARGET_PATTERN = "harvest_tomato*_task_frame_instance_split_jaw_minusx.json"
DEFAULT_HARVEST_PALM_CORRECTION_FACTORS = "0.0"
DEFAULT_HARVEST_GRASP_INSERTION_FACTORS = "0.25,0.5"
DEFAULT_HARVEST_MAX_GRASP_INSERTION_FACTOR = 0.5
DEFAULT_HARVEST_MIN_GRASP_INSERTION_FRACTION = 0.2
DEFAULT_HARVEST_GENERIC_DEPTH_M = 0.070
DEFAULT_HARVEST_GENERIC_WIDTH_M = 0.065
DEFAULT_HARVEST_LOCAL_RADIUS_PX = 70
DEFAULT_HARVEST_LOCAL_DEPTH_WINDOW_M = 0.09
DEFAULT_HARVEST_LOCAL_COLOR_THRESHOLD = 0.30
DEFAULT_HARVEST_LOCAL_RADIUS_M = 0.12
DEFAULT_HARVEST_MIN_LOCAL_POINTS = 40
DEFAULT_HARVEST_MAX_DERIVED_TARGET_SHIFT_M = 0.09
DEFAULT_HARVEST_CONTACT_PALM_PERCENTILE = 20.0
DEFAULT_HARVEST_MIN_PRECLOSE_Z_MARGIN_M = -0.003
DEFAULT_HARVEST_MIN_ACCEPTED_INSERTION_M = 0.020
DEFAULT_HARVEST_MAX_ACCEPTED_BACKOFF_ERROR_RAD = 0.15
DEFAULT_HARVEST_DISTANCE_M = 0.0
DEFAULT_HARVEST_DOWN_M = 0.0
DEFAULT_HARVEST_TARGET_BIAS_ROBOT_M = "0,0,0"
DEFAULT_HARVEST_FULL_ORIENTATION_WEIGHT = 0.5
DEFAULT_HARVEST_LATERAL_OFFSETS_M = "0.04,0.06,0.08,0.10,0.12,-0.04,-0.06,-0.08"
DEFAULT_HARVEST_ELEVATED_OFFSETS_M = "0.05,0.08"
DEFAULT_HARVEST_TCP_ROLLS_DEG = "0,180"
DEFAULT_HARVEST_PREGRASP_TASK_POSITION_TOLERANCE_M = 0.004
DEFAULT_HARVEST_PREGRASP_TASK_JAW_TOLERANCE_RAD = 0.20
DEFAULT_HARVEST_PREGRASP_TASK_BACKOFF_TOLERANCE_RAD = 0.25
DEFAULT_HARVEST_GRASP_TASK_POSITION_TOLERANCE_M = 0.004
DEFAULT_HARVEST_GRASP_TASK_JAW_TOLERANCE_RAD = 0.20
DEFAULT_HARVEST_GRASP_TASK_BACKOFF_TOLERANCE_RAD = 0.30
DEFAULT_HARVEST_PRIMITIVE_PRIOR_DIR = Path(_REPO_ROOT) / "robot" / "calibration" / "targets" / DEFAULT_DATASET_NAME
DEFAULT_TOMATO1_GRASP_PRIOR = (
    Path(_REPO_ROOT)
    / "robot"
    / "kinematics"
    / "runs"
    / "052626"
    / "teleop_tcp_capture"
    / "a270c038_tomato1_teleop_better_grasp_urdf_offset_v1.json"
)
HARVEST_IDENTITY_TASK_FRAMES = {
    "AT1T1": DEFAULT_HARVEST_PRIMITIVE_PRIOR_DIR / "harvest_tomato0_task_frame_instance_split_jaw_minusx.json",
    "AT1T2": DEFAULT_HARVEST_PRIMITIVE_PRIOR_DIR / "harvest_tomato1_task_frame_instance_split_jaw_minusx.json",
}
DEFAULT_HARVEST_FIXED_TAIL_PLAN = (
    HARVEST_SOLVE_LIBRARY_ROOT / "AT1T1" / "a270c038_AT1T1_success_current.json"
)
HARVEST_FIXED_FULL_TAIL_PLANS = {
    "AT1T1": DEFAULT_HARVEST_FIXED_TAIL_PLAN,
    "AT1T2": HARVEST_SOLVE_LIBRARY_ROOT / "AT1T2" / "a270c038_AT1T2_success_current.json",
    "AT1T3": HARVEST_SOLVE_LIBRARY_ROOT / "AT1T3" / "a270c038_AT1T3_success_current.json",
}
DELEAF_SIDE_PRIOR_JSON = (
    Path(_REPO_ROOT)
    / "robot"
    / "kinematics"
    / "runs"
    / "052826"
    / "deleaf_teleop_prior"
    / "a270c038_deleaf_side_facing_stem_priors_01.json"
)
DELEAF_SIDE_PRIOR_SOURCE_MATCH = {
    "AD1": "right_facing",
    "AD2": "left_facing",
    "AD3": "right_facing",
    "BD1": "left_facing",
    "BD2": "right_facing",
    "BD3": "right_facing",
    "BD4": "right_facing",
}
DELEAF_SIDE_AXIS_ROBOT = {
    "AD1": "0.138309,-0.990389,0.0",
    "AD2": "0.138309,-0.990389,0.0",
    "AD3": "0.138309,-0.990389,0.0",
    "BD1": "0.138309,-0.990389,0.0",
    "BD2": "0.138309,-0.990389,0.0",
    "BD3": "-0.138309,0.990389,0.0",
    "BD4": "0.138309,-0.990389,0.0",
}
DELEAF_SIDE_SAFE_ENTRY_AXIS_ROBOT = {
    "AD1": "1,0,0",
    "AD2": "-1,0,0",
    "AD3": "1,0,0",
    "BD1": "-1,0,0",
    "BD2": "1,0,0",
    "BD3": "-1,0,0",
    "BD4": "-1,0,0",
}
HARVEST_TARGET_PROFILES: dict[str, dict[str, Any]] = {
    "AT1T1": {
        "description": "tomato0: known-good deeper side grasp profile",
        "grasp_insertion_factors": "1.25,1.5",
        "max_grasp_insertion_factor": 1.5,
        "min_grasp_insertion_fraction": 0.18,
        "harvest_distance_m": 0.0,
        "harvest_down_m": 0.0,
        "pregrasp_task_position_tolerance_m": 0.008,
        "pregrasp_task_jaw_tolerance_rad": 0.30,
        "pregrasp_task_backoff_tolerance_rad": 0.40,
        "grasp_task_position_tolerance_m": 0.05,
        "grasp_task_jaw_tolerance_rad": 0.35,
        "grasp_task_backoff_tolerance_rad": 0.65,
    },
    "AT1T2": {
        "description": "AT1T2 current-state side grasp search",
        "target_bias_robot_m": "0,0,0.012",
        "full_orientation_weight": 0.5,
        "grasp_insertion_factors": "0.75,1.0,1.25",
        "max_grasp_insertion_factor": 1.25,
        "min_grasp_insertion_fraction": 0.18,
        "harvest_distance_m": 0.0,
        "harvest_down_m": 0.0,
        "palm_correction_factors": "0.0,1.0",
        "lateral_offsets_m": "-0.10,-0.08,-0.06,-0.04,0.04,0.06",
        "elevated_offsets_m": "0.02,0.04",
        "tcp_rolls_deg": "0",
        "grasp_prior_capture_json": DEFAULT_TOMATO1_GRASP_PRIOR,
        "grasp_prior_weight": 0.05,
        "pregrasp_task_position_tolerance_m": 0.008,
        "pregrasp_task_jaw_tolerance_rad": 0.30,
        "pregrasp_task_backoff_tolerance_rad": 0.40,
        "grasp_task_position_tolerance_m": 0.05,
        "grasp_task_jaw_tolerance_rad": 0.30,
        "grasp_task_backoff_tolerance_rad": 0.35,
    },
    "AT1T3": {
        "description": "tomato2: deeper insertion with small roll/lateral sweep",
        "grasp_insertion_factors": "1.0,1.25,1.5",
        "max_grasp_insertion_factor": 1.5,
        "min_grasp_insertion_fraction": 0.18,
        "harvest_distance_m": 0.0,
        "harvest_down_m": 0.0,
        "lateral_offsets_m": "0.02,0.04,0.06,0.08,-0.02,-0.04,-0.06,-0.08",
        "elevated_offsets_m": "0.03,0.05,0.08",
        "tcp_rolls_deg": "-15,0,15,165,180,195",
        "pregrasp_task_position_tolerance_m": 0.008,
        "pregrasp_task_jaw_tolerance_rad": 0.30,
        "pregrasp_task_backoff_tolerance_rad": 0.40,
        "grasp_task_position_tolerance_m": 0.05,
        "grasp_task_jaw_tolerance_rad": 0.35,
        "grasp_task_backoff_tolerance_rad": 0.65,
    },
}
SAME_ROD_MULTI = (
    ("AD1", "AD2"),
    ("AD2", "AD3"),
    ("AD1", "AD3"),
    ("BD1", "BD2"),
    ("BD2", "BD3"),
    ("BD1", "BD3"),
)
CROSS_ROD_MULTI = (
    ("AD1", "BD1"),
    ("AD2", "BD2"),
    ("AD3", "BD3"),
    ("AD1", "BD3"),
    ("AD3", "BD1"),
    ("AD2", "BD1"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def copy_json_payload(source: Path, dest: Path, *, artifact_path: Path | None = None) -> dict[str, Any]:
    payload = read_json(source)
    if artifact_path is not None:
        payload["artifact_path"] = str(artifact_path)
    write_json(dest, payload)
    return payload


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def action_episode_dir(episode: dict[str, Any], action: dict[str, Any]) -> Path:
    return Path(episode["episode_dir"]) / "actions" / str(action["step_id"])


def visit_json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(visit_json_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(visit_json_strings(item))
        return strings
    return []


def rewrite_json_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [rewrite_json_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_json_paths(item, path_map) for key, item in value.items()}
    return value


def copy_episode_file(
    source: Path,
    destination_dir: Path,
    *,
    path_map: dict[str, str],
    copied_jsons: set[Path],
) -> Path:
    source = Path(source)
    if not source.exists():
        return source
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    path_map[str(source)] = str(destination)
    path_map[str(source.resolve())] = str(destination.resolve())
    if destination.suffix.lower() == ".json":
        copied_jsons.add(destination)
    return destination


def matching_sidecars(source: Path) -> list[Path]:
    suffixes = {".json", ".jpg", ".jpeg", ".png", ".npz", ".ply"}
    if not source.exists():
        return []
    return sorted(
        path
        for path in source.parent.iterdir()
        if path.is_file() and path.stem == source.stem and path.suffix.lower() in suffixes
    )


def referenced_episode_files(source: Path, *, step_id: str, episode_dir: Path) -> list[Path]:
    if not source.exists() or source.suffix.lower() != ".json":
        return []
    try:
        payload = read_json(source)
    except json.JSONDecodeError:
        return []
    files: list[Path] = []
    for text in visit_json_strings(payload):
        if step_id not in text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute() or not candidate.exists() or is_relative_to(candidate, episode_dir):
            continue
        files.append(candidate)
    return sorted(set(files), key=lambda path: str(path))


def copy_artifact_bundle(
    source: Path | None,
    *,
    destination_dir: Path,
    step_id: str,
    episode_dir: Path,
    path_map: dict[str, str],
    copied_jsons: set[Path],
) -> Path | None:
    if source is None:
        return None
    source = Path(source)
    if not source.exists():
        return source
    if is_relative_to(source, episode_dir):
        if source.suffix.lower() == ".json":
            copied_jsons.add(source)
        return source

    copied_main = copy_episode_file(
        source,
        destination_dir,
        path_map=path_map,
        copied_jsons=copied_jsons,
    )
    for sidecar in matching_sidecars(source):
        copy_episode_file(sidecar, destination_dir, path_map=path_map, copied_jsons=copied_jsons)
    for referenced in referenced_episode_files(source, step_id=step_id, episode_dir=episode_dir):
        reference_dir = destination_dir / "referenced" / referenced.parent.name
        copy_episode_file(referenced, reference_dir, path_map=path_map, copied_jsons=copied_jsons)
        for sidecar in matching_sidecars(referenced):
            copy_episode_file(sidecar, reference_dir, path_map=path_map, copied_jsons=copied_jsons)
    return copied_main


def localize_action_artifacts(
    *,
    episode: dict[str, Any],
    action: dict[str, Any],
    action_artifact: Path | None,
    target_artifact: Path | None,
    planner_artifact: Path | None,
    trajectory_artifact: Path | None,
    execution_artifact: Path | None,
) -> dict[str, Path | None]:
    episode_dir = Path(episode["episode_dir"])
    base_dir = action_episode_dir(episode, action)
    path_map: dict[str, str] = {}
    copied_jsons: set[Path] = set()
    localized = {
        "action_artifact": copy_artifact_bundle(
            action_artifact,
            destination_dir=base_dir / "action",
            step_id=str(action["step_id"]),
            episode_dir=episode_dir,
            path_map=path_map,
            copied_jsons=copied_jsons,
        ),
        "target_artifact": copy_artifact_bundle(
            target_artifact,
            destination_dir=base_dir / "target",
            step_id=str(action["step_id"]),
            episode_dir=episode_dir,
            path_map=path_map,
            copied_jsons=copied_jsons,
        ),
        "planner_artifact": copy_artifact_bundle(
            planner_artifact,
            destination_dir=base_dir / "planner",
            step_id=str(action["step_id"]),
            episode_dir=episode_dir,
            path_map=path_map,
            copied_jsons=copied_jsons,
        ),
        "trajectory_artifact": copy_artifact_bundle(
            trajectory_artifact,
            destination_dir=base_dir / "trajectory",
            step_id=str(action["step_id"]),
            episode_dir=episode_dir,
            path_map=path_map,
            copied_jsons=copied_jsons,
        ),
        "execution_artifact": copy_artifact_bundle(
            execution_artifact,
            destination_dir=base_dir / "execution",
            step_id=str(action["step_id"]),
            episode_dir=episode_dir,
            path_map=path_map,
            copied_jsons=copied_jsons,
        ),
    }
    task_frame = action.get("artifacts", {}).get("task_frame")
    if isinstance(task_frame, str):
        task_frame_path = Path(task_frame)
        if task_frame_path.exists() and task_frame_path.suffix.lower() == ".json":
            copied_jsons.add(task_frame_path)
    for path in copied_jsons:
        payload = read_json(path)
        rewritten = rewrite_json_paths(payload, path_map)
        if rewritten != payload:
            write_json(path, rewritten)
    return localized


def parse_deleaf_targets(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError("expected at least one deleaf target label")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"duplicate deleaf target label(s): {', '.join(duplicates)}")
    for label in labels:
        meta = label_metadata(label)
        if meta["primitive"] != "deleaf":
            raise argparse.ArgumentTypeError(f"not a deleaf target label: {label}")
    return labels


def parse_harvest_targets(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError("expected at least one harvest target label")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"duplicate harvest target label(s): {', '.join(duplicates)}")
    for label in labels:
        meta = label_metadata(label)
        if meta["primitive"] != "harvest":
            raise argparse.ArgumentTypeError(f"not a harvest target label: {label}")
    return labels


def parse_vector3_arg(value: str, *, name: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly three comma-separated numbers")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain only numbers") from exc


def arg_overridden(value: Any, default: Any) -> bool:
    if isinstance(default, float):
        try:
            return not math.isclose(float(value), float(default), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return True
    return str(value) != str(default)


def profile_value(
    args: argparse.Namespace,
    *,
    attr: str,
    default: Any,
    profile: dict[str, Any],
    profile_key: str,
) -> Any:
    value = getattr(args, attr)
    if arg_overridden(value, default):
        return value
    return profile.get(profile_key, value)


def harvest_profile_settings(action: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    requested = str(args.harvest_profile)
    if requested == "global":
        profile_name = "global"
        profile: dict[str, Any] = {}
    elif requested == "auto":
        profile_name = str(action["symbolic_target"]).upper()
        profile = HARVEST_TARGET_PROFILES.get(profile_name, {})
    else:
        profile_name = requested.upper()
        if profile_name not in HARVEST_TARGET_PROFILES:
            raise ValueError(f"unknown harvest profile: {requested}")
        profile = HARVEST_TARGET_PROFILES[profile_name]
    return {
        "requested": requested,
        "resolved": profile_name,
        "description": profile.get("description", "generic harvest profile"),
        "target_bias_robot_m": profile_value(
            args,
            attr="harvest_target_bias_robot_m",
            default=DEFAULT_HARVEST_TARGET_BIAS_ROBOT_M,
            profile=profile,
            profile_key="target_bias_robot_m",
        ),
        "contact_palm_percentile": float(
            profile_value(
                args,
                attr="harvest_contact_palm_percentile",
                default=DEFAULT_HARVEST_CONTACT_PALM_PERCENTILE,
                profile=profile,
                profile_key="contact_palm_percentile",
            )
        ),
        "harvest_distance_m": float(
            profile_value(
                args,
                attr="harvest_distance_m",
                default=DEFAULT_HARVEST_DISTANCE_M,
                profile=profile,
                profile_key="harvest_distance_m",
            )
        ),
        "harvest_down_m": float(
            profile_value(
                args,
                attr="harvest_down_m",
                default=DEFAULT_HARVEST_DOWN_M,
                profile=profile,
                profile_key="harvest_down_m",
            )
        ),
        "full_orientation_weight": float(
            profile_value(
                args,
                attr="harvest_full_orientation_weight",
                default=DEFAULT_HARVEST_FULL_ORIENTATION_WEIGHT,
                profile=profile,
                profile_key="full_orientation_weight",
            )
        ),
        "grasp_insertion_factors": str(
            profile_value(
                args,
                attr="harvest_grasp_insertion_factors",
                default=DEFAULT_HARVEST_GRASP_INSERTION_FACTORS,
                profile=profile,
                profile_key="grasp_insertion_factors",
            )
        ),
        "max_grasp_insertion_factor": float(
            profile_value(
                args,
                attr="harvest_max_grasp_insertion_factor",
                default=DEFAULT_HARVEST_MAX_GRASP_INSERTION_FACTOR,
                profile=profile,
                profile_key="max_grasp_insertion_factor",
            )
        ),
        "min_grasp_insertion_fraction": float(
            profile_value(
                args,
                attr="harvest_min_grasp_insertion_fraction",
                default=DEFAULT_HARVEST_MIN_GRASP_INSERTION_FRACTION,
                profile=profile,
                profile_key="min_grasp_insertion_fraction",
            )
        ),
        "palm_correction_factors": str(
            profile_value(
                args,
                attr="harvest_palm_correction_factors",
                default=DEFAULT_HARVEST_PALM_CORRECTION_FACTORS,
                profile=profile,
                profile_key="palm_correction_factors",
            )
        ),
        "lateral_offsets_m": str(
            profile_value(
                args,
                attr="harvest_lateral_offsets_m",
                default=DEFAULT_HARVEST_LATERAL_OFFSETS_M,
                profile=profile,
                profile_key="lateral_offsets_m",
            )
        ),
        "elevated_offsets_m": str(
            profile_value(
                args,
                attr="harvest_elevated_offsets_m",
                default=DEFAULT_HARVEST_ELEVATED_OFFSETS_M,
                profile=profile,
                profile_key="elevated_offsets_m",
            )
        ),
        "tcp_rolls_deg": str(
            profile_value(
                args,
                attr="harvest_tcp_rolls_deg",
                default=DEFAULT_HARVEST_TCP_ROLLS_DEG,
                profile=profile,
                profile_key="tcp_rolls_deg",
            )
        ),
        "grasp_prior_capture_json": None
        if profile_value(
            args,
            attr="harvest_grasp_prior_capture_json",
            default=None,
            profile=profile,
            profile_key="grasp_prior_capture_json",
        )
        is None
        else str(
            profile_value(
                args,
                attr="harvest_grasp_prior_capture_json",
                default=None,
                profile=profile,
                profile_key="grasp_prior_capture_json",
            )
        ),
        "grasp_prior_weight": float(
            profile_value(
                args,
                attr="harvest_grasp_prior_weight",
                default=0.05,
                profile=profile,
                profile_key="grasp_prior_weight",
            )
        ),
        "pregrasp_task_position_tolerance_m": float(
            profile_value(
                args,
                attr="harvest_pregrasp_task_axis_position_tolerance_m",
                default=DEFAULT_HARVEST_PREGRASP_TASK_POSITION_TOLERANCE_M,
                profile=profile,
                profile_key="pregrasp_task_position_tolerance_m",
            )
        ),
        "pregrasp_task_jaw_tolerance_rad": float(
            profile_value(
                args,
                attr="harvest_pregrasp_task_axis_jaw_tolerance_rad",
                default=DEFAULT_HARVEST_PREGRASP_TASK_JAW_TOLERANCE_RAD,
                profile=profile,
                profile_key="pregrasp_task_jaw_tolerance_rad",
            )
        ),
        "pregrasp_task_backoff_tolerance_rad": float(
            profile_value(
                args,
                attr="harvest_pregrasp_task_axis_backoff_tolerance_rad",
                default=DEFAULT_HARVEST_PREGRASP_TASK_BACKOFF_TOLERANCE_RAD,
                profile=profile,
                profile_key="pregrasp_task_backoff_tolerance_rad",
            )
        ),
        "grasp_task_position_tolerance_m": float(
            profile_value(
                args,
                attr="harvest_grasp_task_axis_position_tolerance_m",
                default=DEFAULT_HARVEST_GRASP_TASK_POSITION_TOLERANCE_M,
                profile=profile,
                profile_key="grasp_task_position_tolerance_m",
            )
        ),
        "grasp_task_jaw_tolerance_rad": float(
            profile_value(
                args,
                attr="harvest_grasp_task_axis_jaw_tolerance_rad",
                default=DEFAULT_HARVEST_GRASP_TASK_JAW_TOLERANCE_RAD,
                profile=profile,
                profile_key="grasp_task_jaw_tolerance_rad",
            )
        ),
        "grasp_task_backoff_tolerance_rad": float(
            profile_value(
                args,
                attr="harvest_grasp_task_axis_backoff_tolerance_rad",
                default=DEFAULT_HARVEST_GRASP_TASK_BACKOFF_TOLERANCE_RAD,
                profile=profile,
                profile_key="grasp_task_backoff_tolerance_rad",
            )
        ),
    }


def apply_harvest_arg_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "dataset_name": DEFAULT_DATASET_NAME,
        "runs_root": POLICY_RUNS_ROOT,
        "harvest_target_picker": "open3d",
        "harvest_target_source": "click",
        "harvest_identity_task_frame_json": None,
        "harvest_profile": "auto",
        "harvest_toolfit": DEFAULT_HARVEST_TOOLFIT,
        "harvest_contact_frame_json": None,
        "harvest_primitive_prior_dir": DEFAULT_HARVEST_PRIMITIVE_PRIOR_DIR,
        "harvest_primitive_prior_pattern": DEFAULT_HARVEST_TARGET_PATTERN,
        "harvest_basket_anchor_robot_m": DEFAULT_HARVEST_BASKET_ANCHOR_ROBOT_M,
        "harvest_target_bias_robot_m": DEFAULT_HARVEST_TARGET_BIAS_ROBOT_M,
        "harvest_generic_depth_m": DEFAULT_HARVEST_GENERIC_DEPTH_M,
        "harvest_generic_width_m": DEFAULT_HARVEST_GENERIC_WIDTH_M,
        "harvest_palm_correction_factors": DEFAULT_HARVEST_PALM_CORRECTION_FACTORS,
        "harvest_disable_live_target_geometry": False,
        "harvest_local_radius_px": DEFAULT_HARVEST_LOCAL_RADIUS_PX,
        "harvest_local_depth_window_m": DEFAULT_HARVEST_LOCAL_DEPTH_WINDOW_M,
        "harvest_local_color_threshold": DEFAULT_HARVEST_LOCAL_COLOR_THRESHOLD,
        "harvest_local_radius_m": DEFAULT_HARVEST_LOCAL_RADIUS_M,
        "harvest_min_local_points": DEFAULT_HARVEST_MIN_LOCAL_POINTS,
        "harvest_max_derived_target_shift_m": DEFAULT_HARVEST_MAX_DERIVED_TARGET_SHIFT_M,
        "harvest_contact_palm_percentile": DEFAULT_HARVEST_CONTACT_PALM_PERCENTILE,
        "harvest_min_preclose_z_margin_m": DEFAULT_HARVEST_MIN_PRECLOSE_Z_MARGIN_M,
        "harvest_distance_m": DEFAULT_HARVEST_DISTANCE_M,
        "harvest_down_m": DEFAULT_HARVEST_DOWN_M,
        "harvest_full_orientation_weight": DEFAULT_HARVEST_FULL_ORIENTATION_WEIGHT,
        "harvest_grasp_insertion_factors": DEFAULT_HARVEST_GRASP_INSERTION_FACTORS,
        "harvest_max_grasp_insertion_factor": DEFAULT_HARVEST_MAX_GRASP_INSERTION_FACTOR,
        "harvest_min_grasp_insertion_fraction": DEFAULT_HARVEST_MIN_GRASP_INSERTION_FRACTION,
        "harvest_lateral_offsets_m": DEFAULT_HARVEST_LATERAL_OFFSETS_M,
        "harvest_elevated_offsets_m": DEFAULT_HARVEST_ELEVATED_OFFSETS_M,
        "harvest_tcp_rolls_deg": DEFAULT_HARVEST_TCP_ROLLS_DEG,
        "harvest_pregrasp_task_axis_position_tolerance_m": DEFAULT_HARVEST_PREGRASP_TASK_POSITION_TOLERANCE_M,
        "harvest_pregrasp_task_axis_jaw_tolerance_rad": DEFAULT_HARVEST_PREGRASP_TASK_JAW_TOLERANCE_RAD,
        "harvest_pregrasp_task_axis_backoff_tolerance_rad": DEFAULT_HARVEST_PREGRASP_TASK_BACKOFF_TOLERANCE_RAD,
        "harvest_grasp_task_axis_position_tolerance_m": DEFAULT_HARVEST_GRASP_TASK_POSITION_TOLERANCE_M,
        "harvest_grasp_task_axis_jaw_tolerance_rad": DEFAULT_HARVEST_GRASP_TASK_JAW_TOLERANCE_RAD,
        "harvest_grasp_task_axis_backoff_tolerance_rad": DEFAULT_HARVEST_GRASP_TASK_BACKOFF_TOLERANCE_RAD,
        "harvest_grasp_palm_alignment_tolerance_m": 0.04,
        "harvest_grasp_prior_capture_json": None,
        "harvest_grasp_prior_weight": 0.05,
        "width": 640,
        "height": 480,
        "fps": 30,
        "warmup_frames": 30,
        "min_depth_m": 0.15,
        "max_depth_m": 2.0,
        "pointcloud_stride": 2,
        "serial": None,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if not hasattr(args, "harvest_grasp_insertion_factor"):
        setattr(args, "harvest_grasp_insertion_factor", None)
    return args


def harvest_identity_task_frame(action: dict[str, Any], args: argparse.Namespace) -> Path | None:
    if args.harvest_identity_task_frame_json is not None:
        return Path(args.harvest_identity_task_frame_json)
    return HARVEST_IDENTITY_TASK_FRAMES.get(str(action["symbolic_target"]).upper())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def harvest_fixed_plan_for_action(action: dict[str, Any], args: argparse.Namespace) -> tuple[Path | None, str | None]:
    if args.harvest_fixed_plan_json is not None:
        return Path(args.harvest_fixed_plan_json), "explicit"
    mode = str(args.harvest_fixed_plan_mode)
    if mode == "off":
        return None, None
    label = str(action["symbolic_target"]).upper()
    source = HARVEST_FIXED_FULL_TAIL_PLANS.get(label)
    if source is None or not Path(source).exists():
        if mode == "required":
            raise RuntimeError(
                f"no registered fixed full-tail harvest plan for {label}; collect and register one before running this label"
            )
        return None, None
    return source, "registered"


def validate_fixed_harvest_plan(path: Path, action: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"fixed harvest plan does not exist: {path}")
    payload = read_json(path)
    best_candidate = payload.get("best_candidate")
    best_waypoints = best_candidate.get("waypoints", []) if isinstance(best_candidate, dict) else []
    waypoint_names = [waypoint.get("name") for waypoint in best_waypoints]
    required_waypoints = {"approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"}
    missing = sorted(required_waypoints - set(waypoint_names))
    if not payload.get("hard_gate", {}).get("ok", False):
        raise RuntimeError(f"fixed harvest plan does not pass hard gates: {path}")
    if missing:
        raise RuntimeError(
            "fixed harvest plan is not a complete full-tail harvest plan; "
            f"missing={missing}. Artifact: {path}. Waypoints: {waypoint_names}"
        )
    goal = extract_goal_from_artifact(path)
    target = payload.get("target", {})
    target_name = None
    if isinstance(target, dict):
        target_name = target.get("name") or target.get("target_name")
    return {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "symbolic_target": action["symbolic_target"],
        "plan_target_name": target_name or goal.get("target_name"),
        "candidate_name": best_candidate.get("name") if isinstance(best_candidate, dict) else None,
        "waypoint_names": waypoint_names,
        "hard_gate": payload.get("hard_gate", {}),
        "goal": goal,
    }


def capture_harvest_identity_target(
    *,
    python: str,
    action: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None]:
    identity_task_frame = harvest_identity_task_frame(action, args)
    use_identity_target = (
        args.harvest_target_source == "identity"
        or (args.harvest_target_source == "auto" and identity_task_frame is not None and identity_task_frame.exists())
    )
    if not use_identity_target:
        return None, None
    if identity_task_frame is None or not identity_task_frame.exists():
        raise RuntimeError(
            f"no calibrated identity task frame exists for harvest target {action['symbolic_target']}; "
            "use --harvest-target-source click or --harvest-target-source auto"
        )
    validation_output = run_command_tee(
        [
            python,
            "robot/kinematics/apps/validate_live_target_identity.py",
            "--dataset-name",
            args.dataset_name,
            "--runs-root",
            str(args.runs_root),
            "--run-label",
            f"{action['step_id']}_identity",
            "--task-frame-jsons",
            str(identity_task_frame),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
            "--warmup-frames",
            str(args.warmup_frames),
            "--min-depth-m",
            str(args.min_depth_m),
            "--max-depth-m",
            str(args.max_depth_m),
            "--pointcloud-stride",
            str(args.pointcloud_stride),
        ],
        cwd=_REPO_ROOT,
    )
    identity_validation_artifact = path_from_last_payload(validation_output, "artifacts", "json")
    validation_payload = read_json(identity_validation_artifact)
    if not validation_payload.get("ok", False):
        raise RuntimeError(f"live harvest target identity validation failed: {identity_validation_artifact}")
    action.setdefault("artifacts", {})["task_frame"] = str(identity_task_frame)
    goal = extract_goal_from_artifact(identity_validation_artifact)
    print(
        "Using live target identity validation: "
        f"task_frame={identity_task_frame} "
        f"target={goal.get('target_name')} "
        f"surface="
        + ",".join(f"{value:.6f}" for value in goal.get("point_robot_m", []))
    )
    return identity_validation_artifact, identity_task_frame


def capture_harvest_click_target(
    *,
    python: str,
    action: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[float], Path]:
    target_output = run_command_tee(
        [
            python,
            "robot/kinematics/utils/realsense_click_move_arm.py",
            "--run-label",
            f"{action['step_id']}_target",
            "--arm",
            "left",
            "--dataset-name",
            args.dataset_name,
            "--toolfit",
            str(args.harvest_toolfit),
            "--pick-mode",
            args.harvest_target_picker,
            "--max-y-m",
            "0.55",
            "--max-z-m",
            "1.35",
        ],
        cwd=_REPO_ROOT,
    )
    return point_robot_from_capture(target_output)


def fixed_harvest_plan_variant(
    *,
    source_plan: Path,
    episode: dict[str, Any],
    action: dict[str, Any],
    entry_exit_jitter_deg: float,
    seed: int | None,
) -> Path:
    jitter_deg = float(entry_exit_jitter_deg)
    if jitter_deg <= 0.0:
        return source_plan
    payload = read_json(source_plan)
    variant = json.loads(json.dumps(payload))
    candidate = variant.get("best_candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError(f"fixed harvest plan has no best_candidate: {source_plan}")
    if seed is None:
        seed = int(time.time() * 1000) % (2**31 - 1)
    rng = random.Random(int(seed))
    jitter_rad = math.radians(jitter_deg)
    varied: list[dict[str, Any]] = []
    for waypoint in candidate.get("waypoints", []):
        name = waypoint.get("name")
        if name not in {"approach", "retract"}:
            continue
        ik_result = waypoint.get("ik_result")
        if not isinstance(ik_result, dict) or "q_arm_rad" not in ik_result:
            continue
        q_arm = [float(value) for value in ik_result["q_arm_rad"]]
        offsets = [rng.uniform(-jitter_rad, jitter_rad) for _ in q_arm]
        ik_result["q_arm_rad"] = [q_arm[index] + offsets[index] for index in range(len(q_arm))]
        varied.append(
            {
                "waypoint": name,
                "offsets_deg": [math.degrees(value) for value in offsets],
            }
        )
    if not varied:
        raise RuntimeError(f"fixed harvest plan has no approach/retract waypoint to vary: {source_plan}")
    variant["fixed_plan_variation"] = {
        "source_plan": str(source_plan),
        "mode": "entry_exit_joint_jitter",
        "seed": int(seed),
        "entry_exit_jitter_deg": jitter_deg,
        "varied_waypoints": varied,
        "critical_waypoints_unchanged": ["pre_grasp", "grasp", "harvest", "basket_place"],
        "note": "Only approach/retract q_arm_rad values are jittered; grasp and basket waypoints are unchanged.",
    }
    variant["artifact_path"] = None
    out_dir = action_episode_dir(episode, action) / "fixed_plan_variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_source = source_plan.stem
    out_path = out_dir / f"{action['step_id']}_{safe_source}_entry_exit_jitter_seed{int(seed)}.json"
    variant["artifact_path"] = str(out_path)
    write_json(out_path, variant)
    return out_path


def register_successful_harvest_solve(
    *,
    target_label: str,
    planner_artifact: Path,
    summary_path: Path,
    execution_artifact: Path | None,
) -> dict[str, Any]:
    label = str(target_label).upper()
    if label not in HARVEST_TARGETS:
        raise ValueError(f"unknown harvest target label: {label}")
    planner_payload = read_json(planner_artifact)
    if not planner_payload.get("hard_gate", {}).get("ok", False):
        raise RuntimeError(f"refusing to register a harvest solve with hard_gate.ok != true: {planner_artifact}")
    best_candidate = planner_payload.get("best_candidate")
    waypoint_names = (
        [waypoint.get("name") for waypoint in best_candidate.get("waypoints", [])]
        if isinstance(best_candidate, dict)
        else []
    )
    required = {"approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"}
    missing = sorted(required - set(waypoint_names))
    if missing:
        raise RuntimeError(
            f"refusing to register an incomplete harvest solve for {label}; missing={missing}. "
            f"Artifact: {planner_artifact}"
        )
    library_dir = HARVEST_SOLVE_LIBRARY_ROOT / label
    library_dir.mkdir(parents=True, exist_ok=True)
    plan_dest = library_dir / f"{DEFAULT_DATASET_NAME}_{label}_success_current.json"
    summary_dest = library_dir / f"{DEFAULT_DATASET_NAME}_{label}_success_summary.json"
    planner_payload["artifact_path"] = str(plan_dest)
    planner_payload["registered_successful_solve"] = {
        "target_label": label,
        "registered_at_utc": utc_now(),
        "source_planner_artifact": str(planner_artifact),
        "source_summary": str(summary_path),
    }
    write_json(plan_dest, planner_payload)
    shutil.copy2(summary_path, summary_dest)
    execution_dest = None
    if execution_artifact is not None and Path(execution_artifact).exists():
        execution_dest = library_dir / f"{DEFAULT_DATASET_NAME}_{label}_success_execution.json"
        shutil.copy2(execution_artifact, execution_dest)
    return {
        "target_label": label,
        "registered_plan": str(plan_dest),
        "registered_summary": str(summary_dest),
        "registered_execution": str(execution_dest) if execution_dest is not None else None,
        "candidate_name": best_candidate.get("name") if isinstance(best_candidate, dict) else None,
        "waypoint_names": waypoint_names,
        "plan_sha256": sha256_file(plan_dest),
    }


def unit_vector3(values: list[float] | tuple[float, float, float], *, name: str) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 1e-12:
        raise ValueError(f"{name} has near-zero norm")
    return [float(value) / norm for value in values]


def cross3(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def matrix_from_columns(columns: tuple[list[float], list[float], list[float]]) -> list[list[float]]:
    return [[columns[col][row] for col in range(3)] for row in range(3)]


def add3(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def scale3(values: list[float], scalar: float) -> list[float]:
    return [float(scalar) * values[i] for i in range(3)]


def dot3(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def project_out(values: list[float], axis: list[float]) -> list[float]:
    return [values[i] - dot3(values, axis) * axis[i] for i in range(3)]


def median_float(values: list[float], *, default: float) -> float:
    filtered = sorted(float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0)
    if not filtered:
        return float(default)
    mid = len(filtered) // 2
    if len(filtered) % 2:
        return filtered[mid]
    return 0.5 * (filtered[mid - 1] + filtered[mid])


def maybe_flip_axis(axis: list[float], reference: list[float]) -> list[float]:
    return scale3(axis, -1.0) if dot3(axis, reference) < 0.0 else axis


def vector_from_json(data: dict[str, Any], key: str) -> list[float] | None:
    value = data.get(key)
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        vector = [float(part) for part in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(part) for part in vector):
        return None
    return vector


def harvest_primitive_prior_from_task_frames(
    *,
    prior_dir: Path,
    pattern: str,
    fallback_depth_m: float,
    fallback_width_m: float,
) -> dict[str, Any]:
    paths = tuple(sorted(Path(prior_dir).glob(pattern)))
    backoff_axes: list[list[float]] = []
    pull_axes: list[list[float]] = []
    depth_values: list[float] = []
    width_values: list[float] = []
    for path in paths:
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        backoff = vector_from_json(payload, "gripper_backoff_axis_robot")
        if backoff is not None:
            try:
                backoff_axes.append(maybe_flip_axis(unit_vector3(backoff, name="harvest prior backoff"), [-1.0, 0.0, 0.0]))
            except ValueError:
                pass
        task_axes = payload.get("task_axes_robot", {})
        if isinstance(task_axes, dict):
            pull = vector_from_json(task_axes, "harvest_pull_axis_robot")
            if pull is not None:
                try:
                    pull_axes.append(maybe_flip_axis(unit_vector3(pull, name="harvest prior pull"), [-1.0, -1.0, -0.1]))
                except ValueError:
                    pass
        quality = payload.get("quality", {})
        if isinstance(quality, dict):
            for key, target in (
                ("estimated_grasp_depth_m", depth_values),
                ("estimated_grasp_width_m", width_values),
            ):
                try:
                    value = float(quality.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0.0:
                    target.append(value)

    palm_axis = [0.0, 0.0, -1.0]
    if backoff_axes:
        backoff_sum = [0.0, 0.0, 0.0]
        for axis in backoff_axes:
            backoff_sum = add3(backoff_sum, axis)
        backoff_axis = unit_vector3(project_out(backoff_sum, palm_axis), name="harvest prior horizontal backoff")
    else:
        backoff_axis = [-1.0, 0.0, 0.0]
    jaw_axis = unit_vector3(cross3(palm_axis, backoff_axis), name="harvest prior jaw axis")
    palm_axis = unit_vector3(cross3(backoff_axis, jaw_axis), name="harvest prior palm axis")

    if pull_axes:
        pull_sum = [0.0, 0.0, 0.0]
        for axis in pull_axes:
            pull_sum = add3(pull_sum, axis)
        pull_axis = unit_vector3(pull_sum, name="harvest prior pull axis")
    else:
        pull_axis = unit_vector3([-0.70, -0.70, -0.10], name="generic harvest pull axis")

    return {
        "backoff_axis_robot": backoff_axis,
        "jaw_axis_robot": jaw_axis,
        "palm_axis_robot": palm_axis,
        "pull_axis_robot": pull_axis,
        "estimated_grasp_depth_m": median_float(depth_values, default=fallback_depth_m),
        "estimated_grasp_width_m": median_float(width_values, default=fallback_width_m),
        "source_paths": [str(path) for path in paths],
        "source_count": len(paths),
        "mode": "mps_guided_harvest_primitive_prior" if paths else "generic_robot_front_fallback",
    }


def quat_wxyz_from_rotation(rotation: list[list[float]]) -> list[float]:
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s
    quat = [w, x, y, z]
    norm = math.sqrt(sum(value * value for value in quat))
    return [value / norm for value in quat]


def load_robot_from_realsense(transform_json: Path) -> Any:
    import numpy as np

    payload = read_json(transform_json)
    return np.asarray(payload["T_robot_from_realsense"], dtype=float).reshape(4, 4)


def transform_realsense_points_to_robot(points_realsense_m: Any, transform_json: Path) -> Any:
    import numpy as np

    points = np.asarray(points_realsense_m, dtype=float).reshape(-1, 3)
    homogeneous = np.ones((points.shape[0], 4), dtype=float)
    homogeneous[:, :3] = points
    return (load_robot_from_realsense(transform_json) @ homogeneous.T).T[:, :3]


def clamp_float(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def robust_axis_span_m(values: Any, *, lower: float = 5.0, upper: float = 95.0) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, upper) - np.percentile(array, lower))


def bounded_observed_dimension_m(
    observed_m: float,
    *,
    prior_m: float,
    min_fraction: float,
    max_fraction: float,
) -> float:
    prior = max(1e-6, float(prior_m))
    observed = float(observed_m)
    if not math.isfinite(observed) or observed <= 0.0:
        return prior
    return clamp_float(observed, min_fraction * prior, max_fraction * prior)


def live_rgbd_component_points_robot(
    *,
    target_artifact: Path,
    selected_point_robot_m: list[float],
    local_radius_px: int,
    local_depth_window_m: float,
    local_color_threshold: float,
    local_radius_m: float,
    min_points: int,
) -> dict[str, Any]:
    import numpy as np

    try:
        import cv2
    except ImportError:
        cv2 = None

    payload = read_json(target_artifact)
    selected = payload.get("selected_point") or {}
    artifacts = payload.get("artifacts") or {}
    npz_path = artifacts.get("npz")
    transform_json = payload.get("transform_json")
    pixel_xy = selected.get("pixel_xy")
    point_realsense_m = selected.get("point_realsense_m")
    if npz_path is None or transform_json is None or pixel_xy is None or point_realsense_m is None:
        return {
            "ok": False,
            "reason": "target capture is missing npz, transform_json, pixel_xy, or point_realsense_m",
            "points_robot_m": [],
        }

    capture = np.load(str(npz_path))
    color_bgr = np.asarray(capture["color_bgr"])
    depth_m = np.asarray(capture["depth_m"], dtype=float)
    height, width = depth_m.shape
    x_seed, y_seed = int(pixel_xy[0]), int(pixel_xy[1])
    if x_seed < 0 or x_seed >= width or y_seed < 0 or y_seed >= height:
        return {"ok": False, "reason": "selected pixel is outside captured image", "points_robot_m": []}

    selected_depth = float(np.asarray(point_realsense_m, dtype=float).reshape(3)[2])
    selected_robot = np.asarray(selected_point_robot_m, dtype=float).reshape(3)
    radius_px = max(1, int(local_radius_px))
    y0, y1 = max(0, y_seed - radius_px), min(height, y_seed + radius_px + 1)
    x0, x1 = max(0, x_seed - radius_px), min(width, x_seed + radius_px + 1)
    roi_depth = depth_m[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    valid_depth = (
        np.isfinite(roi_depth)
        & (roi_depth > 0.0)
        & (np.abs(roi_depth - selected_depth) <= float(local_depth_window_m))
    )
    selected_rgb = color_bgr[y_seed, x_seed, ::-1].astype(float) / 255.0
    roi_rgb = color_bgr[y0:y1, x0:x1, ::-1].astype(float) / 255.0
    color_distance = np.linalg.norm(roi_rgb - selected_rgb.reshape(1, 1, 3), axis=2)
    seeded_mask = valid_depth & (color_distance <= float(local_color_threshold))
    if int(np.count_nonzero(seeded_mask)) < int(min_points):
        seeded_mask = valid_depth

    component_mask = seeded_mask
    component_source = "local_depth_color"
    if cv2 is not None and int(np.count_nonzero(seeded_mask)) >= int(min_points):
        labels_count, labels = cv2.connectedComponents(seeded_mask.astype("uint8"), connectivity=8)
        seed_label = int(labels[y_seed - y0, x_seed - x0])
        if seed_label == 0 and labels_count > 1:
            best_label = 0
            best_distance = float("inf")
            for label in range(1, labels_count):
                rows, cols = np.nonzero(labels == label)
                if rows.size == 0:
                    continue
                centroid = np.asarray([float(np.mean(cols)) + x0, float(np.mean(rows)) + y0])
                distance = float(np.linalg.norm(centroid - np.asarray([x_seed, y_seed], dtype=float)))
                if distance < best_distance:
                    best_distance = distance
                    best_label = label
            seed_label = best_label
        if seed_label != 0:
            component_mask = labels == seed_label
            component_source = "selected_connected_depth_color_component"

    rows = yy[component_mask].astype(float)
    cols = xx[component_mask].astype(float)
    z = roi_depth[component_mask].astype(float)
    if z.size < int(min_points):
        rows = yy[valid_depth].astype(float)
        cols = xx[valid_depth].astype(float)
        z = roi_depth[valid_depth].astype(float)
        component_source = "local_depth_fallback"
    if z.size == 0:
        return {"ok": False, "reason": "no valid local RGB-D points near selected target", "points_robot_m": []}

    intrinsics = payload.get("intrinsics") or {}
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    points_realsense = np.column_stack(((cols - ppx) * z / fx, (rows - ppy) * z / fy, z))
    points_robot = transform_realsense_points_to_robot(points_realsense, Path(transform_json))
    distances = np.linalg.norm(points_robot - selected_robot.reshape(1, 3), axis=1)
    within_radius = distances <= float(local_radius_m)
    if int(np.count_nonzero(within_radius)) >= int(min_points):
        points_robot = points_robot[within_radius]

    return {
        "ok": True,
        "reason": None,
        "points_robot_m": points_robot,
        "npz": str(npz_path),
        "transform_json": str(transform_json),
        "selected_pixel_xy": [x_seed, y_seed],
        "selected_color_rgb": selected_rgb.tolist(),
        "selected_depth_m": selected_depth,
        "component_source": component_source,
        "component_point_count": int(points_robot.shape[0]),
        "local_radius_px": int(local_radius_px),
        "local_depth_window_m": float(local_depth_window_m),
        "local_color_threshold": float(local_color_threshold),
        "local_radius_m": float(local_radius_m),
    }


def derive_live_harvest_target_geometry(
    *,
    target_artifact: Path,
    selected_point_robot_m: list[float],
    primitive_prior: dict[str, Any],
    local_radius_px: int,
    local_depth_window_m: float,
    local_color_threshold: float,
    local_radius_m: float,
    min_points: int,
    max_target_shift_m: float,
    contact_palm_percentile: float,
) -> dict[str, Any]:
    import numpy as np

    selected = np.asarray(selected_point_robot_m, dtype=float).reshape(3)
    backoff_axis = np.asarray(unit_vector3(primitive_prior["backoff_axis_robot"], name="harvest primitive backoff axis"))
    jaw_axis = np.asarray(unit_vector3(primitive_prior["jaw_axis_robot"], name="harvest primitive jaw axis"))
    palm_axis = np.asarray(unit_vector3(primitive_prior["palm_axis_robot"], name="harvest primitive palm axis"))
    prior_depth_m = float(primitive_prior["estimated_grasp_depth_m"])
    prior_width_m = float(primitive_prior["estimated_grasp_width_m"])

    component = live_rgbd_component_points_robot(
        target_artifact=target_artifact,
        selected_point_robot_m=selected_point_robot_m,
        local_radius_px=local_radius_px,
        local_depth_window_m=local_depth_window_m,
        local_color_threshold=local_color_threshold,
        local_radius_m=local_radius_m,
        min_points=min_points,
    )
    if not component["ok"]:
        surface_target = selected
        depth_m = prior_depth_m
        width_m = prior_width_m
        fruit_center = surface_target - 0.5 * depth_m * backoff_axis
        source = "clicked_point_fallback"
        points_count = 0
        target_shift_m = 0.0
    else:
        points = np.asarray(component["points_robot_m"], dtype=float).reshape(-1, 3)
        backoff_coord = points @ backoff_axis
        jaw_coord = points @ jaw_axis
        palm_coord = points @ palm_axis
        observed_width_m = robust_axis_span_m(jaw_coord, lower=5.0, upper=95.0)
        observed_depth_m = robust_axis_span_m(backoff_coord, lower=10.0, upper=90.0)
        width_m = bounded_observed_dimension_m(
            observed_width_m,
            prior_m=prior_width_m,
            min_fraction=0.70,
            max_fraction=1.40,
        )
        depth_m = bounded_observed_dimension_m(
            observed_depth_m,
            prior_m=prior_depth_m,
            min_fraction=0.80,
            max_fraction=1.30,
        )
        selected_backoff = float(selected @ backoff_axis)
        surface_backoff = max(selected_backoff, float(np.percentile(backoff_coord, 85.0)))
        center_backoff = surface_backoff - 0.5 * depth_m
        center_jaw = float(0.5 * (np.percentile(jaw_coord, 5.0) + np.percentile(jaw_coord, 95.0)))
        center_palm = float(0.5 * (np.percentile(palm_coord, 5.0) + np.percentile(palm_coord, 95.0)))
        contact_palm = float(np.percentile(palm_coord, float(contact_palm_percentile)))
        fruit_center = center_backoff * backoff_axis + center_jaw * jaw_axis + center_palm * palm_axis
        surface_target = (
            center_backoff * backoff_axis
            + center_jaw * jaw_axis
            + contact_palm * palm_axis
            + 0.5 * depth_m * backoff_axis
        )
        target_shift_m = float(np.linalg.norm(surface_target - selected))
        if target_shift_m > float(max_target_shift_m):
            surface_target = selected
            depth_m = prior_depth_m
            width_m = prior_width_m
            fruit_center = surface_target - 0.5 * depth_m * backoff_axis
            source = "clicked_point_fallback_large_rgbd_shift"
            contact_palm_shift_m = 0.0
        else:
            source = "live_rgbd_seeded_component"
            contact_palm_shift_m = float(contact_palm - center_palm)
        points_count = int(points.shape[0])

    contact_points = [
        (surface_target - 0.5 * width_m * jaw_axis).tolist(),
        (surface_target + 0.5 * width_m * jaw_axis).tolist(),
    ]
    return {
        "surface_target_robot_m": surface_target.tolist(),
        "fruit_center_robot_m": fruit_center.tolist(),
        "contact_points_robot_m": contact_points,
        "estimated_grasp_width_m": float(width_m),
        "estimated_grasp_depth_m": float(depth_m),
        "source": source,
        "selected_point_robot_m": selected.tolist(),
        "target_shift_from_click_m": float(np.linalg.norm(surface_target - selected)),
        "local_component_point_count": int(points_count),
        "contact_palm_percentile": float(contact_palm_percentile),
        "contact_palm_shift_m": float(locals().get("contact_palm_shift_m", 0.0)),
        "component": {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in component.items()
            if key != "points_robot_m"
        },
    }


def write_generic_harvest_task_frame(
    *,
    episode_dir: Path,
    action: dict[str, Any],
    surface_target_robot_m: list[float],
    primitive_prior: dict[str, Any],
    target_geometry: dict[str, Any] | None = None,
) -> Path:
    surface = [float(value) for value in surface_target_robot_m]
    backoff_axis = unit_vector3(primitive_prior["backoff_axis_robot"], name="harvest primitive backoff axis")
    jaw_axis = unit_vector3(primitive_prior["jaw_axis_robot"], name="harvest primitive jaw axis")
    palm_axis = unit_vector3(primitive_prior["palm_axis_robot"], name="harvest primitive palm axis")
    pull_axis = unit_vector3(primitive_prior["pull_axis_robot"], name="harvest primitive pull axis")

    geometry = target_geometry or {}
    depth_m = float(geometry.get("estimated_grasp_depth_m", primitive_prior["estimated_grasp_depth_m"]))
    width_m = float(geometry.get("estimated_grasp_width_m", primitive_prior["estimated_grasp_width_m"]))
    half_depth = 0.5 * float(depth_m)
    half_width = 0.5 * float(width_m)
    fruit_center = geometry.get("fruit_center_robot_m")
    if fruit_center is None:
        fruit_center = [surface[i] - half_depth * backoff_axis[i] for i in range(3)]
    else:
        fruit_center = [float(value) for value in fruit_center]
    contact_points = geometry.get("contact_points_robot_m")
    if contact_points is None:
        contact_points = [
            [surface[i] - half_width * jaw_axis[i] for i in range(3)],
            [surface[i] + half_width * jaw_axis[i] for i in range(3)],
        ]
    else:
        contact_points = [[float(part) for part in point] for point in contact_points]

    rotation_task = matrix_from_columns((backoff_axis, jaw_axis, palm_axis))
    tcp_x_axis = [-value for value in jaw_axis]
    tcp_z_axis = backoff_axis
    tcp_y_axis = unit_vector3(cross3(tcp_z_axis, tcp_x_axis), name="generic harvest tcp y axis")
    rotation_tcp = matrix_from_columns((tcp_x_axis, tcp_y_axis, tcp_z_axis))

    payload = {
        "version": "0.1",
        "method": "live_realsense_rgbd_geometry_mps_guided_harvest",
        "frame": "robot_base",
        "target_name": action["symbolic_target"],
        "target_index": int(action["step_index"]),
        "target_robot_m": surface,
        "fruit_center_robot_m": fruit_center,
        "R_robot_from_task": rotation_task,
        "task_axes_robot": {
            "plus_x_backoff_axis_robot": backoff_axis,
            "plus_y_jaw_axis_robot": jaw_axis,
            "plus_z_palm_axis_robot": palm_axis,
            "harvest_pull_axis_robot": pull_axis,
        },
        "grasp_orientation_quat_wxyz": quat_wxyz_from_rotation(rotation_tcp),
        "gripper_jaw_axis_robot": jaw_axis,
        "gripper_backoff_axis_robot": backoff_axis,
        "gripper_jaw_axis_local": "minus-x",
        "gripper_backoff_axis_local": "plus-z",
        "target_record": {
            "target_index": int(action["step_index"]),
            "target_position_semantics": (
                "live_rgbd_seeded_contact_midpoint"
                if target_geometry is not None
                else "live_realsense_clicked_visible_tomato_surface"
            ),
            "estimated_grasp_width_m": float(width_m),
            "estimated_grasp_depth_m": float(depth_m),
            "contact_points_robot_m": contact_points,
            "contact_midpoint_robot_m": surface,
            "gripper_jaw_axis_robot": jaw_axis,
            "gripper_backoff_axis_robot": backoff_axis,
            "recommended_pregrasp_aperture_m": float(width_m) + 0.025,
            "recommended_close_aperture_m": max(0.0, float(width_m) - 0.00625),
            "recommended_close_contact_margin_m": 0.00625,
            "contact_geometry_source": geometry.get("source", "class_level_robot_front_side_grasp_from_live_rgbd_click"),
            "live_target_geometry": geometry,
        },
        "quality": {
            "estimated_grasp_width_m": float(width_m),
            "estimated_grasp_depth_m": float(depth_m),
            "recommended_pregrasp_aperture_m": float(width_m) + 0.025,
            "recommended_close_aperture_m": max(0.0, float(width_m) - 0.00625),
            "recommended_close_contact_margin_m": 0.00625,
            "hand_frame_weak": True,
            "hand_frame_mode": primitive_prior["mode"],
            "aperture_source": "class_level_tomato_width",
            "contact_geometry_source": geometry.get(
                "source",
                "live_realsense_click_plus_mps_guided_harvest_primitive_prior",
            ),
            "primitive_prior": primitive_prior,
            "live_target_geometry": geometry,
            "warnings": [
                "Task frame is generated from live RGB-D target geometry and a primitive-level action prior for unseen tomato instances.",
            ],
        },
        "no_replay_contract": {
            "ok": True,
            "stored_from_hand_demo": ["action-frame axes", "class-level tomato grasp scale"],
            "recorded_trajectory_samples_used_for_robot_motion": False,
            "note": "This task frame is generated from the live RGB-D target and a primitive-level action frame; it does not encode a robot trajectory.",
        },
    }
    task_frame_dir = episode_dir / "generated_task_frames"
    task_frame_path = task_frame_dir / f"{action['step_id']}_generic_harvest_task_frame.json"
    write_json(task_frame_path, payload)
    return task_frame_path


def scene_setup_for_positions(
    *,
    leaf_positions: list[str] | tuple[str, ...],
    tomato_positions: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    tomato_clusters: dict[str, list[str]] = {}
    for label in tomato_positions:
        meta = label_metadata(label)
        if meta["primitive"] == "harvest":
            tomato_clusters.setdefault(meta["cluster"], []).append(label)
    return {
        "rod_A": {
            "deleaf_candidates_present": [label for label in leaf_positions if label.startswith("AD")],
            "tomato_clusters_present": tomato_clusters,
        },
        "rod_B": {
            "deleaf_candidates_present": [label for label in leaf_positions if label.startswith("BD")],
            "tomato_clusters_present": {},
        },
    }


def label_metadata(label: str) -> dict[str, Any]:
    if len(label) >= 3 and label.startswith("AD") and label[2:].isdigit():
        return {"label": label, "rod": "A", "primitive": "deleaf", "candidate_id": label[2:]}
    if len(label) >= 3 and label.startswith("BD") and label[2:].isdigit():
        return {"label": label, "rod": "B", "primitive": "deleaf", "candidate_id": label[2:]}
    harvest_match = re.fullmatch(r"AT(?P<cluster>\d+)T(?P<tomato>\d+)", label)
    if harvest_match:
        cluster = f"AT{harvest_match.group('cluster')}"
        return {
            "label": label,
            "rod": "A",
            "primitive": "harvest",
            "cluster": cluster,
            "tomato_id": f"T{harvest_match.group('tomato')}",
        }
    raise ValueError(f"unknown target label: {label}")


def episode_key(deleaf_sequence: list[str], harvest_target: str) -> str:
    return "deleaf=" + "+".join(deleaf_sequence) + "|harvest=" + harvest_target


def episode_leaf_positions(
    *,
    deleaf_sequence: list[str],
    candidate_positions: tuple[str, ...],
    physical_leaf_count: int,
    offset: int,
) -> list[str]:
    selected = list(dict.fromkeys(deleaf_sequence))
    if len(selected) > physical_leaf_count:
        raise ValueError(
            f"episode deleafs {len(selected)} unique targets but physical_leaf_count is {physical_leaf_count}"
        )
    fillers = [label for label in candidate_positions if label not in selected]
    needed = physical_leaf_count - len(selected)
    if needed > len(fillers):
        raise ValueError(
            f"cannot place {physical_leaf_count} leaves across {len(candidate_positions)} candidate positions"
        )
    if fillers:
        shift = offset % len(fillers)
        fillers = fillers[shift:] + fillers[:shift]
    selected.extend(fillers[:needed])
    selected_set = set(selected)
    return [label for label in candidate_positions if label in selected_set]


def episode_tomato_positions(
    *,
    harvest_target: str,
    tomato_targets: tuple[str, ...],
    physical_tomato_count: int,
    offset: int,
) -> list[str]:
    if physical_tomato_count < 1:
        raise ValueError("--physical-tomato-count must be at least 1")
    if physical_tomato_count > len(tomato_targets):
        raise ValueError("--physical-tomato-count cannot exceed the number of candidate tomato labels")
    selected = [harvest_target]
    fillers = [label for label in tomato_targets if label != harvest_target]
    needed = physical_tomato_count - 1
    if fillers:
        shift = offset % len(fillers)
        fillers = fillers[shift:] + fillers[:shift]
    selected.extend(fillers[:needed])
    selected_set = set(selected)
    return [label for label in tomato_targets if label in selected_set]


def build_episode_templates(
    deleaf_targets: tuple[str, ...],
    tomato_targets: tuple[str, ...],
    physical_leaf_count: int,
    physical_tomato_count: int,
) -> list[dict[str, Any]]:
    if physical_leaf_count < 1:
        raise ValueError("--physical-leaf-count must be at least 1")
    if physical_leaf_count > len(deleaf_targets):
        raise ValueError("--physical-leaf-count cannot exceed the number of candidate petiole labels")
    if physical_tomato_count < 1:
        raise ValueError("--physical-tomato-count must be at least 1")
    if physical_tomato_count > len(tomato_targets):
        raise ValueError("--physical-tomato-count cannot exceed the number of candidate tomato labels")
    templates: list[dict[str, Any]] = []
    if len(deleaf_targets) == 3 and len(tomato_targets) == 2:
        for sequence_length in range(1, len(deleaf_targets) + 1):
            for sequence in permutations(deleaf_targets, sequence_length):
                same_rod = len({label_metadata(label)["rod"] for label in sequence}) == 1
                for harvest_target in tomato_targets:
                    templates.append(
                        {
                            "deleaf_sequence": list(sequence),
                            "harvest_target": harvest_target,
                            "episode_type": "ordered_deleaf_same_rod" if same_rod else "ordered_deleaf_cross_rod",
                        }
                    )
    elif len(tomato_targets) == 2:
        for sequence in combinations(deleaf_targets, 2):
            same_rod = label_metadata(sequence[0])["rod"] == label_metadata(sequence[1])["rod"]
            for harvest_target in tomato_targets:
                templates.append(
                    {
                        "deleaf_sequence": list(sequence),
                        "harvest_target": harvest_target,
                        "episode_type": "multi_deleaf_same_rod" if same_rod else "multi_deleaf_cross_rod",
                    }
                )
    else:
        for deleaf_target in deleaf_targets:
            for harvest_target in tomato_targets:
                templates.append(
                    {
                        "deleaf_sequence": [deleaf_target],
                        "harvest_target": harvest_target,
                        "episode_type": "single_deleaf",
                    }
                )
    if len(tomato_targets) != 2 and tuple(deleaf_targets) == DEFAULT_DELEAF_TARGETS:
        multi_sequences = list(SAME_ROD_MULTI) + list(CROSS_ROD_MULTI)
        for index, sequence in enumerate(multi_sequences):
            templates.append(
                {
                    "deleaf_sequence": list(sequence),
                    "harvest_target": tomato_targets[index % len(tomato_targets)],
                    "episode_type": "multi_deleaf_same_rod" if index < len(SAME_ROD_MULTI) else "multi_deleaf_cross_rod",
                }
            )
    elif len(tomato_targets) != 2:
        for sequence in combinations(deleaf_targets, 2):
            same_rod = label_metadata(sequence[0])["rod"] == label_metadata(sequence[1])["rod"]
            for harvest_target in tomato_targets:
                templates.append(
                    {
                        "deleaf_sequence": list(sequence),
                        "harvest_target": harvest_target,
                        "episode_type": "multi_deleaf_same_rod" if same_rod else "multi_deleaf_cross_rod",
                    }
                )
    if len(templates) != 30:
        raise ValueError(
            f"expected 30 episode templates, got {len(templates)} from {len(deleaf_targets)} deleaf targets. "
            "Use six candidate petiole positions, or exactly three fixed labels with two harvest targets."
        )
    for index, template in enumerate(templates, start=1):
        template["episode_id"] = f"episode_{index:03d}"
        template["episode_key"] = episode_key(template["deleaf_sequence"], template["harvest_target"])
        template["symbolic_sequence"] = [*template["deleaf_sequence"], template["harvest_target"]]
        leaf_positions = episode_leaf_positions(
            deleaf_sequence=template["deleaf_sequence"],
            candidate_positions=deleaf_targets,
            physical_leaf_count=physical_leaf_count,
            offset=index - 1,
        )
        tomato_positions = episode_tomato_positions(
            harvest_target=template["harvest_target"],
            tomato_targets=tomato_targets,
            physical_tomato_count=physical_tomato_count,
            offset=index - 1,
        )
        template["leaf_positions_present"] = leaf_positions
        template["leaf_positions_absent"] = [label for label in deleaf_targets if label not in leaf_positions]
        template["tomato_positions_present"] = tomato_positions
        template["tomato_positions_absent"] = [label for label in tomato_targets if label not in tomato_positions]
        template["scene_setup"] = scene_setup_for_positions(
            leaf_positions=leaf_positions,
            tomato_positions=tomato_positions,
        )
    return templates


def build_dataset_manifest(
    dataset_name: str,
    testbed_id: str,
    deleaf_targets: tuple[str, ...],
    tomato_targets: tuple[str, ...],
    physical_leaf_count: int,
    physical_tomato_count: int,
) -> dict[str, Any]:
    labels = {label: label_metadata(label) for label in (*deleaf_targets, *tomato_targets)}
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "method": "egodelta_end_to_end_policy_episode_plan",
        "dataset_name": dataset_name,
        "testbed_id": testbed_id,
        "created_at_utc": utc_now(),
        "candidate_scene_setup": scene_setup_for_positions(
            leaf_positions=deleaf_targets,
            tomato_positions=tomato_targets,
        ),
        "target_labels": labels,
        "candidate_leaf_positions": list(deleaf_targets),
        "physical_leaf_count": int(physical_leaf_count),
        "candidate_tomato_positions": list(tomato_targets),
        "physical_tomato_count": int(physical_tomato_count),
        "episode_count": 30,
        "episode_uniqueness_key": "leaf_positions_present + tomato_positions_present + ordered_deleaf_sequence + harvest_target",
        "episodes": build_episode_templates(deleaf_targets, tomato_targets, physical_leaf_count, physical_tomato_count),
        "notes": [
            "Each episode executes one or more deleaf actions followed by exactly one harvest action.",
            "Candidate labels name fixed petiole positions; physical leaves can be moved between positions before each episode.",
            "Candidate tomato labels name fixed tomato positions; only tomato_positions_present should be installed for that episode.",
            "Each episode records present and absent leaf/tomato positions so the testbed can be reconstructed.",
            "Symbolic labels define episode uniqueness; clicked 3D coordinates are per-run measurements.",
            "Per-action artifacts should link clicked target, trajectory log, planner output, execution output, and outcome.",
        ],
    }


def action_records_for_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    step_index = 1
    for label in template["deleaf_sequence"]:
        meta = label_metadata(label)
        actions.append(
            {
                "step_index": step_index,
                "step_id": f"{template['episode_id']}_step_{step_index:02d}_{label}",
                "primitive": "deleaf",
                "symbolic_target": label,
                "rod": meta["rod"],
                "status": "planned",
                "outcome": None,
                "goal": {},
                "artifacts": {},
                "operator_notes": "",
            }
        )
        step_index += 1
    harvest_label = template["harvest_target"]
    harvest_meta = label_metadata(harvest_label)
    actions.append(
        {
            "step_index": step_index,
            "step_id": f"{template['episode_id']}_step_{step_index:02d}_{harvest_label}",
            "primitive": "harvest",
            "symbolic_target": harvest_label,
            "rod": harvest_meta["rod"],
            "cluster": harvest_meta["cluster"],
            "status": "planned",
            "outcome": None,
            "goal": {},
            "artifacts": {},
            "operator_notes": "",
        }
    )
    return actions


def build_episode_record(
    *,
    dataset_manifest: dict[str, Any],
    episode_id: str,
    episode_dir: Path,
) -> dict[str, Any]:
    matches = [item for item in dataset_manifest["episodes"] if item["episode_id"] == episode_id]
    if not matches:
        raise ValueError(f"episode_id not found in manifest: {episode_id}")
    template = matches[0]
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "method": "egodelta_end_to_end_policy_episode",
        "episode_id": episode_id,
        "episode_key": template["episode_key"],
        "dataset_name": dataset_manifest["dataset_name"],
        "testbed_id": dataset_manifest["testbed_id"],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "started",
        "episode_dir": str(episode_dir),
        "scene_setup": template.get("scene_setup", dataset_manifest.get("candidate_scene_setup", {})),
        "planned_task": {
            "episode_type": template["episode_type"],
            "leaf_positions_present": template.get("leaf_positions_present", []),
            "leaf_positions_absent": template.get("leaf_positions_absent", []),
            "tomato_positions_present": template.get("tomato_positions_present", []),
            "tomato_positions_absent": template.get("tomato_positions_absent", []),
            "deleaf_sequence": template["deleaf_sequence"],
            "harvest_target": template["harvest_target"],
            "symbolic_sequence": template["symbolic_sequence"],
        },
        "artifacts": {
            "dataset_manifest": None,
            "pre_scene": None,
            "post_scene": None,
        },
        "actions": action_records_for_template(template),
        "human_match_fields": {
            "intended_egocentric_episode_id": None,
            "aria_vrs": None,
            "mps_folder": None,
            "symbolic_sequence": template["symbolic_sequence"],
            "scene_reconstruction_notes": "",
        },
    }


def default_episodes_root_for_manifest(dataset_manifest: Path, *, runs_root: Path | None) -> Path:
    if runs_root is not None and Path(runs_root) != POLICY_RUNS_ROOT:
        return run_directory(TASK_NAME, runs_root=runs_root)
    return POLICY_EPISODES_ROOT


def capture_scene(
    *,
    episode: dict[str, Any],
    scene_name: str,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    serial: str | None,
    min_depth_m: float,
    max_depth_m: float,
    pointcloud_stride: int,
) -> dict[str, str]:
    import cv2
    import numpy as np

    from robot.kinematics.utils.capture_live_realsense import capture_realsense, build_point_cloud

    episode_dir = Path(episode["episode_dir"])
    capture = capture_realsense(
        width=width,
        height=height,
        fps=fps,
        warmup_frames=warmup_frames,
        serial=serial,
    )
    point_cloud = build_point_cloud(
        color_bgr=capture["color_bgr"],
        depth_m=capture["depth_m"],
        intrinsics=capture["intrinsics"],
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        stride=pointcloud_stride,
    )
    scene_prefix = f"{episode['episode_id']}_{scene_name}_scene"
    npz_path = episode_dir / f"{scene_prefix}.npz"
    jpg_path = episode_dir / f"{scene_prefix}.jpg"
    json_path = episode_dir / f"{scene_prefix}.json"
    episode_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        points=point_cloud["points"],
        colors=point_cloud["colors"],
        uv=point_cloud["uv"],
        depth_m=capture["depth_m"],
        color_bgr=capture["color_bgr"],
    )
    cv2.imwrite(str(jpg_path), capture["color_bgr"])
    metadata = {
        "method": "policy_episode_realsense_scene_capture",
        "episode_id": episode["episode_id"],
        "scene_name": scene_name,
        "captured_at_utc": utc_now(),
        "intrinsics": capture["intrinsics"],
        "depth_scale": float(capture["depth_scale"]),
        "frame_metadata": capture["frame_metadata"],
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
        "pointcloud_stride": int(pointcloud_stride),
        "point_count": int(point_cloud["points"].shape[0]),
    }
    write_json(json_path, metadata)
    return {"npz": str(npz_path), "jpg": str(jpg_path), "json": str(json_path)}


def extract_goal_from_artifact(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    goal: dict[str, Any] = {
        "source_artifact": str(path),
        "method": payload.get("method"),
        "run_label": payload.get("run_label"),
    }
    assignments = payload.get("assignments")
    if isinstance(assignments, list):
        passed = [item for item in assignments if isinstance(item, dict) and item.get("ok")]
        if passed:
            assignment = passed[0]
            live_target = (assignment.get("best_match") or {}).get("live_depth_target") or {}
            goal.update(
                {
                    "target_name": assignment.get("target_name"),
                    "target_index": assignment.get("target_index"),
                    "pixel_xy": live_target.get("pixel_uv"),
                    "point_realsense_m": live_target.get("point_realsense_m"),
                    "point_robot_m": live_target.get("point_robot_m"),
                    "selection_source": "live_target_identity_validation",
                    "assigned_detection_id": assignment.get("assigned_detection_id"),
                    "source_task_frame": assignment.get("source_path"),
                }
            )
    selected = payload.get("selected_point")
    if isinstance(selected, dict):
        goal.update(
            {
                "pixel_xy": selected.get("pixel_xy"),
                "point_realsense_m": selected.get("point_realsense_m"),
                "point_robot_m": selected.get("point_robot_m"),
                "selection_source": selected.get("selection_source"),
            }
        )
    surface_target = payload.get("surface_target_override_robot_m")
    if surface_target is not None:
        goal["point_robot_m"] = surface_target
        goal["selection_source"] = "harvest_surface_target_override"
    best_candidate = payload.get("best_candidate")
    if isinstance(best_candidate, dict):
        for waypoint in best_candidate.get("waypoints", []):
            if waypoint.get("name") in {"grasp", "pre_grasp"}:
                if "surface_target_robot_m" in waypoint and "point_robot_m" not in goal:
                    goal["point_robot_m"] = waypoint["surface_target_robot_m"]
                    goal["selection_source"] = "harvest_waypoint_surface_target"
                if "target_contact_position_robot_m" in waypoint:
                    goal["target_contact_position_robot_m"] = waypoint["target_contact_position_robot_m"]
                break
    return {key: value for key, value in goal.items() if value is not None}


def update_action_record(
    *,
    episode: dict[str, Any],
    primitive: str,
    target_label: str,
    action_artifact: Path | None,
    target_artifact: Path | None,
    planner_artifact: Path | None,
    trajectory_artifact: Path | None,
    execution_artifact: Path | None,
    outcome: str,
    notes: str,
    step_index: int | None,
    preserve_target_goal_on_planner: bool = False,
) -> dict[str, Any]:
    matches = [
        action
        for action in episode["actions"]
        if action["primitive"] == primitive
        and action["symbolic_target"] == target_label
        and (step_index is None or int(action["step_index"]) == int(step_index))
    ]
    if not matches:
        raise ValueError(f"no planned action matches primitive={primitive!r}, target={target_label!r}")
    pending = [action for action in matches if action.get("status") == "planned"]
    action = pending[0] if pending else matches[0]
    localized = localize_action_artifacts(
        episode=episode,
        action=action,
        action_artifact=action_artifact,
        target_artifact=target_artifact,
        planner_artifact=planner_artifact,
        trajectory_artifact=trajectory_artifact,
        execution_artifact=execution_artifact,
    )
    action_artifact = localized["action_artifact"]
    target_artifact = localized["target_artifact"]
    planner_artifact = localized["planner_artifact"]
    trajectory_artifact = localized["trajectory_artifact"]
    execution_artifact = localized["execution_artifact"]
    if action_artifact is not None:
        action["artifacts"]["action"] = str(action_artifact)
        action["goal"] = extract_goal_from_artifact(action_artifact)
    if target_artifact is not None:
        action["artifacts"]["target"] = str(target_artifact)
        action["goal"] = extract_goal_from_artifact(target_artifact)
    if planner_artifact is not None:
        action["artifacts"]["planner"] = str(planner_artifact)
        planner_goal = extract_goal_from_artifact(planner_artifact)
        if preserve_target_goal_on_planner and action.get("goal"):
            action["fixed_plan_goal"] = planner_goal
            action["fixed_plan_surface_target_robot_m"] = planner_goal.get("point_robot_m")
            action["fixed_plan_target_contact_position_robot_m"] = planner_goal.get("target_contact_position_robot_m")
            planner_goal = {}
        current_goal = dict(action.get("goal", {}))
        if "point_robot_m" in current_goal and "point_robot_m" in planner_goal:
            current_goal.setdefault("clicked_point_robot_m", current_goal["point_robot_m"])
            current_goal.setdefault("clicked_selection_source", current_goal.get("selection_source"))
            current_goal["planner_surface_target_robot_m"] = planner_goal["point_robot_m"]
            current_goal["point_robot_m"] = planner_goal["point_robot_m"]
            current_goal["selection_source"] = "harvest_live_rgbd_derived_contact_target"
        for key, value in planner_goal.items():
            if key == "point_robot_m" and "point_robot_m" in current_goal:
                continue
            current_goal[key] = value
        action["goal"] = current_goal
    if trajectory_artifact is not None:
        action["artifacts"]["trajectory"] = str(trajectory_artifact)
    if execution_artifact is not None:
        action["artifacts"]["execution"] = str(execution_artifact)
    action["outcome"] = outcome
    action["status"] = "recorded"
    action["operator_notes"] = notes
    action["updated_at_utc"] = utc_now()
    episode["updated_at_utc"] = utc_now()
    if all(item.get("status") == "recorded" for item in episode["actions"]):
        episode["status"] = "actions_recorded"
    return action


def command_init_dataset(args: argparse.Namespace) -> int:
    deleaf_targets = parse_deleaf_targets(args.deleaf_targets)
    tomato_targets = parse_harvest_targets(args.harvest_targets)
    manifest = build_dataset_manifest(
        args.dataset_name,
        args.testbed_id,
        deleaf_targets,
        tomato_targets,
        int(args.physical_leaf_count),
        int(args.physical_tomato_count),
    )
    output = args.output or POLICY_DATASET_ROOT / f"{args.dataset_name}_{args.testbed_id}_30_episode_plan.json"
    write_json(output, manifest)
    print(json.dumps({"manifest": str(output), "episode_count": manifest["episode_count"]}, indent=2))
    return 0


def command_start_episode(args: argparse.Namespace) -> int:
    dataset_manifest = read_json(args.dataset_manifest)
    episodes_root = args.episodes_root or default_episodes_root_for_manifest(
        args.dataset_manifest,
        runs_root=args.runs_root,
    )
    episode_dir = Path(episodes_root) / args.episode_id
    episode_json = episode_dir / "episode.json"
    if episode_json.exists() and not args.overwrite:
        raise FileExistsError(f"episode already exists: {episode_json}")
    episode = build_episode_record(
        dataset_manifest=dataset_manifest,
        episode_id=args.episode_id,
        episode_dir=episode_dir,
    )
    episode["artifacts"]["dataset_manifest"] = str(args.dataset_manifest)
    if args.capture_pre_scene:
        episode["artifacts"]["pre_scene"] = capture_scene(
            episode=episode,
            scene_name="pre",
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            serial=args.serial,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            pointcloud_stride=args.pointcloud_stride,
        )
    write_json(episode_json, episode)
    print(json.dumps({"episode": str(episode_json), "episode_dir": str(episode_dir)}, indent=2))
    return 0


def parse_optional_label_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    labels = [part.strip() for part in value.split(",") if part.strip()]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"duplicate target label(s): {', '.join(duplicates)}")
    for label in labels:
        label_metadata(label)
    return labels


def command_start_action_instance(args: argparse.Namespace) -> int:
    target_label = str(args.target_label)
    meta = label_metadata(target_label)
    primitive = str(args.primitive or meta["primitive"])
    if primitive != meta["primitive"]:
        raise ValueError(f"target {target_label} is a {meta['primitive']} target, not {primitive}")
    run_label = args.run_label or f"{target_label}_{time.strftime('%m%d_%H%M%S')}"
    action_dir = Path(args.action_library_root) / target_label / run_label
    episode_json = action_dir / "episode.json"
    if episode_json.exists() and not args.overwrite:
        raise FileExistsError(f"action instance already exists: {episode_json}")

    leaf_positions = parse_optional_label_list(args.context_leaf_positions)
    tomato_positions = parse_optional_label_list(args.context_tomato_positions)
    if primitive == "deleaf" and target_label not in leaf_positions:
        leaf_positions.insert(0, target_label)
    if primitive == "harvest" and target_label not in tomato_positions:
        tomato_positions.insert(0, target_label)
    for label in leaf_positions:
        if label_metadata(label)["primitive"] != "deleaf":
            raise ValueError(f"context leaf label is not a deleaf target: {label}")
    for label in tomato_positions:
        if label_metadata(label)["primitive"] != "harvest":
            raise ValueError(f"context tomato label is not a harvest target: {label}")

    action: dict[str, Any] = {
        "step_index": 1,
        "step_id": f"{run_label}_step_01_{target_label}",
        "primitive": primitive,
        "symbolic_target": target_label,
        "rod": meta["rod"],
        "status": "planned",
        "outcome": None,
        "operator_notes": "",
        "artifacts": {},
        "goal": {},
    }
    if "cluster" in meta:
        action["cluster"] = meta["cluster"]
    episode = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "method": "egodelta_atomic_policy_action_instance",
        "dataset_name": args.dataset_name,
        "testbed_id": args.testbed_id,
        "episode_id": run_label,
        "episode_key": f"atomic_action={target_label}|run={run_label}",
        "episode_dir": str(action_dir),
        "status": "started",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "planned_task": {
            "symbolic_sequence": [target_label],
            "leaf_positions_present": leaf_positions,
            "tomato_positions_present": tomato_positions,
            "leaf_positions_absent": [],
            "tomato_positions_absent": [],
            "episode_type": "atomic_action_instance",
        },
        "human_match_fields": {
            "symbolic_sequence": [target_label],
            "intended_egocentric_episode_id": None,
            "aria_vrs": None,
            "mps_folder": None,
            "scene_reconstruction_notes": "",
        },
        "scene_setup": scene_setup_for_positions(
            leaf_positions=leaf_positions,
            tomato_positions=tomato_positions,
        ),
        "artifacts": {
            "dataset_manifest": None,
            "pre_scene": None,
            "post_scene": None,
        },
        "actions": [action],
    }
    if args.capture_pre_scene:
        episode["artifacts"]["pre_scene"] = capture_scene(
            episode=episode,
            scene_name="pre",
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            serial=args.serial,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            pointcloud_stride=args.pointcloud_stride,
        )
    write_json(episode_json, episode)
    print(json.dumps({"action_instance": str(episode_json), "action_dir": str(action_dir)}, indent=2))
    return 0


def command_capture_scene(args: argparse.Namespace) -> int:
    episode_json = Path(args.episode_json)
    episode = read_json(episode_json)
    paths = capture_scene(
        episode=episode,
        scene_name=args.scene,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        serial=args.serial,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        pointcloud_stride=args.pointcloud_stride,
    )
    episode["artifacts"][f"{args.scene}_scene"] = paths
    episode["updated_at_utc"] = utc_now()
    write_json(episode_json, episode)
    print(json.dumps({"episode": str(episode_json), "scene": args.scene, "artifacts": paths}, indent=2))
    return 0


def command_record_action(args: argparse.Namespace) -> int:
    episode_json = Path(args.episode_json)
    episode = read_json(episode_json)
    action = update_action_record(
        episode=episode,
        primitive=args.primitive,
        target_label=args.target_label,
        action_artifact=args.action_artifact,
        target_artifact=args.target_artifact,
        planner_artifact=args.planner_artifact,
        trajectory_artifact=args.trajectory_artifact,
        execution_artifact=args.execution_artifact,
        outcome=args.outcome,
        notes=args.notes,
        step_index=args.step_index,
    )
    write_json(episode_json, episode)
    print(json.dumps({"episode": str(episode_json), "recorded_action": action}, indent=2, sort_keys=True))
    return 0


def json_objects_from_output(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        objects.append(value)
    return objects


def run_command_tee(command: list[str], *, cwd: str | Path) -> str:
    print("\n$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=None,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return_code = process.wait()
    output = "".join(lines)
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")
    return output


def path_from_last_payload(output: str, *keys: str) -> Path:
    for value in reversed(json_objects_from_output(output)):
        current: Any = value
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current:
            return Path(str(current))
    if len(keys) == 1 and keys[0] == "artifact_path":
        matches = re.findall(r"\bartifact=\s*([^\r\n]+?\.json)\b", output)
        if matches:
            return Path(matches[-1].strip())
    raise RuntimeError(f"could not find JSON path field: {'.'.join(keys)}")


def artifact_paths_from_output(output: str) -> list[Path]:
    paths: list[Path] = []
    for value in json_objects_from_output(output):
        if isinstance(value, dict) and value.get("artifact_path"):
            paths.append(Path(str(value["artifact_path"])))
    paths.extend(Path(match.strip()) for match in re.findall(r"\bartifact=\s*([^\r\n]+?\.json)\b", output))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def harvest_planner_artifact_from_output(output: str) -> Path:
    paths = artifact_paths_from_output(output)
    if not paths:
        return path_from_last_payload(output, "artifact_path")

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if path.exists():
            loaded.append((path, read_json(path)))
    for path, payload in reversed(loaded):
        if not payload.get("hard_gate", {}).get("ok", False):
            continue
        best_candidate = payload.get("best_candidate")
        waypoints = best_candidate.get("waypoints", []) if isinstance(best_candidate, dict) else []
        if any(waypoint.get("name") == "basket_place" for waypoint in waypoints):
            return path
    for path, payload in reversed(loaded):
        if payload.get("hard_gate", {}).get("ok", False):
            return path
    return paths[-1]


def harvest_no_scoop_report(
    payload: dict[str, Any],
    *,
    min_preclose_z_margin_m: float,
    waypoint_names: tuple[str, ...] = ("pre_grasp", "grasp", "harvest"),
) -> dict[str, Any]:
    target = payload.get("target", {})
    if not isinstance(target, dict):
        target = {}
    identity = payload.get("identity_validation", {})
    if not isinstance(identity, dict):
        identity = {}
    surface = (
        identity.get("surface_target_robot_m")
        or target.get("target_position_robot_m")
        or payload.get("surface_target_override_robot_m")
    )
    if not isinstance(surface, list) or len(surface) != 3:
        return {
            "ok": False,
            "reason": "missing target surface position for no-scoop guard",
            "min_preclose_z_margin_m": float(min_preclose_z_margin_m),
        }
    surface_z = float(surface[2])
    best_candidate = payload.get("best_candidate")
    waypoints = best_candidate.get("waypoints", []) if isinstance(best_candidate, dict) else []
    reports: list[dict[str, Any]] = []
    for name in waypoint_names:
        waypoint = next((item for item in waypoints if item.get("name") == name), None)
        if waypoint is None:
            reports.append({"name": name, "ok": False, "reason": "missing_waypoint"})
            continue
        contact = waypoint.get("target_contact_position_robot_m")
        if not isinstance(contact, list) or len(contact) != 3:
            reports.append({"name": name, "ok": False, "reason": "missing_contact_position"})
            continue
        margin = float(contact[2]) - surface_z
        reports.append(
            {
                "name": name,
                "ok": bool(margin >= float(min_preclose_z_margin_m)),
                "contact_z_m": float(contact[2]),
                "surface_z_m": surface_z,
                "z_margin_m": margin,
            }
        )
    ok = all(bool(item.get("ok", False)) for item in reports)
    return {
        "ok": bool(ok),
        "reason": None if ok else "contact_below_live_side_contact_line",
        "surface_z_m": surface_z,
        "min_preclose_z_margin_m": float(min_preclose_z_margin_m),
        "waypoints": reports,
    }


def harvest_grasp_alignment_report(payload: dict[str, Any]) -> dict[str, Any]:
    best_candidate = payload.get("best_candidate")
    waypoints = best_candidate.get("waypoints", []) if isinstance(best_candidate, dict) else []
    candidate_gate = best_candidate.get("hard_gate", {}) if isinstance(best_candidate, dict) else {}
    reports: list[dict[str, Any]] = []
    for name in ("pre_grasp", "grasp"):
        waypoint = next((item for item in waypoints if item.get("name") == name), None)
        if waypoint is None:
            reports.append({"name": name, "ok": False, "reason": "missing_waypoint"})
            continue
        endpoint_gate = waypoint.get("endpoint_gate", {})
        ik = waypoint.get("ik_result", {})
        raw_ok = bool(endpoint_gate.get("ok", False))
        effective_ok = bool(endpoint_gate.get("effective_ok", raw_ok))
        effective_mode = endpoint_gate.get("effective_mode")
        axis_gate_key = "pregrasp_task_axis_gate" if name == "pre_grasp" else "task_axis_grasp_gate"
        axis_gate = candidate_gate.get(axis_gate_key, {})
        strict_axis_ok = (
            effective_ok
            and effective_mode == ("task_axis_pregrasp" if name == "pre_grasp" else "task_axis_grasp")
            and bool(axis_gate.get("ok", False))
        )
        ok = bool(raw_ok or strict_axis_ok)
        reports.append(
            {
                "name": name,
                "ok": ok,
                "raw_endpoint_ok": raw_ok,
                "effective_endpoint_ok": effective_ok,
                "effective_mode": effective_mode,
                "strict_axis_ok": strict_axis_ok,
                "task_axis_gate": axis_gate,
                "position_error_m": ik.get("position_error_m"),
                "orientation_error_rad": ik.get("orientation_error_rad"),
                "reason": None if ok else "neither_full_pose_nor_strict_task_axis_satisfied",
            }
        )
    ok = all(bool(item.get("ok", False)) for item in reports)
    return {
        "ok": bool(ok),
        "reason": None if ok else "harvest_grasp_requires_full_pose_alignment",
        "waypoints": reports,
    }


def harvest_grasp_physical_quality_report(
    payload: dict[str, Any],
    *,
    min_insertion_m: float,
    max_backoff_error_rad: float,
) -> dict[str, Any]:
    best_candidate = payload.get("best_candidate")
    if not isinstance(best_candidate, dict):
        return {"ok": False, "reason": "missing_best_candidate"}
    quality = best_candidate.get("manipulation_quality", {})
    if not isinstance(quality, dict):
        quality = {}
    gate = best_candidate.get("hard_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    task_axis_gate = gate.get("task_axis_grasp_gate", {})
    if not isinstance(task_axis_gate, dict):
        task_axis_gate = {}
    insertion = quality.get("actual_grasp_insertion_m")
    if insertion is None:
        insertion = quality.get("grasp_insertion_m")
    backoff_error = (
        quality.get("task_axis_grasp_backoff_axis_error_rad")
        if quality.get("task_axis_grasp_backoff_axis_error_rad") is not None
        else task_axis_gate.get("backoff_axis_error_rad")
    )
    insertion_ok = insertion is not None and float(insertion) >= float(min_insertion_m)
    backoff_ok = backoff_error is not None and float(backoff_error) <= float(max_backoff_error_rad)
    return {
        "ok": bool(insertion_ok and backoff_ok),
        "reason": None if insertion_ok and backoff_ok else "shallow_or_angled_grasp",
        "actual_grasp_insertion_m": None if insertion is None else float(insertion),
        "min_insertion_m": float(min_insertion_m),
        "insertion_ok": bool(insertion_ok),
        "backoff_axis_error_rad": None if backoff_error is None else float(backoff_error),
        "max_backoff_axis_error_rad": float(max_backoff_error_rad),
        "backoff_axis_ok": bool(backoff_ok),
    }


def safe_filename_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return stem or "candidate"


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def selected_candidate_hard_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    gate = candidate.get("hard_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    return {
        "ok": bool(gate.get("ok", False)),
        "best_endpoint_ok": bool(gate.get("endpoint_ok", False)),
        "best_path_clearance_ok": bool(gate.get("path_clearance_ok", False)),
        "best_joint_delta_ok": bool(gate.get("joint_delta_ok", False)),
        "best_open_alignment_ok": bool(gate.get("open_alignment_ok", False)),
        "best_open_position_ok": bool(gate.get("open_position_ok", False)),
        "best_open_orientation_ok": bool(gate.get("open_orientation_ok", False)),
        "best_grasp_palm_plausibility_ok": bool(gate.get("grasp_palm_plausibility_ok", False)),
        "best_grasp_palm_alignment_ok": bool(gate.get("grasp_palm_alignment_ok", False)),
        "best_grasp_depth_ok": bool(gate.get("grasp_depth_ok", False)),
        "best_task_axis_grasp_ok": bool(gate.get("task_axis_grasp_ok", False)),
        "best_pregrasp_surface_clearance_ok": bool(gate.get("pregrasp_surface_clearance_ok", False)),
        "best_scene_obstacle_clearance_ok": bool(gate.get("scene_obstacle_clearance_ok", False)),
        "best_tail_forward_reach_ok": bool(gate.get("tail_forward_reach_ok", False)),
    }


def archive_top_harvest_candidates(
    *,
    planner_payload: dict[str, Any],
    stored_planner: Path,
    output_dir: Path,
    top_k: int,
    min_preclose_z_margin_m: float,
    no_scoop_waypoints: tuple[str, ...],
    min_insertion_m: float,
    max_backoff_error_rad: float,
) -> dict[str, Any]:
    passing_candidates = [
        candidate
        for candidate in planner_payload.get("candidates", [])
        if isinstance(candidate, dict) and bool(candidate.get("hard_gate", {}).get("ok", False))
    ]
    archive_dir = output_dir / "candidate_solves"
    archived: list[dict[str, Any]] = []
    for rank, candidate in enumerate(passing_candidates[: max(0, int(top_k))], start=1):
        variant = {key: value for key, value in planner_payload.items() if key != "candidates"}
        variant["best_candidate"] = candidate
        variant["best_candidate_name"] = candidate.get("name")
        variant["candidates"] = [candidate]
        variant["hard_gate"] = selected_candidate_hard_gate(candidate)
        variant["candidate_archive"] = {
            "rank": rank,
            "status": "solver_solved_unconfirmed",
            "source_planner_artifact": str(stored_planner),
            "source_candidate_count": len(planner_payload.get("candidates", [])),
            "note": "This candidate passed solver gates but still needs hardware visual confirmation before reuse.",
        }
        variant["no_scoop_guard"] = harvest_no_scoop_report(
            variant,
            min_preclose_z_margin_m=min_preclose_z_margin_m,
            waypoint_names=no_scoop_waypoints,
        )
        variant["grasp_alignment_guard"] = harvest_grasp_alignment_report(variant)
        variant["grasp_physical_quality_guard"] = harvest_grasp_physical_quality_report(
            variant,
            min_insertion_m=min_insertion_m,
            max_backoff_error_rad=max_backoff_error_rad,
        )
        candidate_name = safe_filename_stem(str(candidate.get("name", f"candidate_{rank:02d}")))
        out_path = archive_dir / f"{rank:02d}_{candidate_name}.json"
        variant["artifact_path"] = str(out_path)
        write_json(out_path, variant)
        quality = candidate.get("manipulation_quality", {})
        if not isinstance(quality, dict):
            quality = {}
        archived.append(
            {
                "rank": rank,
                "candidate_name": candidate.get("name"),
                "artifact": str(out_path),
                "hard_gate_ok": bool(candidate.get("hard_gate", {}).get("ok", False)),
                "no_scoop_ok": bool(variant["no_scoop_guard"].get("ok", False)),
                "alignment_ok": bool(variant["grasp_alignment_guard"].get("ok", False)),
                "physical_quality_ok": bool(variant["grasp_physical_quality_guard"].get("ok", False)),
                "actual_grasp_insertion_m": optional_float(quality.get("actual_grasp_insertion_m")),
                "task_axis_grasp_position_error_m": optional_float(quality.get("task_axis_grasp_position_error_m")),
                "task_axis_grasp_jaw_error_rad": optional_float(quality.get("task_axis_grasp_jaw_axis_error_rad")),
                "task_axis_grasp_backoff_error_rad": optional_float(quality.get("task_axis_grasp_backoff_axis_error_rad")),
            }
        )
    index_path = archive_dir / "top_solved_candidates.json"
    index_payload = {
        "source_planner_artifact": str(stored_planner),
        "requested_top_k": int(top_k),
        "passing_candidate_count": len(passing_candidates),
        "archived_candidate_count": len(archived),
        "status": "solver_solved_unconfirmed",
        "candidates": archived,
    }
    write_json(index_path, index_payload)
    return {
        "directory": str(archive_dir),
        "index": str(index_path),
        "requested_top_k": int(top_k),
        "passing_candidate_count": len(passing_candidates),
        "archived_candidate_count": len(archived),
        "candidates": archived,
    }


def point_robot_from_capture(output: str) -> tuple[list[float], Path]:
    for value in reversed(json_objects_from_output(output)):
        if not isinstance(value, dict):
            continue
        selected = value.get("selected_point")
        artifacts = value.get("artifacts")
        if isinstance(selected, dict) and isinstance(artifacts, dict):
            point = selected.get("point_robot_m")
            path = artifacts.get("json")
            if point is not None and path is not None:
                return [float(item) for item in point], Path(str(path))
    raise RuntimeError("target capture did not return selected_point.point_robot_m and artifacts.json")


def next_planned_action(episode: dict[str, Any], step_index: int | None) -> dict[str, Any]:
    actions = episode["actions"]
    if step_index is not None:
        matches = [action for action in actions if int(action["step_index"]) == int(step_index)]
        if not matches:
            raise ValueError(f"no action has step_index={step_index}")
        return matches[0]
    for action in actions:
        if action.get("status") != "recorded":
            return action
    raise RuntimeError("all actions are already recorded")


def trajectory_args_for_action(
    args: argparse.Namespace,
    *,
    action: dict[str, Any],
    arm: str,
    plan: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        address=args.address,
        model=args.model,
        arm=arm,
        dataset_name=args.dataset_name,
        toolfit=None,
        duration_s=float(args.trajectory_duration_s),
        hz=float(args.trajectory_hz),
        manual_start=False,
        manual_stop=False,
        run_label=f"{action['step_id']}_traj",
        runs_root=args.runs_root,
        plan=plan,
        candidate_name=None,
        waypoint="grasp",
    )


def run_command_with_trajectory(
    command: list[str],
    *,
    args: argparse.Namespace,
    episode: dict[str, Any],
    action: dict[str, Any],
    arm: str,
    plan: Path | None = None,
) -> tuple[str, Path]:
    from robot.policy import capture_robot_tcp_trajectory as trajectory

    stop_event = threading.Event()
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            capture_args = trajectory_args_for_action(args, action=action, arm=arm, plan=plan)
            payload = trajectory.capture(capture_args, stop_event=stop_event)
            output = action_episode_dir(episode, action) / "trajectory" / (
                f"{capture_args.dataset_name}_{capture_args.run_label}.json"
            )
            write_json(output, payload)
            result["artifact"] = output
            result["sample_count"] = payload["sample_count"]
        except Exception as exc:
            result["error"] = repr(exc)

    thread = threading.Thread(target=worker, daemon=True)
    print(f"\nStarting {arm}-arm trajectory logger for {action['step_id']}...")
    thread.start()
    time.sleep(float(args.trajectory_start_delay_s))
    if not thread.is_alive() and "artifact" not in result:
        if "error" in result:
            raise RuntimeError(f"trajectory logger stopped before the action command started: {result['error']}")
        raise RuntimeError("trajectory logger stopped before the action command started")
    try:
        output = run_command_tee(command, cwd=_REPO_ROOT)
    finally:
        stop_event.set()
        thread.join(timeout=float(args.trajectory_stop_timeout_s))
    if thread.is_alive():
        raise RuntimeError("trajectory logger did not stop cleanly")
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    artifact = result.get("artifact")
    if not artifact:
        raise RuntimeError("trajectory logger did not write an artifact")
    print(f"Trajectory artifact: {artifact}")
    return output, Path(artifact)


def run_command_sequence_with_trajectory(
    commands: list[list[str]],
    *,
    args: argparse.Namespace,
    episode: dict[str, Any],
    action: dict[str, Any],
    arm: str,
    plan: Path | None = None,
) -> tuple[list[str], Path]:
    from robot.policy import capture_robot_tcp_trajectory as trajectory

    stop_event = threading.Event()
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            capture_args = trajectory_args_for_action(args, action=action, arm=arm, plan=plan)
            payload = trajectory.capture(capture_args, stop_event=stop_event)
            output = action_episode_dir(episode, action) / "trajectory" / (
                f"{capture_args.dataset_name}_{capture_args.run_label}.json"
            )
            write_json(output, payload)
            result["artifact"] = output
            result["sample_count"] = payload["sample_count"]
        except Exception as exc:
            result["error"] = repr(exc)

    thread = threading.Thread(target=worker, daemon=True)
    print(f"\nStarting {arm}-arm trajectory logger for {action['step_id']} command sequence...")
    thread.start()
    time.sleep(float(args.trajectory_start_delay_s))
    if not thread.is_alive() and "artifact" not in result:
        if "error" in result:
            raise RuntimeError(f"trajectory logger stopped before the action command started: {result['error']}")
        raise RuntimeError("trajectory logger stopped before the action command started")
    outputs: list[str] = []
    try:
        for index, command in enumerate(commands, start=1):
            print(f"\nRunning sequenced motion command {index}/{len(commands)}...")
            outputs.append(run_command_tee(command, cwd=_REPO_ROOT))
    finally:
        stop_event.set()
        thread.join(timeout=float(args.trajectory_stop_timeout_s))
    if thread.is_alive():
        raise RuntimeError("trajectory logger did not stop cleanly")
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    artifact = result.get("artifact")
    if not artifact:
        raise RuntimeError("trajectory logger did not write an artifact")
    print(f"Trajectory artifact: {artifact}")
    return outputs, Path(artifact)


def latest_action_target_artifact(episode: dict[str, Any], action: dict[str, Any]) -> Path | None:
    target_dir = action_episode_dir(episode, action) / "target"
    if not target_dir.exists():
        return None
    candidates = [
        path
        for path in target_dir.glob("*.json")
        if path.is_file() and path.parent.name == "target"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def set_cli_arg(command: list[str], option: str, value: str) -> None:
    for index, item in enumerate(command):
        if item == option:
            if index + 1 >= len(command):
                raise RuntimeError(f"CLI option has no value slot: {option}")
            command[index + 1] = value
            return
        if item.startswith(option + "="):
            command[index] = f"{option}={value}"
            return
    raise RuntimeError(f"CLI option not found in command: {option}")


def confirm_live_action(action: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.execute:
        raise RuntimeError("run-next-action requires --execute for live robot motion")
    if args.yes:
        return
    response = input(
        f"Ready to run step {action['step_index']} ({action['primitive']} {action['symbolic_target']})? "
        "Type RUN and press Enter: "
    )
    if response.strip().upper() != "RUN":
        raise RuntimeError("operator did not confirm RUN")


def restore_home(args: argparse.Namespace, *, reason: str) -> dict[str, Any]:
    restore_grippers = bool(getattr(args, "home_restore_grippers", False) or getattr(args, "home_close_grippers", False))
    gripper_restore_mode = "close" if bool(getattr(args, "home_close_grippers", False)) else str(
        getattr(args, "home_gripper_restore_mode", "saved")
    )
    command = [
        sys.executable,
        "robot/utils/restore_home_state.py",
        "--scope",
        args.home_scope,
        "--minimum-time",
        str(args.home_minimum_time_s),
        "--timeout-s",
        str(args.home_timeout_s),
        "--execute",
    ]
    if restore_grippers:
        command.extend(
            [
                "--restore-grippers",
                "--gripper-restore-mode",
                gripper_restore_mode,
                "--gripper-current-limit",
                str(args.home_gripper_current_limit),
                "--gripper-hold-s",
                str(args.home_gripper_hold_s),
            ]
        )
    output = run_command_tee(command, cwd=_REPO_ROOT)
    return {
        "reason": reason,
        "command": command,
        "completed_at_utc": utc_now(),
        "output_tail": output[-1000:],
    }


def restore_home_before_action(args: argparse.Namespace, action: dict[str, Any]) -> None:
    if not args.home_before_action:
        return
    info = restore_home(args, reason=f"before_step_{action['step_index']}_{action['primitive']}")
    action.setdefault("artifacts", {})["pre_action_home"] = info
    action["updated_at_utc"] = utc_now()


def restore_home_after_action(args: argparse.Namespace, action: dict[str, Any]) -> None:
    if not args.home_after_action:
        return
    info = restore_home(args, reason=f"after_step_{action['step_index']}_{action['primitive']}")
    action.setdefault("artifacts", {})["post_action_home"] = {
        **info,
    }
    action["updated_at_utc"] = utc_now()


def command_run_next_action(args: argparse.Namespace) -> int:
    episode_json = Path(args.episode_json)
    episode = read_json(episode_json)
    action = next_planned_action(episode, args.step_index)
    confirm_live_action(action, args)
    if args.step_index is not None and action.get("status") == "recorded":
        action["status"] = "planned"
        action["outcome"] = None
        action["operator_notes"] = "cleared before operator-confirmed rerun"
        action["updated_at_utc"] = utc_now()
        episode["status"] = "started"
        episode["updated_at_utc"] = utc_now()
    restore_home_before_action(args, action)
    write_json(episode_json, episode)

    python = sys.executable
    if action["primitive"] == "deleaf":
        command = [
            python,
            "robot/kinematics/apps/deleaf_click_cut.py",
            "--deleaf-config",
            "robot/kinematics/configs/deleaf_right_knife_default.json",
            "--run-label",
            action["step_id"],
            "--execute",
            "--no-execute-key-confirm",
            "--execute-until",
            str(args.deleaf_execute_until),
            "--cut-time-s",
            str(args.deleaf_cut_time_s),
        ]
        if args.deleaf_mps_auto_select_action:
            # Fresh per-cut collection: the nearest-demonstrated-cut hand-tracking guidance
            # supplies the cut orientation, so we DO NOT apply the old per-label side
            # overrides (side axis / teleop prior / side_perpendicular). Only generic
            # standoff/tolerance overrides pass through.
            command += [
                "--manifest", str(args.deleaf_manifest),
                "--alignment", str(args.deleaf_alignment),
                "--hand-tracking", str(args.deleaf_hand_tracking),
                "--slam-trajectory", str(args.deleaf_slam_trajectory),
                "--mps-auto-select-action",
                # Tucked-posture + straight-vertical + snap motion validated live
                # (elbow tucked toward torso, world_down descent, fast plunge).
                "--home", str(args.deleaf_home),
                "--posture-weight", str(args.deleaf_posture_weight),
                "--stroke-axis-policy", str(args.deleaf_stroke_axis_policy),
                "--max-stroke-tilt-deg", str(args.deleaf_max_stroke_tilt_deg),
                "--pre-cut-time-s", str(args.deleaf_pre_cut_time_s),
                "--right-velocity-limit-rad-s", str(args.deleaf_right_velocity_limit_rad_s),
                "--right-acceleration-limit-rad-s2", str(args.deleaf_right_acceleration_limit_rad_s2),
            ]
            if args.deleaf_mps_approach_path_guidance:
                command.append("--mps-approach-path-guidance")
            # Per-cut MPS defaults: cut exactly at the click (no sideways extension) and
            # give the blade orientation room (the human cut angle, not forced vertical).
            mps_target_extension = args.deleaf_target_extension_m if args.deleaf_target_extension_m is not None else 0.0
            mps_orientation_tol = args.deleaf_orientation_tolerance_rad if args.deleaf_orientation_tolerance_rad is not None else 0.30
            mps_generic_overrides = (
                ("--pre-cut-standoff-m", args.deleaf_pre_cut_standoff_m),
                ("--target-extension-m", mps_target_extension),
                ("--target-extension-values-m", args.deleaf_target_extension_values_m),
                ("--contact-advance-mode", args.deleaf_contact_advance_mode),
                ("--pointcloud-contact-advance-values-m", args.deleaf_pointcloud_contact_advance_values_m),
                ("--transit-lift-m", args.deleaf_transit_lift_m),
                ("--safe-entry-standoff-m", args.deleaf_safe_entry_standoff_m),
                ("--safe-entry-lift-m", args.deleaf_safe_entry_lift_m),
                ("--return-transit-lift-m", args.deleaf_return_transit_lift_m),
                ("--retract-lift-m", args.deleaf_retract_lift_m),
                ("--retract-standoff-m", args.deleaf_retract_standoff_m),
                ("--pointcloud-min-clearance-m", args.deleaf_pointcloud_min_clearance_m),
                ("--position-tolerance-m", args.deleaf_position_tolerance_m),
                ("--orientation-tolerance-rad", mps_orientation_tol),
            )
            for flag, value in mps_generic_overrides:
                if value is not None:
                    command.extend([flag, str(value)])
            if args.deleaf_pointcloud_candidate_angles_deg is not None:
                command.append(f"--pointcloud-candidate-angles-deg={args.deleaf_pointcloud_candidate_angles_deg}")
        else:
            target_label = str(action["symbolic_target"])
            label_side_axis = DELEAF_SIDE_AXIS_ROBOT.get(target_label)
            label_prior_source_match = DELEAF_SIDE_PRIOR_SOURCE_MATCH.get(target_label)
            label_safe_entry_axis = DELEAF_SIDE_SAFE_ENTRY_AXIS_ROBOT.get(target_label)
            effective_side_axis = args.deleaf_contact_advance_axis_robot or label_side_axis
            effective_prior_json = args.deleaf_teleop_prior_json or (
                DELEAF_SIDE_PRIOR_JSON if label_prior_source_match is not None else None
            )
            effective_prior_source_match = args.deleaf_teleop_prior_source_match or label_prior_source_match
            effective_waypoint_policy = args.deleaf_waypoint_policy or (
                "teleop_offset" if effective_prior_json is not None else None
            )
            effective_safe_entry_axis = args.deleaf_safe_entry_axis_robot or label_safe_entry_axis
            effective_orientation_policy = args.deleaf_task_orientation_policy
            if effective_orientation_policy is None and effective_side_axis is not None:
                effective_orientation_policy = "side_perpendicular"
            deleaf_overrides = (
                ("--waypoint-policy", effective_waypoint_policy),
                ("--teleop-prior-json", effective_prior_json),
                ("--teleop-prior-source-match", effective_prior_source_match),
                ("--pre-cut-standoff-m", args.deleaf_pre_cut_standoff_m),
                ("--target-extension-m", args.deleaf_target_extension_m),
                ("--target-extension-values-m", args.deleaf_target_extension_values_m),
                ("--contact-advance-mode", args.deleaf_contact_advance_mode),
                ("--contact-advance-axis-robot", effective_side_axis),
                ("--task-orientation-policy", effective_orientation_policy),
                ("--pointcloud-contact-advance-values-m", args.deleaf_pointcloud_contact_advance_values_m),
                ("--transit-lift-m", args.deleaf_transit_lift_m),
                ("--safe-entry-standoff-m", args.deleaf_safe_entry_standoff_m),
                ("--safe-entry-axis-robot", effective_safe_entry_axis),
                ("--safe-entry-lift-m", args.deleaf_safe_entry_lift_m),
                ("--return-transit-lift-m", args.deleaf_return_transit_lift_m),
                ("--retract-lift-m", args.deleaf_retract_lift_m),
                ("--retract-standoff-m", args.deleaf_retract_standoff_m),
                ("--pointcloud-min-clearance-m", args.deleaf_pointcloud_min_clearance_m),
                ("--position-tolerance-m", args.deleaf_position_tolerance_m),
                ("--orientation-tolerance-rad", args.deleaf_orientation_tolerance_rad),
            )
            for flag, value in deleaf_overrides:
                if value is not None:
                    text = str(value)
                    if flag in {"--contact-advance-axis-robot", "--safe-entry-axis-robot"} and text.startswith("-"):
                        command.append(f"{flag}={text}")
                    else:
                        command.extend([flag, text])
            if args.deleaf_pointcloud_candidate_angles_deg is not None:
                command.append(f"--pointcloud-candidate-angles-deg={args.deleaf_pointcloud_candidate_angles_deg}")
            if args.deleaf_use_teleop_joint_seeds:
                command.append("--use-teleop-joint-seeds")
        if args.deleaf_mps_auto_select_action and args.deleaf_mps_restore_home:
            # Seat both arms at the deleaf home (right arm tucked) so deleaf_click_cut's
            # whole-robot home check passes and the cut starts from the tucked posture.
            restore_command = [
                python,
                "robot/utils/restore_home_state.py",
                "--home", str(args.deleaf_home),
                "--scope", "arms",
                "--minimum-time", str(args.home_minimum_time_s),
                "--timeout-s", str(args.home_timeout_s),
                "--execute",
            ]
            restore_output = run_command_tee(restore_command, cwd=_REPO_ROOT)
            action.setdefault("artifacts", {})["pre_action_deleaf_home"] = {
                "reason": f"before_step_{action['step_index']}_deleaf_mps",
                "home": str(args.deleaf_home),
                "command": restore_command,
                "completed_at_utc": utc_now(),
                "output_tail": restore_output[-500:],
            }
        output, trajectory_artifact = run_command_with_trajectory(
            command,
            args=args,
            episode=episode,
            action=action,
            arm="right",
        )
        action_artifact = path_from_last_payload(output, "artifacts", "json")
        action_payload = read_json(action_artifact)
        if not action_payload.get("motion", {}).get("executed", False):
            reason = action_payload.get("motion", {}).get("reason", "unknown")
            raise RuntimeError(
                f"deleaf command did not execute robot motion; not recording success. "
                f"Artifact: {action_artifact}. Reason: {reason}"
            )
        updated = update_action_record(
            episode=episode,
            primitive="deleaf",
            target_label=action["symbolic_target"],
            action_artifact=action_artifact,
            target_artifact=None,
            planner_artifact=None,
            trajectory_artifact=trajectory_artifact,
            execution_artifact=None,
            outcome=args.outcome,
            notes=args.notes,
            step_index=int(action["step_index"]),
        )
        restore_home_after_action(args, updated)
        write_json(episode_json, episode)
        print(json.dumps({"episode": str(episode_json), "recorded_action": updated}, indent=2, sort_keys=True))
        return 0

    if action["primitive"] != "harvest":
        raise RuntimeError(f"unsupported primitive: {action['primitive']}")

    harvest_profile = harvest_profile_settings(action, args)
    action["harvest_profile"] = harvest_profile
    print(
        "Using harvest profile: "
        f"requested={harvest_profile['requested']} "
        f"resolved={harvest_profile['resolved']} "
        f"description={harvest_profile['description']} "
        f"insertions={harvest_profile['grasp_insertion_factors']} "
        f"orientation_weight={harvest_profile['full_orientation_weight']} "
        f"tcp_rolls_deg={harvest_profile['tcp_rolls_deg']} "
        f"lateral_offsets_m={harvest_profile['lateral_offsets_m']} "
        f"grasp_prior={harvest_profile['grasp_prior_capture_json']}"
    )

    if args.harvest_replay_trajectory_json is not None:
        source_trajectory = Path(args.harvest_replay_trajectory_json)
        event_plan = Path(args.harvest_replay_event_plan_json) if args.harvest_replay_event_plan_json is not None else None
        target_artifact = (
            Path(args.harvest_replay_target_artifact)
            if args.harvest_replay_target_artifact is not None
            else latest_action_target_artifact(episode, action)
        )
        execution_artifact = Path(episode["episode_dir"]) / f"{action['step_id']}_execution.json"
        replay_command = [
            python,
            "robot/policy/replay_arm_joint_trajectory.py",
            str(source_trajectory),
            "--arm",
            "left",
            "--execute",
            "--output-json",
            str(execution_artifact),
            "--stride",
            str(args.harvest_replay_stride),
            "--minimum-time-s",
            str(args.harvest_replay_minimum_time_s),
            "--entry-exit-jitter-deg",
            str(args.harvest_replay_entry_exit_jitter_deg),
        ]
        if args.harvest_replay_variation_seed is not None:
            replay_command.extend(["--variation-seed", str(args.harvest_replay_variation_seed)])
        if event_plan is not None:
            replay_command.extend(["--event-plan-json", str(event_plan)])
        if not args.harvest_no_gripper:
            replay_command.append("--use-gripper")
        execute_output, trajectory_artifact = run_command_with_trajectory(
            replay_command,
            args=args,
            episode=episode,
            action=action,
            arm="left",
            plan=event_plan,
        )
        _ = execute_output
        updated = update_action_record(
            episode=episode,
            primitive="harvest",
            target_label=action["symbolic_target"],
            action_artifact=None,
            target_artifact=target_artifact,
            planner_artifact=event_plan,
            trajectory_artifact=trajectory_artifact,
            execution_artifact=execution_artifact,
            outcome=args.outcome,
            notes=args.notes or f"replayed harvest trajectory {source_trajectory}",
            step_index=int(action["step_index"]),
        )
        restore_home_after_action(args, updated)
        write_json(episode_json, episode)
        print(json.dumps({"episode": str(episode_json), "recorded_action": updated}, indent=2, sort_keys=True))
        return 0

    fixed_plan_source, fixed_plan_source_kind = harvest_fixed_plan_for_action(action, args)
    if fixed_plan_source is not None:
        source_planner_artifact = Path(fixed_plan_source)
        source_plan_metadata = validate_fixed_harvest_plan(source_planner_artifact, action)
        target_artifact, _identity_task_frame = capture_harvest_identity_target(
            python=python,
            action=action,
            args=args,
        )
        if target_artifact is None and args.harvest_target_source == "click":
            _point_robot_m, target_artifact = capture_harvest_click_target(
                python=python,
                action=action,
                args=args,
            )
        if target_artifact is None:
            target_path_value = action.get("artifacts", {}).get("target")
            target_artifact = Path(target_path_value) if isinstance(target_path_value, str) else latest_action_target_artifact(episode, action)
        planner_artifact = fixed_harvest_plan_variant(
            source_plan=source_planner_artifact,
            episode=episode,
            action=action,
            entry_exit_jitter_deg=float(args.harvest_fixed_plan_entry_exit_jitter_deg),
            seed=args.harvest_fixed_plan_variation_seed,
        )
        executed_plan_metadata = validate_fixed_harvest_plan(planner_artifact, action)
        print(
            "Using fixed full-tail harvest plan: "
            f"target={action['symbolic_target']} "
            f"source={fixed_plan_source_kind} "
            f"candidate={executed_plan_metadata.get('candidate_name')} "
            f"artifact={planner_artifact}"
        )
        execution_artifact = Path(episode["episode_dir"]) / f"{action['step_id']}_execution.json"
        execute_output, trajectory_artifact = run_command_with_trajectory(
            [
                python,
                "robot/kinematics/apps/execute_harvest_plan.py",
                str(planner_artifact),
                "--execute",
                "--output-json",
                str(execution_artifact),
            ]
            + ([] if args.harvest_no_gripper else ["--use-gripper"]),
            args=args,
            episode=episode,
            action=action,
            arm="left",
            plan=planner_artifact,
        )
        _ = execute_output
        updated = update_action_record(
            episode=episode,
            primitive="harvest",
            target_label=action["symbolic_target"],
            action_artifact=None,
            target_artifact=target_artifact,
            planner_artifact=planner_artifact,
            trajectory_artifact=trajectory_artifact,
            execution_artifact=execution_artifact,
            outcome=args.outcome,
            notes=args.notes or f"executed fixed full-tail harvest plan {planner_artifact}",
            step_index=int(action["step_index"]),
            preserve_target_goal_on_planner=target_artifact is not None,
        )
        updated["collection_mode"] = "fixed_full_tail_plan_reuse"
        updated["planner_mode"] = "fixed_plan_reuse"
        updated["live_replanning_performed"] = False
        updated["live_target_logged"] = target_artifact is not None
        updated["fixed_harvest_plan"] = {
            "source_kind": fixed_plan_source_kind,
            "source": source_plan_metadata,
            "executed": executed_plan_metadata,
            "executed_plan_artifact": updated.get("artifacts", {}).get("planner"),
            "critical_waypoints": ["pre_grasp", "grasp", "harvest", "basket_place"],
        }
        updated["data_use_flags"] = {
            "valid_for_imitation": True,
            "valid_for_adaptive_harvest_planner_training": False,
            "split_group": f"fixed_harvest_plan:{action['symbolic_target']}",
        }
        restore_home_after_action(args, updated)
        write_json(episode_json, episode)
        print(json.dumps({"episode": str(episode_json), "recorded_action": updated}, indent=2, sort_keys=True))
        return 0

    identity_validation_artifact: Path | None = None
    point_arg: str | None = None
    task_frame_path: Path
    target_artifact: Path
    identity_task_frame = harvest_identity_task_frame(action, args)
    use_identity_target = (
        args.harvest_target_source == "identity"
        or (args.harvest_target_source == "auto" and identity_task_frame is not None and identity_task_frame.exists())
    )
    if use_identity_target:
        if identity_task_frame is None or not identity_task_frame.exists():
            raise RuntimeError(
                f"no calibrated identity task frame exists for harvest target {action['symbolic_target']}; "
                "use --harvest-target-source click to force clicked RGB-D targeting"
            )
        validation_output = run_command_tee(
            [
                python,
                "robot/kinematics/apps/validate_live_target_identity.py",
                "--dataset-name",
                args.dataset_name,
                "--runs-root",
                str(args.runs_root),
                "--run-label",
                f"{action['step_id']}_identity",
                "--task-frame-jsons",
                str(identity_task_frame),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fps",
                str(args.fps),
                "--warmup-frames",
                str(args.warmup_frames),
                "--min-depth-m",
                str(args.min_depth_m),
                "--max-depth-m",
                str(args.max_depth_m),
                "--pointcloud-stride",
                str(args.pointcloud_stride),
            ],
            cwd=_REPO_ROOT,
        )
        identity_validation_artifact = path_from_last_payload(validation_output, "artifacts", "json")
        validation_payload = read_json(identity_validation_artifact)
        if not validation_payload.get("ok", False):
            raise RuntimeError(f"live harvest target identity validation failed: {identity_validation_artifact}")
        target_artifact = identity_validation_artifact
        task_frame_path = identity_task_frame
        action.setdefault("artifacts", {})["task_frame"] = str(task_frame_path)
        goal = extract_goal_from_artifact(identity_validation_artifact)
        print(
            "Using live target identity validation: "
            f"task_frame={task_frame_path} "
            f"target={goal.get('target_name')} "
            f"surface="
            + ",".join(f"{value:.6f}" for value in goal.get("point_robot_m", []))
        )
    else:
        point_robot_m, target_artifact = capture_harvest_click_target(
            python=python,
            action=action,
            args=args,
        )
        target_bias = parse_vector3_arg(harvest_profile["target_bias_robot_m"], name="harvest target bias")
        biased_point_robot_m = [float(value) + float(delta) for value, delta in zip(point_robot_m, target_bias)]
        if any(abs(value) > 1e-9 for value in target_bias):
            print(
                "Applying harvest target bias: raw="
                + ",".join(f"{value:.6f}" for value in point_robot_m)
                + " bias="
                + ",".join(f"{value:.6f}" for value in target_bias)
                + " biased="
                    + ",".join(f"{value:.6f}" for value in biased_point_robot_m)
            )
        primitive_prior = harvest_primitive_prior_from_task_frames(
            prior_dir=args.harvest_primitive_prior_dir,
            pattern=args.harvest_primitive_prior_pattern,
            fallback_depth_m=float(args.harvest_generic_depth_m),
            fallback_width_m=float(args.harvest_generic_width_m),
        )
        target_geometry = None
        planner_surface_target_robot_m = biased_point_robot_m
        if not args.harvest_disable_live_target_geometry:
            target_geometry = derive_live_harvest_target_geometry(
                target_artifact=target_artifact,
                selected_point_robot_m=biased_point_robot_m,
                primitive_prior=primitive_prior,
                local_radius_px=int(args.harvest_local_radius_px),
                local_depth_window_m=float(args.harvest_local_depth_window_m),
                local_color_threshold=float(args.harvest_local_color_threshold),
                local_radius_m=float(args.harvest_local_radius_m),
                min_points=int(args.harvest_min_local_points),
                max_target_shift_m=float(args.harvest_max_derived_target_shift_m),
                contact_palm_percentile=float(harvest_profile["contact_palm_percentile"]),
            )
            planner_surface_target_robot_m = [
                float(value) for value in target_geometry["surface_target_robot_m"]
            ]
            print(
                "Derived harvest target from live RGB-D geometry: "
                f"source={target_geometry['source']} "
                f"component_points={target_geometry['local_component_point_count']} "
                f"shift_from_click_m={target_geometry['target_shift_from_click_m']:.3f} "
                f"palm_shift_m={target_geometry['contact_palm_shift_m']:.3f} "
                "surface="
                + ",".join(f"{value:.6f}" for value in planner_surface_target_robot_m)
            )
        point_arg = ",".join(f"{value:.6f}" for value in planner_surface_target_robot_m)
        task_frame_path = write_generic_harvest_task_frame(
            episode_dir=episode_json.parent,
            action=action,
            surface_target_robot_m=planner_surface_target_robot_m,
            primitive_prior=primitive_prior,
            target_geometry=target_geometry,
        )
        action.setdefault("artifacts", {})["task_frame"] = str(task_frame_path)
        print(
            "Using MPS-guided generic harvest frame: "
            f"mode={primitive_prior['mode']} "
            f"source_count={primitive_prior['source_count']} "
            f"depth_m={primitive_prior['estimated_grasp_depth_m']:.3f} "
            f"width_m={primitive_prior['estimated_grasp_width_m']:.3f} "
            f"path={task_frame_path}"
        )
    grasp_insertion_factors = str(harvest_profile["grasp_insertion_factors"])
    if args.harvest_grasp_insertion_factor is not None:
        grasp_insertion_factors = str(args.harvest_grasp_insertion_factor)
    if args.harvest_contact_frame_json is None:
        print(
            f"Using harvest TCP: pure URDF jaw-center toolfit={args.harvest_toolfit} "
            f"palm_correction_factors={harvest_profile['palm_correction_factors']}"
        )
    else:
        print(
            f"Using harvest TCP: URDF jaw-center toolfit={args.harvest_toolfit} "
            f"with contact-frame correction={args.harvest_contact_frame_json} "
            f"palm_correction_factors={harvest_profile['palm_correction_factors']}"
        )
    planner_command = [
        python,
        "robot/kinematics/apps/plan_harvest_staged.py",
        str(task_frame_path),
        "--run-label",
        action["step_id"],
        "--toolfit",
        str(args.harvest_toolfit),
        "--disable-scene-obstacle-clearance",
        "--full-orientation-weight",
        str(harvest_profile["full_orientation_weight"]),
        "--harvest-distance-m",
        str(harvest_profile["harvest_distance_m"]),
        "--harvest-down-m",
        str(harvest_profile["harvest_down_m"]),
        "--grasp-insertion-factors",
        grasp_insertion_factors,
        "--max-grasp-insertion-factor",
        str(harvest_profile["max_grasp_insertion_factor"]),
        "--palm-correction-factors",
        str(harvest_profile["palm_correction_factors"]),
        "--min-grasp-insertion-fraction",
        str(harvest_profile["min_grasp_insertion_fraction"]),
        f"--lateral-offsets-m={harvest_profile['lateral_offsets_m']}",
        f"--elevated-offsets-m={harvest_profile['elevated_offsets_m']}",
        f"--tcp-rolls-deg={harvest_profile['tcp_rolls_deg']}",
        "--task-axis-pregrasp-gate",
        "--pregrasp-task-axis-position-tolerance-m",
        str(harvest_profile["pregrasp_task_position_tolerance_m"]),
        "--pregrasp-task-jaw-axis-tolerance-rad",
        str(harvest_profile["pregrasp_task_jaw_tolerance_rad"]),
        "--pregrasp-task-backoff-axis-tolerance-rad",
        str(harvest_profile["pregrasp_task_backoff_tolerance_rad"]),
        "--task-axis-grasp-gate",
        "--grasp-task-axis-position-tolerance-m",
        str(harvest_profile["grasp_task_position_tolerance_m"]),
        "--grasp-task-jaw-axis-tolerance-rad",
        str(harvest_profile["grasp_task_jaw_tolerance_rad"]),
        "--grasp-task-backoff-axis-tolerance-rad",
        str(harvest_profile["grasp_task_backoff_tolerance_rad"]),
        "--grasp-palm-alignment-tolerance-m",
        str(args.harvest_grasp_palm_alignment_tolerance_m),
        "--path-samples",
        "10",
    ]
    if not args.harvest_fixed_tail_after_grasp:
        planner_command.extend(
            [
                "--include-tail",
                "--tail-prefix-count",
                "3",
                f"--basket-anchor-robot-m={args.harvest_basket_anchor_robot_m}",
            ]
        )
    if identity_validation_artifact is not None:
        planner_command.extend(["--identity-validation-json", str(identity_validation_artifact)])
    else:
        if point_arg is None:
            raise RuntimeError("internal error: missing clicked harvest surface target")
        planner_command.extend(["--surface-target-robot-m", point_arg])
    if args.harvest_contact_frame_json is not None:
        planner_command.extend(["--contact-frame-json", str(args.harvest_contact_frame_json)])
    grasp_prior_capture_json = harvest_profile.get("grasp_prior_capture_json")
    if grasp_prior_capture_json is not None:
        planner_command.extend(
            [
                "--grasp-prior-capture-json",
                str(grasp_prior_capture_json),
                "--grasp-prior-weight",
                str(harvest_profile["grasp_prior_weight"]),
            ]
        )
    planner_output = run_command_tee(planner_command, cwd=_REPO_ROOT)
    planner_artifact = harvest_planner_artifact_from_output(planner_output)
    planner_payload = read_json(planner_artifact)
    no_scoop_waypoints = ("pre_grasp", "grasp") if args.harvest_fixed_tail_after_grasp else ("pre_grasp", "grasp", "harvest")
    no_scoop = harvest_no_scoop_report(
        planner_payload,
        min_preclose_z_margin_m=float(args.harvest_min_preclose_z_margin_m),
        waypoint_names=no_scoop_waypoints,
    )
    planner_payload["no_scoop_guard"] = no_scoop
    alignment = harvest_grasp_alignment_report(planner_payload)
    planner_payload["grasp_alignment_guard"] = alignment
    write_json(planner_artifact, planner_payload)
    if not no_scoop["ok"] and args.harvest_scoop_safe_retry:
        print(
            "No-scoop guard rejected the first harvest plan; retrying with scoop-safe "
            "palm/down policy: palm_correction_factors=0.0, harvest_distance_m=0.0, harvest_down_m=0.0"
        )
        retry_command = list(planner_command)
        set_cli_arg(retry_command, "--run-label", f"{action['step_id']}_scoop_safe")
        set_cli_arg(retry_command, "--palm-correction-factors", "0.0")
        set_cli_arg(retry_command, "--harvest-distance-m", "0.0")
        set_cli_arg(retry_command, "--harvest-down-m", "0.0")
        planner_output = run_command_tee(retry_command, cwd=_REPO_ROOT)
        planner_artifact = harvest_planner_artifact_from_output(planner_output)
        planner_payload = read_json(planner_artifact)
        no_scoop = harvest_no_scoop_report(
            planner_payload,
            min_preclose_z_margin_m=float(args.harvest_min_preclose_z_margin_m),
            waypoint_names=no_scoop_waypoints,
        )
        planner_payload["no_scoop_guard"] = no_scoop
        alignment = harvest_grasp_alignment_report(planner_payload)
        planner_payload["grasp_alignment_guard"] = alignment
        write_json(planner_artifact, planner_payload)
    best_candidate = planner_payload.get("best_candidate")
    best_waypoints = best_candidate.get("waypoints", []) if isinstance(best_candidate, dict) else []
    waypoint_names = [
        waypoint.get("name")
        for waypoint in best_waypoints
    ]
    if (
        planner_payload.get("hard_gate", {}).get("ok", False)
        and "basket_place" not in waypoint_names
        and not args.harvest_fixed_tail_after_grasp
    ):
        raise RuntimeError(
            "harvest planner returned a passing plan without basket_place; refusing retract-only harvest. "
            f"Artifact: {planner_artifact}. Waypoints: {waypoint_names}"
        )
    if not planner_payload.get("hard_gate", {}).get("ok", False):
        best_name = planner_payload.get("best_candidate_name")
        hard_gate = planner_payload.get("hard_gate", {})
        raise RuntimeError(
            "harvest planner did not find a valid executable plan; not executing robot motion. "
            f"Artifact: {planner_artifact}. "
            f"best_candidate={best_name} "
            f"best_endpoint_ok={hard_gate.get('best_endpoint_ok')} "
            f"best_path_clearance_ok={hard_gate.get('best_path_clearance_ok')} "
            f"best_scene_obstacle_clearance_ok={hard_gate.get('best_scene_obstacle_clearance_ok')}"
        )
    if not no_scoop["ok"]:
        raise RuntimeError(
            "harvest planner selected a bottom-scoop pre-close motion; not executing robot motion. "
            f"Artifact: {planner_artifact}. Guard: {json.dumps(no_scoop, sort_keys=True)}"
        )
    if not alignment["ok"]:
        raise RuntimeError(
            "harvest planner selected a relaxed or angled grasp; not executing robot motion. "
            f"Artifact: {planner_artifact}. Guard: {json.dumps(alignment, sort_keys=True)}"
        )
    execution_artifact = Path(episode["episode_dir"]) / f"{action['step_id']}_execution.json"
    if args.harvest_fixed_tail_after_grasp:
        fixed_tail_plan = Path(args.harvest_fixed_tail_plan_json)
        if not fixed_tail_plan.exists():
            raise RuntimeError(f"fixed harvest tail plan does not exist: {fixed_tail_plan}")
        fixed_tail_payload = read_json(fixed_tail_plan)
        fixed_tail_candidate = fixed_tail_payload.get("best_candidate")
        fixed_tail_waypoints = (
            fixed_tail_candidate.get("waypoints", []) if isinstance(fixed_tail_candidate, dict) else []
        )
        fixed_tail_waypoint_names = [waypoint.get("name") for waypoint in fixed_tail_waypoints]
        missing_tail_waypoints = [
            name for name in ("grasp", "harvest", "retract", "basket_place") if name not in fixed_tail_waypoint_names
        ]
        if missing_tail_waypoints:
            raise RuntimeError(
                f"fixed harvest tail plan is missing required waypoints {missing_tail_waypoints}: {fixed_tail_plan}"
            )
        prefix_execution_artifact = Path(episode["episode_dir"]) / f"{action['step_id']}_prefix_execution.json"
        fixed_tail_execution_artifact = Path(episode["episode_dir"]) / f"{action['step_id']}_fixed_tail_execution.json"
        gripper_args = [] if args.harvest_no_gripper else ["--use-gripper"]
        prefix_command = [
            python,
            "robot/kinematics/apps/execute_harvest_plan.py",
            str(planner_artifact),
            "--execute",
            "--stop-after",
            "grasp",
            "--allow-prefix-diagnostic",
            "--output-json",
            str(prefix_execution_artifact),
        ] + gripper_args
        fixed_tail_command = [
            python,
            "robot/kinematics/apps/execute_harvest_plan.py",
            str(fixed_tail_plan),
            "--execute",
            "--start-after",
            "grasp",
            "--skip-resume-check",
            "--output-json",
            str(fixed_tail_execution_artifact),
        ] + gripper_args
        execute_outputs, trajectory_artifact = run_command_sequence_with_trajectory(
            [prefix_command, fixed_tail_command],
            args=args,
            episode=episode,
            action=action,
            arm="left",
            plan=planner_artifact,
        )
        execution_payload = {
            "mode": "live_grasp_prefix_plus_fixed_basket_tail",
            "planner_artifact": str(planner_artifact),
            "fixed_tail_plan": str(fixed_tail_plan),
            "fixed_tail_plan_sha256": sha256_file(fixed_tail_plan),
            "prefix_execution_artifact": str(prefix_execution_artifact),
            "fixed_tail_execution_artifact": str(fixed_tail_execution_artifact),
            "prefix_execution": read_json(prefix_execution_artifact),
            "fixed_tail_execution": read_json(fixed_tail_execution_artifact),
            "command_count": len(execute_outputs),
            "completed_at_utc": utc_now(),
        }
        write_json(execution_artifact, execution_payload)
    else:
        execute_output, trajectory_artifact = run_command_with_trajectory(
            [
                python,
                "robot/kinematics/apps/execute_harvest_plan.py",
                str(planner_artifact),
                "--execute",
                "--output-json",
                str(execution_artifact),
            ]
            + ([] if args.harvest_no_gripper else ["--use-gripper"]),
            args=args,
            episode=episode,
            action=action,
            arm="left",
            plan=planner_artifact,
        )
        _ = execute_output
    updated = update_action_record(
        episode=episode,
        primitive="harvest",
        target_label=action["symbolic_target"],
        action_artifact=None,
        target_artifact=target_artifact,
        planner_artifact=planner_artifact,
        trajectory_artifact=trajectory_artifact,
        execution_artifact=execution_artifact,
        outcome=args.outcome,
        notes=args.notes,
        step_index=int(action["step_index"]),
    )
    if args.harvest_fixed_tail_after_grasp:
        updated["collection_mode"] = "live_grasp_prefix_fixed_basket_tail"
        updated["planner_mode"] = "live_grasp_prefix_only"
        updated["fixed_harvest_tail"] = {
            "plan_artifact": str(Path(args.harvest_fixed_tail_plan_json)),
            "plan_sha256": sha256_file(Path(args.harvest_fixed_tail_plan_json)),
            "start_after": "grasp",
            "resume_check_skipped": True,
        }
    restore_home_after_action(args, updated)
    write_json(episode_json, episode)
    print(json.dumps({"episode": str(episode_json), "recorded_action": updated}, indent=2, sort_keys=True))
    return 0


def command_solve_harvest_target(args: argparse.Namespace) -> int:
    args = apply_harvest_arg_defaults(args)
    label = str(args.target_label).upper()
    if label not in HARVEST_TARGETS:
        raise ValueError(f"unknown harvest target label: {label}; expected one of {', '.join(HARVEST_TARGETS)}")
    run_label = str(args.run_label or f"{label}_solve_{datetime.now().strftime('%H%M%S')}")
    output_dir = Path(args.output_root) / label / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    action = {
        "primitive": "harvest",
        "symbolic_target": label,
        "step_id": run_label,
        "step_index": int(HARVEST_TARGETS.index(label) + 1),
        "artifacts": {},
    }
    python = sys.executable
    harvest_profile = harvest_profile_settings(action, args)
    print(
        "Solving harvest target without robot execution: "
        f"target={label} run_label={run_label} "
        f"profile={harvest_profile['resolved']} "
        f"insertions={harvest_profile['grasp_insertion_factors']} "
        f"orientation_weight={harvest_profile['full_orientation_weight']} "
        f"tcp_rolls_deg={harvest_profile['tcp_rolls_deg']} "
        f"lateral_offsets_m={harvest_profile['lateral_offsets_m']} "
        f"include_tail={bool(args.include_tail)}"
    )

    identity_validation_artifact: Path | None = None
    point_arg: str | None = None
    target_artifact: Path
    identity_task_frame = harvest_identity_task_frame(action, args)
    use_identity_target = (
        args.harvest_target_source == "identity"
        or (args.harvest_target_source == "auto" and identity_task_frame is not None and identity_task_frame.exists())
    )
    if use_identity_target:
        if identity_task_frame is None or not identity_task_frame.exists():
            raise RuntimeError(
                f"no calibrated identity task frame exists for harvest target {label}; "
                "use --harvest-target-source click"
            )
        validation_output = run_command_tee(
            [
                python,
                "robot/kinematics/apps/validate_live_target_identity.py",
                "--dataset-name",
                args.dataset_name,
                "--runs-root",
                str(args.runs_root),
                "--run-label",
                f"{run_label}_identity",
                "--task-frame-jsons",
                str(identity_task_frame),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fps",
                str(args.fps),
                "--warmup-frames",
                str(args.warmup_frames),
                "--min-depth-m",
                str(args.min_depth_m),
                "--max-depth-m",
                str(args.max_depth_m),
                "--pointcloud-stride",
                str(args.pointcloud_stride),
            ],
            cwd=_REPO_ROOT,
        )
        identity_validation_artifact = path_from_last_payload(validation_output, "artifacts", "json")
        validation_payload = read_json(identity_validation_artifact)
        if not validation_payload.get("ok", False):
            raise RuntimeError(f"live harvest target identity validation failed: {identity_validation_artifact}")
        target_artifact = identity_validation_artifact
        task_frame_path = identity_task_frame
        goal = extract_goal_from_artifact(identity_validation_artifact)
        point_arg = ",".join(f"{float(value):.6f}" for value in goal.get("point_robot_m", []))
    else:
        target_output = run_command_tee(
            [
                python,
                "robot/kinematics/utils/realsense_click_move_arm.py",
                "--run-label",
                f"{run_label}_target",
                "--arm",
                "left",
                "--dataset-name",
                args.dataset_name,
                "--toolfit",
                str(args.harvest_toolfit),
                "--pick-mode",
                args.harvest_target_picker,
                "--max-y-m",
                "0.55",
                "--max-z-m",
                "1.35",
            ],
            cwd=_REPO_ROOT,
        )
        point_robot_m, target_artifact = point_robot_from_capture(target_output)
        target_bias = parse_vector3_arg(harvest_profile["target_bias_robot_m"], name="harvest target bias")
        biased_point_robot_m = [float(value) + float(delta) for value, delta in zip(point_robot_m, target_bias)]
        primitive_prior = harvest_primitive_prior_from_task_frames(
            prior_dir=args.harvest_primitive_prior_dir,
            pattern=args.harvest_primitive_prior_pattern,
            fallback_depth_m=float(args.harvest_generic_depth_m),
            fallback_width_m=float(args.harvest_generic_width_m),
        )
        target_geometry = None
        planner_surface_target_robot_m = biased_point_robot_m
        if not args.harvest_disable_live_target_geometry:
            target_geometry = derive_live_harvest_target_geometry(
                target_artifact=target_artifact,
                selected_point_robot_m=biased_point_robot_m,
                primitive_prior=primitive_prior,
                local_radius_px=int(args.harvest_local_radius_px),
                local_depth_window_m=float(args.harvest_local_depth_window_m),
                local_color_threshold=float(args.harvest_local_color_threshold),
                local_radius_m=float(args.harvest_local_radius_m),
                min_points=int(args.harvest_min_local_points),
                max_target_shift_m=float(args.harvest_max_derived_target_shift_m),
                contact_palm_percentile=float(harvest_profile["contact_palm_percentile"]),
            )
            planner_surface_target_robot_m = [float(value) for value in target_geometry["surface_target_robot_m"]]
            print(
                "Derived harvest target from live RGB-D geometry: "
                f"source={target_geometry['source']} "
                f"component_points={target_geometry['local_component_point_count']} "
                f"shift_from_click_m={target_geometry['target_shift_from_click_m']:.3f} "
                f"palm_shift_m={target_geometry['contact_palm_shift_m']:.3f} "
                "surface="
                + ",".join(f"{value:.6f}" for value in planner_surface_target_robot_m)
            )
        point_arg = ",".join(f"{value:.6f}" for value in planner_surface_target_robot_m)
        task_frame_path = write_generic_harvest_task_frame(
            episode_dir=output_dir,
            action=action,
            surface_target_robot_m=planner_surface_target_robot_m,
            primitive_prior=primitive_prior,
            target_geometry=target_geometry,
        )

    planner_command = [
        python,
        "robot/kinematics/apps/plan_harvest_staged.py",
        str(task_frame_path),
        "--run-label",
        run_label,
        "--toolfit",
        str(args.harvest_toolfit),
        "--disable-scene-obstacle-clearance",
        "--full-orientation-weight",
        str(harvest_profile["full_orientation_weight"]),
        "--harvest-distance-m",
        str(harvest_profile["harvest_distance_m"]),
        "--harvest-down-m",
        str(harvest_profile["harvest_down_m"]),
        "--grasp-insertion-factors",
        str(harvest_profile["grasp_insertion_factors"]),
        "--max-grasp-insertion-factor",
        str(harvest_profile["max_grasp_insertion_factor"]),
        "--palm-correction-factors",
        str(harvest_profile["palm_correction_factors"]),
        "--min-grasp-insertion-fraction",
        str(harvest_profile["min_grasp_insertion_fraction"]),
        f"--lateral-offsets-m={harvest_profile['lateral_offsets_m']}",
        f"--elevated-offsets-m={harvest_profile['elevated_offsets_m']}",
        f"--tcp-rolls-deg={harvest_profile['tcp_rolls_deg']}",
        "--task-axis-pregrasp-gate",
        "--pregrasp-task-axis-position-tolerance-m",
        str(harvest_profile["pregrasp_task_position_tolerance_m"]),
        "--pregrasp-task-jaw-axis-tolerance-rad",
        str(harvest_profile["pregrasp_task_jaw_tolerance_rad"]),
        "--pregrasp-task-backoff-axis-tolerance-rad",
        str(harvest_profile["pregrasp_task_backoff_tolerance_rad"]),
        "--task-axis-grasp-gate",
        "--grasp-task-axis-position-tolerance-m",
        str(harvest_profile["grasp_task_position_tolerance_m"]),
        "--grasp-task-jaw-axis-tolerance-rad",
        str(harvest_profile["grasp_task_jaw_tolerance_rad"]),
        "--grasp-task-backoff-axis-tolerance-rad",
        str(harvest_profile["grasp_task_backoff_tolerance_rad"]),
        "--grasp-palm-alignment-tolerance-m",
        str(args.harvest_grasp_palm_alignment_tolerance_m),
        "--path-samples",
        "10",
    ]
    if args.include_tail:
        planner_command.extend(
            [
                "--include-tail",
                "--tail-prefix-count",
                str(args.tail_prefix_count),
                f"--basket-anchor-robot-m={args.harvest_basket_anchor_robot_m}",
            ]
        )
    if identity_validation_artifact is not None:
        planner_command.extend(["--identity-validation-json", str(identity_validation_artifact)])
    else:
        if point_arg is None:
            raise RuntimeError("internal error: missing harvest surface target")
        planner_command.extend(["--surface-target-robot-m", point_arg])
    if args.harvest_contact_frame_json is not None:
        planner_command.extend(["--contact-frame-json", str(args.harvest_contact_frame_json)])
    grasp_prior_capture_json = harvest_profile.get("grasp_prior_capture_json")
    if grasp_prior_capture_json is not None:
        planner_command.extend(
            [
                "--grasp-prior-capture-json",
                str(grasp_prior_capture_json),
                "--grasp-prior-weight",
                str(harvest_profile["grasp_prior_weight"]),
            ]
        )
    planner_output = run_command_tee(planner_command, cwd=_REPO_ROOT)
    planner_artifact = harvest_planner_artifact_from_output(planner_output)
    planner_payload = read_json(planner_artifact)
    no_scoop_waypoints = ("pre_grasp", "grasp", "harvest") if args.include_tail else ("pre_grasp", "grasp")
    no_scoop = harvest_no_scoop_report(
        planner_payload,
        min_preclose_z_margin_m=float(args.harvest_min_preclose_z_margin_m),
        waypoint_names=no_scoop_waypoints,
    )
    alignment = harvest_grasp_alignment_report(planner_payload)
    physical_quality = harvest_grasp_physical_quality_report(
        planner_payload,
        min_insertion_m=float(args.harvest_min_accepted_insertion_m),
        max_backoff_error_rad=float(args.harvest_max_accepted_backoff_error_rad),
    )
    planner_payload["no_scoop_guard"] = no_scoop
    planner_payload["grasp_alignment_guard"] = alignment
    planner_payload["grasp_physical_quality_guard"] = physical_quality

    copied_target = copy_artifact_bundle(
        target_artifact,
        destination_dir=output_dir / "target",
        step_id=run_label,
        episode_dir=output_dir,
        path_map={},
        copied_jsons=set(),
    )
    stored_planner = output_dir / "planner" / Path(planner_artifact).name
    write_json(stored_planner, {**planner_payload, "artifact_path": str(stored_planner)})
    candidate_archive = None
    if int(args.store_top_solved_candidates) > 0:
        candidate_archive = archive_top_harvest_candidates(
            planner_payload=planner_payload,
            stored_planner=stored_planner,
            output_dir=output_dir,
            top_k=int(args.store_top_solved_candidates),
            min_preclose_z_margin_m=float(args.harvest_min_preclose_z_margin_m),
            no_scoop_waypoints=no_scoop_waypoints,
            min_insertion_m=float(args.harvest_min_accepted_insertion_m),
            max_backoff_error_rad=float(args.harvest_max_accepted_backoff_error_rad),
        )
    summary = {
        "target_label": label,
        "run_label": run_label,
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "target_artifact": str(copied_target or target_artifact),
        "task_frame": str(task_frame_path),
        "planner_artifact": str(stored_planner),
        "source_planner_artifact": str(planner_artifact),
        "include_tail": bool(args.include_tail),
        "hard_gate_ok": bool(planner_payload.get("hard_gate", {}).get("ok", False)),
        "no_scoop_ok": bool(no_scoop.get("ok", False)),
        "alignment_ok": bool(alignment.get("ok", False)),
        "physical_quality_ok": bool(physical_quality.get("ok", False)),
        "physical_quality": physical_quality,
        "best_candidate_name": planner_payload.get("best_candidate_name"),
        "candidate_solve_archive": candidate_archive,
        "harvest_profile": harvest_profile,
        "planner_command": planner_command,
    }
    if args.execute:
        if not summary["hard_gate_ok"]:
            raise RuntimeError(f"harvest solve did not pass hard gates; refusing execution. Artifact: {stored_planner}")
        if not summary["no_scoop_ok"]:
            raise RuntimeError(f"harvest solve failed no-scoop guard; refusing execution. Artifact: {stored_planner}")
        if not summary["alignment_ok"]:
            raise RuntimeError(f"harvest solve failed grasp alignment guard; refusing execution. Artifact: {stored_planner}")
        if not summary["physical_quality_ok"]:
            raise RuntimeError(
                "harvest solve selected a shallow or angled grasp; refusing execution. "
                f"Artifact: {stored_planner}. Guard: {json.dumps(physical_quality, sort_keys=True)}"
            )
        if not args.include_tail and args.execute_until in {"harvest", "retract", "basket_place"}:
            raise RuntimeError(
                f"--execute-until {args.execute_until} requires --include-tail because the default solve stores only "
                "approach, pre_grasp, and grasp."
            )
        if args.home_before_execute:
            restore_home(
                argparse.Namespace(
                    home_scope=args.home_scope,
                    home_minimum_time_s=args.home_minimum_time_s,
                    home_timeout_s=args.home_timeout_s,
                    home_restore_grippers=args.home_restore_grippers,
                    home_close_grippers=getattr(args, "home_close_grippers", False),
                    home_gripper_restore_mode=getattr(args, "home_gripper_restore_mode", "saved"),
                    home_gripper_current_limit=args.home_gripper_current_limit,
                    home_gripper_hold_s=args.home_gripper_hold_s,
                ),
                reason=f"before_harvest_solve_{label}_{run_label}",
            )
        execution_artifact = output_dir / "execution" / f"{run_label}_execution.json"
        execute_command = [
            python,
            "robot/kinematics/apps/execute_harvest_plan.py",
            str(stored_planner),
            "--execute",
            "--stop-after",
            str(args.execute_until),
            "--output-json",
            str(execution_artifact),
        ]
        if not args.no_gripper:
            execute_command.append("--use-gripper")
        print(
            "Executing stored harvest solve: "
            f"artifact={stored_planner} stop_after={args.execute_until} use_gripper={not args.no_gripper}"
        )
        execute_output = run_command_tee(execute_command, cwd=_REPO_ROOT)
        summary["execution_artifact"] = str(execution_artifact)
        summary["execute_until"] = str(args.execute_until)
        summary["execute_command"] = execute_command
        summary["execute_output_tail"] = execute_output[-2000:]
    summary_path = output_dir / "harvest_solve_summary.json"
    write_json(summary_path, summary)
    if args.register_successful_solve:
        if not args.include_tail:
            raise RuntimeError("--register-successful-solve requires --include-tail so the reusable plan includes basket_place")
        if not args.execute:
            raise RuntimeError("--register-successful-solve requires --execute so the registered plan reflects a hardware-tested run")
        execution_path = Path(summary["execution_artifact"]) if summary.get("execution_artifact") else None
        registration = register_successful_harvest_solve(
            target_label=label,
            planner_artifact=stored_planner,
            summary_path=summary_path,
            execution_artifact=execution_path,
        )
        summary["registered_successful_solve"] = registration
        write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2, sort_keys=True, default=json_default))
    return 0


def command_finish_episode(args: argparse.Namespace) -> int:
    episode_json = Path(args.episode_json)
    episode = read_json(episode_json)
    if args.capture_post_scene:
        episode["artifacts"]["post_scene"] = capture_scene(
            episode=episode,
            scene_name="post",
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            serial=args.serial,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            pointcloud_stride=args.pointcloud_stride,
        )
    episode["status"] = args.status
    episode["operator_notes"] = args.notes
    episode["updated_at_utc"] = utc_now()
    write_json(episode_json, episode)
    print(json.dumps({"episode": str(episode_json), "status": args.status}, indent=2))
    return 0


def command_print_commands(args: argparse.Namespace) -> int:
    episode = read_json(args.episode_json)
    print(f"# {episode['episode_id']} symbolic sequence: {' -> '.join(episode['planned_task']['symbolic_sequence'])}")
    print("# Physical leaf setup for this episode:")
    print(f"#   present: {', '.join(episode['planned_task'].get('leaf_positions_present', []))}")
    print(f"#   absent: {', '.join(episode['planned_task'].get('leaf_positions_absent', []))}")
    print("# Physical tomato setup for this episode:")
    print(f"#   present: {', '.join(episode['planned_task'].get('tomato_positions_present', []))}")
    print(f"#   absent: {', '.join(episode['planned_task'].get('tomato_positions_absent', []))}")
    print("# Capture pre-scene if not already captured:")
    print(f"python robot\\policy\\collect_policy_episode.py capture-scene --episode-json {args.episode_json} --scene pre")
    for action in episode["actions"]:
        label = action["symbolic_target"]
        run_label = action["step_id"]
        print()
        print(f"# Step {action['step_index']}: {action['primitive']} {label}")
        arm = "right" if action["primitive"] == "deleaf" else "left"
        print(
            "python robot\\policy\\capture_robot_tcp_trajectory.py "
            f"--arm {arm} --run-label {run_label}_traj --duration-s 90 --hz 10"
        )
        if action["primitive"] == "deleaf":
            print(
                "python robot\\kinematics\\apps\\deleaf_click_cut.py "
                f"--deleaf-config robot\\kinematics\\configs\\deleaf_right_knife_default.json "
                f"--run-label {run_label} --execute --execute-until return_transit"
            )
        else:
            profile = HARVEST_TARGET_PROFILES.get(str(label).upper(), {})
            grasp_insertion_factors = profile.get("grasp_insertion_factors", DEFAULT_HARVEST_GRASP_INSERTION_FACTORS)
            max_grasp_insertion_factor = profile.get("max_grasp_insertion_factor", DEFAULT_HARVEST_MAX_GRASP_INSERTION_FACTOR)
            min_grasp_insertion_fraction = profile.get(
                "min_grasp_insertion_fraction",
                DEFAULT_HARVEST_MIN_GRASP_INSERTION_FRACTION,
            )
            lateral_offsets_m = profile.get("lateral_offsets_m", DEFAULT_HARVEST_LATERAL_OFFSETS_M)
            elevated_offsets_m = profile.get("elevated_offsets_m", DEFAULT_HARVEST_ELEVATED_OFFSETS_M)
            tcp_rolls_deg = profile.get("tcp_rolls_deg", DEFAULT_HARVEST_TCP_ROLLS_DEG)
            pregrasp_pos_tol = profile.get(
                "pregrasp_task_position_tolerance_m",
                DEFAULT_HARVEST_PREGRASP_TASK_POSITION_TOLERANCE_M,
            )
            pregrasp_jaw_tol = profile.get(
                "pregrasp_task_jaw_tolerance_rad",
                DEFAULT_HARVEST_PREGRASP_TASK_JAW_TOLERANCE_RAD,
            )
            pregrasp_backoff_tol = profile.get(
                "pregrasp_task_backoff_tolerance_rad",
                DEFAULT_HARVEST_PREGRASP_TASK_BACKOFF_TOLERANCE_RAD,
            )
            grasp_pos_tol = profile.get(
                "grasp_task_position_tolerance_m",
                DEFAULT_HARVEST_GRASP_TASK_POSITION_TOLERANCE_M,
            )
            grasp_jaw_tol = profile.get(
                "grasp_task_jaw_tolerance_rad",
                DEFAULT_HARVEST_GRASP_TASK_JAW_TOLERANCE_RAD,
            )
            grasp_backoff_tol = profile.get(
                "grasp_task_backoff_tolerance_rad",
                DEFAULT_HARVEST_GRASP_TASK_BACKOFF_TOLERANCE_RAD,
            )
            print(
                "python robot\\kinematics\\utils\\realsense_click_move_arm.py "
                f"--run-label {run_label}_target --arm left --pick-mode open3d "
                "--max-y-m 0.55 --max-z-m 1.35"
            )
            print(
                "python robot\\kinematics\\apps\\plan_harvest_staged.py "
                f"PATH_TO_GENERATED_GENERIC_TASK_FRAME_JSON --run-label {run_label} --surface-target-robot-m X,Y,Z "
                f"--toolfit {DEFAULT_HARVEST_TOOLFIT} "
                "--include-tail --tail-prefix-count 3 "
                f"--basket-anchor-robot-m {DEFAULT_HARVEST_BASKET_ANCHOR_ROBOT_M} "
                "--disable-scene-obstacle-clearance "
                "--full-orientation-weight 0.5 "
                f"--grasp-insertion-factors {grasp_insertion_factors} "
                f"--max-grasp-insertion-factor {max_grasp_insertion_factor} "
                f"--palm-correction-factors {DEFAULT_HARVEST_PALM_CORRECTION_FACTORS} "
                f"--min-grasp-insertion-fraction {min_grasp_insertion_fraction} "
                f"--lateral-offsets-m {lateral_offsets_m} "
                f"--elevated-offsets-m {elevated_offsets_m} "
                f"--tcp-rolls-deg={tcp_rolls_deg} "
                "--task-axis-pregrasp-gate "
                f"--pregrasp-task-axis-position-tolerance-m {pregrasp_pos_tol} "
                f"--pregrasp-task-jaw-axis-tolerance-rad {pregrasp_jaw_tol} "
                f"--pregrasp-task-backoff-axis-tolerance-rad {pregrasp_backoff_tol} "
                "--task-axis-grasp-gate "
                f"--grasp-task-axis-position-tolerance-m {grasp_pos_tol} "
                f"--grasp-task-jaw-axis-tolerance-rad {grasp_jaw_tol} "
                f"--grasp-task-backoff-axis-tolerance-rad {grasp_backoff_tol} "
                "--grasp-palm-alignment-tolerance-m 0.04 --path-samples 10"
            )
            print(
                "python robot\\kinematics\\apps\\execute_harvest_plan.py "
                f"PATH_TO_PLAN_JSON --execute --use-gripper --output-json PATH_TO_{run_label}_EXECUTION_JSON"
            )
        record_cmd = (
            "python robot\\policy\\collect_policy_episode.py record-action "
            f"--episode-json {args.episode_json} --primitive {action['primitive']} "
            f"--target-label {label} "
        )
        if action["primitive"] == "harvest":
            record_cmd += "--target-artifact PATH_TO_TARGET_JSON --planner-artifact PATH_TO_PLAN_JSON "
        else:
            record_cmd += "--action-artifact PATH_TO_ACTION_JSON "
        record_cmd += "--trajectory-artifact PATH_TO_TRAJ_JSON"
        if action["primitive"] == "harvest":
            record_cmd += " --execution-artifact PATH_TO_EXECUTION_JSON"
        print(record_cmd + " --outcome success")
    print()
    print(f"python robot\\policy\\collect_policy_episode.py finish-episode --episode-json {args.episode_json} --capture-post-scene --status complete")
    return 0


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--pointcloud-stride", type=int, default=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage end-to-end policy training episode manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dataset", help="Write the balanced 30-episode dataset plan.")
    init_parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    init_parser.add_argument("--testbed-id", default=DEFAULT_TESTBED_ID)
    init_parser.add_argument(
        "--deleaf-targets",
        default=",".join(DEFAULT_DELEAF_TARGETS),
        help="Comma-separated candidate petiole labels that can hold a leaf.",
    )
    init_parser.add_argument(
        "--harvest-targets",
        default=",".join(HARVEST_TARGETS),
        help="Comma-separated candidate tomato labels that can hold the single harvest tomato.",
    )
    init_parser.add_argument(
        "--physical-leaf-count",
        type=int,
        default=DEFAULT_PHYSICAL_LEAF_COUNT,
        help="Number of physical leaves installed per episode.",
    )
    init_parser.add_argument(
        "--physical-tomato-count",
        type=int,
        default=DEFAULT_PHYSICAL_TOMATO_COUNT,
        help="Number of physical tomatoes installed per episode.",
    )
    init_parser.add_argument("--runs-root", type=Path, default=POLICY_RUNS_ROOT)
    init_parser.add_argument("--output", type=Path, default=None)
    init_parser.set_defaults(func=command_init_dataset)

    start_parser = subparsers.add_parser("start-episode", help="Create one episode JSON from the dataset plan.")
    start_parser.add_argument("--dataset-manifest", type=Path, required=True)
    start_parser.add_argument("--episode-id", required=True)
    start_parser.add_argument("--runs-root", type=Path, default=POLICY_RUNS_ROOT)
    start_parser.add_argument("--episodes-root", type=Path, default=None)
    start_parser.add_argument("--capture-pre-scene", action="store_true")
    start_parser.add_argument("--overwrite", action="store_true")
    add_camera_args(start_parser)
    start_parser.set_defaults(func=command_start_episode)

    instance_parser = subparsers.add_parser(
        "start-action-instance",
        help="Create a one-action policy-library recording container for one symbolic target.",
    )
    instance_parser.add_argument("--target-label", required=True)
    instance_parser.add_argument("--primitive", choices=("deleaf", "harvest"), default=None)
    instance_parser.add_argument("--run-label", default=None)
    instance_parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    instance_parser.add_argument("--testbed-id", default=DEFAULT_TESTBED_ID)
    instance_parser.add_argument("--action-library-root", type=Path, default=POLICY_ACTION_LIBRARY_ROOT)
    instance_parser.add_argument("--context-leaf-positions", default=None)
    instance_parser.add_argument("--context-tomato-positions", default=None)
    instance_parser.add_argument("--capture-pre-scene", action="store_true")
    instance_parser.add_argument("--overwrite", action="store_true")
    add_camera_args(instance_parser)
    instance_parser.set_defaults(func=command_start_action_instance)

    scene_parser = subparsers.add_parser("capture-scene", help="Capture pre/post RealSense scene artifacts for an episode.")
    scene_parser.add_argument("--episode-json", type=Path, required=True)
    scene_parser.add_argument("--scene", choices=("pre", "post"), required=True)
    add_camera_args(scene_parser)
    scene_parser.set_defaults(func=command_capture_scene)

    record_parser = subparsers.add_parser("record-action", help="Attach one executed action artifact to an episode.")
    record_parser.add_argument("--episode-json", type=Path, required=True)
    record_parser.add_argument("--primitive", choices=("deleaf", "harvest"), required=True)
    record_parser.add_argument("--target-label", required=True)
    record_parser.add_argument("--step-index", type=int, default=None)
    record_parser.add_argument("--action-artifact", type=Path, default=None)
    record_parser.add_argument("--target-artifact", type=Path, default=None)
    record_parser.add_argument("--planner-artifact", type=Path, default=None)
    record_parser.add_argument("--trajectory-artifact", type=Path, default=None)
    record_parser.add_argument("--execution-artifact", type=Path, default=None)
    record_parser.add_argument(
        "--outcome",
        choices=("success", "partial", "miss", "collision", "aborted", "unsafe_skip", "dry_run"),
        default="success",
    )
    record_parser.add_argument("--notes", default="")
    record_parser.set_defaults(func=command_record_action)

    run_next_parser = subparsers.add_parser(
        "run-next-action",
        help="Interactively run and record the next planned action for an episode.",
    )
    run_next_parser.add_argument("--episode-json", type=Path, required=True)
    run_next_parser.add_argument("--step-index", type=int, default=None)
    run_next_parser.add_argument("--execute", action="store_true", help="Required for live robot motion.")
    run_next_parser.add_argument("--yes", action="store_true", help="Skip the RUN text confirmation.")
    run_next_parser.add_argument("--address", default="192.0.2.10:50051")
    run_next_parser.add_argument("--model", default="a")
    run_next_parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    run_next_parser.add_argument("--runs-root", type=Path, default=POLICY_RUNS_ROOT)
    run_next_parser.add_argument("--trajectory-duration-s", type=float, default=180.0)
    run_next_parser.add_argument("--trajectory-hz", type=float, default=10.0)
    run_next_parser.add_argument("--trajectory-start-delay-s", type=float, default=1.5)
    run_next_parser.add_argument("--trajectory-stop-timeout-s", type=float, default=20.0)
    run_next_parser.add_argument(
        "--deleaf-execute-until",
        choices=(
            "transit_lift",
            "safe_entry",
            "approach",
            "pre_cut",
            "cut",
            "post_cut_lift",
            "retract_lift",
            "retract",
            "return_transit",
        ),
        default="return_transit",
    )
    run_next_parser.add_argument("--deleaf-cut-time-s", type=float, default=0.25)
    run_next_parser.add_argument("--deleaf-pre-cut-standoff-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-target-extension-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-target-extension-values-m", default=None)
    run_next_parser.add_argument(
        "--deleaf-waypoint-policy",
        choices=("teleop_offset", "mps_template", "mps_raw"),
        default=None,
    )
    run_next_parser.add_argument("--deleaf-teleop-prior-json", type=Path, default=None)
    run_next_parser.add_argument("--deleaf-teleop-prior-source-match", default=None)
    run_next_parser.add_argument("--deleaf-use-teleop-joint-seeds", action="store_true")
    run_next_parser.add_argument("--deleaf-contact-advance-axis-robot", default=None)
    run_next_parser.add_argument("--deleaf-contact-advance-mode", choices=("target", "entry"), default=None)
    run_next_parser.add_argument(
        "--deleaf-task-orientation-policy",
        choices=("teleop_pre_cut", "mps_axes", "side_perpendicular"),
        default=None,
    )
    run_next_parser.add_argument("--deleaf-pointcloud-contact-advance-values-m", default=None)
    run_next_parser.add_argument("--deleaf-pointcloud-candidate-angles-deg", default=None)
    run_next_parser.add_argument("--deleaf-transit-lift-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-safe-entry-standoff-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-safe-entry-axis-robot", default=None)
    run_next_parser.add_argument("--deleaf-safe-entry-lift-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-return-transit-lift-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-retract-lift-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-retract-standoff-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-pointcloud-min-clearance-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-position-tolerance-m", type=float, default=None)
    run_next_parser.add_argument("--deleaf-orientation-tolerance-rad", type=float, default=None)
    run_next_parser.add_argument(
        "--deleaf-mps-auto-select-action",
        action="store_true",
        help="Collect deleaf with per-cut hand-tracking guidance: the clicked petiole inherits "
        "the nearest demonstrated human cut's axes (uses --deleaf-manifest/-alignment/"
        "-hand-tracking/-slam-trajectory). Skips the old per-label side overrides.",
    )
    run_next_parser.add_argument("--deleaf-manifest", type=Path, default=Path("robot/kinematics/runs/deleaf_0d058720_6cuts_manifest.json"))
    run_next_parser.add_argument("--deleaf-alignment", type=Path, default=Path("robot/calibration/session/0d058720/T_robot_from_aria_manual_world_direct.json"))
    run_next_parser.add_argument("--deleaf-hand-tracking", type=Path, default=Path("C:/path/to/user/aria/handtrack/mps_0d058720-9302-4563-8764-3a30da628444_vrs/hand_tracking/hand_tracking_results.csv"))
    run_next_parser.add_argument("--deleaf-slam-trajectory", type=Path, default=Path("C:/path/to/user/aria/handtrack/mps_0d058720-9302-4563-8764-3a30da628444_vrs/slam/closed_loop_trajectory.csv"))
    run_next_parser.add_argument("--deleaf-mps-approach-path-guidance", action="store_true", help="Also seed-warm-start IK along the demonstrated approach path.")
    # Tucked-posture + straight-vertical + snap defaults validated live for per-cut MPS deleaf.
    run_next_parser.add_argument("--deleaf-home", type=Path, default=Path("robot/utils/deleaf_home_right_tucked.json"))
    run_next_parser.add_argument("--deleaf-mps-restore-home", action=argparse.BooleanOptionalAction, default=True, help="Restore both arms to --deleaf-home (right arm tucked) before each per-cut deleaf action.")
    run_next_parser.add_argument("--deleaf-posture-weight", type=float, default=0.1)
    run_next_parser.add_argument("--deleaf-stroke-axis-policy", choices=("mps", "downward_constrained", "world_down", "teleop_prior"), default="world_down")
    run_next_parser.add_argument("--deleaf-max-stroke-tilt-deg", type=float, default=0.0)
    run_next_parser.add_argument("--deleaf-pre-cut-time-s", type=float, default=0.6)
    run_next_parser.add_argument("--deleaf-right-velocity-limit-rad-s", type=float, default=1.2)
    run_next_parser.add_argument("--deleaf-right-acceleration-limit-rad-s2", type=float, default=3.0)
    run_next_parser.add_argument("--harvest-target-picker", choices=("open3d", "rgb"), default="open3d")
    add_camera_args(run_next_parser)
    run_next_parser.add_argument(
        "--harvest-target-source",
        choices=("auto", "identity", "click"),
        default="click",
        help="auto uses live identity validation when a calibrated task frame exists; click uses the RGB-D picker.",
    )
    run_next_parser.add_argument(
        "--harvest-identity-task-frame-json",
        type=Path,
        default=None,
        help="Override the calibrated task-frame JSON used for live harvest target identity validation.",
    )
    run_next_parser.add_argument(
        "--harvest-profile",
        choices=("auto", "global", "AT1T1", "AT1T2", "AT1T3"),
        default="auto",
        help="Harvest tuning profile. auto selects the profile from the symbolic target label.",
    )
    run_next_parser.add_argument(
        "--harvest-fixed-plan-json",
        type=Path,
        default=None,
        help="Execute an existing passing harvest plan artifact directly, skipping harvest replanning.",
    )
    run_next_parser.add_argument(
        "--harvest-fixed-plan-mode",
        choices=("required", "auto", "off"),
        default="auto",
        help=(
            "auto uses a registered full-tail plan when available; required fails if no registered plan exists; "
            "off always replans from the live target."
        ),
    )
    run_next_parser.add_argument(
        "--harvest-fixed-plan-entry-exit-jitter-deg",
        type=float,
        default=0.0,
        help="Apply bounded joint jitter to only approach/retract waypoints of a fixed harvest plan.",
    )
    run_next_parser.add_argument(
        "--harvest-fixed-plan-variation-seed",
        type=int,
        default=None,
        help="Random seed for fixed-plan entry/exit jitter. Stored in the generated variant artifact.",
    )
    run_next_parser.add_argument(
        "--harvest-replay-trajectory-json",
        type=Path,
        default=None,
        help="Replay an existing captured arm joint trajectory for harvest, skipping target capture and IK planning.",
    )
    run_next_parser.add_argument(
        "--harvest-replay-event-plan-json",
        type=Path,
        default=None,
        help="Plan artifact used only to time replay gripper events from pre_grasp/grasp/basket_place waypoints.",
    )
    run_next_parser.add_argument(
        "--harvest-replay-target-artifact",
        type=Path,
        default=None,
        help="Optional current live RGB-D target artifact to store as the goal when replaying a trajectory.",
    )
    run_next_parser.add_argument("--harvest-replay-stride", type=int, default=8)
    run_next_parser.add_argument("--harvest-replay-minimum-time-s", type=float, default=0.35)
    run_next_parser.add_argument("--harvest-replay-entry-exit-jitter-deg", type=float, default=0.0)
    run_next_parser.add_argument("--harvest-replay-variation-seed", type=int, default=None)
    run_next_parser.add_argument("--harvest-toolfit", type=Path, default=DEFAULT_HARVEST_TOOLFIT)
    run_next_parser.add_argument(
        "--harvest-contact-frame-json",
        type=Path,
        default=None,
        help=(
            "Optional contact-frame correction. Defaults to None so policy collection uses "
            "the URDF jaw-center TCP that worked for the tomato1 harvest."
        ),
    )
    run_next_parser.add_argument("--harvest-primitive-prior-dir", type=Path, default=DEFAULT_HARVEST_PRIMITIVE_PRIOR_DIR)
    run_next_parser.add_argument("--harvest-primitive-prior-pattern", default=DEFAULT_HARVEST_TARGET_PATTERN)
    run_next_parser.add_argument("--harvest-basket-anchor-robot-m", default=DEFAULT_HARVEST_BASKET_ANCHOR_ROBOT_M)
    run_next_parser.add_argument(
        "--harvest-target-bias-robot-m",
        default=DEFAULT_HARVEST_TARGET_BIAS_ROBOT_M,
        help="Comma-separated x,y,z bias added to the clicked harvest target before planning.",
    )
    run_next_parser.add_argument("--harvest-generic-depth-m", type=float, default=DEFAULT_HARVEST_GENERIC_DEPTH_M)
    run_next_parser.add_argument("--harvest-generic-width-m", type=float, default=DEFAULT_HARVEST_GENERIC_WIDTH_M)
    run_next_parser.add_argument(
        "--harvest-palm-correction-factors",
        default=DEFAULT_HARVEST_PALM_CORRECTION_FACTORS,
        help=(
            "Comma-separated palm-axis correction factors for harvest grasp insertion. "
            "The default keeps the live RGB-D derived side-contact line instead of recentering downward."
        ),
    )
    run_next_parser.add_argument(
        "--harvest-disable-live-target-geometry",
        action="store_true",
        help="Use the raw clicked point as the harvest contact target instead of deriving target geometry from RGB-D.",
    )
    run_next_parser.add_argument("--harvest-local-radius-px", type=int, default=DEFAULT_HARVEST_LOCAL_RADIUS_PX)
    run_next_parser.add_argument(
        "--harvest-local-depth-window-m",
        type=float,
        default=DEFAULT_HARVEST_LOCAL_DEPTH_WINDOW_M,
    )
    run_next_parser.add_argument(
        "--harvest-local-color-threshold",
        type=float,
        default=DEFAULT_HARVEST_LOCAL_COLOR_THRESHOLD,
    )
    run_next_parser.add_argument("--harvest-local-radius-m", type=float, default=DEFAULT_HARVEST_LOCAL_RADIUS_M)
    run_next_parser.add_argument("--harvest-min-local-points", type=int, default=DEFAULT_HARVEST_MIN_LOCAL_POINTS)
    run_next_parser.add_argument(
        "--harvest-max-derived-target-shift-m",
        type=float,
        default=DEFAULT_HARVEST_MAX_DERIVED_TARGET_SHIFT_M,
    )
    run_next_parser.add_argument(
        "--harvest-contact-palm-percentile",
        type=float,
        default=DEFAULT_HARVEST_CONTACT_PALM_PERCENTILE,
        help=(
            "Percentile along the gripper palm axis used for the live RGB-D side-contact line. "
            "The default chooses an upper-mid contact line rather than the lower visible surface."
        ),
    )
    run_next_parser.add_argument(
        "--harvest-min-preclose-z-margin-m",
        type=float,
        default=DEFAULT_HARVEST_MIN_PRECLOSE_Z_MARGIN_M,
        help=(
            "No-scoop guard: pre_grasp, grasp, and harvest contact z must be at least this margin "
            "relative to the live RGB-D side-contact target z."
        ),
    )
    run_next_parser.add_argument(
        "--no-harvest-scoop-safe-retry",
        dest="harvest_scoop_safe_retry",
        action="store_false",
        help="Disable the automatic no-scoop retry that removes downward palm/down correction when the guard rejects a plan.",
    )
    run_next_parser.set_defaults(harvest_scoop_safe_retry=True)
    run_next_parser.add_argument(
        "--harvest-distance-m",
        type=float,
        default=DEFAULT_HARVEST_DISTANCE_M,
        help=(
            "Post-close pull distance before retract. The default is 0 for policy data collection "
            "to avoid pulling along a downward-tilted primitive axis."
        ),
    )
    run_next_parser.add_argument(
        "--harvest-down-m",
        type=float,
        default=DEFAULT_HARVEST_DOWN_M,
        help=(
            "Vertical drop after grasp during harvest. The default is 0 to avoid scooping upward from below."
        ),
    )
    run_next_parser.add_argument(
        "--harvest-full-orientation-weight",
        type=float,
        default=DEFAULT_HARVEST_FULL_ORIENTATION_WEIGHT,
        help="Planner orientation residual weight for pre_grasp/grasp. Increase to keep the gripper straighter.",
    )
    run_next_parser.add_argument("--harvest-grasp-insertion-factors", default=DEFAULT_HARVEST_GRASP_INSERTION_FACTORS)
    run_next_parser.add_argument(
        "--harvest-grasp-insertion-factor",
        type=float,
        default=None,
        help="Deprecated singular alias; use --harvest-grasp-insertion-factors.",
    )
    run_next_parser.add_argument(
        "--harvest-max-grasp-insertion-factor",
        type=float,
        default=DEFAULT_HARVEST_MAX_GRASP_INSERTION_FACTOR,
    )
    run_next_parser.add_argument(
        "--harvest-min-grasp-insertion-fraction",
        type=float,
        default=DEFAULT_HARVEST_MIN_GRASP_INSERTION_FRACTION,
        help=(
            "Minimum accepted fraction of nominal depth insertion. The policy default requires a small "
            "positive side insertion so the tomato is inside the jaws without allowing deep scoop motions."
        ),
    )
    run_next_parser.add_argument("--harvest-lateral-offsets-m", default=DEFAULT_HARVEST_LATERAL_OFFSETS_M)
    run_next_parser.add_argument("--harvest-elevated-offsets-m", default=DEFAULT_HARVEST_ELEVATED_OFFSETS_M)
    run_next_parser.add_argument("--harvest-tcp-rolls-deg", default=DEFAULT_HARVEST_TCP_ROLLS_DEG)
    run_next_parser.add_argument(
        "--harvest-pregrasp-task-axis-position-tolerance-m",
        type=float,
        default=DEFAULT_HARVEST_PREGRASP_TASK_POSITION_TOLERANCE_M,
    )
    run_next_parser.add_argument(
        "--harvest-pregrasp-task-axis-jaw-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_PREGRASP_TASK_JAW_TOLERANCE_RAD,
    )
    run_next_parser.add_argument(
        "--harvest-pregrasp-task-axis-backoff-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_PREGRASP_TASK_BACKOFF_TOLERANCE_RAD,
    )
    run_next_parser.add_argument(
        "--harvest-grasp-task-axis-position-tolerance-m",
        type=float,
        default=DEFAULT_HARVEST_GRASP_TASK_POSITION_TOLERANCE_M,
    )
    run_next_parser.add_argument(
        "--harvest-grasp-task-axis-jaw-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_GRASP_TASK_JAW_TOLERANCE_RAD,
    )
    run_next_parser.add_argument(
        "--harvest-grasp-task-axis-backoff-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_GRASP_TASK_BACKOFF_TOLERANCE_RAD,
    )
    run_next_parser.add_argument(
        "--harvest-grasp-palm-alignment-tolerance-m",
        type=float,
        default=0.04,
    )
    run_next_parser.add_argument(
        "--harvest-grasp-prior-capture-json",
        type=Path,
        default=None,
    )
    run_next_parser.add_argument("--harvest-grasp-prior-weight", type=float, default=0.05)
    run_next_parser.add_argument(
        "--harvest-no-gripper",
        action="store_true",
        help="Execute harvest arm waypoints without opening/closing the gripper; useful for alignment debugging.",
    )
    run_next_parser.add_argument(
        "--harvest-fixed-tail-after-grasp",
        dest="harvest_fixed_tail_after_grasp",
        action="store_true",
        help="Plan and execute only the tomato grasp prefix, then splice in a fixed basket/drop tail.",
    )
    run_next_parser.add_argument(
        "--no-harvest-fixed-tail-after-grasp",
        dest="harvest_fixed_tail_after_grasp",
        action="store_false",
        help="Solve and execute the planner's full harvest tail instead of using the fixed basket/drop tail.",
    )
    run_next_parser.set_defaults(harvest_fixed_tail_after_grasp=True)
    run_next_parser.add_argument(
        "--harvest-fixed-tail-plan-json",
        type=Path,
        default=DEFAULT_HARVEST_FIXED_TAIL_PLAN,
        help="Known-good harvest plan used only for the post-grasp basket/drop tail.",
    )
    run_next_parser.add_argument("--home-before-action", action="store_true")
    run_next_parser.add_argument("--home-after-action", action="store_true")
    run_next_parser.add_argument(
        "--home-restore-grippers",
        action="store_true",
        help="When homing, restore gripper encoder positions after the arm reaches home.",
    )
    run_next_parser.add_argument(
        "--home-gripper-restore-mode",
        choices=("saved", "close"),
        default="saved",
        help="With --home-restore-grippers, restore saved gripper positions or force-close the grippers.",
    )
    run_next_parser.add_argument(
        "--home-close-grippers",
        action="store_true",
        help="When homing, force-close the grippers after the arm reaches home.",
    )
    run_next_parser.add_argument("--home-scope", choices=("all", "body_head", "left_arm", "right_arm", "arms"), default="arms")
    run_next_parser.add_argument("--home-minimum-time-s", type=float, default=8.0)
    run_next_parser.add_argument("--home-timeout-s", type=float, default=25.0)
    run_next_parser.add_argument("--home-gripper-current-limit", type=float, default=2.0)
    run_next_parser.add_argument("--home-gripper-hold-s", type=float, default=2.0)
    run_next_parser.add_argument(
        "--outcome",
        choices=("success", "partial", "miss", "collision", "aborted", "unsafe_skip", "dry_run"),
        default="success",
    )
    run_next_parser.add_argument("--notes", default="")
    run_next_parser.set_defaults(func=command_run_next_action)

    finish_parser = subparsers.add_parser("finish-episode", help="Mark an episode complete and optionally capture post scene.")
    finish_parser.add_argument("--episode-json", type=Path, required=True)
    finish_parser.add_argument("--status", choices=("complete", "partial", "aborted"), default="complete")
    finish_parser.add_argument("--capture-post-scene", action="store_true")
    finish_parser.add_argument("--notes", default="")
    add_camera_args(finish_parser)
    finish_parser.set_defaults(func=command_finish_episode)

    commands_parser = subparsers.add_parser("print-commands", help="Print per-step collection commands for an episode.")
    commands_parser.add_argument("--episode-json", type=Path, required=True)
    commands_parser.set_defaults(func=command_print_commands)

    solve_parser = subparsers.add_parser(
        "solve-harvest-target",
        help="Capture/click one tomato target, solve harvest IK without execution, and store the solve artifact.",
    )
    solve_parser.add_argument("--target-label", choices=HARVEST_TARGETS, required=True)
    solve_parser.add_argument("--run-label", default=None)
    solve_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "harvest_solve_tests",
    )
    solve_parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    solve_parser.add_argument("--runs-root", type=Path, default=POLICY_RUNS_ROOT)
    solve_parser.add_argument("--harvest-target-picker", choices=("open3d", "rgb"), default="open3d")
    solve_parser.add_argument(
        "--harvest-target-source",
        choices=("auto", "identity", "click"),
        default="click",
        help="click is recommended for solve validation when you physically place one tomato.",
    )
    solve_parser.add_argument("--harvest-identity-task-frame-json", type=Path, default=None)
    solve_parser.add_argument("--harvest-profile", choices=("auto", "global", "AT1T1", "AT1T2", "AT1T3"), default="auto")
    solve_parser.add_argument("--harvest-toolfit", type=Path, default=DEFAULT_HARVEST_TOOLFIT)
    solve_parser.add_argument("--harvest-contact-frame-json", type=Path, default=None)
    solve_parser.add_argument("--harvest-primitive-prior-dir", type=Path, default=DEFAULT_HARVEST_PRIMITIVE_PRIOR_DIR)
    solve_parser.add_argument("--harvest-primitive-prior-pattern", default=DEFAULT_HARVEST_TARGET_PATTERN)
    solve_parser.add_argument("--harvest-basket-anchor-robot-m", default=DEFAULT_HARVEST_BASKET_ANCHOR_ROBOT_M)
    solve_parser.add_argument("--harvest-target-bias-robot-m", default=DEFAULT_HARVEST_TARGET_BIAS_ROBOT_M)
    solve_parser.add_argument("--harvest-generic-depth-m", type=float, default=DEFAULT_HARVEST_GENERIC_DEPTH_M)
    solve_parser.add_argument("--harvest-generic-width-m", type=float, default=DEFAULT_HARVEST_GENERIC_WIDTH_M)
    solve_parser.add_argument("--harvest-palm-correction-factors", default=DEFAULT_HARVEST_PALM_CORRECTION_FACTORS)
    solve_parser.add_argument("--harvest-disable-live-target-geometry", action="store_true")
    solve_parser.add_argument("--harvest-local-radius-px", type=int, default=DEFAULT_HARVEST_LOCAL_RADIUS_PX)
    solve_parser.add_argument("--harvest-local-depth-window-m", type=float, default=DEFAULT_HARVEST_LOCAL_DEPTH_WINDOW_M)
    solve_parser.add_argument("--harvest-local-color-threshold", type=float, default=DEFAULT_HARVEST_LOCAL_COLOR_THRESHOLD)
    solve_parser.add_argument("--harvest-local-radius-m", type=float, default=DEFAULT_HARVEST_LOCAL_RADIUS_M)
    solve_parser.add_argument("--harvest-min-local-points", type=int, default=DEFAULT_HARVEST_MIN_LOCAL_POINTS)
    solve_parser.add_argument("--harvest-max-derived-target-shift-m", type=float, default=DEFAULT_HARVEST_MAX_DERIVED_TARGET_SHIFT_M)
    solve_parser.add_argument("--harvest-contact-palm-percentile", type=float, default=DEFAULT_HARVEST_CONTACT_PALM_PERCENTILE)
    solve_parser.add_argument("--harvest-min-preclose-z-margin-m", type=float, default=DEFAULT_HARVEST_MIN_PRECLOSE_Z_MARGIN_M)
    solve_parser.add_argument("--harvest-distance-m", type=float, default=DEFAULT_HARVEST_DISTANCE_M)
    solve_parser.add_argument("--harvest-down-m", type=float, default=DEFAULT_HARVEST_DOWN_M)
    solve_parser.add_argument(
        "--harvest-full-orientation-weight",
        type=float,
        default=DEFAULT_HARVEST_FULL_ORIENTATION_WEIGHT,
    )
    solve_parser.add_argument("--harvest-grasp-insertion-factors", default=DEFAULT_HARVEST_GRASP_INSERTION_FACTORS)
    solve_parser.add_argument("--harvest-max-grasp-insertion-factor", type=float, default=DEFAULT_HARVEST_MAX_GRASP_INSERTION_FACTOR)
    solve_parser.add_argument("--harvest-min-grasp-insertion-fraction", type=float, default=DEFAULT_HARVEST_MIN_GRASP_INSERTION_FRACTION)
    solve_parser.add_argument("--harvest-lateral-offsets-m", default=DEFAULT_HARVEST_LATERAL_OFFSETS_M)
    solve_parser.add_argument("--harvest-elevated-offsets-m", default=DEFAULT_HARVEST_ELEVATED_OFFSETS_M)
    solve_parser.add_argument("--harvest-tcp-rolls-deg", default=DEFAULT_HARVEST_TCP_ROLLS_DEG)
    solve_parser.add_argument(
        "--harvest-pregrasp-task-axis-position-tolerance-m",
        type=float,
        default=DEFAULT_HARVEST_PREGRASP_TASK_POSITION_TOLERANCE_M,
    )
    solve_parser.add_argument(
        "--harvest-pregrasp-task-axis-jaw-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_PREGRASP_TASK_JAW_TOLERANCE_RAD,
    )
    solve_parser.add_argument(
        "--harvest-pregrasp-task-axis-backoff-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_PREGRASP_TASK_BACKOFF_TOLERANCE_RAD,
    )
    solve_parser.add_argument(
        "--harvest-grasp-task-axis-position-tolerance-m",
        type=float,
        default=DEFAULT_HARVEST_GRASP_TASK_POSITION_TOLERANCE_M,
    )
    solve_parser.add_argument(
        "--harvest-grasp-task-axis-jaw-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_GRASP_TASK_JAW_TOLERANCE_RAD,
    )
    solve_parser.add_argument(
        "--harvest-grasp-task-axis-backoff-tolerance-rad",
        type=float,
        default=DEFAULT_HARVEST_GRASP_TASK_BACKOFF_TOLERANCE_RAD,
    )
    solve_parser.add_argument("--harvest-grasp-palm-alignment-tolerance-m", type=float, default=0.04)
    solve_parser.add_argument("--harvest-min-accepted-insertion-m", type=float, default=DEFAULT_HARVEST_MIN_ACCEPTED_INSERTION_M)
    solve_parser.add_argument(
        "--harvest-max-accepted-backoff-error-rad",
        type=float,
        default=DEFAULT_HARVEST_MAX_ACCEPTED_BACKOFF_ERROR_RAD,
    )
    solve_parser.add_argument("--harvest-grasp-prior-capture-json", type=Path, default=None)
    solve_parser.add_argument("--harvest-grasp-prior-weight", type=float, default=0.05)
    solve_parser.add_argument(
        "--store-top-solved-candidates",
        type=int,
        default=5,
        help="Store the top K solver-passing candidate plans as unconfirmed alternatives alongside the main solve.",
    )
    solve_parser.add_argument("--include-tail", action="store_true")
    solve_parser.add_argument("--tail-prefix-count", type=int, default=3)
    solve_parser.add_argument(
        "--execute",
        action="store_true",
        help="After a passing solve, execute the stored plan. By default this stops at grasp.",
    )
    solve_parser.add_argument(
        "--execute-until",
        choices=("approach", "pre_grasp", "grasp", "harvest", "retract", "basket_place"),
        default="grasp",
    )
    solve_parser.add_argument("--no-gripper", action="store_true", help="Execute arm waypoints without gripper commands.")
    solve_parser.add_argument(
        "--register-successful-solve",
        action="store_true",
        help="After a full-tail hardware execution, register this solve as the reusable plan for the target label.",
    )
    solve_parser.add_argument("--home-before-execute", action="store_true")
    solve_parser.add_argument("--home-scope", choices=("all", "body_head", "left_arm", "right_arm", "arms"), default="arms")
    solve_parser.add_argument("--home-minimum-time-s", type=float, default=8.0)
    solve_parser.add_argument("--home-timeout-s", type=float, default=25.0)
    solve_parser.add_argument("--home-restore-grippers", action="store_true")
    solve_parser.add_argument("--home-gripper-restore-mode", choices=("saved", "close"), default="saved")
    solve_parser.add_argument("--home-close-grippers", action="store_true")
    solve_parser.add_argument("--home-gripper-current-limit", type=float, default=2.0)
    solve_parser.add_argument("--home-gripper-hold-s", type=float, default=2.0)
    add_camera_args(solve_parser)
    solve_parser.set_defaults(func=command_solve_harvest_target)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
