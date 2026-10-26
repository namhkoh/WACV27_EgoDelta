from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def point3(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("live target point must be a three-element JSON list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("live target point contains a non-finite value")
    return result


def select_assignment(payload: dict[str, Any], target_name: str) -> dict[str, Any]:
    if not payload.get("ok", False):
        raise ValueError("identity validation must have ok=true before it can be frozen")
    matches = [item for item in payload.get("assignments", []) if item.get("target_name") == target_name]
    if len(matches) != 1:
        raise ValueError(f"expected one assignment for {target_name!r}, found {len(matches)}")
    assignment = matches[0]
    if not assignment.get("ok", False):
        raise ValueError(f"assignment for {target_name!r} did not pass")
    live_target = (assignment.get("best_match") or {}).get("live_depth_target")
    if not isinstance(live_target, dict):
        raise ValueError(f"assignment for {target_name!r} has no live depth target")
    point3(live_target.get("point_robot_m"))
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a passed EgoDelta-mask-constrained live target.")
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--task", choices=("harvest", "deleaf"), required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--coordinate-frame", default="robot_base")
    args = parser.parse_args()

    source = args.source_json.resolve()
    output = args.output_json.resolve()
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or provenance.exists():
        raise FileExistsError(f"refusing to overwrite frozen target or provenance: {output}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    assignment = select_assignment(payload, args.target_name)
    best = assignment["best_match"]
    live_target = best["live_depth_target"]
    point_robot_m = point3(live_target["point_robot_m"])
    task_frame = Path(assignment["source_path"]).resolve()
    if not task_frame.is_file():
        raise FileNotFoundError(f"identity assignment task frame does not exist: {task_frame}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    frozen_hash = sha256_file(output)
    source_hash = sha256_file(source)
    if frozen_hash != source_hash:
        output.unlink(missing_ok=True)
        raise RuntimeError("frozen target copy hash does not match source")

    record = {
        "schema": "egodelta_frozen_identity_target_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "task": args.task,
        "scene_id": args.scene_id,
        "target_id": args.target_id,
        "target_name": args.target_name,
        "coordinate_frame": args.coordinate_frame,
        "point_key": "assignments[target_name].best_match.live_depth_target.point_robot_m",
        "point_robot_m": point_robot_m,
        "selection_method": "egodelta_mask_constrained_live_depth_target",
        "task_frame_json": str(task_frame),
        "task_frame_sha256": sha256_file(task_frame),
        "mask_npz": payload.get("mask_npz"),
        "transform_json": payload.get("transform_json"),
        "quality": {
            "center_distance_m": best.get("center_distance_m"),
            "pixel_distance_px": best.get("pixel_distance_px"),
            "depth_error_m": best.get("depth_error_m"),
            "mask_overlap_pixels": best.get("mask_overlap_pixels"),
            "valid_depth_pixels": best.get("valid_depth_pixels"),
        },
        "source_json": str(source),
        "source_sha256": source_hash,
        "frozen_json": str(output),
        "frozen_sha256": frozen_hash,
        "pre_execution": True,
    }
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
