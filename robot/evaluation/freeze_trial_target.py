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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dotted_get(payload: Any, dotted_key: str) -> Any:
    value = payload
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"target point key {dotted_key!r} is missing at {part!r}")
        value = value[part]
    return value


def point3(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("target point must be a three-element JSON list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("target point contains a non-finite value")
    return result


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze one pre-execution EgoDelta target without changing its planner-facing JSON schema."
    )
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--task", choices=("harvest", "deleaf"), required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--point-key", required=True, help="Dotted source JSON key for the robot-frame target point.")
    parser.add_argument("--coordinate-frame", default="robot_base")
    args = parser.parse_args()

    source = args.source_json.resolve()
    output = args.output_json.resolve()
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() or provenance.exists():
        raise FileExistsError(f"refusing to overwrite frozen target: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    point_robot_m = point3(dotted_get(payload, args.point_key))

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    frozen_hash = sha256_file(output)
    if frozen_hash != sha256_file(source):
        output.unlink(missing_ok=True)
        raise RuntimeError("frozen target copy hash does not match its source")
    record = {
        "schema": "egodelta_frozen_trial_target_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(),
        "task": args.task,
        "scene_id": args.scene_id,
        "target_id": args.target_id,
        "coordinate_frame": args.coordinate_frame,
        "point_key": args.point_key,
        "point_robot_m": point_robot_m,
        "source_json": str(source),
        "source_sha256": sha256_file(source),
        "frozen_json": str(output),
        "frozen_sha256": frozen_hash,
        "pre_execution": True,
    }
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
