from __future__ import annotations

import argparse
import hashlib
import json
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


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def directory_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = [file_record(item) for item in sorted(resolved.rglob("*")) if item.is_file()]
    digest = hashlib.sha256()
    for record in files:
        relative = Path(record["path"]).relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"path": str(resolved), "manifest_sha256": digest.hexdigest(), "files": files}


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze policy-training provenance for a robot evaluation session.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--harvest-policy-dir", type=Path, required=True)
    parser.add_argument("--deleaf-policy-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.session_dir.resolve() / "policy_training_provenance.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training provenance: {output}")
    payload = {
        "schema": "egodelta_policy_training_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_output("rev-parse", "HEAD"),
        "worktree_clean": not bool(git_output("status", "--porcelain")),
        "training_data": {
            "training_index": file_record(REPO_ROOT / "robot" / "policy" / "policy_action_library" / "training_index.json"),
            "artifact_integrity": file_record(REPO_ROOT / "robot" / "policy" / "policy_action_library" / "artifact_integrity.json"),
        },
        "trainer": file_record(REPO_ROOT / "robot" / "policy" / "train_action_chunk_policy.py"),
        "evaluator": file_record(REPO_ROOT / "robot" / "policy" / "eval_policy_on_training_data.py"),
        "policies": {
            "harvest": directory_record(args.harvest_policy_dir),
            "deleaf": directory_record(args.deleaf_policy_dir),
        },
        "training_commands": {
            "harvest": (
                "train_action_chunk_policy.py --primitive harvest --sample-hz 10 --chunk-size 30 "
                "--pad-terminal-chunks --harvest-stop-after retract --trim-idle-prefix "
                "--trim-start-delta-rad 0.005 --trim-preroll-s 0.0 --goal-jitter-count 0 "
                "--bc-backend torch --torch-device cuda --bc-hidden-dims 512,512,256 "
                "--bc-epochs 8000 --bc-batch-size 512 --bc-learning-rate 0.0008 "
                "--bc-weight-decay 0.0 --seed 11 --skip-loo-eval --store-training-arrays"
            ),
            "deleaf": (
                "train_action_chunk_policy.py --primitive deleaf --sample-hz 10 --chunk-size 50 "
                "--pad-terminal-chunks --no-trim-idle-prefix --goal-jitter-count 16 "
                "--goal-jitter-std-m 0.004 --goal-jitter-max-m 0.012 --bc-backend torch "
                "--torch-device cuda --bc-hidden-dims 512,512,256 --bc-epochs 8000 "
                "--bc-batch-size 1024 --bc-learning-rate 0.0008 --bc-weight-decay 0.0 "
                "--seed 11 --skip-loo-eval --store-training-arrays"
            ),
        },
        "hardware": {"device": "NVIDIA GeForce RTX 5090", "torch": "2.11.0+cu128"},
        "evaluation_boundary": (
            "The registered 120 robot rollouts are held-out evaluation and are excluded from these policy artifacts."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
