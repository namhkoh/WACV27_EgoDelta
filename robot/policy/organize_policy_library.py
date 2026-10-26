from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parent / "policy_action_library"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_stem(value: str, *, max_len: int = 140) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:max_len]


def rewrite_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for source, dest in replacements.items():
            result = result.replace(source, dest)
        return result
    if isinstance(value, list):
        return [rewrite_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_strings(item, replacements) for key, item in value.items()}
    return value


def localize_success_current_episode(*, label_dir: Path, success_dir: Path, episode_json: Path) -> None:
    episode = read_json(episode_json)
    episode_id = episode.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        return
    source_dir = label_dir / episode_id
    if not source_dir.exists() or source_dir == success_dir:
        return
    localized = rewrite_strings(
        episode,
        {
            str(source_dir): str(success_dir),
            str(source_dir).replace("\\", "/"): str(success_dir).replace("\\", "/"),
        },
    )
    write_json(episode_json, localized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path_value: str | None, *, root: Path) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    record: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.exists() and path.is_relative_to(root) else None,
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        stat = path.stat()
        record["size_bytes"] = stat.st_size
        record["sha256"] = sha256_file(path)
    return record


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize_harvest_candidate_solves(candidate_dir: Path) -> None:
    index_path = candidate_dir / "top_solved_candidates.json"
    candidate_files = sorted(
        path for path in candidate_dir.glob("*.json") if path.name != "top_solved_candidates.json"
    )
    replacements: dict[str, str] = {}
    for index, source in enumerate(candidate_files, start=1):
        dest = candidate_dir / f"candidate_{index:02d}.json"
        if source == dest:
            continue
        if dest.exists():
            dest.unlink()
        source.rename(dest)
        replacements[str(source)] = str(dest)
        replacements[source.name] = dest.name
    if replacements and index_path.exists():
        payload = read_json(index_path)
        payload = rewrite_strings(payload, replacements)
        write_json(index_path, payload)


def export_deleaf_candidates(
    *,
    label: str,
    step_id: str,
    action_artifact: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    payload = read_json(action_artifact)
    candidates = payload.get("candidate_summaries", [])
    selected = payload.get("selected_candidate")
    step_dir = action_artifact.parent.parent
    candidate_dir = step_dir / "candidate_summaries"
    selected_name = selected.get("name") if isinstance(selected, dict) else None

    bundle = {
        "created_at_utc": created_at_utc,
        "label": label,
        "step_id": step_id,
        "primitive": "deleaf",
        "source_action_artifact": str(action_artifact),
        "candidate_count": len(candidates),
        "selected_candidate_name": selected_name,
        "selected_candidate": selected,
        "candidate_summaries": candidates,
        "note": "Deleaf alternatives are planner summaries, not independently verified hardware executions.",
    }
    bundle_path = candidate_dir / "deleaf_candidate_summaries.json"
    write_json(bundle_path, bundle)

    for index, candidate in enumerate(candidates, start=1):
        name = str(candidate.get("name", f"candidate_{index:02d}"))
        candidate_payload = {
            "created_at_utc": created_at_utc,
            "rank": index,
            "label": label,
            "step_id": step_id,
            "source_action_artifact": str(action_artifact),
            "selected": selected_name is not None and candidate.get("name") == selected_name,
            "candidate_summary": candidate,
        }
        write_json(candidate_dir / f"{index:02d}_{safe_stem(name)}.json", candidate_payload)

    return {
        "candidate_count": len(candidates),
        "candidate_summary_bundle": str(bundle_path),
        "selected_candidate_name": selected_name,
    }


def build_index(root: Path) -> dict[str, Any]:
    created_at_utc = datetime.now(timezone.utc).isoformat()
    output: dict[str, Any] = {
        "created_at_utc": created_at_utc,
        "root": str(root),
        "schema": "egodelta_policy_action_library_index_v1",
        "actions": [],
        "missing_success_current": [],
    }

    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = label_dir.name
        success_dir = label_dir / "success_current"
        episode_json = success_dir / "episode.json"
        if not episode_json.exists():
            output["missing_success_current"].append(label)
            continue
        localize_success_current_episode(label_dir=label_dir, success_dir=success_dir, episode_json=episode_json)

        episode = read_json(episode_json)
        entry: dict[str, Any] = {
            "label": label,
            "success_current_dir": str(success_dir),
            "episode_json": str(episode_json),
            "episode_status": episode.get("status"),
            "action_count": len(episode.get("actions", [])),
            "actions": [],
        }

        for action in episode.get("actions", []):
            artifacts = action.get("artifacts", {})
            primitive = action.get("primitive")
            action_entry: dict[str, Any] = {
                "step_id": action.get("step_id"),
                "primitive": primitive,
                "symbolic_target": action.get("symbolic_target"),
                "status": action.get("status"),
                "outcome": action.get("outcome"),
                "goal": action.get("goal", {}),
                "artifacts": artifacts,
            }
            if primitive == "deleaf":
                action_artifact = Path(artifacts["action"]) if artifacts.get("action") else None
                if action_artifact is not None and action_artifact.exists():
                    action_entry.update(
                        export_deleaf_candidates(
                            label=label,
                            step_id=str(action.get("step_id")),
                            action_artifact=action_artifact,
                            created_at_utc=created_at_utc,
                        )
                    )
                else:
                    action_entry["candidate_count"] = 0
                    action_entry["candidate_summary_bundle"] = None
            elif primitive == "harvest":
                candidate_dir = None
                planner = artifacts.get("planner")
                if planner:
                    possible = Path(planner).parent.parent / "candidate_solves"
                    if possible.exists():
                        normalize_harvest_candidate_solves(possible)
                        candidate_dir = possible
                action_entry["candidate_solve_dir"] = str(candidate_dir) if candidate_dir else None

            entry["actions"].append(action_entry)
        output["actions"].append(entry)

    return output


def write_training_indexes(root: Path, index: dict[str, Any]) -> dict[str, Any]:
    episodes_rows: list[dict[str, Any]] = []
    actions_rows: list[dict[str, Any]] = []
    integrity_artifacts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for entry in index["actions"]:
        episode_json = entry["episode_json"]
        episode = read_json(Path(episode_json))
        artifacts = episode.get("artifacts", {})
        episodes_rows.append(
            {
                "label": entry["label"],
                "episode_json": episode_json,
                "episode_id": episode.get("episode_id"),
                "status": episode.get("status"),
                "success_current": True,
                "action_count": entry["action_count"],
                "pre_scene_json": (artifacts.get("pre_scene") or {}).get("json"),
                "pre_scene_jpg": (artifacts.get("pre_scene") or {}).get("jpg"),
                "pre_scene_npz": (artifacts.get("pre_scene") or {}).get("npz"),
            }
        )
        for artifact_path in (
            (artifacts.get("pre_scene") or {}).get("json"),
            (artifacts.get("pre_scene") or {}).get("jpg"),
            (artifacts.get("pre_scene") or {}).get("npz"),
            episode_json,
        ):
            record = artifact_record(artifact_path, root=root)
            if record is not None:
                integrity_artifacts.append(record)
                if not record["exists"]:
                    missing.append(record)

        for action in entry["actions"]:
            action_artifacts = action.get("artifacts", {})
            row = {
                "label": entry["label"],
                "step_id": action.get("step_id"),
                "primitive": action.get("primitive"),
                "symbolic_target": action.get("symbolic_target"),
                "status": action.get("status"),
                "outcome": action.get("outcome"),
                "trajectory": action_artifacts.get("trajectory"),
                "action": action_artifacts.get("action"),
                "target": action_artifacts.get("target"),
                "planner": action_artifacts.get("planner"),
                "execution": action_artifacts.get("execution"),
                "candidate_summary_bundle": action.get("candidate_summary_bundle"),
                "candidate_solve_dir": action.get("candidate_solve_dir"),
                "candidate_count": action.get("candidate_count"),
                "selected_candidate_name": action.get("selected_candidate_name"),
            }
            actions_rows.append(row)
            for artifact_path in (
                row["trajectory"],
                row["action"],
                row["target"],
                row["planner"],
                row["execution"],
                row["candidate_summary_bundle"],
            ):
                record = artifact_record(artifact_path, root=root)
                if record is not None:
                    integrity_artifacts.append(record)
                    if not record["exists"]:
                        missing.append(record)

            candidate_dir = row["candidate_solve_dir"]
            if candidate_dir:
                for path in sorted(Path(candidate_dir).glob("*.json")):
                    record = artifact_record(str(path), root=root)
                    if record is not None:
                        integrity_artifacts.append(record)
                        if not record["exists"]:
                            missing.append(record)

    episodes_index = root / "episodes_index.jsonl"
    actions_index = root / "actions_index.jsonl"
    manifest_path = root / "library_manifest.json"
    integrity_path = root / "artifact_integrity.json"
    splits_path = root / "training_splits.json"

    jsonl_write(episodes_index, episodes_rows)
    jsonl_write(actions_index, actions_rows)
    write_json(
        manifest_path,
        {
            "schema": "egodelta_policy_library_manifest_v1",
            "created_at_utc": index["created_at_utc"],
            "root": str(root),
            "labels": [entry["label"] for entry in index["actions"]],
            "episode_count": len(episodes_rows),
            "action_count": len(actions_rows),
            "indexes": {
                "training_index": str(root / "training_index.json"),
                "episodes_index": str(episodes_index),
                "actions_index": str(actions_index),
                "artifact_integrity": str(integrity_path),
                "training_splits": str(splits_path),
            },
        },
    )
    write_json(
        integrity_path,
        {
            "schema": "egodelta_policy_artifact_integrity_v1",
            "created_at_utc": index["created_at_utc"],
            "artifact_count": len(integrity_artifacts),
            "missing_count": len(missing),
            "missing": missing,
            "artifacts": integrity_artifacts,
        },
    )
    write_json(
        splits_path,
        {
            "schema": "egodelta_policy_training_splits_v1",
            "created_at_utc": index["created_at_utc"],
            "train": [row["label"] for row in actions_rows],
            "val": [],
            "test": [],
            "note": "Current library is a small visually confirmed action set; no held-out split is assigned yet.",
        },
    )
    return {
        "library_manifest": str(manifest_path),
        "episodes_index": str(episodes_index),
        "actions_index": str(actions_index),
        "artifact_integrity": str(integrity_path),
        "training_splits": str(splits_path),
        "missing_artifact_count": len(missing),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root
    index = build_index(root)
    index_path = root / "training_index.json"
    write_json(index_path, index)
    training_outputs = write_training_indexes(root, index)

    summary = {
        "index": str(index_path),
        **training_outputs,
        "labels_indexed": [entry["label"] for entry in index["actions"]],
        "missing_success_current": index["missing_success_current"],
        "deleaf_candidate_bundles": [
            action.get("candidate_summary_bundle")
            for entry in index["actions"]
            for action in entry["actions"]
            if action.get("primitive") == "deleaf"
        ],
        "harvest_candidate_dirs": [
            action.get("candidate_solve_dir")
            for entry in index["actions"]
            for action in entry["actions"]
            if action.get("primitive") == "harvest"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
