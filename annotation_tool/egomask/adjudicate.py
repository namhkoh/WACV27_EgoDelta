from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from PIL import Image

from .export import export_run
from .storage import AnnotationStore


DEFAULT_POLICY_REVIEW_ANNOTATORS = ("annotator_1", "annotator_2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and finalize a human-only second-pass EgoMask review."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create a review run initialized from the saved human masks.",
    )
    prepare.add_argument("--source-run", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--review-queue", type=Path)
    prepare.add_argument(
        "--policy-review-annotator",
        action="append",
        dest="policy_review_annotators",
        help="Review every mask originally assigned to this annotator; repeatable.",
    )
    prepare.add_argument(
        "--reviewer",
        action="append",
        dest="reviewers",
        help="Human reviewer ID; repeatable. Defaults to every source-run annotator.",
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="Export source human masks plus completed human second-pass replacements.",
    )
    finalize.add_argument("--source-run", type=Path, required=True)
    finalize.add_argument("--review-run", type=Path, required=True)
    finalize.add_argument("--override-output", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_human_review(
                args.source_run,
                args.output,
                review_queue=args.review_queue,
                policy_review_annotators=(
                    args.policy_review_annotators or DEFAULT_POLICY_REVIEW_ANNOTATORS
                ),
                reviewers=args.reviewers,
            )
            print(
                f"Prepared {len(result['tasks'])} human-review tasks in "
                f"{Path(args.output).resolve()}"
            )
        else:
            result = finalize_human_review(
                args.source_run,
                args.review_run,
                args.override_output,
                args.output,
            )
            print(
                f"Exported {result['exported_task_count']} human-only masks to "
                f"{Path(args.output).resolve()}"
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def prepare_human_review(
    source_run: str | Path,
    output: str | Path,
    *,
    review_queue: str | Path | None = None,
    policy_review_annotators: tuple[str, ...] | list[str] = DEFAULT_POLICY_REVIEW_ANNOTATORS,
    reviewers: tuple[str, ...] | list[str] | None = None,
) -> dict:
    source = AnnotationStore(source_run)
    output = _require_empty_output(output)
    known_annotators = {row["id"] for row in source.manifest["annotators"]}
    policy_review_annotators = tuple(dict.fromkeys(policy_review_annotators))
    reviewers = tuple(dict.fromkeys(reviewers or sorted(known_annotators)))
    unknown = set(policy_review_annotators + reviewers).difference(known_annotators)
    if unknown:
        raise ValueError(f"Unknown annotators: {sorted(unknown)}")
    if len(reviewers) < 2:
        raise ValueError("Human second-pass review requires at least two reviewers")

    review_rows, review_queue_metadata = _load_review_queue(review_queue)
    review_by_task = {row["task_id"]: row for row in review_rows}
    source_tasks = {task["task_id"]: task for task in source.manifest["tasks"]}
    unknown_tasks = sorted(set(review_by_task).difference(source_tasks))
    if unknown_tasks:
        raise ValueError(f"Review queue contains unknown task IDs: {unknown_tasks[:5]}")

    selected = [
        task
        for task in source.manifest["tasks"]
        if task["assigned_to"] in policy_review_annotators or task["task_id"] in review_by_task
    ]
    if not selected:
        raise ValueError("No tasks selected for human review")
    assignments = assign_reviewers(selected, reviewers)

    prepared_tasks: list[dict] = []
    reviewer_counts = {reviewer: 0 for reviewer in reviewers}
    for task in selected:
        task_id = task["task_id"]
        original_annotator = task["assigned_to"]
        record = source.load_record(original_annotator, task_id)
        if record["status"] != "complete":
            raise ValueError(f"Source task is not complete: {task_id}")
        source_mask = source.annotation_mask_path(original_annotator, task_id)
        if not source_mask.is_file():
            raise FileNotFoundError(f"Source human mask is missing: {source_mask}")
        with Image.open(source_mask) as loaded:
            mask = loaded.convert("L")
            mask.load()
        expected_size = (int(task["camera"]["width"]), int(task["camera"]["height"]))
        if mask.size != expected_size:
            raise ValueError(f"Wrong source mask dimensions for {task_id}: {mask.size}")
        mask = mask.point(lambda value: 1 if value else 0, mode="L")
        relative_initial = Path("initial_masks") / task["scene_id"] / f"{Path(task['image_name']).stem}.png"
        _save_png_atomic(output / relative_initial, mask)

        reviewer = assignments[task_id]
        reviewer_counts[reviewer] += 1
        queued = review_by_task.get(task_id)
        policy_review = original_annotator in policy_review_annotators
        reasons = []
        if policy_review:
            reasons.append("projected-base boundary policy requires human silhouette review")
        if queued:
            reasons.append(queued.get("reason") or queued.get("action") or "targeted quality issue")
        source_components = task["components"]
        prepared = {
            **task,
            "assigned_to": reviewer,
            "initial_mask": str(relative_initial),
            "source_components": source_components,
            "components": [
                {
                    "component_id": 0,
                    "label": 1,
                    "color": "#20d6c7",
                    "adjacent_gt_indices": sorted(
                        {
                            int(index)
                            for component in source_components
                            for index in component["adjacent_gt_indices"]
                        }
                    ),
                    "primitive_count": sum(
                        int(component["primitive_count"]) for component in source_components
                    ),
                    "source_ply": source_components[0]["source_ply"],
                    "role": "human_visible_change_union",
                }
            ],
            "adjudication": {
                "workflow": "human_second_pass",
                "original_annotator": original_annotator,
                "original_annotation_sha256": _sha256(source_mask),
                "original_annotation_revision": int(record["revision"]),
                "original_annotation_updated_utc": record["updated_utc"],
                "policy_review_required": policy_review,
                "queued_quality_issue": bool(queued),
                "reason": "; ".join(reasons),
                "instruction": (
                    "Review the visible RGB silhouette as one foreground component. "
                    "If a projected base remains, clear label 1 and redraw it; exclude occluded "
                    "regions, neighboring plants, hands, tools, and background. Completing the "
                    "task records explicit human approval."
                ),
            },
        }
        prepared_tasks.append(prepared)

    source_annotators = {row["id"]: row for row in source.manifest["annotators"]}
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "human_second_pass_adjudication",
        "source_run": str(source.run_root),
        "source_manifest_sha256": _sha256(source.run_root / "manifest.json"),
        "human_only": True,
        "component_policy": "single_foreground_union",
        "review_policy": (
            "Every task must be explicitly completed by a human reviewer other than its "
            "original annotator. Automatic masks are not accepted as final labels."
        ),
        "review_queue": review_queue_metadata,
        "policy_review_annotators": list(policy_review_annotators),
        "annotators": [
            {
                "id": reviewer,
                "token": source_annotators[reviewer]["token"],
                "assigned_tasks": reviewer_counts[reviewer],
            }
            for reviewer in reviewers
        ],
        "scenes": [
            scene
            for scene in source.manifest.get("scenes", [])
            if any(task["scene_id"] == scene["scene_id"] for task in prepared_tasks)
        ],
        "tasks": prepared_tasks,
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def finalize_human_review(
    source_run: str | Path,
    review_run: str | Path,
    override_output: str | Path,
    output: str | Path,
) -> dict:
    source = AnnotationStore(source_run)
    review = AnnotationStore(review_run)
    if review.manifest.get("workflow") != "human_second_pass_adjudication":
        raise ValueError("Review run is not a human second-pass adjudication run")
    if not review.manifest.get("human_only"):
        raise ValueError("Review run is not marked human-only")
    if Path(review.manifest.get("source_run", "")).resolve() != source.run_root:
        raise ValueError("Review run was prepared from a different source run")
    if review.manifest.get("source_manifest_sha256") != _sha256(source.run_root / "manifest.json"):
        raise ValueError("Source manifest changed after the human-review run was prepared")

    override_output = _require_empty_output(override_output)
    source_tasks = {task["task_id"]: task for task in source.manifest["tasks"]}
    override_rows: list[dict] = []
    for task in review.manifest["tasks"]:
        task_id = task["task_id"]
        reviewer = task["assigned_to"]
        record = review.load_record(reviewer, task_id)
        if record["status"] != "complete":
            raise ValueError(f"Human review is incomplete for {task_id}: {record['status']}")
        reviewed_mask = review.annotation_mask_path(reviewer, task_id)
        if not reviewed_mask.is_file() or _sha256(reviewed_mask) != record.get("mask_sha256"):
            raise ValueError(f"Human-reviewed mask is missing or corrupt: {task_id}")
        with Image.open(reviewed_mask) as loaded:
            mask = loaded.convert("L")
            mask.load()
        source_task = source_tasks[task_id]
        expected_size = (
            int(source_task["camera"]["width"]),
            int(source_task["camera"]["height"]),
        )
        if mask.size != expected_size:
            raise ValueError(f"Wrong reviewed mask dimensions for {task_id}: {mask.size}")
        if mask.getbbox() is None:
            raise ValueError(f"Human-reviewed mask is empty: {task_id}")
        mask = mask.point(lambda value: 1 if value else 0, mode="L")
        relative_mask = Path("masks") / f"{task_id}.png"
        _save_png_atomic(override_output / relative_mask, mask)

        original_annotator = task["adjudication"]["original_annotator"]
        if reviewer == original_annotator:
            raise ValueError(f"Human review is not independent for {task_id}")
        source_mask = source.annotation_mask_path(original_annotator, task_id)
        source_sha256 = _sha256(source_mask)
        if source_sha256 != task["adjudication"]["original_annotation_sha256"]:
            raise ValueError(f"Source human mask changed during adjudication: {task_id}")
        override_rows.append(
            {
                "task_id": task_id,
                "mask": str(relative_mask),
                "reason": "completed human second-pass review",
                "source_annotation_sha256": source_sha256,
                "original_annotator": original_annotator,
                "human_reviewer": reviewer,
                "human_review_revision": int(record["revision"]),
                "human_review_updated_utc": record["updated_utc"],
                "human_review_note": record["note"],
            }
        )

    override_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "human_second_pass_adjudication",
        "status": "human_review_complete",
        "human_only": True,
        "source_run": str(source.run_root),
        "review_run": str(review.run_root),
        "review_manifest_sha256": _sha256(review.run_root / "manifest.json"),
        "overrides": override_rows,
    }
    _write_json_atomic(override_output / "overrides.json", override_manifest)
    return export_run(
        source.run_root,
        output,
        single_component=True,
        mask_overrides=override_output,
    )


def assign_reviewers(tasks: list[dict], reviewers: tuple[str, ...] | list[str]) -> dict[str, str]:
    reviewers = tuple(dict.fromkeys(reviewers))
    loads = {reviewer: 0 for reviewer in reviewers}
    assignments: dict[str, str] = {}
    for task in sorted(tasks, key=lambda row: (row["scene_id"], row["image_name"])):
        candidates = [reviewer for reviewer in reviewers if reviewer != task["assigned_to"]]
        if not candidates:
            raise ValueError(f"No independent reviewer available for {task['task_id']}")
        reviewer = min(candidates, key=lambda candidate: (loads[candidate], candidate))
        assignments[task["task_id"]] = reviewer
        loads[reviewer] += 1
    return assignments


def _load_review_queue(path: str | Path | None) -> tuple[list[dict], dict | None]:
    if path is None:
        return [], None
    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "task_id" not in rows[0]:
        raise ValueError(f"Review queue has no task_id column: {path}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError(f"Review queue contains duplicate task IDs: {path}")
    return rows, {"path": str(path), "sha256": _sha256(path), "task_count": len(rows)}


def _require_empty_output(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_png_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
