#!/usr/bin/env python
"""Reproduce the final identical-cache legacy rows from the archived call graph."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OLD_CODE = ROOT / "old" / "code"
sys.path.insert(0, str(OLD_CODE))

from egodelta_diff.contact_proposal import (  # noqa: E402
    contact_config_from_dict,
    generate_contact_proposals,
)
from egodelta_diff.targets import (  # noqa: E402
    rule_candidate_from_json,
    select_contact_guarded_rule_leaf_candidates,
)


REFERENCE = {
    "deleafing": {
        "predictions": 99, "gt": 106, "tp": 90, "fp": 9, "fn": 16,
        "precision": 0.9090909090909091, "recall": 0.8490566037735849,
        "f1": 0.8780487804878049, "median_error_cm": 1.182,
    },
    "harvesting": {
        "predictions": 54, "gt": 54, "tp": 49, "fp": 5, "fn": 5,
        "precision": 0.9074074074074074, "recall": 0.9074074074074074,
        "f1": 0.9074074074074074, "median_error_cm": 1.584,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=OLD_CODE / "configs" / "objectness_full.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    entries = _parse_gt(args.gt)
    proposal_cfg = contact_config_from_dict(
        yaml.safe_load(args.config.read_text()).get("proposal", {})
    )
    task_rows = {"deleafing": [], "harvesting": []}
    scene_rows = []
    for path in sorted(args.cache_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            scene_id = str(metadata["scene_id"])
            found = _find_gt(scene_id, entries)
            if found is None:
                continue
            task, gt = found
            proposals = generate_contact_proposals(
                points=data["mu"],
                scores=data["delta"],
                source_indices=data["source_indices"],
                rgb01=data["rgb01"],
                removed_mask=data["delta"] > 0.5,
                cluster_labels=data["legacy_cluster_labels"],
                config=proposal_cfg,
                object_type="tomato" if task == "harvesting" else "leaf",
            )
            selected = _legacy_select(scene_id, task, proposals)
            prediction = np.asarray(
                [proposal.representative_point for proposal in selected], dtype=np.float64
            ).reshape((-1, 3))
            evaluation = _evaluate(prediction, gt)
            task_rows[task].append(evaluation)
            scene_rows.append({"scene_id": scene_id, "task": task, **evaluation})

    summary = {task: _aggregate(rows) for task, rows in task_rows.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_scene.csv", scene_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if args.verify_reference:
        _verify(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _legacy_select(scene_id, task, proposals):
    if task == "harvesting":
        return sorted(
            [proposal for proposal in proposals if proposal.primary],
            key=lambda proposal: (proposal.rank, proposal.proposal_id),
        )
    candidates = [
        rule_candidate_from_json(scene_id, asdict(proposal)) for proposal in proposals
    ]
    ids = {
        candidate.proposal_id
        for candidate in select_contact_guarded_rule_leaf_candidates(candidates)
    }
    return sorted(
        [proposal for proposal in proposals if proposal.proposal_id in ids],
        key=lambda proposal: (proposal.rank, proposal.proposal_id),
    )


def _parse_gt(path):
    entries = {}
    scene_uuid = None
    task = None
    header = re.compile(r"^\d+\.\s+([0-9a-fA-F-]+)\.vrs$")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        match = header.match(line)
        if match:
            scene_uuid = match.group(1).lower()
            entries[scene_uuid] = {"deleafing": [], "harvesting": []}
            task = None
        elif line in ("deleafing", "harvesting"):
            task = line
        elif line.startswith("[") and scene_uuid and task:
            entries[scene_uuid][task].append(json.loads(line))
    return {
        key: {
            task_name: np.asarray(value, dtype=np.float32).reshape((-1, 3))
            for task_name, value in tasks.items()
        }
        for key, tasks in entries.items()
    }


def _find_gt(scene_id, entries):
    match = re.match(r"^egodelta_([0-9a-f]+)_(leaf|tomato)$", scene_id)
    if not match:
        return None
    candidates = [
        key for key in entries if key.replace("-", "").startswith(match.group(1))
    ]
    if not candidates:
        return None
    task = "deleafing" if match.group(2) == "leaf" else "harvesting"
    return task, entries[candidates[0]][task]


def _evaluate(prediction, gt):
    if not len(prediction):
        return _metric(0, 0, len(gt), [])
    distance = np.linalg.norm(prediction[:, None] - gt[None], axis=2)
    invalid = float(distance.max() + 2.0)
    rows, columns = linear_sum_assignment(
        np.where(distance <= 0.03, distance, invalid)
    )
    valid = distance[rows, columns] <= 0.03
    return _metric(
        int(valid.sum()), len(prediction) - int(valid.sum()),
        len(gt) - int(valid.sum()), distance[rows[valid], columns[valid]].tolist()
    )


def _metric(tp, fp, fn, errors):
    return {
        "predictions": tp + fp, "gt": tp + fn, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "median_error_cm": float(np.median(errors) * 100.0) if errors else None,
        "matched_errors_m": errors,
    }


def _aggregate(rows):
    errors = [value for row in rows for value in row["matched_errors_m"]]
    output = _metric(
        sum(row["tp"] for row in rows), sum(row["fp"] for row in rows),
        sum(row["fn"] for row in rows), errors
    )
    output.pop("matched_errors_m")
    return output


def _verify(summary):
    for task, expected in REFERENCE.items():
        for key in ("predictions", "gt", "tp", "fp", "fn"):
            if summary[task][key] != expected[key]:
                raise AssertionError(f"{task} {key}: {summary[task][key]} != {expected[key]}")
        for key in ("precision", "recall", "f1"):
            if not np.isclose(summary[task][key], expected[key], rtol=0, atol=1e-12):
                raise AssertionError(f"{task} {key}: {summary[task][key]} != {expected[key]}")
        if not np.isclose(
            summary[task]["median_error_cm"], expected["median_error_cm"], rtol=0,
            atol=5e-4,
        ):
            raise AssertionError(
                f"{task} median: {summary[task]['median_error_cm']} != {expected['median_error_cm']}"
            )


def _write_csv(path, rows):
    clean = [{k: v for k, v in row.items() if k != "matched_errors_m"} for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0]))
        writer.writeheader()
        writer.writerows(clean)


if __name__ == "__main__":
    main()
