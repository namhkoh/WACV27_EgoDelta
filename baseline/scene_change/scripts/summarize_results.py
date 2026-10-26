#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = (
    ("3dgscd", "3DGS-CD"),
    ("mv3dcd", "MV3DCD"),
    ("gsdiff", "GS-DIFF (reimplementation)"),
    ("egodelta", "EgoDelta"),
)
METRICS = ("iou", "f1", "precision", "recall")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate complete final-720 evaluations")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    sources = {}
    for slug, display_name in METHODS:
        path = args.evaluation_dir / f"{slug}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["prediction_coverage"] != 1.0:
            raise ValueError(f"{display_name} coverage is not complete")
        if result["image_count"] != 720 or result["scene_intervention_count"] != 60:
            raise ValueError(f"{display_name} does not contain the final 720/60 protocol")
        row = {"method": display_name}
        for metric in METRICS:
            row[f"overall_{metric}"] = result["macro_scene"][metric]
            row[f"deleafing_{metric}"] = result["by_task_kind"]["deleafing"][metric]
            row[f"harvesting_{metric}"] = result["by_task_kind"]["harvesting"][metric]
        rows.append(row)
        sources[slug] = {"path": str(path.resolve()), "sha256": _sha256(path)}

    ego = next(row for row in rows if row["method"] == "EgoDelta")
    best_baseline = max(
        (row for row in rows if row["method"] != "EgoDelta"),
        key=lambda row: row["overall_iou"],
    )
    payload = {
        "schema_version": 1,
        "evaluation_unit": "scene-intervention macro average; 12 fixed views pooled per unit",
        "image_count": 720,
        "scene_intervention_count": 60,
        "rows": rows,
        "best_baseline": best_baseline["method"],
        "egodelta_absolute_gain_over_best_baseline": {
            metric: ego[f"overall_{metric}"] - best_baseline[f"overall_{metric}"]
            for metric in METRICS
        },
        "sources": sources,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["egodelta_absolute_gain_over_best_baseline"], indent=2))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
