#!/usr/bin/env python
"""Run the removed-side harvesting decoder over frozen Stage-2 caches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egodelta_diff import (  # noqa: E402
    EndpointSide,
    RemovedSurfaceConfig,
    aggregate_evaluations,
    evaluate_points,
    extract_for_endpoint_side,
    find_scene_targets,
    load_stage2_cache,
    parse_gt_markdown,
)


REFERENCE = {
    "scene_count": 30,
    "gt": 54,
    "predictions": 54,
    "tp": 51,
    "fp": 3,
    "fn": 3,
    "precision": 0.9444444444444444,
    "recall": 0.9444444444444444,
    "f1": 0.9444444444444444,
    "median_error_cm": 1.395150356056057,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="fail unless the frozen all-30 metrics are reproduced exactly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = parse_gt_markdown(args.gt)
    config = RemovedSurfaceConfig()
    scene_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []

    for cache_path in sorted(args.cache_dir.glob("*.npz")):
        cache = load_stage2_cache(cache_path)
        found = find_scene_targets(cache.scene_id, entries)
        if found is None or found[1] != "harvesting":
            continue
        _, task, ground_truth = found
        result = extract_for_endpoint_side(
            EndpointSide.REMOVED,
            mu=cache.mu,
            normals=cache.normals,
            delta=cache.delta,
            omega=cache.omega,
            overlap=cache.overlap,
            source_indices=cache.source_indices,
            removed_config=config,
        )
        evaluation = evaluate_points(result.points, ground_truth)
        evaluations.append(evaluation)
        scene_rows.append(
            {
                "scene_id": cache.scene_id,
                "task": task,
                "endpoint_side": EndpointSide.REMOVED.value,
                "sigma_m": result.sigma_m,
                "overlap_primitives": result.overlap_primitive_count,
                "formed_components": result.formed_component_count,
                "accepted_components": result.accepted_component_count,
                **{key: value for key, value in evaluation.items() if key != "matched_errors_m"},
            }
        )
        for target in result.targets:
            target_rows.append(
                {
                    "scene_id": cache.scene_id,
                    "component_id": target.component_id,
                    "mode_id": target.mode_id,
                    "primitive_index": target.primitive_index,
                    "source_index": target.source_index,
                    "x": float(target.xyz[0]),
                    "y": float(target.xyz[1]),
                    "z": float(target.xyz[2]),
                    "component_size": target.component_size,
                    "component_mean_delta": target.component_mean_delta,
                    "component_median_density": target.component_median_density,
                    "density_fence": target.density_fence,
                }
            )

    summary = aggregate_evaluations(evaluations)
    summary.pop("matched_errors_m")
    summary["scene_count"] = len(scene_rows)
    summary["endpoint_side"] = EndpointSide.REMOVED.value
    summary["config"] = asdict(config)
    summary["scope"] = "all 30 harvesting scenes; in-sample operating point"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_scene.csv", scene_rows)
    _write_csv(args.output_dir / "targets.csv", target_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    if args.verify_reference:
        _verify_reference(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _verify_reference(summary: dict[str, object]) -> None:
    for key in ("scene_count", "gt", "predictions", "tp", "fp", "fn"):
        if int(summary[key]) != int(REFERENCE[key]):
            raise AssertionError(
                f"reference mismatch for {key}: {summary[key]} != {REFERENCE[key]}"
            )
    for key in ("precision", "recall", "f1", "median_error_cm"):
        if not np.isclose(
            float(summary[key]), float(REFERENCE[key]), rtol=0.0, atol=1e-12
        ):
            raise AssertionError(
                f"reference mismatch for {key}: {summary[key]} != {REFERENCE[key]}"
            )


if __name__ == "__main__":
    main()
