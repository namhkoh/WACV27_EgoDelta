#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_aria_change_baseline.config import PipelineConfig, VALID_TASKS, parse_world_up
from sam_aria_change_baseline.io_aria import load_frames_metadata
from sam_aria_change_baseline.mps_geometry import AriaMpsGeometry
from sam_aria_change_baseline.prepost_change import run_prepost_change
from sam_aria_change_baseline.target_rules import run_target_rules
from sam_aria_change_baseline.visualization import (
    save_change_debug_overlays_on_frames,
    save_summary_figure,
    save_target_overlays_on_frames,
    write_change_debug_plys,
    write_visualization_plys,
)


def _load(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute pre/post changes and task targets from saved pipeline intermediates."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default="harvesting", choices=VALID_TASKS)
    parser.add_argument("--mode", default="single_recording", choices=("single_recording", "separate_recordings"))
    parser.add_argument("--vrs")
    parser.add_argument("--mps")
    parser.add_argument("--post-vrs")
    parser.add_argument("--post-mps")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--sam-backend", default="native_sam3")
    parser.add_argument("--change-score-mode", default="auto", choices=("auto", "average", "sam_absence_only", "point_disappearance_only"))
    parser.add_argument("--disappeared-score-thres", type=float, default=0.60)
    parser.add_argument("--min-visible-post-views", type=int, default=3)
    parser.add_argument("--undistort-rgb", action="store_true")
    parser.add_argument("--world-up", default="0,0,1")
    args = parser.parse_args()

    out = Path(args.out)
    frames_root = out / "frames"
    if args.undistort_rgb or (out / "frames_undistorted" / "post_frames.json").exists():
        frames_root = out / "frames_undistorted"
    config = PipelineConfig(
        task=args.task,
        rgb_stream_id=args.rgb_stream_id,
        sam_backend=args.sam_backend,
        use_undistorted_rgb=frames_root.name == "frames_undistorted",
        change_score_mode=args.change_score_mode,
        disappeared_score_thres=args.disappeared_score_thres,
        min_visible_post_views=args.min_visible_post_views,
        world_up=parse_world_up(args.world_up),
    )
    config.validate()

    if args.mode == "single_recording":
        if not args.vrs or not args.mps:
            raise SystemExit("--vrs and --mps are required for single_recording recompute.")
        pre_geometry = AriaMpsGeometry(args.vrs, args.mps, args.rgb_stream_id, config)
        post_geometry = pre_geometry
        point_reliable = False
    else:
        if not args.vrs or not args.mps or not args.post_vrs or not args.post_mps:
            raise SystemExit(
                "--vrs/--mps and --post-vrs/--post-mps are required for separate_recordings recompute."
            )
        pre_geometry = AriaMpsGeometry(args.vrs, args.mps, args.rgb_stream_id, config)
        post_geometry = AriaMpsGeometry(args.post_vrs, args.post_mps, args.rgb_stream_id, config)
        point_reliable = True

    change_path, changes, _ = run_prepost_change(
        pre_instances_json=out / "lifted_3d" / "pre_instances.json",
        post_instances_json=out / "lifted_3d" / "post_instances.json",
        post_frames_json=frames_root / "post_frames.json",
        post_geometry=post_geometry,
        post_predictions_json=out / "sam_predictions" / "post_predictions_filtered.json",
        config=config,
        task=args.task,
        out_dir=out,
        point_disappearance_reliable=point_reliable,
    )
    final_path, final, final_stats = run_target_rules(
        pre_instances_json=out / "lifted_3d" / "pre_instances.json",
        change_candidates_json=change_path,
        config=config,
        task=args.task,
        out_dir=out,
        method_metadata={"sam_backend": args.sam_backend, "mode": args.mode},
    )

    pre_instances = _load(out / "lifted_3d" / "pre_instances.json")
    post_instances = _load(out / "lifted_3d" / "post_instances.json")
    write_visualization_plys(
        pre_instances=pre_instances,
        post_instances=post_instances,
        change_candidates=changes,
        predictions=final.get("predictions", []),
        out_dir=out,
    )
    change_debug_stats = write_change_debug_plys(
        pre_instances=pre_instances,
        post_instances=post_instances,
        change_candidates=changes,
        predictions=final.get("predictions", []),
        out_dir=out,
    )
    pre_frames = load_frames_metadata(frames_root / "pre_frames.json")
    post_frames = load_frames_metadata(frames_root / "post_frames.json")
    target_overlay_stats = {
        "pre": save_target_overlays_on_frames(
            frames=pre_frames,
            geometry=pre_geometry,
            predictions=final.get("predictions", []),
            split="pre",
            out_dir=out,
        ),
        "post": save_target_overlays_on_frames(
            frames=post_frames,
            geometry=post_geometry,
            predictions=final.get("predictions", []),
            split="post",
            out_dir=out,
        ),
    }
    change_debug_overlay_stats = {
        "pre": save_change_debug_overlays_on_frames(
            frames=pre_frames,
            geometry=pre_geometry,
            pre_instances=pre_instances,
            post_instances=post_instances,
            change_candidates=changes,
            predictions=final.get("predictions", []),
            split="pre",
            out_dir=out,
        ),
        "post": save_change_debug_overlays_on_frames(
            frames=post_frames,
            geometry=post_geometry,
            pre_instances=pre_instances,
            post_instances=post_instances,
            change_candidates=changes,
            predictions=final.get("predictions", []),
            split="post",
            out_dir=out,
        ),
    }

    debug_path = out / "debug_stats.json"
    if debug_path.exists():
        debug = _load(debug_path)
    else:
        debug = {}
    debug["num_disappeared_candidates"] = sum(1 for item in changes if item.get("status") == "disappeared")
    debug["num_final_predictions"] = final_stats["num_final_predictions"]
    debug["change_debug_visualization"] = change_debug_stats
    debug["change_debug_overlay_stats"] = change_debug_overlay_stats
    debug["target_overlay_stats"] = target_overlay_stats
    warnings = list(debug.get("warnings", []))
    note = (
        "Recomputed change targets. In single_recording, point disappearance is not reliable "
        "because pre/post share one MPS map."
    )
    if note not in warnings:
        warnings.append(note)
    debug["warnings"] = warnings
    with debug_path.open("w") as handle:
        json.dump(debug, handle, indent=2)
    save_summary_figure(
        task=args.task,
        summary=debug,
        predictions=final.get("predictions", []),
        out_path=out / "visualizations" / "summary.png",
    )
    print(f"Wrote {change_path}")
    print(f"Wrote {final_path}")
    print(f"Disappeared candidates: {debug['num_disappeared_candidates']}")
    print(f"Final predictions: {debug['num_final_predictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
