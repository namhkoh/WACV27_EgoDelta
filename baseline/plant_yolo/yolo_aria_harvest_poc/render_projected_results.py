"""CLI for projecting final 3D results back onto extracted RGB frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Render final 3D result overlays on RGB frames")
    parser.add_argument("--out", required=True, help="Pipeline output directory")
    parser.add_argument("--vrs", required=True, help="VRS used for calibration")
    parser.add_argument("--mps", required=True, help="MPS directory used by the run")
    parser.add_argument("--rgb-stream-id", default="214-1")
    parser.add_argument("--image-view", choices=["raw", "egodelta_pinhole"], default="raw")
    parser.add_argument("--aria-camera-label", default="camera-rgb")
    parser.add_argument("--rect-size", type=int, default=720)
    parser.add_argument("--rect-focal", type=float, default=380.0)
    parser.add_argument("--task", choices=["harvest", "deleaf"], default="harvest")
    parser.add_argument("--max-frames-per-split", type=int, default=20)
    parser.add_argument("--visualization-stride", type=int, default=30)
    args = parser.parse_args()
    if args.visualization_stride < 1:
        parser.error("--visualization-stride must be >= 1")

    from .mps_geometry import MPSGeometry
    from .result_projection import write_projected_result_overlays, write_task_projected_result_overlays

    config = PipelineConfig(
        rgb_stream_id=args.rgb_stream_id,
        image_view=args.image_view,
        aria_camera_label=args.aria_camera_label,
        rect_size=args.rect_size,
        rect_focal=args.rect_focal,
        visualization_stride=args.visualization_stride,
    )
    geometry = MPSGeometry.from_paths(args.mps, args.vrs, config)
    if args.task == "harvest" and (Path(args.out) / "lifted_3d" / "pre_instances.json").exists():
        summary = write_projected_result_overlays(
            out_dir=Path(args.out),
            pre_geometry=geometry,
            post_geometry=geometry,
            max_frames_per_split=args.max_frames_per_split,
            visualization_stride=config.visualization_stride,
        )
        output = Path(args.out) / "visualizations" / "projected_results"
    else:
        summary = write_task_projected_result_overlays(
            out_dir=Path(args.out),
            task_name=args.task,
            pre_geometry=geometry,
            post_geometry=geometry,
            max_frames_per_split=args.max_frames_per_split,
            visualization_stride=config.visualization_stride,
        )
        output = Path(args.out) / "visualizations" / args.task / "projected_removed_targets"
    combined_count = len(summary.get("combined_overlays", []))
    print(
        f"Wrote {len(summary['overlays'])} per-target projected overlay images "
        f"and {combined_count} combined overlay images"
    )
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
