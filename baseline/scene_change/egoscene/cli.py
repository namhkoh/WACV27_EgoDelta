from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .benchmark import discover_scenes
from .evaluate import evaluate_predictions
from .gsdiff import run_gsdiff_scene
from .learned import collect_3dgscd_predictions, render_mv3dcd_queries
from .prepare import prepare_3dgscd_scene, prepare_mv3dcd_scene


DEFAULT_DATA = Path("/path/to/data/mps/egodelta_benchmark_data")
DEFAULT_MANIFEST = Path("/path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views/manifest.json")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="egoscene", description="EgoDelta scene-change baselines")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    common.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    common.add_argument("--scene", action="append", dest="scenes", help="Scene ID; repeat to select several")
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", parents=[common])
    prepare.add_argument("--baseline", required=True, choices=("3dgscd", "mv3dcd"))
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--post-views", type=int, default=10)

    gsdiff = commands.add_parser("gsdiff", parents=[common])
    gsdiff.add_argument("--output-root", type=Path, required=True)
    gsdiff.add_argument("--device", default="cuda")
    gsdiff.add_argument("--threshold", type=float, default=0.5)
    gsdiff.add_argument("--max-neighbors", type=int, default=0, help="0 keeps every paper-defined neighbor")
    gsdiff.add_argument("--max-primitives", type=int, default=0, help="Debug only; 0 uses all primitives")

    egodelta = commands.add_parser("egodelta", parents=[common])
    egodelta.add_argument("--cache-root", type=Path, required=True)
    egodelta.add_argument("--output-root", type=Path, required=True)
    egodelta.add_argument(
        "--annotator",
        help="Render only the fixed manifest views assigned to this annotator.",
    )

    collect = commands.add_parser("collect-3dgscd", parents=[common])
    collect.add_argument("--official-output-root", type=Path, required=True)
    collect.add_argument("--output-root", type=Path, required=True)
    collect.add_argument("--threshold", type=float, default=0.5)

    render = commands.add_parser("render-mv3dcd", parents=[common])
    render.add_argument("--model-root", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--upstream-root", type=Path, required=True)
    render.add_argument("--iteration", type=int, default=13_000)
    render.add_argument("--alpha-iteration", type=int, default=7_000)
    render.add_argument("--threshold", type=float, default=0.5)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--gt-export", type=Path, required=True)
    evaluate.add_argument("--predictions-root", type=Path, required=True)
    evaluate.add_argument("--output-json", type=Path)
    evaluate.add_argument("--output-csv", type=Path)
    evaluate.add_argument(
        "--allow-incomplete-predictions",
        action="store_true",
        help="Score only scene directories with completion metadata and report coverage.",
    )
    return root


def _scenes(args: argparse.Namespace):
    return discover_scenes(
        args.manifest,
        pairs_root=args.data_root / "pairs_fast",
        pre_models_root=args.data_root / "pre_models",
        post_models_root=args.data_root / "post_models",
        scene_ids=args.scenes,
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_predictions(
            args.gt_export,
            args.predictions_root,
            output_json=args.output_json,
            output_csv=args.output_csv,
            allow_incomplete_predictions=args.allow_incomplete_predictions,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "image_count",
                        "expected_image_count",
                        "scene_intervention_count",
                        "expected_scene_intervention_count",
                        "prediction_coverage",
                        "macro_scene",
                    )
                },
                indent=2,
            )
        )
        return 0
    scenes = _scenes(args)
    for index, scene in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {scene.scene_id}", flush=True)
        if args.command == "prepare":
            destination = args.output_root / scene.scene_id
            if args.baseline == "3dgscd":
                prepare_3dgscd_scene(scene, destination, post_views=args.post_views)
            else:
                prepare_mv3dcd_scene(scene, destination, post_views=args.post_views)
        elif args.command == "gsdiff":
            run_gsdiff_scene(
                scene,
                args.output_root,
                device=args.device,
                threshold=args.threshold,
                max_neighbors=args.max_neighbors or None,
                max_primitives=args.max_primitives or None,
            )
        elif args.command == "egodelta":
            from .egodelta import run_egodelta_scene

            run_egodelta_scene(
                scene,
                args.cache_root,
                args.output_root,
                annotator=args.annotator,
            )
        elif args.command == "collect-3dgscd":
            collect_3dgscd_predictions(
                scene,
                args.official_output_root / scene.scene_id,
                args.output_root,
                threshold=args.threshold,
            )
        elif args.command == "render-mv3dcd":
            render_mv3dcd_queries(
                scene,
                args.model_root / scene.scene_id,
                args.output_root,
                args.upstream_root,
                iteration=args.iteration,
                alpha_iteration=args.alpha_iteration,
                threshold=args.threshold,
            )
        else:
            raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
