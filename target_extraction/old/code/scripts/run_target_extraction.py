#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egodelta_diff.data import discover_ready_pairs, make_scene_pair
from egodelta_diff.pipeline import EgoDeltaDiffPipeline

DEFAULT_PRE_ROOT = Path("pre_models")
DEFAULT_POST_ROOT = Path("post_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EGODELTA-Diff for 2DGS pre/post model pairs.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for a single pair.")
    parser.add_argument("--out-root", type=Path, default=Path("outputs"), help="Output root for batch mode.")
    parser.add_argument("--query-views", choices=["pre", "post", "all", "pre_added"], default=None)
    parser.add_argument("--max-primitives", type=int, default=None, help="Smoke-test primitive cap per model.")
    parser.add_argument("--device", default=None, help="Override compute.device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow CPU fallback when CUDA is requested but unavailable.")
    parser.add_argument("--disable-render", action="store_true")
    parser.add_argument("--disable-overlay", action="store_true")

    single = parser.add_argument_group("single pair")
    single.add_argument("--pre-model", type=Path)
    single.add_argument("--post-model", type=Path)
    single.add_argument("--pre-ply", type=Path)
    single.add_argument("--post-ply", type=Path)
    single.add_argument("--pre-cameras", type=Path)
    single.add_argument("--post-cameras", type=Path)

    batch = parser.add_argument_group("batch")
    batch.add_argument("--pre-root", type=Path, default=DEFAULT_PRE_ROOT)
    batch.add_argument("--post-root", type=Path, default=DEFAULT_POST_ROOT)
    batch.add_argument("--only-ready", action="store_true", help="Run all ready basename-matched pairs.")
    batch.add_argument("--skip-existing", action="store_true", help="Skip pairs whose output directory already has core artifacts.")
    batch.add_argument("--num-shards", type=int, default=1, help="Split ready pairs into this many modulo shards.")
    batch.add_argument("--shard-index", type=int, default=0, help="Run only pairs where index %% num_shards equals this value.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.max_primitives is not None:
        config.setdefault("compute", {})["max_primitives"] = args.max_primitives
    if args.device is not None:
        config.setdefault("compute", {})["device"] = args.device
    if args.allow_cpu_fallback:
        config.setdefault("compute", {})["allow_cpu_fallback"] = True
    if args.disable_render:
        config.setdefault("render", {})["enabled"] = False
    if args.disable_overlay:
        config.setdefault("overlay", {})["enabled"] = False
    return config


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    pipeline = EgoDeltaDiffPipeline(config)

    if args.only_ready:
        pairs = discover_ready_pairs(args.pre_root, args.post_root)
        if not pairs:
            raise SystemExit("No ready pre/post pairs found.")
        if args.num_shards < 1:
            raise SystemExit("--num-shards must be >= 1")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards")
        if args.num_shards > 1:
            pairs = [pair for idx, pair in enumerate(pairs) if idx % args.num_shards == args.shard_index]
            if not pairs:
                print(f"[egodelta-diff] shard {args.shard_index}/{args.num_shards} has no ready pairs")
                return
        for pair in pairs:
            out_dir = args.out_root / pair.name
            if args.skip_existing and (out_dir / "diagnostics.json").exists() and (out_dir / "removed_primitives.ply").exists():
                print(f"[egodelta-diff] skip existing {pair.name} -> {out_dir}")
                continue
            print(f"[egodelta-diff] {pair.name} -> {out_dir}")
            result = pipeline.run_pair(pair)
            pipeline.save(result, out_dir, query_views=args.query_views)
        return

    if args.pre_model and args.post_model:
        pair = make_scene_pair(args.pre_model, args.post_model)
        out_dir = args.out_dir or (args.out_root / pair.name)
        result = pipeline.run_pair(pair)
        pipeline.save(result, out_dir, query_views=args.query_views)
        return

    required = [args.pre_ply, args.post_ply, args.pre_cameras, args.post_cameras]
    if all(required):
        out_dir = args.out_dir or args.out_root / "single_pair"
        result = pipeline.run(args.pre_ply, args.post_ply, args.pre_cameras, args.post_cameras)
        pipeline.save(result, out_dir, query_views=args.query_views)
        return

    raise SystemExit("Provide --only-ready, --pre-model/--post-model, or explicit --pre-ply/--post-ply/--pre-cameras/--post-cameras.")


if __name__ == "__main__":
    main()

