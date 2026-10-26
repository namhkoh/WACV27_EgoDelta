#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoscene.qualitative import create_figure3_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the final Figure-3-style qualitative grid")
    parser.add_argument("--gt-export", type=Path, required=True)
    parser.add_argument("--3dgscd-root", dest="threedgscd_root", type=Path, required=True)
    parser.add_argument("--mv3dcd-root", type=Path, required=True)
    parser.add_argument("--gsdiff-root", type=Path, required=True)
    parser.add_argument("--egodelta-root", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--scenes-per-kind", type=int, default=5)
    args = parser.parse_args()
    result = create_figure3_comparison(
        args.gt_export,
        {
            "3DGS-CD": args.threedgscd_root,
            "MV3DCD": args.mv3dcd_root,
            "GS-DIFF": args.gsdiff_root,
            "EgoDelta": args.egodelta_root,
        },
        args.output_stem,
        scenes_per_kind=args.scenes_per_kind,
        seed=args.seed,
    )
    print(json.dumps(result["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
