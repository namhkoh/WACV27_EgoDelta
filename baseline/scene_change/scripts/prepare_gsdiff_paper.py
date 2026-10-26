#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoscene.paper import prepare_paslcd_pgsr


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare independent PASLCD inputs for paper-faithful PGSR")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_paslcd_pgsr(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
