from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .storage import AnnotationStore, summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show progress for an EgoMask annotation run")
    parser.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_run(AnnotationStore(args.run))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    columns = ("annotator", "total", "complete", "in_progress", "needs_review", "not_started")
    widths = {column: len(column) for column in columns}
    for row in summary["annotators"]:
        for column in columns:
            widths[column] = max(widths[column], len(str(row[column])))
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in summary["annotators"]:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))
    print()
    total = summary["total"]
    print(
        f"Overall: {total['complete']}/{total['total']} complete, "
        f"{total['needs_review']} flagged, {total['in_progress']} in progress, "
        f"{total['not_started']} not started"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
