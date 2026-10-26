from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one immutable 30-rollout batch from the registered ledger.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    session = args.session_dir.resolve()
    ledger = session / "robot_trial_ledger.csv"
    output = session / "batches" / f"{args.batch_id}.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite batch export: {output}")
    with ledger.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["batch_id"] == args.batch_id]
    if not fieldnames:
        raise ValueError("registered ledger has no columns")
    if len(rows) != 30:
        raise ValueError(f"expected exactly 30 rows in {args.batch_id}, found {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"batch_id": args.batch_id, "output": str(output), "trial_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
