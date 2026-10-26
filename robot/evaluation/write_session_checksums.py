from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a deterministic SHA-256 inventory for a trial session.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="SHA256SUMS.txt")
    args = parser.parse_args()
    root = args.session_dir.resolve()
    output = root / args.output_name
    files = sorted(item for item in root.rglob("*") if item.is_file() and item != output)
    lines = [f"{sha256_file(item)}  {item.relative_to(root).as_posix()}" for item in files]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(output)
    print(f"wrote {len(lines)} checksums to {output}")


if __name__ == "__main__":
    main()
