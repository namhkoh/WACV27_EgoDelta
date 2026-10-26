#!/usr/bin/env python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPL_PATH = ROOT / "scripts" / "tune_deleafing_8sample_50.py"


def _load_impl():
    spec = importlib.util.spec_from_file_location("_tune_deleafing_8sample_impl", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tuning implementation from {IMPL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(_load_impl().main())
