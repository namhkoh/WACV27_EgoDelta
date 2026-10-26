from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def write_prediction(
    root: str | Path,
    scene_id: str,
    image_name: str,
    probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, str]:
    root = Path(root)
    stem = Path(image_name).stem
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError(f"Prediction for {scene_id}/{image_name} must be a finite HxW array")
    probability = np.clip(probability, 0.0, 1.0)
    score_path = root / scene_id / "scores" / f"{stem}.npy"
    mask_path = root / scene_id / "masks" / f"{stem}.png"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_path, probability, allow_pickle=False)
    Image.fromarray((probability > threshold).astype(np.uint8) * 255, mode="L").save(mask_path)
    return {"score": str(score_path.relative_to(root)), "mask": str(mask_path.relative_to(root))}


def write_scene_metadata(root: str | Path, scene_id: str, payload: dict[str, Any]) -> Path:
    path = Path(root) / scene_id / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_id": scene_id,
        **payload,
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
