from __future__ import annotations

from pathlib import Path

import numpy as np


def _hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        plt.figure(figsize=(6, 4))
        if len(values):
            plt.hist(values, bins=60, color="#2f6f9f", alpha=0.85)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        path.write_text(f"Plot unavailable: {title}\n", encoding="utf-8")


def write_diagnostic_plots(
    plots_dir: str | Path,
    nn_distances: np.ndarray,
    m3c2_distances: np.ndarray | None,
    cluster_sizes: list[int],
) -> None:
    plots_dir = Path(plots_dir)
    _hist(plots_dir / "nn_distance_hist.png", nn_distances, "Pre-to-post NN distances", "distance (m)")
    if m3c2_distances is not None:
        _hist(plots_dir / "m3c2_distance_hist.png", m3c2_distances, "M3C2 distances", "distance (m)")
    _hist(plots_dir / "cluster_size_hist.png", np.asarray(cluster_sizes, dtype=float), "Cluster sizes", "points")
