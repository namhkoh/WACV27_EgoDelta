"""Render a visual WITH-MPS vs MINUS-MPS comparison for one harvest and one deleaf example.

Reads the egodelta_mps_ablation JSON (raw-prior variant, which carries the per-arm action
axis vectors) and draws a 2-panel 3D figure:
  - Harvest (tomato0): grasp action axes (approach / jaw / pull). Human = EgoDelta WITH
    (solid) vs the prior-free robot-front default (dashed).
  - Deleaf (deleafing_01): blade descent + horizontal heading. Human = EgoDelta WITH (solid)
    vs the prior-free vertical straight-down default (dashed), with a world -Z reference.

Needs only numpy + matplotlib (run with an env that has matplotlib, e.g. `aria`). Output PNG
lands in the gitignored runs dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "robot" / "ablation" / "egodelta_mps_ablation_results.json"
OUT = REPO / "robot" / "ablation" / "egodelta_mps_ablation_figure.png"

WITH_COLOR = "#1f77b4"   # EgoDelta WITH MPS (= human)
MINUS_COLOR = "#d62728"  # no prior
REF_COLOR = "#888888"


def _row(rows, action, arm):
    return next(r for r in rows if r["action"] == action and r["arm"] == arm)


def _u(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _angle(u, v, bidirectional=False):
    a = float(np.degrees(np.arccos(np.clip(float(_u(u) @ _u(v)), -1.0, 1.0))))
    if bidirectional and a > 90.0:
        a = 180.0 - a
    return a


def _arrow(ax, vec, color, ls, lw, label=None):
    v = _u(vec)
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, linewidth=lw, linestyle=ls,
              arrow_length_ratio=0.16, length=1.0)
    if label:
        ax.text(v[0] * 1.20, v[1] * 1.20, v[2] * 1.20, label, color=color, fontsize=8.5,
                ha="center", va="center", weight="bold")


def _style(ax, title):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("robot X (fwd)", fontsize=8)
    ax.set_ylabel("robot Y (left)", fontsize=8)
    ax.set_zlabel("robot Z (up)", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=18, azim=-60)
    ax.set_title(title, fontsize=11, pad=2)


def main() -> int:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    harvest, deleaf = data["harvest"], data["deleaf"]

    fig = plt.figure(figsize=(13.5, 6.2))

    # ---- Panel A: harvest tomato0 ----
    axA = fig.add_subplot(121, projection="3d")
    hw = _row(harvest, "tomato0", "WITH")
    hm = _row(harvest, "tomato0", "MINUS")
    for key, name in (("approach_axis_robot", "approach"), ("jaw_axis_robot", "jaw"), ("pull_axis_robot", "pull")):
        _arrow(axA, hw[key], WITH_COLOR, "-", 3.0, name)
        _arrow(axA, hm[key], MINUS_COLOR, "--", 1.8, None)
    _style(axA, "Harvest grasp — tomato0")
    txtA = (
        f"approach  Δ = {hm['approach_axis_err_deg']:.0f}°\n"
        f"jaw          Δ = {hm['jaw_axis_err_deg']:.0f}°\n"
        f"pull         Δ = {hm['pull_axis_err_deg']:.0f}°"
    )
    axA.text2D(0.02, 0.96, txtA, transform=axA.transAxes, fontsize=9, va="top",
               family="monospace", bbox=dict(boxstyle="round", fc="white", ec=REF_COLOR, alpha=0.85))

    # ---- Panel B: deleaf deleafing_01 ----
    axB = fig.add_subplot(122, projection="3d")
    dw = _row(deleaf, "deleafing_01", "WITH")
    dm = _row(deleaf, "deleafing_01", "MINUS")
    _arrow(axB, [0, 0, -1], REF_COLOR, ":", 1.4, None)
    _arrow(axB, dw["human_cut_axis_robot"], WITH_COLOR, "-", 3.0, "descent")
    _arrow(axB, dw["human_heading_axis_robot"], WITH_COLOR, "-", 2.2, "heading")
    _arrow(axB, dm["planned_cut_axis_robot"], MINUS_COLOR, "--", 1.8, None)
    _arrow(axB, dm["planned_heading_axis_robot"], MINUS_COLOR, "--", 1.6, None)
    _style(axB, "Deleaf cut — deleafing_01")
    txtB = (
        f"descent   Δ = {dm['cut_axis_err_deg']:.0f}°\n"
        f"heading   Δ = {dm['heading_axis_err_deg']:.0f}°\n"
        f"orient.     Δ = {dm['orient_dev_from_human_deg']:.0f}°"
    )
    axB.text2D(0.02, 0.96, txtB, transform=axB.transAxes, fontsize=9, va="top",
               family="monospace", bbox=dict(boxstyle="round", fc="white", ec=REF_COLOR, alpha=0.85))

    # ---- shared legend ----
    proxies = [
        plt.Line2D([0], [0], color=WITH_COLOR, lw=3.0, ls="-"),
        plt.Line2D([0], [0], color=MINUS_COLOR, lw=1.8, ls="--"),
        plt.Line2D([0], [0], color=REF_COLOR, lw=1.4, ls=":"),
    ]
    fig.legend(
        proxies,
        ["EgoDelta WITH MPS prior  (= human demonstration)", "No MPS prior  (robot-front / vertical default)", "world straight-down reference"],
        loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "EgoDelta MPS prior aligns the robot action with the human demonstration "
        "(Δ = deviation from human)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
