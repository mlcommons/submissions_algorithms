#!/usr/bin/env python3
"""Analysis 2: the effect of measuring progress in steps vs wall-clock.

Two figures:
  1. step_vs_wallclock_scatter.png -- per (submission, workload), time-to-target
     normalized to the fastest submission for that workload, steps (x) vs
     wall-clock (y), log-log. Points above the diagonal reach the target in
     competitive step counts but lose ground on wall-clock (expensive steps).
  2. leaderboard_rank_shift.png -- leaderboard rank under a step-based score vs a
     wall-clock-based score; lines colored by direction of movement.

Inputs come from compute_metrics.py (analysis/data/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from style import FAMILY_COLORS, INK, MUTED, apply_rc  # noqa: E402
from submissions import meta_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "analysis" / "data"
DEFAULT_OUT = REPO_ROOT / "analysis" / "figures"


def _normalize_to_best(mat: pd.DataFrame) -> pd.DataFrame:
    """Per workload (column), divide by the fastest finite value -> ratio >= 1."""
    finite = mat.replace(np.inf, np.nan)
    return finite.div(finite.min(axis=0), axis=1)


def scatter(steps: pd.DataFrame, wall: pd.DataFrame, out_path: Path) -> None:
    s = _normalize_to_best(steps)
    w = _normalize_to_best(wall)
    fig, ax = plt.subplots(figsize=(8.5, 8))

    seen_families = set()
    for submission in s.index:
        fam = meta_for(submission).family
        color = FAMILY_COLORS.get(fam, MUTED)
        for workload in s.columns:
            x, y = s.loc[submission, workload], w.loc[submission, workload]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            label = fam if fam not in seen_families else None
            seen_families.add(fam)
            ax.scatter(x, y, s=70, color=color, alpha=0.8, edgecolor="white",
                       linewidth=0.6, label=label, zorder=3)

    lim_hi = float(np.nanmax([s.values, w.values])) * 1.15
    diag = np.array([1.0, lim_hi])
    ax.plot(diag, diag, color=MUTED, linestyle="--", linewidth=1.2, zorder=1,
            label="equal standing")
    ax.set(xscale="log", yscale="log", xlim=(0.9, lim_hi), ylim=(0.9, lim_hi))
    ax.set_xlabel("steps to target  (× fastest on workload)")
    ax.set_ylabel("wall-clock to target  (× fastest on workload)")
    ax.set_title(
        "Step vs wall-clock cost to reach target\n"
        "above diagonal = competitive in steps, slower in wall-clock",
        fontweight="bold", pad=12,
    )
    ax.set_aspect("equal")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="#e5e5e5", loc="lower right", title="algorithm family")
    fig.savefig(out_path)
    plt.close(fig)


def rank_shift(board: pd.DataFrame, out_path: Path) -> None:
    # Keep submissions that reach at least one target under either clock.
    board = board[(board["score_by_steps"] > 0) | (board["score_by_wallclock"] > 0)]
    fig, ax = plt.subplots(figsize=(9, 8))

    for submission, row in board.iterrows():
        r_step, r_wall = row["rank_by_steps"], row["rank_by_wallclock"]
        shift = r_wall - r_step  # <0 improves under wall-clock, >0 worsens
        if shift < 0:
            color, lw = "#0072B2", 2.6
        elif shift > 0:
            color, lw = "#D55E00", 2.6
        else:
            color, lw = MUTED, 1.4
        ax.plot([0, 1], [r_step, r_wall], color=color, linewidth=lw,
                marker="o", markersize=7, zorder=3 if shift else 2, alpha=0.9)
        label = meta_for(submission).label
        ax.text(-0.02, r_step, label, ha="right", va="center", fontsize=9, color=INK)
        # shift>0 => higher rank number => dropped (▼); shift<0 => rose (▲).
        tag = f"  (▼{shift})" if shift > 0 else (f"  (▲{-shift})" if shift < 0 else "")
        ax.text(1.02, r_wall, label + tag, ha="left", va="center", fontsize=9,
                color=color if shift else INK)

    n = len(board)
    ax.set(xlim=(-0.55, 1.6), ylim=(n + 0.5, 0.5), xticks=[0, 1])
    ax.set_xticklabels(["rank by\nsteps", "rank by\nwall-clock"], fontweight="bold")
    ax.set_yticks(range(1, n + 1))
    ax.set_ylabel("leaderboard rank")
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Leaderboard reorders under wall-clock vs steps\n"
        "red = drops (expensive steps), blue = rises (cheap steps)",
        fontweight="bold", pad=12,
    )
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    apply_rc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = pd.read_csv(args.data / "time_to_target_steps.csv", index_col=0)
    wall = pd.read_csv(args.data / "time_to_target_wallclock.csv", index_col=0)
    board = pd.read_csv(args.data / "leaderboard.csv", index_col=0)

    scatter(steps, wall, args.out_dir / "step_vs_wallclock_scatter.png")
    rank_shift(board, args.out_dir / "leaderboard_rank_shift.png")
    print(f"Wrote step-vs-wall-clock figures to {args.out_dir}")


if __name__ == "__main__":
    main()
