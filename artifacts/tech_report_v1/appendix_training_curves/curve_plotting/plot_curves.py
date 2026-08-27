#!/usr/bin/env python3
"""Pass 2 of the curve-plotting pipeline: ``curve_data.csv`` -> PNGs.

Reads the tidy CSV from ``parse_logs.py`` and renders, for each
(algorithm, workload), a 2x2 figure with one distinctly-colored curve per
submission (mean over trials with a +/-1 std band):

    top row     validation metric vs. wall-clock time / training steps
    bottom row  distance to target (log) vs. wall-clock time / training steps

The metric rows read like the training curves; the distance-to-target rows
blow up near the target, so runs that otherwise pile up on the target line stay
distinguishable (a curve simply ends when it reaches the target).

    python parse_logs.py                       # pass 1 (once per log change)
    python plot_curves.py --algo all           # pass 2 (cheap, re-runnable)
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; avoids the Qt "no display" crash.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_config import (  # noqa: E402
    ALGO_CONFIGS,
    DEFAULT_DATA_PATH,
    DEFAULT_PLOT_DIR,
    PLOT_RC_PARAMS,
    is_higher_better,
)

GRID_POINTS = 150
TIME_COL = "accumulated_submission_time"
STEP_COL = "global_step"
TARGET_COLOR = "#D0021B"
_TRIAL_KEYS = ["study", "workload_variant", "trial"]

# Distinct, CVD-safe hues (Okabe-Ito), assigned per submission within an algo so
# variants of the same optimizer never collide as look-alike shades.
DISTINCT_HUES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def recolored(submissions: dict) -> dict:
    """Give each submission a distinct hue, keeping its label and linestyle."""
    out = {}
    for i, (sub, style) in enumerate(submissions.items()):
        s = dict(style)
        s["color"] = DISTINCT_HUES[i % len(DISTINCT_HUES)]
        out[sub] = s
    return out


def interpolate_trials(trials, x_col, y_col, n=GRID_POINTS):
    """Interpolate each trial onto a shared grid; return (grid, mean, std).

    The grid spans the pooled min/max of ``x_col`` across trials. Values past
    the end of a trial are held at that trial's last observation. Returns None
    if there is no data.
    """
    xs = np.concatenate([t[x_col].to_numpy() for t in trials])
    if xs.size == 0:
        return None
    grid = np.linspace(xs.min(), xs.max(), n)
    curves = [
        np.interp(grid, t[x_col], t[y_col], right=t[y_col].iloc[-1]) for t in trials
    ]
    stacked = np.vstack(curves)
    mean = np.nanmean(stacked, axis=0)
    std = np.nan_to_num(np.nanstd(stacked, axis=0), nan=0.0)
    return grid, mean, std


def compute_ylimits(values, target_value, higher_is_better, zoom):
    """Adaptive y-axis bounds for the metric panels (matches the original)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 1.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    all_le_one = bool(np.all(values <= 1.0))

    if zoom == "percentile":
        if higher_is_better:
            pct_5 = sorted_vals[int(n * 0.05)]
            ymin = max(0.0, pct_5 * 0.95) if pct_5 > 0.1 else 0.0
            ymax = sorted_vals[-1]
            if target_value is not None:
                ymax = max(ymax, target_value)
            ymax *= 1.05
            if all_le_one:
                ymax = min(1.0, ymax)
        else:
            ymin = sorted_vals[0] * 0.95
            if target_value is not None:
                ymin = min(ymin, target_value * 0.9)
            ymin = max(0.0, ymin)
            pct_90 = sorted_vals[int(n * 0.90)]
            ymax = pct_90
            if target_value is not None:
                ymax = max(ymax, target_value * 1.5)
            if ymax <= ymin:
                ymax = ymin * 2.0 if ymin > 0 else 1.0
    else:  # log
        min_val, max_val = sorted_vals[0], sorted_vals[-1]
        if higher_is_better:
            ymin = max(0.0, min_val * 0.95)
            ymax = max_val * 1.05
            if target_value is not None:
                ymin = min(ymin, target_value * 0.95)
                ymax = max(ymax, target_value * 1.05)
            if all_le_one:
                ymax = min(1.0, ymax)
        else:
            ymin = min_val * 0.95
            if target_value is not None:
                ymin = min(ymin, target_value * 0.9)
            ymin = max(1e-6, ymin)  # strictly positive for log scale
            ymax = max_val * 1.05
            if target_value is not None:
                ymax = max(ymax, target_value * 1.05)
            if ymax <= ymin:
                ymax = ymin * 2.0
    return ymin, ymax


def _style_axis(ax):
    ax.grid(True, which="major", color="#e8e8e8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def _legend(ax):
    ax.legend(fontsize=9, frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="#e5e5e5")


def _plot_metric(ax, submissions, wdf, x_col, x_scale):
    """Metric mean + std band per submission; returns pooled values for ylim."""
    all_vals = []
    for sub, style in submissions.items():
        sdf = wdf[wdf["submission"] == sub]
        if sdf.empty:
            continue
        trials = [t for _, t in sdf.groupby(_TRIAL_KEYS)]
        res = interpolate_trials(trials, x_col, "metric_value")
        if res is None:
            continue
        grid, mean, std = res
        x = grid / x_scale
        ax.plot(x, mean, color=style["color"], linestyle=style["linestyle"],
                label=style["label"], alpha=0.95, linewidth=2.5)
        ax.fill_between(x, mean - std, mean + std, color=style["color"],
                        alpha=0.10, edgecolor="none")
        all_vals.extend(sdf["metric_value"].tolist())
    return all_vals


def _plot_gap(ax, submissions, wdf, x_col, x_scale, target, higher):
    """Distance-to-target mean + std band (log y); a curve ends when reached."""
    for sub, style in submissions.items():
        sdf = wdf[wdf["submission"] == sub]
        if sdf.empty:
            continue
        trials = [t for _, t in sdf.groupby(_TRIAL_KEYS)]
        res = interpolate_trials(trials, x_col, "metric_value")
        if res is None:
            continue
        grid, mean, std = res
        x = grid / x_scale
        gap = (target - mean) if higher else (mean - target)
        valid = gap > 0  # curve ends once the target is reached
        # The +/-1 std band transforms to gap +/- std; floor it to stay on the
        # log axis (half the run's best mean gap keeps the band proportionate).
        floor = (np.nanmin(gap[valid]) * 0.5) if valid.any() else 1e-6
        floor = max(floor, 1e-9)
        line = np.where(valid, gap, np.nan)
        lo = np.where(valid, np.clip(gap - std, floor, None), np.nan)
        hi = np.where(valid, np.clip(gap + std, floor, None), np.nan)
        ax.plot(x, line, color=style["color"], linestyle=style["linestyle"],
                label=style["label"], alpha=0.95, linewidth=2.5)
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.10, edgecolor="none")
    ax.set_yscale("log")


def plot_workload(workload, wdf, submissions, out_dir, dpi=200):
    """Render the 2x2 figure for one workload; return the saved path or None."""
    metric = wdf["metric_name"].iloc[0]
    higher = is_higher_better(metric)
    tv = wdf["target_value"].iloc[0]
    target = float(tv) if pd.notna(tv) else None
    use_log = not higher  # log y for loss-type (minimize) metrics

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    (ax_mt, ax_ms), (ax_gt, ax_gs) = axes
    fig.suptitle(f"{workload} — validation {metric}", fontweight="bold", y=0.995)

    vals = _plot_metric(ax_mt, submissions, wdf, TIME_COL, 3600.0)
    _plot_metric(ax_ms, submissions, wdf, STEP_COL, 1000.0)
    if not vals:
        plt.close(fig)
        return None

    ymin, ymax = compute_ylimits(vals, target, higher, "log")
    mlabel = f"Validation {metric.upper()}" + (" (log)" if use_log else "")
    for ax, xlab in [(ax_mt, "Wall-clock time (hours)"), (ax_ms, "Training steps (x10³)")]:
        ax.set_yscale("log" if use_log else "linear")
        ax.set_ylim(ymin, ymax)
        if target is not None:
            ax.axhline(target, color=TARGET_COLOR, linestyle=":", linewidth=1.5,
                       label=f"Target ({target:g})")
        ax.set_xlabel(xlab, fontweight="semibold")
        ax.set_ylabel(mlabel, fontweight="semibold")
        _legend(ax)
        _style_axis(ax)
    ax_mt.set_title(f"Validation {metric} vs. wall-clock time", fontsize=12)
    ax_ms.set_title(f"Validation {metric} vs. training steps", fontsize=12)

    if target is not None:
        _plot_gap(ax_gt, submissions, wdf, TIME_COL, 3600.0, target, higher)
        _plot_gap(ax_gs, submissions, wdf, STEP_COL, 1000.0, target, higher)
        glabel = f"Distance to target  |target − {metric}|  (log)"
        for ax, xlab in [(ax_gt, "Wall-clock time (hours)"), (ax_gs, "Training steps (x10³)")]:
            ax.set_xlabel(xlab, fontweight="semibold")
            ax.set_ylabel(glabel, fontweight="semibold")
            _legend(ax)
            _style_axis(ax)
        ax_gt.set_title("Distance to target vs. wall-clock time", fontsize=12)
        ax_gs.set_title("Distance to target vs. training steps", fontsize=12)
    else:
        for ax in (ax_gt, ax_gs):
            ax.axis("off")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{workload}_curves.png"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return png_path


def _render_job(job, dpi):
    workload, wdf, submissions, out_dir = job
    plt.rcParams.update(PLOT_RC_PARAMS)  # rcParams don't survive a spawn start.
    return plot_workload(workload, wdf, submissions, out_dir, dpi=dpi)


def build_jobs(data, algos, out_root):
    """Flatten the (algo, workload) grid into independent, picklable render jobs."""
    jobs = []
    for algo in algos:
        cfg = ALGO_CONFIGS[algo]
        submissions = recolored(cfg["submissions"])
        out_dir = out_root / cfg["sub_dir"]
        algo_data = data[data["submission"].isin(submissions)]
        if algo_data.empty:
            print(f"  {algo}: no matching data, skipping.")
            continue
        for workload, wdf in algo_data.groupby("workload"):
            jobs.append((workload, wdf, submissions, out_dir))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algo", default="all", choices=list(ALGO_CONFIGS) + ["all"],
                        help="Algorithm to plot (default: all).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="Tidy curve_data.csv from parse_logs.py.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOT_DIR,
                        help="Root directory for the <algo>/<workload>_curves.png outputs.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="Parallel render workers (each figure is independent). 1 = serial.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output resolution.")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"No curve data at {args.data}. Run parse_logs.py first (pass 1).")

    plt.rcParams.update(PLOT_RC_PARAMS)
    data = pd.read_csv(args.data)
    algos = list(ALGO_CONFIGS) if args.algo == "all" else [args.algo]
    print(f"Plotting algorithms: {algos}")

    jobs = build_jobs(data, algos, args.out_dir)
    if not jobs:
        print("\nNothing to plot.")
        return

    n_workers = max(1, min(args.jobs, len(jobs)))
    render = partial(_render_job, dpi=args.dpi)
    print(f"Rendering {len(jobs)} figure(s) across {n_workers} worker(s)...")
    if n_workers == 1:
        results = [render(job) for job in jobs]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(render, jobs)

    saved = [p for p in results if p is not None]
    print(f"\nDone. {len(saved)} plot(s) written.")


if __name__ == "__main__":
    main()
