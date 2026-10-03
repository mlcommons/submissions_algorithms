# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: algoperf
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Appendix: training curves
#
# Loads the self-tuning logs with `report_data.load_runs` and plots one figure
# per (algorithm, workload). Every submission in an algorithm family gets its
# own curve: the mean over trials with a ±1 std band. Targets and metric
# direction come from the workload config that scoring uses.
#
# Two figure sets are written to `out/`:
#
# - `training_curves/<algo>/<workload>_curves.{pdf,png}` is a 2×2 grid. The top
#   row is the validation metric and the bottom row is the distance to the
#   target on a log axis. The left column is against wall-clock time and the
#   right column is against training steps.
# - `schedule_free/<workload>_curves.{pdf,png}` compares the Schedule-Free
#   AdamW implementations side by side (metric against time and steps).
#
# Setup: `uv sync --extra dev --extra analysis`. Select that environment's
# Python kernel, then Restart and Run All.

# %%
# Notebook cells intentionally import dependencies near their use.
# ruff: noqa: E402
import re
import sys
from pathlib import Path

REPO_ROOT = next(
  p
  for p in [Path.cwd(), *Path.cwd().parents]
  if (p / 'scoring/score_submissions.py').is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from artifacts.tech_report_v1.report_data import load_runs
from artifacts.tech_report_v1.report_utils import (
  COLORS,
  NOTEBOOK_STYLE,
  latex_name,
  pretty,
  save_figure,
  section_output_dir,
  set_plot_style,
  submission_directory,
  workload_name,
)
from scoring.config import DEFAULT_TARGETS_PATH, WorkloadConfig

# %% [markdown]
# ## Inputs and outputs
#
# `ALGORITHMS` sets which submissions share a figure. Submissions with no log
# directory are skipped with a message. Set `SHOW_FIGURES = False` to only
# write the files without displaying them inline.

# %%
SCHEDULE_FREE = (
  'schedule_free_adamw',
  'schedule_free_adamw_v2',
  'schedule_free_adamw_jax',
  'schedule_free_adamw_jax_v2',
)
ALGORITHMS = {
  'sfadamw_ademamix': (*SCHEDULE_FREE, 'ademamix', 'ademamix_golden'),
  'muon_lion': ('muon_torch', 'muon_torch_jax_hps', 'muon', 'lion'),
  'nadamw_diloco': (
    'nadamw',
    'nadamw_baselinev05',
    'cautious_nadamw',
    'single_worker_dilocov2',
  ),
}
SCHEDULE_FREE_ZOOM = 'log'  # 'log' or 'percentile' (trims early spikes)
SHOW_FIGURES = True

SUBMISSION_DIRECTORY = submission_directory()
OUTPUT_DIR = section_output_dir('appendix')
WORKLOAD_CONFIG = WorkloadConfig.from_json(DEFAULT_TARGETS_PATH)

GRID_POINTS = 150
TIME_COL = 'accumulated_submission_time'
STEP_COL = 'global_step'
TARGET_COLOR = '#555555'
X_AXES = (
  (TIME_COL, 3600.0, 'Wall-clock time (hours)', 'wall-clock time'),
  (STEP_COL, 1000.0, 'Training steps (×10³)', 'training steps'),
)

set_plot_style({**NOTEBOOK_STYLE, 'figure.titlesize': 13})

# %% [markdown]
# ## Load runs
#
# `load_runs` returns one row per trial, and each column holds that trial's
# evaluation series. The frames are reshaped into
# `curves[raw_submission][workload]`, a list of per-trial frames with the time,
# step and target-metric columns. JAX and PyTorch runs of a workload share one
# key so that both appear in the same figure.

# %%
requested = sorted({s for subs in ALGORITHMS.values() for s in subs})
available = [s for s in requested if (SUBMISSION_DIRECTORY / s).is_dir()]
for missing in sorted(set(requested) - set(available)):
  print(f'Skipping {missing}: no logs under {SUBMISSION_DIRECTORY}')

runs = load_runs(SUBMISSION_DIRECTORY, include=available)


def base_workload(name):
  return re.sub(r'_(jax|pytorch)$', '', name)


def trial_curves(frame):
  """Group one submission's trials by workload as tidy per-trial frames."""
  curves = {}
  for _, row in frame.iterrows():
    workload = base_workload(row['workload'])
    metric_col, _ = WORKLOAD_CONFIG.metric_and_target(workload)
    if not isinstance(row.get(metric_col), np.ndarray):
      continue
    trial = pd.DataFrame(
      {
        TIME_COL: row[TIME_COL],
        STEP_COL: row[STEP_COL],
        'metric': row[metric_col],
      }
    ).dropna()
    if not trial.empty:
      curves.setdefault(workload, []).append(trial)
  return curves


curves = {raw: trial_curves(runs[pretty(raw)]) for raw in available}
display(
  pd.DataFrame(
    {raw: {w: len(t) for w, t in c.items()} for raw, c in curves.items()}
  )
  .fillna(0)
  .astype(int)
  .rename_axis('trials per workload')
)

# %% [markdown]
# ## Plotting helpers
#
# Each submission's trials are interpolated onto a shared grid over their
# pooled x-range. After a trial's last evaluation its final value is held. The
# distance-to-target curve stops once the mean reaches the target. Its ±1 std
# band is clipped to stay positive on the log axis.


# %%
def submission_styles(subs):
  """Distinct color per submission; dashed lines mark JAX implementations."""
  return {
    raw: dict(
      color=COLORS[i % len(COLORS)],
      linestyle='--' if '(JAX)' in pretty(raw) else '-',
      label=pretty(raw),
    )
    for i, raw in enumerate(subs)
  }


# (v1, v2) shades per framework, from report_utils.COLORS.
FRAMEWORK_SHADES = {'PyTorch': ('#4477AA', '#66CCEE'), 'JAX': ('#CC3311', '#EE7733')}


def framework_styles(subs):
  """Blue shades for PyTorch, red/orange for JAX; v1 dashed, v2 solid."""
  styles = {}
  for raw in subs:
    v2 = raw.endswith('_v2')
    framework = 'JAX' if '(JAX)' in pretty(raw) else 'PyTorch'
    styles[raw] = dict(
      color=FRAMEWORK_SHADES[framework][v2],
      linestyle='-' if v2 else '--',
      label=pretty(raw),
    )
  return styles


def mean_std(trials, x_col):
  x = np.concatenate([t[x_col].to_numpy() for t in trials])
  grid = np.linspace(x.min(), x.max(), GRID_POINTS)
  stacked = np.vstack(
    [
      np.interp(grid, t[x_col], t['metric'], right=t['metric'].iloc[-1])
      for t in trials
    ]
  )
  return grid, np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def metric_ylimits(values, target, minimized, zoom='log'):
  """Y-limits that keep the target visible; 'percentile' trims early spikes."""
  v = np.sort(np.asarray(values, dtype=float))
  bounded = v[-1] <= 1.0
  if zoom == 'percentile':
    if minimized:
      lo = max(0.0, min(v[0] * 0.95, target * 0.9))
      hi = max(v[int(len(v) * 0.9)], target * 1.5)
    else:
      p5 = v[int(len(v) * 0.05)]
      lo = max(0.0, p5 * 0.95) if p5 > 0.1 else 0.0
      hi = max(v[-1], target) * 1.05
  elif minimized:
    lo = max(1e-6, min(v[0] * 0.95, target * 0.9))
    hi = max(v[-1], target) * 1.05
  else:
    lo = max(0.0, min(v[0], target) * 0.95)
    hi = max(v[-1], target) * 1.05
  if not minimized and bounded:
    hi = min(1.0, hi)
  if hi <= lo:
    hi = lo * 2 or 1.0
  return lo, hi


def plot_metric(ax, workload, styles, x_col, x_scale, zoom='log'):
  _, target = WORKLOAD_CONFIG.metric_and_target(workload)
  minimized = WORKLOAD_CONFIG.target_is_minimized(workload)
  values = []
  for raw, style in styles.items():
    trials = curves.get(raw, {}).get(workload)
    if not trials:
      continue
    grid, mean, std = mean_std(trials, x_col)
    x = grid / x_scale
    ax.plot(x, mean, **style)
    ax.fill_between(x, mean - std, mean + std, color=style['color'], alpha=0.12)
    values.extend(np.concatenate([t['metric'] for t in trials]))
  ax.axhline(
    target, color=TARGET_COLOR, linestyle=':', label=f'Target ({target:g})'
  )
  ax.set_yscale('log' if minimized and zoom == 'log' else 'linear')
  ax.set_ylim(*metric_ylimits(values, target, minimized, zoom))


def plot_gap(ax, workload, styles, x_col, x_scale):
  _, target = WORKLOAD_CONFIG.metric_and_target(workload)
  sign = 1 if WORKLOAD_CONFIG.target_is_minimized(workload) else -1
  for raw, style in styles.items():
    trials = curves.get(raw, {}).get(workload)
    if not trials:
      continue
    grid, mean, std = mean_std(trials, x_col)
    gap = sign * (mean - target)
    valid = gap > 0
    floor = max(gap[valid].min() * 0.5 if valid.any() else 1e-6, 1e-9)
    x = grid / x_scale
    ax.plot(x, np.where(valid, gap, np.nan), **style)
    ax.fill_between(
      x,
      np.where(valid, np.clip(gap - std, floor, None), np.nan),
      np.where(valid, np.clip(gap + std, floor, None), np.nan),
      color=style['color'],
      alpha=0.12,
    )
  ax.set_yscale('log')


def finish(fig, path):
  paths = save_figure(fig, path)
  if SHOW_FIGURES:
    display(fig)
  plt.close(fig)
  return paths


# %% [markdown]
# ## All training curves
#
# One 2×2 figure per (algorithm, workload).

# %%
saved = []
for algo, subs in ALGORITHMS.items():
  styles = submission_styles([s for s in subs if s in curves])
  workloads = sorted({w for raw in styles for w in curves[raw]})
  for workload in workloads:
    metric = WORKLOAD_CONFIG.metric_and_target(workload)[0].split('/')[-1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(
      f'{workload_name(workload)}: validation {metric}', fontweight='bold'
    )
    for col, (x_col, x_scale, xlabel, name) in enumerate(X_AXES):
      top, bottom = axes[0, col], axes[1, col]
      plot_metric(top, workload, styles, x_col, x_scale)
      top.set(
        title=f'Validation {metric} vs. {name}',
        xlabel=xlabel,
        ylabel=f'Validation {metric}',
      )
      plot_gap(bottom, workload, styles, x_col, x_scale)
      bottom.set(
        title=f'Distance to target vs. {name}',
        xlabel=xlabel,
        ylabel=f'|target − {metric}| (log)',
      )
      top.legend()
      bottom.legend()
    fig.tight_layout()
    saved += finish(
      fig, OUTPUT_DIR / 'training_curves' / algo / f'{workload}_curves'
    )
print(f'Wrote {len(saved)} files under {OUTPUT_DIR / "training_curves"}')

# %% [markdown]
# ## Aggregate training curves
#
# One figure per algorithm family, with a subplot per workload showing the
# validation target metric against wall-clock time. Each figure is saved to
# `out/training_curves_aggregate/<algo>.{pdf,png}`.

# %%
from matplotlib.lines import Line2D

NCOLS = 3
x_col, x_scale, xlabel, _ = X_AXES[0]
for algo, subs in ALGORITHMS.items():
  styles = submission_styles([s for s in subs if s in curves])
  workloads = sorted({w for raw in styles for w in curves[raw]})
  nrows = -(-len(workloads) // NCOLS)
  fig, axes = plt.subplots(
    nrows, NCOLS, figsize=(4.2 * NCOLS, 3.2 * nrows), squeeze=False
  )
  for ax, workload in zip(axes.flat, workloads):
    metric = WORKLOAD_CONFIG.metric_and_target(workload)[0].split('/')[-1]
    plot_metric(ax, workload, styles, x_col, x_scale)
    ax.set(title=workload_name(workload), xlabel=xlabel, ylabel=metric)
  for ax in axes.flat[len(workloads) :]:
    ax.axis('off')
  handles = [Line2D([], [], **style) for style in styles.values()]
  handles.append(Line2D([], [], color=TARGET_COLOR, linestyle=':', label='Target'))
  fig.legend(
    handles=handles,
    loc='lower center',
    ncol=len(handles),
    bbox_to_anchor=(0.5, 0.0),
  )
  fig.tight_layout(rect=(0, 0.05, 1, 1))
  finish(fig, OUTPUT_DIR / 'training_curves_aggregate' / algo)

# %% [markdown]
# ## Schedule-Free AdamW implementations
#
# The PyTorch and JAX Schedule-Free AdamW submissions (v1 and v2) for each
# workload, plotted against wall-clock time and training steps. PyTorch runs
# are blue and JAX runs red/orange; v1 is dashed and v2 solid. Set
# `SCHEDULE_FREE_ZOOM = 'percentile'` to use a linear axis that trims the
# early-training spikes.

# %%
styles = framework_styles(
  [s for s in SCHEDULE_FREE if s in curves]
)
workloads = sorted({w for raw in styles for w in curves[raw]})
for workload in workloads:
  metric = WORKLOAD_CONFIG.metric_and_target(workload)[0].split('/')[-1]
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
  fig.suptitle(
    f'{workload_name(workload)}: validation {metric}', fontweight='bold'
  )
  for ax, (x_col, x_scale, xlabel, _) in zip(axes, X_AXES):
    plot_metric(ax, workload, styles, x_col, x_scale, zoom=SCHEDULE_FREE_ZOOM)
    ax.set(xlabel=xlabel, ylabel=f'Validation {metric}')
    ax.legend()
  fig.tight_layout()
  finish(fig, OUTPUT_DIR / 'schedule_free' / f'{workload}_curves')

# %% [markdown]
# ## Step-time table
#
# For each trial, the average time per step is the submission time between
# consecutive evaluations divided by the steps between them. The interval
# between the first two evaluations is dropped because it can include
# compilation time. Intervals are pooled with weights by their step count
# (total time over total steps), and the result is averaged over trials.
# Values are normalized per workload to the NAdamW baseline
# (`nadamw_baselinev05`). Each submission runs at its own batch size. The raw
# seconds per step are saved next to the LaTeX table.

# %%
from artifacts.tech_report_v1.report_tables import WORKLOAD_LATEX, workload_table

STEP_TIME_REFERENCE = 'nadamw_baselinev05'


def seconds_per_step(row):
  t = np.asarray(row[TIME_COL], dtype=float)
  s = np.asarray(row[STEP_COL], dtype=float)
  keep = ~(np.isnan(t) | np.isnan(s))
  dt, ds = np.diff(t[keep])[1:], np.diff(s[keep])[1:]
  valid = ds > 0
  return dt[valid].sum() / ds[valid].sum() if valid.any() else np.nan


step_rows = [
  dict(
    submission=raw,
    workload=base_workload(row['workload']),
    seconds_per_step=seconds_per_step(row),
  )
  for raw in available
  for _, row in runs[pretty(raw)].iterrows()
]
step_times = (
  pd.DataFrame(step_rows)
  .groupby(['submission', 'workload'])['seconds_per_step']
  .mean()
  .unstack()[list(WORKLOAD_LATEX)]
)
normalized = step_times / step_times.loc[STEP_TIME_REFERENCE]

order = [s for subs in ALGORITHMS.values() for s in subs if s in step_times.index]
sections = {
  'JAX submissions': [s for s in order if '(JAX)' in pretty(s)],
  'PyTorch submissions': [s for s in order if '(JAX)' not in pretty(s)],
}
rows = [s for names in sections.values() for s in names]
table = normalized.loc[rows].map(lambda v: f'${v:.2f}$' if pd.notna(v) else '--')
table.index = table.index.map(pretty)
step_time_latex = workload_table(
  table,
  caption=(
    'Step execution times, normalized to '
    f'{latex_name(pretty(STEP_TIME_REFERENCE))}. Each value is the ratio of '
    "a submission's average time per step to the reference's on that "
    'workload, excluding the first evaluation interval to remove '
    'compilation time; submissions run at their own batch sizes.'
  ),
  label='tab:step_time_comparison',
  sections={
    title: [pretty(s) for s in names] for title, names in sections.items()
  },
).replace(r'\begin{table}', r'\begin{table*}').replace(
  r'\end{table}', r'\end{table*}'
)

byproducts = OUTPUT_DIR / 'byproducts'
byproducts.mkdir(parents=True, exist_ok=True)
step_times.rename(index=pretty).to_csv(byproducts / 'seconds_per_step.csv')
(OUTPUT_DIR / 'step_time_table.tex').write_text(step_time_latex + '\n')
display(normalized.loc[rows].rename(index=pretty).round(2))
print(step_time_latex)
