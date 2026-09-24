"""Render the cached leaderboard as a ranked legend beside its profiles.

Run with ``uv run python artifacts/tech_report_v1/section_4_leaderboards/
plot_leaderboard_profile.py`` from the repository root. No logs are rescored.
Pass ``--metric global_step`` for the step-count leaderboard and profiles.
Pass ``--metric score_relaxed`` for the 10%-relaxed-target leaderboard.
The layout follows Figure 1 of https://arxiv.org/abs/2502.15015: the table
is the legend, with identical line-and-marker keys in both panels.
"""

import argparse
import math

import matplotlib as mpl
from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

from artifacts.tech_report_v1.report_utils import (
  REPORT_ROOT,
  SERIF_STYLE,
  NOTEBOOK_STYLE,
  set_plot_style,
  save_figure,
  submission_styles,
)

HERE = REPORT_ROOT / 'section_4_leaderboards'
PROFILE_MARKER_SPACING = 14
PROFILE_MARKER_OFFSETS = 12
POINTS_PER_INCH = 72
LEGEND_ROWS = 4
AXIS_LABEL_SPACE_PT = 40


def load_data(metric='score', byproducts=None):
  byproducts = (
    Path(byproducts) if byproducts is not None else HERE / 'out/byproducts'
  )
  score_file = {
    'score': 'scores.csv',
    'global_step': 'scores_steps.csv',
    'score_relaxed': 'scores_relaxed.csv',
  }[metric]
  scores = pd.read_csv(byproducts / score_file, index_col=0)['score']
  profiles = pd.read_csv(
    byproducts / f'performance_profile_{metric}.csv', index_col=0
  )
  return validate_profiles(scores, profiles)


def validate_profiles(scores, profiles):
  """Check profile integrals and sort rows consistently for display."""
  profiles = profiles.copy()
  profiles.columns = profiles.columns.astype(float)
  assert scores.index.is_unique and profiles.index.is_unique
  assert set(scores.index) == set(profiles.index)
  scores = scores.sort_values(ascending=False, kind='stable')
  profiles = profiles.loc[scores.index]
  tau = profiles.columns.to_numpy()
  assert np.all(np.diff(tau) > 0)
  assert np.all(np.diff(profiles.to_numpy(), axis=1) >= -1e-12)
  assert np.all((profiles.to_numpy() >= 0) & (profiles.to_numpy() <= 1))
  np.testing.assert_allclose(
    np.trapezoid(profiles.to_numpy(), x=tau) / (tau[-1] - tau[0]),
    scores.to_numpy(),
    rtol=0,
    atol=1e-12,
  )
  return scores, profiles


def _draw_profiles(ax, profiles, *, ranked):
  styles = submission_styles(profiles.index)
  for rank, name in enumerate(profiles.index, start=1):
    style = styles[name].copy()
    if ranked:
      style.update(
        markevery=(rank % PROFILE_MARKER_OFFSETS, PROFILE_MARKER_SPACING),
        markersize=4.8,
        zorder=len(profiles) - rank + 2,
      )
    else:
      style.pop('marker')
    ax.plot(
      profiles.columns,
      profiles.loc[name],
      label=name,
      **style,
      linewidth=1.5 if ranked else 1.6,
      alpha=0.95 if ranked else 0.92,
    )
  ax.set_xlim(profiles.columns.min(), profiles.columns.max())
  ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def plot_performance_profiles(
  perf_df,
  figsize=(9, 5.5),
  title=None,
):
  """Render profiles with a conventional legend below the axes."""
  set_plot_style(NOTEBOOK_STYLE)
  fig, ax = plt.subplots(figsize=figsize)

  _draw_profiles(ax, perf_df, ranked=False)

  ax.set_xlabel('Performance ratio τ  (relative to best submission)')
  ax.set_ylabel('Fraction of workloads solved ρ(τ)')
  ax.set_ylim(-0.02, 1.05)

  if title:
    ax.set_title(title, pad=8)

  n = len(perf_df.index)
  ncol = max(3, math.ceil(n / LEGEND_ROWS))
  ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.22),
    ncol=ncol,
    borderaxespad=0,
    frameon=True,
    handlelength=2.0,
    handleheight=0.9,
    columnspacing=1.0,
    labelspacing=0.35,
  )

  # Reserve space below the axes for the legend and axis label.
  n_rows = math.ceil(n / ncol)
  pts_per_row = mpl.rcParams['legend.fontsize'] * 1.55
  fig_height_pts = figsize[1] * POINTS_PER_INCH
  legend_frac = (n_rows * pts_per_row + AXIS_LABEL_SPACE_PT) / fig_height_pts
  fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.91 if title else 0.97,
    bottom=min(0.14 + legend_frac, 0.58),
  )

  return fig, ax


def plot_score_comparison(comparison):
  """Render wall-clock versus step scores using the shared submission styles."""
  styles = submission_styles(comparison.index)
  # The shared color cycle repeats every 10 entries, which lands "AdEMAMix
  # (AdamW-equiv.) (PyTorch)" on the same blue as "Schedule-Free AdamW
  # (PyTorch)"; both also render as plain circles, so they're
  # indistinguishable here. Give it a distinct color from the same palette
  # family, local to this plot.
  if 'AdEMAMix (AdamW-equiv.) (PyTorch)' in styles:
    styles['AdEMAMix (AdamW-equiv.) (PyTorch)'] = {
      **styles['AdEMAMix (AdamW-equiv.) (PyTorch)'],
      'color': '#882255',
    }
  set_plot_style(NOTEBOOK_STYLE)
  families = [
    (
      'Schedule-Free AdamW',
      'o',
      [
        'Schedule-Free AdamW v2 (PyTorch)',
        'Schedule-Free AdamW v2 (JAX)',
        'Schedule-Free AdamW (PyTorch)',
        'Schedule-Free AdamW (JAX)',
      ],
    ),
    (
      'NAdamW',
      's',
      [
        'NAdamW (Baseline AlgoPerf v0.5) (JAX)',
        'NAdamW (JAX)',
        'NAdamW (Tuned for ResNet) (JAX)',
        'Cautious NAdamW (JAX)',
      ],
    ),
    ('Muon', 'D', ['Muon (PyTorch)', 'Muon (JAX)']),
    (
      'Single Worker DiLoCo',
      '^',
      [
        'Single Worker DiLoCo (JAX)',
        'Single Worker DiLoCo v2 (JAX)',
      ],
    ),
    (
      'AdEMAMix',
      'P',
      ['AdEMAMix (PyTorch)', 'AdEMAMix (AdamW-equiv.) (PyTorch)'],
    ),
    ('Lion', 'X', ['Lion (PyTorch)']),
  ]
  _FAMILY_MARKER = {
    name: marker for _, marker, members in families for name in members
  }

  fig, ax = plt.subplots(figsize=(13.65, 4.23))

  lims = (0.10, 0.60)
  ax.plot(lims, lims, linestyle='--', color='#999999', linewidth=1.0, zorder=1)
  # White background so these labels stay legible if a marker lands nearby.
  _corner_label_bbox = dict(
    boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75
  )
  ax.text(
    0.15,
    0.485,
    'wall-clock advantage\n(lower cost per step)',
    ha='left',
    va='top',
    fontsize=8.5,
    style='italic',
    color='#777777',
    zorder=4,
    bbox=_corner_label_bbox,
  )
  ax.text(
    0.55,
    0.205,
    'step advantage\n(fewer steps to target)',
    ha='right',
    va='bottom',
    fontsize=8.5,
    style='italic',
    color='#777777',
    zorder=4,
    bbox=_corner_label_bbox,
  )

  scatter_handles = {}
  for name, row in comparison.iterrows():
    scatter_handles[name] = ax.scatter(
      row.steps,
      row.wallclock,
      color=styles[name]['color'],
      marker=_FAMILY_MARKER.get(name, 'o'),
      s=58,
      linewidths=0.6,
      edgecolors='#333333',
      zorder=3,
      label=name,
    )

  _annotate = {
    'Schedule-Free AdamW v2 (PyTorch)': (-8, 2, 'right'),
    'AdEMAMix (PyTorch)': (8, 3, 'left'),
    'Muon (JAX)': (8, -3, 'left'),
    'Muon (PyTorch)': (8, -3, 'left'),
    'NAdamW (JAX)': (8, 1, 'left'),
    'Single Worker DiLoCo (JAX)': (-2, 9, 'left'),
  }
  for name, (dx, dy, ha) in _annotate.items():
    row = comparison.loc[name]
    ax.annotate(
      name,
      (row.steps, row.wallclock),
      xytext=(dx, dy),
      textcoords='offset points',
      fontsize=8,
      ha=ha,
      color='#333333',
    )

  ax.set_xlim(lims)
  ax.set_ylim(lims)
  ax.set_aspect('equal')
  ax.set_xlabel('Step-based benchmark score')
  ax.set_ylabel('Wall-clock benchmark score')
  ax.set_title('Wall-clock vs. step-based benchmark scores', pad=8)

  legend_order = [
    name
    for _, _, members in families
    for name in members
    if name in scatter_handles
  ]
  legend_order.extend(
    name for name in comparison.index if name not in legend_order
  )
  ax.legend(
    [scatter_handles[name] for name in legend_order],
    legend_order,
    loc='center left',
    bbox_to_anchor=(1.03, 0.5),
    ncol=1,
    borderaxespad=0,
    frameon=True,
    handlelength=0.9,
    labelspacing=0.45,
    fontsize=7.7,
    markerscale=0.9,
  )
  fig.subplots_adjust(left=0.085, right=0.535, top=0.92, bottom=0.12)

  return fig


def plot_leaderboard_profile(scores, profiles, metric='score'):
  scores, profiles = validate_profiles(scores, profiles)
  styles = submission_styles(scores.index)
  set_plot_style(
    {
      **SERIF_STYLE,
      'font.size': 11.5,
      'savefig.dpi': 350,
      'mathtext.fontset': 'dejavuserif',
      'axes.linewidth': 0.8,
    }
  )
  fig = plt.figure(figsize=(10.6, max(4.3, 0.25 * (len(scores) + 3))))
  # Compact name/key/score columns; align the table rules with the
  # profile's top and bottom edges instead of centering unequal panels.
  table = fig.add_axes([0.006, 0.19, 0.38, 0.79])
  ax = fig.add_axes([0.48, 0.19, 0.495, 0.79])
  n = len(scores)
  table.set(xlim=(0, 1), ylim=(n + 0.65, -0.65))
  table.axis('off')
  columns = [
    (0.01, 'Submission', 'left'),
    (0.775, 'Line', 'center'),
    (0.99, 'Score', 'right'),
  ]
  for x, label, align in columns:
    table.text(x, 0.22, label, ha=align, va='baseline', weight='bold')
  table.hlines(
    [-0.65, 0.5, n + 0.65], 0, 1, colors='black', linewidths=[0.9, 0.6, 0.9]
  )

  for rank, (name, score) in enumerate(scores.items(), start=1):
    style = styles[name]
    if 'Baseline AlgoPerf' in name:
      table.axhspan(rank - 0.48, rank + 0.48, color='#EEEEEE', zorder=0)
    # Match the names already used in the paper's leaderboard.
    label = name.replace('Baseline AlgoPerf v0.5', 'Baseline v0.5')
    label = label.replace('(Tuned for ResNet)', 'tuned for ResNet')
    label = label.replace('Single Worker', 'Single-worker')
    # Shared baselines avoid optical misalignment between names with
    # descenders, numeric scores, and the vertically centered keys.
    table.text(
      0.01, rank + 0.22, label, ha='left', va='baseline', fontsize=10.8
    )
    table.text(
      0.99,
      rank + 0.22,
      f'{score:.4f}',
      ha='right',
      va='baseline',
      fontsize=10.8,
      weight='bold' if rank == 1 else 'normal',
    )
    table.plot(
      [0.735, 0.775, 0.815],
      [rank] * 3,
      **style,
      markevery=[1],
      linewidth=1.8,
      markersize=5.3,
    )
  _draw_profiles(ax, profiles, ranked=True)

  ax.set_ylim(-0.02, 1.02)
  ax.xaxis.set_major_locator(MultipleLocator(0.5))
  ax.set_yticks(np.arange(0, 1.01, 0.25))
  unit = 'steps ' if metric == 'global_step' else ''
  ax.set_xlabel(f'Performance ratio τ\n({unit}relative to best submission)')
  ax.set_ylabel('Fraction of workloads solved ρ(τ)', labelpad=6)
  ax.grid(axis='y', color='#DDDDDD', linewidth=0.65)
  ax.set_axisbelow(True)
  assert len(ax.lines) == len(scores) == len(table.lines)
  return fig


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--metric',
    choices=['score', 'global_step', 'score_relaxed'],
    default='score',
  )
  parser.add_argument(
    '--byproducts', type=Path, default=HERE / 'out/byproducts'
  )
  parser.add_argument('--output-dir', type=Path, default=HERE / 'out/results')
  args = parser.parse_args()
  scores, profiles = load_data(args.metric, args.byproducts)
  fig = plot_leaderboard_profile(scores, profiles, args.metric)
  suffix = {'score': '', 'global_step': '_steps', 'score_relaxed': '_relaxed'}[
    args.metric
  ]
  output = args.output_dir / f'leaderboard_profile{suffix}'
  save_figure(fig, output)
  plt.close(fig)
  print(
    f'Validated {len(scores)} unchanged scores and profiles; saved {output}.pdf/.png'
  )


if __name__ == '__main__':
  main()
