"""Rescore saved runs at 0--20% target relaxation, at 1% increments.

The pool and scoring protocol remain fixed across relaxation levels.
Coverage counts finite median times before the profile's 4x cutoff.
"""

import argparse
from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifacts.tech_report_v1.report_utils import (
  REPO_ROOT,
  REPORT_ROOT,
  SERIF_STYLE,
  DISPLAY_TO_RAW,
  SELECTED_SWEEP_SUBMISSIONS,
  set_plot_style,
  save_figure,
  submission_styles,
)

from artifacts.tech_report_v1.report_data import (
  DEFAULT_MIN_TAU,
  DEFAULT_MAX_TAU,
  DEFAULT_PROFILE_POINTS,
  load_runs as load_submission_runs,
  score_runs,
)

HERE = REPORT_ROOT / 'section_4_leaderboards'
SWEEP_PERCENTAGES = range(0, 21)
SWEEP_STYLE = {
  **SERIF_STYLE,
  'axes.labelsize': 12,
  'axes.titlesize': 12,
  'xtick.labelsize': 11,
  'ytick.labelsize': 11,
}


def load_runs(byproducts=None, submission_directory=None):
  """Load the cached leaderboard pool; notebooks can pass existing runs."""
  byproducts = (
    Path(byproducts) if byproducts is not None else HERE / 'out/byproducts'
  )
  submission_directory = (
    Path(submission_directory)
    if submission_directory is not None
    else REPO_ROOT / 'logs/self_tuning'
  )
  standard = pd.read_csv(byproducts / 'scores.csv', index_col=0)
  if not standard.index.is_unique:
    raise ValueError('Submission names must be unique.')
  runs = load_submission_runs(
    submission_directory,
    include=[DISPLAY_TO_RAW.get(name, name) for name in standard.index],
  )
  return runs, standard['score'].sort_values(ascending=False).index


def sweep(
  runs,
  *,
  config=None,
  output_dir=None,
  percentages=SWEEP_PERCENTAGES,
  min_tau=DEFAULT_MIN_TAU,
  max_tau=DEFAULT_MAX_TAU,
  num_points=DEFAULT_PROFILE_POINTS,
  scale='linear',
  strict=False,
  self_tuning_ruleset=True,
  references=None,
):
  """Rescore a fixed pool and optionally check supplied reference endpoints.

  references maps a relaxation percentage to (scores Series, target-times
  DataFrame), allowing the notebook to validate against its current results.
  """
  from scoring.config import DEFAULT_TARGETS_PATH, WorkloadConfig

  if config is None:
    config = WorkloadConfig.from_json(DEFAULT_TARGETS_PATH)
  output = (
    Path(output_dir)
    if output_dir is not None
    else HERE / 'out/byproducts/relaxation_sweep'
  )
  references = {} if references is None else references
  percentages = list(percentages)
  if not percentages or percentages != sorted(set(percentages)):
    raise ValueError(
      'Relaxation percentages must be nonempty, unique, and increasing.'
    )
  if any(p != int(p) or not 0 <= p < 100 for p in percentages):
    raise ValueError(
      'Relaxation percentages must be integers between 0 and 99.'
    )
  output.mkdir(parents=True, exist_ok=True)
  records, thresholds = [], []
  for percent in percentages:
    fraction = percent / 100
    relaxed = config.with_target_relaxations({'all': fraction})
    suffix = f'_{percent:02d}pct'
    profiles, scores_frame, times = score_runs(
      runs,
      relaxed,
      output,
      artifact_suffix=suffix,
      min_tau=min_tau,
      max_tau=max_tau,
      num_points=num_points,
      scale=scale,
      strict=strict,
      self_tuning_ruleset=self_tuning_ruleset,
    )
    scores = scores_frame['score']
    times = times.loc[:, list(config.base_workloads)]
    coverage = np.isfinite(times).sum(axis=1)
    ranks = scores.rank(ascending=False, method='min').astype(int)
    profiles.to_csv(output / f'performance_profiles{suffix}.csv')
    for name in scores.index:
      records.append(
        dict(
          relaxation_percent=percent,
          submission=name,
          score=scores[name],
          rank=ranks[name],
          workloads_reached=coverage[name],
          workload_count=config.num_base_workloads,
        )
      )
    for workload in config.base_workloads:
      metric, target = relaxed.metric_and_target(workload)
      thresholds.append(
        dict(
          relaxation_percent=percent,
          workload=workload,
          metric=metric,
          target=target,
        )
      )
    if percent in references:
      expected_scores, expected_times = references[percent]
      if set(scores.index) != set(expected_scores.index):
        raise ValueError(f'Reference pool differs at {percent}%.')
      np.testing.assert_allclose(
        scores.loc[expected_scores.index], expected_scores, rtol=0, atol=1e-12
      )
      np.testing.assert_allclose(
        times.loc[expected_times.index, expected_times.columns],
        expected_times,
        rtol=0,
        atol=1e-9,
      )
    best = scores.idxmax()
    print(
      f'{percent:2d}%: winner={best}; score={scores[best]:.4f}; '
      f'coverage={coverage.min()}--{coverage.max()}/{config.num_base_workloads}',
      flush=True,
    )

  frame = pd.DataFrame(records)
  counts = frame.pivot(
    index='relaxation_percent', columns='submission', values='workloads_reached'
  )
  assert np.all(np.diff(counts.to_numpy(), axis=0) >= 0)
  frame.to_csv(output / 'scores_and_coverage.csv', index=False)
  pd.DataFrame(thresholds).to_csv(output / 'target_thresholds.csv', index=False)
  return frame


def plot_sweep(frame, order):
  workload_count = int(frame.workload_count.iloc[0])
  styles = submission_styles(order)
  set_plot_style(SWEEP_STYLE)
  fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
  fig.subplots_adjust(
    left=0.065, right=0.985, top=0.91, bottom=0.34, wspace=0.20
  )
  for i, name in enumerate(order):
    rows = frame[frame.submission == name].sort_values('relaxation_percent')
    for ax, column in zip(axes, ['score', 'workloads_reached']):
      ax.plot(
        rows.relaxation_percent,
        rows[column],
        label=name,
        **styles[name],
        linewidth=1.8,
        markersize=4.8,
        markevery=(i % 3, 3),
        alpha=0.95,
      )
  axes[0].set(title='(a) Relative leaderboard score', ylabel='AlgoPerf score')
  axes[0].set_ylim(0, max(0.7, frame.score.max() + 0.04))
  axes[1].set(
    title='(b) Workloads reaching the target',
    ylabel=f'Workloads reached (out of {workload_count})',
    ylim=(0, workload_count + 0.3),
  )
  axes[1].set_yticks(range(workload_count + 1))
  for ax in axes:
    ax.set(
      xlabel='Target relaxation (%)',
      xlim=(
        frame.relaxation_percent.min() - 0.25,
        frame.relaxation_percent.max() + 0.25,
      ),
    )
    ax.set_xticks(
      range(
        int(frame.relaxation_percent.min()),
        int(frame.relaxation_percent.max()) + 1,
        2,
      )
    )
    ax.grid(axis='both', color='#DDDDDD', linewidth=0.6)
    ax.set_axisbelow(True)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.07),
    ncol=3,
    frameon=False,
    fontsize=9,
    handlelength=2.8,
    columnspacing=2.0,
    labelspacing=0.7,
  )
  fig.suptitle(
    'Sensitivity to target relaxation: the same saved training runs',
    fontsize=14,
    y=0.98,
  )
  fig.text(
    0.5,
    0.025,
    'Lines connect sampled relaxation levels. '
    f'Fixed {frame.submission.nunique()}-submission pool and {workload_count} workloads; '
    'fastest-run references recomputed at each level.',
    ha='center',
    fontsize=9,
    color='#555555',
  )
  return fig


def plot_selected_scores(frame, selected=None):
  # Contrasting examples, not an exhaustive list of sweep winners:
  # original winner; high-relaxation leader; strict-target runner-up;
  # baseline with a conspicuous 3% -> 4% jump; and DiLoCo v2 as a lower
  # reference. Filter only the display, keeping the full scoring pool.
  selected = SELECTED_SWEEP_SUBMISSIONS if selected is None else selected
  styles = submission_styles(frame.submission.unique())
  subset = frame[frame.submission.isin(selected)].copy()
  expected = pd.MultiIndex.from_product(
    [frame.relaxation_percent.unique(), selected]
  )
  actual = pd.MultiIndex.from_frame(
    subset[['relaxation_percent', 'submission']]
  )
  if not actual.is_unique or set(actual) != set(expected):
    raise ValueError(
      'Every selected submission must have one row per relaxation level.'
    )
  set_plot_style({**SWEEP_STYLE, 'axes.titlesize': 13})
  fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
  fig.subplots_adjust(
    left=0.065, right=0.985, bottom=0.31, top=0.97, wspace=0.25
  )
  for i, name in enumerate(selected):
    rows = subset[subset.submission == name].sort_values('relaxation_percent')
    for ax, metric in zip(axes, ('score', 'rank')):
      ax.plot(
        rows.relaxation_percent,
        rows[metric],
        **styles[name],
        linewidth=2.2,
        markersize=5.2,
        markevery=(i % 2, 2),
        label=name,
        zorder=len(selected) + 1 - i,
      )
  for ax in axes:
    ax.set(
      xlabel='Target relaxation (%)',
      xlim=(
        frame.relaxation_percent.min() - 0.25,
        frame.relaxation_percent.max() + 0.25,
      ),
    )
    ax.set_xticks(
      range(
        int(frame.relaxation_percent.min()),
        int(frame.relaxation_percent.max()) + 1,
        2,
      )
    )
    ax.grid(color='#DDDDDD', linewidth=0.6)
    ax.set_axisbelow(True)
  score_limit = max(0.75, subset.score.max() + 0.05)
  axes[0].set(ylabel='AlgoPerf score', ylim=(0, score_limit))
  axes[0].set_yticks(np.arange(0, score_limit, 0.1))
  # Cached ranks use every submission, not just the displayed subset.
  # Lower numeric ranks appear higher, so upward motion is improvement.
  pool_size = frame.submission.nunique()
  axes[1].set(ylabel='Leaderboard rank', ylim=(pool_size + 0.4, 0.6))
  axes[1].set_yticks(range(1, pool_size + 1))
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.015),
    ncol=2,
    frameon=False,
    fontsize=10,
    handlelength=3,
    columnspacing=2,
    labelspacing=0.65,
  )
  return fig, subset


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--byproducts', type=Path, default=HERE / 'out/byproducts'
  )
  parser.add_argument(
    '--submission-directory', type=Path, default=REPO_ROOT / 'logs/self_tuning'
  )
  parser.add_argument('--output-dir', type=Path, default=HERE / 'out')
  args = parser.parse_args()
  runs, order = load_runs(args.byproducts, args.submission_directory)
  references = {
    percent: (
      pd.read_csv(args.byproducts / f'scores{tag}.csv', index_col=0)['score'],
      pd.read_csv(args.byproducts / f'time_to_targets{tag}.csv', index_col=0),
    )
    for percent, tag in [(0, ''), (10, '_relaxed')]
  }
  sweep_dir = args.output_dir / 'byproducts/relaxation_sweep'
  frame = sweep(runs, output_dir=sweep_dir, references=references)
  fig = plot_sweep(frame, order)
  save_figure(fig, args.output_dir / 'results/target_relaxation_sweep')
  plt.close(fig)
  fig, selected = plot_selected_scores(frame)
  save_figure(fig, args.output_dir / 'results/target_relaxation_selected')
  plt.close(fig)
  selected.to_csv(sweep_dir / 'selected_curves.csv', index=False)


if __name__ == '__main__':
  main()
