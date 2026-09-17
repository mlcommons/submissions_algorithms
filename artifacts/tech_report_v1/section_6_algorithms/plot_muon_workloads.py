"""Plot Muon's workload profile with a matched vanilla PyTorch comparison."""

import argparse
from pathlib import Path
import sys

if __name__ == '__main__' and not __package__:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from artifacts.tech_report_v1.report_utils import (
  DISPLAY_TO_RAW,
  REPORT_ROOT,
  set_plot_style,
  save_figure,
  submission_directory,
  pretty,
)

HERE = REPORT_ROOT / 'section_6_algorithms'
SOURCE = REPORT_ROOT / 'section_4_leaderboards/out/byproducts'
OUTPUT = HERE / 'out'
SHARDED = 'Muon (PyTorch)'
VANILLA_RAW = 'muon_torch_replicated_torch_hps'
VANILLA = pretty(VANILLA_RAW)
JAX = 'Muon (JAX)'
MUONS = [SHARDED, VANILLA, JAX]
LEGEND_LABELS = ['Muon (PyTorch, sharded)', 'Muon (PyTorch, vanilla)', JAX]
COLORS = ['#176B96', '#176B96', '#B65A27']
MARKERS = ['o', 'o', 'D']
WORKLOADS = [
  ('criteo1tb', 'Criteo 1TB', 'DLRM'),
  ('ogbg', 'OGBG', 'Graph neural network'),
  ('finewebedu_lm', 'FineWebEdu', 'Decoder Transformer'),
  ('wmt', 'WMT', 'Transformer'),
  ('imagenet_vit', 'ImageNet ViT', 'Vision Transformer'),
  ('fastmri', 'fastMRI', 'U-Net'),
  ('imagenet_resnet', 'ImageNet ResNet', 'ResNet-50'),
  ('librispeech_conformer', 'LibriSpeech Conformer', 'Conformer'),
  ('librispeech_deepspeech', 'LibriSpeech DeepSpeech', 'DeepSpeech'),
]


def load_verified_data(source=SOURCE, config=None, submission_dir=None):
  """Validate leaderboard medians and load the matched vanilla run from logs.

  The vanilla comparison is excluded from the non-Muon reference pool. The
  input directory follows ALGOPERF_REPORT_INPUT unless explicitly supplied.
  """
  from scoring import performance_profile, scoring_utils
  from scoring.config import (
    DEFAULT_TARGETS_PATH,
    SELF_TUNING_RUNTIME_FACTOR,
    WorkloadConfig,
  )

  source = Path(source)
  times = pd.read_csv(source / 'time_to_targets.csv', index_col=0)
  config = (
    config
    if config is not None
    else WorkloadConfig.from_json(DEFAULT_TARGETS_PATH)
  )
  if not times.index.is_unique or not {SHARDED, JAX}.issubset(times.index):
    raise ValueError(
      'Expected unique submission names including both Muon submissions.'
    )
  assert set(times.columns) == set(config.base_workloads)
  assert set(times.columns) == {w[0] for w in WORKLOADS}
  budgets = pd.Series(
    {
      w: config.workloads[w].max_allowed_runtime_sec
      * SELF_TUNING_RUNTIME_FACTOR
      for w in times.columns
    }
  )
  leaderboard_names = set(times.index)
  study_counts = {}
  for name in times.index:
    raw_name = DISPLAY_TO_RAW.get(name, name)
    summary = pd.read_csv(source / 'summaries' / f'{raw_name}_summary.csv')
    summary['base'] = summary['workload'].str.replace(
      r'_(jax|pytorch)$', '', regex=True
    )
    grouped = summary.groupby('base')['time to target on val (s)']
    assert grouped.size().between(1, 3).all()
    if name in (SHARDED, JAX):
      assert grouped.size().eq(3).all()
    study_counts[name] = grouped.size()
    medians = grouped.apply(lambda values: np.median(values.to_numpy()))
    np.testing.assert_allclose(
      times.loc[name], medians.reindex(times.columns, fill_value=np.inf)
    )

  if VANILLA not in times.index:
    directory = (
      submission_directory() if submission_dir is None else Path(submission_dir)
    )
    vanilla_path = directory / VANILLA_RAW
    if not vanilla_path.is_dir():
      raise ValueError(f'Matched vanilla Muon logs are missing: {vanilla_path}')
    runs = scoring_utils.get_experiment_df(str(vanilla_path))
    # The self-tuning comparison has one trial per study. Keep failed studies
    # in the median and record incomplete study counts in the audit CSV.
    if runs.empty or runs.duplicated(['workload', 'study']).any():
      raise ValueError('Expected one vanilla Muon trial per workload/study.')
    counts = runs.groupby('workload')['study'].nunique()
    counts.index = counts.index.str.replace(r'_(jax|pytorch)$', '', regex=True)
    if (
      set(counts.index) != set(times.columns) or not counts.between(1, 3).all()
    ):
      raise ValueError('Expected 1–3 vanilla Muon studies on each workload.')
    study_counts[VANILLA] = counts
    times.loc[VANILLA] = (
      performance_profile.get_workloads_time_to_target(
        runs,
        VANILLA_RAW,
        config,
        time_col='score',
        verbosity=0,
        self_tuning_ruleset=True,
        strict=False,
      )
      .iloc[0]
      .reindex(times.columns)
    )

  normalized = times / budgets
  if (
    not ((normalized > 0) & ((normalized <= 1) | np.isposinf(normalized)))
    .all()
    .all()
  ):
    raise ValueError(
      'Expected positive budget fractions <= 1, or infinity for missed targets.'
    )
  others = times.loc[~times.index.str.startswith('Muon')]
  best = others.min()
  rows = []
  for workload, label, model in WORKLOADS:
    for name in times.index:
      seconds = times.loc[name, workload]
      solved = np.isfinite(seconds)
      has_reference = np.isfinite(best[workload])
      rows.append(
        {
          'workload': workload,
          'label': label,
          'model_family': model,
          'submission': name,
          'median_seconds': seconds,
          'self_tuning_budget_seconds': budgets[workload],
          'budget_fraction': normalized.loc[name, workload],
          'target_reached': solved,
          'study_count': int(study_counts[name].get(workload, 0)),
          'in_source_leaderboard': name in leaderboard_names,
          'best_non_muon_submission': (
            others[workload].idxmin() if has_reference else ''
          ),
          'best_non_muon_seconds': best[workload],
          'speedup_vs_best_non_muon': (
            best[workload] / seconds if solved and has_reference else np.nan
          ),
        }
      )
  return times, normalized, best, pd.DataFrame(rows)


def plot(times, normalized, best):
  set_plot_style(
    {
      'font.family': 'DejaVu Sans',
      'font.size': 10,
      'savefig.bbox': 'tight',
      'savefig.pad_inches': 0.14,
      'pdf.fonttype': 42,
      'axes.spines.top': False,
      'axes.spines.right': False,
      'axes.spines.left': False,
    }
  )
  fig = plt.figure(figsize=(12.4, 5.4), facecolor='white')
  handles = [
    Line2D(
      [],
      [],
      marker=m,
      color=c,
      linestyle='none',
      markersize=7,
      label=label,
      markerfacecolor='white' if name == VANILLA else c,
      markeredgewidth=1.5 if name == VANILLA else 1,
    )
    for name, label, c, m in zip(
      MUONS, LEGEND_LABELS, COLORS, MARKERS, strict=True
    )
  ] + [
    Line2D(
      [],
      [],
      marker='o',
      color='#ADB5BE',
      linestyle='none',
      markersize=5,
      label='Other submissions',
    )
  ]
  fig.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(0.018, 1.0),
    ncol=4,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.5,
    fontsize=9.5,
  )
  # Text columns and plot use identical y coordinates for accurate alignment.
  labels = fig.add_axes([0.025, 0.12, 0.190, 0.73])
  ax = fig.add_axes([0.222, 0.12, 0.522, 0.73])
  ratios = fig.add_axes([0.768, 0.12, 0.215, 0.73])
  for panel in (labels, ax, ratios):
    panel.set_ylim(len(times.columns) - 0.35, -0.65)
    panel.set_yticks([])
  for panel in (labels, ratios):
    panel.set_xlim(0, 1)
    panel.axis('off')
  labels.text(
    0,
    1.045,
    'Workload',
    transform=labels.transAxes,
    weight='bold',
    fontsize=10,
  )
  ax.text(
    0,
    1.045,
    'Time to target / workload budget',
    transform=ax.transAxes,
    weight='bold',
    fontsize=10,
  )
  ratios.text(
    0.5,
    1.105,
    'Speedup vs. best other',
    transform=ratios.transAxes,
    weight='bold',
    ha='center',
    fontsize=10,
  )
  xs = [1 / 6, 0.5, 5 / 6]
  for x, text, color in zip(
    xs, ['Sharded', 'Vanilla', 'JAX'], COLORS, strict=True
  ):
    ratios.text(
      x,
      1.045,
      text,
      color=color,
      transform=ratios.transAxes,
      ha='center',
      fontsize=10,
    )
  ax.set_xlim(-0.015, 1.025)
  ax.set_xticks(np.arange(0, 1.01, 0.2))
  ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
  ax.tick_params(axis='x', length=0, pad=7, colors='#56616A')
  ax.spines['bottom'].set_color('#B7C0C7')
  ax.grid(axis='x', color='#E0E5EA', linewidth=0.7, zorder=0)
  ax.axvline(1, color='#9BA5AF', linestyle=(0, (3, 3)), linewidth=1)
  ax.set_xlabel('← Faster', loc='left', labelpad=8, color='#56616A')
  other_names = times.index[~times.index.str.startswith('Muon')]
  for y, (workload, label, _) in enumerate(WORKLOADS):
    for panel in (labels, ax, ratios):
      if y % 2 == 0:
        panel.axhspan(y - 0.49, y + 0.49, color='#F4F6F8', zorder=-1)
    labels.text(0, y, label, va='center', color='#263846', fontsize=10)
    # All submissions share the row center; larger Muon markers are
    # drawn last so they remain visible among the other submissions.
    other = normalized.loc[other_names, workload]
    finite = other[np.isfinite(other)]
    ax.scatter(
      finite,
      np.full(len(finite), y),
      s=25,
      color='#ADB5BE',
      edgecolor='white',
      linewidth=0.5,
      zorder=3,
    )
    if not np.isfinite(normalized[workload]).any():
      ax.text(
        0.5,
        y,
        'No submission reached target',
        ha='center',
        va='center',
        color='#737D87',
        fontsize=9,
        style='italic',
      )
    for j, name in enumerate(MUONS):
      value = normalized.loc[name, workload]
      xcell = xs[j]
      if np.isfinite(value):
        is_vanilla = name == VANILLA
        # A larger hollow circle preserves visibility when paired medians
        # almost coincide, as on FineWebEdu.
        ax.scatter(
          [value],
          [y],
          s=112 if is_vanilla else 58,
          marker=MARKERS[j],
          facecolors='none' if is_vanilla else COLORS[j],
          edgecolors=COLORS[j] if is_vanilla else 'white',
          linewidth=1.5 if is_vanilla else 0.8,
          zorder=6 if is_vanilla else 5,
        )
      if np.isfinite(value) and np.isfinite(best[workload]):
        speedup = best[workload] / times.loc[name, workload]
        wins = speedup > 1
        ratios.add_patch(
          Rectangle(
            (xcell - 0.145, y - 0.33),
            0.29,
            0.66,
            facecolor='#DCEDE5' if wins else '#ECEFF2',
            edgecolor='none',
          )
        )
        ratios.text(
          xcell,
          y,
          f'{speedup:.2f}×',
          ha='center',
          va='center',
          color='#166446' if wins else '#455460',
          weight='bold' if wins else 'normal',
          fontsize=10.5,
        )
      else:
        ratios.text(
          xcell, y, '—', ha='center', va='center', color='#7C858E', fontsize=9
        )
  return fig


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, default=SOURCE)
  parser.add_argument('--output-dir', type=Path, default=OUTPUT)
  parser.add_argument(
    '--submission-directory',
    type=Path,
    default=None,
    help='Log pool containing the matched vanilla submission.',
  )
  args = parser.parse_args()
  times, normalized, best, audit = load_verified_data(
    args.source,
    submission_dir=args.submission_directory,
  )
  byproducts = args.output_dir / 'byproducts'
  byproducts.mkdir(parents=True, exist_ok=True)
  audit.to_csv(byproducts / 'muon_workloads_data.csv', index=False)
  fig = plot(times, normalized, best)
  paths = save_figure(fig, args.output_dir / 'results/muon_workloads')
  plt.close(fig)
  print(
    f'Verified {len(audit)} time-to-target entries using leaderboard summaries '
    'and matched vanilla logs.'
  )
  print(*paths, sep='\n')


if __name__ == '__main__':
  main()
