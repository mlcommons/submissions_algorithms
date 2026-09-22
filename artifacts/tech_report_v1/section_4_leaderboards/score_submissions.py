# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# # AlgoPerf Scoring v2: Tables & Performance Profiles
#
# Second scoring iteration for the tech report. This notebook runs **locally against
# the repo's self-contained `scoring/` package**, so scores always match
# `python -m scoring.score_submissions` and divide by the config's full base
# workload count (9 for the current config).
#
# Outputs are written to `artifacts/tech_report_v1/section_4_leaderboards/out/`.
#
# Run from anywhere inside the repo:
#
# ```bash
# uv sync --extra dev --extra analysis  # use this environment in Jupyter or VS Code
# uv run jupytext --sync --execute \
#   artifacts/tech_report_v1/section_4_leaderboards/score_submissions.ipynb
# ```
#
# **Input format** — your submission data must follow this directory structure:
# ```
# submission_directory/
#   <submission_name>/
#     <study_name>/
#       <workload_name>/
#         <trial_name>/
#           eval_measurements.csv
# ```
#

# %% [markdown]
# ## 1. Imports & Repo Root
#
# Locates the repo root (so the notebook works whether launched from the root or
# from this folder), then imports the scoring package.
#

# %%
# Notebook cells intentionally import dependencies near their use.
# ruff: noqa: E402
import os
import sys
from pathlib import Path

# Run everything relative to the repo root so `scoring` imports and the
# repo-relative paths in Section 2 work from any launch directory.
REPO_ROOT = next(
  p
  for p in [Path.cwd(), *Path.cwd().parents]
  if (p / 'scoring' / 'score_submissions.py').exists()
)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from scoring.config import (
  DEFAULT_TARGETS_PATH,
  SELF_TUNING_RUNTIME_FACTOR,
  WorkloadConfig,
)


# %% [markdown]
# ## 2. Configuration
#
# Set the paths and flags below before running the rest of the notebook.

# %%
# ── Required ──────────────────────────────────────────────────────────────────
# Path to the directory that contains one sub-folder per submission
# (relative to the repo root).
from artifacts.tech_report_v1.report_utils import (
  EXCLUDED_SUBMISSIONS,
  section_output_dir,
  submission_directory,
)

SUBMISSION_DIRECTORY = str(submission_directory())

# Where to write output CSVs, plots, and LaTeX tables. Lives next to this
# notebook rather than in the old top-level artifacts/leaderboard_v2/ tree.
OUTPUT_DIR = str(section_output_dir('section_4_leaderboards'))
# RESULTS_DIR: the notebook's explicit deliverables — LaTeX tables and plots
# that go straight into the tech report.
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
# BYPRODUCTS_DIR: intermediate/raw artifacts produced while computing the
# scores (score & performance-profile CSVs, time-to-target CSVs) — useful for
# debugging or reloading, but not themselves report-ready.
BYPRODUCTS_DIR = os.path.join(OUTPUT_DIR, 'byproducts')
# Per-submission summary CSVs (Section 4) are byproducts too; keep them in
# their own subfolder so they don't clutter BYPRODUCTS_DIR.
SUMMARIES_DIR = os.path.join(BYPRODUCTS_DIR, 'summaries')

# ── Submission filters (leave empty strings to include/exclude nothing) ───────
# Comma-separated names to include (empty = include all).
INCLUDE_SUBMISSIONS = ''
# Comma-separated names to exclude.
EXCLUDE_SUBMISSIONS = ','.join(EXCLUDED_SUBMISSIONS)

# ── Scoring flags ─────────────────────────────────────────────────────────────
# Set True to enforce the competition's strict trial/study count rules.
STRICT = False
# Set True when scoring the self-tuning ruleset.
SELF_TUNING_RULESET = True
# Benchmark version config: base/held-out workloads, targets, step hints.
# The score divides by the number of base workloads in this config.
WORKLOAD_CONFIG = WorkloadConfig.from_json(DEFAULT_TARGETS_PATH)

# ── Performance profile parameters ────────────────────────────────────────────
from artifacts.tech_report_v1.report_data import (
  DEFAULT_MIN_TAU,
  DEFAULT_MAX_TAU,
  DEFAULT_PROFILE_POINTS,
)

MIN_TAU = DEFAULT_MIN_TAU
MAX_TAU = DEFAULT_MAX_TAU  # set None to auto-detect from data
NUM_POINTS = DEFAULT_PROFILE_POINTS
SCALE = 'linear'  # 'linear' or 'log'

# ── Caching (optional) ────────────────────────────────────────────────────────
# Save the parsed results dict so you can reload it later without re-parsing.
SAVE_RESULTS_TO = None  # e.g. 'results.pkl'
# Load a previously saved results dict instead of re-parsing.
LOAD_RESULTS_FROM = None  # e.g. 'results.pkl'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(
  SUMMARIES_DIR, exist_ok=True
)  # os.makedirs also creates BYPRODUCTS_DIR


# %%
from artifacts.tech_report_v1.report_utils import (
  latex_name,
  save_figure,
  SELECTED_SWEEP_SUBMISSIONS,
)
from artifacts.tech_report_v1.report_tables import (
  scores_to_latex,
  workload_table,
)
from artifacts.tech_report_v1.report_data import (
  load_runs,
  write_summaries,
  score_runs,
)
from artifacts.tech_report_v1.section_4_leaderboards.plot_leaderboard_profile import (
  plot_performance_profiles,
  plot_score_comparison,
  plot_leaderboard_profile,
)


# %% [markdown]
# ## 4. Load & Summarize Submissions

# %%
results = load_runs(
  SUBMISSION_DIRECTORY,
  include=INCLUDE_SUBMISSIONS,
  exclude=EXCLUDE_SUBMISSIONS,
  cache=Path(OUTPUT_DIR) / LOAD_RESULTS_FROM if LOAD_RESULTS_FROM else None,
  save_cache=Path(OUTPUT_DIR) / SAVE_RESULTS_TO if SAVE_RESULTS_TO else None,
)
write_summaries(results, WORKLOAD_CONFIG, SUMMARIES_DIR)
SCORING_OPTIONS = dict(
  min_tau=MIN_TAU,
  max_tau=MAX_TAU,
  num_points=NUM_POINTS,
  scale=SCALE,
  strict=STRICT,
  self_tuning_ruleset=SELF_TUNING_RULESET,
)

# %% [markdown]
# ## 5. Performance Profiles, Leaderboard Scores & LaTeX Table

# %%
if not STRICT:
  print(
    'WARNING: STRICT=False relaxes criteria on held-out workloads, '
    'trial counts, and study counts. Scores may not match official '
    'competition scoring. Set STRICT=True to enforce all rules.'
  )

performance_profile_df, scores, ttt = score_runs(
  results,
  WORKLOAD_CONFIG,
  BYPRODUCTS_DIR,
  time_col='score',
  artifact_suffix='',
  **SCORING_OPTIONS,
)

# Save profile CSV for reuse without re-running the full pipeline.
profile_csv = os.path.join(BYPRODUCTS_DIR, 'performance_profile_score.csv')
performance_profile_df.to_csv(profile_csv)

# ── Styled plot ────────────────────────────────────────────────────────────
fig, ax = plot_performance_profiles(
  performance_profile_df,
  title='AlgoPerf: Self-Tuning Ruleset Performance Profiles',
)
save_figure(
  fig,
  Path(RESULTS_DIR) / 'performance_profile_by_score',
)
plt.show()

# ── Scores table ──────────────────────────────────────────────────────────
scores_path = os.path.join(BYPRODUCTS_DIR, 'scores.csv')
scores.to_csv(scores_path)

print('\n--- Leaderboard Scores ---')
display(scores.sort_values('score', ascending=False))
print(f'Saved to {scores_path}')


# ── LaTeX table ────────────────────────────────────────────────────────────
latex_table = scores_to_latex(scores)
print(latex_table)

latex_path = os.path.join(RESULTS_DIR, 'scores_table.tex')
with open(latex_path, 'w') as f:
  f.write(latex_table)
print(f'\nSaved to {latex_path}')

# %% [markdown]
# ## 5b. Time-to-Target Table
# Shows the median time each submission took to reach the validation target on each workload, as a fraction of that workload's maximum allowed runtime.\n`inf` means the target was not reached within the run.

# %%


# ── Per-workload max time budget ───────────────────────────────────────────────
# `max_allowed_runtime_sec` comes straight from WORKLOAD_CONFIG (i.e.
# scoring/workload_targets.json), which vendors each base workload's
# `algoperf/workloads/<workload>/workload.py` value. WorkloadConfig applies
# the self-tuning ruleset's SELF_TUNING_RUNTIME_FACTOR (1.5x) internally.
# Format: finite values as a fraction (2 decimals) of the workload's time
# budget, inf as em-dash.
def _fmt(workload, v):
  if pd.isna(v) or v == float('inf'):
    return r'\textemdash{}'
  budget = WORKLOAD_CONFIG.max_runtime_sec(
    workload, self_tuning_ruleset=SELF_TUNING_RULESET
  )
  return f'{v / budget:.2f}'


ttt_display = ttt.apply(lambda col: col.map(lambda v: _fmt(col.name, v)))
ttt_display.index.name = 'Submission'
# Order rows by wall-clock leaderboard score (best first), matching Section 5.
ttt_display = ttt_display.loc[
  scores.sort_values('score', ascending=False).index
]

print('--- Time to Target (fraction of max runtime budget) ---')
display(ttt_display)


_budget_caption = (
  r"Time to target as a fraction of each workload's self-tuning-ruleset "
  rf'time budget (${SELF_TUNING_RUNTIME_FACTOR:g}\times$ the external-tuning maximum allowed runtime).'
  if SELF_TUNING_RULESET
  else r"Time to target as a fraction of each workload's maximum allowed runtime."
)

latex = workload_table(
  ttt_display,
  caption=_budget_caption + r' \textemdash{} = target not reached.',
  label='tab:time_to_target',
)

ttt_latex_path = os.path.join(RESULTS_DIR, 'time_to_target_table.tex')
with open(ttt_latex_path, 'w') as f:
  f.write(latex)
print(f'\nLaTeX saved → {ttt_latex_path}')


# %% [markdown]
# ## 6. Step-Based Scoring: Step vs. Wall-Clock Efficiency
#
# Re-scores every submission with the identical performance-profile machinery,
# but using `global_step` (optimizer steps to target) as the time column instead
# of wall-clock seconds. Same workload config, same $\tau$ range, same
# denominator. Artifacts are written with a `_steps` suffix, plus:
#
# - `steps_to_target_table.tex` — per-workload median steps to target
# - `scores_steps_table.tex` — step-based leaderboard ranking
# - `scores_comparison_time_vs_steps.tex` — wall-clock vs. step-based scores
#   and ranks, plus the per-submission rank shift ($\Delta$) between the two
# - `wallclock_vs_steps.{pdf,png}` — scatter of the two benchmark scores
#
# Caveat: submissions choose their own batch sizes, so step counts compare
# optimizer updates, not examples seen; steps are not sample-normalized.
#

# %%
# ── Step-based performance profiles and leaderboard scores ────────────────────
performance_profile_steps_df, scores_steps, stt = score_runs(
  results,
  WORKLOAD_CONFIG,
  BYPRODUCTS_DIR,
  time_col='global_step',
  artifact_suffix='_steps',
  **SCORING_OPTIONS,
)
performance_profile_steps_df.to_csv(
  os.path.join(BYPRODUCTS_DIR, 'performance_profile_global_step.csv')
)

fig, ax = plot_performance_profiles(
  performance_profile_steps_df,
  title='AlgoPerf: Self-Tuning Performance Profiles (steps to target)',
)
save_figure(
  fig,
  Path(RESULTS_DIR) / 'performance_profile_by_global_step',
)
plt.show()

scores_steps.to_csv(os.path.join(BYPRODUCTS_DIR, 'scores_steps.csv'))
print('--- Step-based Leaderboard Scores ---')
display(scores_steps.sort_values('score', ascending=False))

# ── LaTeX table ────────────────────────────────────────────────────────────
scores_steps_latex = scores_to_latex(
  scores_steps,
  caption='AlgoPerf Self-Tuning Leaderboard (steps to target)',
  label='tab:scores_steps',
)
print(scores_steps_latex)

scores_steps_latex_path = os.path.join(RESULTS_DIR, 'scores_steps_table.tex')
with open(scores_steps_latex_path, 'w') as f:
  f.write(scores_steps_latex)
print(f'\nSaved to {scores_steps_latex_path}')

# %%
# ── Wall-clock vs. step-based leaderboard comparison ──────────────────────────
# Both use the same performance-profile scoring; only the notion of training
# time differs (seconds vs. optimizer steps to target). rank_shift captures
# how much a submission's rank changes between the two.
cmp = pd.DataFrame(
  {
    'wallclock': scores['score'],
    'steps': scores_steps['score'],
  }
)
cmp['rank_wallclock'] = cmp.wallclock.rank(
  ascending=False, method='min'
).astype(int)
cmp['rank_steps'] = cmp.steps.rank(ascending=False, method='min').astype(int)
cmp['rank_shift'] = cmp.rank_wallclock - cmp.rank_steps
cmp = cmp.sort_values('wallclock', ascending=False)

cmp_csv_path = os.path.join(BYPRODUCTS_DIR, 'scores_wallclock_vs_steps.csv')
cmp.to_csv(cmp_csv_path)
print('--- Wall-clock vs. Step-based Leaderboard Comparison ---')
display(cmp.round(4))
print(f'Saved to {cmp_csv_path}')


def _fmt_shift(d):
  if d > 0:
    return rf'$\uparrow${d}'
  if d < 0:
    return rf'$\downarrow${-d}'
  return '--'


_best_wall = cmp.wallclock.max()
_best_steps = cmp.steps.max()
_cmp_rows = []
for name, row in cmp.iterrows():
  wall = f'{row.wallclock:.4f}'
  steps = f'{row.steps:.4f}'
  if row.wallclock == _best_wall:
    wall = r'\textbf{' + wall + '}'
  if row.steps == _best_steps:
    steps = r'\textbf{' + steps + '}'
  _cmp_rows.append(
    f'    {latex_name(name)} & {wall} & {int(row.rank_wallclock)} & {steps} & '
    f'{int(row.rank_steps)} & {_fmt_shift(int(row.rank_shift))} ' + r'\\'
  )

cmp_latex = '\n'.join(
  [
    r'\begin{table}[htbp]',
    r'  \centering',
    r'  \caption{Wall-clock vs.\ step-based benchmark scores. Both use the same'
    r' performance-profile scoring; only the notion of training time differs'
    r' (seconds vs.\ optimizer steps to target). $\Delta$ is the rank change'
    r' when moving from wall-clock to step-based scoring.}',
    r'  \label{tab:leaderboard_steps}',
    r'  \begin{tabular}{lrrrrc}',
    r'    \toprule',
    r'    & \multicolumn{2}{c}{Wall-clock} & \multicolumn{2}{c}{Steps} & \\',
    r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'    Submission & Score & Rank & Score & Rank & $\Delta$ \\',
    r'    \midrule',
    *_cmp_rows,
    r'    \bottomrule',
    r'  \end{tabular}',
    r'\end{table}',
  ]
)

leaderboard_steps_latex_path = os.path.join(
  RESULTS_DIR, 'scores_comparison_time_vs_steps.tex'
)
with open(leaderboard_steps_latex_path, 'w') as f:
  f.write(cmp_latex)
print(f'\nSaved to {leaderboard_steps_latex_path}')


# %%
# ── Steps-to-target table (analog of the time-to-target table) ────────────────


def _fmt_steps(v):
  if pd.isna(v) or v == float('inf'):
    return r'\textemdash{}'
  return f'{v:,.0f}'


stt_display = stt.map(_fmt_steps)
stt_display.index.name = 'Submission'
# Order rows by step-based leaderboard score (best first), matching this
# section's own leaderboard above.
stt_display = stt_display.loc[
  scores_steps.sort_values('score', ascending=False).index
]
display(stt_display)

stt_latex = workload_table(
  stt_display,
  caption=r'Median number of optimizer steps to reach the validation target.'
  r' \textemdash{} = target not reached. Submissions choose their own batch'
  r' sizes, so step counts compare optimizer updates, not examples seen.',
  label='tab:steps_to_target',
)

with open(os.path.join(RESULTS_DIR, 'steps_to_target_table.tex'), 'w') as f:
  f.write(stt_latex)
print('LaTeX saved -> steps_to_target_table.tex')


# %%
fig = plot_score_comparison(cmp)
save_figure(
  fig,
  Path(RESULTS_DIR) / 'wallclock_vs_steps',
)
display(fig)
plt.close(fig)


# %% [markdown]
# ## 7. Relaxed Convergence Targets: Score Table & Performance Profile
#
# Repeats Section 5's performance profile, leaderboard score, and LaTeX table
# (mirroring Section 6's step-based version too), but against a workload
# configuration whose validation targets are relaxed by
# `TARGET_RELAXATION_FRACTION` using `WorkloadConfig.with_target_relaxations`
# (see `scoring/config.py`) — the same mechanism as
# `scoring/score_submissions.py --target_relaxations=all=0.10`. Loss-style
# (minimize) targets increase by this fraction; accuracy-style (maximize)
# targets decrease by it. Artifacts use a `_relaxed` suffix
# (`time_to_targets_relaxed.csv`, `scores_relaxed.csv`,
# `scores_standard_vs_relaxed.csv`, `scores_relaxed_table.tex`,
# `performance_profile_by_score_relaxed.{pdf,png}`)
# and this never modifies the frozen `workload_targets*.json` files.
# Also emits `time_to_target_table_relaxed.tex`, the relaxed-target analog of
# Section 5b's time-to-target table, plus
# `newly_reached_targets_relaxed.csv` to record which cells should be bolded.

# %%
TARGET_RELAXATION_FRACTION = 0.10  # 10% relaxation, applied to every workload
_RELAXATION_LABEL = f'{TARGET_RELAXATION_FRACTION:.0%}'
_RELAXATION_LABEL_LATEX = _RELAXATION_LABEL.replace('%', r'\%')

RELAXED_WORKLOAD_CONFIG = WORKLOAD_CONFIG.with_target_relaxations(
  {'all': TARGET_RELAXATION_FRACTION}
)

performance_profile_relaxed_df, scores_relaxed, ttt_relaxed = score_runs(
  results,
  RELAXED_WORKLOAD_CONFIG,
  BYPRODUCTS_DIR,
  time_col='score',
  artifact_suffix='_relaxed',
  **SCORING_OPTIONS,
)
performance_profile_relaxed_df.to_csv(
  os.path.join(BYPRODUCTS_DIR, 'performance_profile_score_relaxed.csv')
)

# ── Styled plot ────────────────────────────────────────────────────────────
fig, ax = plot_performance_profiles(
  performance_profile_relaxed_df,
  title=f'AlgoPerf: Self-Tuning Ruleset Performance Profiles '
  f'(targets relaxed {TARGET_RELAXATION_FRACTION:.0%})',
)
save_figure(
  fig,
  Path(RESULTS_DIR) / 'performance_profile_by_score_relaxed',
)
plt.show()

# ── Relaxed scores table ───────────────────────────────────────────────────
scores_relaxed_path = os.path.join(BYPRODUCTS_DIR, 'scores_relaxed.csv')
scores_relaxed.to_csv(scores_relaxed_path)

print(
  f'\n--- Leaderboard Scores (targets relaxed {TARGET_RELAXATION_FRACTION:.0%}) ---'
)
display(scores_relaxed.sort_values('score', ascending=False))
print(f'Saved to {scores_relaxed_path}')

# ── Standard-vs-relaxed comparison table ──────────────────────────────────
# Keep the score and rank movement in one auditable DataFrame, then use it for
# both the CSV byproduct and the report table. Positive rank_shift means that a
# submission moves up when targets are relaxed.
relaxed_cmp = pd.DataFrame(
  {
    'standard_score': scores['score'],
    'relaxed_score': scores_relaxed['score'],
  }
)
relaxed_cmp['standard_rank'] = relaxed_cmp.standard_score.rank(
  ascending=False, method='min'
).astype(int)
relaxed_cmp['relaxed_rank'] = relaxed_cmp.relaxed_score.rank(
  ascending=False, method='min'
).astype(int)
relaxed_cmp['rank_shift'] = relaxed_cmp.standard_rank - relaxed_cmp.relaxed_rank
relaxed_cmp = relaxed_cmp.sort_values('relaxed_rank')

relaxed_cmp_path = os.path.join(
  BYPRODUCTS_DIR, 'scores_standard_vs_relaxed.csv'
)
relaxed_cmp.to_csv(relaxed_cmp_path)
print(f'Saved to {relaxed_cmp_path}')

_best_standard = relaxed_cmp.standard_score.max()
_best_relaxed = relaxed_cmp.relaxed_score.max()
_relaxed_cmp_rows = []
for name, row in relaxed_cmp.iterrows():
  standard_score = f'{row.standard_score:.4f}'
  relaxed_score = f'{row.relaxed_score:.4f}'
  if row.standard_score == _best_standard:
    standard_score = r'\textbf{' + standard_score + '}'
  if row.relaxed_score == _best_relaxed:
    relaxed_score = r'\textbf{' + relaxed_score + '}'
  _relaxed_cmp_rows.append(
    f'    {latex_name(name)} & {standard_score} & '
    f'{int(row.standard_rank)} & {relaxed_score} & '
    f'{int(row.relaxed_rank)} & {_fmt_shift(int(row.rank_shift))} ' + r'\\'
  )

scores_relaxed_latex = '\n'.join(
  [
    r'\begin{table}[htbp]',
    r'  \centering',
    r'  \caption{Standard vs.\ '
    + _RELAXATION_LABEL_LATEX
    + r' relaxed-target AlgoPerf self-tuning leaderboard. $\Delta$ is the rank'
    + r' change after relaxing targets.}',
    r'  \label{tab:scores_relaxed}',
    r'  \begin{tabular}{lrrrrc}',
    r'    \toprule',
    r'    & \multicolumn{2}{c}{Standard} & \multicolumn{2}{c}{10\% relaxed} & \\',
    r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'    Submission & Score & Rank & Score & Rank & $\Delta$ \\',
    r'    \midrule',
    *_relaxed_cmp_rows,
    r'    \bottomrule',
    r'  \end{tabular}',
    r'\end{table}',
  ]
)
print(scores_relaxed_latex)

scores_relaxed_latex_path = os.path.join(
  RESULTS_DIR, 'scores_relaxed_table.tex'
)
with open(scores_relaxed_latex_path, 'w') as f:
  f.write(scores_relaxed_latex)
print(f'\nSaved to {scores_relaxed_latex_path}')

# %%
# ── Time-to-Target Table (relaxed targets) ─────────────────────────────────
# Relaxed-target analog of Section 5b's time-to-target table: same _fmt()
# (fraction of the workload's time budget), against
# time_to_targets_relaxed.csv instead of the official time_to_targets.csv.

ttt_relaxed_display = ttt_relaxed.apply(
  lambda col: col.map(lambda v: _fmt(col.name, v))
)
ttt_relaxed_display.index.name = 'Submission'
# Order rows by relaxed leaderboard score (best first), matching this
# section's own leaderboard above.
ttt_relaxed_display = ttt_relaxed_display.loc[
  scores_relaxed.sort_values('score', ascending=False).index
]

print(
  f'--- Time to Target, targets relaxed {TARGET_RELAXATION_FRACTION:.0%} '
  f'(fraction of max runtime budget) ---'
)
display(ttt_relaxed_display)


# A target is newly reached when its standard time-to-target is non-finite but
# its relaxed time-to-target is finite. Persist this mask so the bolding in the
# LaTeX table is mechanically checkable without parsing LaTeX.
def _target_reached(v):
  return pd.notna(v) and np.isfinite(v)


_newly_reached_relaxed = ~ttt.map(_target_reached) & ttt_relaxed.map(
  _target_reached
)
_newly_reached_path = os.path.join(
  BYPRODUCTS_DIR, 'newly_reached_targets_relaxed.csv'
)
_newly_reached_relaxed.to_csv(_newly_reached_path)
print(
  f'Newly reached targets: {_newly_reached_relaxed.to_numpy().sum()} '
  f'(saved to {_newly_reached_path})'
)

_relaxed_budget_caption = (
  rf'Time to target with targets relaxed {_RELAXATION_LABEL_LATEX}, as a '
  r"fraction of each workload's self-tuning-ruleset time budget "
  rf'(${SELF_TUNING_RUNTIME_FACTOR:g}\times$ the external-tuning maximum allowed runtime).'
  if SELF_TUNING_RULESET
  else rf'Time to target with targets relaxed {_RELAXATION_LABEL_LATEX}, as a '
  r"fraction of each workload's maximum allowed runtime."
)

ttt_relaxed_latex = workload_table(
  ttt_relaxed_display,
  caption=_relaxed_budget_caption
  + r' \textbf{Bold} values mark targets reached only after relaxation;'
  + r' \textemdash{} = target not reached.',
  label='tab:time_to_target_relaxed',
  bold=_newly_reached_relaxed,
)

ttt_relaxed_latex_path = os.path.join(
  RESULTS_DIR, 'time_to_target_table_relaxed.tex'
)
with open(ttt_relaxed_latex_path, 'w') as f:
  f.write(ttt_relaxed_latex)
print(f'\nLaTeX saved → {ttt_relaxed_latex_path}')

# %% [markdown]
# ## 8. Paper leaderboard figures
#
# Render the ranked legends from the profiles computed above. The helpers
# validate the score integrals and return figures for notebook display.
# Figures are saved locally; this notebook does not update the paper checkout.

# %%

paper_figure_paths = []
for metric, score_frame, profile_frame, suffix in [
  ('score', scores, performance_profile_df, ''),
  ('global_step', scores_steps, performance_profile_steps_df, '_steps'),
  ('score_relaxed', scores_relaxed, performance_profile_relaxed_df, '_relaxed'),
]:
  fig = plot_leaderboard_profile(score_frame['score'], profile_frame, metric)
  paper_figure_paths.extend(
    save_figure(fig, Path(RESULTS_DIR) / f'leaderboard_profile{suffix}')
  )
  display(fig)
  plt.close(fig)

# %% [markdown]
# ## 9. Target-relaxation sweep
#
# Reuse the loaded runs, full submission pool, workload configuration, and
# scoring settings. Check the standard and relaxed endpoints against this
# notebook's results. Filtering the selected figure affects only its display;
# scores and ranks always use the full pool.

# %%
from artifacts.tech_report_v1.section_4_leaderboards.sweep_target_relaxation import (
  sweep,
  SWEEP_PERCENTAGES,
  plot_sweep,
  plot_selected_scores,
)

SELECTED_SUBMISSIONS = SELECTED_SWEEP_SUBMISSIONS
sweep_dir = Path(BYPRODUCTS_DIR) / 'relaxation_sweep'
sweep_frame = sweep(
  results,
  config=WORKLOAD_CONFIG,
  output_dir=sweep_dir,
  percentages=SWEEP_PERCENTAGES,
  **SCORING_OPTIONS,
  references={
    0: (scores['score'], ttt),
    round(100 * TARGET_RELAXATION_FRACTION, 8): (
      scores_relaxed['score'],
      ttt_relaxed,
    ),
  },
)
fig = plot_sweep(
  sweep_frame, scores['score'].sort_values(ascending=False).index
)
paper_figure_paths.extend(
  save_figure(fig, Path(RESULTS_DIR) / 'target_relaxation_sweep')
)
display(fig)
plt.close(fig)

fig, selected_curves = plot_selected_scores(sweep_frame, SELECTED_SUBMISSIONS)
selected_curves.to_csv(sweep_dir / 'selected_curves.csv', index=False)
paper_figure_paths.extend(
  save_figure(fig, Path(RESULTS_DIR) / 'target_relaxation_selected')
)
display(fig)
plt.close(fig)
