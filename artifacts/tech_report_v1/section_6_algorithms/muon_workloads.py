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
# # Muon: empirical performance across workloads
#
# Run the paired Section 4 scoring notebook first to generate the standard
# leaderboard medians and study summaries. This notebook validates those
# medians, displays both Muon submissions against the full leaderboard pool,
# and writes the figure and its audit data. It does not rescore training logs
# or modify the paper checkout.
#
# Setup: `uv sync --extra dev --extra analysis`. Select that environment's
# Python kernel, then Restart and Run All. See `../README.md` for pairing,
# headless execution, and the separate paper-export step.

# %%
# Notebook cells intentionally import dependencies near their use.
# ruff: noqa: E402
import sys
from pathlib import Path

REPO_ROOT = next(
  p
  for p in [Path.cwd(), *Path.cwd().parents]
  if (p / 'scoring/score_submissions.py').is_file()
)
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from IPython.display import display

from artifacts.tech_report_v1.report_utils import (
  save_figure,
  section_output_dir,
)
from artifacts.tech_report_v1.section_6_algorithms.plot_muon_workloads import (
  load_verified_data,
  plot,
)

# %% [markdown]
# ## Inputs and outputs
#
# The source directory contains all submissions from the standard-target
# leaderboard, not just Muon; the fastest non-Muon reference is recomputed
# within that pool for each workload. Time budgets use the self-tuning factor.

# %%
SOURCE_DIR = section_output_dir('section_4_leaderboards') / 'byproducts'
OUTPUT_DIR = section_output_dir('section_6_algorithms')

# %% [markdown]
# ## Figure and audit data
#
# Each workload's markers share a centerline. A dash means the median target
# was not reached. Speedup is the fastest non-Muon median time divided by the
# Muon median time, so values above one favor Muon. The two Muon entries use
# different fixed hyperparameters as well as different frameworks.

# %%
times, budget_fractions, best_other, audit = load_verified_data(SOURCE_DIR)
byproducts = OUTPUT_DIR / 'byproducts'
byproducts.mkdir(parents=True, exist_ok=True)
audit.to_csv(byproducts / 'muon_workloads_data.csv', index=False)

fig = plot(times, budget_fractions, best_other)
paper_figure_paths = save_figure(fig, OUTPUT_DIR / 'results/muon_workloads')
display(fig)
plt.close(fig)

display(
  audit[audit['submission'].str.startswith('Muon')][
    [
      'label',
      'submission',
      'median_seconds',
      'budget_fraction',
      'speedup_vs_best_non_muon',
      'best_non_muon_submission',
    ]
  ]
)
