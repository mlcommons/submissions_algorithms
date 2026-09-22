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
# # AlgoPerf: JAX vs. PyTorch Framework Comparison
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
# repo-relative paths in Section 3 work from any launch directory.
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

from scoring.config import DEFAULT_TARGETS_PATH, WorkloadConfig


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

# Where to write output CSVs, plots, and LaTeX tables. This is the committed
# artifact directory for the second scoring iteration.
OUTPUT_DIR = str(section_output_dir('section_7_frameworks'))

# ── Submission filters (leave empty strings to include/exclude nothing) ───────
# Comma-separated names to include (empty = include all).
INCLUDE_SUBMISSIONS = 'schedule_free_adamw,schedule_free_adamw_jax,schedule_free_adamw_v2,schedule_free_adamw_jax_v2'
# Comma-separated names to exclude.
EXCLUDE_SUBMISSIONS = ','.join(EXCLUDED_SUBMISSIONS)

WORKLOAD_CONFIG = WorkloadConfig.from_json(DEFAULT_TARGETS_PATH)

# ── Caching (optional) ────────────────────────────────────────────────────────
# Save the parsed results dict so you can reload it later without re-parsing.
SAVE_RESULTS_TO = None  # e.g. 'results.pkl'
# Load a previously saved results dict instead of re-parsing.
LOAD_RESULTS_FROM = None  # e.g. 'results.pkl'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# %%
from artifacts.tech_report_v1.report_data import load_runs, write_summaries
from artifacts.tech_report_v1.report_utils import (
  pretty,
  save_figure,
)
from artifacts.tech_report_v1.section_7_frameworks.plot_framework_comparison import (
  step_times,
  plot_framework_comparison,
)

results = load_runs(
  SUBMISSION_DIRECTORY,
  include=INCLUDE_SUBMISSIONS,
  exclude=EXCLUDE_SUBMISSIONS,
  cache=Path(OUTPUT_DIR) / LOAD_RESULTS_FROM if LOAD_RESULTS_FROM else None,
  save_cache=Path(OUTPUT_DIR) / SAVE_RESULTS_TO if SAVE_RESULTS_TO else None,
)
for summary in write_summaries(results, WORKLOAD_CONFIG, OUTPUT_DIR).values():
  display(summary)

# %% [markdown]
# ## 7. Framework Comparison: JAX vs. PyTorch
#
# Compare Schedule-Free AdamW v1 and v2 across JAX and PyTorch, retaining pairs
# with shared core optimizer hyperparameters. Muon is excluded because its
# submissions differ in hyperparameters and orthogonalization sharding. The
# comparison is restricted to the six workloads with matching global training
# batch sizes: Criteo, ResNet, ViT, DeepSpeech, OGBG, and WMT. Exclude fastMRI,
# FineWeb-Edu, and Conformer rather than assume step time scales with batch size.
# Remaining recipe and implementation differences mean this is an observational
# comparison of submitted implementations, not a framework-only effect. Emits
# `framework_comparison.{csv,pdf,png}`:
# a heatmap of the raw median step-time ratio (JAX / PyTorch); the colorbar
# ticks repeat the same ratio values shown in the cells.
#

# %%
# ── JAX vs PyTorch: paired-algorithm framework comparison ────────────────────

WORKLOADS = [
  'criteo1tb',
  'imagenet_resnet',
  'imagenet_vit',
  'librispeech_deepspeech',
  'ogbg',
  'wmt',
]
WL_SHORT = ['Criteo', 'ResNet', 'ViT', 'DeepSpeech', 'OGBG', 'WMT']

# (row label, pytorch submission, jax submission)
# Restrict to shared core optimizer HPs. Muon also changes HPs and sharding.
# Exclude fastMRI, FineWeb-Edu, and Conformer: their batch sizes differ.
# Schedule-Free still has v1 averaging differences
# and an apparent JAX label-smoothing fallback; do not infer framework-only causality.
PAIRS = [
  (
    'Schedule-Free AdamW v1',
    pretty('schedule_free_adamw'),
    pretty('schedule_free_adamw_jax'),
  ),
  (
    'Schedule-Free AdamW v2',
    pretty('schedule_free_adamw_v2'),
    pretty('schedule_free_adamw_jax_v2'),
  ),
]

# Training batch sizes from each submission's get_batch_size().
BATCH_PT_SFA = {
  'criteo1tb': 262144,
  'fastmri': 16,
  'finewebedu_lm': 64,
  'imagenet_resnet': 1024,
  'imagenet_vit': 1024,
  'librispeech_conformer': 224,
  'librispeech_deepspeech': 128,
  'ogbg': 512,
  'wmt': 128,
}
BATCH_JAX_SFA = {
  **BATCH_PT_SFA,
  'fastmri': 32,
  'finewebedu_lm': 32,
  'librispeech_conformer': 256,
}
BATCHES = {
  'Schedule-Free AdamW v1': (BATCH_PT_SFA, BATCH_JAX_SFA),
  'Schedule-Free AdamW v2': (BATCH_PT_SFA, BATCH_JAX_SFA),
}


step_ratio = np.full((len(PAIRS), len(WORKLOADS)), np.nan)

for i, (label, pt, jx) in enumerate(PAIRS):
  st_pt, st_jx = step_times(results[pt]), step_times(results[jx])
  b_pt, b_jx = BATCHES[label]
  for j, w in enumerate(WORKLOADS):
    assert b_pt[w] == b_jx[w], f'Batch-size mismatch: {label}, {w}'
    step_ratio[i, j] = st_jx[w] / st_pt[w]

print('Raw median step-time ratio (JAX/PT), matching global batch sizes:')
print({short: BATCH_PT_SFA[w] for short, w in zip(WL_SHORT, WORKLOADS)})
ratio_df = pd.DataFrame(
  step_ratio, index=[p[0] for p in PAIRS], columns=WL_SHORT
)
print(ratio_df.round(2))
ratio_df.to_csv(
  os.path.join(OUTPUT_DIR, 'framework_comparison.csv'), index_label='Submission'
)

# ── Figure: single heatmap, print-true at \textwidth = 6.5in ──────────────────
# Cells show the raw JAX ÷ PyTorch ratio; the colorbar ticks repeat those
# ratio values (0.25 … 4) so the legend numbers match the box numbers.
fig = plot_framework_comparison(ratio_df)
save_figure(
  fig,
  Path(OUTPUT_DIR) / 'framework_comparison',
)
display(fig)
plt.close(fig)
