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
#     display_name: myenv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Curve Plotter
#
# This notebook imports the helper functions from `plot_helper.py` to generate and display publication-grade comparison plots for self-tuning algorithms (e.g., `sfadamw`, `muon`, `ademamix`, etc.).
# You can customize the styling parameters below and select which algorithms/workloads to plot.

# %%
import sys
from pathlib import Path

# Ensure the scratch directory is in python path
scratch_dir = Path('./').resolve()
if str(scratch_dir) not in sys.path:
    sys.path.append(str(scratch_dir))

import plot_helper
from plot_helper import (
    ALGO_CONFIGS,
    configure_styling,
    plot_workload,
    find_workloads,
    find_repo_root,
)

# %% [markdown]
# ## Customize Styling
#
# You can override any Matplotlib RC parameters in the dictionary below. Call `configure_styling(custom_params)` to apply them.

# %%
# configure_styling()'s defaults already match score_submissions.py's theme
# (Section 4 leaderboards); add overrides here only if you want to deviate
# from that shared look for this notebook.
custom_styles = {}

# Apply styling configurations
configure_styling(custom_styles)

# %% [markdown]
# ## Select Algorithm & Generate Plots
#
# Generates plots for every algorithm in `ALGO_CONFIGS` (set `ALGO_NAMES` to a
# subset to limit it) and the given zoom type (`'log'` or `'percentile'`).
# Outputs go to `out/<algo>/<workload>_curves.png`, next to this notebook —
# one subdirectory per algorithm, mirroring each algorithm's `sub_dir`.

# %%
# Parameters - feel free to change these
ALGO_NAMES = list(ALGO_CONFIGS.keys())  # e.g. ['sfadamw', 'muon'] to limit to a subset
ZOOM_TYPE = 'log'                       # Options: 'log', 'percentile'

# Configure directories (repo root found by walking up from plot_helper.py,
# not hardcoded to any user's home directory). base_save_dir is anchored to
# plot_helper.py's own location (reliable even when this notebook's own
# __file__ isn't set) so outputs land in out/ next to this notebook.
REPO_ROOT = find_repo_root()
base_log_dir = REPO_ROOT / 'logs' / 'self_tuning'
base_save_dir = Path(plot_helper.__file__).resolve().parent / 'out'

for algo_name in ALGO_NAMES:
    if algo_name not in ALGO_CONFIGS:
        print(f"Algorithm '{algo_name}' not found in ALGO_CONFIGS. Available: {list(ALGO_CONFIGS.keys())}")
        continue

    config = ALGO_CONFIGS[algo_name]
    submissions = config['submissions']

    # Find all workloads for this algorithm
    workloads = find_workloads(base_log_dir, submissions)
    print(f"\n=== {algo_name}: found workloads {workloads} ===")

    for workload in workloads:
        print(f"Processing workload: {workload}")
        plot_workload(
            workload=workload,
            algo_name=algo_name,
            submissions=submissions,
            base_log_dir=base_log_dir,
            base_save_dir=base_save_dir,
            zoom=ZOOM_TYPE,
            show_plot=True,  # displays every figure inline, in addition to saving it
        )
