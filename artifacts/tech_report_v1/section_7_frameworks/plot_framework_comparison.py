"""Framework step-time summaries and heatmap rendering."""

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from artifacts.tech_report_v1.report_utils import NOTEBOOK_STYLE, set_plot_style


LOG_RATIO_LIMIT = 2.5
WHITE_TEXT_THRESHOLD = 1.6
COLORBAR_TICKS = np.arange(-2, 3)


def step_times(df):
  """Median seconds per step, per base workload."""
  out = {}
  for workload, group in df.groupby('workload'):
    base = re.sub(r'_(jax|pytorch)$', '', workload)
    ratios = []
    for _, trial in group.iterrows():
      t = np.diff(np.asarray(trial['accumulated_submission_time']), prepend=0)
      s = np.diff(np.asarray(trial['global_step']), prepend=0)
      with np.errstate(divide='ignore', invalid='ignore'):
        ratios.append(np.nanmedian(t / s))
    out[base] = float(np.median(ratios))
  return out


def plot_framework_comparison(ratios):
  """Render a labeled JAX/PyTorch ratio matrix."""
  values = ratios.to_numpy()
  set_plot_style(NOTEBOOK_STYLE)
  JAX_C, PT_C, MID_C = '#3D6FC4', '#CC3311', '#f7f7f7'
  cmap = LinearSegmentedColormap.from_list('fw', [JAX_C, MID_C, PT_C])
  norm = TwoSlopeNorm(vmin=-LOG_RATIO_LIMIT, vcenter=0.0, vmax=LOG_RATIO_LIMIT)

  fig, ax = plt.subplots(figsize=(6.5, 1.65))
  ax.grid(False)

  log = np.log2(values)
  masked = np.ma.masked_invalid(log)
  ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')
  for i in range(values.shape[0]):
    for j in range(values.shape[1]):
      v = values[i, j]
      ax.text(
        j,
        i,
        f'{v:.2f}' if v < 10 else f'{v:.0f}',
        ha='center',
        va='center',
        fontsize=8,
        color='white' if abs(log[i, j]) > WHITE_TEXT_THRESHOLD else '#333333',
      )
  ax.set_yticks(range(values.shape[0]))
  ax.set_yticklabels(ratios.index, fontsize=8.5)
  ax.set_xticks(range(len(ratios.columns)))
  ax.set_xticklabels(
    ratios.columns,
    fontsize=8.5,
    rotation=28,
    ha='right',
    rotation_mode='anchor',
  )
  ax.set_title(
    'Step time, JAX ÷ PyTorch (matched batch sizes)',
    fontsize=9,
    loc='left',
    pad=4,
  )
  ax.tick_params(length=0)
  for spine in ax.spines.values():
    spine.set_visible(False)
  for j in range(values.shape[1] + 1):
    ax.axvline(j - 0.5, color='white', lw=1.6)
  for i in range(values.shape[0] + 1):
    ax.axhline(i - 0.5, color='white', lw=1.6)

  cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    fraction=0.05,
    pad=0.03,
    ticks=COLORBAR_TICKS,
  )
  cbar.ax.set_yticklabels(
    [f'{2.0**tick:g}' for tick in COLORBAR_TICKS], fontsize=8
  )
  cbar.set_label('JAX ÷ PyTorch ratio', fontsize=8)
  cbar.ax.text(
    0.5,
    1.04,
    'PyTorch\nfaster',
    transform=cbar.ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=7,
    color=PT_C,
  )
  cbar.ax.text(
    0.5,
    -0.04,
    'JAX\nfaster',
    transform=cbar.ax.transAxes,
    ha='center',
    va='top',
    fontsize=7,
    color=JAX_C,
  )
  cbar.outline.set_visible(False)

  return fig
