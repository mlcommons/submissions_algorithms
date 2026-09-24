"""Shared presentation metadata and artifact helpers for report notebooks.

Importing this module does not change the Matplotlib backend or global style.
"""

import os
from pathlib import Path

import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = Path(__file__).resolve().parent
COLORS = [
  '#4477AA',
  '#EE6677',
  '#228833',
  '#CCBB44',
  '#66CCEE',
  '#AA3377',
  '#BBBBBB',
  '#EE7733',
  '#009988',
  '#CC3311',
]
LINE_STYLES = ['-', '--', '-.', ':']

SERIF_STYLE = {
  'font.family': 'serif',
  'font.serif': ['DejaVu Serif'],
  'font.size': 11,
  'axes.spines.top': False,
  'axes.spines.right': False,
  'pdf.fonttype': 42,
}

NOTEBOOK_STYLE = {
  'figure.figsize': (9, 4.5),
  'figure.dpi': 150,
  'savefig.dpi': 300,
  'savefig.bbox': 'tight',
  'savefig.pad_inches': 0.05,
  'font.family': 'serif',
  'font.serif': ['Times New Roman', 'DejaVu Serif'],
  'font.size': 11,
  'axes.titlesize': 11,
  'axes.labelsize': 11,
  'xtick.labelsize': 10,
  'ytick.labelsize': 10,
  'legend.fontsize': 8.5,
  'legend.title_fontsize': 9,
  'legend.framealpha': 0.92,
  'legend.edgecolor': '#cccccc',
  'legend.borderpad': 0.5,
  'legend.labelspacing': 0.35,
  'axes.grid': True,
  'grid.alpha': 0.3,
  'grid.linestyle': '--',
  'grid.linewidth': 0.6,
  'axes.spines.top': False,
  'axes.spines.right': False,
  'axes.linewidth': 0.8,
  'lines.linewidth': 1.6,
}

SUBMISSION_NAME_MAP = {
  'ademamix': 'AdEMAMix (AdamW-equiv.) (PyTorch)',
  'ademamix_golden': 'AdEMAMix (PyTorch)',
  'cautious_nadamw': 'Cautious NAdamW (JAX)',
  'lion': 'Lion (PyTorch)',
  'muon': 'Muon (JAX)',
  'muon_torch': 'Muon (PyTorch)',
  'muon_torch_jax_hps': 'Muon (PyTorch, JAX HPs)',
  'muon_torch_jax_hps_achandr': 'Muon (PyTorch, JAX HPs, achandr)',
  'muon_torch_jax_hps_lr_fix': 'Muon (PyTorch, JAX HPs, LR Fix)',
  'muon_torch_replicated_jax_hps': 'Muon (Replicated, JAX HPs)',
  'muon_torch_replicated_torch_hps': 'Muon (Replicated, Torch HPs)',
  'nadamw': 'NAdamW (JAX)',
  'nadamw_baselinev05': 'NAdamW (Baseline AlgoPerf v0.5) (JAX)',
  'nadamw_resnet': 'NAdamW (Tuned for ResNet) (JAX)',
  'schedule_free_adamw': 'Schedule-Free AdamW (PyTorch)',
  'schedule_free_adamw_jax': 'Schedule-Free AdamW (JAX)',
  'schedule_free_adamw_jax_v2': 'Schedule-Free AdamW v2 (JAX)',
  'schedule_free_adamw_v2': 'Schedule-Free AdamW v2 (PyTorch)',
  'single_worker_diloco': 'Single Worker DiLoCo (JAX)',
  'single_worker_dilocov2': 'Single Worker DiLoCo (JAX)',
}

SUBMISSION_LATEX_MACRO = {
  'ademamix': r'\ademamixadamw',
  'ademamix_golden': r'\ademamix',
  'cautious_nadamw': r'\cautiousnadamw',
  'lion': r'\lion',
  'muon': r'\muonjax',
  'muon_torch': r'\muonpt',
  'nadamw': r'\nadamw',
  'nadamw_baselinev05': r'\nadamwbase',
  'nadamw_resnet': r'\nadamwresnet',
  'schedule_free_adamw': r'\sfadamw',
  'schedule_free_adamw_jax': r'\sfadamwjax',
  'schedule_free_adamw_jax_v2': r'\sfadamwjaxii',
  'schedule_free_adamw_v2': r'\sfadamwii',
  'single_worker_diloco': r'\dilocosw',
  'single_worker_dilocov2': r'\dilocosw',
}

DISPLAY_TO_RAW = {display: raw for raw, display in SUBMISSION_NAME_MAP.items()}


def marker_for(name):
  """Use one symbol per algorithm family, as in the step/time scatter."""
  if 'Schedule-Free' in name:
    return 'o'
  if 'Muon' in name:
    return 'D'
  if 'DiLoCo' in name:
    return 'v'
  if 'AdEMAMix' in name:
    return 's'
  if 'Lion' in name:
    return '*'
  if 'Cautious' in name:
    return 'P'
  return '^'


def submission_styles(names):
  """Shared line/color/symbol map for paired profiles and sensitivity plots."""
  return {
    name: dict(
      color=COLORS[i % len(COLORS)],
      linestyle=LINE_STYLES[(i // len(COLORS)) % len(LINE_STYLES)],
      marker=marker_for(name),
    )
    for i, name in enumerate(sorted(names))
  }


def set_plot_style(overrides):
  """Set global defaults for the next report figure and its PDF/PNG exports."""
  mpl.rcdefaults()
  mpl.rcParams.update({'savefig.dpi': 220, **overrides})


def save_figure(fig, output):
  """Save PDF and PNG using the renderer's style; leave the figure open."""
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  paths = [output.with_suffix(f'.{extension}') for extension in ('pdf', 'png')]
  for path in paths:
    fig.savefig(path, facecolor='white')
  return paths


def pretty(name):
  return SUBMISSION_NAME_MAP.get(name, name)


def latex_name(name):
  return SUBMISSION_LATEX_MACRO.get(DISPLAY_TO_RAW.get(name, name), name)


def submission_directory():
  return (
    REPO_ROOT / os.environ.get('ALGOPERF_REPORT_INPUT', 'logs/self_tuning')
  ).resolve()


def section_output_dir(section):
  root = REPO_ROOT / os.environ.get('ALGOPERF_REPORT_OUTPUT', str(REPORT_ROOT))
  return root / section / 'out'


EXCLUDED_SUBMISSIONS = (
  'muon_torch_jax_hps',
  'muon_torch_jax_hps_achandr',
  'muon_torch_jax_hps_lr_fix',
  'muon_torch_replicated_jax_hps',
  'muon_torch_replicated_torch_hps',
  'single_worker_diloco',

)
SELECTED_SWEEP_SUBMISSIONS = tuple(
  map(
    pretty,
    (
      'schedule_free_adamw_v2',
      'muon_torch',
      'ademamix',
      'nadamw_baselinev05',
      'single_worker_dilocov2',
    ),
  )
)
