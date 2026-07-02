"""Shared configuration for the self-tuning curve-plotting pipeline.

The pipeline runs in two passes so that parsing and plotting are decoupled:

  1. ``parse_logs.py``  -- walk the raw run logs once and emit a tidy
     ``curve_data.csv`` (one row per validation measurement).
  2. ``plot_curves.py`` -- read ``curve_data.csv`` and render the per-workload
     comparison plots.  Re-plotting (e.g. a different ``--zoom``) never touches
     the raw logs again.

Both passes import their registry, styling, path defaults, and helpers from
here so there is a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Derived from this file's location (``<repo>/logs/curve_plotting/plot_config.py``)
# so the scripts work from any checkout without a hardcoded ``~`` path.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "self_tuning"
DEFAULT_PLOT_DIR = REPO_ROOT / "logs" / "curve_plotting"
DEFAULT_DATA_PATH = DEFAULT_PLOT_DIR / "curve_data.csv"

# --- Metric direction ------------------------------------------------------
# Substrings that mark a validation metric where "higher is better".
_HIGHER_IS_BETTER_KEYS = (
    "accuracy",
    "auc",
    "map",
    "bleu",
    "ssim",
    "precision",
    "score",
)


def is_higher_better(metric_name: str) -> bool:
    """Return True if larger values of ``metric_name`` are better."""
    metric = metric_name.lower()
    return any(key in metric for key in _HIGHER_IS_BETTER_KEYS)


# --- Matplotlib styling ----------------------------------------------------
PLOT_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# --- Algorithm registry ----------------------------------------------------
# algo -> {"submissions": {submission_name: style}, "sub_dir": output folder}.
# ``style`` keys: color, linestyle, label, alpha.
ALGO_CONFIGS = {
    "sfadamw": {
        "submissions": {
            "schedule_free_adamw": {
                "color": "#1F77B4",  # Classic Blue
                "linestyle": "-",
                "label": "PyTorch v1",
                "alpha": 0.9,
            },
            "schedule_free_adamw_v2": {
                "color": "#0B3C5D",  # Deep Navy
                "linestyle": "--",
                "label": "PyTorch v2",
                "alpha": 0.9,
            },
            "schedule_free_adamw_jax": {
                "color": "#FF7F0E",  # Safety Orange
                "linestyle": "-",
                "label": "JAX v1",
                "alpha": 0.9,
            },
            "schedule_free_adamw_jax_v2": {
                "color": "#D9531E",  # Vibrant Rust
                "linestyle": "--",
                "label": "JAX v2",
                "alpha": 0.9,
            },
        },
        "sub_dir": "sfadamw",
    },
    "muon": {
        "submissions": {
            "muon_torch": {
                "color": "#1F77B4",
                "linestyle": "-",
                "label": "PyTorch v1",
                "alpha": 0.9,
            },
            "muon_torch_jax_hps": {
                "color": "#0B3C5D",
                "linestyle": "--",
                "label": "PyTorch v2 (JAX HPS)",
                "alpha": 0.9,
            },
            "muon": {
                "color": "#FF7F0E",
                "linestyle": "-",
                "label": "JAX v1",
                "alpha": 0.9,
            },
        },
        "sub_dir": "muon",
    },
    "ademamix": {
        "submissions": {
            "ademamix": {
                "color": "#1F77B4",
                "linestyle": "-",
                "label": "PyTorch",
                "alpha": 0.9,
            },
        },
        "sub_dir": "ademamix",
    },
    "cautious_nadamw": {
        "submissions": {
            "cautious_nadamw": {
                "color": "#FF7F0E",
                "linestyle": "-",
                "label": "JAX",
                "alpha": 0.9,
            },
        },
        "sub_dir": "cautious_nadamw",
    },
    "lion": {
        "submissions": {
            "lion": {
                "color": "#1F77B4",
                "linestyle": "-",
                "label": "PyTorch",
                "alpha": 0.9,
            },
        },
        "sub_dir": "lion",
    },
    "nadamw": {
        "submissions": {
            "nadamw": {
                "color": "#FF7F0E",
                "linestyle": "-",
                "label": "JAX v1",
                "alpha": 0.9,
            },
            "nadamw_baselinev05": {
                "color": "#D9531E",
                "linestyle": "--",
                "label": "JAX Baseline v0.5",
                "alpha": 0.9,
            },
        },
        "sub_dir": "nadamw",
    },
}


def registered_submissions() -> list[str]:
    """All submission names referenced by the registry, de-duplicated."""
    seen = {}
    for cfg in ALGO_CONFIGS.values():
        for submission in cfg["submissions"]:
            seen[submission] = None
    return list(seen)
