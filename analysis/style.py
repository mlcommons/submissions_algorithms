"""Shared plotting style for the analysis figures.

Follows the dataviz principles: sequential ramp for magnitude (step-time
heatmap), fixed-order CVD-safe categorical hues for algorithm families
(Okabe-Ito), diverging for rank-shift polarity. Text stays in ink colors, marks
carry identity.
"""

from __future__ import annotations

import matplotlib as mpl

# Okabe-Ito colorblind-safe categorical palette, assigned in fixed order by
# family (never cycled). Blue/orange/green/vermillion/sky/purple/black.
FAMILY_COLORS = {
    "nadamw": "#0072B2",
    "schedule_free_adamw": "#E69F00",
    "muon": "#009E73",
    "ademamix": "#D55E00",
    "cautious_nadamw": "#56B4E9",
    "lion": "#CC79A7",
    "diloco": "#000000",
}

# Sequential ramp for magnitude (step time); perceptually uniform, CVD-safe.
SEQUENTIAL_CMAP = "magma_r"
# Diverging ramp for polarity (rank shift): cool = improves, warm = worsens.
DIVERGING_CMAP = "RdBu_r"

INK = "#222222"
MUTED = "#888888"
GRID = "#e8e8e8"


def apply_rc() -> None:
    """Apply shared rcParams for clean, publication-style static figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
