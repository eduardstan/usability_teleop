"""Unified publication-style plotting theme."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PRIMARY_BLUE = "#2563eb"
PRIMARY_RED = "#dc2626"
NEUTRAL_DARK = "#1f2937"
NEUTRAL_GRID = "#d9e2ec"
NEUTRAL_MID = "#9ca3af"
BG_AXES = "#f8fafc"
BG_FIG = "#ffffff"


def apply_publication_theme() -> None:
    """Apply consistent style across all generated figures."""
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": BG_AXES,
            "figure.facecolor": BG_FIG,
            "grid.color": NEUTRAL_GRID,
            "grid.alpha": 0.7,
        }
    )
