"""Unified publication-style plotting theme."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


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
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "#ffffff",
            "grid.color": "#d9e2ec",
            "grid.alpha": 0.7,
        }
    )
