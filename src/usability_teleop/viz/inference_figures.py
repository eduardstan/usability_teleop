"""Figure builders for inference bundle outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from usability_teleop.viz.theme import apply_publication_theme


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_inference_regression_ci(inf_reg_df: pd.DataFrame, output_path: Path) -> None:
    """Point-range plot for regression R2 with bootstrap CI per target."""
    apply_publication_theme()
    if inf_reg_df.empty:
        return
    plot_df = inf_reg_df.sort_values("r2_observed", ascending=True).copy()
    sig = plot_df["paired_significant_fdr"].fillna(False).astype(bool)
    colors = sig.map({True: "#dc2626", False: "#2563eb"})
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df) + 1.5)))
    ax.hlines(plot_df["target"], plot_df["r2_ci_low"], plot_df["r2_ci_high"], color="#9ca3af", linewidth=2)
    ax.scatter(plot_df["r2_observed"], plot_df["target"], c=colors, s=52, zorder=3)
    ax.axvline(0.0, color="#1f2937", linestyle="--", linewidth=1)
    ax.set_title("Inference: Regression R2 with 95% Bootstrap CI")
    ax.set_xlabel("R2")
    ax.set_ylabel("Target")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", label="FDR-significant", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb", label="Not FDR-significant", markersize=8),
    ]
    ax.legend(handles=legend_items, loc="lower right")
    _save(fig, output_path)


def plot_inference_classification_ci(inf_cls_df: pd.DataFrame, output_path: Path) -> None:
    """Point-range plot for classification AUC with bootstrap CI per target."""
    apply_publication_theme()
    if inf_cls_df.empty:
        return
    plot_df = inf_cls_df.sort_values("auc_observed", ascending=True).copy()
    sig = plot_df["paired_significant_fdr"].fillna(False).astype(bool)
    colors = sig.map({True: "#dc2626", False: "#2563eb"})
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df) + 1.5)))
    ax.hlines(plot_df["target"], plot_df["auc_ci_low"], plot_df["auc_ci_high"], color="#9ca3af", linewidth=2)
    ax.scatter(plot_df["auc_observed"], plot_df["target"], c=colors, s=52, zorder=3)
    ax.axvline(0.5, color="#1f2937", linestyle="--", linewidth=1)
    ax.set_title("Inference: Classification AUC with 95% Bootstrap CI")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Target")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", label="FDR-significant", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb", label="Not FDR-significant", markersize=8),
    ]
    ax.legend(handles=legend_items, loc="lower right")
    _save(fig, output_path)


def plot_inference_pvalues(inf_reg_df: pd.DataFrame, inf_cls_df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of raw vs FDR-adjusted paired p-values."""
    apply_publication_theme()
    frames: list[pd.DataFrame] = []
    if not inf_reg_df.empty:
        t = inf_reg_df[["target", "paired_p_value", "paired_p_value_fdr"]].copy()
        t["track"] = "regression"
        frames.append(t)
    if not inf_cls_df.empty:
        t = inf_cls_df[["target", "paired_p_value", "paired_p_value_fdr"]].copy()
        t["track"] = "classification"
        frames.append(t)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data, x="paired_p_value", y="target", hue="track", style="track", s=85, ax=ax)
    ax.scatter(data["paired_p_value_fdr"], data["target"], marker="x", c="#111827", s=45)
    ax.axvline(0.05, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.02)
    ax.set_title("Inference: Paired Tests (Raw and FDR-adjusted p-values)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Target")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="x", color="#111827", linestyle="None", label="FDR-adjusted p-value", markersize=8))
    labels.append("FDR-adjusted p-value")
    ax.legend(handles=handles, labels=labels, loc="upper right")
    _save(fig, output_path)


def plot_inference_bayesian(inf_reg_df: pd.DataFrame, inf_cls_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart for Bayesian probability of improvement over baseline."""
    apply_publication_theme()
    frames: list[pd.DataFrame] = []
    if not inf_reg_df.empty:
        t = inf_reg_df[["target", "bayes_prob_improvement"]].copy()
        t["track"] = "regression"
        frames.append(t)
    if not inf_cls_df.empty:
        t = inf_cls_df[["target", "bayes_prob_improvement"]].copy()
        t["track"] = "classification"
        frames.append(t)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data["label"] = data["target"] + " | " + data["track"]
    data = data.sort_values("bayes_prob_improvement", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(data) + 1.5)))
    sns.barplot(data=data, x="bayes_prob_improvement", y="label", hue="track", dodge=False, palette="Set2", ax=ax)
    ax.axvline(0.5, color="#1f2937", linestyle="--", linewidth=1)
    ax.axvline(0.95, color="#dc2626", linestyle=":", linewidth=1)
    ax.set_xlim(0.0, 1.02)
    ax.set_title("Inference: Bayesian Probability of Improvement vs Baseline")
    ax.set_xlabel("Posterior P(improvement)")
    ax.set_ylabel("Target | Track")
    _save(fig, output_path)
