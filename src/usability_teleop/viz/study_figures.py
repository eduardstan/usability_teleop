"""Figure builders for ablation study outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from usability_teleop.viz.theme import apply_publication_theme


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_study_stage_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    apply_publication_theme()
    if summary_df.empty:
        return
    data = summary_df.copy()
    melted = data.melt(
        id_vars=["stage"],
        value_vars=["regression_mean_best_r2", "classification_mean_best_auc"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "regression_mean_best_r2": "Regression (mean best R2)",
        "classification_mean_best_auc": "Classification (mean best AUC)",
    }
    melted["metric"] = melted["metric"].map(metric_labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="value", y="stage", hue="metric", ax=ax)
    ax.set_title("Ablation Study: Stage-Level Performance")
    ax.set_xlabel("Metric Value")
    ax.set_ylabel("Stage")
    _save(fig, output_path)


def plot_study_delta_heatmap(breakdown_df: pd.DataFrame, output_path: Path) -> None:
    apply_publication_theme()
    if breakdown_df.empty:
        return
    data = breakdown_df[breakdown_df["stage"] != "baseline"].copy()
    if data.empty:
        return
    data["label"] = data["track"] + " | " + data["target"]
    data["stage_metric"] = data["stage"] + " | " + data["metric"]
    matrix = data.pivot(index="label", columns="stage_metric", values="delta_vs_baseline")
    fig_h = max(4, 0.35 * len(matrix) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(matrix, cmap="RdBu_r", center=0.0, annot=True, fmt=".3f", ax=ax)
    ax.set_title("Ablation Study: Delta vs Baseline")
    ax.set_xlabel("Stage | Metric")
    ax.set_ylabel("Track | Target")
    _save(fig, output_path)


def plot_study_target_distributions(target_dist_df: pd.DataFrame, output_path: Path) -> None:
    apply_publication_theme()
    if target_dist_df.empty:
        return
    reg = target_dist_df[target_dist_df["track"] == "regression"].copy()
    cls = target_dist_df[target_dist_df["track"] == "classification"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * max(len(reg), len(cls)) + 2)))
    if not reg.empty:
        sns.barplot(data=reg, x="std", y="target", color="#2563eb", ax=axes[0])
    axes[0].set_title("Regression Target Dispersion (std)")
    axes[0].set_xlabel("std")
    axes[0].set_ylabel("target")
    if not cls.empty:
        sns.barplot(data=cls, x="minority_ratio", y="target", color="#dc2626", ax=axes[1])
        axes[1].axvline(0.35, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_title("Classification Class Balance")
    axes[1].set_xlabel("minority ratio")
    axes[1].set_ylabel("target")
    _save(fig, output_path)
