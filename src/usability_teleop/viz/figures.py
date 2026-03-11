"""Publication-quality figure builders from result tables."""

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


def plot_correlation_heatmap(correlation_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """Plot Pearson heatmap for top highlighted feature-target pairs."""
    apply_publication_theme()
    highlights = correlation_df[correlation_df["pearson_highlight"]].copy()
    if highlights.empty:
        return

    top_features = (
        highlights.assign(abs_r=highlights["pearson_r"].abs())
        .sort_values("abs_r", ascending=False)
        .drop_duplicates("feature")
        .head(top_n)["feature"]
        .tolist()
    )

    matrix = (
        correlation_df[correlation_df["feature"].isin(top_features)]
        .pivot(index="feature", columns="target", values="pearson_r")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(matrix) * 0.2)))
    sns.heatmap(matrix, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Top Pearson Correlations (Highlighted)")
    _save(fig, output_path)


def plot_regression_overview(regression_global_df: pd.DataFrame, output_path: Path, top_k: int = 10) -> None:
    """Bar plot for top global regression configurations by mean R2."""
    apply_publication_theme()
    top = regression_global_df.sort_values("r2_mean", ascending=False).head(top_k).copy()
    top["label"] = top["model"] + " | " + top["feature_set"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=top,
        x="r2_mean",
        y="label",
        hue="label",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.axvline(0.0, color="#1f2937", linestyle="--", linewidth=1)
    ax.set_title("Global Regression Benchmark (Top by Mean R2)")
    ax.set_xlabel("Mean R2")
    ax.set_ylabel("Configuration")
    _save(fig, output_path)


def plot_classification_overview(classification_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot for best AUC per target from classification benchmark."""
    apply_publication_theme()
    valid = classification_df[classification_df["status"] == "ok"].copy()
    if valid.empty:
        return
    best = valid.sort_values(["target", "auc"], ascending=[True, False]).groupby("target", as_index=False).head(1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=best, x="auc", y="target", hue="target", palette="mako", legend=False, ax=ax)
    ax.axvline(0.5, color="#1f2937", linestyle=":", linewidth=1)
    ax.set_title("Best Classification AUC per Target")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Target")
    _save(fig, output_path)


def plot_permutation_summary(
    perm_reg_df: pd.DataFrame | None,
    perm_cls_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Plot p-value summary for permutation tests."""
    apply_publication_theme()
    frames: list[pd.DataFrame] = []

    if perm_reg_df is not None and not perm_reg_df.empty:
        tmp = perm_reg_df[["target", "p_value"]].copy()
        tmp["track"] = "regression"
        frames.append(tmp)

    if perm_cls_df is not None and not perm_cls_df.empty:
        tmp = perm_cls_df[["target", "p_value"]].copy()
        tmp["track"] = "classification"
        frames.append(tmp)

    if not frames:
        return

    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data, x="p_value", y="target", hue="track", style="track", s=80, ax=ax)
    ax.axvline(0.05, color="#dc2626", linestyle="--", linewidth=1, label="alpha=0.05")
    ax.set_xlim(0.0, 1.02)
    ax.set_title("Permutation Test P-values")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Target")
    _save(fig, output_path)


def plot_global_vs_target_specific_r2(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Dumbbell chart: best global vs best target-specific R2 per target."""
    apply_publication_theme()
    if comparison_df.empty:
        return
    plot_df = comparison_df.sort_values("delta_r2", ascending=False).copy()
    fig_height = max(4, 0.4 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.hlines(
        y=plot_df["target"],
        xmin=plot_df["r2_global"],
        xmax=plot_df["r2_specific"],
        color="#9ca3af",
        linewidth=2,
    )
    ax.scatter(plot_df["r2_global"], plot_df["target"], color="#2563eb", s=45, label="Best global config")
    ax.scatter(plot_df["r2_specific"], plot_df["target"], color="#dc2626", s=45, label="Best per-target config")
    ax.axvline(0.0, color="#1f2937", linestyle="--", linewidth=1)
    ax.set_title("Per-Target R2: Best Global vs Best Target-Specific")
    ax.set_xlabel("R2")
    ax.set_ylabel("Target")
    ax.legend(loc="best")
    _save(fig, output_path)
