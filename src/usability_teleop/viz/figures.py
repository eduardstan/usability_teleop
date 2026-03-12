"""Publication-quality figure builders from result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from usability_teleop.viz.theme import (
    NEUTRAL_DARK,
    NEUTRAL_MID,
    PRIMARY_BLUE,
    PRIMARY_RED,
    apply_publication_theme,
)


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
    sns.heatmap(matrix, cmap="RdBu_r", center=0.0, ax=ax)
    ax.set_title("Top Pearson Correlations (Highlighted)")
    _save(fig, output_path)


def plot_regression_overview(regression_global_df: pd.DataFrame, output_path: Path, top_k: int = 10) -> None:
    """Bar plot for top global regression configurations by mean R2."""
    apply_publication_theme()
    data = regression_global_df.copy()
    if "r2_mean" not in data.columns and {"model", "feature_set", "r2"}.issubset(data.columns):
        data = (
            data.groupby(["model", "feature_set"], as_index=False)["r2"]
            .mean()
            .rename(columns={"r2": "r2_mean"})
        )
    if "r2_mean" not in data.columns:
        return
    top = data.sort_values("r2_mean", ascending=False).head(top_k).copy()
    top["label"] = top["model"] + " | " + top["feature_set"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top, x="r2_mean", y="label", color=PRIMARY_BLUE, ax=ax)
    ax.axvline(0.0, color=NEUTRAL_DARK, linestyle="--", linewidth=1)
    ax.set_title("Global Regression Benchmark (Top by Mean R2)")
    ax.set_xlabel("Mean R2")
    ax.set_ylabel("Configuration")
    _save(fig, output_path)


def plot_classification_overview(classification_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot for best AUC per target from classification benchmark."""
    apply_publication_theme()
    if "status" in classification_df.columns:
        valid = classification_df[classification_df["status"] == "ok"].copy()
    else:
        valid = classification_df.copy()
    if valid.empty:
        return
    best = valid.sort_values(["target", "auc"], ascending=[True, False]).groupby("target", as_index=False).head(1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=best, x="auc", y="target", color=PRIMARY_RED, ax=ax)
    ax.axvline(0.5, color=NEUTRAL_DARK, linestyle=":", linewidth=1)
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
    ax.axvline(0.05, color=PRIMARY_RED, linestyle="--", linewidth=1, label="alpha=0.05")
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
        color=NEUTRAL_MID,
        linewidth=2,
    )
    ax.scatter(plot_df["r2_global"], plot_df["target"], color=PRIMARY_BLUE, s=45, label="Best global config")
    ax.scatter(plot_df["r2_specific"], plot_df["target"], color=PRIMARY_RED, s=45, label="Best per-target config")
    ax.axvline(0.0, color=NEUTRAL_DARK, linestyle="--", linewidth=1)
    ax.set_title("Per-Target R2: Best Global vs Best Target-Specific")
    ax.set_xlabel("R2")
    ax.set_ylabel("Target")
    ax.legend(loc="best")
    _save(fig, output_path)


def plot_protocol_dashboard(
    comparison_df: pd.DataFrame,
    perm_reg_df: pd.DataFrame,
    perm_cls_df: pd.DataFrame,
    inf_reg_df: pd.DataFrame,
    inf_cls_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2x2 executive dashboard for key statistical outcomes."""
    apply_publication_theme()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    if not comparison_df.empty:
        top = comparison_df.sort_values("delta_r2", ascending=False).head(8)
        sns.barplot(data=top, x="delta_r2", y="target", color=PRIMARY_BLUE, ax=ax1)
        ax1.axvline(0.0, color=NEUTRAL_DARK, linestyle="--", linewidth=1)
    ax1.set_title("Delta R2 (Local - Global)")
    ax1.set_xlabel("delta R2")
    ax1.set_ylabel("Target")

    perm_frames: list[pd.DataFrame] = []
    if not perm_reg_df.empty:
        t = perm_reg_df[["target", "p_value"]].copy()
        t["track"] = "regression"
        perm_frames.append(t)
    if not perm_cls_df.empty:
        t = perm_cls_df[["target", "p_value"]].copy()
        t["track"] = "classification"
        perm_frames.append(t)
    if perm_frames:
        perm_all = pd.concat(perm_frames, ignore_index=True)
        sns.histplot(data=perm_all, x="p_value", hue="track", bins=10, element="step", stat="count", ax=ax2)
        ax2.axvline(0.05, color=PRIMARY_RED, linestyle="--", linewidth=1)
    ax2.set_title("Permutation p-value Distribution")
    ax2.set_xlabel("p-value")
    ax2.set_ylabel("count")

    inf_frames: list[pd.DataFrame] = []
    if not inf_reg_df.empty:
        t = inf_reg_df[["target", "bayes_prob_improvement"]].copy()
        t["track"] = "regression"
        inf_frames.append(t)
    if not inf_cls_df.empty:
        t = inf_cls_df[["target", "bayes_prob_improvement"]].copy()
        t["track"] = "classification"
        inf_frames.append(t)
    if inf_frames:
        inf_all = pd.concat(inf_frames, ignore_index=True)
        sns.boxplot(data=inf_all, x="track", y="bayes_prob_improvement", color=PRIMARY_BLUE, ax=ax3)
        sns.stripplot(data=inf_all, x="track", y="bayes_prob_improvement", color=NEUTRAL_DARK, size=3, alpha=0.65, ax=ax3)
        ax3.axhline(0.5, color=NEUTRAL_DARK, linestyle="--", linewidth=1)
        ax3.axhline(0.95, color=PRIMARY_RED, linestyle=":", linewidth=1)
    ax3.set_title("Bayesian Improvement Probabilities")
    ax3.set_xlabel("track")
    ax3.set_ylabel("P(improvement)")

    sig_reg = int(perm_reg_df["significant"].sum()) if not perm_reg_df.empty and "significant" in perm_reg_df.columns else 0
    sig_cls = int(perm_cls_df["significant"].sum()) if not perm_cls_df.empty and "significant" in perm_cls_df.columns else 0
    summary = pd.DataFrame(
        {
            "metric": ["sig_reg_targets", "sig_cls_targets"],
            "value": [sig_reg, sig_cls],
        }
    )
    sns.barplot(data=summary, x="metric", y="value", color=PRIMARY_RED, ax=ax4)
    ax4.set_title("Significant Targets (p < 0.05)")
    ax4.set_xlabel("")
    ax4.set_ylabel("count")
    for label in ax4.get_xticklabels():
        label.set_rotation(15)

    _save(fig, output_path)
