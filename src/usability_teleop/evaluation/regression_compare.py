"""Comparison helpers for global vs target-specific regression performance."""

from __future__ import annotations

import pandas as pd


def build_global_vs_target_specific_r2(
    regression_global_df: pd.DataFrame,
    regression_target_specific_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare per-target R2 from best global config against best target-specific config."""
    if regression_global_df.empty or regression_target_specific_df.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "r2_global",
                "r2_specific",
                "delta_r2",
                "model_global",
                "feature_set_global",
                "model_specific",
                "feature_set_specific",
            ]
        )

    best_global = regression_global_df.sort_values("r2_mean", ascending=False).iloc[0]
    rows: list[dict[str, object]] = []
    targets = sorted(regression_target_specific_df["target"].unique().tolist())
    for target in targets:
        best_specific = (
            regression_target_specific_df[regression_target_specific_df["target"] == target]
            .sort_values("r2", ascending=False)
            .iloc[0]
        )
        global_col = f"r2_{target}"
        if global_col not in best_global.index:
            continue
        r2_global = float(best_global[global_col])
        r2_specific = float(best_specific["r2"])
        rows.append(
            {
                "target": target,
                "r2_global": r2_global,
                "r2_specific": r2_specific,
                "delta_r2": float(r2_specific - r2_global),
                "model_global": str(best_global["model"]),
                "feature_set_global": str(best_global["feature_set"]),
                "model_specific": str(best_specific["model"]),
                "feature_set_specific": str(best_specific["feature_set"]),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_r2", ascending=False).reset_index(drop=True)
