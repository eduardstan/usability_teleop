"""Correlation workflow (RQ1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class CorrelationConfig:
    """Configuration for correlation significance filtering."""

    alpha: float = 0.05
    effect_threshold: float = 0.30


def run_correlation_analysis(
    x_user: pd.DataFrame,
    y_corr: pd.DataFrame,
    config: CorrelationConfig | None = None,
) -> pd.DataFrame:
    """Compute Pearson/Spearman correlations for all feature-target pairs."""
    cfg = config or CorrelationConfig()

    common_users = sorted(set(x_user.index.tolist()) & set(y_corr.index.tolist()))
    x_aligned = x_user.loc[common_users]
    y_aligned = y_corr.loc[common_users]

    rows: list[dict[str, object]] = []

    for target in y_aligned.columns:
        y_vals = y_aligned[target].to_numpy(dtype=float)
        for feature in x_aligned.columns:
            x_vals = x_aligned[feature].to_numpy(dtype=float)

            if np.allclose(x_vals, x_vals[0]) or np.allclose(y_vals, y_vals[0]):
                pearson_r, pearson_p = np.nan, np.nan
                spearman_rho, spearman_p = np.nan, np.nan
            else:
                pearson_r, pearson_p = pearsonr(x_vals, y_vals)
                spearman_rho, spearman_p = spearmanr(x_vals, y_vals)

            pearson_sig = bool(np.isfinite(pearson_p) and pearson_p < cfg.alpha)
            spearman_sig = bool(np.isfinite(spearman_p) and spearman_p < cfg.alpha)
            pearson_effective = bool(np.isfinite(pearson_r) and abs(pearson_r) >= cfg.effect_threshold)
            spearman_effective = bool(
                np.isfinite(spearman_rho) and abs(spearman_rho) >= cfg.effect_threshold
            )

            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "pearson_significant": pearson_sig,
                    "spearman_significant": spearman_sig,
                    "pearson_effective": pearson_effective,
                    "spearman_effective": spearman_effective,
                    "pearson_highlight": pearson_sig and pearson_effective,
                    "spearman_highlight": spearman_sig and spearman_effective,
                }
            )

    return pd.DataFrame(rows)
