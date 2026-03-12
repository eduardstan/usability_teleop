"""Fold-safe feature selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionConfig:
    top_k_per_axis: int | None = None


def select_train_test_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    cfg: SelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply unsupervised train-only feature screening and mirror on test split."""
    if cfg.top_k_per_axis is None:
        cols = list(x_train.columns)
        return x_train.copy(), x_test[cols].copy(), cols

    keep = _top_variance_cols_by_axis(x_train, cfg.top_k_per_axis)
    return x_train[keep].copy(), x_test[keep].copy(), keep


def select_full_features(x_full: pd.DataFrame, cfg: SelectionConfig) -> tuple[pd.DataFrame, list[str]]:
    if cfg.top_k_per_axis is None:
        cols = list(x_full.columns)
        return x_full.copy(), cols
    keep = _top_variance_cols_by_axis(x_full, cfg.top_k_per_axis)
    return x_full[keep].copy(), keep


def _top_variance_cols_by_axis(x_df: pd.DataFrame, top_k_per_axis: int) -> list[str]:
    if top_k_per_axis <= 0:
        raise ValueError("top_k_per_axis must be > 0")

    axes = ("x", "y", "z", "w", "avg")
    keep: set[str] = set()
    for axis in axes:
        axis_cols = [col for col in x_df.columns if f"_ee_quat.{axis}" in col]
        if not axis_cols:
            continue
        variances = x_df[axis_cols].var(axis=0).sort_values(ascending=False)
        keep.update(variances.index[:top_k_per_axis].tolist())
    keep.update([col for col in x_df.columns if "_ee_quat." not in col])
    if not keep:
        return list(x_df.columns)
    return sorted(keep)


def pack_fold_feature_counts(counts: list[int]) -> str:
    return ",".join(str(int(v)) for v in counts)


def unpack_fold_feature_counts(raw: str) -> np.ndarray:
    if not raw:
        return np.asarray([], dtype=int)
    return np.asarray([int(v) for v in raw.split(",") if v], dtype=int)
