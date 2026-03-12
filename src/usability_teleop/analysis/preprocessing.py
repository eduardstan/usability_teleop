"""Preprocessing helpers for prototype experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

AXES: tuple[str, ...] = ("x", "y", "z", "w")


def filter_axis_top_variance(
    x_user: pd.DataFrame,
    top_k_per_axis: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_k_per_axis <= 0:
        raise ValueError("top_k_per_axis must be > 0")

    keep: set[str] = set()
    rows: list[dict[str, object]] = []
    for axis in AXES:
        axis_cols = [col for col in x_user.columns if f"_ee_quat.{axis}" in col]
        if not axis_cols:
            rows.append({"axis": axis, "n_before": 0, "n_after": 0})
            continue
        variances = x_user[axis_cols].var(axis=0).sort_values(ascending=False)
        kept_axis_cols = variances.index[:top_k_per_axis].tolist()
        keep.update(kept_axis_cols)
        rows.append({"axis": axis, "n_before": len(axis_cols), "n_after": len(kept_axis_cols)})

    non_axis_cols = [col for col in x_user.columns if "_ee_quat." not in col]
    keep.update(non_axis_cols)
    return x_user[sorted(keep)].copy(), pd.DataFrame(rows)


def build_target_distribution_table(y_reg: pd.DataFrame, y_cls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in y_reg.columns:
        values = y_reg[target].to_numpy(dtype=float)
        rows.append(
            {
                "track": "regression",
                "target": target,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "skew": float(pd.Series(values).skew()),
                "n": int(values.shape[0]),
            }
        )
    for target in y_cls.columns:
        values = y_cls[target].to_numpy(dtype=float)
        threshold = float(np.median(values))
        y_bin = (values >= threshold).astype(int)
        n0 = int(np.sum(y_bin == 0))
        n1 = int(np.sum(y_bin == 1))
        minority_ratio = float(min(n0, n1) / max(1, (n0 + n1)))
        rows.append(
            {
                "track": "classification",
                "target": target,
                "threshold": threshold,
                "n_class0": n0,
                "n_class1": n1,
                "minority_ratio": minority_ratio,
                "imbalanced_lt_35pct": bool(minority_ratio < 0.35),
                "n": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)
