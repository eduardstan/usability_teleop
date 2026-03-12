from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from usability_teleop.analysis.preprocessing import filter_axis_top_variance
from usability_teleop.evaluation.classification import ClassBalanceMode, run_classification_benchmark
from usability_teleop.evaluation.regression import run_regression_target_specific
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs


@dataclass(frozen=True)
class PrototypeOutputs:
    summary: pd.DataFrame
    breakdown: pd.DataFrame
    feature_filter_summary: pd.DataFrame


def _best_regression_scores(df: pd.DataFrame) -> pd.DataFrame:
    best = df.sort_values("r2", ascending=False).groupby("target", as_index=False).first()
    best["track"] = "regression"
    best["metric"] = "r2"
    best["value"] = best["r2"]
    return best[["track", "target", "metric", "value"]]


def _best_classification_scores(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame(columns=["track", "target", "metric", "value"])
    best = valid.sort_values("auc", ascending=False).groupby("target", as_index=False).first()
    best["track"] = "classification"
    best["metric"] = "auc"
    best["value"] = best["auc"]
    return best[["track", "target", "metric", "value"]]


def run_incremental_prototype(
    x_base: pd.DataFrame,
    y_reg: pd.DataFrame,
    y_cls: pd.DataFrame,
    max_models: int,
    max_feature_sets: int,
    top_k_per_axis: int,
    class_balance: ClassBalanceMode,
    seed: int,
    workers: int,
    tuning_regression_scoring: str,
    tuning_classification_scoring: str,
    inner_regression_splits: int,
    inner_classification_splits: int,
    inner_shuffle: bool,
    inner_seed: int,
    logger: object | None = None,
) -> PrototypeOutputs:
    feature_sets = generate_ee_quat_feature_sets(include_average=True)
    reg_models = regression_model_specs()[:max_models]
    cls_models = classification_model_specs()[:max_models]
    x_filtered, feature_filter_summary = filter_axis_top_variance(x_base, top_k_per_axis=top_k_per_axis)

    stages = [
        ("baseline", x_base, "none"),
        ("feature_filter", x_filtered, "none"),
        ("feature_filter_balance", x_filtered, class_balance),
    ]
    summary_rows: list[dict[str, object]] = []
    breakdown_frames: list[pd.DataFrame] = []
    baseline_map: dict[tuple[str, str], float] = {}
    for stage_name, x_user, rebalance in stages:
        t0 = time.perf_counter()
        if logger is not None:
            logger.info(
                "prototype stage=%s | users=%s cols=%s rebalance=%s",
                stage_name,
                len(x_user),
                x_user.shape[1],
                rebalance,
            )
        reg_df = run_regression_target_specific(
            x_user,
            y_reg,
            feature_sets,
            reg_models,
            random_seed=seed,
            max_feature_sets=max_feature_sets,
            logger=logger,
            workers=workers,
            tuning_scoring=tuning_regression_scoring,
            inner_cv_max_splits=inner_regression_splits,
            inner_cv_shuffle=inner_shuffle,
            inner_cv_seed=inner_seed,
        )
        cls_df = run_classification_benchmark(
            x_user,
            y_cls,
            feature_sets,
            cls_models,
            random_seed=seed,
            max_feature_sets=max_feature_sets,
            tuning_scoring=tuning_classification_scoring,
            inner_cv_max_splits=inner_classification_splits,
            inner_cv_shuffle=inner_shuffle,
            inner_cv_seed=inner_seed,
            class_balance=rebalance,
        )
        reg_best = _best_regression_scores(reg_df)
        cls_best = _best_classification_scores(cls_df)
        stage_breakdown = pd.concat([reg_best, cls_best], ignore_index=True)
        stage_breakdown["stage"] = stage_name
        breakdown_frames.append(stage_breakdown)
        elapsed = time.perf_counter() - t0
        summary_rows.append(
            {
                "stage": stage_name,
                "rebalance": rebalance,
                "n_users": int(len(x_user)),
                "n_columns": int(x_user.shape[1]),
                "regression_mean_best_r2": float(reg_best["value"].mean()),
                "classification_mean_best_auc": float(cls_best["value"].mean())
                if not cls_best.empty
                else float("nan"),
                "elapsed_seconds": elapsed,
            }
        )

    breakdown = pd.concat(breakdown_frames, ignore_index=True)
    for _, row in breakdown.iterrows():
        key = (str(row["track"]), str(row["target"]))
        if str(row["stage"]) == "baseline":
            baseline_map[key] = float(row["value"])
    deltas: list[float] = []
    for _, row in breakdown.iterrows():
        key = (str(row["track"]), str(row["target"]))
        base = baseline_map.get(key, float("nan"))
        deltas.append(float(row["value"]) - base if np.isfinite(base) else float("nan"))
    breakdown["delta_vs_baseline"] = deltas
    return PrototypeOutputs(
        summary=pd.DataFrame(summary_rows),
        breakdown=breakdown,
        feature_filter_summary=feature_filter_summary,
    )
