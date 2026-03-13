from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from usability_teleop.analysis.preprocessing import filter_axis_top_variance
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs
from usability_teleop.protocol.estimation_classification import run_classification_estimation
from usability_teleop.protocol.estimation_regression import run_regression_estimation
from usability_teleop.protocol.selection import SelectionConfig


@dataclass(frozen=True)
class StudyOutputs:
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
    valid = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if valid.empty:
        return pd.DataFrame(columns=["track", "target", "metric", "value"])
    best = valid.sort_values("auc", ascending=False).groupby("target", as_index=False).first()
    best["track"] = "classification"
    best["metric"] = "auc"
    best["value"] = best["auc"]
    return best[["track", "target", "metric", "value"]]


def run_ablation_study(
    x_base: pd.DataFrame,
    y_reg: pd.DataFrame,
    y_cls: pd.DataFrame,
    max_models: int,
    max_feature_sets: int,
    topk_values: list[int],
    seed: int,
    workers: int,
    tuning_regression_scoring: str,
    tuning_classification_scoring: str,
    inner_regression_splits: int,
    inner_classification_splits: int,
    inner_shuffle: bool,
    inner_seed: int,
    models_config: Path | None,
    logger: object | None = None,
) -> StudyOutputs:
    feature_sets = generate_ee_quat_feature_sets(include_average=True)
    reg_models = regression_model_specs(config_path=models_config)[:max_models]
    cls_models = classification_model_specs(config_path=models_config)[:max_models]
    if max_feature_sets is not None:
        feature_sets = feature_sets[:max_feature_sets]

    stages: list[tuple[str, SelectionConfig]] = [("baseline", SelectionConfig(top_k_per_axis=None))]
    for k in sorted(set(topk_values)):
        stages.append((f"variance_topk_{k}", SelectionConfig(top_k_per_axis=int(k))))

    summary_rows: list[dict[str, object]] = []
    breakdown_frames: list[pd.DataFrame] = []
    filter_rows: list[pd.DataFrame] = []
    baseline_map: dict[tuple[str, str], float] = {}
    for stage_name, selection_cfg in stages:
        t0 = time.perf_counter()
        if logger is not None:
            logger.info(
                "study stage=%s | users=%s cols=%s selection.top_k_per_axis=%s",
                stage_name,
                len(x_base),
                x_base.shape[1],
                selection_cfg.top_k_per_axis,
            )
        reg_df = run_regression_estimation(
            x_base,
            y_reg,
            feature_sets,
            reg_models,
            random_seed=seed,
            tuning_scoring=tuning_regression_scoring,
            inner_cv_max_splits=inner_regression_splits,
            inner_cv_shuffle=inner_shuffle,
            inner_cv_seed=inner_seed,
            selection_cfg=selection_cfg,
            logger=logger,
        )
        cls_df = run_classification_estimation(
            x_base,
            y_cls,
            feature_sets,
            cls_models,
            random_seed=seed,
            tuning_scoring=tuning_classification_scoring,
            inner_cv_max_splits=inner_classification_splits,
            inner_cv_shuffle=inner_shuffle,
            inner_cv_seed=inner_seed,
            class_balance="none",
            selection_cfg=selection_cfg,
            logger=logger,
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
                "selection_top_k_per_axis": selection_cfg.top_k_per_axis,
                "n_users": int(len(x_base)),
                "n_columns": int(x_base.shape[1]),
                "regression_mean_best_r2": float(reg_best["value"].mean()),
                "classification_mean_best_auc": float(cls_best["value"].mean())
                if not cls_best.empty
                else float("nan"),
                "elapsed_seconds": elapsed,
            }
        )
        if selection_cfg.top_k_per_axis is None:
            _, summary = filter_axis_top_variance(x_base, top_k_per_axis=10_000)
        else:
            _, summary = filter_axis_top_variance(x_base, top_k_per_axis=selection_cfg.top_k_per_axis)
        summary["stage"] = stage_name
        filter_rows.append(summary)

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
    return StudyOutputs(
        summary=pd.DataFrame(summary_rows),
        breakdown=breakdown,
        feature_filter_summary=pd.concat(filter_rows, ignore_index=True) if filter_rows else pd.DataFrame(),
    )
