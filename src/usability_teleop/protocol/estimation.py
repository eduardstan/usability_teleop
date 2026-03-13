"""Unified estimation lane orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs
from usability_teleop.protocol.estimation_classification import ClassBalanceMode, run_classification_estimation
from usability_teleop.protocol.estimation_regression import run_regression_estimation
from usability_teleop.protocol.selection import SelectionConfig


@dataclass(frozen=True)
class EstimationOutputs:
    regression: pd.DataFrame
    classification: pd.DataFrame
    best_configs: pd.DataFrame


def run_estimation_lane(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    y_cls: pd.DataFrame,
    seed: int,
    max_models: int | None,
    max_feature_sets: int | None,
    regression_scoring: str,
    classification_scoring: str,
    regression_inner_max_splits: int,
    classification_inner_max_splits: int,
    inner_shuffle: bool,
    inner_seed: int,
    top_k_per_axis: int | None,
    class_balance: ClassBalanceMode,
    workers: int = 1,
    models_config: Path | None = None,
    logger: object | None = None,
) -> EstimationOutputs:
    feature_sets = generate_ee_quat_feature_sets(include_average=True)
    if max_feature_sets is not None:
        feature_sets = feature_sets[:max_feature_sets]
    reg_pool = regression_model_specs(config_path=models_config)
    cls_pool = classification_model_specs(config_path=models_config)
    reg_models = reg_pool[:max_models] if max_models is not None else reg_pool
    cls_models = cls_pool[:max_models] if max_models is not None else cls_pool
    selection_cfg = SelectionConfig(top_k_per_axis=top_k_per_axis)

    workers = max(1, int(workers))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, 2)) as executor:
            reg_future = executor.submit(
                run_regression_estimation,
                x_user,
                y_reg,
                feature_sets,
                reg_models,
                random_seed=seed,
                tuning_scoring=regression_scoring,
                inner_cv_max_splits=regression_inner_max_splits,
                inner_cv_shuffle=inner_shuffle,
                inner_cv_seed=inner_seed,
                selection_cfg=selection_cfg,
                logger=logger,
            )
            cls_future = executor.submit(
                run_classification_estimation,
                x_user,
                y_cls,
                feature_sets,
                cls_models,
                random_seed=seed,
                tuning_scoring=classification_scoring,
                inner_cv_max_splits=classification_inner_max_splits,
                inner_cv_shuffle=inner_shuffle,
                inner_cv_seed=inner_seed,
                class_balance=class_balance,
                selection_cfg=selection_cfg,
                logger=logger,
            )
            reg_df = reg_future.result()
            cls_df = cls_future.result()
    else:
        reg_df = run_regression_estimation(
            x_user,
            y_reg,
            feature_sets,
            reg_models,
            random_seed=seed,
            tuning_scoring=regression_scoring,
            inner_cv_max_splits=regression_inner_max_splits,
            inner_cv_shuffle=inner_shuffle,
            inner_cv_seed=inner_seed,
            selection_cfg=selection_cfg,
            logger=logger,
        )
        cls_df = run_classification_estimation(
            x_user,
            y_cls,
            feature_sets,
            cls_models,
            random_seed=seed,
            tuning_scoring=classification_scoring,
            inner_cv_max_splits=classification_inner_max_splits,
            inner_cv_shuffle=inner_shuffle,
            inner_cv_seed=inner_seed,
            class_balance=class_balance,
            selection_cfg=selection_cfg,
            logger=logger,
        )
    best_reg = reg_df.sort_values("r2", ascending=False).groupby("target", as_index=False).first()
    best_reg["selection_metric"] = "r2"
    best_reg["selection_score"] = best_reg["r2"]
    best_cls = cls_df.sort_values("auc", ascending=False).groupby("target", as_index=False).first()
    best_cls["selection_metric"] = "auc"
    best_cls["selection_score"] = best_cls["auc"]
    best = pd.concat([best_reg, best_cls], ignore_index=True, sort=False)
    return EstimationOutputs(
        regression=reg_df,
        classification=cls_df,
        best_configs=best,
    )
