"""Nested-LOSO estimation for regression track."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.cv import fit_with_tuning, loso_indices, regression_inner_cv
from usability_teleop.modeling.registry import ModelSpec, build_estimator
from usability_teleop.protocol.selection import SelectionConfig, pack_fold_feature_counts, select_train_test_features


def run_regression_estimation(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    random_seed: int,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
    selection_cfg: SelectionConfig,
    logger: object | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in y_reg.columns:
        y_target = y_reg[target].to_numpy(dtype=float)
        for fs in feature_sets:
            x_fs = build_feature_set(x_user, fs)
            for spec in model_specs:
                y_pred = np.zeros_like(y_target, dtype=float)
                fold_params: list[dict[str, Any]] = []
                fold_counts: list[int] = []
                for train_idx, test_idx in loso_indices(len(x_fs)):
                    x_train_raw = x_fs.iloc[train_idx]
                    x_test_raw = x_fs.iloc[test_idx]
                    x_train, x_test, selected_cols = select_train_test_features(
                        x_train_raw,
                        x_test_raw,
                        selection_cfg,
                    )
                    fold_counts.append(len(selected_cols))
                    scaler = StandardScaler()
                    x_train_s = scaler.fit_transform(x_train.to_numpy(dtype=float))
                    x_test_s = scaler.transform(x_test.to_numpy(dtype=float))
                    y_train = y_target[train_idx]
                    model, best_params = fit_with_tuning(
                        build_estimator(spec, random_seed),
                        spec.param_grid,
                        x_train_s,
                        y_train,
                        scoring=tuning_scoring,
                        cv=regression_inner_cv(
                            len(train_idx),
                            max_splits=inner_cv_max_splits,
                            shuffle=inner_cv_shuffle,
                            random_seed=inner_cv_seed,
                        ),
                    )
                    fold_params.append(best_params)
                    y_pred[test_idx] = model.predict(x_test_s)
                rows.append(
                    {
                        "track": "regression",
                        "target": target,
                        "feature_set": fs.name,
                        "model": spec.name,
                        "rmse": float(np.sqrt(mean_squared_error(y_target, y_pred))),
                        "mae": float(mean_absolute_error(y_target, y_pred)),
                        "r2": float(r2_score(y_target, y_pred)),
                        "fold_best_params": json.dumps(fold_params, sort_keys=True),
                        "fold_feature_counts": pack_fold_feature_counts(fold_counts),
                    }
                )
                if logger is not None:
                    logger.info(
                        "estimation regression target=%s feature_set=%s model=%s done",
                        target,
                        fs.name,
                        spec.name,
                    )
    return pd.DataFrame(rows).sort_values(["target", "r2"], ascending=[True, False]).reset_index(drop=True)
