"""Helpers for LOSO prediction traces used by inference routines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.cv import (
    classification_inner_cv,
    fit_with_tuning,
    loso_indices,
    regression_inner_cv,
)
from usability_teleop.modeling.registry import ModelSpec, build_estimator


def loso_regression_trace(
    x_user: pd.DataFrame,
    y: np.ndarray,
    fs: FeatureSetSpec,
    spec: ModelSpec,
    random_seed: int,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_fs = build_feature_set(x_user, fs)
    y_true = y.astype(float).copy()
    y_pred = np.zeros_like(y_true, dtype=float)
    for train_idx, test_idx in loso_indices(len(x_fs)):
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_fs.iloc[train_idx].to_numpy(dtype=float))
        x_test_s = scaler.transform(x_fs.iloc[test_idx].to_numpy(dtype=float))
        y_train = y_true[train_idx]
        model, _ = fit_with_tuning(
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
        y_pred[test_idx] = model.predict(x_test_s)
    return y_true, y_pred


def loso_classification_trace(
    x_user: pd.DataFrame,
    y_bin: np.ndarray,
    fs: FeatureSetSpec,
    spec: ModelSpec,
    random_seed: int,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_fs = build_feature_set(x_user, fs)
    y_true = y_bin.astype(int).copy()
    y_pred = np.zeros_like(y_true)
    y_score = np.zeros_like(y_true, dtype=float)
    for train_idx, test_idx in loso_indices(len(x_fs)):
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_fs.iloc[train_idx].to_numpy(dtype=float))
        x_test_s = scaler.transform(x_fs.iloc[test_idx].to_numpy(dtype=float))
        y_train = y_true[train_idx]
        model, _ = fit_with_tuning(
            build_estimator(spec, random_seed),
            spec.param_grid,
            x_train_s,
            y_train,
            scoring=tuning_scoring,
            cv=classification_inner_cv(
                y_train,
                max_splits=inner_cv_max_splits,
                shuffle=inner_cv_shuffle,
                random_seed=inner_cv_seed,
            ),
        )
        y_pred[test_idx] = model.predict(x_test_s)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test_s)
            y_score[test_idx] = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(x_test_s)
            y_score[test_idx] = score if np.ndim(score) else [score]
        else:
            y_score[test_idx] = y_pred[test_idx]
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    return y_true, y_pred, y_score, auc
