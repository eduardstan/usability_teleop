"""Global multi-output regression workflow."""

from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.cv import fit_with_tuning, loso_indices, regression_inner_cv
from usability_teleop.modeling.registry import ModelSpec, build_estimator
from usability_teleop.utils.timing import ProgressTracker, format_seconds


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _run_global_task(
    fs: FeatureSetSpec,
    spec: ModelSpec,
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    random_seed: int,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
) -> tuple[str, str, dict[str, object]]:
    x_fs = build_feature_set(x_user, fs)
    y_true_all = y_reg.to_numpy(dtype=float)
    y_pred_all = np.zeros_like(y_true_all, dtype=float)
    fold_params: list[dict[str, Any]] = []

    for train_idx, test_idx in loso_indices(len(x_fs)):
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_fs.iloc[train_idx].to_numpy(dtype=float))
        x_test_s = scaler.transform(x_fs.iloc[test_idx].to_numpy(dtype=float))
        y_train = y_reg.iloc[train_idx].to_numpy(dtype=float)
        base = build_estimator(spec, random_seed)
        wrapped = MultiOutputRegressor(base)
        grid = {f"estimator__{k}": v for k, v in spec.param_grid.items()}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            best_model, best_params = fit_with_tuning(
                wrapped,
                grid,
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
        y_pred_all[test_idx, :] = best_model.predict(x_test_s)

    row: dict[str, object] = {
        "feature_set": fs.name,
        "model": spec.name,
        "rmse_mean": float(np.mean([_rmse(y_true_all[:, i], y_pred_all[:, i]) for i in range(y_true_all.shape[1])])),
        "mae_mean": float(np.mean([mean_absolute_error(y_true_all[:, i], y_pred_all[:, i]) for i in range(y_true_all.shape[1])])),
        "r2_mean": float(np.mean([r2_score(y_true_all[:, i], y_pred_all[:, i]) for i in range(y_true_all.shape[1])])),
        "best_params_last_fold": json.dumps(fold_params[-1] if fold_params else {}, sort_keys=True),
    }
    for i, target in enumerate(y_reg.columns):
        row[f"rmse_{target}"] = _rmse(y_true_all[:, i], y_pred_all[:, i])
        row[f"mae_{target}"] = float(mean_absolute_error(y_true_all[:, i], y_pred_all[:, i]))
        row[f"r2_{target}"] = float(r2_score(y_true_all[:, i], y_pred_all[:, i]))
    return fs.name, spec.name, row


def run_regression_global(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    random_seed: int = 42,
    max_feature_sets: int | None = None,
    logger: logging.Logger | None = None,
    workers: int = 1,
    tuning_scoring: str = "r2",
    inner_cv_max_splits: int = 3,
    inner_cv_shuffle: bool = True,
    inner_cv_seed: int = 42,
) -> pd.DataFrame:
    """Run global multi-output regression benchmark across feature sets and models."""
    selected_sets = feature_sets[: max_feature_sets or len(feature_sets)]
    rows: list[dict[str, object]] = []
    progress = ProgressTracker(total=max(1, len(selected_sets) * len(model_specs)))
    tasks = [(fs, spec) for fs in selected_sets for spec in model_specs]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_global_task,
                    fs,
                    spec,
                    x_user,
                    y_reg,
                    random_seed,
                    tuning_scoring,
                    inner_cv_max_splits,
                    inner_cv_shuffle,
                    inner_cv_seed,
                )
                for fs, spec in tasks
            ]
            for future in as_completed(futures):
                fs_name, model_name, row = future.result()
                rows.append(row)
                elapsed, eta = progress.step()
                if logger is not None:
                    logger.info(
                        "RQ2 global progress %s/%s | feature_set=%s model=%s | elapsed=%s eta=%s",
                        progress.completed, progress.total, fs_name, model_name, format_seconds(elapsed), format_seconds(eta)
                    )
    else:
        for fs, spec in tasks:
            fs_name, model_name, row = _run_global_task(
                fs,
                spec,
                x_user,
                y_reg,
                random_seed,
                tuning_scoring,
                inner_cv_max_splits,
                inner_cv_shuffle,
                inner_cv_seed,
            )
            rows.append(row)
            elapsed, eta = progress.step()
            if logger is not None:
                logger.info(
                    "RQ2 global progress %s/%s | feature_set=%s model=%s | elapsed=%s eta=%s",
                    progress.completed, progress.total, fs_name, model_name, format_seconds(elapsed), format_seconds(eta)
                )
    return pd.DataFrame(rows).sort_values(by="r2_mean", ascending=False).reset_index(drop=True)
