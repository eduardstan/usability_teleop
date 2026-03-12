"""Final-model lane: global tuning and refit on full data."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets, build_feature_set
from usability_teleop.modeling.cv import classification_inner_cv, fit_with_tuning, regression_inner_cv
from usability_teleop.modeling.registry import (
    classification_model_specs,
    regression_model_specs,
    build_estimator,
)
from usability_teleop.protocol.estimation_classification import ClassBalanceMode, _rebalance_binary_train
from usability_teleop.protocol.selection import SelectionConfig, select_full_features


def fit_final_models(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    y_cls: pd.DataFrame,
    estimation_best: pd.DataFrame,
    seed: int,
    regression_scoring: str,
    classification_scoring: str,
    regression_inner_max_splits: int,
    classification_inner_max_splits: int,
    inner_shuffle: bool,
    inner_seed: int,
    top_k_per_axis: int | None,
    class_balance: ClassBalanceMode,
    logger: object | None = None,
) -> pd.DataFrame:
    fs_specs = {fs.name: fs for fs in generate_ee_quat_feature_sets(include_average=True)}
    reg_specs = {spec.name: spec for spec in regression_model_specs()}
    cls_specs = {spec.name: spec for spec in classification_model_specs()}
    selection_cfg = SelectionConfig(top_k_per_axis=top_k_per_axis)
    rows: list[dict[str, object]] = []

    for _, row in estimation_best.iterrows():
        track = str(row["track"])
        target = str(row["target"])
        fs_name = str(row["feature_set"])
        model_name = str(row["model"])
        if fs_name not in fs_specs:
            raise ValueError(f"Unknown feature_set in estimation_best_configs: {fs_name}")
        fs = fs_specs[fs_name]
        x_fs = build_feature_set(x_user, fs)
        x_selected, selected_cols = select_full_features(x_fs, selection_cfg)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_selected.to_numpy(dtype=float))
        if track == "regression":
            if target not in y_reg.columns:
                raise ValueError(f"Unknown regression target in estimation_best_configs: {target}")
            if model_name not in reg_specs:
                raise ValueError(f"Unknown regression model in estimation_best_configs: {model_name}")
            y = y_reg[target].to_numpy(dtype=float)
            spec = reg_specs[model_name]
            model, params = fit_with_tuning(
                build_estimator(spec, seed),
                spec.param_grid,
                x_scaled,
                y,
                scoring=regression_scoring,
                cv=regression_inner_cv(
                    len(y),
                    max_splits=regression_inner_max_splits,
                    shuffle=inner_shuffle,
                    random_seed=inner_seed,
                ),
            )
            model.fit(x_scaled, y)
            rows.append(
                _row(track, target, fs_name, model_name, params, selected_cols, {"selection_score": float(row["selection_score"])})
            )
        else:
            if track != "classification":
                raise ValueError(f"Unsupported track in estimation_best_configs: {track}")
            if target not in y_cls.columns:
                raise ValueError(f"Unknown classification target in estimation_best_configs: {target}")
            if model_name not in cls_specs:
                raise ValueError(f"Unknown classification model in estimation_best_configs: {model_name}")
            y_cont = y_cls[target].to_numpy(dtype=float)
            threshold = float(row["threshold"]) if "threshold" in row and np.isfinite(float(row["threshold"])) else float(np.median(y_cont))
            y_bin = (y_cont >= threshold).astype(int)
            x_bal, y_bal = _rebalance_binary_train(x_scaled, y_bin, class_balance, seed)
            spec = cls_specs[model_name]
            model, params = fit_with_tuning(
                build_estimator(spec, seed),
                spec.param_grid,
                x_bal,
                y_bal,
                scoring=classification_scoring,
                cv=classification_inner_cv(
                    y_bal,
                    max_splits=classification_inner_max_splits,
                    shuffle=inner_shuffle,
                    random_seed=inner_seed,
                ),
            )
            model.fit(x_bal, y_bal)
            rows.append(
                _row(
                    track,
                    target,
                    fs_name,
                    model_name,
                    params,
                    selected_cols,
                    {"threshold": threshold, "selection_score": float(row["selection_score"])},
                )
            )
        if logger is not None:
            logger.info("final model fitted track=%s target=%s model=%s feature_set=%s", track, target, model_name, fs_name)
    return pd.DataFrame(rows).sort_values(["track", "target"]).reset_index(drop=True)


def _row(
    track: str,
    target: str,
    feature_set: str,
    model: str,
    final_params: dict[str, Any],
    selected_cols: list[str],
    extra: dict[str, object],
) -> dict[str, object]:
    out: dict[str, object] = {
        "track": track,
        "target": target,
        "feature_set": feature_set,
        "model": model,
        "final_params": json.dumps(final_params, sort_keys=True),
        "selected_features": json.dumps(selected_cols),
        "n_selected_features": len(selected_cols),
    }
    out.update(extra)
    return out
