"""Permutation testing for target-specific classification winners."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.registry import ModelSpec, build_estimator
from usability_teleop.stats.inference_utils import loso_classification_trace
from usability_teleop.stats.permutation_config import PermutationConfig
from usability_teleop.stats.permutation_shared import (
    feature_set_by_name,
    params_from_result_row,
    spec_by_name,
)


def run_classification_permutation_tests(
    x_user: pd.DataFrame,
    y_cls: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    classification_results: pd.DataFrame,
    config: PermutationConfig | None = None,
    nested_tuning_scoring: str = "roc_auc",
    inner_cv_max_splits: int = 3,
    inner_cv_shuffle: bool = True,
    inner_cv_seed: int = 42,
) -> pd.DataFrame:
    """Run permutation tests on best per-target classification configurations."""
    cfg = config or PermutationConfig()
    rows: list[dict[str, object]] = []
    if "status" in classification_results.columns:
        valid = classification_results[classification_results["status"] == "ok"]
    else:
        valid = classification_results

    for target in y_cls.columns:
        target_rows = valid[valid["target"] == target]
        if target_rows.empty:
            continue
        best = target_rows.sort_values("auc", ascending=False).iloc[0]

        model_name = str(best["model"])
        feature_name = str(best["feature_set"])
        threshold = float(best["threshold"])
        params = params_from_result_row(best)
        y = (y_cls[target].to_numpy(dtype=float) >= threshold).astype(int)
        if len(np.unique(y)) < 2:
            continue

        spec = spec_by_name(model_specs, model_name)
        fs = feature_set_by_name(feature_sets, feature_name)
        x_fs = build_feature_set(x_user, fs).to_numpy(dtype=float)

        estimator = build_estimator(spec, cfg.random_seed)
        if params:
            estimator.set_params(**params)

        if not cfg.nested:
            observed_auc = _loso_auc_score(estimator, x_fs, y)
            rng = np.random.default_rng(cfg.random_seed)
            perm_scores: list[float] = []
            for _ in range(cfg.n_permutations):
                perm_auc = _loso_auc_score(estimator, x_fs, rng.permutation(y))
                if np.isfinite(perm_auc):
                    perm_scores.append(float(perm_auc))
        else:
            _, _, _, observed_auc = loso_classification_trace(
                x_user,
                y,
                fs,
                spec,
                cfg.random_seed,
                nested_tuning_scoring,
                inner_cv_max_splits,
                inner_cv_shuffle,
                inner_cv_seed,
            )
            rng = np.random.default_rng(cfg.random_seed)
            perm_scores = []
            for _ in range(cfg.n_permutations):
                _, _, _, perm_auc = loso_classification_trace(
                    x_user,
                    rng.permutation(y),
                    fs,
                    spec,
                    cfg.random_seed,
                    nested_tuning_scoring,
                    inner_cv_max_splits,
                    inner_cv_shuffle,
                    inner_cv_seed,
                )
                if np.isfinite(perm_auc):
                    perm_scores.append(float(perm_auc))

        perm_arr = np.asarray(perm_scores, dtype=float)
        if perm_arr.size == 0 or not np.isfinite(observed_auc):
            p_value = float("nan")
            perm_mean = float("nan")
        else:
            p_value = float((1 + np.sum(perm_arr >= observed_auc)) / (len(perm_arr) + 1))
            perm_mean = float(np.mean(perm_arr))

        rows.append(
            {
                "target": target,
                "model": model_name,
                "feature_set": feature_name,
                "auc_observed": observed_auc,
                "auc_perm_mean": perm_mean,
                "p_value": float(p_value),
                "significant": bool(np.isfinite(p_value) and p_value < cfg.alpha),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)


def _loso_auc_score(estimator: Any, x: np.ndarray, y: np.ndarray) -> float:
    y_true = np.zeros_like(y)
    y_score = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(x):
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x[train_idx])
        x_test_s = scaler.transform(x[test_idx])
        model = estimator.__class__(**estimator.get_params())
        model.fit(x_train_s, y[train_idx])
        y_true[test_idx] = y[test_idx]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test_s)
            y_score[test_idx] = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(x_test_s)
            y_score[test_idx] = score if np.ndim(score) else [score]
        else:
            y_score[test_idx] = model.predict(x_test_s)

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
