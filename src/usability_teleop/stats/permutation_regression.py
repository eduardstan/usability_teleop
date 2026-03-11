"""Permutation testing for target-specific regression winners."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, permutation_test_score

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.registry import ModelSpec, build_estimator
from usability_teleop.stats.inference_utils import loso_regression_trace
from usability_teleop.stats.permutation_config import PermutationConfig
from usability_teleop.stats.permutation_shared import (
    feature_set_by_name,
    params_from_json,
    spec_by_name,
)


def run_regression_permutation_tests(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    regression_target_results: pd.DataFrame,
    config: PermutationConfig | None = None,
    nested_tuning_scoring: str = "r2",
    inner_cv_max_splits: int = 3,
    inner_cv_shuffle: bool = True,
    inner_cv_seed: int = 42,
) -> pd.DataFrame:
    """Run permutation tests on best per-target regression configurations."""
    cfg = config or PermutationConfig()
    rows: list[dict[str, object]] = []

    for target in y_reg.columns:
        target_rows = regression_target_results[regression_target_results["target"] == target]
        if target_rows.empty:
            continue
        best = target_rows.sort_values("r2", ascending=False).iloc[0]

        model_name = str(best["model"])
        feature_name = str(best["feature_set"])
        params = params_from_json(str(best.get("best_params_last_fold", "{}")))

        spec = spec_by_name(model_specs, model_name)
        fs = feature_set_by_name(feature_sets, feature_name)
        x_fs = build_feature_set(x_user, fs)
        y = y_reg[target].to_numpy(dtype=float)

        estimator = build_estimator(spec, cfg.random_seed)
        if params:
            estimator.set_params(**params)

        if not cfg.nested:
            score, perm_scores, p_value = permutation_test_score(
                estimator,
                x_fs.to_numpy(dtype=float),
                y,
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error",
                n_permutations=cfg.n_permutations,
                n_jobs=1,
                random_state=cfg.random_seed,
            )
            rmse_observed = float(np.sqrt(-score))
            rmse_perm = np.sqrt(-perm_scores)
        else:
            rng = np.random.default_rng(cfg.random_seed)
            y_true_obs, y_pred_obs = loso_regression_trace(
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
            rmse_observed = float(np.sqrt(np.mean((y_true_obs - y_pred_obs) ** 2)))
            perm_rmses: list[float] = []
            for _ in range(cfg.n_permutations):
                y_perm = rng.permutation(y)
                y_true_p, y_pred_p = loso_regression_trace(
                    x_user,
                    y_perm,
                    fs,
                    spec,
                    cfg.random_seed,
                    nested_tuning_scoring,
                    inner_cv_max_splits,
                    inner_cv_shuffle,
                    inner_cv_seed,
                )
                perm_rmses.append(float(np.sqrt(np.mean((y_true_p - y_pred_p) ** 2))))
            rmse_perm = np.asarray(perm_rmses, dtype=float)
            p_value = float((1 + np.sum(rmse_perm <= rmse_observed)) / (len(rmse_perm) + 1))
        rows.append(
            {
                "target": target,
                "model": model_name,
                "feature_set": feature_name,
                "rmse_observed": rmse_observed,
                "rmse_perm_mean": float(np.mean(rmse_perm)),
                "p_value": float(p_value),
                "significant": bool(p_value < cfg.alpha),
            }
        )

    return pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)
