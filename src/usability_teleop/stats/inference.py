"""Extended statistical inference bundle for RQ2/RQ3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from usability_teleop.features.ee_quat import FeatureSetSpec
from usability_teleop.modeling.registry import ModelSpec
from usability_teleop.stats.inference_utils import loso_classification_trace, loso_regression_trace
from usability_teleop.stats.permutation_shared import feature_set_by_name, spec_by_name


@dataclass(frozen=True)
class InferenceBundleConfig:
    baseline_regression_model: str
    baseline_classification_model: str
    bootstrap_iterations: int
    bayesian_bootstrap_samples: int
    paired_alpha: float
    fdr_alpha: float
    random_seed: int


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(np.nan_to_num(p_values, nan=1.0))
    ranked = p_values[order]
    adjusted = np.full(n, np.nan, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        p = ranked[i]
        if np.isnan(p):
            adj = np.nan
        else:
            adj = min(prev, p * n / (i + 1))
            prev = adj
        adjusted[order[i]] = adj
    return adjusted


def _bayesian_prob_improvement(diff: np.ndarray, n_samples: int, seed: int) -> float:
    if np.allclose(diff, 0.0):
        return 0.5
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(len(diff)))
        if float(np.dot(w, diff)) > 0:
            wins += 1
    return float(wins / n_samples)


def run_regression_inference(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    regression_target_results: pd.DataFrame,
    cfg: InferenceBundleConfig,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_spec = spec_by_name(model_specs, cfg.baseline_regression_model)
    rng = np.random.default_rng(cfg.random_seed)
    for target in y_reg.columns:
        best = regression_target_results[regression_target_results["target"] == target].sort_values("r2", ascending=False).iloc[0]
        fs = feature_set_by_name(feature_sets, str(best["feature_set"]))
        best_spec = spec_by_name(model_specs, str(best["model"]))
        y = y_reg[target].to_numpy(dtype=float)
        y_true, y_pred = loso_regression_trace(x_user, y, fs, best_spec, cfg.random_seed, tuning_scoring, inner_cv_max_splits, inner_cv_shuffle, inner_cv_seed)
        _, y_pred_base = loso_regression_trace(x_user, y, fs, baseline_spec, cfg.random_seed, tuning_scoring, inner_cv_max_splits, inner_cv_shuffle, inner_cv_seed)
        abs_err = np.abs(y_true - y_pred)
        abs_err_base = np.abs(y_true - y_pred_base)
        diff = abs_err_base - abs_err
        if np.allclose(diff, 0.0):
            p_wil = 1.0
        else:
            try:
                p_wil = float(wilcoxon(abs_err_base, abs_err, alternative="greater").pvalue)
            except ValueError:
                p_wil = float(binomtest(np.sum(diff > 0), np.sum(diff != 0), p=0.5, alternative="greater").pvalue) if np.sum(diff != 0) > 0 else float("nan")

        if np.isnan(p_wil):
            p_wil = float(binomtest(np.sum(diff > 0), np.sum(diff != 0), p=0.5, alternative="greater").pvalue) if np.sum(diff != 0) > 0 else float("nan")

        boot_idx = rng.integers(0, len(y_true), size=(cfg.bootstrap_iterations, len(y_true)))
        r2_boot = np.array([r2_score(y_true[idx], y_pred[idx]) for idx in boot_idx], dtype=float)
        rmse_boot = np.array([np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])) for idx in boot_idx], dtype=float)
        mae_boot = np.array([mean_absolute_error(y_true[idx], y_pred[idx]) for idx in boot_idx], dtype=float)
        rows.append(
            {
                "target": target,
                "model": str(best["model"]),
                "feature_set": str(best["feature_set"]),
                "baseline_model": baseline_spec.name,
                "r2_observed": float(r2_score(y_true, y_pred)),
                "r2_ci_low": _bootstrap_ci(r2_boot)[0],
                "r2_ci_high": _bootstrap_ci(r2_boot)[1],
                "rmse_observed": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "rmse_ci_low": _bootstrap_ci(rmse_boot)[0],
                "rmse_ci_high": _bootstrap_ci(rmse_boot)[1],
                "mae_observed": float(mean_absolute_error(y_true, y_pred)),
                "mae_ci_low": _bootstrap_ci(mae_boot)[0],
                "mae_ci_high": _bootstrap_ci(mae_boot)[1],
                "paired_p_value": p_wil,
                "bayes_prob_improvement": _bayesian_prob_improvement(diff, cfg.bayesian_bootstrap_samples, cfg.random_seed),
            }
        )
    out = pd.DataFrame(rows)
    out["paired_p_value_fdr"] = _bh_adjust(out["paired_p_value"].to_numpy(dtype=float))
    out["paired_significant_fdr"] = out["paired_p_value_fdr"] < cfg.fdr_alpha
    return out.sort_values("paired_p_value_fdr", na_position="last").reset_index(drop=True)


def run_classification_inference(
    x_user: pd.DataFrame,
    y_cls: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    classification_results: pd.DataFrame,
    cfg: InferenceBundleConfig,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_spec = spec_by_name(model_specs, cfg.baseline_classification_model)
    rng = np.random.default_rng(cfg.random_seed)
    if "status" in classification_results.columns:
        valid = classification_results[classification_results["status"] == "ok"]
    else:
        valid = classification_results
    for target in y_cls.columns:
        target_rows = valid[valid["target"] == target]
        if target_rows.empty:
            continue
        best = target_rows.sort_values("auc", ascending=False).iloc[0]
        fs = feature_set_by_name(feature_sets, str(best["feature_set"]))
        best_spec = spec_by_name(model_specs, str(best["model"]))
        threshold = float(best["threshold"])
        y_bin = (y_cls[target].to_numpy(dtype=float) >= threshold).astype(int)
        y_true, y_pred, _, auc = loso_classification_trace(x_user, y_bin, fs, best_spec, cfg.random_seed, tuning_scoring, inner_cv_max_splits, inner_cv_shuffle, inner_cv_seed)
        _, y_pred_base, _, _ = loso_classification_trace(x_user, y_bin, fs, baseline_spec, cfg.random_seed, tuning_scoring, inner_cv_max_splits, inner_cv_shuffle, inner_cv_seed)
        best_correct = (y_pred == y_true).astype(int)
        base_correct = (y_pred_base == y_true).astype(int)
        b = int(np.sum((best_correct == 1) & (base_correct == 0)))
        c = int(np.sum((best_correct == 0) & (base_correct == 1)))
        p_mcnemar = float(binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue) if (b + c) > 0 else 1.0
        diff = best_correct - base_correct
        boot_idx = rng.integers(0, len(y_true), size=(cfg.bootstrap_iterations, len(y_true)))
        auc_boot_vals: list[float] = []
        for idx in boot_idx:
            y_b = y_true[idx]
            if len(np.unique(y_b)) < 2:
                continue
            auc_boot_vals.append(float(roc_auc_score(y_b, y_pred[idx])))
        auc_boot = np.asarray(auc_boot_vals if auc_boot_vals else [np.nan], dtype=float)
        acc_boot = np.array([accuracy_score(y_true[idx], y_pred[idx]) for idx in boot_idx], dtype=float)
        rows.append(
            {
                "target": target,
                "model": str(best["model"]),
                "feature_set": str(best["feature_set"]),
                "baseline_model": baseline_spec.name,
                "auc_observed": float(auc),
                "auc_ci_low": _bootstrap_ci(auc_boot)[0],
                "auc_ci_high": _bootstrap_ci(auc_boot)[1],
                "accuracy_observed": float(accuracy_score(y_true, y_pred)),
                "accuracy_ci_low": _bootstrap_ci(acc_boot)[0],
                "accuracy_ci_high": _bootstrap_ci(acc_boot)[1],
                "paired_p_value": p_mcnemar,
                "bayes_prob_improvement": _bayesian_prob_improvement(diff.astype(float), cfg.bayesian_bootstrap_samples, cfg.random_seed),
            }
        )
    out = pd.DataFrame(rows)
    out["paired_p_value_fdr"] = _bh_adjust(out["paired_p_value"].to_numpy(dtype=float))
    out["paired_significant_fdr"] = out["paired_p_value_fdr"] < cfg.fdr_alpha
    return out.sort_values("paired_p_value_fdr", na_position="last").reset_index(drop=True)
