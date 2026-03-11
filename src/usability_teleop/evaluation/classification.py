"""Binary classification benchmark workflow (RQ3)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.cv import classification_inner_cv, fit_with_tuning, loso_indices
from usability_teleop.modeling.registry import ModelSpec, build_estimator


def _auc_from_estimator(estimator: Any, x_test: np.ndarray) -> float:
    if hasattr(estimator, "predict_proba"):
        prob = estimator.predict_proba(x_test)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return float(prob[:, 1][0])
        return float(prob[0])
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(x_test)
        if isinstance(score, np.ndarray):
            return float(score[0])
        return float(score)
    pred = estimator.predict(x_test)
    return float(pred[0])


def run_classification_benchmark(
    x_user: pd.DataFrame,
    y_cls: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    random_seed: int = 42,
    max_feature_sets: int | None = None,
    tuning_scoring: str = "roc_auc",
    inner_cv_max_splits: int = 3,
    inner_cv_shuffle: bool = True,
    inner_cv_seed: int = 42,
) -> pd.DataFrame:
    """Run median-split binary classification LOSO benchmark."""
    selected_sets = feature_sets[: max_feature_sets or len(feature_sets)]
    rows: list[dict[str, object]] = []

    for target in y_cls.columns:
        y_cont = y_cls[target].to_numpy(dtype=float)
        threshold = float(np.median(y_cont))
        y_bin = (y_cont >= threshold).astype(int)

        if len(np.unique(y_bin)) < 2:
            rows.append(
                {
                    "target": target,
                    "feature_set": "-",
                    "model": "-",
                    "threshold": threshold,
                    "status": "skipped_single_class",
                }
            )
            continue

        for fs in selected_sets:
            x_fs = build_feature_set(x_user, fs)

            for spec in model_specs:
                y_true_all = np.zeros_like(y_bin)
                y_pred_all = np.zeros_like(y_bin)
                y_score_all = np.zeros_like(y_cont, dtype=float)
                fold_params: list[dict[str, Any]] = []

                for train_idx, test_idx in loso_indices(len(x_fs)):
                    x_train = x_fs.iloc[train_idx].to_numpy(dtype=float)
                    x_test = x_fs.iloc[test_idx].to_numpy(dtype=float)
                    y_train = y_bin[train_idx]

                    scaler = StandardScaler()
                    x_train_s = scaler.fit_transform(x_train)
                    x_test_s = scaler.transform(x_test)

                    base = build_estimator(spec, random_seed)
                    best_model, best_params = fit_with_tuning(
                        base,
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
                    fold_params.append(best_params)

                    y_true_all[test_idx] = y_bin[test_idx]
                    y_pred_all[test_idx] = best_model.predict(x_test_s)
                    y_score_all[test_idx] = _auc_from_estimator(best_model, x_test_s)

                if len(np.unique(y_true_all)) > 1:
                    auc = float(roc_auc_score(y_true_all, y_score_all))
                else:
                    auc = float("nan")

                rows.append(
                    {
                        "target": target,
                        "feature_set": fs.name,
                        "model": spec.name,
                        "threshold": threshold,
                        "status": "ok",
                        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
                        "f1_macro": float(f1_score(y_true_all, y_pred_all, average="macro")),
                        "auc": auc,
                        "best_params_last_fold": json.dumps(
                            fold_params[-1] if fold_params else {}, sort_keys=True
                        ),
                    }
                )

    return pd.DataFrame(rows).sort_values(by=["target", "auc"], ascending=[True, False]).reset_index(
        drop=True
    )
