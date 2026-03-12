"""Nested-LOSO estimation for classification track."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.cv import classification_inner_cv, fit_with_tuning, loso_indices
from usability_teleop.modeling.registry import ModelSpec, build_estimator
from usability_teleop.protocol.selection import SelectionConfig, pack_fold_feature_counts, select_train_test_features

ClassBalanceMode = Literal["none", "smote"]


def run_classification_estimation(
    x_user: pd.DataFrame,
    y_cls: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    random_seed: int,
    tuning_scoring: str,
    inner_cv_max_splits: int,
    inner_cv_shuffle: bool,
    inner_cv_seed: int,
    class_balance: ClassBalanceMode,
    selection_cfg: SelectionConfig,
    logger: object | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in y_cls.columns:
        y_cont = y_cls[target].to_numpy(dtype=float)
        threshold = float(np.median(y_cont))
        y_bin = (y_cont >= threshold).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        for fs in feature_sets:
            x_fs = build_feature_set(x_user, fs)
            for spec in model_specs:
                y_true = np.zeros_like(y_bin)
                y_pred = np.zeros_like(y_bin)
                y_score = np.zeros_like(y_cont, dtype=float)
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
                    y_train = y_bin[train_idx]
                    x_train_np, y_train_np = _rebalance_binary_train(
                        x_train.to_numpy(dtype=float),
                        y_train,
                        class_balance,
                        random_seed + int(test_idx[0]),
                    )
                    scaler = StandardScaler()
                    x_train_s = scaler.fit_transform(x_train_np)
                    x_test_s = scaler.transform(x_test.to_numpy(dtype=float))
                    model, best_params = fit_with_tuning(
                        build_estimator(spec, random_seed),
                        spec.param_grid,
                        x_train_s,
                        y_train_np,
                        scoring=tuning_scoring,
                        cv=classification_inner_cv(
                            y_train_np,
                            max_splits=inner_cv_max_splits,
                            shuffle=inner_cv_shuffle,
                            random_seed=inner_cv_seed,
                        ),
                    )
                    fold_params.append(best_params)
                    y_true[test_idx] = y_bin[test_idx]
                    y_pred[test_idx] = model.predict(x_test_s)
                    y_score[test_idx] = _score_from_estimator(model, x_test_s)
                rows.append(
                    {
                        "track": "classification",
                        "target": target,
                        "feature_set": fs.name,
                        "model": spec.name,
                        "threshold": threshold,
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
                        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
                        "fold_best_params": json.dumps(fold_params, sort_keys=True),
                        "fold_feature_counts": pack_fold_feature_counts(fold_counts),
                    }
                )
                if logger is not None:
                    logger.info(
                        "estimation classification target=%s feature_set=%s model=%s done",
                        target,
                        fs.name,
                        spec.name,
                    )
    return pd.DataFrame(rows).sort_values(["target", "auc"], ascending=[True, False]).reset_index(drop=True)


def _score_from_estimator(estimator: Any, x_test: np.ndarray) -> float:
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(x_test)
        return float(p[:, 1][0]) if p.ndim == 2 and p.shape[1] > 1 else float(p[0])
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(x_test)
        return float(s[0]) if isinstance(s, np.ndarray) else float(s)
    return float(estimator.predict(x_test)[0])


def _rebalance_binary_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: ClassBalanceMode,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return x_train, y_train
    if method != "smote":
        raise ValueError(f"Unsupported class balance mode: {method}")
    counts = Counter(y_train.tolist())
    if len(counts) < 2:
        return x_train, y_train
    keys = np.asarray(sorted(counts.keys()), dtype=int)
    c0 = counts[int(keys[0])]
    c1 = counts[int(keys[1])]
    if c0 == c1:
        return x_train, y_train
    rng = np.random.default_rng(random_seed)
    idx0 = np.where(y_train == int(keys[0]))[0]
    idx1 = np.where(y_train == int(keys[1]))[0]
    minor_idx, major_idx = (idx0, idx1) if c0 < c1 else (idx1, idx0)
    needed = len(major_idx) - len(minor_idx)
    if needed <= 0:
        return x_train, y_train
    x_minor = x_train[minor_idx]
    if len(minor_idx) < 2:
        extra = rng.choice(minor_idx, size=needed, replace=True)
        keep = np.concatenate([major_idx, minor_idx, extra])
        rng.shuffle(keep)
        return x_train[keep], y_train[keep]
    k_neighbors = min(5, len(minor_idx) - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(x_minor)
    neigh = nn.kneighbors(x_minor, return_distance=False)
    synth: list[np.ndarray] = []
    for _ in range(needed):
        i = int(rng.integers(0, len(minor_idx)))
        j = int(neigh[i][1:][int(rng.integers(0, k_neighbors))])
        lam = float(rng.random())
        synth.append(x_minor[i] + lam * (x_minor[j] - x_minor[i]))
    x_synth = np.vstack(synth)
    y_synth = np.full(len(synth), y_train[minor_idx[0]], dtype=y_train.dtype)
    return np.vstack([x_train, x_synth]), np.concatenate([y_train, y_synth])
