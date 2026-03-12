"""Cross-validation and tuning helpers for LOSO workflows."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold


def loso_indices(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic LOSO index splits."""
    loo = LeaveOneOut()
    return list(loo.split(np.arange(n_samples)))


def regression_inner_cv(
    n_train: int,
    max_splits: int = 3,
    shuffle: bool = True,
    random_seed: int = 42,
) -> KFold | None:
    """Choose inner KFold for regression tuning."""
    if n_train < 4:
        return None
    # R2 requires at least two samples in each validation fold.
    n_splits = min(max_splits, n_train // 2)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed if shuffle else None)


def classification_inner_cv(
    y_train: np.ndarray,
    max_splits: int = 3,
    shuffle: bool = True,
    random_seed: int = 42,
) -> StratifiedKFold | None:
    """Choose inner StratifiedKFold if class support is sufficient."""
    counts = Counter(y_train.tolist())
    if len(counts) < 2:
        return None
    min_count = min(counts.values())
    if min_count < 2:
        return None
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        return None
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_seed if shuffle else None,
    )


def fit_with_tuning(
    estimator: Any,
    param_grid: dict[str, list[Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    scoring: str,
    cv: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit estimator directly or with grid search depending on CV availability."""
    if cv is None or not param_grid:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings(
                "ignore",
                message="Only one sample available. You may want to reshape your data array",
                category=UserWarning,
            )
            estimator.fit(x_train, y_train)
        return estimator, {}

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message="Only one sample available. You may want to reshape your data array",
            category=UserWarning,
        )
        search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_)
