"""Experiment protocol configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureSetConfig:
    include_average: bool = True


@dataclass(frozen=True)
class TuningConfig:
    regression_scoring: str = "r2"
    classification_scoring: str = "roc_auc"


@dataclass(frozen=True)
class CVConfig:
    regression_inner_max_splits: int = 3
    classification_inner_max_splits: int = 3
    inner_shuffle: bool = True
    inner_random_seed: int = 42


@dataclass(frozen=True)
class PermutationProtocolConfig:
    alpha: float = 0.05
    n_permutations_default: int = 100
    nested_default: bool = False


@dataclass(frozen=True)
class ShapConfig:
    max_targets_default: int = 5


@dataclass(frozen=True)
class InferenceConfig:
    baseline_regression_model: str = "LinearRegression"
    baseline_classification_model: str = "RidgeLikeClassifier"
    bootstrap_iterations: int = 2000
    bayesian_bootstrap_samples: int = 5000
    paired_alpha: float = 0.05
    fdr_alpha: float = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    feature_sets: FeatureSetConfig
    tuning: TuningConfig
    cv: CVConfig
    permutation: PermutationProtocolConfig
    shap: ShapConfig
    inference: InferenceConfig


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "experiment.yaml"


def _from_dict(data: dict[str, Any]) -> ExperimentConfig:
    fs = data.get("feature_sets", {})
    tuning = data.get("tuning", {})
    cv = data.get("cv", {})
    perm = data.get("permutation", {})
    shap = data.get("shap", {})
    inf = data.get("inference", {})
    return ExperimentConfig(
        feature_sets=FeatureSetConfig(include_average=bool(fs.get("include_average", True))),
        tuning=TuningConfig(
            regression_scoring=str(tuning.get("regression_scoring", "r2")),
            classification_scoring=str(tuning.get("classification_scoring", "roc_auc")),
        ),
        cv=CVConfig(
            regression_inner_max_splits=int(cv.get("regression_inner_max_splits", 3)),
            classification_inner_max_splits=int(cv.get("classification_inner_max_splits", 3)),
            inner_shuffle=bool(cv.get("inner_shuffle", True)),
            inner_random_seed=int(cv.get("inner_random_seed", 42)),
        ),
        permutation=PermutationProtocolConfig(
            alpha=float(perm.get("alpha", 0.05)),
            n_permutations_default=int(perm.get("n_permutations_default", 100)),
            nested_default=bool(perm.get("nested_default", False)),
        ),
        shap=ShapConfig(max_targets_default=int(shap.get("max_targets_default", 5))),
        inference=InferenceConfig(
            baseline_regression_model=str(inf.get("baseline_regression_model", "LinearRegression")),
            baseline_classification_model=str(inf.get("baseline_classification_model", "RidgeLikeClassifier")),
            bootstrap_iterations=int(inf.get("bootstrap_iterations", 2000)),
            bayesian_bootstrap_samples=int(inf.get("bayesian_bootstrap_samples", 5000)),
            paired_alpha=float(inf.get("paired_alpha", 0.05)),
            fdr_alpha=float(inf.get("fdr_alpha", 0.05)),
        ),
    )


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    config_path = path or _default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid experiment config format: {config_path}")
    return _from_dict(raw)
