"""Model registry loaded from external YAML configuration."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

ModelTask = Literal["regression", "classification"]


@dataclass(frozen=True)
class ModelSpec:
    """Model declaration with constructor and tuning grid."""

    name: str
    estimator_cls: type[Any]
    fixed_params: dict[str, Any]
    param_grid: dict[str, list[Any]]


def _default_models_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def _import_estimator(path: str) -> type[Any]:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    estimator_cls = getattr(module, class_name)
    if not isinstance(estimator_cls, type):
        raise TypeError(f"Estimator must be a class: {path}")
    return estimator_cls


def _parse_model_spec(raw: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=str(raw["name"]),
        estimator_cls=_import_estimator(str(raw["estimator"])),
        fixed_params=dict(raw.get("fixed_params", {})),
        param_grid={str(k): list(v) for k, v in dict(raw.get("param_grid", {})).items()},
    )


@lru_cache(maxsize=4)
def _load_specs(config_path: str, task: ModelTask) -> tuple[ModelSpec, ...]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or task not in config:
        raise ValueError(f"Missing '{task}' section in model config: {config_path}")
    raw_specs = config[task]
    if not isinstance(raw_specs, list):
        raise ValueError(f"'{task}' section must be a list in: {config_path}")
    return tuple(_parse_model_spec(item) for item in raw_specs)


def _specs_for(task: ModelTask, config_path: Path | None = None) -> list[ModelSpec]:
    path = config_path or _default_models_config_path()
    return list(_load_specs(str(path), task))


def regression_model_specs(config_path: Path | None = None) -> list[ModelSpec]:
    """Regression model specs loaded from external YAML config."""
    return _specs_for("regression", config_path=config_path)


def classification_model_specs(config_path: Path | None = None) -> list[ModelSpec]:
    """Classification model specs loaded from external YAML config."""
    return _specs_for("classification", config_path=config_path)


def build_estimator(spec: ModelSpec, random_seed: int) -> Any:
    """Instantiate estimator with fixed params and deterministic seeds."""
    kwargs: dict[str, Any] = dict(spec.fixed_params)
    try:
        params = spec.estimator_cls().get_params()
    except Exception:
        params = {}
    if "random_state" in params and "random_state" not in kwargs:
        kwargs["random_state"] = random_seed
    if spec.name == "SVC" and "probability" not in kwargs:
        kwargs["probability"] = True
    return spec.estimator_cls(**kwargs)
