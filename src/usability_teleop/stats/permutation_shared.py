"""Shared helpers for permutation-test workflows."""

from __future__ import annotations

import json
from typing import Any

from usability_teleop.features.ee_quat import FeatureSetSpec
from usability_teleop.modeling.registry import ModelSpec


def spec_by_name(specs: list[ModelSpec], model_name: str) -> ModelSpec:
    for spec in specs:
        if spec.name == model_name:
            return spec
    raise ValueError(f"Unknown model name: {model_name}")


def feature_set_by_name(specs: list[FeatureSetSpec], name: str) -> FeatureSetSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown feature set: {name}")


def params_from_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    return {key.replace("estimator__", ""): value for key, value in parsed.items()}


def params_from_result_row(row: Any) -> dict[str, Any]:
    """Extract estimator params from either legacy or unified result row schemas."""
    raw_last = row.get("best_params_last_fold", None)
    if raw_last is not None and str(raw_last).strip():
        return params_from_json(str(raw_last))

    raw_fold = row.get("fold_best_params", None)
    if raw_fold is None or not str(raw_fold).strip():
        return {}
    parsed = json.loads(str(raw_fold))
    if isinstance(parsed, list) and parsed:
        last = parsed[-1]
        if isinstance(last, dict):
            return {str(k).replace("estimator__", ""): v for k, v in last.items()}
    return {}
