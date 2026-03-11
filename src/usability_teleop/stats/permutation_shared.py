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
