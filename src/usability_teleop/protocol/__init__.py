"""Unified protocol lanes for estimation and final-model workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_estimation_lane", "fit_final_models", "run_final_explainability"]


def __getattr__(name: str) -> Any:
    if name == "run_estimation_lane":
        return import_module("usability_teleop.protocol.estimation").run_estimation_lane
    if name == "fit_final_models":
        return import_module("usability_teleop.protocol.final_models").fit_final_models
    if name == "run_final_explainability":
        return import_module("usability_teleop.protocol.explainability").run_final_explainability
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
