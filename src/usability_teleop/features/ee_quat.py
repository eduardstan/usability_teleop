"""Deterministic ee_quat feature-set generation and extraction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import pandas as pd

from usability_teleop.data.contracts import FEATURE_NAMES, LABEL_COLUMNS

EE_AXIS_ORDER: tuple[str, ...] = ("x", "y", "z", "w")


@dataclass(frozen=True)
class FeatureSetSpec:
    """Specification for an ee_quat feature set."""

    name: str
    axes: tuple[str, ...]
    mode: Literal["subset", "average"]


def generate_ee_quat_feature_sets(include_average: bool = True) -> list[FeatureSetSpec]:
    """Generate deterministic non-empty axis subsets plus optional average variant."""
    specs: list[FeatureSetSpec] = []
    for size in range(1, len(EE_AXIS_ORDER) + 1):
        for axes in combinations(EE_AXIS_ORDER, size):
            specs.append(FeatureSetSpec(name="+".join(axes), axes=axes, mode="subset"))

    if include_average:
        specs.append(FeatureSetSpec(name="avg", axes=EE_AXIS_ORDER, mode="average"))

    return specs


def select_ee_quat_columns(features_df: pd.DataFrame, axes: tuple[str, ...]) -> list[str]:
    """Return columns matching ee_quat axes in deterministic source-column order."""
    selected: list[str] = []
    for column in features_df.columns:
        if any(f"_ee_quat.{axis}" in column for axis in axes):
            selected.append(column)
    return selected


def build_feature_set(features_df: pd.DataFrame, spec: FeatureSetSpec) -> pd.DataFrame:
    """Materialize a feature matrix for one ee_quat feature-set specification."""
    if spec.mode == "subset":
        cols = select_ee_quat_columns(features_df, spec.axes)
        if not cols:
            raise ValueError(f"No ee_quat columns found for axes={spec.axes}")
        return features_df[cols].copy()

    # Axis-average variant: average x/y/z/w for each base feature.
    avg_df = pd.DataFrame(index=features_df.index)
    for feature_name in FEATURE_NAMES:
        source_cols = [f"{feature_name}_ee_quat.{axis}" for axis in EE_AXIS_ORDER]
        missing = [col for col in source_cols if col not in features_df.columns]
        if missing:
            raise ValueError(f"Missing columns for average feature set: {missing}")
        avg_df[f"{feature_name}_ee_quat.avg"] = features_df[source_cols].mean(axis=1)

    return avg_df


def aggregate_user_level_features(raw_features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sample-level features to one row per user by arithmetic mean."""
    if list(labels.columns) != LABEL_COLUMNS:
        raise ValueError(f"labels must use columns {LABEL_COLUMNS}")

    combined = pd.concat([labels, raw_features], axis=1)
    user_level = combined.groupby("user_id", sort=True).mean(numeric_only=True)
    return user_level.drop(columns=["task_id", "rep_id"])
