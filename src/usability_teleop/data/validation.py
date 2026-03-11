"""Validation utilities for dataset contracts and alignment checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from usability_teleop.data.contracts import (
    DEFAULT_CONTRACT,
    LABEL_COLUMNS,
    LIKERT_MAPPING,
    QUESTIONNAIRE_TARGET_COLUMNS,
    TIMES_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when dataset files violate required contracts."""


@dataclass(frozen=True)
class ValidationSummary:
    """Compact report for data validation CLI output."""

    n_users: int
    n_samples: int
    n_features: int


_TIME_PATTERN = r"^\d{2}:\d{2}\.\d{2}$"


def _fail(message: str) -> None:
    raise DataValidationError(message)


def validate_raw_features(df: pd.DataFrame) -> None:
    if df.shape[1] != DEFAULT_CONTRACT.n_feature_columns:
        _fail(
            f"raw_features must have {DEFAULT_CONTRACT.n_feature_columns} columns, got {df.shape[1]}"
        )
    if df.empty:
        _fail("raw_features is empty")
    if df.isna().any().any():
        _fail("raw_features contains missing values")


def validate_labels(df: pd.DataFrame) -> None:
    if list(df.columns) != LABEL_COLUMNS:
        _fail(f"labels columns must be {LABEL_COLUMNS}, got {list(df.columns)}")
    if df.empty:
        _fail("labels is empty")
    if df.isna().any().any():
        _fail("labels contains missing values")

    for col in LABEL_COLUMNS:
        if not pd.api.types.is_integer_dtype(df[col]):
            _fail(f"labels.{col} must be integer dtype")

    task_values = set(df["task_id"].unique().tolist())
    rep_values = set(df["rep_id"].unique().tolist())
    expected = set(range(1, DEFAULT_CONTRACT.n_tasks + 1))
    if task_values != expected:
        _fail(f"task_id values must be {sorted(expected)}, got {sorted(task_values)}")
    if rep_values != expected:
        _fail(f"rep_id values must be {sorted(expected)}, got {sorted(rep_values)}")


def validate_questionnaire(df: pd.DataFrame) -> None:
    if df.empty:
        _fail("questionnaire is empty")
    if len(df.columns) != DEFAULT_CONTRACT.n_questionnaire_columns:
        _fail(
            "questionnaire must have "
            f"{DEFAULT_CONTRACT.n_questionnaire_columns} columns, got {len(df.columns)}"
        )

    if df["timestamp"].isna().any():
        _fail("questionnaire.timestamp contains missing values")

    allowed_likert = set(LIKERT_MAPPING)
    for col in QUESTIONNAIRE_TARGET_COLUMNS:
        normalized = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )
        unknown = sorted(set(normalized) - allowed_likert)
        if unknown:
            _fail(f"questionnaire.{col} has unmapped Likert values: {unknown}")


def validate_times(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) != TIMES_COLUMNS:
        _fail(f"times columns must be {TIMES_COLUMNS}, got {list(df.columns)}")

    cleaned = df[df["user_id"].notna()].copy()
    if cleaned.empty:
        _fail("times has no participant rows after removing summary rows")

    cleaned["user_id"] = cleaned["user_id"].astype(int)

    for col in TIMES_COLUMNS[1:]:
        invalid = cleaned[col].dropna().astype(str).str.match(_TIME_PATTERN).eq(False)
        if invalid.any():
            _fail(f"times.{col} contains invalid time strings (expected mm:ss.xx)")

    return cleaned


def validate_alignment(
    raw_features: pd.DataFrame,
    labels: pd.DataFrame,
    questionnaire: pd.DataFrame,
    times: pd.DataFrame,
) -> ValidationSummary:
    if len(raw_features) != len(labels):
        _fail(
            f"row mismatch: raw_features has {len(raw_features)} rows, labels has {len(labels)} rows"
        )

    users_in_labels = sorted(labels["user_id"].unique().tolist())
    users_in_questionnaire = sorted(questionnaire["user_id"].unique().tolist())
    users_in_times = sorted(times["user_id"].unique().tolist())

    if users_in_labels != users_in_questionnaire:
        _fail(
            "user mismatch between labels and questionnaire: "
            f"{users_in_labels} vs {users_in_questionnaire}"
        )

    if users_in_labels != users_in_times:
        _fail(f"user mismatch between labels and times: {users_in_labels} vs {users_in_times}")

    expected_samples = len(users_in_labels) * DEFAULT_CONTRACT.n_tasks * DEFAULT_CONTRACT.n_repetitions
    if len(labels) != expected_samples:
        _fail(f"expected {expected_samples} samples from users*tasks*reps, got {len(labels)}")

    return ValidationSummary(
        n_users=len(users_in_labels),
        n_samples=len(labels),
        n_features=raw_features.shape[1],
    )


def ensure_path_exists(path: Path) -> None:
    if not path.exists():
        _fail(f"missing input file: {path}")
