"""Dataset contracts and canonical column naming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FEATURE_NAMES: list[str] = [
    "min",
    "max",
    "mean",
    "median",
    "std",
    "variance",
    "4th_moment",
    "5th_moment",
    "skewness",
    "kurtosis",
    "rms",
    "iqr",
    "total_sum",
    "range",
    "entropy",
    "std_peaks",
    "spectral_power",
    "spectral_mean",
    "spectral_median",
]

SIGNAL_NAMES: list[str] = [
    "upperarm_quat.x",
    "upperarm_quat.y",
    "upperarm_quat.z",
    "upperarm_quat.w",
    "forearm_quat.x",
    "forearm_quat.y",
    "forearm_quat.z",
    "forearm_quat.w",
    "hand_quat.x",
    "hand_quat.y",
    "hand_quat.z",
    "hand_quat.w",
    "ee_cart.x",
    "ee_cart.y",
    "ee_cart.z",
    "ee_quat.x",
    "ee_quat.y",
    "ee_quat.z",
    "ee_quat.w",
]

FEATURE_COLUMN_NAMES: list[str] = [f"{feat}_{sig}" for sig in SIGNAL_NAMES for feat in FEATURE_NAMES]

LABEL_COLUMNS: list[str] = ["task_id", "user_id", "rep_id"]

QUESTIONNAIRE_COLUMNS: list[str] = [
    "timestamp",
    "gender",
    "age_group",
    "experience_level",
    "physical_demand",
    "mental_demand",
    "temporal_demand",
    "effort",
    "frustration",
    "usability",
    "responsiveness",
    "realistic",
    "intuitive",
    "performance",
    "comments",
]

QUESTIONNAIRE_TARGET_COLUMNS: list[str] = [
    "physical_demand",
    "mental_demand",
    "temporal_demand",
    "effort",
    "frustration",
    "usability",
    "responsiveness",
    "realistic",
    "intuitive",
    "performance",
]

LIKERT_MAPPING: dict[str, int] = {
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5,
}

TIMES_COLUMNS: list[str] = [
    "user_id",
    "task 1 rip 1",
    "task 1 rip 2",
    "task 1 rip 3",
    "task 2 rip 1",
    "task 2 rip 2",
    "task 2 rip 3",
    "task 3 rip 1",
    "task 3 rip 2",
    "task 3 rip 3",
    "media",
]


@dataclass(frozen=True)
class DataPaths:
    """Dataset paths for the raw-data ingestion process."""

    raw_features: Path
    labels: Path
    questionnaire: Path
    times: Path


@dataclass(frozen=True)
class DatasetContract:
    """Expected shape/count constraints for base datasets."""

    n_feature_columns: int = len(FEATURE_COLUMN_NAMES)
    n_label_columns: int = len(LABEL_COLUMNS)
    n_questionnaire_columns: int = len(QUESTIONNAIRE_COLUMNS)
    n_times_columns: int = len(TIMES_COLUMNS)
    n_tasks: int = 3
    n_repetitions: int = 3


DEFAULT_CONTRACT = DatasetContract()
