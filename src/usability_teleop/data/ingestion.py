"""I/O and ingestion pipeline for canonical raw datasets."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from usability_teleop.data.contracts import (
    FEATURE_COLUMN_NAMES,
    LABEL_COLUMNS,
    QUESTIONNAIRE_COLUMNS,
    DataPaths,
)
from usability_teleop.data.validation import (
    ValidationSummary,
    ensure_path_exists,
    validate_alignment,
    validate_labels,
    validate_questionnaire,
    validate_raw_features,
    validate_times,
)


@dataclass(frozen=True)
class DatasetBundle:
    """Validated in-memory datasets for downstream pipeline stages."""

    raw_features: pd.DataFrame
    labels: pd.DataFrame
    questionnaire: pd.DataFrame
    times: pd.DataFrame
    summary: ValidationSummary


def raw_data_paths(base_dir: Path) -> DataPaths:
    """Construct expected raw input file paths from a base directory."""
    return DataPaths(
        raw_features=base_dir / "raw_features_full.csv",
        labels=base_dir / "labels_full.csv",
        questionnaire=base_dir / "User_risposte.xlsx",
        times=base_dir / "tempi_media.csv",
    )


def load_and_validate(paths: DataPaths) -> DatasetBundle:
    """Load datasets and enforce schema/alignment contracts."""
    for path in [paths.raw_features, paths.labels, paths.questionnaire, paths.times]:
        ensure_path_exists(path)

    raw_features = pd.read_csv(paths.raw_features, header=None)
    raw_features = raw_features.set_axis(FEATURE_COLUMN_NAMES, axis=1, copy=False)

    labels = pd.read_csv(paths.labels, header=None, names=LABEL_COLUMNS)
    labels = labels.astype(int)

    questionnaire = pd.read_excel(paths.questionnaire)
    questionnaire = questionnaire.set_axis(QUESTIONNAIRE_COLUMNS, axis=1, copy=False)

    times = pd.read_csv(paths.times)

    validate_raw_features(raw_features)
    validate_labels(labels)
    validate_questionnaire(questionnaire)
    questionnaire["user_id"] = range(1, len(questionnaire) + 1)
    cleaned_times = validate_times(times)
    summary = validate_alignment(raw_features, labels, questionnaire, cleaned_times)

    return DatasetBundle(
        raw_features=raw_features,
        labels=labels,
        questionnaire=questionnaire,
        times=cleaned_times,
        summary=summary,
    )


def copy_raw_inputs(paths: DataPaths, destination: Path) -> None:
    """Copy source inputs into canonical raw-data directory with stable names."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.raw_features, destination / "raw_features_full.csv")
    shutil.copy2(paths.labels, destination / "labels_full.csv")
    shutil.copy2(paths.questionnaire, destination / "User_risposte.xlsx")
    shutil.copy2(paths.times, destination / "tempi_media.csv")
