from pathlib import Path

import pandas as pd
import pytest

from usability_teleop.data.contracts import QUESTIONNAIRE_COLUMNS, TIMES_COLUMNS
from usability_teleop.data.ingestion import load_and_validate, raw_data_paths
from usability_teleop.data.validation import (
    DataValidationError,
    validate_questionnaire,
    validate_times,
)


def test_load_and_validate_raw_data_smoke() -> None:
    bundle = load_and_validate(raw_data_paths(Path("data/raw")))
    assert bundle.summary.n_users == 16
    assert bundle.summary.n_samples == 144
    assert bundle.summary.n_features == 361


def test_validate_questionnaire_rejects_unmapped_likert() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "gender": ["Male"],
            "age_group": ["18-29"],
            "experience_level": ["No level"],
            "physical_demand": ["agree"],
            "mental_demand": ["neutral"],
            "temporal_demand": ["disagree"],
            "effort": ["agree"],
            "frustration": ["agree"],
            "usability": ["agree"],
            "responsiveness": ["agree"],
            "realistic": ["agree"],
            "intuitive": ["agree"],
            "performance": ["not a valid likert"],
            "comments": [None],
        }
    )
    assert list(df.columns) == QUESTIONNAIRE_COLUMNS

    with pytest.raises(DataValidationError):
        validate_questionnaire(df)


def test_validate_times_rejects_wrong_format() -> None:
    df = pd.DataFrame(
        [
            [1, "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00"],
            [2, "invalid", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00", "00:10.00"],
        ],
        columns=TIMES_COLUMNS,
    )

    with pytest.raises(DataValidationError):
        validate_times(df)
