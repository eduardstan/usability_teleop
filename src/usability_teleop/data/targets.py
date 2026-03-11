"""Target encoding and stage-specific inversion policy."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from usability_teleop.data.contracts import LIKERT_MAPPING, QUESTIONNAIRE_TARGET_COLUMNS

TargetStage = Literal["correlation", "regression", "classification"]
NEGATIVE_TARGETS: set[str] = {
    "physical_demand",
    "mental_demand",
    "temporal_demand",
    "effort",
    "frustration",
}


def _encode_likert_value(value: object) -> int:
    key = str(value).strip().lower()
    if key not in LIKERT_MAPPING:
        raise ValueError(f"Unmapped Likert value: {value!r}")
    return LIKERT_MAPPING[key]


def prepare_targets(questionnaire_df: pd.DataFrame, stage: TargetStage) -> pd.DataFrame:
    """Encode questionnaire targets with stage-aware inversion behavior."""
    if "user_id" not in questionnaire_df.columns:
        raise ValueError("questionnaire_df must include 'user_id'")

    missing = [col for col in QUESTIONNAIRE_TARGET_COLUMNS if col not in questionnaire_df.columns]
    if missing:
        raise ValueError(f"Missing questionnaire target columns: {missing}")

    encoded = questionnaire_df[["user_id", *QUESTIONNAIRE_TARGET_COLUMNS]].copy()
    for col in QUESTIONNAIRE_TARGET_COLUMNS:
        encoded[col] = [_encode_likert_value(value) for value in questionnaire_df[col].tolist()]

    encoded["user_id"] = questionnaire_df["user_id"].astype(int)
    encoded = encoded.set_index("user_id")
    encoded = encoded.astype(int)

    if stage == "correlation":
        for col in sorted(NEGATIVE_TARGETS):
            encoded[col] = 6 - encoded[col]

    if stage not in {"correlation", "regression", "classification"}:
        raise ValueError(f"Unsupported stage: {stage}")

    return encoded
