"""Validation helpers for unified protocol artifact tables."""

from __future__ import annotations

import pandas as pd


def validate_estimation_best_configs(df: pd.DataFrame) -> None:
    required = {
        "track",
        "target",
        "feature_set",
        "model",
        "selection_metric",
        "selection_score",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "estimation_best_configs is missing required columns: "
            + ", ".join(missing)
        )

    valid_tracks = {"regression", "classification"}
    bad_tracks = sorted(set(df["track"].astype(str)) - valid_tracks)
    if bad_tracks:
        raise ValueError(
            "estimation_best_configs contains invalid track values: "
            + ", ".join(bad_tracks)
        )


def validate_final_models_table(df: pd.DataFrame) -> None:
    required = {
        "track",
        "target",
        "feature_set",
        "model",
        "final_params",
        "selected_features",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "final_models is missing required columns: "
            + ", ".join(missing)
        )

    valid_tracks = {"regression", "classification"}
    bad_tracks = sorted(set(df["track"].astype(str)) - valid_tracks)
    if bad_tracks:
        raise ValueError(
            "final_models contains invalid track values: " + ", ".join(bad_tracks)
        )
