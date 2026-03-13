"""Shared CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from usability_teleop.config.experiment import ExperimentConfig, load_experiment_config
from usability_teleop.data.ingestion import (
    DataPaths,
    DatasetBundle,
    copy_raw_inputs,
    load_and_validate,
)
from usability_teleop.data.validation import DataValidationError
from usability_teleop.evaluation.regression import build_global_vs_target_specific_r2
from usability_teleop.features.ee_quat import aggregate_user_level_features


def raw_data_paths(base_dir: Path) -> DataPaths:
    """Return canonical input paths under data/raw-like directory."""
    return DataPaths(
        raw_features=base_dir / "raw_features_full.csv",
        labels=base_dir / "labels_full.csv",
        questionnaire=base_dir / "User_risposte.xlsx",
        times=base_dir / "tempi_media.csv",
    )


def prepare_aligned_inputs(source_dir: Path) -> tuple[DatasetBundle, pd.DataFrame]:
    bundle = load_and_validate(raw_data_paths(source_dir))
    x_user = aggregate_user_level_features(bundle.raw_features, bundle.labels)
    return bundle, x_user


def resolve_experiment_config(path: str | None) -> ExperimentConfig:
    return load_experiment_config(Path(path).resolve()) if path else load_experiment_config()


def write_regression_comparison_artifacts(
    regression_global_df: pd.DataFrame,
    regression_target_specific_df: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    from usability_teleop.viz.figures import plot_global_vs_target_specific_r2

    comparison = build_global_vs_target_specific_r2(regression_global_df, regression_target_specific_df)
    out_table = table_dir / "regression_best_global_vs_target_specific.csv"
    comparison.to_csv(out_table, index=False)
    plot_global_vs_target_specific_r2(comparison, figure_dir / "figure_regression_global_vs_target_specific.png")
    return comparison


__all__ = [
    "DataValidationError",
    "copy_raw_inputs",
    "prepare_aligned_inputs",
    "raw_data_paths",
    "resolve_experiment_config",
    "write_regression_comparison_artifacts",
]
