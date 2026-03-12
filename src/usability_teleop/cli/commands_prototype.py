"""CLI command for fast incremental prototyping experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from usability_teleop.analysis.incremental import run_incremental_prototype
from usability_teleop.analysis.preprocessing import build_target_distribution_table
from usability_teleop.cli.common import (
    DataValidationError,
    prepare_aligned_inputs,
    resolve_experiment_config,
)
from usability_teleop.data.targets import prepare_targets


def cmd_run_incremental_prototype(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    table_dir = Path(args.tables_dir).resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-incremental-prototype FAILED: %s", exc)
        return 1

    y_reg = prepare_targets(bundle.questionnaire, stage="regression")
    y_cls = prepare_targets(bundle.questionnaire, stage="classification")
    target_dist = build_target_distribution_table(y_reg, y_cls)
    target_dist.to_csv(table_dir / "prototype_target_distribution.csv", index=False)

    outputs = run_incremental_prototype(
        x_base=x_user,
        y_reg=y_reg,
        y_cls=y_cls,
        max_models=args.max_models,
        max_feature_sets=args.max_feature_sets,
        top_k_per_axis=args.top_k_per_axis,
        class_balance=args.class_balance,
        seed=args.seed,
        workers=args.workers,
        tuning_regression_scoring=exp.tuning.regression_scoring,
        tuning_classification_scoring=exp.tuning.classification_scoring,
        inner_regression_splits=exp.cv.regression_inner_max_splits,
        inner_classification_splits=exp.cv.classification_inner_max_splits,
        inner_shuffle=exp.cv.inner_shuffle,
        inner_seed=exp.cv.inner_random_seed,
        logger=logger,
    )
    outputs.summary.to_csv(table_dir / "prototype_incremental_summary.csv", index=False)
    outputs.breakdown.to_csv(table_dir / "prototype_incremental_breakdown.csv", index=False)
    outputs.feature_filter_summary.to_csv(table_dir / "prototype_feature_filter_summary.csv", index=False)

    logger.info("prototype outputs written to %s", table_dir)
    logger.info("summary rows=%s breakdown rows=%s", len(outputs.summary), len(outputs.breakdown))
    return 0
