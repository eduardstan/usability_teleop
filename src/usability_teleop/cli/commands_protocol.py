"""Unified protocol CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from usability_teleop.cli.common import DataValidationError, prepare_aligned_inputs, resolve_experiment_config
from usability_teleop.data.targets import prepare_targets
from usability_teleop.evaluation.correlation import CorrelationConfig, run_correlation_analysis
from usability_teleop.protocol.estimation import run_estimation_lane
from usability_teleop.protocol.explainability import run_final_explainability
from usability_teleop.protocol.final_models import fit_final_models


def cmd_run_estimation(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-estimation FAILED: %s", exc)
        return 1
    outputs = run_estimation_lane(
        x_user=x_user,
        y_reg=prepare_targets(bundle.questionnaire, "regression"),
        y_cls=prepare_targets(bundle.questionnaire, "classification"),
        seed=args.seed,
        max_models=args.max_models,
        max_feature_sets=args.max_feature_sets,
        regression_scoring=exp.tuning.regression_scoring,
        classification_scoring=exp.tuning.classification_scoring,
        regression_inner_max_splits=exp.cv.regression_inner_max_splits,
        classification_inner_max_splits=exp.cv.classification_inner_max_splits,
        inner_shuffle=exp.cv.inner_shuffle,
        inner_seed=exp.cv.inner_random_seed,
        top_k_per_axis=args.top_k_per_axis,
        class_balance=args.class_balance,
        logger=logger,
    )
    outputs.regression.to_csv(tables_dir / "estimation_regression.csv", index=False)
    outputs.classification.to_csv(tables_dir / "estimation_classification.csv", index=False)
    outputs.best_configs.to_csv(tables_dir / "estimation_best_configs.csv", index=False)
    logger.info("estimation tables written to %s", tables_dir)
    return 0


def cmd_fit_final_models(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    best_path = tables_dir / "estimation_best_configs.csv"
    if not best_path.exists():
        logger.error("Missing %s", best_path)
        return 1
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("fit-final-models FAILED: %s", exc)
        return 1
    final_df = fit_final_models(
        x_user=x_user,
        y_reg=prepare_targets(bundle.questionnaire, "regression"),
        y_cls=prepare_targets(bundle.questionnaire, "classification"),
        estimation_best=pd.read_csv(best_path),
        seed=args.seed,
        regression_scoring=exp.tuning.regression_scoring,
        classification_scoring=exp.tuning.classification_scoring,
        regression_inner_max_splits=exp.cv.regression_inner_max_splits,
        classification_inner_max_splits=exp.cv.classification_inner_max_splits,
        inner_shuffle=exp.cv.inner_shuffle,
        inner_seed=exp.cv.inner_random_seed,
        top_k_per_axis=args.top_k_per_axis,
        class_balance=args.class_balance,
        logger=logger,
    )
    final_df.to_csv(tables_dir / "final_models.csv", index=False)
    logger.info("final models table written to %s", tables_dir / "final_models.csv")
    return 0


def cmd_run_final_explainability(args: argparse.Namespace, logger: object) -> int:
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    final_path = tables_dir / "final_models.csv"
    if not final_path.exists():
        logger.error("Missing %s", final_path)
        return 1
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-final-explainability FAILED: %s", exc)
        return 1
    shap_df = run_final_explainability(
        x_user=x_user,
        y_reg=prepare_targets(bundle.questionnaire, "regression"),
        final_models=pd.read_csv(final_path),
        figure_dir=figures_dir,
        max_targets=args.max_targets,
        seed=args.seed,
    )
    shap_df.to_csv(tables_dir / "final_explainability_shap.csv", index=False)
    logger.info("final explainability artifacts written to %s and %s", tables_dir, figures_dir)
    return 0


def cmd_run_paper_pipeline(args: argparse.Namespace, logger: object) -> int:
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-paper-pipeline FAILED: %s", exc)
        return 1
    y_corr = prepare_targets(bundle.questionnaire, stage="correlation")
    corr_df = run_correlation_analysis(x_user, y_corr, CorrelationConfig(alpha=args.alpha, effect_threshold=args.effect_threshold))
    corr_df.to_csv(tables_dir / "correlation_results.csv", index=False)
    rc = cmd_run_estimation(args, logger)
    if rc != 0:
        return rc
    rc = cmd_fit_final_models(args, logger)
    if rc != 0:
        return rc
    rc = cmd_run_final_explainability(args, logger)
    if rc != 0:
        return rc
    logger.info("paper pipeline completed | tables=%s figures=%s", tables_dir, figures_dir)
    return 0
