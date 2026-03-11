"""CLI command for extended inference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from usability_teleop.cli.common import (
    DataValidationError,
    prepare_aligned_inputs,
    resolve_experiment_config,
)
from usability_teleop.data.targets import prepare_targets
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs
from usability_teleop.stats.inference import (
    InferenceBundleConfig,
    run_classification_inference,
    run_regression_inference,
)


def compute_inference_tables(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    y_cls: pd.DataFrame,
    feature_sets: list,
    reg_models: list,
    cls_models: list,
    reg_results: pd.DataFrame,
    cls_results: pd.DataFrame,
    exp: object,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inf_cfg = InferenceBundleConfig(
        baseline_regression_model=exp.inference.baseline_regression_model,
        baseline_classification_model=exp.inference.baseline_classification_model,
        bootstrap_iterations=exp.inference.bootstrap_iterations,
        bayesian_bootstrap_samples=exp.inference.bayesian_bootstrap_samples,
        paired_alpha=exp.inference.paired_alpha,
        fdr_alpha=exp.inference.fdr_alpha,
        random_seed=seed,
    )
    reg_inf = run_regression_inference(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        reg_results,
        inf_cfg,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    cls_inf = run_classification_inference(
        x_user,
        y_cls,
        feature_sets,
        cls_models,
        cls_results,
        inf_cfg,
        exp.tuning.classification_scoring,
        exp.cv.classification_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    return reg_inf, cls_inf


def cmd_run_inference(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    table_dir = Path(args.tables_dir).resolve()
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-inference FAILED: %s", exc)
        return 1
    reg_path = table_dir / "regression_target_specific_results.csv"
    cls_path = table_dir / "classification_results.csv"
    if not reg_path.exists() or not cls_path.exists():
        logger.error("Missing benchmark results in %s", table_dir)
        return 1

    reg_results = pd.read_csv(reg_path)
    cls_results = pd.read_csv(cls_path)
    y_reg = prepare_targets(bundle.questionnaire, stage="regression")
    y_cls = prepare_targets(bundle.questionnaire, stage="classification")
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    reg_models = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()
    cls_models = classification_model_specs()[: args.max_models] if args.max_models is not None else classification_model_specs()
    if args.max_feature_sets is not None:
        allowed = {fs.name for fs in feature_sets[: args.max_feature_sets]}
        reg_results = reg_results[reg_results["feature_set"].isin(allowed)]
        cls_results = cls_results[cls_results["feature_set"].isin(allowed)]
        feature_sets = feature_sets[: args.max_feature_sets]

    reg_inf, cls_inf = compute_inference_tables(
        x_user,
        y_reg,
        y_cls,
        feature_sets,
        reg_models,
        cls_models,
        reg_results,
        cls_results,
        exp,
        args.seed,
    )
    reg_out = table_dir / "inference_regression.csv"
    cls_out = table_dir / "inference_classification.csv"
    reg_inf.to_csv(reg_out, index=False)
    cls_inf.to_csv(cls_out, index=False)
    logger.info("inference regression rows=%s written to %s", len(reg_inf), reg_out)
    logger.info("inference classification rows=%s written to %s", len(cls_inf), cls_out)
    return 0
