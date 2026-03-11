"""CLI commands for regression-centric workflows (RQ2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from usability_teleop.cli.common import (
    DataValidationError,
    prepare_aligned_inputs,
    resolve_experiment_config,
    write_regression_comparison_artifacts,
)
from usability_teleop.data.targets import prepare_targets
from usability_teleop.evaluation.regression import (
    run_regression_global,
    run_regression_target_specific,
)
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import regression_model_specs
from usability_teleop.stats.inference import InferenceBundleConfig, run_regression_inference
from usability_teleop.stats.permutation import PermutationConfig, run_regression_permutation_tests
from usability_teleop.stats.shap_analysis import run_regression_shap
from usability_teleop.utils.timing import ProgressTracker, format_seconds
from usability_teleop.viz.figures import plot_permutation_summary, plot_regression_overview


def cmd_run_regression(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-regression FAILED: %s", exc)
        return 1
    y_reg = prepare_targets(bundle.questionnaire, stage="regression")
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    model_specs = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()
    logger.info(
        "running RQ2 regression with feature_sets=%s models=%s workers=%s",
        len(feature_sets[: args.max_feature_sets or len(feature_sets)]),
        len(model_specs),
        args.workers,
    )
    global_df = run_regression_global(
        x_user,
        y_reg,
        feature_sets,
        model_specs,
        args.seed,
        args.max_feature_sets,
        logger,
        args.workers,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    specific_df = run_regression_target_specific(
        x_user,
        y_reg,
        feature_sets,
        model_specs,
        args.seed,
        args.max_feature_sets,
        logger,
        args.workers,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "regression_global_results.csv"
    specific_path = output_dir / "regression_target_specific_results.csv"
    global_df.to_csv(global_path, index=False)
    specific_df.to_csv(specific_path, index=False)
    write_regression_comparison_artifacts(global_df, specific_df, output_dir, output_dir.parent / "figures")
    logger.info("global regression rows=%s written to %s", len(global_df), global_path)
    logger.info("target-specific regression rows=%s written to %s", len(specific_df), specific_path)
    return 0


def cmd_run_rq2_end2end(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    table_dir = Path(args.tables_dir).resolve()
    figure_dir = Path(args.figures_dir).resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-rq2-end2end FAILED: %s", exc)
        return 1

    y_reg = prepare_targets(bundle.questionnaire, stage="regression")
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    if args.max_feature_sets is not None:
        feature_sets = feature_sets[: args.max_feature_sets]
    reg_models = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()

    stage = ProgressTracker(total=4)
    logger.info("RQ2 end-to-end starting | users=%s feature_sets=%s models=%s workers=%s", len(x_user), len(feature_sets), len(reg_models), args.workers)
    global_df = run_regression_global(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        args.seed,
        None,
        logger,
        args.workers,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    specific_df = run_regression_target_specific(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        args.seed,
        None,
        logger,
        args.workers,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    global_df.to_csv(table_dir / "regression_global_results.csv", index=False)
    specific_df.to_csv(table_dir / "regression_target_specific_results.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 1/4 regression done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))

    n_perm = args.n_permutations or exp.permutation.n_permutations_default
    reg_perm = run_regression_permutation_tests(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        specific_df,
        PermutationConfig(n_perm, exp.permutation.alpha, args.seed, exp.permutation.nested_default),
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    reg_perm.to_csv(table_dir / "permutation_regression_results.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 2/4 permutation done | rows=%s elapsed=%s eta=%s", len(reg_perm), format_seconds(elapsed), format_seconds(eta))

    max_targets = args.max_targets or exp.shap.max_targets_default
    shap_df = run_regression_shap(
        x_user, y_reg, feature_sets, reg_models, specific_df, reg_perm, figure_dir, max_targets, args.seed
    )
    shap_df.to_csv(table_dir / "shap_feature_importance.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 3/4 shap done | rows=%s elapsed=%s eta=%s", len(shap_df), format_seconds(elapsed), format_seconds(eta))

    plot_regression_overview(global_df, figure_dir / "figure_regression_overview.png")
    plot_permutation_summary(reg_perm, None, figure_dir / "figure_permutation_pvalues.png")
    write_regression_comparison_artifacts(global_df, specific_df, table_dir, figure_dir)
    inf_cfg = InferenceBundleConfig(
        baseline_regression_model=exp.inference.baseline_regression_model,
        baseline_classification_model=exp.inference.baseline_classification_model,
        bootstrap_iterations=exp.inference.bootstrap_iterations,
        bayesian_bootstrap_samples=exp.inference.bayesian_bootstrap_samples,
        paired_alpha=exp.inference.paired_alpha,
        fdr_alpha=exp.inference.fdr_alpha,
        random_seed=args.seed,
    )
    reg_inf = run_regression_inference(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        specific_df,
        inf_cfg,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    reg_inf.to_csv(table_dir / "inference_regression.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 4/4 figures done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    logger.info("RQ2 end-to-end completed | tables=%s figures=%s", table_dir, figure_dir)
    return 0
