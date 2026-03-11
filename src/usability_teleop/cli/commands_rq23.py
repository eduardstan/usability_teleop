"""CLI command for full RQ2+RQ3 end-to-end execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from usability_teleop.cli.commands_inference import compute_inference_tables
from usability_teleop.cli.common import (
    DataValidationError,
    prepare_aligned_inputs,
    resolve_experiment_config,
    write_regression_comparison_artifacts,
)
from usability_teleop.data.targets import prepare_targets
from usability_teleop.evaluation.classification import run_classification_benchmark
from usability_teleop.evaluation.regression import (
    run_regression_global,
    run_regression_target_specific,
)
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs
from usability_teleop.stats.permutation import (
    PermutationConfig,
    run_classification_permutation_tests,
    run_regression_permutation_tests,
)
from usability_teleop.stats.shap_analysis import run_regression_shap
from usability_teleop.utils.timing import ProgressTracker, format_seconds
from usability_teleop.viz.figures import (
    plot_classification_overview,
    plot_permutation_summary,
    plot_regression_overview,
)


def cmd_run_rq23_end2end(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    table_dir = Path(args.tables_dir).resolve()
    figure_dir = Path(args.figures_dir).resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-rq23-end2end FAILED: %s", exc)
        return 1

    y_reg = prepare_targets(bundle.questionnaire, stage="regression")
    y_cls = prepare_targets(bundle.questionnaire, stage="classification")
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    if args.max_feature_sets is not None:
        feature_sets = feature_sets[: args.max_feature_sets]
    reg_models = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()
    cls_models = classification_model_specs()[: args.max_models] if args.max_models is not None else classification_model_specs()

    stage = ProgressTracker(total=7)
    logger.info("RQ2+RQ3 end-to-end starting | users=%s feature_sets=%s reg_models=%s cls_models=%s workers=%s", len(x_user), len(feature_sets), len(reg_models), len(cls_models), args.workers)
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
    logger.info("stage 1/6 regression done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    cls_df = run_classification_benchmark(
        x_user,
        y_cls,
        feature_sets,
        cls_models,
        args.seed,
        None,
        exp.tuning.classification_scoring,
        exp.cv.classification_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    cls_df.to_csv(table_dir / "classification_results.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 2/6 classification done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    n_perm = args.n_permutations or exp.permutation.n_permutations_default
    cfg = PermutationConfig(n_perm, exp.permutation.alpha, args.seed, exp.permutation.nested_default)
    reg_perm = run_regression_permutation_tests(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        specific_df,
        cfg,
        exp.tuning.regression_scoring,
        exp.cv.regression_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    cls_perm = run_classification_permutation_tests(
        x_user,
        y_cls,
        feature_sets,
        cls_models,
        cls_df,
        cfg,
        exp.tuning.classification_scoring,
        exp.cv.classification_inner_max_splits,
        exp.cv.inner_shuffle,
        exp.cv.inner_random_seed,
    )
    reg_perm.to_csv(table_dir / "permutation_regression_results.csv", index=False)
    cls_perm.to_csv(table_dir / "permutation_classification_results.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 3/7 permutation done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    max_targets = args.max_targets or exp.shap.max_targets_default
    shap_df = run_regression_shap(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        specific_df,
        reg_perm,
        figure_dir,
        max_targets,
        args.seed,
    )
    shap_df.to_csv(table_dir / "shap_feature_importance.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 4/7 shap done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    comparison = write_regression_comparison_artifacts(global_df, specific_df, table_dir, figure_dir)
    logger.info("global-vs-target-specific rows=%s written to %s", len(comparison), table_dir / "regression_best_global_vs_target_specific.csv")
    elapsed, eta = stage.step()
    logger.info("stage 5/7 comparison artifacts done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    reg_inf, cls_inf = compute_inference_tables(
        x_user,
        y_reg,
        y_cls,
        feature_sets,
        reg_models,
        cls_models,
        specific_df,
        cls_df,
        exp,
        args.seed,
    )
    reg_inf.to_csv(table_dir / "inference_regression.csv", index=False)
    cls_inf.to_csv(table_dir / "inference_classification.csv", index=False)
    elapsed, eta = stage.step()
    logger.info("stage 6/7 inference done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))

    plot_regression_overview(global_df, figure_dir / "figure_regression_overview.png")
    plot_classification_overview(cls_df, figure_dir / "figure_classification_overview.png")
    plot_permutation_summary(reg_perm, cls_perm, figure_dir / "figure_permutation_pvalues.png")
    elapsed, eta = stage.step()
    logger.info("stage 7/7 figures done | elapsed=%s eta=%s", format_seconds(elapsed), format_seconds(eta))
    logger.info("RQ2+RQ3 end-to-end completed | tables=%s figures=%s", table_dir, figure_dir)
    return 0
