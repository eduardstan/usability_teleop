"""CLI commands for classification/permutation/SHAP/figure generation."""

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
from usability_teleop.evaluation.classification import run_classification_benchmark
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs
from usability_teleop.stats.permutation import (
    PermutationConfig,
    run_classification_permutation_tests,
    run_regression_permutation_tests,
)
from usability_teleop.stats.shap_analysis import run_regression_shap
from usability_teleop.viz.figures import (
    plot_classification_overview,
    plot_correlation_heatmap,
    plot_global_vs_target_specific_r2,
    plot_permutation_summary,
    plot_regression_overview,
)
from usability_teleop.viz.inference_figures import (
    plot_inference_bayesian,
    plot_inference_classification_ci,
    plot_inference_pvalues,
    plot_inference_regression_ci,
)


def cmd_run_classification(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-classification FAILED: %s", exc)
        return 1
    y_cls = prepare_targets(bundle.questionnaire, stage="classification")
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    model_specs = classification_model_specs()[: args.max_models] if args.max_models is not None else classification_model_specs()
    out = run_classification_benchmark(
        x_user, y_cls, feature_sets, model_specs, args.seed, args.max_feature_sets,
        exp.tuning.classification_scoring, exp.cv.classification_inner_max_splits, exp.cv.inner_shuffle, exp.cv.inner_random_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("classification rows=%s written to %s", len(out), output_path)
    return 0


def cmd_run_permutation(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    table_dir = Path(args.tables_dir).resolve()
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-permutation FAILED: %s", exc)
        return 1
    reg_path = table_dir / "regression_target_specific_results.csv"
    cls_path = table_dir / "classification_results.csv"
    if not reg_path.exists() or not cls_path.exists():
        logger.error("Required benchmark tables missing in %s", table_dir)
        return 1
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    reg_results = pd.read_csv(reg_path)
    cls_results = pd.read_csv(cls_path)
    reg_models = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()
    cls_models = classification_model_specs()[: args.max_models] if args.max_models is not None else classification_model_specs()
    if args.max_feature_sets is not None:
        allowed = {fs.name for fs in feature_sets[: args.max_feature_sets]}
        reg_results = reg_results[reg_results["feature_set"].isin(allowed)]
        cls_results = cls_results[cls_results["feature_set"].isin(allowed)]
    cfg = PermutationConfig(args.n_permutations or exp.permutation.n_permutations_default, exp.permutation.alpha, args.seed, exp.permutation.nested_default)
    y_reg = prepare_targets(bundle.questionnaire, stage="regression")
    y_cls = prepare_targets(bundle.questionnaire, stage="classification")
    reg_perm = run_regression_permutation_tests(x_user, y_reg, feature_sets, reg_models, reg_results, cfg, exp.tuning.regression_scoring, exp.cv.regression_inner_max_splits, exp.cv.inner_shuffle, exp.cv.inner_random_seed)
    cls_perm = run_classification_permutation_tests(x_user, y_cls, feature_sets, cls_models, cls_results, cfg, exp.tuning.classification_scoring, exp.cv.classification_inner_max_splits, exp.cv.inner_shuffle, exp.cv.inner_random_seed)
    reg_perm.to_csv(table_dir / "permutation_regression_results.csv", index=False)
    cls_perm.to_csv(table_dir / "permutation_classification_results.csv", index=False)
    logger.info("permutation regression rows=%s written to %s", len(reg_perm), table_dir / "permutation_regression_results.csv")
    logger.info("permutation classification rows=%s written to %s", len(cls_perm), table_dir / "permutation_classification_results.csv")
    return 0


def cmd_run_shap(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    table_dir = Path(args.tables_dir).resolve()
    figure_dir = Path(args.figures_dir).resolve()
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-shap FAILED: %s", exc)
        return 1
    reg_path = table_dir / "regression_target_specific_results.csv"
    if not reg_path.exists():
        logger.error("Missing %s", reg_path)
        return 1
    reg_results = pd.read_csv(reg_path)
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    reg_models = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()
    if args.max_feature_sets is not None:
        allowed = {fs.name for fs in feature_sets[: args.max_feature_sets]}
        reg_results = reg_results[reg_results["feature_set"].isin(allowed)]
    shap_df = run_regression_shap(
        x_user, prepare_targets(bundle.questionnaire, stage="regression"), feature_sets, reg_models, reg_results,
        pd.read_csv(table_dir / "permutation_regression_results.csv") if (table_dir / "permutation_regression_results.csv").exists() else None,
        figure_dir, args.max_targets or exp.shap.max_targets_default, args.seed,
    )
    shap_df.to_csv(table_dir / "shap_feature_importance.csv", index=False)
    logger.info("shap rows=%s written to %s", len(shap_df), table_dir / "shap_feature_importance.csv")
    return 0


def cmd_build_figures(args: argparse.Namespace, logger: object) -> int:
    table_dir = Path(args.tables_dir).resolve()
    figure_dir = Path(args.figures_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    if (table_dir / "correlation_results.csv").exists():
        plot_correlation_heatmap(pd.read_csv(table_dir / "correlation_results.csv"), figure_dir / "figure_correlation_heatmap.png")
    if (table_dir / "regression_global_results.csv").exists():
        plot_regression_overview(pd.read_csv(table_dir / "regression_global_results.csv"), figure_dir / "figure_regression_overview.png")
    if (table_dir / "classification_results.csv").exists():
        plot_classification_overview(pd.read_csv(table_dir / "classification_results.csv"), figure_dir / "figure_classification_overview.png")
    perm_reg = pd.read_csv(table_dir / "permutation_regression_results.csv") if (table_dir / "permutation_regression_results.csv").exists() else None
    perm_cls = pd.read_csv(table_dir / "permutation_classification_results.csv") if (table_dir / "permutation_classification_results.csv").exists() else None
    plot_permutation_summary(perm_reg, perm_cls, figure_dir / "figure_permutation_pvalues.png")
    cmp_path = table_dir / "regression_best_global_vs_target_specific.csv"
    if cmp_path.exists():
        plot_global_vs_target_specific_r2(pd.read_csv(cmp_path), figure_dir / "figure_regression_global_vs_target_specific.png")
    inf_reg_path = table_dir / "inference_regression.csv"
    inf_cls_path = table_dir / "inference_classification.csv"
    if inf_reg_path.exists():
        inf_reg = pd.read_csv(inf_reg_path)
        plot_inference_regression_ci(inf_reg, figure_dir / "figure_inference_regression_ci.png")
    else:
        inf_reg = pd.DataFrame()
    if inf_cls_path.exists():
        inf_cls = pd.read_csv(inf_cls_path)
        plot_inference_classification_ci(inf_cls, figure_dir / "figure_inference_classification_ci.png")
    else:
        inf_cls = pd.DataFrame()
    if not inf_reg.empty or not inf_cls.empty:
        plot_inference_pvalues(inf_reg, inf_cls, figure_dir / "figure_inference_pvalues.png")
        plot_inference_bayesian(inf_reg, inf_cls, figure_dir / "figure_inference_bayesian.png")
    logger.info("figures built in %s", figure_dir)
    return 0
