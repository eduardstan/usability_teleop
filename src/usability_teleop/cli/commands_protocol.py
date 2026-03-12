"""Unified protocol CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from usability_teleop.cli.common import DataValidationError, prepare_aligned_inputs, resolve_experiment_config
from usability_teleop.data.targets import prepare_targets
from usability_teleop.evaluation.correlation import CorrelationConfig, run_correlation_analysis
from usability_teleop.features.ee_quat import generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs
from usability_teleop.protocol.estimation import run_estimation_lane
from usability_teleop.protocol.final_models import fit_final_models
from usability_teleop.protocol.validation import (
    validate_estimation_best_configs,
    validate_final_models_table,
)
from usability_teleop.stats.permutation import (
    PermutationConfig,
    run_classification_permutation_tests,
    run_regression_permutation_tests,
)


def _run_estimation(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    args: argparse.Namespace,
    logger: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp = resolve_experiment_config(args.experiment_config)
    outputs = run_estimation_lane(
        x_user=x_user,
        y_reg=prepare_targets(questionnaire, "regression"),
        y_cls=prepare_targets(questionnaire, "classification"),
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
    return outputs.regression, outputs.classification, outputs.best_configs


def _fit_final_models(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    estimation_best: pd.DataFrame,
    args: argparse.Namespace,
    logger: object,
) -> pd.DataFrame:
    exp = resolve_experiment_config(args.experiment_config)
    validate_estimation_best_configs(estimation_best)
    return fit_final_models(
        x_user=x_user,
        y_reg=prepare_targets(questionnaire, "regression"),
        y_cls=prepare_targets(questionnaire, "classification"),
        estimation_best=estimation_best,
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


def _run_permutation(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = resolve_experiment_config(args.experiment_config)
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    if args.max_feature_sets is not None:
        feature_sets = feature_sets[: args.max_feature_sets]
    reg_models = regression_model_specs()[: args.max_models] if args.max_models is not None else regression_model_specs()
    cls_models = classification_model_specs()[: args.max_models] if args.max_models is not None else classification_model_specs()
    cfg = PermutationConfig(
        n_permutations=args.n_permutations or exp.permutation.n_permutations_default,
        alpha=exp.permutation.alpha,
        random_seed=args.seed,
        nested=args.nested_permutation if args.nested_permutation else exp.permutation.nested_default,
    )
    y_reg = prepare_targets(questionnaire, "regression")
    y_cls = prepare_targets(questionnaire, "classification")
    reg_perm = run_regression_permutation_tests(
        x_user,
        y_reg,
        feature_sets,
        reg_models,
        reg_df,
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
    return reg_perm, cls_perm


def _build_publication_figures(
    corr_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    reg_perm: pd.DataFrame,
    cls_perm: pd.DataFrame,
    figures_dir: Path,
) -> None:
    from usability_teleop.viz.figures import (
        plot_classification_overview,
        plot_correlation_heatmap,
        plot_permutation_summary,
        plot_regression_overview,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_correlation_heatmap(corr_df, figures_dir / "figure_correlation_heatmap.png")
    plot_regression_overview(reg_df, figures_dir / "figure_regression_overview.png")
    plot_classification_overview(cls_df, figures_dir / "figure_classification_overview.png")
    plot_permutation_summary(reg_perm, cls_perm, figures_dir / "figure_permutation_pvalues.png")


def _run_final_shap(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    final_models: pd.DataFrame,
    args: argparse.Namespace,
    figures_dir: Path,
) -> pd.DataFrame:
    from usability_teleop.protocol.explainability import run_final_explainability

    exp = resolve_experiment_config(args.experiment_config)
    validate_final_models_table(final_models)
    return run_final_explainability(
        x_user=x_user,
        y_reg=prepare_targets(questionnaire, "regression"),
        final_models=final_models,
        figure_dir=figures_dir,
        max_targets=args.max_targets or exp.shap.max_targets_default,
        seed=args.seed,
    )


def cmd_run_estimation(args: argparse.Namespace, logger: object) -> int:
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-estimation FAILED: %s", exc)
        return 1
    reg_df, cls_df, best_df = _run_estimation(x_user, bundle.questionnaire, args, logger)
    reg_df.to_csv(tables_dir / "estimation_regression.csv", index=False)
    cls_df.to_csv(tables_dir / "estimation_classification.csv", index=False)
    best_df.to_csv(tables_dir / "estimation_best_configs.csv", index=False)
    logger.info("estimation tables written to %s", tables_dir)
    return 0


def cmd_fit_final_models(args: argparse.Namespace, logger: object) -> int:
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
    try:
        estimation_best = pd.read_csv(best_path)
        final_df = _fit_final_models(
            x_user,
            bundle.questionnaire,
            estimation_best,
            args,
            logger,
        )
    except (ValueError, KeyError) as exc:
        logger.error("fit-final-models FAILED: %s", exc)
        return 1
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
    try:
        final_models = pd.read_csv(final_path)
        shap_df = _run_final_shap(
            x_user,
            bundle.questionnaire,
            final_models,
            args,
            figures_dir,
        )
    except (ValueError, KeyError) as exc:
        logger.error("run-final-explainability FAILED: %s", exc)
        return 1
    shap_df.to_csv(tables_dir / "final_explainability_shap.csv", index=False)
    logger.info("final explainability artifacts written to %s and %s", tables_dir, figures_dir)
    return 0


def cmd_run_paper_pipeline(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
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
    corr_df = run_correlation_analysis(
        x_user,
        y_corr,
        CorrelationConfig(alpha=args.alpha, effect_threshold=args.effect_threshold),
    )
    corr_df.to_csv(tables_dir / "correlation_results.csv", index=False)
    logger.info("correlation table written to %s", tables_dir / "correlation_results.csv")

    try:
        reg_df, cls_df, best_df = _run_estimation(x_user, bundle.questionnaire, args, logger)
        reg_df.to_csv(tables_dir / "estimation_regression.csv", index=False)
        cls_df.to_csv(tables_dir / "estimation_classification.csv", index=False)
        best_df.to_csv(tables_dir / "estimation_best_configs.csv", index=False)
        logger.info("estimation tables written to %s", tables_dir)

        reg_perm, cls_perm = _run_permutation(x_user, bundle.questionnaire, reg_df, cls_df, args)
        reg_perm.to_csv(tables_dir / "permutation_regression_results.csv", index=False)
        cls_perm.to_csv(tables_dir / "permutation_classification_results.csv", index=False)
        logger.info("permutation tables written to %s", tables_dir)

        _build_publication_figures(
            corr_df=corr_df,
            reg_df=reg_df,
            cls_df=cls_df,
            reg_perm=reg_perm,
            cls_perm=cls_perm,
            figures_dir=figures_dir,
        )
        logger.info("publication overview figures written to %s", figures_dir)

        final_df = _fit_final_models(x_user, bundle.questionnaire, best_df, args, logger)
        final_df.to_csv(tables_dir / "final_models.csv", index=False)
        logger.info("final models table written to %s", tables_dir / "final_models.csv")

        shap_df = _run_final_shap(x_user, bundle.questionnaire, final_df, args, figures_dir)
        shap_df.to_csv(tables_dir / "final_explainability_shap.csv", index=False)
        logger.info("final explainability table written to %s", tables_dir / "final_explainability_shap.csv")
    except (ValueError, KeyError) as exc:
        logger.error("run-paper-pipeline FAILED: %s", exc)
        return 1
    logger.info(
        "paper protocol pipeline completed | tables=%s figures=%s | includes permutation p-values | "
        "inference bundle not included in unified protocol lane | "
        "shap.max_targets_default=%s",
        tables_dir,
        figures_dir,
        exp.shap.max_targets_default,
    )
    return 0
