"""Unified protocol CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

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
from usability_teleop.stats.inference import (
    InferenceBundleConfig,
    run_classification_inference,
    run_regression_inference,
)


def _resolve_models_config(path: str | None) -> Path | None:
    return Path(path).resolve() if path else None


def _run_estimation(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    args: argparse.Namespace,
    logger: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp = resolve_experiment_config(args.experiment_config)
    models_config = _resolve_models_config(getattr(args, "models_config", None))
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
        models_config=models_config,
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
    models_config = _resolve_models_config(getattr(args, "models_config", None))
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
        models_config=models_config,
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
    models_config = _resolve_models_config(getattr(args, "models_config", None))
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    if args.max_feature_sets is not None:
        feature_sets = feature_sets[: args.max_feature_sets]
    reg_pool = regression_model_specs(config_path=models_config)
    cls_pool = classification_model_specs(config_path=models_config)
    reg_models = reg_pool[: args.max_models] if args.max_models is not None else reg_pool
    cls_models = cls_pool[: args.max_models] if args.max_models is not None else cls_pool
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
    reg_comparison_df: pd.DataFrame,
    cls_comparison_df: pd.DataFrame,
    inf_reg_df: pd.DataFrame,
    inf_cls_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    from usability_teleop.viz.figures import (
        plot_classification_overview,
        plot_correlation_heatmap,
        plot_global_vs_target_specific_auc,
        plot_global_vs_target_specific_r2,
        plot_permutation_summary,
        plot_protocol_dashboard,
        plot_regression_overview,
    )
    from usability_teleop.viz.inference_figures import (
        plot_inference_bayesian,
        plot_inference_classification_ci,
        plot_inference_pvalues,
        plot_inference_regression_ci,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_correlation_heatmap(corr_df, figures_dir / "figure_correlation_heatmap.png")
    plot_regression_overview(reg_df, figures_dir / "figure_regression_overview.png")
    plot_classification_overview(cls_df, figures_dir / "figure_classification_overview.png")
    plot_permutation_summary(reg_perm, cls_perm, figures_dir / "figure_permutation_pvalues.png")
    plot_global_vs_target_specific_r2(reg_comparison_df, figures_dir / "figure_regression_global_vs_target_specific.png")
    plot_global_vs_target_specific_auc(
        cls_comparison_df,
        figures_dir / "figure_classification_global_vs_target_specific.png",
    )
    plot_inference_regression_ci(inf_reg_df, figures_dir / "figure_inference_regression_ci.png")
    plot_inference_classification_ci(inf_cls_df, figures_dir / "figure_inference_classification_ci.png")
    plot_inference_pvalues(inf_reg_df, inf_cls_df, figures_dir / "figure_inference_pvalues.png")
    plot_inference_bayesian(inf_reg_df, inf_cls_df, figures_dir / "figure_inference_bayesian.png")
    plot_protocol_dashboard(
        reg_comparison_df,
        reg_perm,
        cls_perm,
        inf_reg_df,
        inf_cls_df,
        figures_dir / "figure_protocol_dashboard.png",
    )


def _load_csv_or_warn(path: Path, logger: object) -> pd.DataFrame:
    if not path.exists():
        logger.warning("missing table input: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("failed to read %s (%s); skipping dependent figures", path, exc)
        return pd.DataFrame()


def _run_plot(
    plot_fn: Callable[..., None],
    output_path: Path,
    logger: object,
    *args: Any,
) -> bool:
    try:
        plot_fn(*args, output_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("failed to build figure %s (%s)", output_path.name, exc)
        return False
    if output_path.exists():
        logger.info("figure written: %s", output_path)
        return True
    logger.warning("figure skipped (insufficient data): %s", output_path.name)
    return False


def _build_global_vs_target_specific_comparison(reg_df: pd.DataFrame) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "r2_global",
                "r2_specific",
                "delta_r2",
                "model_global",
                "feature_set_global",
                "model_specific",
                "feature_set_specific",
            ]
        )

    grouped = (
        reg_df.groupby(["model", "feature_set"], as_index=False)["r2"]
        .mean()
        .rename(columns={"r2": "r2_mean"})
        .sort_values("r2_mean", ascending=False)
    )
    best_global = grouped.iloc[0]
    g_model = str(best_global["model"])
    g_feature_set = str(best_global["feature_set"])

    rows: list[dict[str, object]] = []
    for target in sorted(reg_df["target"].unique().tolist()):
        target_df = reg_df[reg_df["target"] == target]
        if target_df.empty:
            continue
        best_specific = target_df.sort_values("r2", ascending=False).iloc[0]
        global_target = target_df[
            (target_df["model"] == g_model)
            & (target_df["feature_set"] == g_feature_set)
        ]
        if global_target.empty:
            continue
        r2_global = float(global_target.iloc[0]["r2"])
        r2_specific = float(best_specific["r2"])
        rows.append(
            {
                "target": target,
                "r2_global": r2_global,
                "r2_specific": r2_specific,
                "delta_r2": float(r2_specific - r2_global),
                "model_global": g_model,
                "feature_set_global": g_feature_set,
                "model_specific": str(best_specific["model"]),
                "feature_set_specific": str(best_specific["feature_set"]),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_r2", ascending=False).reset_index(drop=True)


def _build_classification_global_vs_target_specific_comparison(cls_df: pd.DataFrame) -> pd.DataFrame:
    if cls_df.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "auc_global",
                "auc_specific",
                "delta_auc",
                "model_global",
                "feature_set_global",
                "model_specific",
                "feature_set_specific",
            ]
        )

    valid = cls_df[cls_df["status"] == "ok"].copy() if "status" in cls_df.columns else cls_df.copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "auc_global",
                "auc_specific",
                "delta_auc",
                "model_global",
                "feature_set_global",
                "model_specific",
                "feature_set_specific",
            ]
        )

    grouped = (
        valid.groupby(["model", "feature_set"], as_index=False)["auc"]
        .mean()
        .rename(columns={"auc": "auc_mean"})
        .sort_values("auc_mean", ascending=False)
    )
    best_global = grouped.iloc[0]
    g_model = str(best_global["model"])
    g_feature_set = str(best_global["feature_set"])

    rows: list[dict[str, object]] = []
    for target in sorted(valid["target"].unique().tolist()):
        target_df = valid[valid["target"] == target]
        if target_df.empty:
            continue
        best_specific = target_df.sort_values("auc", ascending=False).iloc[0]
        global_target = target_df[
            (target_df["model"] == g_model)
            & (target_df["feature_set"] == g_feature_set)
        ]
        if global_target.empty:
            continue
        auc_global = float(global_target.iloc[0]["auc"])
        auc_specific = float(best_specific["auc"])
        rows.append(
            {
                "target": target,
                "auc_global": auc_global,
                "auc_specific": auc_specific,
                "delta_auc": float(auc_specific - auc_global),
                "model_global": g_model,
                "feature_set_global": g_feature_set,
                "model_specific": str(best_specific["model"]),
                "feature_set_specific": str(best_specific["feature_set"]),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)


def _run_inference_bundle(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = resolve_experiment_config(args.experiment_config)
    models_config = _resolve_models_config(getattr(args, "models_config", None))
    feature_sets = generate_ee_quat_feature_sets(include_average=exp.feature_sets.include_average)
    if args.max_feature_sets is not None:
        feature_sets = feature_sets[: args.max_feature_sets]
    reg_pool = regression_model_specs(config_path=models_config)
    cls_pool = classification_model_specs(config_path=models_config)
    reg_models = reg_pool[: args.max_models] if args.max_models is not None else reg_pool
    cls_models = cls_pool[: args.max_models] if args.max_models is not None else cls_pool
    cfg = InferenceBundleConfig(
        baseline_regression_model=exp.inference.baseline_regression_model,
        baseline_classification_model=exp.inference.baseline_classification_model,
        bootstrap_iterations=exp.inference.bootstrap_iterations,
        bayesian_bootstrap_samples=exp.inference.bayesian_bootstrap_samples,
        paired_alpha=exp.inference.paired_alpha,
        fdr_alpha=exp.inference.fdr_alpha,
        random_seed=args.seed,
    )
    y_reg = prepare_targets(questionnaire, "regression")
    y_cls = prepare_targets(questionnaire, "classification")
    inf_reg = run_regression_inference(
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
    inf_cls = run_classification_inference(
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
    return inf_reg, inf_cls


def _run_stat_validation_bundle(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg_perm, cls_perm = _run_permutation(x_user, questionnaire, reg_df, cls_df, args)
    reg_comparison_df = _build_global_vs_target_specific_comparison(reg_df)
    cls_comparison_df = _build_classification_global_vs_target_specific_comparison(cls_df)
    inf_reg_df, inf_cls_df = _run_inference_bundle(x_user, questionnaire, reg_df, cls_df, args)
    return reg_perm, cls_perm, reg_comparison_df, cls_comparison_df, inf_reg_df, inf_cls_df


def _run_final_shap(
    x_user: pd.DataFrame,
    questionnaire: pd.DataFrame,
    final_models: pd.DataFrame,
    args: argparse.Namespace,
    figures_dir: Path,
) -> pd.DataFrame:
    from usability_teleop.protocol.explainability import run_final_explainability

    exp = resolve_experiment_config(args.experiment_config)
    models_config = _resolve_models_config(getattr(args, "models_config", None))
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


def cmd_run_stat_validation(args: argparse.Namespace, logger: object) -> int:
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    reg_path = tables_dir / "estimation_regression.csv"
    cls_path = tables_dir / "estimation_classification.csv"
    if not reg_path.exists() or not cls_path.exists():
        logger.error(
            "Missing estimation tables. Run run-estimation first. Expected: %s and %s",
            reg_path,
            cls_path,
        )
        return 1
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-stat-validation FAILED: %s", exc)
        return 1

    y_corr = prepare_targets(bundle.questionnaire, stage="correlation")
    corr_df = run_correlation_analysis(
        x_user,
        y_corr,
        CorrelationConfig(alpha=0.05, effect_threshold=0.30),
    )
    corr_df.to_csv(tables_dir / "correlation_results.csv", index=False)
    logger.info("correlation table written to %s", tables_dir / "correlation_results.csv")

    reg_df = pd.read_csv(reg_path)
    cls_df = pd.read_csv(cls_path)
    try:
        reg_perm, cls_perm, reg_comparison_df, cls_comparison_df, inf_reg_df, inf_cls_df = _run_stat_validation_bundle(
            x_user,
            bundle.questionnaire,
            reg_df,
            cls_df,
            args,
        )
    except (ValueError, KeyError) as exc:
        logger.error("run-stat-validation FAILED: %s", exc)
        return 1

    reg_perm.to_csv(tables_dir / "permutation_regression_results.csv", index=False)
    cls_perm.to_csv(tables_dir / "permutation_classification_results.csv", index=False)
    reg_comparison_df.to_csv(tables_dir / "regression_best_global_vs_target_specific.csv", index=False)
    cls_comparison_df.to_csv(tables_dir / "classification_best_global_vs_target_specific.csv", index=False)
    inf_reg_df.to_csv(tables_dir / "inference_regression.csv", index=False)
    inf_cls_df.to_csv(tables_dir / "inference_classification.csv", index=False)
    logger.info("stat-validation tables written to %s", tables_dir)
    return 0


def cmd_build_paper_artifacts(args: argparse.Namespace, logger: object) -> int:
    exp = resolve_experiment_config(args.experiment_config)
    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("build-paper-artifacts FAILED: %s", exc)
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

        reg_perm, cls_perm, reg_comparison_df, cls_comparison_df, inf_reg_df, inf_cls_df = _run_stat_validation_bundle(
            x_user,
            bundle.questionnaire,
            reg_df,
            cls_df,
            args,
        )
        reg_perm.to_csv(tables_dir / "permutation_regression_results.csv", index=False)
        cls_perm.to_csv(tables_dir / "permutation_classification_results.csv", index=False)
        reg_comparison_df.to_csv(tables_dir / "regression_best_global_vs_target_specific.csv", index=False)
        cls_comparison_df.to_csv(tables_dir / "classification_best_global_vs_target_specific.csv", index=False)
        inf_reg_df.to_csv(tables_dir / "inference_regression.csv", index=False)
        inf_cls_df.to_csv(tables_dir / "inference_classification.csv", index=False)
        logger.info("stat-validation tables written to %s", tables_dir)

        _build_publication_figures(
            corr_df=corr_df,
            reg_df=reg_df,
            cls_df=cls_df,
            reg_perm=reg_perm,
            cls_perm=cls_perm,
            reg_comparison_df=reg_comparison_df,
            cls_comparison_df=cls_comparison_df,
            inf_reg_df=inf_reg_df,
            inf_cls_df=inf_cls_df,
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
        logger.error("build-paper-artifacts FAILED: %s", exc)
        return 1
    logger.info(
        "paper artifacts completed | tables=%s figures=%s | includes permutation p-values and "
        "inference CI/statistical figures | "
        "shap.max_targets_default=%s",
        tables_dir,
        figures_dir,
        exp.shap.max_targets_default,
    )
    return 0


def cmd_build_figures(args: argparse.Namespace, logger: object) -> int:
    from usability_teleop.viz.figures import (
        plot_classification_overview,
        plot_correlation_heatmap,
        plot_global_vs_target_specific_auc,
        plot_global_vs_target_specific_r2,
        plot_permutation_summary,
        plot_protocol_dashboard,
        plot_regression_overview,
    )
    from usability_teleop.viz.inference_figures import (
        plot_inference_bayesian,
        plot_inference_classification_ci,
        plot_inference_pvalues,
        plot_inference_regression_ci,
    )

    tables_dir = Path(args.tables_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    corr_df = _load_csv_or_warn(tables_dir / "correlation_results.csv", logger)
    reg_df = _load_csv_or_warn(tables_dir / "estimation_regression.csv", logger)
    cls_df = _load_csv_or_warn(tables_dir / "estimation_classification.csv", logger)
    reg_perm_df = _load_csv_or_warn(tables_dir / "permutation_regression_results.csv", logger)
    cls_perm_df = _load_csv_or_warn(tables_dir / "permutation_classification_results.csv", logger)
    comparison_df = _load_csv_or_warn(tables_dir / "regression_best_global_vs_target_specific.csv", logger)
    cls_comparison_df = _load_csv_or_warn(tables_dir / "classification_best_global_vs_target_specific.csv", logger)
    inf_reg_df = _load_csv_or_warn(tables_dir / "inference_regression.csv", logger)
    inf_cls_df = _load_csv_or_warn(tables_dir / "inference_classification.csv", logger)

    built: list[str] = []
    skipped: list[str] = []

    def _mark_skip(fig_name: str, reason: str) -> None:
        logger.warning("figure skipped: %s (%s)", fig_name, reason)
        skipped.append(fig_name)

    if {"pearson_highlight", "pearson_r", "feature", "target"}.issubset(corr_df.columns):
        if _run_plot(plot_correlation_heatmap, figures_dir / "figure_correlation_heatmap.png", logger, corr_df):
            built.append("figure_correlation_heatmap.png")
        else:
            skipped.append("figure_correlation_heatmap.png")
    else:
        _mark_skip("figure_correlation_heatmap.png", "missing correlation columns")

    if _run_plot(plot_regression_overview, figures_dir / "figure_regression_overview.png", logger, reg_df):
        built.append("figure_regression_overview.png")
    else:
        skipped.append("figure_regression_overview.png")

    if _run_plot(plot_classification_overview, figures_dir / "figure_classification_overview.png", logger, cls_df):
        built.append("figure_classification_overview.png")
    else:
        skipped.append("figure_classification_overview.png")

    if _run_plot(
        plot_permutation_summary,
        figures_dir / "figure_permutation_pvalues.png",
        logger,
        reg_perm_df,
        cls_perm_df,
    ):
        built.append("figure_permutation_pvalues.png")
    else:
        skipped.append("figure_permutation_pvalues.png")

    if _run_plot(
        plot_global_vs_target_specific_r2,
        figures_dir / "figure_regression_global_vs_target_specific.png",
        logger,
        comparison_df,
    ):
        built.append("figure_regression_global_vs_target_specific.png")
    else:
        skipped.append("figure_regression_global_vs_target_specific.png")

    if _run_plot(
        plot_global_vs_target_specific_auc,
        figures_dir / "figure_classification_global_vs_target_specific.png",
        logger,
        cls_comparison_df,
    ):
        built.append("figure_classification_global_vs_target_specific.png")
    else:
        skipped.append("figure_classification_global_vs_target_specific.png")

    if _run_plot(plot_inference_regression_ci, figures_dir / "figure_inference_regression_ci.png", logger, inf_reg_df):
        built.append("figure_inference_regression_ci.png")
    else:
        skipped.append("figure_inference_regression_ci.png")

    if _run_plot(
        plot_inference_classification_ci,
        figures_dir / "figure_inference_classification_ci.png",
        logger,
        inf_cls_df,
    ):
        built.append("figure_inference_classification_ci.png")
    else:
        skipped.append("figure_inference_classification_ci.png")

    if _run_plot(
        plot_inference_pvalues,
        figures_dir / "figure_inference_pvalues.png",
        logger,
        inf_reg_df,
        inf_cls_df,
    ):
        built.append("figure_inference_pvalues.png")
    else:
        skipped.append("figure_inference_pvalues.png")

    if _run_plot(
        plot_inference_bayesian,
        figures_dir / "figure_inference_bayesian.png",
        logger,
        inf_reg_df,
        inf_cls_df,
    ):
        built.append("figure_inference_bayesian.png")
    else:
        skipped.append("figure_inference_bayesian.png")

    has_dashboard_inputs = any(
        not df.empty for df in [comparison_df, reg_perm_df, cls_perm_df, inf_reg_df, inf_cls_df]
    )
    if has_dashboard_inputs:
        if _run_plot(
            plot_protocol_dashboard,
            figures_dir / "figure_protocol_dashboard.png",
            logger,
            comparison_df,
            reg_perm_df,
            cls_perm_df,
            inf_reg_df,
            inf_cls_df,
        ):
            built.append("figure_protocol_dashboard.png")
        else:
            skipped.append("figure_protocol_dashboard.png")
    else:
        _mark_skip("figure_protocol_dashboard.png", "no dashboard inputs available")

    ab_built, ab_skipped = _build_ablation_figures_from_tables(tables_dir, figures_dir, logger)
    built.extend(ab_built)
    skipped.extend(ab_skipped)

    report_path = runs_dir / "build_figures_report.json"
    report = {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "built_figures": built,
        "skipped_figures": skipped,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "build-figures completed | built=%s skipped=%s report=%s",
        len(built),
        len(skipped),
        report_path,
    )
    return 0


def _build_ablation_figures_from_tables(
    tables_dir: Path,
    figures_dir: Path,
    logger: object,
) -> tuple[list[str], list[str]]:
    from usability_teleop.viz.study_figures import (
        plot_study_delta_heatmap,
        plot_study_stage_summary,
        plot_study_target_distributions,
    )

    summary_df = _load_csv_or_warn(tables_dir / "ablation_summary.csv", logger)
    breakdown_df = _load_csv_or_warn(tables_dir / "ablation_breakdown.csv", logger)
    target_dist_df = _load_csv_or_warn(tables_dir / "ablation_target_distributions.csv", logger)

    built: list[str] = []
    skipped: list[str] = []
    figure_specs: list[tuple[str, Callable[..., None], tuple[Any, ...]]] = [
        ("figure_ablation_stage_summary.png", plot_study_stage_summary, (summary_df,)),
        ("figure_ablation_delta_heatmap.png", plot_study_delta_heatmap, (breakdown_df,)),
        ("figure_ablation_target_distributions.png", plot_study_target_distributions, (target_dist_df,)),
    ]
    for fig_name, plot_fn, plot_args in figure_specs:
        out = figures_dir / fig_name
        if _run_plot(plot_fn, out, logger, *plot_args):
            built.append(fig_name)
        else:
            skipped.append(fig_name)
    return built, skipped


def cmd_run_ablation(args: argparse.Namespace, logger: object) -> int:
    from usability_teleop.analysis import build_target_distribution_table, run_ablation_study

    source_dir = Path(args.data_dir).resolve()
    tables_dir = Path(args.tables_dir).resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-ablation FAILED: %s", exc)
        return 1

    exp = resolve_experiment_config(args.experiment_config)
    models_config = _resolve_models_config(getattr(args, "models_config", None))
    y_reg = prepare_targets(bundle.questionnaire, "regression")
    y_cls = prepare_targets(bundle.questionnaire, "classification")
    try:
        outputs = run_ablation_study(
            x_base=x_user,
            y_reg=y_reg,
            y_cls=y_cls,
            max_models=args.max_models,
            max_feature_sets=args.max_feature_sets,
            top_k_per_axis=args.top_k_per_axis,
            class_balance=args.class_balance,
            models_config=models_config,
            seed=args.seed,
            workers=1,
            tuning_regression_scoring=exp.tuning.regression_scoring,
            tuning_classification_scoring=exp.tuning.classification_scoring,
            inner_regression_splits=exp.cv.regression_inner_max_splits,
            inner_classification_splits=exp.cv.classification_inner_max_splits,
            inner_shuffle=exp.cv.inner_shuffle,
            inner_seed=exp.cv.inner_random_seed,
            logger=logger,
        )
    except (ValueError, KeyError) as exc:
        logger.error("run-ablation FAILED: %s", exc)
        return 1

    target_dist_df = build_target_distribution_table(y_reg, y_cls)
    outputs.summary.to_csv(tables_dir / "ablation_summary.csv", index=False)
    outputs.breakdown.to_csv(tables_dir / "ablation_breakdown.csv", index=False)
    outputs.feature_filter_summary.to_csv(tables_dir / "ablation_feature_filter_summary.csv", index=False)
    target_dist_df.to_csv(tables_dir / "ablation_target_distributions.csv", index=False)
    logger.info("ablation tables written to %s", tables_dir)
    return 0


def cmd_build_ablation_figures(args: argparse.Namespace, logger: object) -> int:
    tables_dir = Path(args.tables_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    built, skipped = _build_ablation_figures_from_tables(tables_dir, figures_dir, logger)
    report_path = runs_dir / "build_ablation_figures_report.json"
    report = {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "built_figures": built,
        "skipped_figures": skipped,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "build-ablation-figures completed | built=%s skipped=%s report=%s",
        len(built),
        len(skipped),
        report_path,
    )
    return 0
