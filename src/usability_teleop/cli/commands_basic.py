"""CLI commands: doctor, data validation, and correlation."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from usability_teleop.cli.common import (
    DataValidationError,
    copy_raw_inputs,
    prepare_aligned_inputs,
    raw_data_paths,
)
from usability_teleop.config.settings import discover_project_paths
from usability_teleop.data.targets import prepare_targets
from usability_teleop.evaluation.correlation import CorrelationConfig, run_correlation_analysis

REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "shap",
    "matplotlib",
    "seaborn",
]


def cmd_doctor(_: argparse.Namespace, logger: object) -> int:
    paths = discover_project_paths(Path.cwd())
    logger.info("Project root: %s", paths.root)
    required = [
        paths.configs,
        paths.src,
        paths.data_raw,
        paths.data_interim,
        paths.data_processed,
        paths.outputs,
        paths.outputs_figures,
        paths.outputs_tables,
        paths.outputs_runs,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.error("Missing directories:")
        for item in missing:
            logger.error(" - %s", item)
        return 1
    logger.info("Directory layout OK")
    for pkg in REQUIRED_IMPORTS:
        module = importlib.import_module(pkg)
        logger.info("%s=%s", pkg, getattr(module, "__version__", "unknown"))
    return 0


def cmd_validate_data(args: argparse.Namespace, logger: object) -> int:
    source_dir = Path(args.source_dir).resolve()
    try:
        bundle = prepare_aligned_inputs(source_dir)[0]
    except DataValidationError as exc:
        logger.error("validate-data FAILED: %s", exc)
        return 1
    logger.info("validate-data OK")
    logger.info(
        "summary: users=%s samples=%s features=%s",
        bundle.summary.n_users,
        bundle.summary.n_samples,
        bundle.summary.n_features,
    )
    if args.copy_to_raw:
        copy_raw_inputs(raw_data_paths(source_dir), discover_project_paths(Path.cwd()).data_raw)
        logger.info("copied source files to %s", discover_project_paths(Path.cwd()).data_raw)
    return 0


def cmd_run_correlation(args: argparse.Namespace, logger: object) -> int:
    source_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()
    try:
        bundle, x_user = prepare_aligned_inputs(source_dir)
    except DataValidationError as exc:
        logger.error("run-correlation FAILED: %s", exc)
        return 1
    y_corr = prepare_targets(bundle.questionnaire, stage="correlation")
    result = run_correlation_analysis(
        x_user,
        y_corr,
        CorrelationConfig(alpha=args.alpha, effect_threshold=args.effect_threshold),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("correlation rows=%s written to %s", len(result), output_path)
    return 0
