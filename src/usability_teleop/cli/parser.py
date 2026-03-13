"""CLI parser builder."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable


def _lazy_handler(module_name: str, handler_name: str) -> Callable[[argparse.Namespace, object], int]:
    def _runner(args: argparse.Namespace, logger: object) -> int:
        module = importlib.import_module(module_name)
        handler = getattr(module, handler_name)
        return handler(args, logger)

    return _runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="usability-teleop")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Validate project layout and runtime deps").set_defaults(
        func=_lazy_handler("usability_teleop.cli.commands_basic", "cmd_doctor")
    )

    v = sub.add_parser("validate-data", help="Validate raw data files against strict contracts")
    v.add_argument("--source-dir", default="data/raw", help="Directory containing required raw data files")
    v.add_argument("--copy-to-raw", action="store_true", help="Copy validated files to canonical data/raw")
    v.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_basic", "cmd_validate_data"))

    est = sub.add_parser("run-estimation", help="Run unified estimation lane (nested LOSO)")
    est.add_argument("--data-dir", default="data/raw")
    est.add_argument("--tables-dir", default="outputs/tables")
    est.add_argument("--runs-dir", default="outputs/runs")
    est.add_argument("--seed", type=int, default=42)
    est.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    est.add_argument("--models-config", default=None, help="Path to models YAML (fast/full profile)")
    est.add_argument("--max-models", type=int, default=None)
    est.add_argument("--max-feature-sets", type=int, default=None)
    est.add_argument("--top-k-per-axis", type=int, default=None)
    est.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_estimation"))

    ff = sub.add_parser("fit-final-models", help="Fit final models from estimation winners")
    ff.add_argument("--data-dir", default="data/raw")
    ff.add_argument("--tables-dir", default="outputs/tables")
    ff.add_argument("--runs-dir", default="outputs/runs")
    ff.add_argument("--seed", type=int, default=42)
    ff.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    ff.add_argument("--models-config", default=None, help="Path to models YAML (fast/full profile)")
    ff.add_argument("--top-k-per-axis", type=int, default=None)
    ff.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_fit_final_models"))

    fe = sub.add_parser("run-final-explainability", help="Run SHAP from final fitted models only")
    fe.add_argument("--data-dir", default="data/raw")
    fe.add_argument("--tables-dir", default="outputs/tables")
    fe.add_argument("--figures-dir", default="outputs/figures")
    fe.add_argument("--runs-dir", default="outputs/runs")
    fe.add_argument("--seed", type=int, default=42)
    fe.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    fe.add_argument("--max-targets", type=int, default=None)
    fe.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_final_explainability"))

    stat = sub.add_parser(
        "run-stat-validation",
        help="Run unified statistical validation (permutation + inference + global-vs-local tables)",
    )
    stat.add_argument("--data-dir", default="data/raw")
    stat.add_argument("--tables-dir", default="outputs/tables")
    stat.add_argument("--runs-dir", default="outputs/runs")
    stat.add_argument("--seed", type=int, default=42)
    stat.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    stat.add_argument("--models-config", default=None, help="Path to models YAML (fast/full profile)")
    stat.add_argument("--max-models", type=int, default=None)
    stat.add_argument("--max-feature-sets", type=int, default=None)
    stat.add_argument("--n-permutations", type=int, default=None)
    stat.add_argument("--nested-permutation", action="store_true")
    stat.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_stat_validation"))

    art = sub.add_parser(
        "build-paper-artifacts",
        help="Build full publication artifact bundle (tables + figures)",
    )
    art.add_argument("--data-dir", default="data/raw")
    art.add_argument("--tables-dir", default="outputs/tables")
    art.add_argument("--figures-dir", default="outputs/figures")
    art.add_argument("--runs-dir", default="outputs/runs")
    art.add_argument("--seed", type=int, default=42)
    art.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    art.add_argument("--models-config", default=None, help="Path to models YAML (fast/full profile)")
    art.add_argument("--max-models", type=int, default=None)
    art.add_argument("--max-feature-sets", type=int, default=None)
    art.add_argument("--top-k-per-axis", type=int, default=None)
    art.add_argument("--max-targets", type=int, default=5)
    art.add_argument("--alpha", type=float, default=0.05)
    art.add_argument("--effect-threshold", type=float, default=0.30)
    art.add_argument("--n-permutations", type=int, default=None)
    art.add_argument("--nested-permutation", action="store_true")
    art.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_build_paper_artifacts"))

    figs = sub.add_parser(
        "build-figures",
        help="Build publication figures from existing CSV tables (no training/stat recompute)",
    )
    figs.add_argument("--tables-dir", default="outputs/tables")
    figs.add_argument("--figures-dir", default="outputs/figures")
    figs.add_argument("--runs-dir", default="outputs/runs")
    figs.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_build_figures"))

    ab = sub.add_parser(
        "run-ablation",
        help="Run ablation study tables (baseline vs fold-safe feature-selection stages)",
    )
    ab.add_argument("--data-dir", default="data/raw")
    ab.add_argument("--tables-dir", default="outputs/tables")
    ab.add_argument("--runs-dir", default="outputs/runs")
    ab.add_argument("--seed", type=int, default=42)
    ab.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    ab.add_argument("--models-config", default=None, help="Path to models YAML (fast/full profile)")
    ab.add_argument("--max-models", type=int, default=None)
    ab.add_argument("--max-feature-sets", type=int, default=None)
    ab.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers for ablation stages (1 keeps deterministic sequential execution).",
    )
    ab.add_argument(
        "--top-k-per-axis",
        default="1,2,3,5",
        help="Comma-separated top-k-per-axis values for fold-safe selection ablation.",
    )
    ab.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_ablation"))

    abf = sub.add_parser(
        "build-ablation-figures",
        help="Build ablation publication figures from existing ablation CSV tables",
    )
    abf.add_argument("--tables-dir", default="outputs/tables")
    abf.add_argument("--figures-dir", default="outputs/figures")
    abf.add_argument("--runs-dir", default="outputs/runs")
    abf.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_build_ablation_figures"))
    return parser
