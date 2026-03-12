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
    est.add_argument("--seed", type=int, default=42)
    est.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    est.add_argument("--max-models", type=int, default=None)
    est.add_argument("--max-feature-sets", type=int, default=None)
    est.add_argument("--top-k-per-axis", type=int, default=None)
    est.add_argument("--class-balance", choices=["none", "oversample", "undersample", "smote"], default="none")
    est.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_estimation"))

    ff = sub.add_parser("fit-final-models", help="Fit final models from estimation winners")
    ff.add_argument("--data-dir", default="data/raw")
    ff.add_argument("--tables-dir", default="outputs/tables")
    ff.add_argument("--seed", type=int, default=42)
    ff.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    ff.add_argument("--top-k-per-axis", type=int, default=None)
    ff.add_argument("--class-balance", choices=["none", "oversample", "undersample", "smote"], default="none")
    ff.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_fit_final_models"))

    fe = sub.add_parser("run-final-explainability", help="Run SHAP from final fitted models only")
    fe.add_argument("--data-dir", default="data/raw")
    fe.add_argument("--tables-dir", default="outputs/tables")
    fe.add_argument("--figures-dir", default="outputs/figures")
    fe.add_argument("--seed", type=int, default=42)
    fe.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    fe.add_argument("--max-targets", type=int, default=None)
    fe.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_final_explainability"))

    pipe = sub.add_parser(
        "run-paper-pipeline",
        help="Run protocol pipeline (RQ1 + estimation + final-model explainability)",
    )
    pipe.add_argument("--data-dir", default="data/raw")
    pipe.add_argument("--tables-dir", default="outputs/tables")
    pipe.add_argument("--figures-dir", default="outputs/figures")
    pipe.add_argument("--seed", type=int, default=42)
    pipe.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    pipe.add_argument("--max-models", type=int, default=None)
    pipe.add_argument("--max-feature-sets", type=int, default=None)
    pipe.add_argument("--top-k-per-axis", type=int, default=None)
    pipe.add_argument("--class-balance", choices=["none", "oversample", "undersample", "smote"], default="none")
    pipe.add_argument("--max-targets", type=int, default=5)
    pipe.add_argument("--alpha", type=float, default=0.05)
    pipe.add_argument("--effect-threshold", type=float, default=0.30)
    pipe.set_defaults(func=_lazy_handler("usability_teleop.cli.commands_protocol", "cmd_run_paper_pipeline"))
    return parser
