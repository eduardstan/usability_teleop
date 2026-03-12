"""CLI parser builder."""

from __future__ import annotations

import argparse

from usability_teleop.cli.commands_basic import cmd_doctor, cmd_run_correlation, cmd_validate_data
from usability_teleop.cli.commands_inference import cmd_run_inference
from usability_teleop.cli.commands_study import cmd_run_ablation_study
from usability_teleop.cli.commands_rq2 import cmd_run_regression, cmd_run_rq2_end2end
from usability_teleop.cli.commands_rq3 import (
    cmd_build_figures,
    cmd_run_classification,
    cmd_run_permutation,
    cmd_run_shap,
)
from usability_teleop.cli.commands_rq23 import cmd_run_rq23_end2end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="usability-teleop")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Validate project layout and runtime deps").set_defaults(func=cmd_doctor)

    v = sub.add_parser("validate-data", help="Validate raw data files against strict contracts")
    v.add_argument("--source-dir", default="data/raw", help="Directory containing required raw data files")
    v.add_argument("--copy-to-raw", action="store_true", help="Copy validated files to canonical data/raw")
    v.set_defaults(func=cmd_validate_data)

    c = sub.add_parser("run-correlation", help="Run RQ1 correlation analysis and write result table")
    c.add_argument("--data-dir", default="data/raw")
    c.add_argument("--output", default="outputs/tables/correlation_results.csv")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--effect-threshold", type=float, default=0.30)
    c.set_defaults(func=cmd_run_correlation)

    r = sub.add_parser("run-regression", help="Run RQ2 regression benchmarks (global + target-specific)")
    r.add_argument("--data-dir", default="data/raw")
    r.add_argument("--output-dir", default="outputs/tables")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    r.add_argument("--max-feature-sets", type=int, default=None)
    r.add_argument("--max-models", type=int, default=None)
    r.add_argument("--workers", type=int, default=1)
    r.set_defaults(func=cmd_run_regression)

    cls = sub.add_parser("run-classification", help="Run RQ3 binary classification benchmark")
    cls.add_argument("--data-dir", default="data/raw")
    cls.add_argument("--output", default="outputs/tables/classification_results.csv")
    cls.add_argument("--seed", type=int, default=42)
    cls.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    cls.add_argument("--max-feature-sets", type=int, default=None)
    cls.add_argument("--max-models", type=int, default=None)
    cls.set_defaults(func=cmd_run_classification)

    p = sub.add_parser("run-permutation", help="Run permutation tests for best regression/classification configurations")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--tables-dir", default="outputs/tables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--max-feature-sets", type=int, default=None)
    p.add_argument("--max-models", type=int, default=None)
    p.set_defaults(func=cmd_run_permutation)

    inf = sub.add_parser("run-inference", help="Run extended inference bundle for RQ2/RQ3 best configs")
    inf.add_argument("--data-dir", default="data/raw")
    inf.add_argument("--tables-dir", default="outputs/tables")
    inf.add_argument("--seed", type=int, default=42)
    inf.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    inf.add_argument("--max-feature-sets", type=int, default=None)
    inf.add_argument("--max-models", type=int, default=None)
    inf.set_defaults(func=cmd_run_inference)

    s = sub.add_parser("run-shap", help="Run SHAP explainability for selected best regression targets")
    s.add_argument("--data-dir", default="data/raw")
    s.add_argument("--tables-dir", default="outputs/tables")
    s.add_argument("--figures-dir", default="outputs/figures")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    s.add_argument("--max-targets", type=int, default=None)
    s.add_argument("--max-feature-sets", type=int, default=None)
    s.add_argument("--max-models", type=int, default=None)
    s.set_defaults(func=cmd_run_shap)

    f = sub.add_parser("build-figures", help="Build publication-ready figures from table artifacts")
    f.add_argument("--tables-dir", default="outputs/tables")
    f.add_argument("--figures-dir", default="outputs/figures")
    f.set_defaults(func=cmd_build_figures)

    rq2 = sub.add_parser("run-rq2-end2end", help="Run full regression-focused end-to-end pipeline with ETA logging")
    rq2.add_argument("--data-dir", default="data/raw")
    rq2.add_argument("--tables-dir", default="outputs/tables")
    rq2.add_argument("--figures-dir", default="outputs/figures")
    rq2.add_argument("--seed", type=int, default=42)
    rq2.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    rq2.add_argument("--n-permutations", type=int, default=None)
    rq2.add_argument("--max-targets", type=int, default=None)
    rq2.add_argument("--max-feature-sets", type=int, default=None)
    rq2.add_argument("--max-models", type=int, default=None)
    rq2.add_argument("--workers", type=int, default=1)
    rq2.set_defaults(func=cmd_run_rq2_end2end)

    rq23 = sub.add_parser("run-rq23-end2end", help="Run full RQ2+RQ3 pipeline (regression + classification + stats + figures)")
    rq23.add_argument("--data-dir", default="data/raw")
    rq23.add_argument("--tables-dir", default="outputs/tables")
    rq23.add_argument("--figures-dir", default="outputs/figures")
    rq23.add_argument("--seed", type=int, default=42)
    rq23.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    rq23.add_argument("--n-permutations", type=int, default=None)
    rq23.add_argument("--max-targets", type=int, default=None)
    rq23.add_argument("--max-feature-sets", type=int, default=None)
    rq23.add_argument("--max-models", type=int, default=None)
    rq23.add_argument("--workers", type=int, default=1)
    rq23.set_defaults(func=cmd_run_rq23_end2end)

    study = sub.add_parser("run-ablation-study", help="Run ablation study with feature screening and balancing")
    study.add_argument("--data-dir", default="data/raw")
    study.add_argument("--tables-dir", default="outputs/tables")
    study.add_argument("--figures-dir", default="outputs/figures")
    study.add_argument("--seed", type=int, default=42)
    study.add_argument("--experiment-config", default=None, help="Path to experiment protocol YAML")
    study.add_argument("--max-models", type=int, default=2)
    study.add_argument("--max-feature-sets", type=int, default=2)
    study.add_argument("--top-k-per-axis", type=int, default=25)
    study.add_argument("--class-balance", choices=["none", "oversample", "undersample", "smote"], default="smote")
    study.add_argument("--workers", type=int, default=1)
    study.set_defaults(func=cmd_run_ablation_study)
    return parser
