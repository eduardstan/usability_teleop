import pytest

from usability_teleop.cli.main import build_parser


def test_cli_parser_accepts_doctor_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"


def test_cli_parser_accepts_unified_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["validate-data"]).command == "validate-data"
    assert parser.parse_args(["run-estimation"]).command == "run-estimation"
    assert parser.parse_args(["fit-final-models"]).command == "fit-final-models"
    assert parser.parse_args(["run-final-explainability"]).command == "run-final-explainability"
    assert parser.parse_args(["run-stat-validation"]).command == "run-stat-validation"
    assert parser.parse_args(["build-paper-artifacts"]).command == "build-paper-artifacts"
    assert parser.parse_args(["build-figures"]).command == "build-figures"
    assert parser.parse_args(["run-ablation"]).command == "run-ablation"
    assert parser.parse_args(["build-ablation-figures"]).command == "build-ablation-figures"


def test_cli_build_paper_artifacts_accepts_permutation_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["build-paper-artifacts", "--n-permutations", "10", "--nested-permutation"])
    assert args.command == "build-paper-artifacts"
    assert args.n_permutations == 10
    assert args.nested_permutation is True


def test_cli_class_balance_restricts_to_none_or_smote() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-estimation", "--class-balance", "smote"])
    assert args.class_balance == "smote"
    with pytest.raises(SystemExit):
        parser.parse_args(["run-estimation", "--class-balance", "oversample"])
