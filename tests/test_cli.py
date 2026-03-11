from usability_teleop.cli.main import build_parser


def test_cli_parser_accepts_doctor_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"


def test_cli_parser_accepts_phase34_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["run-correlation"]).command == "run-correlation"
    assert parser.parse_args(["run-regression"]).command == "run-regression"
    assert parser.parse_args(["run-classification"]).command == "run-classification"
    assert parser.parse_args(["run-permutation"]).command == "run-permutation"
    assert parser.parse_args(["run-inference"]).command == "run-inference"
    assert parser.parse_args(["run-shap"]).command == "run-shap"
    assert parser.parse_args(["build-figures"]).command == "build-figures"
    assert parser.parse_args(["run-rq2-end2end"]).command == "run-rq2-end2end"
    assert parser.parse_args(["run-rq23-end2end"]).command == "run-rq23-end2end"
