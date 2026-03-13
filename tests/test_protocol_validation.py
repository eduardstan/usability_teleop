import sys

import pandas as pd
import pytest

from usability_teleop.cli.parser import build_parser
from usability_teleop.protocol.validation import (
    validate_estimation_best_configs,
    validate_final_models_table,
)


def test_validate_estimation_best_configs_requires_columns() -> None:
    bad = pd.DataFrame({"track": ["regression"], "target": ["usability"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_estimation_best_configs(bad)


def test_validate_final_models_table_requires_columns() -> None:
    bad = pd.DataFrame({"track": ["regression"], "target": ["usability"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_final_models_table(bad)


def test_parser_build_is_lazy_for_command_modules() -> None:
    sys.modules.pop("usability_teleop.cli.commands_basic", None)
    parser = build_parser()
    _ = parser.parse_args(["run-estimation"])
    assert "usability_teleop.cli.commands_basic" not in sys.modules
