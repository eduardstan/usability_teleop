import pandas as pd

from usability_teleop.data.targets import prepare_targets


def _sample_questionnaire() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "physical_demand": ["strongly disagree", "agree"],
            "mental_demand": ["disagree", "agree"],
            "temporal_demand": ["neutral", "agree"],
            "effort": ["agree", "strongly agree"],
            "frustration": ["disagree", "neutral"],
            "usability": ["agree", "strongly agree"],
            "responsiveness": ["neutral", "agree"],
            "realistic": ["disagree", "agree"],
            "intuitive": ["agree", "agree"],
            "performance": ["neutral", "agree"],
        }
    )


def test_prepare_targets_regression_keeps_original_orientation() -> None:
    out = prepare_targets(_sample_questionnaire(), stage="regression")
    assert out.loc[1, "physical_demand"] == 1
    assert out.loc[1, "usability"] == 4


def test_prepare_targets_correlation_inverts_negative_targets_only() -> None:
    out = prepare_targets(_sample_questionnaire(), stage="correlation")
    assert out.loc[1, "physical_demand"] == 5
    assert out.loc[1, "mental_demand"] == 4
    assert out.loc[1, "usability"] == 4
