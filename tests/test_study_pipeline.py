import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from usability_teleop.analysis.preprocessing import (
    build_target_distribution_table,
    filter_axis_top_variance,
)
from usability_teleop.evaluation.classification import run_classification_benchmark
from usability_teleop.evaluation.regression import run_regression_target_specific
from usability_teleop.features.ee_quat import FeatureSetSpec
from usability_teleop.modeling.registry import ModelSpec


def _tiny_features() -> pd.DataFrame:
    users = [1, 2, 3, 4, 5, 6]
    vals = np.array(users, dtype=float)
    return pd.DataFrame(
        {
            "min_ee_quat.x": vals,
            "max_ee_quat.x": vals + 0.1,
            "min_ee_quat.y": vals * 0.2,
            "max_ee_quat.y": vals * 0.3,
            "min_ee_quat.z": vals * 0.4,
            "max_ee_quat.z": vals * 0.5,
            "min_ee_quat.w": vals * 0.6,
            "max_ee_quat.w": vals * 0.7,
        },
        index=users,
    )


def test_variance_filter_and_target_distribution_smoke() -> None:
    x_user = _tiny_features()
    x_filt, summary = filter_axis_top_variance(x_user, top_k_per_axis=1)
    assert x_filt.shape[1] == 4
    assert set(summary["axis"]) == {"x", "y", "z", "w"}

    y = pd.DataFrame({"usability": [1, 2, 3, 4, 5, 5]}, index=x_user.index)
    dist = build_target_distribution_table(y, y)
    assert {"track", "target", "n"}.issubset(dist.columns)


def test_classification_balance_mode_smoke() -> None:
    x_user = _tiny_features()
    y_cls = pd.DataFrame({"usability": [1, 1, 1, 4, 4, 5]}, index=x_user.index)
    fs = [FeatureSetSpec(name="x", axes=("x",), mode="subset")]
    models = [
        ModelSpec(
            name="LogReg",
            estimator_cls=LogisticRegression,
            fixed_params={},
            param_grid={"C": [0.1], "penalty": ["l2"], "solver": ["liblinear"]},
        )
    ]
    out = run_classification_benchmark(
        x_user,
        y_cls,
        fs,
        models,
        max_feature_sets=1,
        class_balance="oversample",
    )
    assert not out.empty
    assert (out["status"] == "ok").all()


def test_regression_still_runs_on_filtered_features() -> None:
    x_user = _tiny_features()
    x_filt, _ = filter_axis_top_variance(x_user, top_k_per_axis=1)
    y_reg = pd.DataFrame({"usability": [1, 2, 3, 4, 4, 5]}, index=x_user.index)
    fs = [FeatureSetSpec(name="x", axes=("x",), mode="subset")]
    models = [ModelSpec(name="Ridge", estimator_cls=Ridge, fixed_params={}, param_grid={"alpha": [0.1]})]
    out = run_regression_target_specific(x_filt, y_reg, fs, models, max_feature_sets=1)
    assert len(out) == 1
