import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from usability_teleop.evaluation.classification import run_classification_benchmark
from usability_teleop.evaluation.correlation import run_correlation_analysis
from usability_teleop.evaluation.regression import (
    build_global_vs_target_specific_r2,
    run_regression_global,
    run_regression_target_specific,
)
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
        },
        index=users,
    )


def test_correlation_workflow_smoke() -> None:
    x_user = _tiny_features()
    y_corr = pd.DataFrame({"usability": [1, 2, 3, 4, 5, 5]}, index=x_user.index)
    out = run_correlation_analysis(x_user, y_corr)
    assert not out.empty
    assert {"target", "feature", "pearson_r", "spearman_rho"}.issubset(out.columns)


def test_regression_workflows_smoke() -> None:
    x_user = _tiny_features()
    y_reg = pd.DataFrame(
        {
            "usability": [1, 2, 3, 4, 4, 5],
            "responsiveness": [1, 1, 2, 3, 4, 5],
        },
        index=x_user.index,
    )

    fs = [FeatureSetSpec(name="x", axes=("x",), mode="subset")]
    models = [ModelSpec(name="Ridge", estimator_cls=Ridge, fixed_params={}, param_grid={"alpha": [0.1, 1.0]})]

    global_df = run_regression_global(x_user, y_reg, fs, models, max_feature_sets=1)
    specific_df = run_regression_target_specific(x_user, y_reg, fs, models, max_feature_sets=1)

    assert len(global_df) == 1
    assert len(specific_df) == 2
    assert {"rmse_mean", "mae_mean", "r2_mean"}.issubset(global_df.columns)
    assert {"target", "rmse", "mae", "r2"}.issubset(specific_df.columns)


def test_global_vs_target_specific_summary_smoke() -> None:
    global_df = pd.DataFrame(
        {
            "model": ["Ridge"],
            "feature_set": ["x"],
            "r2_mean": [0.1],
            "r2_usability": [0.2],
            "r2_responsiveness": [0.0],
        }
    )
    specific_df = pd.DataFrame(
        {
            "target": ["usability", "responsiveness"],
            "model": ["Ridge", "Ridge"],
            "feature_set": ["x", "x"],
            "r2": [0.3, 0.1],
        }
    )
    out = build_global_vs_target_specific_r2(global_df, specific_df)
    assert len(out) == 2
    assert {"target", "r2_global", "r2_specific", "delta_r2"}.issubset(out.columns)


def test_classification_workflow_smoke() -> None:
    x_user = _tiny_features()
    y_cls = pd.DataFrame(
        {
            "usability": [1, 2, 3, 4, 4, 5],
        },
        index=x_user.index,
    )

    fs = [FeatureSetSpec(name="x", axes=("x",), mode="subset")]
    models = [
        ModelSpec(
            name="LogReg",
            estimator_cls=LogisticRegression,
            fixed_params={},
            param_grid={"C": [0.1, 1.0], "penalty": ["l2"], "solver": ["liblinear"]},
        )
    ]

    out = run_classification_benchmark(x_user, y_cls, fs, models, max_feature_sets=1)
    assert not out.empty
    assert {"accuracy", "balanced_accuracy", "f1_macro", "auc"}.issubset(out.columns)
