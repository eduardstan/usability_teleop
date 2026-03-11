from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from usability_teleop.features.ee_quat import FeatureSetSpec
from usability_teleop.modeling.registry import ModelSpec
from usability_teleop.stats.inference import (
    InferenceBundleConfig,
    run_classification_inference,
    run_regression_inference,
)
from usability_teleop.stats.permutation import (
    PermutationConfig,
    run_classification_permutation_tests,
    run_regression_permutation_tests,
)
from usability_teleop.viz.figures import (
    plot_classification_overview,
    plot_correlation_heatmap,
    plot_global_vs_target_specific_r2,
    plot_permutation_summary,
    plot_regression_overview,
)
from usability_teleop.viz.inference_figures import (
    plot_inference_bayesian,
    plot_inference_classification_ci,
    plot_inference_pvalues,
    plot_inference_regression_ci,
)


def _tiny_x() -> pd.DataFrame:
    idx = [1, 2, 3, 4, 5, 6]
    return pd.DataFrame(
        {
            "min_ee_quat.x": [1, 2, 3, 4, 5, 6],
            "max_ee_quat.x": [1.2, 2.1, 3.0, 4.2, 5.2, 6.1],
        },
        index=idx,
    )


def test_permutation_workflows_smoke() -> None:
    x = _tiny_x()
    y_reg = pd.DataFrame({"usability": [1, 2, 3, 3, 4, 5]}, index=x.index)
    y_cls = pd.DataFrame({"usability": [1, 2, 3, 3, 4, 5]}, index=x.index)

    fs = [FeatureSetSpec(name="x", axes=("x",), mode="subset")]
    reg_specs = [ModelSpec(name="Ridge", estimator_cls=Ridge, fixed_params={}, param_grid={"alpha": [1.0]})]
    cls_specs = [
        ModelSpec(
            name="RidgeLikeClassifier",
            estimator_cls=LogisticRegression,
            fixed_params={},
            param_grid={"C": [1.0], "penalty": ["l2"], "solver": ["liblinear"]},
        )
    ]

    reg_best = pd.DataFrame(
        {
            "target": ["usability"],
            "feature_set": ["x"],
            "model": ["Ridge"],
            "r2": [0.1],
            "best_params_last_fold": ["{\"alpha\": 1.0}"],
        }
    )
    cls_best = pd.DataFrame(
        {
            "target": ["usability"],
            "feature_set": ["x"],
            "model": ["RidgeLikeClassifier"],
            "threshold": [3.0],
            "status": ["ok"],
            "auc": [0.7],
            "best_params_last_fold": ["{\"C\": 1.0, \"penalty\": \"l2\", \"solver\": \"liblinear\"}"],
        }
    )

    cfg = PermutationConfig(n_permutations=5, alpha=0.05, random_seed=42)
    out_reg = run_regression_permutation_tests(x, y_reg, fs, reg_specs, reg_best, cfg)
    out_cls = run_classification_permutation_tests(x, y_cls, fs, cls_specs, cls_best, cfg)

    assert not out_reg.empty
    assert not out_cls.empty
    assert "p_value" in out_reg.columns
    assert "p_value" in out_cls.columns


def test_figure_builders_smoke(tmp_path: Path) -> None:
    corr = pd.DataFrame(
        {
            "target": ["usability", "usability"],
            "feature": ["f1", "f2"],
            "pearson_r": [0.5, -0.4],
            "pearson_highlight": [True, True],
        }
    )
    reg = pd.DataFrame({"model": ["Ridge"], "feature_set": ["x"], "r2_mean": [0.3]})
    cls = pd.DataFrame(
        {
            "target": ["usability"],
            "status": ["ok"],
            "auc": [0.7],
        }
    )
    perm_reg = pd.DataFrame({"target": ["usability"], "p_value": [0.04]})
    perm_cls = pd.DataFrame({"target": ["usability"], "p_value": [0.06]})
    comparison = pd.DataFrame(
        {
            "target": ["usability"],
            "r2_global": [0.1],
            "r2_specific": [0.3],
            "delta_r2": [0.2],
        }
    )
    inf_reg = pd.DataFrame(
        {
            "target": ["usability"],
            "r2_observed": [0.3],
            "r2_ci_low": [0.1],
            "r2_ci_high": [0.5],
            "paired_p_value": [0.04],
            "paired_p_value_fdr": [0.05],
            "paired_significant_fdr": [True],
            "bayes_prob_improvement": [0.92],
        }
    )
    inf_cls = pd.DataFrame(
        {
            "target": ["usability"],
            "auc_observed": [0.7],
            "auc_ci_low": [0.6],
            "auc_ci_high": [0.8],
            "paired_p_value": [0.06],
            "paired_p_value_fdr": [0.07],
            "paired_significant_fdr": [False],
            "bayes_prob_improvement": [0.74],
        }
    )

    plot_correlation_heatmap(corr, tmp_path / "corr.png", top_n=2)
    plot_regression_overview(reg, tmp_path / "reg.png")
    plot_classification_overview(cls, tmp_path / "cls.png")
    plot_permutation_summary(perm_reg, perm_cls, tmp_path / "perm.png")
    plot_global_vs_target_specific_r2(comparison, tmp_path / "reg_cmp.png")
    plot_inference_regression_ci(inf_reg, tmp_path / "inf_reg_ci.png")
    plot_inference_classification_ci(inf_cls, tmp_path / "inf_cls_ci.png")
    plot_inference_pvalues(inf_reg, inf_cls, tmp_path / "inf_pvalues.png")
    plot_inference_bayesian(inf_reg, inf_cls, tmp_path / "inf_bayes.png")

    assert (tmp_path / "corr.png").exists()
    assert (tmp_path / "reg.png").exists()
    assert (tmp_path / "cls.png").exists()
    assert (tmp_path / "perm.png").exists()
    assert (tmp_path / "reg_cmp.png").exists()
    assert (tmp_path / "inf_reg_ci.png").exists()
    assert (tmp_path / "inf_cls_ci.png").exists()
    assert (tmp_path / "inf_pvalues.png").exists()
    assert (tmp_path / "inf_bayes.png").exists()


def test_inference_bundle_smoke() -> None:
    x = _tiny_x()
    y_reg = pd.DataFrame({"usability": [1, 2, 3, 3, 4, 5]}, index=x.index)
    y_cls = pd.DataFrame({"usability": [1, 2, 3, 3, 4, 5]}, index=x.index)
    fs = [FeatureSetSpec(name="x", axes=("x",), mode="subset")]
    reg_specs = [ModelSpec(name="Ridge", estimator_cls=Ridge, fixed_params={}, param_grid={"alpha": [1.0]})]
    cls_specs = [
        ModelSpec(
            name="RidgeLikeClassifier",
            estimator_cls=LogisticRegression,
            fixed_params={"max_iter": 2000, "C": 1.0, "penalty": "l2", "solver": "liblinear"},
            param_grid={},
        )
    ]
    reg_best = pd.DataFrame({"target": ["usability"], "feature_set": ["x"], "model": ["Ridge"], "r2": [0.1]})
    cls_best = pd.DataFrame(
        {"target": ["usability"], "feature_set": ["x"], "model": ["RidgeLikeClassifier"], "threshold": [3.0], "status": ["ok"], "auc": [0.7]}
    )
    cfg = InferenceBundleConfig(
        baseline_regression_model="Ridge",
        baseline_classification_model="RidgeLikeClassifier",
        bootstrap_iterations=20,
        bayesian_bootstrap_samples=50,
        paired_alpha=0.05,
        fdr_alpha=0.05,
        random_seed=42,
    )
    reg_inf = run_regression_inference(x, y_reg, fs, reg_specs, reg_best, cfg, "r2", 3, True, 42)
    cls_inf = run_classification_inference(x, y_cls, fs, cls_specs, cls_best, cfg, "roc_auc", 3, True, 42)
    assert not reg_inf.empty
    assert not cls_inf.empty
    assert {"paired_p_value", "paired_p_value_fdr", "bayes_prob_improvement"}.issubset(reg_inf.columns)
    assert {"paired_p_value", "paired_p_value_fdr", "bayes_prob_improvement"}.issubset(cls_inf.columns)
