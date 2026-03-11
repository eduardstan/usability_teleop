"""SHAP explainability workflow for selected regression targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import FeatureSetSpec, build_feature_set
from usability_teleop.modeling.registry import ModelSpec, build_estimator

shap.initjs = lambda *args, **kwargs: None  # noqa: E731


def _spec_by_name(specs: list[ModelSpec], model_name: str) -> ModelSpec:
    for spec in specs:
        if spec.name == model_name:
            return spec
    raise ValueError(f"Unknown model name: {model_name}")


def _feature_set_by_name(specs: list[FeatureSetSpec], name: str) -> FeatureSetSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown feature set: {name}")


def _params_from_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    clean: dict[str, Any] = {}
    for key, value in parsed.items():
        clean[key.replace("estimator__", "")] = value
    return clean


def run_regression_shap(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
    model_specs: list[ModelSpec],
    regression_target_results: pd.DataFrame,
    permutation_results: pd.DataFrame | None,
    figure_dir: Path,
    max_targets: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute SHAP summary importance for selected best per-target regression models."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    candidates = regression_target_results.copy()
    if permutation_results is not None and not permutation_results.empty:
        sig_targets = permutation_results[permutation_results["significant"]]["target"].tolist()
        candidates = candidates[candidates["target"].isin(sig_targets)]

    selected = (
        candidates.sort_values("r2", ascending=False)
        .groupby("target", as_index=False)
        .head(1)
        .head(max_targets)
    )

    rows: list[dict[str, object]] = []

    for _, row in selected.iterrows():
        target = str(row["target"])
        model_name = str(row["model"])
        feature_name = str(row["feature_set"])
        params = _params_from_json(str(row.get("best_params_last_fold", "{}")))

        spec = _spec_by_name(model_specs, model_name)
        fs = _feature_set_by_name(feature_sets, feature_name)

        x_fs = build_feature_set(x_user, fs)
        y = y_reg[target].to_numpy(dtype=float)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_fs.to_numpy(dtype=float))

        estimator = build_estimator(spec, random_seed)
        if params:
            estimator.set_params(**params)
        estimator.fit(x_scaled, y)

        explainer = shap.Explainer(estimator, x_scaled)
        shap_values = explainer(x_scaled)
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, 0]

        mean_abs = np.mean(np.abs(values), axis=0)
        feature_names = list(x_fs.columns)

        top_idx = np.argsort(mean_abs)[::-1][:10]
        for idx in top_idx:
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature_set": feature_name,
                    "feature": feature_names[idx],
                    "mean_abs_shap": float(mean_abs[idx]),
                }
            )

        fig_path = figure_dir / f"shap_summary_{target}_{model_name}_{feature_name}.png"
        plt.figure(figsize=(8, 5))
        shap.summary_plot(values, features=x_scaled, feature_names=feature_names, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

    if not rows:
        return pd.DataFrame(
            columns=["target", "model", "feature_set", "feature", "mean_abs_shap"]
        )

    return pd.DataFrame(rows).sort_values(
        ["target", "mean_abs_shap"], ascending=[True, False]
    ).reset_index(drop=True)
