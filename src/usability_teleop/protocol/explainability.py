"""Final-model explainability (no OOF mode)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from usability_teleop.features.ee_quat import build_feature_set, generate_ee_quat_feature_sets
from usability_teleop.modeling.registry import build_estimator, regression_model_specs

shap.initjs = lambda *args, **kwargs: None  # noqa: E731


def _compute_shap_values(estimator: Any, x_scaled: np.ndarray) -> np.ndarray:
    """Return SHAP values with a robust fallback for non-callable estimators."""
    try:
        explainer = shap.Explainer(estimator, x_scaled)
        shap_values = explainer(x_scaled)
    except TypeError:
        # Some estimators (for example SVR) are not directly supported as models.
        # Fallback to a prediction-function based explainer.
        explainer = shap.Explainer(estimator.predict, x_scaled)
        shap_values = explainer(x_scaled)
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values


def run_final_explainability(
    x_user: pd.DataFrame,
    y_reg: pd.DataFrame,
    final_models: pd.DataFrame,
    figure_dir: Path,
    max_targets: int,
    seed: int,
) -> pd.DataFrame:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fs_specs = {fs.name: fs for fs in generate_ee_quat_feature_sets(include_average=True)}
    reg_specs = {spec.name: spec for spec in regression_model_specs()}
    rows: list[dict[str, object]] = []
    reg_final = final_models[final_models["track"] == "regression"].head(max_targets)
    for _, row in reg_final.iterrows():
        target = str(row["target"])
        fs_name = str(row["feature_set"])
        model_name = str(row["model"])
        if target not in y_reg.columns:
            raise ValueError(f"Unknown regression target in final_models: {target}")
        if fs_name not in fs_specs:
            raise ValueError(f"Unknown feature_set in final_models: {fs_name}")
        if model_name not in reg_specs:
            raise ValueError(f"Unknown regression model in final_models: {model_name}")
        params = _json_to_dict(str(row["final_params"]))
        selected = _json_to_list(str(row["selected_features"]))

        fs = fs_specs[fs_name]
        spec = reg_specs[model_name]
        x_full = build_feature_set(x_user, fs)
        missing_selected = [col for col in selected if col not in x_full.columns]
        if missing_selected:
            raise ValueError(
                "final_models selected_features contains columns not present in "
                f"feature_set '{fs_name}': {missing_selected}"
            )
        x_fs = x_full[selected].copy()
        y = y_reg[target].to_numpy(dtype=float)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_fs.to_numpy(dtype=float))
        estimator = build_estimator(spec, seed)
        if params:
            estimator.set_params(**params)
        estimator.fit(x_scaled, y)

        values = _compute_shap_values(estimator, x_scaled)
        mean_abs = np.mean(np.abs(values), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        for idx in top_idx:
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature_set": fs_name,
                    "feature": x_fs.columns[idx],
                    "mean_abs_shap": float(mean_abs[idx]),
                }
            )
        fig_path = figure_dir / f"figure_final_shap_{target}_{model_name}_{fs_name}.png"
        plt.figure(figsize=(8, 5))
        shap.summary_plot(values, x_scaled, feature_names=list(x_fs.columns), show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
    if not rows:
        return pd.DataFrame(columns=["target", "model", "feature_set", "feature", "mean_abs_shap"])
    return pd.DataFrame(rows).sort_values(["target", "mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)


def _json_to_dict(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw) if raw else {}
    return {str(k): v for k, v in parsed.items()}


def _json_to_list(raw: str) -> list[str]:
    parsed = json.loads(raw) if raw else []
    return [str(v) for v in parsed]
