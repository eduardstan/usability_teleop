"""Visualization utilities for publication-quality artifact generation."""

from usability_teleop.viz.figures import (
    plot_classification_overview,
    plot_correlation_heatmap,
    plot_global_vs_target_specific_r2,
    plot_permutation_summary,
    plot_protocol_dashboard,
    plot_regression_overview,
)
from usability_teleop.viz.inference_figures import (
    plot_inference_bayesian,
    plot_inference_classification_ci,
    plot_inference_pvalues,
    plot_inference_regression_ci,
)
from usability_teleop.viz.study_figures import (
    plot_study_delta_heatmap,
    plot_study_stage_summary,
    plot_study_target_distributions,
)
from usability_teleop.viz.theme import apply_publication_theme

__all__ = [
    "apply_publication_theme",
    "plot_correlation_heatmap",
    "plot_regression_overview",
    "plot_global_vs_target_specific_r2",
    "plot_classification_overview",
    "plot_permutation_summary",
    "plot_protocol_dashboard",
    "plot_inference_regression_ci",
    "plot_inference_classification_ci",
    "plot_inference_pvalues",
    "plot_inference_bayesian",
    "plot_study_stage_summary",
    "plot_study_delta_heatmap",
    "plot_study_target_distributions",
]
