# Usability Teleop

Clean, reproducible, publication-grade pipeline for validating the claims in `draft.tex`.

This repository is the experiment source of truth:
- data ingestion and validation,
- feature construction and fold-safe selection,
- LOSO estimation,
- statistical validation (correlation/permutation/inference),
- final refit and explainability,
- publication-ready figures and tables,
- ablation study outputs.

## 1) Repository Structure

```text
.
├── AGENTS.md
├── DATA_CONTRACTS.md
├── DEV_HISTORY.md
├── IMPLEMENTATION_PLAN.md
├── TASK_LIST.md
├── README.md
├── configs/
│   ├── default.yaml
│   ├── experiment.yaml
│   ├── models.yaml
│   ├── models_fast.yaml
│   └── models_full.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── runs/
├── src/usability_teleop/
└── tests/
```

## 2) Environment Setup

```bash
conda env create -f environment.yml
conda activate usability_teleop_clean
python -m pip install -e .
python -m pip install -e .[dev]
```

Sanity checks:

```bash
usability-teleop doctor
pytest -q
```

## 3) Data Inputs

Expected raw inputs are validated from `data/raw/` (see `DATA_CONTRACTS.md`):
- `raw_features_full.csv`
- `labels_full.csv`
- `User_risposte.xlsx`
- `tempi_media.csv`

Validate and normalize:

```bash
usability-teleop validate-data --source-dir data/raw --copy-to-raw
```

## 4) Active CLI Commands

### Core Protocol
- `run-estimation`
- `run-stat-validation`
- `fit-final-models`
- `run-final-explainability`
- `build-figures`
- `build-paper-artifacts`

### Ablation
- `run-ablation`
- `build-ablation-figures`

### Utilities
- `doctor`
- `validate-data`

## 5) Key Configuration Knobs

### Model profile (`--models-config`)
- `configs/models_fast.yaml`: faster smoke/development runs.
- `configs/models_full.yaml`: larger paper-grade search profile.
- `configs/models.yaml`: baseline/default profile.

### Experiment protocol (`--experiment-config`)
Use `configs/experiment.yaml` (or custom path) for:
- scoring,
- inner CV settings,
- permutation defaults,
- inference defaults,
- SHAP defaults.

### Class balancing (`--class-balance`)
Allowed values: `none|smote`.
- applied only to classification paths,
- never applied to regression.

### Feature-selection knob
- `--top-k-per-axis` controls fold-safe variance-based screening per quaternion axis.
- `--max-feature-sets` only caps run breadth (not statistical selection).

## 6) Stage-by-Stage Reproducible Pipeline

Use this sequence for transparent, resume-by-stage execution.

```bash
conda activate usability_teleop_clean

# 1) Estimation (nested LOSO)
usability-teleop run-estimation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config configs/models_full.yaml \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 3 \
  --class-balance smote \
  --seed 42

# 2) Statistical validation (correlation + permutation + inference)
usability-teleop run-stat-validation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config configs/models_full.yaml \
  --max-models 10 \
  --max-feature-sets 16 \
  --n-permutations 1000 \
  --seed 42

# 3) Final refit from winners
usability-teleop fit-final-models \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config configs/models_full.yaml \
  --top-k-per-axis 3 \
  --class-balance smote \
  --seed 42

# 4) Explainability from final models
usability-teleop run-final-explainability \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --experiment-config configs/experiment.yaml \
  --max-targets 5 \
  --seed 42

# 5) Build protocol figures from existing CSV tables
usability-teleop build-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs

# 6) Ablation tables
usability-teleop run-ablation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config configs/models_full.yaml \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 3 \
  --class-balance smote \
  --seed 42

# 7) Ablation figures
usability-teleop build-ablation-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs
```

## 7) One-Command Convenience Run

```bash
usability-teleop build-paper-artifacts \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --models-config configs/models_full.yaml \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 3 \
  --class-balance smote \
  --n-permutations 1000 \
  --seed 42
```

## 8) Output Contracts

### Core tables (`outputs/tables/`)
- `correlation_results.csv`
- `estimation_regression.csv`
- `estimation_classification.csv`
- `estimation_best_configs.csv`
- `permutation_regression_results.csv`
- `permutation_classification_results.csv`
- `inference_regression.csv`
- `inference_classification.csv`
- `regression_best_global_vs_target_specific.csv`
- `classification_best_global_vs_target_specific.csv`
- `final_models.csv`
- `final_explainability_shap.csv`

### Ablation tables (`outputs/tables/`)
- `ablation_summary.csv`
- `ablation_breakdown.csv`
- `ablation_feature_filter_summary.csv`
- `ablation_target_distributions.csv`

### Core figures (`outputs/figures/`)
- `figure_correlation_heatmap.png`
- `figure_regression_overview.png`
- `figure_classification_overview.png`
- `figure_permutation_pvalues.png`
- `figure_regression_global_vs_target_specific.png`
- `figure_classification_global_vs_target_specific.png`
- `figure_inference_regression_ci.png`
- `figure_inference_classification_ci.png`
- `figure_inference_pvalues.png`
- `figure_inference_bayesian.png`
- `figure_protocol_dashboard.png`

### Ablation figures (`outputs/figures/`)
- `figure_ablation_stage_summary.png`
- `figure_ablation_delta_heatmap.png`
- `figure_ablation_target_distributions.png`

### Run metadata (`outputs/runs/`)
- `build_figures_report.json`
- `build_ablation_figures_report.json`

## 9) Cluster Execution Pattern

Recommended pattern:
1. Run each stage as an independent job (estimation, stat-validation, final-fit, explainability, figure build, ablation).
2. Keep shared storage paths fixed (`outputs/tables`, `outputs/figures`, `outputs/runs`).
3. Re-run only failed or updated stages; downstream stages consume existing CSV artifacts.

Example shell skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

conda activate usability_teleop_clean

PROFILE=configs/models_full.yaml
SEED=42

usability-teleop run-estimation --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --class-balance smote --seed "$SEED"
usability-teleop run-stat-validation --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --n-permutations 1000 --seed "$SEED"
usability-teleop fit-final-models --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --class-balance smote --seed "$SEED"
usability-teleop run-final-explainability --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --seed "$SEED"
usability-teleop build-figures --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs
usability-teleop run-ablation --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --class-balance smote --seed "$SEED"
usability-teleop build-ablation-figures --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs
```

## 10) Testing and Quality Gates

Run before commit/merge:

```bash
pytest -q
ruff check .
ruff format --check .
mypy
```

## 11) Deprecation Policy

- No legacy/versioned planning files (`*_v*.md`) are maintained.
- Historical milestones are centralized in `DEV_HISTORY.md`.
- Canonical planning docs are `IMPLEMENTATION_PLAN.md` and `TASK_LIST.md` only.

## 12) Troubleshooting

- Missing figure/table input: rerun the upstream stage; figure builders intentionally skip missing dependencies with warnings.
- Unexpected parser rejection for balancing mode: only `none|smote` is allowed.
- Slow iteration: use `configs/models_fast.yaml`, lower `--max-models`, `--max-feature-sets`, and `--n-permutations`.
