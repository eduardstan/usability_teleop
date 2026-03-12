# Usability Teleop (Clean Pipeline)

This repository is the clean, reproducible implementation track for validating claims in `draft.tex`.

## Current Status
- Unified protocol lanes implemented:
  - Estimation lane (`run-estimation`): nested LOSO only.
  - Final-model lane (`fit-final-models` + `run-final-explainability`).
- Statistical validation lane (`run-stat-validation`).
- Figure assembly lane (`build-figures`) from existing CSVs.

## Repository Layout
- `src/usability_teleop/`: package code (data, features, modeling, stats, viz, cli)
- `configs/`: experiment configurations
- `scripts/`: orchestration scripts
- `data/raw/`: immutable source datasets
- `data/interim/`, `data/processed/`: generated datasets
- `outputs/figures/`, `outputs/tables/`, `outputs/runs/`: reproducible artifacts
- `tests/`: unit/smoke tests

## Environment
Create environment from file:

```bash
conda env create -f environment.yml
conda activate usability_teleop_clean
```

## Install Package

```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

## Sanity Commands

```bash
usability-teleop doctor
usability-teleop validate-data --source-dir data/raw --copy-to-raw
usability-teleop run-estimation --data-dir data/raw --tables-dir outputs/tables --max-models 2 --max-feature-sets 2 --class-balance smote
usability-teleop run-stat-validation --data-dir data/raw --tables-dir outputs/tables --max-models 2 --max-feature-sets 2 --n-permutations 100
usability-teleop fit-final-models --data-dir data/raw --tables-dir outputs/tables --class-balance smote
usability-teleop run-final-explainability --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --experiment-config configs/experiment.yaml --max-targets 5
usability-teleop build-figures --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs
usability-teleop run-ablation --data-dir data/raw --tables-dir outputs/tables --max-models 2 --max-feature-sets 4 --top-k-per-axis 3 --class-balance smote
usability-teleop build-ablation-figures --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs
usability-teleop build-paper-artifacts --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --max-models 2 --max-feature-sets 2
pytest
ruff check .
ruff format --check .
mypy
```

Notes:
- `--max-feature-sets N` limits how many predefined axis-combination configurations are executed.
- This cap is distinct from fold-safe feature selection (`top_k_per_axis`) used inside protocol selection logic.
- `configs/models.yaml` defines model families and hyperparameter grids.
- `configs/experiment.yaml` defines protocol settings (tuning metric, inner-CV behavior, permutation alpha/defaults, SHAP defaults).
- Every run command accepts `--experiment-config path/to/experiment.yaml` to override protocol defaults.
- `--class-balance` is constrained to `none|smote` and only applies to classification paths.
- `run-final-explainability` explains only final refit models; no OOF explainability mode is used.
- `run-stat-validation` computes correlation, permutation, and inference tables (p-values, CI, global-vs-local summary) from estimation outputs.
- `build-figures` only reads `outputs/tables/*.csv` and generates figures; it logs skipped figures when required inputs are missing and writes a report to `outputs/runs/build_figures_report.json`.
- `run-ablation` writes `ablation_summary.csv`, `ablation_breakdown.csv`, `ablation_feature_filter_summary.csv`, and `ablation_target_distributions.csv`.
- `build-ablation-figures` writes ablation figures from existing ablation CSVs and reports to `outputs/runs/build_ablation_figures_report.json`.
- `build-paper-artifacts` is a convenience end-to-end command (data -> tables -> figures -> final models -> SHAP).

## Full Experimental Run (Stage-by-Stage)
Use this when running on workstation/cluster and you want each stage to be independently inspectable.

```bash
conda activate usability_teleop_clean

# 1) Train + evaluate estimation benchmarks
usability-teleop run-estimation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --max-models 2 \
  --max-feature-sets 2 \
  --class-balance smote

# 2) Statistical validation (correlation + permutation + inference)
usability-teleop run-stat-validation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --max-models 2 \
  --max-feature-sets 2 \
  --n-permutations 100 \
  --seed 42

# 3) Refit final models from estimation winners
usability-teleop fit-final-models \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --class-balance smote

# 4) SHAP explainability on final models
usability-teleop run-final-explainability \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --max-targets 5 \
  --seed 42

# 5) Build publication figures from existing tables
usability-teleop build-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs

# 6) Run ablation study (feature-selection + rebalancing factors)
usability-teleop run-ablation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --max-models 2 \
  --max-feature-sets 4 \
  --top-k-per-axis 3 \
  --class-balance smote \
  --seed 42

# 7) Build ablation figures from ablation tables
usability-teleop build-ablation-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs
```

Expected artifacts:
- `outputs/tables/`: estimation, correlation, permutation, inference, final model, and SHAP tables.
- `outputs/figures/`: all publication figures (including correlation heatmap and inference figures).
- `outputs/tables/ablation_*.csv` and `outputs/figures/figure_ablation_*.png`: ablation study tables/figures.
- `outputs/runs/`: run/build reports (for example, `build_figures_report.json`).

## Working Rules
- All production implementation goes under `src/`.
- Every figure/table must be reproducible by command + config.
- Use deterministic seeds and explicit config files for all experiments.
