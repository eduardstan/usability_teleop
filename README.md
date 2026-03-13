# Usability Teleop

Last updated: 2026-03-13

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

Python baseline: `3.12.x`

### Option A: `venv` (recommended when conda is unavailable)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e .[dev]
```

### Option B: conda

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

## 5) CLI API Reference

This section documents the active API surface and exact parameter semantics.

### Global default semantics
- `--models-config`:
  - unset: use `configs/models.yaml` (registry default).
  - set: load the specified model profile YAML.
- `--max-models`:
  - unset (`None`): use all models defined in the selected `models-config` YAML.
  - set (`N`): keep first `N` model families per task from the selected YAML.
- `--max-feature-sets`:
  - unset (`None`): use all generated feature-set definitions.
  - set (`N`): keep first `N` feature-set definitions.
- `--n-permutations`:
  - unset (`None`): use the default from `--experiment-config` (`configs/experiment.yaml` by default).
- `--top-k-per-axis`:
  - for estimation/final-fit/full-pipeline: single integer `k` or unset (`None`) for no fold-safe top-k screening.
  - for ablation: comma-separated list (for example `1,2,3,5`), each value defines one ablation stage.
- `--num-workers`:
  - currently exposed on `run-ablation`.
  - `1` means deterministic sequential stage execution.
  - `>1` runs ablation stages in parallel workers.

### `doctor`
- Runs environment/project sanity checks.
- Parameters: none.

### `validate-data`
- Purpose: strict raw-data contract validation (+ optional canonical copy).
- Parameters:
  - `--source-dir` (default: `data/raw`): input directory to validate.
  - `--copy-to-raw` (flag): if set, copy validated artifacts to canonical `data/raw`.

### `run-estimation`
- Purpose: nested LOSO estimation for regression + classification tracks.
- Parameters:
  - `--data-dir` (default: `data/raw`)
  - `--tables-dir` (default: `outputs/tables`)
  - `--runs-dir` (default: `outputs/runs`)
  - `--seed` (default: `42`)
  - `--experiment-config` (default: unset; resolves to `configs/experiment.yaml`)
  - `--models-config` (default: unset; resolves to `configs/models.yaml`)
  - `--max-models` (default: unset; all models from YAML)
  - `--max-feature-sets` (default: unset; all feature sets)
  - `--top-k-per-axis` (default: unset; no fold-safe top-k screening)

### `run-stat-validation`
- Purpose: correlation + permutation tests + inference + global-vs-target tables.
- Parameters:
  - `--data-dir` (default: `data/raw`)
  - `--tables-dir` (default: `outputs/tables`)
  - `--runs-dir` (default: `outputs/runs`)
  - `--seed` (default: `42`)
  - `--experiment-config` (default: unset; resolves to `configs/experiment.yaml`)
  - `--models-config` (default: unset; resolves to `configs/models.yaml`)
  - `--max-models` (default: unset; all models from YAML)
  - `--max-feature-sets` (default: unset; all feature sets)
  - `--n-permutations` (default: unset; uses experiment config default)
  - `--nested-permutation` (flag): enable nested permutation mode.

### `fit-final-models`
- Purpose: fit final models from `estimation_best_configs.csv`.
- Parameters:
  - `--data-dir` (default: `data/raw`)
  - `--tables-dir` (default: `outputs/tables`)
  - `--runs-dir` (default: `outputs/runs`)
  - `--seed` (default: `42`)
  - `--experiment-config` (default: unset; resolves to `configs/experiment.yaml`)
  - `--models-config` (default: unset; resolves to `configs/models.yaml`)
  - `--top-k-per-axis` (default: unset; no fold-safe top-k screening)

### `run-final-explainability`
- Purpose: SHAP explainability using `final_models.csv`.
- Parameters:
  - `--data-dir` (default: `data/raw`)
  - `--tables-dir` (default: `outputs/tables`)
  - `--figures-dir` (default: `outputs/figures`)
  - `--runs-dir` (default: `outputs/runs`)
  - `--seed` (default: `42`)
  - `--experiment-config` (default: unset; resolves to `configs/experiment.yaml`)
  - `--max-targets` (default: unset; uses experiment config SHAP default)

### `build-figures`
- Purpose: build protocol figures from existing CSV tables only (no recompute).
- Parameters:
  - `--tables-dir` (default: `outputs/tables`)
  - `--figures-dir` (default: `outputs/figures`)
  - `--runs-dir` (default: `outputs/runs`)

### `build-paper-artifacts`
- Purpose: one-command full protocol run (tables + figures).
- Parameters:
  - `--data-dir` (default: `data/raw`)
  - `--tables-dir` (default: `outputs/tables`)
  - `--figures-dir` (default: `outputs/figures`)
  - `--runs-dir` (default: `outputs/runs`)
  - `--seed` (default: `42`)
  - `--experiment-config` (default: unset; resolves to `configs/experiment.yaml`)
  - `--models-config` (default: unset; resolves to `configs/models.yaml`)
  - `--max-models` (default: unset; all models from YAML)
  - `--max-feature-sets` (default: unset; all feature sets)
  - `--top-k-per-axis` (default: unset; no fold-safe top-k screening)
  - `--max-targets` (default: `5`)
  - `--alpha` (default: `0.05`)
  - `--effect-threshold` (default: `0.30`)
  - `--n-permutations` (default: unset; uses experiment config default)
  - `--nested-permutation` (flag): enable nested permutation mode.

### `run-ablation`
- Purpose: ablation tables for baseline vs fold-safe top-k-per-axis filtering stages.
- Parameters:
  - `--data-dir` (default: `data/raw`)
  - `--tables-dir` (default: `outputs/tables`)
  - `--runs-dir` (default: `outputs/runs`)
  - `--seed` (default: `42`)
  - `--experiment-config` (default: unset; resolves to `configs/experiment.yaml`)
  - `--models-config` (default: unset; resolves to `configs/models.yaml`)
  - `--max-models` (default: unset; all models from YAML)
  - `--max-feature-sets` (default: unset; all feature sets)
  - `--num-workers` (default: `1`; parallel ablation stage workers)
  - `--top-k-per-axis` (default: `1,2,3,5`; comma-separated ablation stage values)

### `build-ablation-figures`
- Purpose: ablation figures from existing ablation CSV tables only.
- Parameters:
  - `--tables-dir` (default: `outputs/tables`)
  - `--figures-dir` (default: `outputs/figures`)
  - `--runs-dir` (default: `outputs/runs`)

## 6) Key Configuration Knobs

### Model profile (`--models-config`)
- `configs/models_fast.yaml`: faster smoke/development runs.
- `configs/models_full.yaml`: expanded paper-grade search profile.
- `configs/models.yaml`: baseline/default profile.

### Experiment protocol (`--experiment-config`)
Use `configs/experiment.yaml` (or custom path) for:
- scoring,
- inner CV settings,
- permutation defaults,
- inference defaults,
- SHAP defaults.

### Class balancing
`--class-balance` is deprecated and removed from active CLI commands.
- current protocol runs with balancing disabled (`none`) for both estimation and ablation.
- rationale: avoid asymmetry with regression and keep current comparisons methodologically aligned.
- future work can reintroduce balancing as an explicit, isolated experiment axis.

### Feature-selection knob
- `--top-k-per-axis` controls fold-safe variance-based screening per quaternion axis.
- `--max-feature-sets` only caps run breadth (not statistical selection).
- `run-ablation` varies fold-safe selection via `--top-k-per-axis` as a comma list (for example `1,2,3,5,8`).

## 7) Stage-by-Stage Reproducible Pipeline

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
  --num-workers 4 \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 1,2,3,5,8 \
  --seed 42

# 7) Ablation figures
usability-teleop build-ablation-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs
```

## 8) One-Command Convenience Run

```bash
usability-teleop build-paper-artifacts \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --models-config configs/models_full.yaml \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 3 \
  --n-permutations 1000 \
  --seed 42
```

## 9) Output Contracts

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
- `run_manifest_<command>_<timestamp>.json`
- `run_manifest_<command>_latest.json`

Each `run_manifest_*` stores command args, UTC start/end timestamps, elapsed seconds, git commit hash, status, output paths, and error details.
- `run_manifest_<command>_<timestamp>.json` is immutable run history.
- `run_manifest_<command>_latest.json` is overwritten by the most recent run of that command.

## 10) Cluster Execution Pattern

Recommended pattern:
1. Run each stage as an independent job (estimation, stat-validation, final-fit, explainability, figure build, ablation).
2. Keep shared storage paths fixed (`outputs/tables`, `outputs/figures`, `outputs/runs`).
3. Re-run only failed or updated stages; downstream stages consume existing CSV artifacts.
4. Capture per-command runtime metadata from `outputs/runs/run_manifest_*` for paper reporting.

Example shell skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

conda activate usability_teleop_clean

PROFILE=configs/models_full.yaml
SEED=42

usability-teleop run-estimation --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --seed "$SEED"
usability-teleop run-stat-validation --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --n-permutations 1000 --seed "$SEED"
usability-teleop fit-final-models --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --seed "$SEED"
usability-teleop run-final-explainability --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --seed "$SEED"
usability-teleop build-figures --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs
usability-teleop run-ablation --data-dir data/raw --tables-dir outputs/tables --models-config "$PROFILE" --top-k-per-axis 1,2,3,5,8 --seed "$SEED"
usability-teleop build-ablation-figures --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs
```

## 11) Testing and Quality Gates

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
- Unexpected parser rejection for balancing mode: this flag is currently deprecated and unavailable.
- Slow iteration: use `configs/models_fast.yaml`, lower `--max-models`, `--max-feature-sets`, and `--n-permutations`.
