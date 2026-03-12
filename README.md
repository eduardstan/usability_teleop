# Usability Teleop (Clean Pipeline)

This repository is the clean, reproducible implementation track for validating claims in `draft.tex`.

## Current Status
- Unified protocol lanes implemented:
  - Estimation lane (`run-estimation`): nested LOSO only.
  - Final-model lane (`fit-final-models` + `run-final-explainability`).
- Unified orchestrator available: `run-paper-pipeline`.

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
usability-teleop run-estimation --data-dir data/raw --tables-dir outputs/tables --max-models 2 --max-feature-sets 2 --class-balance none
usability-teleop run-stat-validation --data-dir data/raw --tables-dir outputs/tables --max-models 2 --max-feature-sets 2 --n-permutations 100
usability-teleop fit-final-models --data-dir data/raw --tables-dir outputs/tables
usability-teleop run-final-explainability --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --experiment-config configs/experiment.yaml --max-targets 5
usability-teleop build-paper-artifacts --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --max-models 2 --max-feature-sets 2
usability-teleop run-paper-pipeline --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --max-models 2 --max-feature-sets 2
pytest
ruff check .
ruff format --check .
mypy
```

Notes:
- `--max-feature-sets N` limits how many predefined axis-combination configurations are executed.
- It is not statistical feature selection; it is a run-scope cap for faster iteration.
- `--workers K` parallelizes Stage 2 outer tasks (feature-set/model combinations); ETA remains based on completed tasks.
- `configs/models.yaml` defines model families and hyperparameter grids.
- `configs/experiment.yaml` defines protocol settings (tuning metric, inner-CV behavior, permutation alpha/defaults, SHAP defaults).
- Every run command accepts `--experiment-config path/to/experiment.yaml` to override protocol defaults.
- `run-final-explainability` explains only final refit models; no OOF explainability mode is used.
- `run-stat-validation` computes permutation and inference tables (p-values, CI, global-vs-local summary) from estimation outputs.
- `build-paper-artifacts` is the primary unified end-to-end command for publication tables and figures.
- `run-paper-pipeline` is a compatibility alias to `build-paper-artifacts`.

## Working Rules
- All production implementation goes under `src/`.
- Every figure/table must be reproducible by command + config.
- Use deterministic seeds and explicit config files for all experiments.
