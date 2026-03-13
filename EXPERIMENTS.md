# EXPERIMENTS.md

Last updated: 2026-03-13

This document is the reproducible runbook for generating all paper-ready artifacts
(tables, figures, statistical validation outputs, final models, explainability, and ablation results).

## 1) Preconditions

- Repository root is the working directory.
- Raw data files are available in `data/raw/` according to `DATA_CONTRACTS.md`.
- Python `3.12.x` is available.
- Environment is created either with `venv` (recommended when conda is unavailable) or conda.

`venv` setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e .[dev]
```

Conda setup:

```bash
conda env create -f environment.yml
conda activate usability_teleop_clean
python -m pip install -e .
python -m pip install -e .[dev]
```

Sanity checks:

```bash
usability-teleop doctor
usability-teleop validate-data --source-dir data/raw --copy-to-raw
pytest -q
```

If using `venv`, ensure `source .venv/bin/activate` is done before running any `usability-teleop` command.

## 2) Configuration Profiles

- `configs/models_fast.yaml`: smoke/dev profile (small search space).
- `configs/models_full.yaml`: paper-grade profile (expanded search space).
- `configs/experiment.yaml`: protocol controls (CV, permutation, inference, SHAP defaults).

## 3) Output Directories

All commands below write to:
- `outputs/tables`
- `outputs/figures`
- `outputs/runs`

Create/clean as desired:

```bash
mkdir -p outputs/tables outputs/figures outputs/runs
```

## 4) Quick Smoke Run (Recommended First)

Use this to validate end-to-end execution quickly before launching full experiments.

```bash
conda activate usability_teleop_clean

PROFILE=configs/models_fast.yaml
SEED=42

usability-teleop run-estimation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config "$PROFILE" \
  --num-workers 2 \
  --max-models 2 \
  --max-feature-sets 2 \
  --top-k-per-axis 2 \
  --seed "$SEED"

usability-teleop run-stat-validation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config "$PROFILE" \
  --num-workers 2 \
  --max-models 2 \
  --max-feature-sets 2 \
  --n-permutations 20 \
  --seed "$SEED"

usability-teleop fit-final-models \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config "$PROFILE" \
  --top-k-per-axis 2 \
  --seed "$SEED"

usability-teleop run-final-explainability \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --experiment-config configs/experiment.yaml \
  --max-targets 5 \
  --seed "$SEED"

usability-teleop build-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs

usability-teleop build-ablation-artifacts \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs \
  --models-config "$PROFILE" \
  --max-models 2 \
  --max-feature-sets 2 \
  --num-workers 2 \
  --top-k-per-axis 1,2,3 \
  --seed "$SEED"
```

## 5) Full Paper Run (Stage-by-Stage)

This is the recommended production run for paper artifacts.

```bash
conda activate usability_teleop_clean

PROFILE=configs/models_full.yaml
SEED=42

# Stage 1: Estimation
usability-teleop run-estimation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config "$PROFILE" \
  --num-workers 24 \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 3 \
  --seed "$SEED"

# Stage 2: Statistical validation (correlation + permutation + inference + global-vs-target)
usability-teleop run-stat-validation \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config "$PROFILE" \
  --num-workers 24 \
  --max-models 10 \
  --max-feature-sets 16 \
  --n-permutations 1000 \
  --seed "$SEED"

# Stage 3: Final refit from winners
usability-teleop fit-final-models \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --models-config "$PROFILE" \
  --top-k-per-axis 3 \
  --seed "$SEED"

# Stage 4: Explainability
usability-teleop run-final-explainability \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --experiment-config configs/experiment.yaml \
  --max-targets 5 \
  --seed "$SEED"

# Stage 5: Protocol figures from existing tables
usability-teleop build-figures \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs

# Stage 6: Feature-selection ablation sweep + ablation figures
usability-teleop build-ablation-artifacts \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --runs-dir outputs/runs \
  --models-config "$PROFILE" \
  --max-models 10 \
  --max-feature-sets 16 \
  --num-workers 24 \
  --top-k-per-axis 1,2,3,4,5,6,7,8,9,10 \
  --seed "$SEED"
```

## 6) One-Command Convenience Alternative

If you want a single command for core pipeline (not including ablation stages):

```bash
usability-teleop build-paper-artifacts \
  --data-dir data/raw \
  --tables-dir outputs/tables \
  --figures-dir outputs/figures \
  --models-config configs/models_full.yaml \
  --num-workers 24 \
  --max-models 10 \
  --max-feature-sets 16 \
  --top-k-per-axis 3 \
  --n-permutations 1000 \
  --seed 42
```

Then run ablation explicitly:

```bash
usability-teleop build-ablation-artifacts --data-dir data/raw --tables-dir outputs/tables --figures-dir outputs/figures --runs-dir outputs/runs --models-config configs/models_full.yaml --max-models 10 --max-feature-sets 16 --num-workers 24 --top-k-per-axis 1,2,3,4,5,6,7,8,9,10 --seed 42
```

Notes:
- If `--max-models` is omitted, all models defined in the selected models YAML are evaluated.
- If `--max-feature-sets` is omitted, all generated feature-set definitions are evaluated.
- `--num-workers` is available on `run-estimation`, `run-stat-validation`, `build-paper-artifacts`, `run-ablation`, and `build-ablation-artifacts`.
- `run-ablation` / `build-ablation-artifacts` do not expose `--n-permutations`: ablation scope is fold-safe feature-selection sensitivity, while permutation inference is handled by `run-stat-validation` / `build-paper-artifacts`.

## 7) Expected Artifacts

### Core Tables (`outputs/tables`)
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

### Ablation Tables (`outputs/tables`)
- `ablation_summary.csv`
- `ablation_breakdown.csv`
- `ablation_feature_filter_summary.csv`
- `ablation_target_distributions.csv`

### Core Figures (`outputs/figures`)
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

### Ablation Figures (`outputs/figures`)
- `figure_ablation_stage_summary.png`
- `figure_ablation_delta_heatmap.png`
- `figure_ablation_target_distributions.png`

### Run Metadata (`outputs/runs`)
- `build_figures_report.json`
- `build_ablation_figures_report.json`
- `run_manifest_<command>_<timestamp>.json`
- `run_manifest_<command>_latest.json`

Each run manifest includes:
- command name and args
- start/end UTC timestamps
- elapsed seconds
- git commit hash
- status (`ok`/`error`)
- output paths
- error details (if any)
- timestamped manifests are immutable history snapshots
- `*_latest.json` manifests are rolling pointers to latest command runs

## 8) Minimal Verification Commands

```bash
ls -1 outputs/tables | sort
ls -1 outputs/figures | sort
cat outputs/runs/build_figures_report.json
cat outputs/runs/build_ablation_figures_report.json
ls -1 outputs/runs/run_manifest_* | sort
```

## 9) Notes for Cluster Use

- Keep stage boundaries: each stage is restartable independently by rerunning the failed stage.
- Prefer writing stdout/stderr to per-stage log files.
- Start with a smoke run using `models_fast.yaml` before full profile.
- For full runs, ensure enough walltime for `--n-permutations 1000` and expanded `models_full.yaml` grids.

Example detached execution pattern:

```bash
nohup bash -lc 'conda activate usability_teleop_clean && usability-teleop run-estimation --data-dir data/raw --tables-dir outputs/tables --models-config configs/models_full.yaml --max-models 10 --max-feature-sets 16 --top-k-per-axis 3 --seed 42' > outputs/runs/log_estimation.txt 2>&1 &
```
