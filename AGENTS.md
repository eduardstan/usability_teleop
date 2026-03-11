# AGENTS.md

## Mission
Build a clean, reproducible, publication-grade research codebase for testing the claims in `draft.tex`, replacing legacy scripts with a modular pipeline suitable for open release.

## Scope for This Repository
- Source of truth for experiments and figures used in the paper.
- Data required for experiments must be standardized into a documented `data/` layout.
- Outputs must be deterministic, reproducible, and easy to regenerate end-to-end.

## Non-Negotiable Engineering Standards
- Reproducibility first: fixed seeds, explicit configs, version-pinned dependencies, deterministic splits.
- Clean architecture: small composable modules, no notebook-only logic, minimal side effects.
- Traceability: every figure/table must map to a command, config, and artifact hash/run id.
- Validation: tests for data loading, feature generation, CV split integrity, and metric computation.
- Publication quality visuals: unified style system so all figures look authored by the same hand.

## Required Pipeline Capabilities (Aligned to draft.tex)
1. Data ingestion + schema checks (features, labels, questionnaire, demographics, timing).
2. Feature engineering:
   - user-level aggregation
   - end-effector orientation feature subsets (x/y/z/w combinations + axis-average variant)
   - target encoding and optional inversion logic only for correlation stage.
3. Stage 1: correlation analysis (Pearson/Spearman, p-values, effect-size thresholding).
4. Stage 2: regression benchmark:
   - 10 model families
   - LOSO outer CV
   - inner CV tuning
   - global multi-output + target-specific settings
   - RMSE/MAE/R2.
5. Stage 3: binary classification benchmark:
   - median split per target
   - LOSO + inner tuning
   - Accuracy/Balanced Accuracy/F1/AUC.
6. Statistical validation:
   - permutation tests (regression + classification)
   - p-values and significance reporting.
7. Interpretability:
   - SHAP for best statistically significant models.
8. Artifact generation:
   - publication-ready figures + tables
   - machine-readable result files.

## Initial Target Repository Layout
```text
.
├── AGENTS.md
├── IMPLEMENTATION_PLAN.md
├── TASK_LIST.md
├── README.md
├── environment.yml
├── pyproject.toml
├── src/usability_teleop/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── modeling/
│   ├── evaluation/
│   ├── stats/
│   ├── viz/
│   └── cli/
├── scripts/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── runs/
└── tests/
```

## Execution Rules
- Implement production code under `src/`.
- No hidden manual steps for reproducing results.
- Every experiment must run via CLI command + config file.
- Prefer typed Python, docstrings for non-trivial APIs, and lint/format checks.
- Keep all plotting defaults centralized in one style module.

## Decision Log (Initialize)
- Repository naming will be finalized before first public release; code package naming can be updated once chosen.
- Plan and tasks are tracked in:
  - `IMPLEMENTATION_PLAN.md`
  - `TASK_LIST.md`
