# Task List (Review Before Implementation)

## Immediate Setup
- [ ] Choose final public repository name and Python package name.
- [ ] Approve proposed folder structure and artifact conventions.
- [ ] Approve dependency policy (strict pinning in conda + optional pip extras).

## Data + Contracts
- [ ] Inventory current data assets and define canonical location in `data/raw`.
- [ ] Write dataset schema specs (features, labels, questionnaire, times).
- [ ] Implement and test ingestion/validation module.

## Core Pipeline
- [ ] Implement deterministic feature builder for ee_quat axis subsets + avg variant.
- [ ] Implement target mapping/inversion policy aligned to draft methods.
- [ ] Implement shared LOSO engine and split integrity tests.

## Experiments
- [ ] Implement correlation workflow (Pearson/Spearman + p-values + thresholds).
- [ ] Implement regression benchmark workflow (global + per-target).
- [ ] Implement classification benchmark workflow (median split, LOSO).
- [ ] Implement hyperparameter search abstraction and model registry.

## Statistical Validation + Explainability
- [ ] Implement permutation testing for regression and classification selections.
- [ ] Implement SHAP analysis workflow for significant models/targets.
- [ ] Verify significance and output consistency with deterministic seeds.

## Visualization System
- [ ] Define single plotting theme file (fonts/colors/grid/spacing/export defaults).
- [ ] Implement figure builders for each paper artifact type.
- [ ] Add output QA checks (resolution, label completeness, naming convention).

## Reproducibility + Release
- [ ] Add CLI entrypoints for each stage and full-run orchestration.
- [ ] Add README reproducibility guide with exact commands.
- [ ] Add tests (unit + smoke end-to-end).
- [ ] Freeze environment and create release checklist.
