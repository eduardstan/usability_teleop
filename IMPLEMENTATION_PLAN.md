# Implementation Plan

## Goal
Deliver a clean, reproducible, release-ready research pipeline that can regenerate all paper claims and visuals from raw project data.

## Phase 0 - Foundation (Repo hygiene + environment)
- Define project conventions, directory structure, and coding standards.
- Create pinned conda environment and export lockfile snapshot.
- Add base tooling: formatter, linter, test runner, type checks, pre-commit.
- Implement and standardize a project-owned logger with pretty console output for all CLI/pipeline stages.
- Write README with one-command bootstrap and run instructions.

## Phase 1 - Data Contracts and Ingestion
- Standardize usable inputs into canonical `data/raw`.
- Define explicit schemas for each dataset (columns, dtypes, constraints).
- Implement ingestion module with strict validation + informative failures.
- Add unit tests for schema validation and row/participant alignment.

## Phase 2 - Feature/Target Pipeline
- Re-implement feature matrix builder from scratch (no notebook carry-over).
- Implement axis-combination generator for `ee_quat.{x,y,z,w}` + avg variant.
- Implement questionnaire mapping and target preparation with stage-specific inversion policy.
- Add tests for deterministic feature set generation and target mapping.

## Phase 3 - Modeling Core
- Build shared LOSO engine with strict train/test isolation.
- Implement model registry for regression and classification families.
- Add inner CV hyperparameter search abstraction.
- Provide deterministic outputs with fixed seed plumbing.

## Phase 4 - Experiment Tracks (RQ-aligned)
- Correlation track (RQ1): Pearson/Spearman matrix + significance filtering outputs.
- Regression track (RQ2): global + target-specific benchmarks, metrics tables.
- Classification track (RQ3): median-split modeling with accuracy/F1/AUC suite.

## Phase 5 - Statistical Validation and Explainability
- Implement permutation test module for both regression and classification best models.
- Add SHAP pipeline for significant targets with model-specific explainers.
- Produce structured outputs for both numeric stats and corresponding figures.

## Phase 6 - Publication-Quality Visualization System
- Define a global plotting theme (fonts, color palette, spacing, annotation style).
- Rebuild all core paper figures with consistent layout grammar.
- Add regression/classification/comparison/permutation/SHAP figure builders.
- Add figure snapshot checks (dimensions, labels, deterministic filenames).

## Phase 7 - Reproducibility and Release Hardening
- Add end-to-end run command that rebuilds all artifacts from clean state.
- Add smoke integration test for full pipeline on reduced sample.
- Freeze environment, document hardware/runtime notes, and add citation/license metadata.
- Keep repository free of deprecated code paths and stale references.

## Definition of Done
- One documented command sequence recreates all tables/figures claimed in paper.
- Results are deterministic across reruns in the pinned environment.
- CI/local checks pass: lint, tests, type checks, smoke pipeline.
- No dependency on notebook state or manual intervention.
