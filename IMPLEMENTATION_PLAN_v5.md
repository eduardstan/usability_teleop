# Implementation Plan v5 (Ablation + Cluster-Ready Execution)

## Scope
Deliver a robust, reproducible ablation-focused experimental pipeline with:
- per-dimension/feature-set feature selection as a first-class protocol component,
- constrained and explicit class rebalancing policy (`none` or `smote`),
- expanded hyperparameter spaces with documented fast/full profiles,
- cluster-friendly execution guidance that regenerates all paper artifacts.

## Current Gap Summary
- Fold-safe feature selection logic exists (`top_k_per_axis`) but is not exposed as a dedicated ablation protocol with standardized outputs/figures.
- CLI still exposes deprecated class-balance options (`oversample`, `undersample`) that are no longer desired.
- README does not provide a full, production-style experiment recipe for cluster execution and artifact regeneration.
- Hyperparameter grids exist but are not explicitly organized into fast-vs-full reproducible profiles for ablation and large runs.

## Design Principles
- Deterministic by default: fixed seeds, explicit configs, reproducible CSV/figure outputs.
- No hidden coupling: each stage consumes explicit input artifacts and emits explicit output artifacts.
- Ablation is a protocol, not ad-hoc scripting: same CLI/config system, same logging, same output contracts.
- Cluster-first ergonomics: clear run commands, output directories, and resumable stage boundaries (without implementing stop/resume yet).

## Phase 1: Protocol/API Alignment
- Remove `oversample` and `undersample` from CLI/API choices.
- Standardize class-balance mode to `none|smote` across:
  - parser argument choices,
  - typing aliases,
  - docs/examples.
- Ensure regression code paths never receive rebalancing behavior.
- Add validation guardrails that fail fast if unknown class-balance values are passed via config/code.

## Phase 2: First-Class Ablation Protocol
- Introduce an explicit ablation command family (proposed):
  - `run-ablation` for generating ablation tables,
  - `build-ablation-figures` for ablation visuals from CSV inputs.
- Ablation factors (initial):
  - feature selection: `none` vs `variance_top_k_per_axis`,
  - feature-set scope: per axis combination (`x`, `y`, `z`, `w`, combos, `avg`),
  - classification rebalancing: `none` vs `smote`.
- Output contracts:
  - `outputs/tables/ablation_summary.csv`,
  - `outputs/tables/ablation_breakdown.csv`,
  - `outputs/tables/ablation_feature_filter_summary.csv`,
  - publication-grade ablation figures in `outputs/figures/`.

## Phase 3: Feature Selection Implementation Hardening
- Keep fold-safe selection (train-fold only statistics) as mandatory behavior.
- Ensure selection metadata is persisted in all relevant tables:
  - selection method,
  - selected feature counts per fold,
  - selected feature lists for final models.
- Add integrity tests:
  - no test fold leakage in selection,
  - deterministic selected columns under fixed seed,
  - expected behavior for `avg` and sparse axis subsets.

## Phase 4: Hyperparameter Profile Expansion
- Split model grids into explicit profiles:
  - `fast` profile for dev/smoke,
  - `full` profile for paper-grade runs.
- Keep profiles in versioned config files (e.g., `configs/models_fast.yaml`, `configs/models_full.yaml`).
- Add guardrails:
  - schema validation for config grids,
  - run logs must include profile id and model-grid hash.

## Phase 5: Visualization + Artifact Standardization
- Extend figure builders for ablation:
  - stage comparison (performance deltas),
  - selection-strength sensitivity curves (`top_k_per_axis`),
  - per-target gains/losses from selection and SMOTE.
- Keep unified style system and deterministic filenames.
- Ensure `build-figures` can include ablation figures when ablation CSVs are present; skip cleanly when absent.

## Phase 6: Cluster Execution Documentation
- Document end-to-end stage-by-stage execution in README:
  - estimation,
  - stat validation,
  - final fit,
  - explainability,
  - figure build.
- Add a “full experiment” recipe with:
  - recommended output paths (`outputs/tables`, `outputs/figures`, `outputs/runs`),
  - fast vs full command examples,
  - per-stage expected outputs.

## Phase 7: Validation + Exit Criteria
- Tests:
  - CLI parser + argument constraints,
  - selection correctness and leakage safety,
  - ablation output schema checks,
  - smoke E2E (`max-models=2`).
- Exit criteria:
  - full stage-based flow regenerates all expected tables/figures,
  - ablation outputs are reproducible and publication-ready,
  - docs are cluster-ready and complete for clean-room reruns.
