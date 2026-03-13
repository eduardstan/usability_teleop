# Development History

This file summarizes major repository milestones that were previously tracked across versioned planning/validation artifacts.

## 2026-03 (v3): Protocol Unification
- Introduced unified protocol architecture with explicit lanes:
  - `run-estimation`
  - `fit-final-models`
  - `run-final-explainability`
- Standardized nested LOSO estimation outputs and final-model artifacts.
- Removed dependence on ad-hoc per-command surfaces and converged to unified command semantics.

## 2026-03 (v4): Protocol Hardening + Hygiene
- Hardened feature selection behavior for `avg` feature set and validation pathways.
- Added parser lazy-loading to isolate heavy runtime dependencies.
- Improved error-path logging and protocol table validation.
- Performed repository cleanup of stale generated outputs and legacy planning clutter.
- Added targeted validation tests and smoke runs.

## 2026-03 (v5): Ablation + Cluster Readiness
- Enforced class-balance policy to `none|smote`.
- Added first-class ablation commands:
  - `run-ablation`
  - `build-ablation-figures`
- Added fast/full model profile support:
  - `configs/models_fast.yaml`
  - `configs/models_full.yaml`
  - `--models-config` CLI switch.
- Expanded README with stage-by-stage cluster execution guidance.

## 2026-03 (Current): Cleanup + Symmetry Improvements
- Added classification global-vs-target-specific comparison artifacts for symmetry with regression.
- Integrated classification comparison into statistical outputs and publication figure build flow.
- Consolidated historical planning artifacts into canonical files and this history document.
- Refreshed onboarding docs (`README.md`, `AGENTS.md`) for clean new-chat startup context.

## Historical Artifacts Consolidated
The following versioned artifacts were consolidated into this history and canonical files:
- `IMPLEMENTATION_PLAN_v3.md`
- `IMPLEMENTATION_PLAN_v4.md`
- `IMPLEMENTATION_PLAN_v5.md`
- `TASK_LIST_v3.md`
- `TASK_LIST_v4.md`
- `TASK_LIST_v5.md`
- `VALIDATION_v4.md`
