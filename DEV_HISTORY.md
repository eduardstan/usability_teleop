# Development History

Last updated: 2026-03-13

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
- Refined ablation to fold-safe feature-selection stages over `top_k_per_axis` values.
- Unified ablation UX naming to `--top-k-per-axis` and removed alternate argument naming.
- Deprecated class-balance in active CLI and disabled balancing in current protocol runs.
- Expanded `models_full.yaml` to a broader paper-grade hyperparameter space.

## 2026-03-13 (Current Patch): Runtime Metadata + API Documentation Hardening
- Added systematic run-manifest emission for stage commands with:
  - UTC start/end timestamps,
  - elapsed seconds,
  - command args,
  - status and error payload,
  - git commit hash,
  - output artifact paths.
- Introduced and documented the two-manifest pattern:
  - immutable `run_manifest_<command>_<timestamp>.json`,
  - latest pointer `run_manifest_<command>_latest.json`.
- Extended ablation CLI with `--num-workers` and documented worker semantics.
- Corrected ablation defaults to full-search behavior when caps are omitted:
  - `--max-models` unset -> all models from selected YAML,
  - `--max-feature-sets` unset -> all feature sets.
- Reworked `README.md` and `EXPERIMENTS.md` API/default sections for cluster reproducibility clarity.

## Historical Artifacts Consolidated
The following versioned artifacts were consolidated into this history and canonical files:
- `IMPLEMENTATION_PLAN_v3.md`
- `IMPLEMENTATION_PLAN_v4.md`
- `IMPLEMENTATION_PLAN_v5.md`
- `TASK_LIST_v3.md`
- `TASK_LIST_v4.md`
- `TASK_LIST_v5.md`
- `VALIDATION_v4.md`
