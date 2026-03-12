# Implementation Plan v4 (Protocol Hardening + Repo Hygiene)

## Objective
Deliver a production-grade, reproducible protocol layer with clear command semantics, robust artifact contracts, modular CLI loading, and clean repository outputs/logging hygiene.

## Constraints
- Keep `develop` compatibility for existing RQ2/RQ3 commands.
- Preserve reproducibility and deterministic behavior.
- Do not keep stale generated artifacts in versioned `outputs/`.

## Work Packages

### WP1. Plan + Task Governance
- Create `IMPLEMENTATION_PLAN_v4.md` and `TASK_LIST_v4.md`.
- Track each task status explicitly.

### WP2. Protocol Correctness Hardening
- Fix feature selection bug for `avg` feature-set under `top_k_per_axis`.
- Add defensive validation for `estimation_best_configs.csv` and `final_models.csv` fields.
- Ensure clear failure messages for invalid model/feature-set references.

### WP3. CLI Modularity + Isolation
- Refactor parser to use lazy command handlers.
- Prevent parser import from requiring plotting/runtime-heavy dependencies.
- Keep CLI UX unchanged from user perspective.

### WP4. Logging and Error-Path Consistency
- Ensure command entrypoints and protocol services emit structured, informative logs.
- Remove silent failure-prone paths and improve actionable error logs.

### WP5. Repository Hygiene
- Remove stale generated outputs/log files from `outputs/` while preserving `.gitkeep` structure.
- Remove deprecated planning/task artifacts superseded by v4 where safe.

### WP6. Validation Coverage
- Add tests for:
  - avg feature selection with `top_k_per_axis`
  - protocol artifact validation/failure paths
  - parser lazy import behavior for protocol commands
- Run targeted tests and record results.

### WP7. Documentation Alignment
- Update README command behavior and config support notes.
- Align task list completion with real test evidence.

## Acceptance Criteria
- `run-estimation`/`fit-final-models` no longer fail for `avg` with feature screening.
- Invalid or stale protocol artifacts fail fast with explicit diagnostics.
- CLI parser can be imported without seaborn/matplotlib installed.
- `outputs/` contains only structural placeholders (`.gitkeep`) after cleanup.
- New/updated tests pass in the available environment.
