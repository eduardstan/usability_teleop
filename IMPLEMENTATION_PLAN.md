# Implementation Plan (Canonical)

Last updated: 2026-03-13

## Mission
Maintain a clean, reproducible, publication-grade pipeline that regenerates all tables/figures used in `draft.tex` from documented commands and configs.

## Current Architecture
- Stage 0: data validation (`doctor`, `validate-data`).
- Stage 1: estimation (`run-estimation`).
- Stage 2: statistical validation (`run-stat-validation`).
- Stage 3: final refit (`fit-final-models`) and explainability (`run-final-explainability`).
- Stage 4: artifact build (`build-figures`, `build-ablation-figures`).
- Full convenience orchestration: `build-paper-artifacts`.

## Active Priorities
1. Reproducibility hardening
- Ensure deterministic outputs with fixed seeds and explicit config references.
- Persist run metadata in `outputs/runs/` for every major command.
- Keep manifest semantics stable (`*_latest` pointer + immutable timestamped records).

2. Statistical completeness
- Keep regression and classification outputs symmetric where methodologically valid.
- Maintain permutation and inference outputs with corresponding figures.

3. Ablation rigor
- Keep feature-selection ablations as first-class pipeline outputs (class balancing currently disabled).
- Expand ablation visual suite for publication-ready narratives.
- Support configurable ablation worker parallelism via `--num-workers` while preserving deterministic defaults (`--num-workers 1`).

4. Cluster operations
- Keep stage-separated command flow so runs can be manually resumed by stage.
- Preserve fast/full model profile options through `--models-config`.

5. Documentation integrity
- Keep `README.md` and `AGENTS.md` as accurate single sources of truth.
- Avoid versioned planning artifacts; use canonical files plus `DEV_HISTORY.md`.
- Keep full CLI/API parameter semantics explicit in markdown docs (defaults and config fallbacks).

## Definition of Done
- A new contributor can reproduce all artifacts from README commands only.
- No deprecated/retired command surfaces or stale planning documents remain.
- Unit/smoke tests pass for CLI, protocol, stats, and visualization paths.
