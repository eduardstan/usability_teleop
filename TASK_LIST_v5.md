# Task List v5 (Execution Checklist)

## Priority 0: API and Behavior Corrections
- [ ] Restrict `--class-balance` choices to `none` and `smote` in all CLI commands.
- [ ] Update type aliases and validation paths to accept only `none|smote`.
- [ ] Remove any dead code paths for oversampling/undersampling.
- [ ] Add tests verifying invalid balance modes fail fast.

## Priority 1: Ablation as First-Class Pipeline
- [ ] Add `run-ablation` command with config-driven factors.
- [ ] Add `build-ablation-figures` command (CSV-in, figures-out).
- [ ] Emit standardized ablation tables under `outputs/tables/`.
- [ ] Emit publication-grade ablation figures under `outputs/figures/`.
- [ ] Write run summary metadata under `outputs/runs/`.

## Priority 2: Feature Selection Rigor
- [ ] Confirm fold-safe selection in estimation/regression/classification paths.
- [ ] Persist selection metadata (`method`, fold counts, selected feature lists) in outputs.
- [ ] Add tests for no leakage and deterministic selection under fixed seed.
- [ ] Add tests for axis subsets and `avg` feature-set behavior.

## Priority 3: Hyperparameter Profile Expansion
- [ ] Introduce `models_fast.yaml` and `models_full.yaml`.
- [ ] Expand full profile grids for robust paper experiments.
- [ ] Add CLI/config switch for choosing profile path.
- [ ] Log model profile and config hash in run outputs.

## Priority 4: README and Cluster Execution
- [ ] Add full experiment command sequence for stage-by-stage execution.
- [ ] Document fast vs full profiles and expected runtime/usage.
- [ ] Document exact artifacts produced per stage.
- [ ] Document ablation execution commands and outputs.

## Priority 5: Verification and Quality Gates
- [ ] Add/refresh unit tests for parser, selection, and ablation schemas.
- [ ] Add smoke script for `max-models=2` full stage flow.
- [ ] Validate that `build-figures` handles partial/missing CSV inputs cleanly.
- [ ] Ensure lint/type/tests pass before merge.

## Definition of Done
- [ ] `none|smote` policy enforced consistently.
- [ ] Ablation command path produces reproducible tables + figures.
- [ ] README supports cluster-style rerun without hidden steps.
- [ ] Full artifact set reproducible from documented command sequence.
