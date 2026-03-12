# Protocol Unification Plan (v3)

## Objective
Unify the pipeline into two explicit, non-overlapping lanes:
1. Estimation lane (nested LOSO, unbiased performance/statistics).
2. Final-model lane (global refit + explainability for publication/deployment).

No OOF explainability lane is included.

## Scope
- Refactor API to clear top-level commands:
  - `run-estimation`
  - `fit-final-models`
  - `run-final-explainability`
  - `run-paper-pipeline`
- Add structured artifacts for per-fold and final-model parameters.
- Remove dependence on `best_params_last_fold` as primary signal.
- Keep existing commands functional for backwards compatibility, but promote new API in docs.

## Design Decisions
- Estimation lane:
  - Feature selection and HP tuning happen inside each outer LOSO training fold.
  - Outputs include fold-level selected features and best params.
- Final-model lane:
  - Single global feature selection + tuning on all users.
  - Final model artifacts written as tables (and JSON-ready columns).
  - SHAP computed only from final models.
- Inference/permutation claims remain tied to estimation outputs only.

## Execution Phases
1. `DONE` Branch + baseline alignment.
2. Add protocol data structures and artifact schema.
3. Implement estimation lane command with unified output bundle.
4. Implement final-model fitting command and artifact tables.
5. Implement final explainability command (SHAP from final models only).
6. Add `run-paper-pipeline` orchestrator and concise docs.
7. Tests + smoke runs + artifact verification.

## Acceptance Criteria
- New commands run end-to-end on `--max-models 2 --max-feature-sets 2`.
- Estimation outputs include fold-wise params/features and aggregate summaries.
- Final-model outputs include one reproducible config per target/track.
- SHAP command reads only final-model artifacts.
- CLI/API naming is coherent and non-fragmented.
