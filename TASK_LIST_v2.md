# Task List v2 (Inference Bundle)

## Objective
Augment statistical validation with a comprehensive, standard inference bundle for RQ2/RQ3.

## Execution Plan (Follow Strictly)
1. Add protocol config for inference
- [x] Extend `configs/experiment.yaml` with inference settings:
  - baseline models (regression/classification)
  - bootstrap iterations
  - Bayesian bootstrap samples
  - paired-test alpha + FDR alpha
  - nested permutation toggle

2. Add robust inference modules
- [x] Implement LOSO per-subject prediction utilities for:
  - regression target-specific configs
  - classification target-specific configs
- [x] Implement bootstrap confidence intervals:
  - regression (`r2`, `rmse`, `mae`)
  - classification (`auc`, `accuracy`)
- [x] Implement paired baseline tests:
  - regression: Wilcoxon signed-rank on per-subject absolute errors (+ sign test fallback)
  - classification: exact McNemar on per-subject correctness
- [x] Implement Bayesian evidence:
  - probability(best model improves vs baseline) via Bayesian bootstrap
- [x] Implement Benjamini-Hochberg FDR correction over p-values.

3. Extend permutation framework
- [x] Add optional nested-permutation mode (retune in each permutation) controlled by config.
- [x] Keep default mode as fixed-best-params for runtime feasibility.

4. CLI integration
- [x] Add `run-inference` command to generate inference tables.
- [x] Integrate inference stage into `run-rq23-end2end`.
- [x] Integrate regression-only inference into `run-rq2-end2end`.

5. Artifacts
- [x] Write tables:
  - `inference_regression.csv`
  - `inference_classification.csv`
- [x] Include observed metric, bootstrap CI, paired p-value, FDR-adjusted p-value, Bayesian posterior probability of improvement.

6. Tests and QA
- [x] Add unit/smoke tests for inference calculations.
- [x] Run `ruff check .`
- [x] Run `pytest -q`
- [x] Run end-to-end smoke command proving artifacts are generated.

## Acceptance Criteria
- Inference outputs are deterministic with fixed seed.
- Both tracks (RQ2/RQ3) report uncertainty + paired significance + multiplicity control + Bayesian evidence.
- CLI and end-to-end flows generate inference artifacts without manual steps.
