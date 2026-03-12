# Validation v4

## Commands Executed

```bash
PYTHONPATH=src pytest -q tests/test_cli.py tests/test_feature_sets.py tests/test_protocol_validation.py
PYTHONPATH=src pytest -q tests/test_workflows.py tests/test_study_pipeline.py
bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate usability_teleop_clean && PYTHONPATH=src python ... run-estimation ...'
bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate usability_teleop_clean && PYTHONPATH=src python ... fit-final-models ...'
bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate usability_teleop_clean && PYTHONPATH=src python ... run-final-explainability ...'
```

## Results
- `tests/test_cli.py tests/test_feature_sets.py tests/test_protocol_validation.py`: `9 passed`
- `tests/test_workflows.py tests/test_study_pipeline.py`: `7 passed, 1 warning`
- Unified smoke run artifacts generated in `/tmp/usability_smoke_tables` and `/tmp/usability_smoke_figs`:
  - `estimation_regression.csv` (10 rows)
  - `estimation_classification.csv` (10 rows)
  - `estimation_best_configs.csv` (20 rows)
  - `final_models.csv` (20 rows)
  - `final_explainability_shap.csv` (10 rows)
  - 2 SHAP figure PNGs

## Notes
- Warning source: joblib multiprocessing helper reports permission limitations in the current sandbox; tests still pass.
- Parser/module loading no longer requires eager import of heavy runtime dependencies.
