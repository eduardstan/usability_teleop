# Data Contracts (Phase 1)

Canonical raw input files (expected in `data/raw/` once ingested):
- `raw_features_full.csv`
- `labels_full.csv`
- `User_risposte.xlsx`
- `tempi_media.csv`

## Structural Constraints
- `raw_features_full.csv`
  - shape: `N x 361`
  - no missing values
  - columns generated as `19 features x 19 signals`
- `labels_full.csv`
  - columns: `task_id,user_id,rep_id`
  - integer dtypes only
  - `task_id` in `{1,2,3}`
  - `rep_id` in `{1,2,3}`
- `User_risposte.xlsx`
  - exactly 15 columns, normalized to canonical names by position
  - timestamp must be present
  - all target responses must map to Likert values (`strongly disagree`..`strongly agree`)
- `tempi_media.csv`
  - columns exactly as dataset contract export
  - summary row without `user_id` is allowed and removed
  - time fields must match `mm:ss.xx`

## Alignment Constraints
After cleaning:
- number of `raw_features` rows must equal `labels` rows
- users in labels/questionnaire/times must match exactly
- expected sample count must equal `users * 3 tasks * 3 repetitions`

## CLI
Validate and optionally ingest data:

```bash
usability-teleop validate-data --source-dir data/raw --copy-to-raw
```
