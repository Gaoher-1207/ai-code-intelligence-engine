# Hajera Main Branch Integration Handover

## Integration status

The required runtime implementation has been selectively integrated into the current `main` checkout's worktree. The branches have unrelated histories, so no whole-branch merge was used. Shared root schema files were preserved. The changes are prepared for the requested integration branch and are not yet committed or pushed.

## Included Hajera files

- `hajera/README.md`, `hajera/requirements.txt`
- `hajera/training/schema_features.py`, `predict.py`, `dataset_preparation.py`, `devign_preparation.py`, `train_schema_models.py`, `make_report.py`
- Hajera tests: `test_baseline_training.py`, `test_dataset_preparation.py`, `test_devign_preparation.py`, `test_ml.py`, `test_prediction_interface.py`
- Six model bundles used by `predict.py`: Python complexity baseline/improved/schema and C defect baseline/improved/schema.
- Root `.gitattributes` LFS rule for `hajera/models/*.joblib`.

The two unused `*_tfidf_lr.joblib` artifacts, exploratory notebook, evaluation outputs, local caches, and datasets were excluded. Public inference does not reference the TF-IDF bundles. Training still needs external CodeComplex input and a cached Devign dataset. The existing root `feature_schema.py` and `FEATURE_SCHEMA.md` were retained without changes.

The existing `frontend/app.py` from the source branch was integrated as the UI entry point, with standard imports and calls to Gaoher's public bridge functions. It uses the normal root package structure and does not add import-path mutation.

## LFS and model validation

The six required model files are materialized binary artifacts, not pointer text. Their recorded sizes are:

| Model | Bytes |
|---|---:|
| `complexity_python_rf_baseline.joblib` | 100,591,210 |
| `complexity_python_rf_improved.joblib` | 5,843,346 |
| `complexity_python_schema_lr.joblib` | 7,530,479 |
| `defect_c_rf_baseline.joblib` | 186,687,106 |
| `defect_c_rf_improved.joblib` | 3,792,730 |
| `defect_c_schema_lr.joblib` | 2,368,783 |

All six loaded successfully through Hajera's actual prediction API; `git lfs fsck --objects` reported `Git LFS fsck OK`. The preferred complexity bundle has 17 numeric features and seven output classes; preferred defect bundle has 13 numeric features and binary output classes.

## Dependencies

Install the full declared set from `hajera/requirements.txt` into Python 3.11 `.venv`. Inference dependencies include NumPy, SciPy, joblib, and scikit-learn 1.7.2 (pinned to the serialized estimators); the requirements also include pandas, datasets, pyarrow, matplotlib, Streamlit, and pytest for data, reporting, UI, and tests.

Verified environment: Python 3.11.9, NumPy 2.2.6, SciPy 1.17.1, joblib 1.6.0, scikit-learn 1.7.2. Full test environment installation succeeded.

## Verification performed

From the repository root:

```powershell
git lfs fsck --objects
.\.venv\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -q
```

Results: Gaoher **62 passed, 0 failed, 0 skipped**; Hajera **64 passed**. Gaoher integration tests compared shared feature fields, exact 17-value vector order, and actual NumPy `prep()` output, then ran real Python complexity and C defect model predictions.

Observed representative outputs: Python `total(xs)` sample gave `O(log n)` / `logn`, confidence `0.257265`; C `strcpy` sample gave risk `0.581239`, label `vulnerable`. The public Hajera predictor performed extraction, preprocessing, loading, and prediction. No models were retrained.

## New branch and publish steps

The approved final integration procedure is:

```powershell
git switch -c integration/ai-code-intelligence
git add .gitattributes HAJERA_MAIN_BRANCH_HANDOVER.md FINAL_INTEGRATION_REPORT.md frontend/app.py hajera/README.md hajera/requirements.txt hajera/training hajera/models/*.joblib gaoher
git diff --cached --check
git commit -m "feat: integrate code intelligence and ML prediction pipeline"
git push -u origin integration/ai-code-intelligence
```

Review staged paths before committing; only the six predictor-required bundles should be staged from `hajera/models/`. Do not merge unrelated histories or push to `main` / `hajera/ml-model`. Verify the remote branch tip after push. See `FINAL_INTEGRATION_REPORT.md` for the final branch and commit record.
