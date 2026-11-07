# Phase 6: Gaoher–Hajera Runtime Handover

**Latest status (2026-11-08):** `.venv311` Python 3.11.9 verification passed: Gaoher 62/62, Hajera 64/64, real bridge predictions, model loading, and frontend AppTests all passed. The bridge uses normal `hajera.training` imports and delegates prediction to Hajera's public APIs. The two branches have unrelated histories; integration was selective and did not merge histories. See [`FINAL_PROJECT_AUDIT.md`](../FINAL_PROJECT_AUDIT.md) for exact outputs and remaining dataset limitations.

## Verified environment and tests

- Python 3.11.9 in the repository `.venv`.
- NumPy 2.2.6, SciPy 1.17.1, joblib 1.6.0, scikit-learn 1.7.2; Hajera's complete requirements are installed.
- Gaoher: **62 passed, 0 failed, 0 skipped**.
- Hajera: **64 passed**.
- LFS integrity: `git lfs fsck --objects` → `Git LFS fsck OK`.
- Actual Python and C predictors loaded real model bundles and returned predictions.

## Runtime contract

- `gaoher.ml_integration.predict_python_complexity(source)` parses with Gaoher, exposes compatibility feature/vector/preprocessing results, and calls Hajera's `predict_complexity(source)`.
- `gaoher.ml_integration.predict_c_defect(source)` delegates to Hajera's `predict_defect(source)` and returns validated defect features and prediction.
- Run from the repository root with the declared dependencies and hydrated Git LFS bundles.
- Hajera owns ML inference-time extraction, preprocessing, model loading, and prediction logic.

## Reproduce

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv311\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -v
git lfs fsck --objects
```

See [`ML_INTEGRATION_REPORT.md`](ML_INTEGRATION_REPORT.md) for model inventory, actual outputs, setup, and limitations. Training is not verified because CodeComplex source data and the Devign cache are external to this repository; inference does not need them.
