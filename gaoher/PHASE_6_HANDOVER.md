# Phase 6: Gaoher–Hajera Runtime Handover

**Status: integrated and verified in the current worktree.** The bridge uses normal `hajera.training` imports and delegates prediction to Hajera's public APIs. The two branches have unrelated histories; integration is selective and does not merge histories.

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
.\.venv\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -q
git lfs fsck --objects
```

See [`ML_INTEGRATION_REPORT.md`](ML_INTEGRATION_REPORT.md) for model inventory, actual outputs, setup, and limitations. Training is not verified because CodeComplex source data and the Devign cache are external to this repository; inference does not need them.
