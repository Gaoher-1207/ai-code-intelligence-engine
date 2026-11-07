# Gaoher–Hajera ML Integration Report

## Result

The integration runs from the normal repository root using standard Python package imports. Gaoher parses Python source and exposes its validated features, Hajera's ordered vector and actual NumPy preprocessing output; prediction is delegated to Hajera's public API. The C path delegates defect extraction and prediction to Hajera. No model, prediction, feature extractor, or preprocessing implementation is mocked or copied.

## Verified runtime

- Python 3.11.9, repository `.venv`.
- NumPy 2.2.6, SciPy 1.17.1, joblib 1.6.0, scikit-learn 1.7.2.
- Hajera's full declared requirements were installed into `.venv`; full list is in `hajera/requirements.txt`.
- Normal imports: `hajera.training.schema_features` and `hajera.training.predict`.
- Hajera's `prep()` was run with NumPy and compared against Gaoher's adapter output; shape is `(1, 17)` and values match.

## Models

The public predictor's six configured bundles are present as hydrated Git LFS binaries under `hajera/models/`, and were successfully loaded by Hajera's predictor. The preferred Python complexity model is `complexity_python_rf_improved.joblib` (5,843,346 bytes; 17 numeric inputs; seven classes). The preferred C defect model is `defect_c_rf_improved.joblib` (3,792,730 bytes; 13 numeric inputs; binary classes). Baseline and schema fallback bundles are also included. `git lfs fsck --objects` reports `Git LFS fsck OK`.

## Real predictions

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -c "from gaoher.ml_integration import predict_python_complexity; print(predict_python_complexity('def total(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n')['prediction'])"
.\.venv\Scripts\python.exe -c "from gaoher.ml_integration import predict_c_defect; print(predict_c_defect('void copy(char *s) { char buf[8]; strcpy(buf, s); }')['prediction'])"
```

Observed results from the same samples: Python `O(log n)`, class `logn`, confidence `0.257265`; C risk probability `0.581239`, predicted label `vulnerable`. These are model outputs from the actual public predictor and checked-in LFS bundles.

## Test results

- Gaoher: `python -m unittest discover -s gaoher\tests -v` — **62 passed, 0 failed, 0 skipped**. This includes real feature parity, the exact 17-value vector, NumPy preprocessing comparison, real Python and C predictions, and Streamlit UI smoke tests.
- Hajera: `python -m pytest -p no:cacheprovider hajera\training -q` — **64 passed**.
- `git diff --check` completed successfully; Git printed only line-ending normalization notices for modified Gaoher text files.

Streamlit's AppTest emits its expected bare-mode context warning and the current `use_container_width` deprecation warning; both UI tests pass.

## Reproducible setup and verification

From a normal clone with Git LFS installed and hydrated:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r hajera\requirements.txt
git lfs pull --include="hajera/models/*.joblib"
git lfs fsck --objects
.\.venv\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -q
```

The project must be run from its repository root so Python resolves the `gaoher` and `hajera` namespace packages. Inference requires the hydrated models and compatible installed dependencies. Training additionally requires external CodeComplex input and a cached Devign dataset; data preparation and model retraining were not part of this integration verification.

## Bridge interface

- `gaoher.ml_integration.predict_python_complexity(source)` returns analysis summary, validated feature mapping, ordered feature names/vector, actual preprocessed values, and Hajera's unchanged prediction mapping.
- `gaoher.ml_integration.predict_c_defect(source)` returns Hajera's defect feature mapping and prediction mapping.

Hajera owns authoritative ML extraction, preprocessing, model loading, and prediction. Gaoher performs compatibility validation and delegates; it does not bypass Hajera's public prediction functions.

## Current audit status (2026-11-08)

The final verification was run after activating `.venv311`, which reported Python 3.11.9. Dependency imports passed (NumPy 2.2.6, SciPy 1.17.1, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.6.0, pytest 9.2.0, Streamlit 1.67.0).

- Gaoher complete suite: **62 passed, 0 failed, 0 skipped**.
- Hajera complete suite: **64 passed, 0 failed, 0 skipped**.
- Explicit Gaoher ML bridge: **7 passed, 0 failed**.
- Hajera model prediction interface: **11 passed, 0 failed**.
- Frontend AppTests: **2 passed, 0 failed** with two expected bare-mode `ScriptRunContext` warnings and no container-width deprecation warning.
- Real predictions through Hajera and the Gaoher bridge agreed: Python `total(xs)` → `O(log n)`, `logn`, confidence `0.257265`; C `strcpy` sample → risk `0.574168`, `vulnerable`.
- LFS fsck and `git diff --check` passed.

The earlier restricted-shell attempt could not access the configured Python install; that result is superseded by this verified venv run. `.venv311_old` remains preserved. Training reproduction still depends on external datasets, and licensing terms require review. See [`FINAL_PROJECT_AUDIT.md`](../FINAL_PROJECT_AUDIT.md) and [`FINAL_INTEGRATION_REPORT.md`](../FINAL_INTEGRATION_REPORT.md) for detailed results.
