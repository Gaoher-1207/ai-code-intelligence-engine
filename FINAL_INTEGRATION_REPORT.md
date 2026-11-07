# Final Integration Report

## Summary

Gaoher's static-analysis implementation and Hajera's ML runtime are integrated on a new branch based on the latest fetched `origin/main`. The source histories had no common ancestor, so files were selected into the main-based worktree instead of merging branches. The integration works from the repository root through normal package imports, with real Git LFS model bundles and real public prediction APIs.

## Hajera files integrated

- `hajera/README.md` and `hajera/requirements.txt`.
- Runtime and training modules: `dataset_preparation.py`, `devign_preparation.py`, `make_report.py`, `predict.py`, `schema_features.py`, and `train_schema_models.py`.
- Hajera tests: baseline training, dataset preparation, Devign preparation, ML features, and prediction interface.
- Six bundles configured by the public predictor: Python complexity baseline/improved/schema and C defect baseline/improved/schema.
- Root `.gitattributes` LFS rule for `hajera/models/*.joblib`.
- `frontend/app.py` is included as the app entry point and calls Gaoher's public bridge functions using normal imports.

The unused two `*_tfidf_lr.joblib` artifacts, exploratory notebook, evaluation outputs, caches, and external training data were excluded. The runtime does not reference the TF-IDF bundles. Shared root schema files were retained unchanged. Training data (CodeComplex source and cached Devign) remains external and is required only to reproduce training, not inference.

## Gaoher files preserved

All existing Gaoher work was retained. The bridge returns Gaoher parser/features plus Hajera's actual ordered vector and NumPy preprocessing output for Python, and delegates C defect extraction/prediction through Hajera's API. Tests include parser/analysis, compatibility checks, public predictions, and Streamlit smoke coverage. Gaoher documentation now records current results and reproducible commands.

## Models and LFS

Each included hydrated model was compared with its source-branch LFS pointer for exact size and SHA-256, and all passed:

| Bundle | Bytes | SHA-256 |
|---|---:|---|
| `complexity_python_rf_baseline.joblib` | 100,591,210 | `89ede53fb992a2adb21f0e4e8fc23995f36a7940510778a2e727928720fe0a00` |
| `complexity_python_rf_improved.joblib` | 5,843,346 | `1694826794d2b49a04890f775c9301adcff2eb5a071c5caa0c97fcc9e0802026` |
| `complexity_python_schema_lr.joblib` | 7,530,479 | `aa5a90dfdb27be9878e68f27cc698efa4df45d46e04f2147c82eb81801493315` |
| `defect_c_rf_baseline.joblib` | 186,687,106 | `4ba570eb4c31b64a0f2f17bd1dbf59ea9ca8b7112f6c3daa492126592d6b8d92` |
| `defect_c_rf_improved.joblib` | 3,792,730 | `48696c458ea13fd9eb7e8a32eb9e66e06370712b04d2c292096543727652b533` |
| `defect_c_schema_lr.joblib` | 2,368,783 | `ca1baf2e6010ca5a8919b1f27780dbad84085208132972a31dc357eaa282806f` |

`git lfs fsck --objects` passed. All six bundles loaded through Hajera's predictor. Preferred Python complexity bundle dimensions: 17 numeric features and seven classes. Preferred C defect bundle: 13 numeric features and binary classes.

## Environment and dependencies

- Python 3.11.9 in project `.venv`.
- NumPy 2.2.6, SciPy 1.17.1, joblib 1.6.0, scikit-learn 1.7.2.
- The full dependency set in `hajera/requirements.txt` was installed into `.venv`; no global package installation was used.
- Standard imports verified from the repository root: `hajera.training.schema_features`, `hajera.training.predict`, and `gaoher.ml_integration`.

## Real compatibility and predictions

Gaoher's complexity features matched Hajera's extractor and shared schema. The exact 17-value order matched Hajera's adapter. Actual NumPy preprocessing via Hajera's `prep()` matched Gaoher's adapter output at shape `(1, 17)`. The defect adapter matched Hajera's real feature output.

Actual public prediction results from the verified samples:

- Python `total(xs)` sample: `O(log n)`, class `logn`, confidence `0.257265`.
- C `strcpy` sample: risk probability `0.581239`, predicted label `vulnerable`.

Both predictions loaded actual serialized Hajera models; no prediction was mocked and no model was retrained.

## Tests and checks

- Gaoher: `python -m unittest discover -s gaoher\tests -v` — **62 passed, 0 failed, 0 skipped**.
- Hajera: `python -m pytest -p no:cacheprovider hajera\training -q` — **64 passed**.
- `git lfs fsck --objects` — passed.
- `git diff --cached --check` — passed before integration commit; `git diff --check` — passed before commit. Git emitted only CRLF normalization notices for text files.
- Streamlit AppTest reports a bare-mode `ScriptRunContext` warning and a `use_container_width` deprecation warning; both UI flows passed.

## Reproduce from a normal checkout

```powershell
git lfs pull --include="hajera/models/*.joblib"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r hajera\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -q
git lfs fsck --objects
```

## Git branch, commits, and push

- Base: `main` / `origin/main` at `5d336be8d4d7aa10744928f1f446fe3d09e7a99e` when the integration branch was created.
- Branch: `integration/ai-code-intelligence`.
- Integration commit: `a40a653c7b88c64c3abdaa11c94f3694662f0094`, message `feat: integrate code intelligence and ML prediction pipeline`.
- This report is committed separately after the integration commit so it can record that commit hash.
- Push target: `origin/integration/ai-code-intelligence`; remote ref will be checked against the local branch tip after pushing. No force push is used.

## Remaining limitations

The app and inference path work from a checkout when Python dependencies are installed and Git LFS models are hydrated. Model training was not rerun because required datasets are external and the trained models are already available. Streamlit emitted the warnings listed above. No other known runtime blocker remains.
