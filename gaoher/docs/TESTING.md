# Testing and Validation

Run the full Gaoher suite from the repository root with Python 3.11 and Hajera's dependencies installed:

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv311\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -v
```

The counts below are the latest run under `.venv311` Python 3.11.9: Gaoher **62 passed**, Hajera **64 passed**, ML bridge **7 passed**, prediction interface **11 passed**, and frontend AppTests **2 passed**. The frontend emitted two bare-mode `ScriptRunContext` warnings; see [`FINAL_PROJECT_AUDIT.md`](../../FINAL_PROJECT_AUDIT.md) for the exact commands, outputs, and release status.

The tests cover Gaoher's AST parser and complexity estimates, shared schema fields, the exact Hajera 17-value complexity vector, actual NumPy preprocessing, the defect feature adapter, public model predictions for Python and C, and Streamlit UI smoke tests.

## Verified result

Python 3.11.9 in `.venv`; tested dependency versions include NumPy 2.2.6, SciPy 1.17.1, joblib 1.6.0, and scikit-learn 1.7.2. Hajera's complete `requirements.txt` was installed in the project environment.

- Gaoher: **62 tests passed, 0 failed, 0 skipped**.
- Hajera: **64 tests passed**.
- Actual Python complexity and C defect predictions ran through Hajera's public API and real hydrated model bundles.
- Streamlit AppTest emits a bare-mode `ScriptRunContext` warning and a `use_container_width` deprecation warning; UI checks pass.

Normal package imports must resolve from the repository root. Ensure Git LFS is installed and model files are hydrated (`git lfs pull --include="hajera/models/*.joblib"`). Training requires external CodeComplex data and the cached Devign dataset; neither was needed for inference validation.

See [`ML_INTEGRATION_REPORT.md`](../ML_INTEGRATION_REPORT.md) for observed output values, setup steps, model contracts, and bridge interface details.
