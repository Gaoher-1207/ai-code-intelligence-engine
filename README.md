# AI Code Intelligence Engine

This project combines Gaoher's Python source analysis with Hajera's trained
models for Python complexity classification and C defect-risk estimation. The
Streamlit frontend runs both workflows locally from submitted source text. It
does not execute user code.

## Architecture and request flow

```text
Streamlit UI (frontend/app.py)
  ├─ Python → Gaoher AST analysis and shared features
  │            → Hajera vector/preprocessing adapters
  │            → Hajera public complexity prediction API
  └─ C       → Hajera lexical feature extractor and defect prediction API
```

`feature_schema.py` defines the shared input contract. Gaoher owns the Python
AST parser, structural metrics and heuristic time/auxiliary-space estimates.
`gaoher.ml_integration` validates the Gaoher-to-Hajera feature contract and
delegates model inference to `hajera.training.predict`. Hajera owns the
authoritative feature extraction, preprocessing, model loading and predictions.
There is no separate HTTP backend; the Streamlit app calls Python modules
directly in the same process.

The C extractor is lexical and heuristic; it is not a C parser. Model outputs
are estimates. The complexity classifier predicts one of the dataset's seven
classes, and the defect score is not calibrated proof of a vulnerability.

## Environment setup

Use Python 3.10 or 3.11. The saved estimators were serialized with
scikit-learn 1.7.2, which is pinned in the requirements. From PowerShell at the
repository root:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r hajera\requirements.txt
git lfs pull --include="hajera/models/*.joblib"
git lfs fsck --objects
```

The requirements include NumPy, SciPy, pandas, scikit-learn, joblib, pytest,
Streamlit, matplotlib, Hugging Face `datasets`, and PyArrow. Dataset
preparation reads local files/caches and does not download datasets by itself.
If `.venv311` already exists, verify that its interpreter can start before
using it; virtual environments depend on the base Python installation used to
create them.

## Run the application

```powershell
.\.venv311\Scripts\python.exe -m streamlit run frontend\app.py
```

Choose Python or C, enter source code, and select **Analyze**. Python results
include estimated complexity, confidence, parser metrics and contract features.
C results include a risk estimate, label and extracted features. Python syntax
errors and model/input errors are shown in the page. C input is treated as a
snippet by lexical heuristics; it is not syntax-validated.

## Tests and verification

From the repository root:

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s gaoher\tests -v
.\.venv311\Scripts\python.exe -m pytest -p no:cacheprovider hajera\training -v
git lfs fsck --objects
```

Gaoher's `test_frontend_integration.py` exercises the Python and C UI flows
through Streamlit AppTest and the actual predictor. The suite includes real
model prediction tests; do not replace their results with mocked outputs when
claiming end-to-end verification. Historical test evidence and current audit
conditions are recorded in `FINAL_INTEGRATION_REPORT.md` and
`FINAL_PROJECT_AUDIT.md`.

## Data and training

Raw datasets are not included in this repository. CodeComplex JSONL is expected
at `hajera/data/python_data.jsonl`, or at the path in
`CODEINTEL_CODECOMPLEX_PATH`. Devign preparation reads the local Hugging Face
cache for `google/code_x_glue_cc_defect_detection`; its cache location can be
selected with `CODEINTEL_DEVIGN_DATASET`. See `hajera/README.md` for dataset
provenance, expected fields, quarantine and split policies, known licensing
limits, and exact preparation commands.

The tracked notebook `hajera/training/01_train_models.ipynb` is exploratory.
The reproducible training entry point is `hajera/training/train_schema_models.py`;
it requires the external data described above and can update model/evaluation
artifacts. Review its data availability and intended outputs before running.
`hajera/training/make_report.py` reproduces evaluation reports from prepared
data and models. Training and evaluation were not rerun as part of release
preparation.

## Project status

See `FINAL_PROJECT_AUDIT.md` for feature-by-feature readiness and remaining
work. The audit distinguishes historical passing test results from checks
rerun in the current environment. The CodeComplex held-out experiment did not
improve on its saved baseline, and Devign performance remains limited; consult
the JSON evaluation artifacts and `hajera/evaluation/RESULTS.md` before relying
on predictions.
