# ML module (Hajera)

This folder contains the ML dataset checks, training and prediction code, saved models, and evaluation artifacts. Shared input names and types remain defined by the repository-level `feature_schema.py` and `FEATURE_SCHEMA.md`; those files are read-only from this module's workflow.

## Current models

| Task | Data | Prediction interface |
|---|---|---|
| Python complexity estimate | CodeComplex Python records | `predict_complexity(python_code)` |
| C defect-risk estimate | CodeXGLUE / Devign defect detection | `predict_defect(c_code)` |

The preferred prediction bundles are the validation-selected ExtraTrees experiment for CodeComplex and constrained Random Forest experiment for Devign. Original baseline bundles remain intact. The defect score is a risk estimate, not proof that code is vulnerable. Complexity labels are dataset classifications, not a guarantee of exact asymptotic behavior.

## Prediction API

From the repository root, another Python component can call the public functions directly:

```python
from hajera.training.predict import predict_complexity, predict_defect

complexity = predict_complexity("def total(values):\n    return sum(values)\n")
defect = predict_defect("int f(int *p) { return *p; }")
```

The API loads each model lazily and independently. It prefers the validation-selected experiment bundles, then falls back to the baseline bundle and the legacy schema model:

- `models/complexity_python_rf_improved.joblib`
- `models/defect_c_rf_improved.joblib`
- `models/complexity_python_rf_baseline.joblib` and `models/defect_c_rf_baseline.joblib` are preserved fallbacks.

`predict_complexity(python_code)` accepts a non-empty string containing syntactically valid Python and returns:

```python
{
    "predicted_complexity": "O(n)",
    "complexity_class": "linear",  # original CodeComplex class
    "confidence": 0.62,            # winning class probability
}
```

`predict_defect(c_code)` accepts a non-empty C source string or snippet and returns:

```python
{
    "risk_probability": 0.37,       # probability of Devign target 1 (Insecure)
    "predicted_label": "non-vulnerable",  # threshold: risk >= 0.5
}
```

These values illustrate the response shape. Both functions raise `PredictionInputError` for empty, non-string, or NUL-containing input; complexity prediction also rejects Python syntax errors. Model loading and feature-contract failures raise `ModelBundleError`. `load_model_bundle("complexity")`, `load_model_bundle("defect")`, and `get_model_info()` are available for startup checks. `get_model_info()` returns artifact name, model family, expected feature dimension, and class labels.

The model bundle’s feature order is checked against the existing extractor contract at load time. `schema_features.prep` applies the same `log1p` transformation used during training; a saved scaler, when present, is applied after it. Complexity confidence and defect risk probabilities are model scores and are not calibrated probabilities. The C extractor is lexical rather than a C parser, so it accepts snippets and does not certify syntax or vulnerability. The controlled Devign evaluation covers unseen commits within FFmpeg and qemu, not unseen projects.

The experiment reports are `evaluation/complexity_python_generalization_experiment.json` and `evaluation/defect_c_generalization_experiment.json`. The CodeComplex experiment improved validation macro-F1 over its baseline but scored lower on the held-out test split. Devign improved on the fixed test split, although its ROC-AUC remains weak. See the reports for candidate comparisons and per-class metrics.

## Folder contents

- `data/`: optional local datasets; raw data is not committed here.
- `training/dataset_preparation.py`: read-only dataset loading, validation, quality summaries, and split-integrity checks.
- `training/devign_preparation.py`: read-only cached Arrow loading, conflicting-label quarantine, connected-component splitting, artifact generation, and leakage verification.
- `training/train_schema_models.py`: training entry point. It uses validation data for model selection and does not use the defect test split for saving decisions.
- `training/make_report.py`: final evaluation/report entry point. Defect results are evaluated on the official test split.
- `training/schema_features.py`, `training/predict.py`: existing feature extraction and prediction implementation.
- `training/test_dataset_preparation.py`: small in-memory tests; does not need dataset downloads or model files.
- `models/`: existing serialized estimators. Do not replace them except through an explicitly approved training run.
- `evaluation/`: existing evaluation outputs.

## Environment

Use a Python 3.10 or 3.11 environment. The saved scikit-learn estimators were serialized with scikit-learn 1.7.2; use that version when loading those artifacts to avoid pickle compatibility warnings. The dependency list is in `hajera/requirements.txt`. Installing it is a separate environment action; the Phase 1 workflow does not install or upgrade packages.

The Hugging Face `datasets` and `pyarrow` packages are needed to read the cached Devign Arrow dataset. `streamlit` is needed only for the frontend, and `pytest` is needed for tests. Dataset loading is configured cache-only: the scripts will not download data automatically.

## Dataset locations and preparation

### CodeComplex

The Python dataset is JSONL with `src`, `complexity`, and `problem` required, plus `from` and `tags` in the currently inspected file. The default path is `hajera/data/python_data.jsonl`. To use a separately stored copy, set `CODEINTEL_CODECOMPLEX_PATH` to its path. For example, in PowerShell:

```powershell
$env:CODEINTEL_CODECOMPLEX_PATH = 'D:\datasets\CodeComplex\data\python_data.jsonl'
```

The current corpus contains 4,900 records and seven labels: `constant`, `logn`, `linear`, `nlogn`, `quadratic`, `cubic`, and `np`. Labels are kept as stored; `np` is not converted, dropped, or merged. The current prediction display maps `np` to “exponential / NP-hard.” The dataset has no `Other/Unknown` class.

The original corpus contains 4,900 records. Dataset preparation quarantines all 59 normalized-source groups with conflicting labels (127 records), including the 59 exact-source conflict groups (121 records). It does not change labels or the original JSONL. The remaining 4,773 records retain all seven classes and contain no normalized-source group with conflicting labels. Only this modeling pool enters training, validation-based selection, final refitting, and primary test evaluation. Quarantined rows and deterministic normalized-source hash IDs remain available in memory for separate analysis. The fixed problem-grouped split currently produces 2,888 training, 993 validation, and 892 test records. Exact and normalized source overlap across splits fails closed. Class counts and quarantine statistics are included in the validation report and in `complexity_quarantine_report.txt`, which is generated only when `make_report.py` is explicitly run. Similarity caused by identifier renaming or small logic changes is not detected, so semantic near-duplicate detection remains a limitation.

The upstream CodeComplex project identifies the source data as Codeforces code and documents seven complexity classes. The directly referenced data repository does not clearly establish the license for the current 2025 dataset release. The previous license claim in this README was not verified; confirm terms with the dataset authors before redistribution or other use. [Dataset repository](https://github.com/sybaik1/CodeComplex-Data) · [Project and dataset description](https://github.com/sybaik1/CodeComplex)

### Devign / CodeXGLUE

The dataset identifier is `google/code_x_glue_cc_defect_detection`. Preparation reads the existing cached Arrow IPC streams directly with PyArrow and never downloads or modifies the Hugging Face cache. Official split names and sizes are train 21,854, validation 2,732, and test 2,732. Records contain `id`, `func`, `target`, `project`, and `commit_id`; `target=True` (1) is Insecure and `False` (0) is Secure.

Run the controlled preparation from the repository root with the configured Python environment:

```powershell
python hajera/training/devign_preparation.py
```

The command writes `hajera/data/devign_prepared/train.jsonl`, `validation.jsonl`, `test.jsonl`, `quarantine.jsonl`, and `preparation_report.json`. The three modelling files contain 27,148 retained original records split into 19,004 train, 4,072 validation, and 4,072 test records. The quarantine JSONL contains all 170 original records from the 85 normalized-source groups with conflicting labels, nested intact with their original metadata, original split, stable group ID, and quarantine reason. The four same-label normalized duplicate groups remain in the modelling pool and are kept together by the grouping strategy.

The controlled split assigns entire connected components, joining records that share either the conservative whitespace-normalized C token stream or a `(project, commit_id)` pair. Its deterministic assignment balances record count, labels, and projects and verifies zero normalized-source, commit-pair, or component overlap across controlled splits. The original official splits remain unchanged in the Hugging Face cache for a separate benchmark comparison. They contain source and commit overlap; the preparation report records the audited pairwise overlap counts. Any benchmark using all original official rows is a separate protocol and retains the known contradictory-label and leakage limitations.

The controlled evaluation measures generalization to unseen commits within FFmpeg and qemu. It does not measure generalization to unseen projects. The audit also flagged 753 records with structural C-source anomalies. The lexical feature extractor accepts these records but is not a C parser, so the warnings do not establish whether a source record is truncated or an intentionally incomplete snippet. Identifier-renamed or slightly modified code is not detected.

CodeXGLUE lists the Computational Use of Data Agreement (C-UDA) for its datasets and describes the intended use as computational/non-commercial research. Review the provider's terms before redistribution or non-research use. [Hugging Face dataset card](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection) · [Microsoft CodeXGLUE](https://github.com/microsoft/CodeXGLUE)

## Run dataset checks and tests

From the repository root, run the in-memory tests with:

```powershell
python -m pytest -p no:cacheprovider hajera/training/test_dataset_preparation.py hajera/training/test_devign_preparation.py
```

The tests use small in-memory records. Running the Devign preparation command is a separate data-artifact generation step; it does not train models. `streamlit` is needed only for the frontend.

```powershell
python train_schema_models.py
python make_report.py
```

`train_schema_models.py` is a training command and overwrites model/report artifacts when its validation rule permits saving. Do not run it as part of dataset-only validation. `make_report.py` writes evaluation artifacts. The prediction API and frontend integration remain as currently implemented.

## Current limitations

- The 59 CodeComplex conflicting source groups are quarantined without relabeling; author adjudication remains outstanding.
- Exact code duplicates are checked, but semantically near-identical code is not yet detected.
- Devign's official split has exact and normalized-source duplicates across boundaries, contradictory labels for 85 normalized-source groups, and extensive shared project/commit pairs. Use the controlled split for primary evaluation and treat results on the official split as a separate benchmark with those limitations disclosed.
- Devign's controlled split contains records from both FFmpeg and qemu in every partition; it does not support claims about unseen-project generalization.
- Devign source quality includes 753 heuristic structural warnings. The lexical extractor processes them, but source completeness has not been verified against upstream files.
- Existing models were saved under scikit-learn 1.7.2; loading with 1.8.0 emitted a version warning.
- The C feature extractor uses heuristics and token counts rather than a C AST parser.
