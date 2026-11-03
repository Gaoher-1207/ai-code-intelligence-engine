# ML module (Hajera)

This folder contains the ML dataset checks, training and prediction code, saved models, and evaluation artifacts. Shared input names and types remain defined by the repository-level `feature_schema.py` and `FEATURE_SCHEMA.md`; those files are read-only from this module's workflow.

## Current models

| Task | Data | Prediction interface |
|---|---|---|
| Python complexity estimate | CodeComplex Python records | `predict_complexity(python_code)` |
| C defect-risk estimate | CodeXGLUE / Devign defect detection | `predict_defect(c_code)` |

The prediction functions and existing serialized model files are unchanged by dataset preparation. The defect score is a risk estimate, not proof that code is vulnerable. Complexity labels are dataset classifications, not a guarantee of exact asymptotic behavior.

## Folder contents

- `data/`: optional local datasets; raw data is not committed here.
- `training/dataset_preparation.py`: read-only dataset loading, validation, quality summaries, and split-integrity checks.
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

The quality summary reports repeated source code and conflicting labels without changing rows. The inspected corpus has 173 exact duplicate-source groups and 59 groups whose identical source has conflicting complexity labels. These require review. The deterministic problem-grouped split keeps the historical 80/20 outer test partition and creates a validation subset within the outer training portion. Exact and formatting/comment-normalized code overlap across splits fails closed. Similarity caused by identifier renaming or small logic changes is not detected, so semantic near-duplicate detection remains a limitation.

The upstream CodeComplex project identifies the source data as Codeforces code and documents seven complexity classes. The directly referenced data repository does not clearly establish the license for the current 2025 dataset release. The previous license claim in this README was not verified; confirm terms with the dataset authors before redistribution or other use. [Dataset repository](https://github.com/sybaik1/CodeComplex-Data) · [Project and dataset description](https://github.com/sybaik1/CodeComplex)

### Devign / CodeXGLUE

The dataset identifier defaults to `google/code_x_glue_cc_defect_detection`; override it with `CODEINTEL_DEVIGN_DATASET` if needed. The loader requires the cached dataset and uses `local_files_only=True`, so it does not download data. Official split names are `train`, `validation`, and `test`. The documented sizes are 21,854 / 2,732 / 2,732. Required fields are `id`, `func`, `target`, `project`, and `commit_id`.

The official card defines `target=True` (1) as insecure and `False` (0) as secure. Training fits on train, selects/saves using validation, and leaves test for final evaluation. Exact and formatting/comment-normalized source overlap across official splits is reported and blocks training for review; rows are never moved or deleted. The current environment did not permit inspection of cached Arrow row-level missing values or duplicate counts, so run the offline validation after installing the documented dependencies. Identifier-renamed or slightly modified code is not detected.

CodeXGLUE lists the Computational Use of Data Agreement (C-UDA) for its datasets and describes the intended use as computational/non-commercial research. Review the provider's terms before redistribution or non-research use. [Hugging Face dataset card](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection) · [Microsoft CodeXGLUE](https://github.com/microsoft/CodeXGLUE)

## Run dataset checks and tests

From the repository root:

```powershell
python -m pytest -p no:cacheprovider hajera/training/test_dataset_preparation.py
```

The tests use small in-memory records. Dataset checks do not write prepared data. The current environment lacks `datasets`, `pyarrow`, and `streamlit`; do not install packages without approval. Once the packages are available and paths/cache are configured, run scripts from `hajera/training`:

```powershell
python train_schema_models.py
python make_report.py
```

`train_schema_models.py` is a training command and overwrites model/report artifacts when its validation rule permits saving. Do not run it as part of dataset-only validation. `make_report.py` writes evaluation artifacts. The prediction API and frontend integration remain as currently implemented.

## Current limitations

- The CodeComplex corpus has conflicting labels for identical source records; preprocessing reports these without resolving them.
- Exact code duplicates are checked, but semantically near-identical code is not yet detected.
- Devign's official cached Arrow files were present, but row-level split validation could not be run in the current environment because `datasets`/`pyarrow` are missing.
- Existing models were saved under scikit-learn 1.7.2; loading with 1.8.0 emitted a version warning.
- The C feature extractor uses heuristics and token counts rather than a C AST parser.
