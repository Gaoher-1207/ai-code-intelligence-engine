# ML module (Hajera)

Machine-learning part of the AI Code Intelligence Engine. It follows the shared feature contract in
`feature_schema.py` / `FEATURE_SCHEMA.md`.

## What it does
| Model | Input | Output | Data |
|---|---|---|---|
| Time complexity (Python) | Python code | complexity class (O(1), O(log n), O(n), O(n log n), O(n^2), O(n^3), exponential/NP-hard) + confidence | CodeComplex (Python part), split by problem |
| Defect risk (C) | C function | `risk_probability` (insecure) + label | CodeXGLUE defect detection (Devign), official train/test split |

Both models combine the schema features with a TF-IDF representation of the code text, then use logistic regression.
Results and limits: see `evaluation/RESULTS.md`. These are baselines (about 51% for complexity over 7 classes and
about 64% for defects), not production-grade predictors.

## Folders
- `data/`: place for local data (raw datasets are NOT committed; see below)
- `training/`: `schema_features.py` (schema feature extraction), `train_schema_models.py`, `predict.py`,
  `make_report.py`, `test_ml.py`, `01_train_models.ipynb` (first baselines)
- `models/`: trained models (`*_schema_lr.joblib`)
- `evaluation/`: reports, confusion matrices, `RESULTS.md`

## Setup
```
conda create -n codeintel python=3.10 -y
conda activate codeintel
pip install pandas scikit-learn joblib datasets matplotlib pyarrow scipy streamlit pytest
```

## Data
- CodeComplex: `git clone https://github.com/sybaik1/CodeComplex.git D:/projects/datasets/CodeComplex`
  (file used: `data/python_data.jsonl`; license CC BY-NC-ND 4.0, non-commercial use).
- CodeXGLUE defect detection: loaded from Hugging Face (`google/code_x_glue_cc_defect_detection`).
  `target=True` (1) means insecure code. The dataset lists "No known license".

## Use
```
cd hajera/training
python train_schema_models.py     # train both models
python make_report.py             # confusion matrices + evaluation/RESULTS.md
python predict.py                 # quick demo
cd ../..
pytest hajera/training -q         # tests
streamlit run frontend/app.py     # web demo
```
`predict.py` exposes `predict_complexity(python_code)` and `predict_defect(c_code)`.

## Notes for the shared schema
- `schema_features.py` is a stand-in extractor. Teammate's extractor can replace `extract_complexity_features`
  and `extract_defect_features`, as long as they return the schema fields.
- For C, `ast_node_count` is a token count and several counts (pointers, uninitialized variables, null checks)
  are pattern-matching estimates, not parser-based.
- Static metrics cannot prove a bug or guarantee exact Big-O.