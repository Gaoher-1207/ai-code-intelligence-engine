# Gaoher Analysis Modules

Gaoher provides static Python source analysis using Python's `ast` module, shared-schema feature extraction, and symbolic time and auxiliary-space growth estimates. Complexity results are heuristics and cannot prove behavior for arbitrary programs.

## Modules

- `code_analysis`: parser and per-function structural metrics.
- `feature_extraction`: shared-schema complexity features and compatibility adapters.
- `complexity_analysis`: conservative AST-based time and auxiliary-space estimates.
- `ml_integration`: public bridge to Hajera's trained Python complexity and C defect predictors.

See [Architecture](docs/ARCHITECTURE.md), [Parser](docs/PARSER.md), [Feature Extraction](docs/FEATURE_EXTRACTION.md), [Complexity Analysis](docs/COMPLEXITY_ANALYSIS.md), and [Testing](docs/TESTING.md).

## Run tests

From the repository root with Python 3.11, dependencies installed, and model LFS files hydrated:

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s gaoher\tests -v
```

## Example

```python
from gaoher.code_analysis import analyze_source
from gaoher.complexity_analysis import estimate_complexity
from gaoher.feature_extraction import extract_complexity_features

source = """\
def visit(n, m):
    for i in range(n):
        for j in range(m):
            pass
"""

parsed = analyze_source(source)
features = extract_complexity_features(source)
estimate = estimate_complexity(source)
```

Invalid Python is returned as a diagnostic by `analyze_source`; feature extraction and complexity estimation raise `ValueError` for invalid Python.

## ML integration

`gaoher.ml_integration.predict_python_complexity(source)` validates Gaoher's analysis and feature adapters, exposes Hajera's authoritative 17-value vector and NumPy preprocessing result, and delegates model inference to Hajera's public prediction API. `predict_c_defect(source)` uses Hajera's defect feature extraction and prediction APIs. Hajera retains ownership of its ML extraction, preprocessing, models, and predictions.

The normal runtime uses `hajera.training` imports from the repository root. Install `hajera/requirements.txt`, hydrate `hajera/models/*.joblib` with Git LFS, and run from the project root. Integration and prediction verification details are in [ML integration report](ML_INTEGRATION_REPORT.md).
