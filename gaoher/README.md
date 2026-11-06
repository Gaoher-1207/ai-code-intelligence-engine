# Gaoher Analysis Modules

Gaoher's modules provide static Python source analysis for the AI Code Intelligence Engine. They parse source with Python's `ast` module, expose structural features under the shared feature schema, and estimate symbolic time and auxiliary-space growth. These results are heuristics: they do not prove runtime behavior or exact complexity for arbitrary programs.

## Modules

- `code_analysis`: parser and per-function structural metrics.
- `feature_extraction`: shared-schema complexity features and adapters to Hajera's existing vector, preprocessing, and defect extractors.
- `complexity_analysis`: conservative AST-based time and auxiliary-space estimates.
- `tests`: unit and pipeline tests for the modules above.

See [Architecture](docs/ARCHITECTURE.md), [Parser](docs/PARSER.md), [Feature Extraction](docs/FEATURE_EXTRACTION.md), [Complexity Analysis](docs/COMPLEXITY_ANALYSIS.md), and [Testing](docs/TESTING.md).

## Requirements and running tests

The parser and analyzer use Python's standard library. Feature extraction imports the repository's root `feature_schema.py`. The Hajera adapters also import `hajera.training.schema_features`, whose numerical preprocessing uses NumPy. No separate Gaoher dependency manifest is present.

From the repository root, run the existing suite with:

```powershell
python -m unittest discover -s gaoher/tests -v
```

This command runs the tests using the active Python interpreter; activate the project's supported environment first if it has been prepared. Tests that need NumPy are skipped when it is unavailable.

## Usage

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

print(parsed.functions[0].loop_count)       # 2
print(features["loop_count"])               # 2
print(estimate.time_complexity)              # O(m*n) estimated
print(estimate.space_complexity)              # O(1) estimated
```

Invalid Python is returned as a diagnostic by `analyze_source`. The feature extractor and complexity estimator raise `ValueError` for invalid Python. See each module guide for return fields, assumptions, and limitations.

## Integration status

The complexity extractor is tested against Hajera's real feature extractor and ordered 17-value vector. The defect adapter delegates to Hajera's real extractor. This does not establish trained-model compatibility: model loading and predictions are outside Gaoher's modules. In the recorded validation environment NumPy was unavailable, so the real numerical preprocessing test was skipped. See [Testing](docs/TESTING.md) for the exact distinction between real, mocked, and skipped checks.
