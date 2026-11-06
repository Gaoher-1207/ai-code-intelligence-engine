# Testing and Validation

The suite is `gaoher/tests/test_analysis.py` and uses Python's built-in `unittest`. From the repository root:

```powershell
python -m unittest discover -s gaoher/tests -v
```

## Coverage

Tests exercise parser behavior, independent nested function and method metrics, async functions, nested loops, comprehensions, direct and conventional method recursion, invalid syntax, shared-schema feature validation, Hajera feature/vector/defect adapters, and time/space estimates. End-to-end tests call the parser, complexity feature extractor, and complexity analyzer on the same source examples.

The complexity cases include single and nested loops, sequential loops, distinct input dimensions, unknown iterables, while-loop heuristics, recognized recursion, collection growth and literals, comprehensions, built-in operations, and unsupported calls.

## Recorded validation results

In the validation environment used for Phase 4, the complete suite ran **53 tests: 52 passed, 0 failed, 1 skipped**. The skipped test is the comparison against Hajera's actual numerical `prep` result, because NumPy was not installed in that environment.

The complexity feature, 17-value vector, and defect extractor comparisons called Hajera's real source functions. Tests used an import-only NumPy shim to load Hajera's module when NumPy was absent; no extractor or vector result was mocked in those comparisons. A distinct mocked test checks that Gaoher delegates preprocessing correctly. These are different checks and should not be described as a real numerical preprocessing test.

The actual preprocessing operation, trained model loading, and model predictions were not verified. Passing the feature/vector adapter tests does not establish trained-model compatibility.

## Interpretation

Test results apply to the source examples and Python environment in which the suite ran. They do not establish correctness for arbitrary Python programs or runtime complexity. Re-run the command in the intended project environment when dependencies are installed; do not treat the historical result above as a current environment check.
