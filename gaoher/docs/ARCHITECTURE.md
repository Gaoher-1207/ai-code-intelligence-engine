# Architecture and Data Flow

## Pipeline

The public pipeline can be called in stages:

```text
Python source string
        |
        v
analyze_source(source)  ----> AnalysisResult / FunctionInfo
        |                           structural parser result
        +---- extract_complexity_features(source)
        |          |                shared-schema mapping
        |          +---- complexity_vector(features)
        |          |                Hajera's 17-number vector
        |          +---- preprocess_complexity_features(features)
        |                           Hajera's NumPy log1p transform
        +---- estimate_complexity(source)
                                   ComplexityEstimate
```

The parser and feature extractor parse source independently. This keeps their public interfaces simple, but means the full pipeline currently reparses source rather than passing a shared AST object between stages.

## Package responsibilities

### `gaoher.code_analysis`

`analyze_source(source: str) -> AnalysisResult` parses Python and reports validity, diagnostics, aggregate structural metrics, AST node count, and a tuple of per-function `FunctionInfo` records. Function records contain source lines, loop count and nesting, condition count, call names, and detected direct recursion count. Nested functions and class methods receive independent metrics.

### `gaoher.feature_extraction`

`extract_complexity_features(source: str, language: str = "python") -> dict[str, Any]` extracts the complexity feature mapping and invokes `feature_schema.validate_complexity_features` before returning it. Invalid source raises `ValueError`.

`complexity_vector_feature_names()` and `complexity_vector(features)` delegate to Hajera's `schema_features` module. `preprocess_complexity_features(features)` delegates vector creation and `prep` there as well. `extract_defect_features(source, language="c")` delegates C defect feature extraction and validates the returned mapping with the shared schema. These adapter functions require the repository root and Hajera module to be importable; numerical preprocessing additionally requires NumPy.

### `gaoher.complexity_analysis`

`estimate_complexity(source: str) -> ComplexityEstimate` returns estimated time and auxiliary-space strings, confidence, loop nesting, recursion detection, uncertainty descriptions, and reasoning tuples. It raises `ValueError` on invalid Python.

## Dependency direction

The parser depends on `ast` and dataclasses. Feature extraction uses the parser and the shared root schema; its Hajera imports are lazy. The complexity analyzer uses the parser and the feature module's `_halves` loop-bound heuristic, plus the standard library. Public imports are re-exported from each package's `__init__.py`.

## Scope

This is a static analysis layer. It does not load trained models, make predictions, parse C source, or perform frontend/backend integration. Defect feature extraction is only an adapter to Hajera's implementation.
