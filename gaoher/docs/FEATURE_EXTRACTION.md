# Feature Extraction and Hajera Compatibility

`gaoher/feature_extraction/features.py` exposes the shared complexity feature contract and adapters for existing Hajera functionality. The shared schema is defined in repository-root `feature_schema.py`; Gaoher reads and validates it but does not modify it.

## Complexity feature mapping

Call:

```python
extract_complexity_features(source: str, language: str = "python") -> dict[str, Any]
```

The returned keys are `language`, `function_length`, `loop_count`, `max_loop_nesting`, `branch_count`, `recursion_count`, `function_call_count`, `ast_node_count`, `input_size_variable`, `loop_bound_type`, and `data_structure_types`. The validator runs before the mapping is returned. The count and classification semantics intentionally match Hajera's training-time extractor:

- `loop_count` counts `ast.For` and `ast.While`; `ast.AsyncFor` is excluded for parity.
- `max_loop_nesting` uses nested `For` and `While` AST nodes.
- `branch_count` counts `If`, conditional expressions, and `match` cases.
- Comprehension generator/filters are not included in the loop or branch feature counts.
- Function calls and AST nodes count across the entire parsed source unit.
- Recursion counts direct calls and conventional `self`/`cls` self-method calls, not mutual or aliased recursion.
- Input-size, loop-bound, and data-structure classifications are AST heuristics. Recognized structures include list, dict (including `defaultdict` and `Counter`), set, heap, and deque.

The feature extractor accepts `language` as an argument and validates the returned mapping; it parses the supplied source as Python regardless of that label.

## 17-value vector adapter

`complexity_vector_feature_names()` returns Hajera's authoritative ordering. `complexity_vector(features)` validates the mapping and delegates numeric conversion to Hajera. The exact order is:

1. `function_length`
2. `loop_count`
3. `max_loop_nesting`
4. `branch_count`
5. `recursion_count`
6. `function_call_count`
7. `ast_node_count`
8. `input_size_variable`
9. `loop_bound_type=constant`
10. `loop_bound_type=input_dependent`
11. `loop_bound_type=logarithmic`
12. `loop_bound_type=unknown`
13. `data_structure_types=list`
14. `data_structure_types=dict`
15. `data_structure_types=set`
16. `data_structure_types=heap`
17. `data_structure_types=deque`

The final nine values are indicator values (0 or 1); the boolean input-size field becomes an integer. The Gaoher adapter delegates the ordering and vector conversion rather than maintaining a second implementation.

`preprocess_complexity_features(features)` calls Hajera's `complexity_vector` then `prep([vector])`. Hajera's `prep` applies NumPy `log1p` to a float array; the expected shape is `(1, 17)`. This is a delegated contract, not an independently reproduced transform.

## Defect adapter

`extract_defect_features(source: str, language: str = "c")` calls `hajera.training.schema_features.extract_defect_features` and validates the returned mapping with `validate_defect_features`. Gaoher does not duplicate the defect heuristics. This helper returns features only; it neither loads a defect model nor predicts a risk score.

## Dependencies and verification boundary

The shared schema must be importable from the repository root. Hajera adapter calls require `hajera.training.schema_features` to be importable. That module imports NumPy at import time. The previously recorded comparison tests invoked Hajera's real pure-Python extractors and vector functions with an import-only NumPy shim because NumPy was absent; those checks do not verify numerical preprocessing. A separate mock test checks preprocessing delegation only. No trained-model compatibility is claimed.
