# Feature Extraction and Hajera Compatibility

`gaoher/feature_extraction/features.py` implements Gaoher's static complexity features and adapters. Shared field definitions live in the root `feature_schema.py`; Gaoher validates against that contract without modifying it.

## Complexity fields

Call `extract_complexity_features(source, language="python")` for the validated mapping. The shared fields are `language`, `function_length`, `loop_count`, `max_loop_nesting`, `branch_count`, `recursion_count`, `function_call_count`, `ast_node_count`, `input_size_variable`, `loop_bound_type`, and `data_structure_types`.

Extraction semantics match Hajera's training feature function: `ast.For` and `ast.While` contribute loop counts (`AsyncFor` is excluded); branch counts include `If`, conditional expressions, and match cases; function calls and AST nodes are counted across the parsed source. Input-size, loop-bound, and data-structure classifications are AST heuristics. See the implementation and tests for their detailed recognition rules.

## 17-value vector and preprocessing

`complexity_vector_feature_names()` delegates ordering to Hajera. Its exact numeric vector order is:

1. `function_length`
2. `loop_count`
3. `max_loop_nesting`
4. `branch_count`
5. `recursion_count`
6. `function_call_count`
7. `ast_node_count`
8. `input_size_variable`
9–12. `loop_bound_type`: constant, input_dependent, logarithmic, unknown
13–17. `data_structure_types`: list, dict, set, heap, deque

Boolean values become integers and categorical fields become indicator values. Gaoher delegates vector conversion to Hajera; it does not keep a parallel implementation. `preprocess_complexity_features()` calls Hajera's `prep()` and returns a one-row NumPy matrix of shape `(1, 17)`. The integration test compared this adapter output against Hajera's actual `prep()` result.

## Defect feature adapter

`extract_defect_features(source, language="c")` calls Hajera's C extractor and validates the returned mapping against the shared defect schema. The `predict_c_defect()` bridge delegates actual prediction to Hajera's public API and returns the features together with its prediction.

## Public prediction bridge

`predict_python_complexity(source)` parses with Gaoher, validates and returns Gaoher's feature/vector/preprocessing artifacts, then delegates inference on the source to Hajera's public `predict_complexity()` function. Hajera remains authoritative for its inference-time feature extraction, preprocessing, model loading, and prediction. No estimator internals or prediction logic are duplicated.

Use standard imports from the repository root. Install dependencies from `hajera/requirements.txt` and hydrate model files with Git LFS. Verified Python complexity output for the tested sample: `O(log n)` / `logn`, confidence `0.257265`. Verified C defect output: risk probability `0.581239`, label `vulnerable`. Full details and commands are in [`ML_INTEGRATION_REPORT.md`](../ML_INTEGRATION_REPORT.md).
