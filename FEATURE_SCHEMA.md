# Feature Schema

## A. Complexity model inputs

| Field | Meaning |
|---|---|
| `language` | Source language, e.g. Python or C |
| `function_length` | Lines in the function |
| `loop_count` | Number of loops |
| `max_loop_nesting` | Maximum nested-loop depth |
| `branch_count` | Conditional branches |
| `recursion_count` | Detected recursive calls |
| `function_call_count` | Calls to other functions |
| `ast_node_count` | AST node count |
| `input_size_variable` | Whether an input-size variable such as `n` is detected |
| `loop_bound_type` | Constant, input-dependent, logarithmic, or unknown |
| `data_structure_types` | Detected structures such as lists, arrays, trees, or graphs |

Training label: `complexity_class` (for example `O(n)` or `O(n^2)`). Never include it in the input matrix.

## B. Defect-risk model inputs

| Field | Meaning |
|---|---|
| `language` | Source language |
| `source_code` | Function source, for code-aware models/feature extraction |
| `function_length` | Lines in the function |
| `branch_count` | Conditional branches |
| `loop_count` | Number of loops |
| `pointer_usage` | Pointer-related operations; language-specific |
| `memory_operation_count` | Allocation/deallocation operations; language-specific |
| `dangerous_api_count` | Matches to a documented risky-API list |
| `exception_handling_count` | Exception-handling constructs; language-specific |
| `null_check_count` | Null/None/reference checks |
| `cyclomatic_complexity` | Control-flow complexity estimate |
| `duplicate_code_ratio` | Estimated duplicate-code proportion, from 0 to 1 |
| `uninitialized_variable_count` | Static-analysis estimate |
| `resource_cleanup_count` | Resource-release operations |
| `ast_node_count` | AST node count |

Training label: `target`. Inspect the selected dataset's actual label mapping; do not assume what 0 and 1 mean.

## C. Model outputs (not input features)
- `predicted_complexity`: predicted complexity class.
- `confidence`: confidence score, only if meaningful for the model.
- `predicted_label`: defect-risk class.
- `risk_probability`: probability for the documented positive class.

## Coordination rules
1. You (DSA teammate) implement parsing and feature extraction.
2. Hajera implements data preparation, training, validation, and evaluation.
3. Agree on this schema before coding the modules.
4. Do not replace unavailable values with zero unless zero is the real value; document a missing-value policy.
5. Keep labels and prediction outputs out of model inputs.
6. Use repository-separated or project-separated test data where possible to reduce leakage.
7. Static metrics do not prove a bug or guarantee exact Big-O for arbitrary programs.
