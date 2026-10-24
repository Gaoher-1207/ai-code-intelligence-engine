# Results

Both models are baselines. All numbers are on held-out test data.

| Model | Data | Test size | Accuracy | Macro F1 |
|---|---|---|---|---|
| Time complexity (Python, 7 classes) | CodeComplex, split by problem | 905 | 51.0% | 0.51 |
| Defect risk (C, insecure vs secure) | CodeXGLUE / Devign test split | 2732 | 63.7% | 0.63 |

Confusion matrices: `complexity_confusion.png`, `defect_confusion.png`.

## What we learned
- The schema fields alone are weak predictors: 36.5% for complexity (7 classes) and 55.1% for defects
  (about the same as always guessing "secure"). The code text carries most of the signal.
- An earlier baseline that used only our own loop and nesting counts plus code text scored 53.4% (complexity)
  and 63.0% (defect), within noise of the schema-based models.
- Complexity: best on constant and quadratic programs, weakest on `np` and `logn`.
- Defect: `target=True` means insecure code (dataset card). Recall for insecure code is only about 54%.

## Limits
- Static features and text cannot prove a bug or guarantee exact Big-O.
- Defect labels come from the Devign dataset (C functions from open-source projects); results may not carry
  over to other code styles.
- For C, `ast_node_count` is a token count and several counts are pattern-matching estimates until a C parser
  is added.
