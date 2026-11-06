# Complexity Analysis

`estimate_complexity(source: str) -> ComplexityEstimate` in `gaoher/complexity_analysis/analyzer.py` uses Python's AST and symbolic input-size expressions. It reports estimated time and auxiliary-space growth; it is not a proof of exact Big-O for arbitrary Python.

## Result fields

`ComplexityEstimate` contains:

- `time_complexity`, `space_complexity`: strings such as `O(n) estimated` or `unknown`.
- `confidence`: currently `medium` when no uncertainty was recorded, otherwise `low`.
- `loop_nesting`, `recursion_detected`.
- `uncertain_patterns`: tuple of unsupported or inferred patterns.
- `time_reasoning`, `space_reasoning`: tuples explaining the estimate.

Invalid Python raises `ValueError` with an `Invalid Python source` diagnostic.

## Time estimates

The estimator adds sequential work and multiplies nested loop bounds. It preserves distinct simple dimensions, so nested loops over `n` and `m` produce `O(m*n) estimated`; two sequential loops over the same `n` simplify to `O(n) estimated`. Simple `range` arguments, named iterables, selected traversal operations (`sum`, `min`, `max`, `all`, `any`), sorting, heap operations, simple unit-step countdown loops, and selected logarithmic halving patterns are recognized. A loop over an opaque call such as `get_items()` remains `unknown`.

Recognized single-call recursive patterns with a simple base case can produce linear or logarithmic estimates. Unsupported recurrence shapes remain unknown or receive low confidence. Callee bodies are not expanded interprocedurally.

Examples verified by tests:

```python
estimate_complexity("def f(n, m):\n    for i in range(n):\n        for j in range(m):\n            pass\n").time_complexity
# 'O(m*n) estimated'

estimate_complexity("def f(items):\n    out = []\n    for x in items: out.append(x)\n").space_complexity
# 'O(items) estimated'
```

## Space estimates

Space is auxiliary space; recognized input parameter storage is excluded. The analyzer looks for local collection literals, comprehensions, collection constructors, and growth operations such as `append`, `extend`, dictionary insertion/update, and set growth. It estimates allocation scaling in recognized loops and accounts for recognized recursive call-stack depth. Collection lifetimes are approximated, and collection aliases or opaque mutations can change actual peak memory.

An unrelated constant collection literal does not automatically scale with a nearby loop. Unknown-size growth can make space unknown. Constant local work/space is reported as `O(1) estimated` under the estimator's model.

## Assumptions and limitations

- Bounds are syntactic estimates. Runtime values, exceptions, side effects, custom iterators, overloaded operators, decorators, dynamic dispatch, and aliases may alter cost.
- Unknown iterable or loop bounds are not guessed. `while` recognition uses simple comparison/update heuristics; it cannot prove progress or termination.
- Recursion recognition is limited to simple direct self-recursion. Mutual recursion, aliased calls, complex recurrences, and recursion under loops may be unsupported.
- Built-in costs are recognized only for selected operations and recognizable sizes. User-defined calls are not inlined; dynamic calls are uncertain.
- Space analysis is approximate and does not model allocator details, object size, garbage collection, or all temporary values.
- `confidence` is a coarse signal based on whether uncertainty notes were recorded, not a calibrated probability.

Use the reasoning and uncertainty fields together with both estimates; do not interpret an estimate as a runtime guarantee.
