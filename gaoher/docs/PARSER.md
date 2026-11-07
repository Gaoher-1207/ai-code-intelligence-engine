# Python AST Parser

The parser in `gaoher/code_analysis/parser.py` uses `ast.parse` and returns immutable dataclasses. It reports syntax structure, not execution traces.

## Public API

```python
analyze_source(source: str) -> AnalysisResult
```

`AnalysisResult` fields are:

- `valid`, `error`: parse status and diagnostic.
- `functions`: tuple of `FunctionInfo` records, including async functions and methods.
- `loop_count`, `max_loop_nesting`, `condition_count`: source-unit totals.
- `function_calls`: tuple of readable call names found across the module.
- `recursive_call_count`: total calls recognized as recursive.
- `ast_node_count`: number of nodes yielded by `ast.walk` for valid source.

Each `FunctionInfo` contains `name`, `line`, `end_line`, `loop_count`, `max_loop_nesting`, `condition_count`, `function_calls`, and `recursive_calls`. Per-function structural metrics exclude nested function, class, and lambda bodies; nested function definitions and methods are also reported as their own records.

## Supported syntax

The visitor recognizes synchronous and asynchronous function definitions; `for`, `async for`, and `while` loops; nested loops; `if`, conditional expressions, and `match` cases; calls; and list, set, dictionary, and generator comprehensions. Comprehension generators contribute loop layers and their filters contribute conditions. Calls are represented by dotted names where possible (for example, `obj.method`); dynamic call expressions use placeholder names.

Direct self-calls and conventional `self.method(...)` / `cls.method(...)` self-recursion are counted. Mutual recursion, recursion through aliases, and dynamic dispatch are not inferred.

## Invalid syntax

`analyze_source` catches `SyntaxError`, `ValueError`, and `TypeError` from parsing and returns `valid=False`, an error string, empty function/call results, and zero counts. Example:

```python
result = analyze_source("def broken(:")
assert result.valid is False
assert result.functions == ()
```

For valid input, a small verified example:

```python
result = analyze_source("def f(n):\n    for item in n:\n        pass\n")
result.valid                         # True
result.functions[0].name             # 'f'
result.functions[0].loop_count       # 1
result.functions[0].max_loop_nesting # 1
```

Aggregate totals include all nested scopes in the source unit. The parser does not infer whether a call executes, whether a loop terminates, or the runtime size of an iterable.
