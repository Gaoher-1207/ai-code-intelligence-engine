"""Unit tests for Gaoher's parser, schema adapter, and complexity estimates."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

from gaoher.code_analysis import analyze_source
from gaoher.complexity_analysis import estimate_complexity
from gaoher.feature_extraction import (
    complexity_vector,
    complexity_vector_feature_names,
    extract_complexity_features,
    extract_defect_features,
    preprocess_complexity_features,
)
from feature_schema import validate_complexity_features


def _real_hajera_schema_features():
    """Import Hajera's real pure-Python extractors, shimming only absent NumPy."""
    try:
        return importlib.import_module("hajera.training.schema_features")
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        numpy_stub = types.ModuleType("numpy")
        with patch.dict(sys.modules, {"numpy": numpy_stub}):
            return importlib.import_module("hajera.training.schema_features")


class ParserTests(unittest.TestCase):
    def test_finds_functions_nested_loops_conditions_calls_and_recursion(self) -> None:
        result = analyze_source(
            "def visit(n):\n"
            "    if n <= 1:\n"
            "        return helper(n)\n"
            "    for i in range(n):\n"
            "        for j in range(n):\n"
            "            visit(n - 1)\n"
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.loop_count, 2)
        self.assertEqual(result.max_loop_nesting, 2)
        self.assertEqual(result.condition_count, 1)
        self.assertEqual(result.recursive_call_count, 1)
        self.assertIn("helper", result.function_calls)
        self.assertEqual(result.functions[0].name, "visit")

    def test_nested_function_metrics_are_independent(self) -> None:
        result = analyze_source(
            "def outer():\n"
            "    before()\n"
            "    def inner():\n"
            "        if ready():\n"
            "            for item in items():\n"
            "                consume(item)\n"
            "    after()\n"
        )
        outer, inner = result.functions
        self.assertEqual((outer.loop_count, outer.condition_count), (0, 0))
        self.assertEqual(outer.function_calls, ("before", "after"))
        self.assertEqual((inner.loop_count, inner.condition_count), (1, 1))
        self.assertEqual(inner.function_calls, ("ready", "items", "consume"))

    def test_class_methods_are_analyzed_without_enclosing_scope_mixing(self) -> None:
        result = analyze_source(
            "def enclosing():\n"
            "    class Worker:\n"
            "        def run(self):\n"
            "            for item in items:\n"
            "                while active(item):\n"
            "                    process(item)\n"
            "        def idle(self):\n"
            "            return ping()\n"
        )
        enclosing, run, idle = result.functions
        self.assertEqual((enclosing.loop_count, enclosing.function_calls), (0, ()))
        self.assertEqual((run.loop_count, run.max_loop_nesting), (2, 2))
        self.assertEqual(run.function_calls, ("active", "process"))
        self.assertEqual((idle.loop_count, idle.function_calls), (0, ("ping",)))

    def test_multiple_and_async_functions_have_independent_metrics(self) -> None:
        result = analyze_source(
            "def first():\n"
            "    for item in items:\n"
            "        use(item)\n"
            "async def second():\n"
            "    async for item in stream():\n"
            "        await use_async(item)\n"
        )
        first, second = result.functions
        self.assertEqual((first.loop_count, first.function_calls), (1, ("use",)))
        self.assertEqual((second.loop_count, second.function_calls),
                         (1, ("stream", "use_async")))

    def test_self_and_cls_recursive_calls_are_detected(self) -> None:
        result = analyze_source(
            "class Node:\n"
            "    def walk(self, child):\n"
            "        if child:\n"
            "            self.walk(child.next)\n"
            "    @classmethod\n"
            "    def build(cls, count):\n"
            "        if count:\n"
            "            return cls.build(count - 1)\n"
        )
        walk, build = result.functions
        self.assertEqual(walk.recursive_calls, 1)
        self.assertEqual(build.recursive_calls, 1)
        self.assertEqual(result.recursive_call_count, 2)

    def test_list_set_and_dictionary_comprehensions_are_counted(self) -> None:
        result = analyze_source(
            "values = [x for x in source if keep(x)]\n"
            "unique = {x for x in source}\n"
            "mapping = {x: convert(x) for x in source}\n"
        )
        self.assertEqual(result.loop_count, 3)
        self.assertEqual(result.condition_count, 1)
        self.assertEqual(result.function_calls.count("keep"), 1)
        self.assertEqual(result.function_calls.count("convert"), 1)

    def test_calls_in_nested_functions_are_counted_once_per_scope(self) -> None:
        result = analyze_source(
            "def outer():\n"
            "    def inner():\n"
            "        nested_call()\n"
            "    outer_call()\n"
            "    inner()\n"
        )
        outer, inner = result.functions
        self.assertEqual(outer.function_calls, ("outer_call", "inner"))
        self.assertEqual(inner.function_calls, ("nested_call",))
        self.assertCountEqual(result.function_calls,
                              ("nested_call", "outer_call", "inner"))

    def test_invalid_syntax_returns_diagnostic_without_raising(self) -> None:
        result = analyze_source("def broken(:\n pass")
        self.assertFalse(result.valid)
        self.assertIn("line 1", result.error or "")
        self.assertEqual(result.functions, ())

    def test_comprehension_generators_count_as_nested_loops(self) -> None:
        result = analyze_source("values = [x + y for x in xs for y in ys if y > 0]")
        self.assertEqual(result.loop_count, 2)
        self.assertEqual(result.max_loop_nesting, 2)
        self.assertEqual(result.condition_count, 1)


class EndToEndPipelineTests(unittest.TestCase):
    """Exercise parser -> shared features -> complexity estimate as one path."""

    def test_simple_function_pipeline_and_schema_validation(self) -> None:
        source = "def identity(value):\n    return value\n"
        parsed = analyze_source(source)
        features = extract_complexity_features(source)
        estimate = estimate_complexity(source)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.functions[0].name, "identity")
        validate_complexity_features(features)
        self.assertEqual(estimate.time_complexity, "O(1) estimated")
        self.assertEqual(estimate.space_complexity, "O(1) estimated")

    def test_nested_loops_pipeline_preserves_independent_dimensions(self) -> None:
        source = (
            "def visit(n, m):\n"
            "    for i in range(n):\n"
            "        for j in range(m):\n"
            "            consume(i, j)\n"
        )
        parsed = analyze_source(source)
        features = extract_complexity_features(source)
        estimate = estimate_complexity(source)
        self.assertEqual((parsed.functions[0].loop_count,
                          parsed.functions[0].max_loop_nesting), (2, 2))
        self.assertEqual(features["loop_count"], 2)
        validate_complexity_features(features)
        self.assertEqual(estimate.time_complexity, "O(m*n) estimated")

    def test_recursive_function_pipeline(self) -> None:
        source = "def descend(n):\n    if n: return descend(n - 1)\n"
        parsed = analyze_source(source)
        features = extract_complexity_features(source)
        estimate = estimate_complexity(source)
        self.assertEqual(parsed.functions[0].recursive_calls, 1)
        self.assertEqual(features["recursion_count"], 1)
        validate_complexity_features(features)
        self.assertEqual(estimate.time_complexity, "O(n) estimated")
        self.assertEqual(estimate.space_complexity, "O(n) estimated")

    def test_async_function_pipeline_keeps_training_feature_semantics(self) -> None:
        source = (
            "async def consume_all(items):\n"
            "    async for item in items:\n"
            "        await process(item)\n"
        )
        parsed = analyze_source(source)
        features = extract_complexity_features(source)
        estimate = estimate_complexity(source)
        self.assertEqual(parsed.functions[0].loop_count, 1)
        # Hajera's current training extractor counts ast.For and ast.While,
        # but not ast.AsyncFor; Gaoher preserves that feature contract.
        self.assertEqual(features["loop_count"], 0)
        validate_complexity_features(features)
        self.assertEqual(estimate.time_complexity, "O(items) estimated")

    def test_invalid_python_is_rejected_consistently_across_pipeline(self) -> None:
        source = "def invalid(:"
        self.assertFalse(analyze_source(source).valid)
        with self.assertRaisesRegex(ValueError, "Invalid Python source"):
            extract_complexity_features(source)
        with self.assertRaisesRegex(ValueError, "Invalid Python source"):
            estimate_complexity(source)

    def test_collection_growth_pipeline_reports_auxiliary_space(self) -> None:
        source = (
            "def build(items):\n"
            "    output = []\n"
            "    lookup = {}\n"
            "    for item in items:\n"
            "        output.append(item)\n"
            "        lookup[item] = item\n"
            "    return output, lookup\n"
        )
        parsed = analyze_source(source)
        features = extract_complexity_features(source)
        estimate = estimate_complexity(source)
        self.assertEqual(parsed.functions[0].loop_count, 1)
        self.assertIn("list", features["data_structure_types"])
        self.assertIn("dict", features["data_structure_types"])
        validate_complexity_features(features)
        self.assertEqual(estimate.time_complexity, "O(items) estimated")
        self.assertEqual(estimate.space_complexity, "O(items) estimated")

    def test_unknown_dynamic_construct_is_reported_end_to_end(self) -> None:
        source = "def dynamic():\n    for item in get_items(): pass\n"
        parsed = analyze_source(source)
        features = extract_complexity_features(source)
        estimate = estimate_complexity(source)
        self.assertTrue(parsed.valid)
        validate_complexity_features(features)
        self.assertEqual(estimate.time_complexity, "unknown")
        self.assertTrue(estimate.uncertain_patterns)


class FeatureTests(unittest.TestCase):
    def test_complexity_features_match_shared_field_names_and_types(self) -> None:
        features = extract_complexity_features(
            "def total(values):\n    return [v for v in values if v > 0]\n"
        )
        expected = {
            "language", "function_length", "loop_count", "max_loop_nesting",
            "branch_count", "recursion_count", "function_call_count", "ast_node_count",
            "input_size_variable", "loop_bound_type", "data_structure_types",
        }
        self.assertEqual(set(features), expected)
        self.assertEqual(features["language"], "python")
        self.assertIsInstance(features["input_size_variable"], bool)
        self.assertIn("list", features["data_structure_types"])
        self.assertTrue(all(isinstance(features[key], int) and features[key] >= 0 for key in (
            "function_length", "loop_count", "max_loop_nesting", "branch_count",
            "recursion_count", "function_call_count", "ast_node_count")))
        validate_complexity_features(features)

    def test_complexity_features_match_real_hajera_extractor(self) -> None:
        hajera = _real_hajera_schema_features()
        examples = (
            "def solve(data):\n    return len(data)\n",
            "def halve(n):\n    while n > 1:\n        n //= 2\n",
            "import sys\ndef read():\n    value = input()\n    return len(sys.stdin.readline())\n",
            "from collections import Counter, defaultdict\n"
            "def count(items):\n    return Counter(items), defaultdict(list)\n",
            "from collections import deque\nfrom heapq import heappush\n"
            "def work(xs):\n    q = deque(xs)\n    heappush([], 1)\n    return [x for x in q if x]\n",
            "async def run(stream):\n    async for item in stream:\n        if item:\n            await handle(item)\n",
            "values = [x for x in xs if keep(x)]\n"
            "unique = {x for x in xs}\n"
            "mapping = {x: x for x in xs}\n",
        )
        for source in examples:
            with self.subTest(source=source):
                self.assertEqual(extract_complexity_features(source),
                                 hajera.extract_complexity_features(source))

    def test_complexity_extractor_validates_language_argument(self) -> None:
        with self.assertRaisesRegex(ValueError, "language"):
            extract_complexity_features("pass", language="")

    def test_vector_adapter_matches_real_hajera_order_and_dimension(self) -> None:
        hajera = _real_hajera_schema_features()
        features = extract_complexity_features(
            "def solve(values):\n    for value in values:\n        consume(value)\n"
        )
        self.assertEqual(complexity_vector_feature_names(),
                         tuple(hajera.COMPLEXITY_VECTOR_FEATURE_NAMES))
        expected = hajera.complexity_vector(features)
        actual = complexity_vector(features)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 17)

    def test_preprocessing_adapter_delegates_vector_and_transform(self) -> None:
        features = extract_complexity_features("pass")
        vector = list(range(17))
        matrix = object()
        fake = types.SimpleNamespace(
            complexity_vector=lambda values: vector,
            prep=lambda rows: self.assertEqual(rows, [vector]) or matrix,
        )
        with patch("gaoher.feature_extraction.features._hajera_schema_features",
                   return_value=fake):
            self.assertIs(preprocess_complexity_features(features), matrix)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
    def test_preprocessing_adapter_matches_real_hajera_prep(self) -> None:
        hajera = _real_hajera_schema_features()
        features = extract_complexity_features("def solve(n):\n    return range(n)\n")
        expected = hajera.prep([hajera.complexity_vector(features)])
        actual = preprocess_complexity_features(features)
        self.assertEqual(actual.shape, (1, 17))
        self.assertTrue((actual == expected).all())

    def test_invalid_source_is_rejected_by_feature_extractor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Python source"):
            extract_complexity_features("def broken(:")

    def test_defect_adapter_delegates_and_validates_existing_hajera_result(self) -> None:
        valid_features = {"language": "c", "source_code": "int main(){}",
                          "function_length": 1, "branch_count": 0, "loop_count": 0,
                          "pointer_usage": 0, "memory_operation_count": 0,
                          "dangerous_api_count": 0, "exception_handling_count": 0,
                          "null_check_count": 0, "cyclomatic_complexity": 1,
                          "duplicate_code_ratio": 0.0, "uninitialized_variable_count": 0,
                          "resource_cleanup_count": 0, "ast_node_count": 4}
        fake = types.SimpleNamespace(
            extract_defect_features=lambda source, language="c": valid_features
        )
        with patch("gaoher.feature_extraction.features._hajera_schema_features",
                   return_value=fake), \
                patch("feature_schema.validate_defect_features") as validate:
            self.assertEqual(extract_defect_features("int main(){}"), valid_features)
            validate.assert_called_once_with(valid_features)

    def test_defect_adapter_matches_real_hajera_extractor(self) -> None:
        hajera = _real_hajera_schema_features()
        source = "int copy(char *dst, const char *src) { strcpy(dst, src); return 0; }"
        self.assertEqual(extract_defect_features(source),
                         hajera.extract_defect_features(source))


class ComplexityTests(unittest.TestCase):
    def test_input_dependent_nested_loops_report_polynomial_estimate(self) -> None:
        estimate = estimate_complexity(
            "def f(n):\n    for i in range(n):\n        for j in range(n):\n            pass\n"
        )
        self.assertEqual(estimate.time_complexity, "O(n^2) estimated")
        self.assertEqual(estimate.loop_nesting, 2)

    def test_while_loop_is_reported_as_uncertain(self) -> None:
        estimate = estimate_complexity("def f(n):\n    while n > 0:\n        n -= 1\n")
        self.assertEqual(estimate.time_complexity, "O(n) estimated")
        self.assertTrue(estimate.uncertain_patterns)

    def test_wrong_direction_while_update_is_unknown(self) -> None:
        estimate = estimate_complexity("def f(n):\n    while n > 0:\n        n += 1\n")
        self.assertIn("unknown", estimate.time_complexity)

    def test_constant_outer_loop_does_not_inflate_variable_growth_degree(self) -> None:
        estimate = estimate_complexity(
            "def f(n):\n    for _ in range(10):\n        for i in range(n):\n            pass\n"
        )
        self.assertEqual(estimate.time_complexity, "O(n) estimated")

    def test_comprehension_growth_is_reported(self) -> None:
        estimate = estimate_complexity("def f(xs, ys):\n    return [(x, y) for x in xs for y in ys]\n")
        self.assertIn("O(xs*ys)", estimate.time_complexity)
        self.assertIn("O(xs*ys)", estimate.space_complexity)

    def test_simple_linear_recursion_estimates_time_and_stack_space(self) -> None:
        estimate = estimate_complexity("def f(n):\n    if n: return f(n - 1)\n")
        self.assertTrue(estimate.recursion_detected)
        self.assertEqual(estimate.time_complexity, "O(n) estimated")
        self.assertEqual(estimate.space_complexity, "O(n) estimated")
        self.assertTrue(estimate.uncertain_patterns)

    def test_halving_recursion_estimates_logarithmic_time_and_stack(self) -> None:
        estimate = estimate_complexity(
            "def f(n):\n    if n <= 1: return 1\n    return f(n // 2)\n"
        )
        self.assertEqual(estimate.time_complexity, "O(log(n)) estimated")
        self.assertEqual(estimate.space_complexity, "O(log(n)) estimated")

    def test_recursion_without_recognized_base_case_stays_unknown(self) -> None:
        estimate = estimate_complexity("def f(n):\n    return f(n - 1)\n")
        self.assertIn("unknown", estimate.time_complexity)
        self.assertIn("unknown", estimate.space_complexity)

    def test_linear_recursion_with_linear_frame_work_and_storage_is_quadratic(self) -> None:
        estimate = estimate_complexity(
            "def f(n):\n    if n <= 0: return []\n    values = []\n"
            "    for i in range(n): values.append(i)\n    f(n - 1)\n    return values\n"
        )
        self.assertEqual(estimate.time_complexity, "O(n^2) estimated")
        self.assertEqual(estimate.space_complexity, "O(n^2) estimated")

    def test_recursion_nested_in_loop_is_left_unknown(self) -> None:
        estimate = estimate_complexity(
            "def f(n):\n    if n <= 1: return 1\n"
            "    for item in range(n):\n        f(n - 1)\n"
        )
        self.assertIn("unknown", estimate.time_complexity)
        self.assertTrue(any("nested in iterations" in item
                            for item in estimate.uncertain_patterns))

    def test_constant_operations_and_auxiliary_space(self) -> None:
        estimate = estimate_complexity("def f(x):\n    y = x + 1\n    return y\n")
        self.assertEqual(estimate.time_complexity, "O(1) estimated")
        self.assertEqual(estimate.space_complexity, "O(1) estimated")
        self.assertTrue(estimate.time_reasoning)
        self.assertTrue(estimate.space_reasoning)
        self.assertEqual(estimate.confidence, "medium")

    def test_empty_builtin_collections_have_constant_construction_time(self) -> None:
        for source in ("def f(): return list()\n", "def f(): return dict()\n",
                       "def f(): return set()\n"):
            with self.subTest(source=source):
                self.assertEqual(estimate_complexity(source).time_complexity, "O(1) estimated")

    def test_single_loop_and_append_growth(self) -> None:
        estimate = estimate_complexity(
            "def f(items):\n    result = []\n    for item in items:\n        result.append(item)\n    return result\n"
        )
        self.assertEqual(estimate.time_complexity, "O(items) estimated")
        self.assertEqual(estimate.space_complexity, "O(items) estimated")

    def test_input_mutation_is_excluded_but_reallocated_parameter_is_auxiliary(self) -> None:
        mutation = estimate_complexity("def f(items):\n    items.append(1)\n")
        reallocation = estimate_complexity(
            "def f(items, n):\n    items = []\n"
            "    for i in range(n): items.append(i)\n"
        )
        self.assertEqual(mutation.space_complexity, "O(1) estimated")
        self.assertEqual(reallocation.space_complexity, "O(n) estimated")

    def test_nested_loops_preserve_independent_input_sizes(self) -> None:
        estimate = estimate_complexity(
            "def f(n, m):\n    for i in range(n):\n        for j in range(m):\n            pass\n"
        )
        self.assertEqual(estimate.time_complexity, "O(m*n) estimated")

    def test_sequential_loops_over_same_input_simplify(self) -> None:
        estimate = estimate_complexity(
            "def f(n):\n    for i in range(n): pass\n    for j in range(n): pass\n"
        )
        self.assertEqual(estimate.time_complexity, "O(n) estimated")

    def test_unknown_iterable_bound_stays_unknown(self) -> None:
        estimate = estimate_complexity("def f():\n    for item in get_items(): pass\n")
        self.assertIn("unknown", estimate.time_complexity)
        self.assertTrue(any("Opaque iterable" in item for item in estimate.uncertain_patterns))

    def test_halving_while_loop_is_logarithmic(self) -> None:
        estimate = estimate_complexity("def f(n):\n    while n > 1:\n        n //= 2\n")
        self.assertEqual(estimate.time_complexity, "O(log(n)) estimated")

    def test_extend_growth_multiplies_loop_and_source_sizes(self) -> None:
        estimate = estimate_complexity(
            "def f(n, values):\n    result = []\n    for i in range(n):\n        result.extend(values)\n"
        )
        self.assertEqual(estimate.space_complexity, "O(n*values) estimated")

    def test_dictionary_and_set_growth(self) -> None:
        mapping = estimate_complexity(
            "def f(n):\n    result = {}\n    for i in range(n):\n        result[i] = i\n"
        )
        unique = estimate_complexity(
            "def f(items):\n    result = set()\n    for item in items:\n        result.add(item)\n"
        )
        self.assertEqual(mapping.space_complexity, "O(n) estimated")
        self.assertEqual(unique.space_complexity, "O(items) estimated")

    def test_collection_literal_unrelated_to_loop_is_constant_auxiliary_space(self) -> None:
        estimate = estimate_complexity(
            "def f(items):\n    cache = {'key': 1}\n    for item in items: pass\n"
        )
        self.assertEqual(estimate.space_complexity, "O(1) estimated")

    def test_collection_created_inside_loop_is_not_multiplied_by_outer_count(self) -> None:
        estimate = estimate_complexity(
            "def f(items):\n    for item in items:\n        temporary = []\n        temporary.append(item)\n"
        )
        self.assertEqual(estimate.space_complexity, "O(1) estimated")

    def test_collection_growth_in_match_try_else_and_with_scopes_is_seen(self) -> None:
        sources = (
            "def f(items, flag):\n    result = []\n"
            "    match flag:\n        case True:\n"
            "            for item in items: result.append(item)\n",
            "def f(items):\n    result = []\n    try:\n        pass\n"
            "    except Exception:\n        pass\n    else:\n"
            "        for item in items: result.append(item)\n",
            "def f(items, context):\n    result = []\n"
            "    with context:\n        for item in items: result.append(item)\n",
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(estimate_complexity(source).space_complexity,
                                 "O(items) estimated")

    def test_unknown_bound_for_collection_growth_makes_space_unknown(self) -> None:
        estimate = estimate_complexity(
            "def f():\n    result = []\n    for item in get_items():\n        result.append(item)\n"
        )
        self.assertIn("unknown", estimate.space_complexity)

    def test_sorting_and_linear_builtin_traversal(self) -> None:
        sorted_result = estimate_complexity("def f(items):\n    return sorted(items)\n")
        summed = estimate_complexity("def f(items):\n    return sum(items)\n")
        self.assertIn("items*log(items)", sorted_result.time_complexity)
        self.assertEqual(sorted_result.space_complexity, "O(items) estimated")
        self.assertEqual(summed.time_complexity, "O(items) estimated")

    def test_heap_operations_use_logarithmic_and_linear_bounds(self) -> None:
        pop = estimate_complexity("def f(heap):\n    return heappop(heap)\n")
        build = estimate_complexity("def f(values):\n    heapify(values)\n")
        self.assertEqual(pop.time_complexity, "O(log(heap)) estimated")
        self.assertEqual(build.time_complexity, "O(values) estimated")

    def test_opaque_calls_are_reported_as_unexpanded(self) -> None:
        estimate = estimate_complexity("def f(x):\n    return dynamic_dispatch(x)\n")
        self.assertTrue(any("not inlined" in item for item in estimate.uncertain_patterns))

    def test_invalid_syntax_is_rejected_safely(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Python source"):
            estimate_complexity("def f(:")


if __name__ == "__main__":
    unittest.main()
