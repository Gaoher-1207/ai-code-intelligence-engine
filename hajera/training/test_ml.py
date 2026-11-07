"""Tests for the ML part. Run from the repo root:  pytest hajera/training -q"""
import pytest

from feature_schema import COMPLEXITY_INPUT_FEATURES, DEFECT_INPUT_FEATURES
from schema_features import (complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, validate_complexity_features,
                             validate_defect_features, CX_NUM, BOUNDS, STRUCTS, DEFECT_NUM)

LINEAR = "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n"
NESTED = "def f(xs):\n    c = 0\n    for i in xs:\n        for j in xs:\n            c += i * j\n    return c\n"
BINARY = ("def f(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
          "        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n")
RECURSIVE = "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n"
UNSAFE_C = "void copy(char *s) {\n    char buf[8];\n    strcpy(buf, s);\n}\n"
SAFE_C = ("int *make(int n) {\n    int *p = malloc(n * sizeof(int));\n    if (p == NULL) {\n"
          "        return NULL;\n    }\n    return p;\n}\n")


# ---- feature extraction follows the shared schema ----
@pytest.mark.parametrize("code", [LINEAR, NESTED, BINARY, RECURSIVE, "x = 1\n", ""])
def test_complexity_features_match_schema(code):
    feats = extract_complexity_features(code)
    validate_complexity_features(feats)  # Gaoher's validator
    assert list(feats) == COMPLEXITY_INPUT_FEATURES
    assert len(complexity_vector(feats)) == 17


@pytest.mark.parametrize("code", [UNSAFE_C, SAFE_C, "void example(void) {}", ""])
def test_defect_features_match_schema(code):
    feats = extract_defect_features(code)
    validate_defect_features(feats)
    assert list(feats) == DEFECT_INPUT_FEATURES
    assert 0 <= feats["duplicate_code_ratio"] <= 1
    assert len(defect_vector(feats)) == 13


def test_model_vector_feature_orders_remain_fixed():
    assert CX_NUM == [
        "function_length", "loop_count", "max_loop_nesting", "branch_count",
        "recursion_count", "function_call_count", "ast_node_count",
    ]
    assert BOUNDS == ["constant", "input_dependent", "logarithmic", "unknown"]
    assert STRUCTS == ["list", "dict", "set", "heap", "deque"]
    assert DEFECT_NUM == [
        "function_length", "branch_count", "loop_count", "pointer_usage",
        "memory_operation_count", "dangerous_api_count", "exception_handling_count",
        "null_check_count", "cyclomatic_complexity", "duplicate_code_ratio",
        "uninitialized_variable_count", "resource_cleanup_count", "ast_node_count",
    ]


# ---- feature values are sensible ----
def test_loop_nesting():
    assert extract_complexity_features(LINEAR)["max_loop_nesting"] == 1
    assert extract_complexity_features(NESTED)["max_loop_nesting"] == 2


def test_recursion_detected():
    assert extract_complexity_features(RECURSIVE)["recursion_count"] == 1
    assert extract_complexity_features(LINEAR)["recursion_count"] == 0


def test_loop_bound_types():
    assert extract_complexity_features(BINARY)["loop_bound_type"] == "logarithmic"
    assert extract_complexity_features(LINEAR)["loop_bound_type"] == "input_dependent"
    assert extract_complexity_features("x = 1\n")["loop_bound_type"] == "constant"


def test_c_features():
    unsafe = extract_defect_features(UNSAFE_C)
    assert unsafe["dangerous_api_count"] == 1
    safe = extract_defect_features(SAFE_C)
    assert safe["memory_operation_count"] == 1
    assert safe["null_check_count"] == 1
    assert safe["dangerous_api_count"] == 0


def test_invalid_python_raises():
    with pytest.raises(SyntaxError):
        extract_complexity_features("def f(:\n")


# ---- prediction functions (skipped if the model files are not present) ----
@pytest.fixture(scope="module")
def predict():
    try:
        import predict as p
    except FileNotFoundError:
        pytest.skip("trained model files not found in hajera/models")
    return p


def test_predict_complexity_output(predict):
    res = predict.predict_complexity(NESTED)
    assert set(res) == {"predicted_complexity", "complexity_class", "confidence"}
    assert 0.0 <= res["confidence"] <= 1.0


def test_predict_defect_output(predict):
    res = predict.predict_defect(UNSAFE_C)
    assert 0.0 <= res["risk_probability"] <= 1.0
    expected = "vulnerable" if res["risk_probability"] >= 0.5 else "non-vulnerable"
    assert res["predicted_label"] == expected


def test_predict_is_deterministic(predict):
    assert predict.predict_complexity(LINEAR) == predict.predict_complexity(LINEAR)


# ---- AST and lexical-regex regression cases ----
def test_multiple_python_functions_use_the_whole_input_as_the_analysis_unit():
    code = (
        "def first(xs):\n    for x in xs:\n        consume(x)\n"
        "def second(n):\n    return n + 1\n"
    )

    feats = extract_complexity_features(code)

    assert feats["function_length"] == len(code.splitlines())
    assert feats["loop_count"] == 1
    assert feats["function_call_count"] == 1


def test_async_direct_recursion_is_counted():
    code = "async def descend(n):\n    return await descend(n - 1) if n else 0\n"

    assert extract_complexity_features(code)["recursion_count"] == 1


def test_direct_recursion_is_counted_separately_in_multiple_functions():
    code = (
        "def first(n):\n    return first(n - 1) if n else 0\n"
        "def second(n):\n    return second(n - 1) if n else 0\n"
    )

    assert extract_complexity_features(code)["recursion_count"] == 2


def test_match_cases_contribute_to_branch_count():
    code = (
        "def choose(value):\n"
        "    match value:\n"
        "        case 0:\n            return 0\n"
        "        case _:\n            return 1\n"
    )

    assert extract_complexity_features(code)["branch_count"] == 2


def test_loop_bound_ignores_unrelated_division_and_growth():
    unrelated = "def f(n, x):\n    while n > 0:\n        y = x // 2\n        n -= 1\n"
    growth = "def f(n):\n    while n < 100:\n        n *= 2\n"
    divide_by_one = "def f(n):\n    while n > 0:\n        n //= 1\n"

    assert extract_complexity_features(unrelated)["loop_bound_type"] == "unknown"
    assert extract_complexity_features(growth)["loop_bound_type"] == "unknown"
    assert extract_complexity_features(divide_by_one)["loop_bound_type"] == "unknown"


def test_empty_python_function_and_empty_input_are_well_formed():
    empty_function = extract_complexity_features("def f():\n    pass\n")
    empty_source = extract_complexity_features("")

    assert empty_function["function_length"] == 2
    assert empty_function["loop_count"] == 0
    assert empty_function["recursion_count"] == 0
    assert empty_source["ast_node_count"] == 1  # The Python AST still contains its Module root.


def test_c_pointer_and_memory_operations_are_counted_lexically():
    code = (
        "int f(int *p) {\n"
        "    int *q = p;\n"
        "    *q = 1;\n"
        "    void *a = malloc(8);\n"
        "    a = realloc(a, 16);\n"
        "    free(a);\n"
        "    return *p;\n"
        "}\n"
    )
    feats = extract_defect_features(code)

    assert feats["pointer_usage"] > 0
    assert feats["memory_operation_count"] == 3


def test_c_api_names_inside_strings_and_comments_do_not_count():
    code = (
        'void f(void) { const char *s = "strcpy(p, q); http://host/malloc(8); if (x)"; '
        '// system("x")\n }\n'
    )
    feats = extract_defect_features(code)

    assert feats["dangerous_api_count"] == 0
    assert feats["memory_operation_count"] == 0
    assert feats["branch_count"] == 0
    assert feats["loop_count"] == 0


def test_c_switch_ternary_and_logical_operators_feed_complexity_estimate():
    code = "int f(int x) { switch (x) { case 1: return x ? 1 : 0; case 2: break; } return x && 1; }"
    feats = extract_defect_features(code)

    # Regex-derived decision-point estimate: switch + two cases + ternary + &&.
    assert feats["branch_count"] == 4
    assert feats["cyclomatic_complexity"] == 6


def test_c_null_and_error_flow_patterns_are_estimates():
    code = "int f(char *p) { if (p == NULL) goto done; if (!p) errno = 1; done: return 0; }"
    feats = extract_defect_features(code)

    assert feats["null_check_count"] == 2
    assert feats["exception_handling_count"] == 2


def test_c_duplicate_lines_uninitialized_declaration_and_cleanup_are_estimates():
    code = (
        "void f(void) {\n"
        "    int uninit;\n"
        "    int initialized = 1;\n"
        "    free(ptr);\n"
        "    fclose(file);\n"
        "    cleanup();\n"
        "    cleanup();\n"
        "}\n"
    )
    feats = extract_defect_features(code)

    assert feats["uninitialized_variable_count"] == 1
    assert feats["resource_cleanup_count"] == 2
    assert feats["duplicate_code_ratio"] > 0
    # This is only a lexical token count, not an AST node count.
    assert feats["ast_node_count"] > 0


def test_empty_c_source_is_well_formed_but_not_a_parsed_function():
    feats = extract_defect_features("")

    validate_defect_features(feats)
    assert feats["function_length"] == 0
    assert feats["cyclomatic_complexity"] == 1
    assert feats["ast_node_count"] == 0
