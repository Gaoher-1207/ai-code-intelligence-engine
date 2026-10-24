"""Tests for the ML part. Run from the repo root:  pytest hajera/training -q"""
import pytest

from schema_features import (complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, validate_complexity_features,
                             validate_defect_features)

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
    assert len(complexity_vector(feats)) == 17


@pytest.mark.parametrize("code", [UNSAFE_C, SAFE_C, "void example(void) {}", ""])
def test_defect_features_match_schema(code):
    feats = extract_defect_features(code)
    validate_defect_features(feats)
    assert 0 <= feats["duplicate_code_ratio"] <= 1
    assert len(defect_vector(feats)) == 13


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