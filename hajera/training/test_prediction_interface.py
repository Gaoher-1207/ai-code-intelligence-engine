import pytest

from hajera.training.predict import (
    PredictionInputError,
    get_model_info,
    load_model_bundle,
    predict_complexity,
    predict_defect,
)


PYTHON_EXAMPLE = "def total(values):\n    return sum(values)\n"
C_EXAMPLE = "int copy_value(int *out, const int *in) { *out = *in; return 0; }"


def test_preferred_experiment_bundles_load_independently_and_match_features():
    complexity = load_model_bundle("complexity")
    defect = load_model_bundle("defect")

    assert complexity["model_family"] == "extratrees_depth_12_leaf_5"
    assert defect["model_family"] == "rf_depth_8_leaf_20"
    assert len(complexity["feature_names"]) == complexity["model"].n_features_in_ == 17
    assert len(defect["feature_names"]) == defect["model"].n_features_in_ == 13
    assert get_model_info("complexity")["artifact"] == "complexity_python_rf_improved.joblib"
    assert get_model_info("defect")["artifact"] == "defect_c_rf_improved.joblib"


def test_python_prediction_has_documented_class_and_confidence_fields():
    output = predict_complexity(PYTHON_EXAMPLE)

    assert set(output) == {"predicted_complexity", "complexity_class", "confidence"}
    assert output["complexity_class"] in {
        "constant", "logn", "linear", "nlogn", "quadratic", "cubic", "np"
    }
    assert isinstance(output["predicted_complexity"], str)
    assert 0.0 <= output["confidence"] <= 1.0


def test_c_prediction_has_documented_label_and_probability_fields():
    output = predict_defect(C_EXAMPLE)

    assert set(output) == {"risk_probability", "predicted_label"}
    assert output["predicted_label"] in {"vulnerable", "non-vulnerable"}
    assert 0.0 <= output["risk_probability"] <= 1.0
    assert output["predicted_label"] == (
        "vulnerable" if output["risk_probability"] >= 0.5 else "non-vulnerable"
    )


@pytest.mark.parametrize("bad_input", [None, 42, b"def f(): pass", "", " \t\n"])
def test_both_prediction_functions_reject_empty_or_non_string_input(bad_input):
    with pytest.raises(PredictionInputError):
        predict_complexity(bad_input)
    with pytest.raises(PredictionInputError):
        predict_defect(bad_input)


def test_python_prediction_rejects_syntax_errors():
    with pytest.raises(PredictionInputError, match="syntactically valid Python"):
        predict_complexity("def unfinished(:\n")


def test_both_prediction_functions_reject_nul_bytes():
    with pytest.raises(PredictionInputError, match="NUL"):
        predict_complexity("x = 1\x00")
    with pytest.raises(PredictionInputError, match="NUL"):
        predict_defect("int f() {\x00 return 0; }")


def test_unknown_task_has_a_clear_loading_error():
    with pytest.raises(ValueError, match="complexity.*defect"):
        load_model_bundle("unknown")
