"""Integration checks through Hajera's normal project package imports."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import numpy as np

from feature_schema import validate_complexity_features, validate_defect_features
from gaoher.code_analysis import analyze_source
from gaoher.feature_extraction import (
    complexity_vector,
    complexity_vector_feature_names,
    extract_complexity_features,
    extract_defect_features,
    preprocess_complexity_features,
)
from gaoher.ml_integration import predict_c_defect, predict_python_complexity


PYTHON_EXAMPLE = """\
def total(values):
    result = 0
    for value in values:
        result += value
    return result
"""
C_EXAMPLE = """\
int copy_name(char *out, const char *in) {
    if (in == NULL) return 0;
    strcpy(out, in);
    return 1;
}
"""


def _import_hajera(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "hajera" or (exc.name and exc.name.startswith("hajera.")):
            raise unittest.SkipTest(
                f"{module_name} is absent from the normal project checkout"
            ) from exc
        raise


def _has_materialized_model(predictor, task: str) -> bool:
    for filename in predictor.MODEL_FILES[task]:
        model_path = Path(predictor.MODELS) / filename
        if not model_path.is_file():
            continue
        with model_path.open("rb") as model_file:
            first_line = model_file.readline()
        if not first_line.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return True
    return False


class RealFeatureCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hajera_features = _import_hajera("hajera.training.schema_features")

    def test_gaoher_complexity_features_match_hajera_and_shared_schema(self) -> None:
        parsed = analyze_source(PYTHON_EXAMPLE)
        self.assertTrue(parsed.valid)
        gaoher_features = extract_complexity_features(PYTHON_EXAMPLE)
        validate_complexity_features(gaoher_features)
        hajera_features = self.hajera_features.extract_complexity_features(PYTHON_EXAMPLE)
        self.assertEqual(gaoher_features, hajera_features)

    def test_17_value_vector_and_actual_numpy_preprocessing_match(self) -> None:
        features = extract_complexity_features(PYTHON_EXAMPLE)
        expected_names = tuple(self.hajera_features.COMPLEXITY_VECTOR_FEATURE_NAMES)
        names = complexity_vector_feature_names()
        vector = complexity_vector(features)
        hajera_vector = self.hajera_features.complexity_vector(features)

        self.assertEqual(len(expected_names), 17)
        self.assertEqual(names, expected_names)
        self.assertEqual(vector, hajera_vector)
        self.assertEqual(len(vector), 17)

        gaoher_matrix = preprocess_complexity_features(features)
        hajera_matrix = self.hajera_features.prep([hajera_vector])
        self.assertEqual(gaoher_matrix.shape, (1, 17))
        np.testing.assert_array_equal(gaoher_matrix, hajera_matrix)
        expected_matrix = np.log1p(np.asarray([hajera_vector], dtype=float))
        np.testing.assert_array_equal(hajera_matrix, expected_matrix)

    def test_real_defect_adapter_matches_hajera_and_shared_schema(self) -> None:
        actual = extract_defect_features(C_EXAMPLE)
        expected = self.hajera_features.extract_defect_features(C_EXAMPLE)
        validate_defect_features(actual)
        self.assertEqual(actual, expected)


class RealPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hajera_predictor = _import_hajera("hajera.training.predict")
        missing = [
            task for task in ("complexity", "defect")
            if not _has_materialized_model(cls.hajera_predictor, task)
        ]
        if missing:
            raise unittest.SkipTest(
                "No materialized model bundle is present for: " + ", ".join(missing)
            )

    def test_gaoher_entrypoint_runs_hajera_public_model_prediction(self) -> None:
        result = predict_python_complexity(PYTHON_EXAMPLE)
        prediction = result["prediction"]
        self.assertTrue(result["analysis"]["valid"])
        self.assertEqual(result["analysis"]["function_names"], ["total"])
        self.assertEqual(len(result["feature_vector"]), 17)
        self.assertEqual(len(result["feature_names"]), 17)
        self.assertEqual(np.asarray(result["preprocessed_features"]).shape, (1, 17))
        self.assertEqual(
            set(prediction),
            {"predicted_complexity", "complexity_class", "confidence"},
        )
        self.assertIn(prediction["complexity_class"], self.hajera_predictor.BIGO)
        self.assertTrue(0 <= prediction["confidence"] <= 1)

    def test_gaoher_entrypoint_runs_hajera_public_c_defect_prediction(self) -> None:
        result = predict_c_defect(C_EXAMPLE)
        prediction = result["prediction"]
        self.assertEqual(set(prediction), {"risk_probability", "predicted_label"})
        self.assertIn("dangerous_api_count", result["features"])
        self.assertTrue(0 <= prediction["risk_probability"] <= 1)
        self.assertIn(prediction["predicted_label"], {"vulnerable", "non-vulnerable"})


class InvalidSourceTests(unittest.TestCase):
    def test_invalid_python_is_rejected_before_hajera_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Python source"):
            predict_python_complexity("def broken(:")

    def test_empty_and_non_string_inputs_are_rejected(self) -> None:
        for source in ("", " \t\n", None, 12):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    predict_python_complexity(source)
                with self.assertRaises(ValueError):
                    predict_c_defect(source)


if __name__ == "__main__":
    unittest.main()
