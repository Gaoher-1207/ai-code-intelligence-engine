"""Public inference API for Hajera's complexity and defect-risk models.

From the repository root, import ``predict_complexity`` and ``predict_defect``
from ``hajera.training.predict``. Bundles are loaded lazily and independently.
"""
from __future__ import annotations

import ast
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

try:  # Package import used by other project components.
    from .schema_features import (
        COMPLEXITY_VECTOR_FEATURE_NAMES,
        DEFECT_INPUT_FEATURES,
        DEFECT_VECTOR_FEATURE_NAMES,
        complexity_vector,
        defect_vector,
        extract_complexity_features,
        extract_defect_features,
        prep,
        validate_complexity_features,
        validate_defect_features,
    )
except ImportError:  # Direct script/import compatibility for existing workflows.
    from schema_features import (
        COMPLEXITY_VECTOR_FEATURE_NAMES,
        DEFECT_INPUT_FEATURES,
        DEFECT_VECTOR_FEATURE_NAMES,
        complexity_vector,
        defect_vector,
        extract_complexity_features,
        extract_defect_features,
        prep,
        validate_complexity_features,
        validate_defect_features,
    )


MODELS = Path(__file__).resolve().parents[1] / "models"
MODEL_FILES = {
    "complexity": (
        "complexity_python_rf_improved.joblib",
        "complexity_python_rf_baseline.joblib",
        "complexity_python_schema_lr.joblib",
    ),
    "defect": (
        "defect_c_rf_improved.joblib",
        "defect_c_rf_baseline.joblib",
        "defect_c_schema_lr.joblib",
    ),
}
_EXPECTED_FEATURES = {
    "complexity": COMPLEXITY_VECTOR_FEATURE_NAMES,
    "defect": DEFECT_VECTOR_FEATURE_NAMES,
}
_MODEL_CACHE: dict[str, tuple[dict[str, Any], Path]] = {}
_MODEL_LOCK = threading.RLock()

BIGO = {
    "constant": "O(1)", "logn": "O(log n)", "linear": "O(n)",
    "nlogn": "O(n log n)", "quadratic": "O(n^2)", "cubic": "O(n^3)",
    "np": "exponential / NP-hard",
}


class PredictionInputError(ValueError):
    """Raised when caller input is empty, malformed, or of the wrong type."""


class ModelBundleError(RuntimeError):
    """Raised when no compatible model bundle can be loaded for a task."""


def _validate_bundle(task: str, bundle: Any, path: Path) -> dict[str, Any]:
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelBundleError(f"Model bundle {path.name} must be a mapping with a 'model' entry")
    model = bundle["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelBundleError(f"Model bundle {path.name} does not support probability prediction")
    classes = list(getattr(model, "classes_", ()))
    if task == "complexity":
        expected_classes = set(BIGO)
        if {str(label) for label in classes} != expected_classes:
            raise ModelBundleError(f"{path.name} does not contain the expected seven complexity classes")
    elif (len(classes) != 2 or not any(label is True or label == 1 for label in classes)
          or not any(label is False or label == 0 for label in classes)):
        raise ModelBundleError(f"{path.name} must contain secure (0) and insecure (1) classes")

    expected = _EXPECTED_FEATURES[task]
    if bundle.get("feature_mode") == "schema_numeric":
        model_dimensions = getattr(model, "n_features_in_", None)
        if model_dimensions != len(expected):
            raise ModelBundleError(
                f"{path.name} expects {model_dimensions} numeric features; {task} extractor "
                f"produces {len(expected)}"
            )
        saved_names = bundle.get("feature_names")
        # Earlier Devign baseline metadata stored the raw shared input fields,
        # although the estimator correctly consumes the 13 ordered numeric fields.
        allowed_names = [expected]
        if task == "defect":
            allowed_names.append(DEFECT_INPUT_FEATURES)
        if saved_names is not None and list(saved_names) not in allowed_names:
            raise ModelBundleError(f"Feature order in {path.name} does not match the extractor contract")
        scaler = bundle.get("scaler")
        if scaler is not None and getattr(scaler, "n_features_in_", len(expected)) != len(expected):
            raise ModelBundleError(f"Preprocessor dimensions in {path.name} do not match its feature contract")
    return bundle


def load_model_bundle(task: str) -> dict[str, Any]:
    """Load and validate the preferred saved bundle for ``complexity`` or ``defect``.

    The validation-selected experiment is preferred. Existing baseline and legacy
    bundles remain supported as fallbacks and are never modified.
    """
    if task not in MODEL_FILES:
        raise ValueError("task must be 'complexity' or 'defect'")
    with _MODEL_LOCK:
        if task in _MODEL_CACHE:
            return _MODEL_CACHE[task][0]
        paths = [MODELS / name for name in MODEL_FILES[task]]
        selected = next((path for path in paths if path.is_file()), None)
        if selected is None:
            searched = ", ".join(path.name for path in paths)
            raise ModelBundleError(f"No {task} model bundle found in {MODELS}; searched: {searched}")
        try:
            bundle = _validate_bundle(task, joblib.load(selected), selected)
        except ModelBundleError:
            raise
        except Exception as exc:
            raise ModelBundleError(f"Could not load {task} model bundle {selected.name}: {exc}") from exc
        _MODEL_CACHE[task] = (bundle, selected)
        return bundle


def load_model_bundles() -> dict[str, dict[str, Any]]:
    """Load both task bundles independently and return them keyed by task."""
    return {task: load_model_bundle(task) for task in MODEL_FILES}


def get_model_info(task: str | None = None) -> dict[str, Any]:
    """Return stable integration metadata without exposing estimator internals."""
    tasks = (task,) if task is not None else tuple(MODEL_FILES)
    result = {}
    for name in tasks:
        bundle = load_model_bundle(name)
        path = _MODEL_CACHE[name][1]
        result[name] = {
            "artifact": path.name,
            "model_family": bundle.get("model_family", bundle.get("version", "legacy")),
            "feature_dimensions": len(_EXPECTED_FEATURES[name]),
            "classes": [str(label) for label in bundle["model"].classes_],
        }
    return result if task is None else result[task]


def _validate_text_input(value: Any, argument_name: str) -> str:
    if not isinstance(value, str):
        raise PredictionInputError(f"{argument_name} must be a string containing source code")
    if not value.strip():
        raise PredictionInputError(f"{argument_name} must not be empty or whitespace-only")
    if "\x00" in value:
        raise PredictionInputError(f"{argument_name} must not contain NUL characters")
    return value


def _model_matrix(task: str, source: str) -> tuple[dict[str, Any], np.ndarray | Any]:
    bundle = load_model_bundle(task)
    if task == "complexity":
        features = extract_complexity_features(source)
        validate_complexity_features(features)
        vector = complexity_vector(features)
    else:
        features = extract_defect_features(source)
        validate_defect_features(features)
        vector = defect_vector(features)

    matrix = prep([vector])
    if matrix.shape[1] != len(_EXPECTED_FEATURES[task]):
        raise ModelBundleError(f"{task} feature extraction returned an unexpected vector dimension")
    if bundle.get("feature_mode") == "schema_numeric":
        scaler = bundle.get("scaler")
        if scaler is not None:
            matrix = scaler.transform(matrix)
    else:
        numeric = bundle["scaler"].transform(matrix)
        matrix = hstack([
            bundle["vectorizer"].transform([source]), csr_matrix(numeric)
        ]).tocsr()
    return bundle, matrix


def _probabilities(bundle: dict[str, Any], matrix: Any) -> np.ndarray:
    probabilities = np.asarray(bundle["model"].predict_proba(matrix)[0], dtype=float)
    if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)):
        raise ModelBundleError("Estimator returned invalid class probabilities")
    if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
        raise ModelBundleError("Estimator class probabilities do not sum to one")
    return probabilities


def predict_complexity(python_code: str) -> dict[str, Any]:
    """Estimate the CodeComplex class for valid Python source.

    Returns ``predicted_complexity`` (display form), ``complexity_class`` (raw
    CodeComplex label), and the winning class probability as ``confidence``.
    """
    source = _validate_text_input(python_code, "python_code")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise PredictionInputError(
            f"python_code must be syntactically valid Python (line {exc.lineno}): {exc.msg}"
        ) from exc
    bundle, matrix = _model_matrix("complexity", source)
    probabilities = _probabilities(bundle, matrix)
    classes = list(bundle["model"].classes_)
    if len(classes) != len(probabilities):
        raise ModelBundleError("Complexity estimator returned a class/probability length mismatch")
    index = int(probabilities.argmax())
    label = str(classes[index])
    return {
        "predicted_complexity": BIGO.get(label, label),
        "complexity_class": label,
        "confidence": round(float(probabilities[index]), 6),
    }


def predict_defect(c_code: str) -> dict[str, Any]:
    """Estimate insecure-code risk for non-empty C source or a C snippet.

    ``risk_probability`` is the probability of Devign target 1 (Insecure).
    The output label uses a fixed 0.5 threshold; this threshold is not calibrated.
    """
    source = _validate_text_input(c_code, "c_code")
    bundle, matrix = _model_matrix("defect", source)
    probabilities = _probabilities(bundle, matrix)
    classes = list(bundle["model"].classes_)
    positive_index = next((i for i, label in enumerate(classes)
                           if label is True or label == 1), None)
    if positive_index is None or len(classes) != 2 or len(probabilities) != 2:
        raise ModelBundleError("Defect estimator must expose binary classes 0 (Secure) and 1 (Insecure)")
    risk = round(float(probabilities[positive_index]), 6)
    return {
        "risk_probability": risk,
        "predicted_label": "vulnerable" if risk >= 0.5 else "non-vulnerable",
    }
