"""Bridge Gaoher's Python analysis to Hajera's public complexity predictor.

The prediction API remains owned by Hajera. This module checks Gaoher's parse,
feature, vector, and preprocessing adapters, then delegates prediction from the
original source to Hajera's public ``predict_complexity`` interface. Hajera's
predictor performs its own extraction and preprocessing internally; this bridge
does not reimplement or replace that model path.
"""

from __future__ import annotations

from typing import Any

from gaoher.code_analysis import analyze_source
from gaoher.feature_extraction import (
    complexity_vector,
    complexity_vector_feature_names,
    extract_complexity_features,
    extract_defect_features,
    preprocess_complexity_features,
)


def predict_python_complexity(source: str) -> dict[str, Any]:
    """Analyze Python source with Gaoher and predict via Hajera's real API.

    The returned mapping includes Gaoher's analysis artifacts and the prediction
    mapping returned unchanged by Hajera. The delegated prediction API accepts
    source text and repeats its authoritative extraction/preprocessing path
    internally; the adapter values here make the Gaoher-to-Hajera contract
    visible and verifiable without duplicating Hajera's model implementation.

    Raises:
        ValueError: if ``source`` is not syntactically valid Python.
        ImportError: if Hajera's prediction module or its dependencies are not
            available in the current checkout/environment.
        Hajera prediction errors: input, bundle, or estimator failures from the
            public prediction API are allowed to propagate unchanged.
    """
    analysis = analyze_source(source)
    if not analysis.valid:
        raise ValueError(f"Invalid Python source: {analysis.error}")

    features = extract_complexity_features(source, language="python")
    names = complexity_vector_feature_names()
    vector = complexity_vector(features)
    if len(names) != 17 or len(vector) != len(names):
        raise ValueError(
            "Hajera complexity vector contract mismatch: "
            f"received {len(vector)} values for {len(names)} feature names"
        )

    matrix = preprocess_complexity_features(features)
    if getattr(matrix, "shape", None) != (1, 17):
        raise ValueError(
            "Hajera preprocessing must return a one-row, 17-column matrix; "
            f"received shape {getattr(matrix, 'shape', None)!r}"
        )

    try:
        from hajera.training.predict import predict_complexity
    except ImportError as exc:
        raise ImportError(
            "Hajera's public prediction API is unavailable. Make "
            "hajera.training.predict importable from the repository root and "
            "install its declared prediction dependencies."
        ) from exc

    prediction = predict_complexity(source)
    return {
        "analysis": {
            "valid": analysis.valid,
            "function_names": [function.name for function in analysis.functions],
            "loop_count": analysis.loop_count,
            "max_loop_nesting": analysis.max_loop_nesting,
            "condition_count": analysis.condition_count,
        },
        "features": features,
        "feature_names": list(names),
        "feature_vector": vector,
        "preprocessed_features": matrix.tolist(),
        "prediction": prediction,
    }


def predict_c_defect(source: str) -> dict[str, Any]:
    """Extract C defect features with Hajera and delegate real prediction.

    Hajera owns the C feature extractor and defect predictor. This function
    returns the schema-validated features together with the public prediction
    result; it does not duplicate preprocessing or model logic.
    """
    try:
        from hajera.training.predict import predict_defect
    except ImportError as exc:
        raise ImportError(
            "Hajera's public prediction API is unavailable. Make "
            "hajera.training.predict importable from the repository root and "
            "install its declared prediction dependencies."
        ) from exc

    prediction = predict_defect(source)
    features = extract_defect_features(source, language="c")
    return {"features": features, "prediction": prediction}


__all__ = ["predict_c_defect", "predict_python_complexity"]
