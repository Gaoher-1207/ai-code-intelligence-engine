"""Adapters that expose analysis results to the shared feature contract."""

from .features import (
    complexity_vector,
    complexity_vector_feature_names,
    extract_complexity_features,
    extract_defect_features,
    preprocess_complexity_features,
)

__all__ = [
    "complexity_vector",
    "complexity_vector_feature_names",
    "extract_complexity_features",
    "extract_defect_features",
    "preprocess_complexity_features",
]
