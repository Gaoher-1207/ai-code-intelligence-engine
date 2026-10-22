"""Shared feature contract for AI Code Intelligence Engine."""
from typing import Any, Dict, List

COMPLEXITY_INPUT_FEATURES = [
    "language", "function_length", "loop_count", "max_loop_nesting",
    "branch_count", "recursion_count", "function_call_count", "ast_node_count",
    "input_size_variable", "loop_bound_type", "data_structure_types",
]
DEFECT_INPUT_FEATURES = [
    "language", "source_code", "function_length", "branch_count", "loop_count",
    "pointer_usage", "memory_operation_count", "dangerous_api_count",
    "exception_handling_count", "null_check_count", "cyclomatic_complexity",
    "duplicate_code_ratio", "uninitialized_variable_count",
    "resource_cleanup_count", "ast_node_count",
]
COMPLEXITY_TARGET = "complexity_class"  # label only; never an input feature
DEFECT_TARGET = "target"                # dataset label only; verify label mapping
DEFECT_PREDICTION_OUTPUT = "risk_probability"  # model output only


def _nonnegative_ints(features: Dict[str, Any], fields: List[str]) -> None:
    for field in fields:
        value = features.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")


def validate_complexity_features(features: Dict[str, Any]) -> None:
    missing = set(COMPLEXITY_INPUT_FEATURES) - features.keys()
    if missing:
        raise ValueError(f"Missing complexity features: {', '.join(sorted(missing))}")
    _nonnegative_ints(features, [
        "function_length", "loop_count", "max_loop_nesting", "branch_count",
        "recursion_count", "function_call_count", "ast_node_count",
    ])
    if not isinstance(features["language"], str) or not features["language"]:
        raise ValueError("language must be a non-empty string")
    if not isinstance(features["input_size_variable"], bool):
        raise ValueError("input_size_variable must be boolean")
    if not isinstance(features["data_structure_types"], list):
        raise ValueError("data_structure_types must be a list")


def validate_defect_features(features: Dict[str, Any]) -> None:
    missing = set(DEFECT_INPUT_FEATURES) - features.keys()
    if missing:
        raise ValueError(f"Missing defect features: {', '.join(sorted(missing))}")
    _nonnegative_ints(features, [
        "function_length", "branch_count", "loop_count", "pointer_usage",
        "memory_operation_count", "dangerous_api_count", "exception_handling_count",
        "null_check_count", "cyclomatic_complexity", "uninitialized_variable_count",
        "resource_cleanup_count", "ast_node_count",
    ])
    ratio = features["duplicate_code_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError("duplicate_code_ratio must be between 0 and 1")
    if not isinstance(features["language"], str) or not features["language"]:
        raise ValueError("language must be a non-empty string")
    if not isinstance(features["source_code"], str):
        raise ValueError("source_code must be a string")
