import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_schema import validate_complexity_features, validate_defect_features


def test_complexity_schema_accepts_valid_record():
    validate_complexity_features({
        "language": "python", "function_length": 8, "loop_count": 2,
        "max_loop_nesting": 2, "branch_count": 1, "recursion_count": 0,
        "function_call_count": 1, "ast_node_count": 34,
        "input_size_variable": True, "loop_bound_type": "input_dependent",
        "data_structure_types": ["list"]
    })


def test_complexity_schema_rejects_negative_counts():
    record = {
        "language": "python", "function_length": -1, "loop_count": 2,
        "max_loop_nesting": 2, "branch_count": 1, "recursion_count": 0,
        "function_call_count": 1, "ast_node_count": 34,
        "input_size_variable": True, "loop_bound_type": "input_dependent",
        "data_structure_types": ["list"]
    }
    try:
        validate_complexity_features(record)
    except ValueError:
        return
    raise AssertionError("Expected negative count to be rejected")


def test_defect_schema_rejects_invalid_duplicate_ratio():
    record = {
        "language": "c", "source_code": "void f(void) {}",
        "function_length": 1, "branch_count": 0, "loop_count": 0,
        "pointer_usage": 0, "memory_operation_count": 0, "dangerous_api_count": 0,
        "exception_handling_count": 0, "null_check_count": 0,
        "cyclomatic_complexity": 1, "duplicate_code_ratio": 1.5,
        "uninitialized_variable_count": 0, "resource_cleanup_count": 0,
        "ast_node_count": 2
    }
    try:
        validate_defect_features(record)
    except ValueError:
        return
    raise AssertionError("Expected duplicate ratio above 1 to be rejected")
