"""Feature adapters aligned with the shared schema and Hajera's ML pipeline.

Complexity features use Python's AST and match Hajera's training extractor.
Defect extraction, vector ordering, and numeric preprocessing delegate to
Hajera's existing implementation so those contracts are not reimplemented here.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping

from gaoher.code_analysis import analyze_source


def _halves(loop: ast.While) -> bool:
    """Mirror Hajera's conservative heuristic for shrinking while-loop bounds.

    This recognizes direct division/shifts of a tested variable and common
    midpoint-update patterns. It is a static hint, not a runtime proof.
    """
    test_names = {node.id for node in ast.walk(loop.test) if isinstance(node, ast.Name)}
    if not test_names:
        return False

    body_nodes: list[ast.AST] = []
    stack: list[ast.AST] = list(loop.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        body_nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))

    def assigned_names(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            return set()
        return {target.id for target in targets if isinstance(target, ast.Name)}

    divisions: list[ast.BinOp] = []
    for node in body_nodes:
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            divisor = node.value.value if isinstance(node.value, ast.Constant) else None
            minimum = 1 if isinstance(node.op, ast.RShift) else 2
            if (node.target.id in test_names and isinstance(divisor, (int, float))
                    and divisor >= minimum
                    and isinstance(node.op, (ast.FloorDiv, ast.Div, ast.RShift))):
                return True
        if (isinstance(node, ast.BinOp)
                and isinstance(node.op, (ast.FloorDiv, ast.Div, ast.RShift))
                and isinstance(node.right, ast.Constant)):
            minimum = 1 if isinstance(node.op, ast.RShift) else 2
            names = {child.id for child in ast.walk(node.left) if isinstance(child, ast.Name)}
            if (isinstance(node.right.value, (int, float))
                    and node.right.value >= minimum and names & test_names):
                divisions.append(node)

    midpoint_names: set[str] = set()
    for division in divisions:
        for parent in body_nodes:
            if (isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is not None
                    and any(child is division for child in ast.walk(parent.value))):
                targets = assigned_names(parent)
                if targets & test_names:
                    return True
                midpoint_names.update(targets)

    if midpoint_names:
        for node in body_nodes:
            if assigned_names(node) & test_names:
                value = getattr(node, "value", None)
                if value is not None:
                    names = {child.id for child in ast.walk(value) if isinstance(child, ast.Name)}
                    if names & midpoint_names:
                        return True
                if isinstance(node, ast.AugAssign):
                    names = {child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)}
                    if names & midpoint_names:
                        return True
    return False


def _loop_bound_type(nodes: list[ast.AST]) -> str:
    """Classify loop bounds using the same precedence and heuristic as Hajera."""
    kinds: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.For):
            iterator = node.iter
            if isinstance(iterator, ast.Call) and getattr(iterator.func, "id", "") == "range":
                constant = iterator.args and all(isinstance(arg, ast.Constant) for arg in iterator.args)
                kinds.add("constant" if constant else "input_dependent")
            else:
                kinds.add("input_dependent")
        elif isinstance(node, ast.While):
            kinds.add("logarithmic" if _halves(node) else "unknown")
    if not kinds:
        return "constant"
    for kind in ("input_dependent", "logarithmic", "constant", "unknown"):
        if kind in kinds:
            return kind
    return "unknown"


def _max_loop_nesting(node: ast.AST, depth: int = 0) -> int:
    """Return the same recursive For/While nesting metric used by Hajera."""
    best = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, (ast.For, ast.While)) else depth
        best = max(best, _max_loop_nesting(child, child_depth))
    return best


def _call_name(call: ast.Call) -> str | None:
    """Return the final identifier for a call, matching Hajera's detector."""
    return getattr(call.func, "id", None) or getattr(call.func, "attr", None)


def _data_structure_types(nodes: list[ast.AST], call_names: set[str]) -> list[str]:
    structures: list[str] = []
    if any(isinstance(node, (ast.List, ast.ListComp)) for node in nodes):
        structures.append("list")
    if (any(isinstance(node, (ast.Dict, ast.DictComp)) for node in nodes)
            or call_names & {"defaultdict", "Counter"}):
        structures.append("dict")
    if any(isinstance(node, (ast.Set, ast.SetComp)) for node in nodes):
        structures.append("set")
    if call_names & {"heappush", "heappop", "heapify"}:
        structures.append("heap")
    if "deque" in call_names:
        structures.append("deque")
    return structures


def extract_complexity_features(source: str, language: str = "python") -> dict[str, Any]:
    """Return the shared complexity fields with Hajera-compatible semantics.

    Loop and branch counts intentionally follow the training extractor: regular
    ``for``/``while`` nodes count as loops, and ``if``, conditional expressions,
    and match cases count as branches. Comprehension generators and filters do
    not affect those two fields. The shared validator is always applied before
    returning the feature mapping.
    """
    analysis = analyze_source(source)
    if not analysis.valid:
        raise ValueError(f"Invalid Python source: {analysis.error}")
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    calls = [node for node in nodes if isinstance(node, ast.Call)]
    call_names = {_call_name(call) for call in calls}
    functions = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    recursion_count = sum(function.recursive_calls for function in analysis.functions)

    input_size_variable = (
        bool(call_names & {"input", "len", "readline", "read", "readlines"})
        or any(isinstance(call.func, ast.Name) and call.func.id == "range"
               and any(not isinstance(arg, ast.Constant) for arg in call.args) for call in calls)
        or any(function.args.posonlyargs or function.args.args or function.args.kwonlyargs
               or function.args.vararg or function.args.kwarg for function in functions)
        or any(isinstance(node, ast.Attribute) and node.attr == "stdin" for node in nodes)
    )
    features: dict[str, Any] = {
        "language": language,
        "function_length": len(source.splitlines()),
        "loop_count": sum(isinstance(node, (ast.For, ast.While)) for node in nodes),
        "max_loop_nesting": _max_loop_nesting(tree),
        "branch_count": sum(isinstance(node, (ast.If, ast.IfExp)) for node in nodes)
                        + sum(len(node.cases) for node in nodes if isinstance(node, ast.Match)),
        "recursion_count": recursion_count,
        "function_call_count": len(calls),
        "ast_node_count": len(nodes),
        "input_size_variable": bool(input_size_variable),
        "loop_bound_type": _loop_bound_type(nodes),
        "data_structure_types": _data_structure_types(nodes, call_names),
    }
    try:
        from feature_schema import validate_complexity_features
    except ImportError as exc:
        raise ImportError("Could not import the shared feature_schema validator") from exc
    validate_complexity_features(features)
    return features


def _hajera_schema_features() -> Any:
    """Import Hajera's feature implementation lazily with a clear failure."""
    try:
        from hajera.training import schema_features
    except ImportError as exc:
        raise ImportError(
            "Hajera's required module 'hajera.training.schema_features' is "
            "unavailable. Include Hajera's training package in the project "
            "checkout, make the repository root the import root, and install "
            "its declared runtime dependencies."
        ) from exc
    return schema_features


def complexity_vector_feature_names() -> tuple[str, ...]:
    """Return Hajera's authoritative ordered names for the 17-value vector."""
    schema_features = _hajera_schema_features()
    return tuple(schema_features.COMPLEXITY_VECTOR_FEATURE_NAMES)


def complexity_vector(features: Mapping[str, Any]) -> list[int | float]:
    """Convert schema features using Hajera's authoritative feature ordering."""
    from feature_schema import validate_complexity_features

    validate_complexity_features(dict(features))
    schema_features = _hajera_schema_features()
    return schema_features.complexity_vector(features)


def preprocess_complexity_features(features: Mapping[str, Any]) -> Any:
    """Return Hajera's preprocessed one-row model matrix for schema features.

    This delegates both vector construction and ``log1p`` preprocessing to
    Hajera, preserving the saved model's feature order and transform exactly.
    The return value is a NumPy array with shape ``(1, 17)``.
    """
    schema_features = _hajera_schema_features()
    vector = complexity_vector(features)
    return schema_features.prep([vector])


def extract_defect_features(source: str, language: str = "c") -> dict[str, Any]:
    """Delegate defect-schema extraction to Hajera's existing implementation."""
    schema_features = _hajera_schema_features()
    features = schema_features.extract_defect_features(source, language=language)
    from feature_schema import validate_defect_features

    validate_defect_features(features)
    return features
