"""Parse Python source and collect syntax-level structural information.

This module reports source structure; its measurements are static hints and do
not establish runtime behavior for arbitrary programs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionInfo:
    """Structural facts collected for one function definition."""

    name: str
    line: int
    end_line: int
    loop_count: int
    max_loop_nesting: int
    condition_count: int
    function_calls: tuple[str, ...]
    recursive_calls: int


@dataclass(frozen=True)
class AnalysisResult:
    """Result of parsing source, including safe diagnostics for invalid input."""

    valid: bool
    error: str | None
    functions: tuple[FunctionInfo, ...]
    loop_count: int
    max_loop_nesting: int
    condition_count: int
    function_calls: tuple[str, ...]
    recursive_call_count: int
    ast_node_count: int


def _call_name(call: ast.Call) -> str:
    """Return a readable name for a call expression."""
    node: ast.expr = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append("<call-result>")
    else:
        parts.append("<expression>")
    return ".".join(reversed(parts))


class _StructureVisitor(ast.NodeVisitor):
    """Collect structural metrics, optionally limiting traversal to one scope."""

    def __init__(self, *, skip_nested_scopes: bool = False) -> None:
        self.loop_count = 0
        self.max_loop_nesting = 0
        self.condition_count = 0
        self.function_calls: list[str] = []
        self._loop_depth = 0
        self._skip_nested_scopes = skip_nested_scopes

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._skip_nested_scopes:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not self._skip_nested_scopes:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not self._skip_nested_scopes:
            self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if not self._skip_nested_scopes:
            self.generic_visit(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        self.loop_count += 1
        self._loop_depth += 1
        self.max_loop_nesting = max(self.max_loop_nesting, self._loop_depth)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        # Each generator is a nested iteration layer in a comprehension.
        layers = len(node.generators)
        self.loop_count += layers
        self._loop_depth += layers
        self.max_loop_nesting = max(self.max_loop_nesting, self._loop_depth)
        self.condition_count += sum(len(generator.ifs) for generator in node.generators)
        self.generic_visit(node)
        self._loop_depth -= layers

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_If(self, node: ast.If) -> None:
        self.condition_count += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.condition_count += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.condition_count += len(node.cases)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.function_calls.append(_call_name(node))
        self.generic_visit(node)


def _function_calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Collect calls in a function, excluding nested function/class bodies."""
    found: list[str] = []

    class Calls(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            found.append(_call_name(node))
            self.generic_visit(node)

    Calls().visit(function)
    return tuple(found)


def _recursion_count(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count direct and conventional self/cls recursive calls."""
    count = 0

    class Recursion(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal count
            fn = node.func
            direct = isinstance(fn, ast.Name) and fn.id == function.name
            method = (isinstance(fn, ast.Attribute) and fn.attr == function.name
                      and isinstance(fn.value, ast.Name) and fn.value.id in {"self", "cls"})
            count += int(direct or method)
            self.generic_visit(node)

    Recursion().visit(function)
    return count


def analyze_source(source: str) -> AnalysisResult:
    """Parse Python source and report functions, loops, conditions and calls.

    Invalid syntax is returned as ``valid=False`` with an error message and
    empty structural results; syntax errors are not raised to callers. Per-function
    metrics exclude nested function, class, and lambda bodies. Recursion detection
    covers direct calls and conventional ``self``/``cls`` calls; mutual and aliased
    recursion are not inferred.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError) as exc:
        location = f"line {exc.lineno}: " if isinstance(exc, SyntaxError) and exc.lineno else ""
        return AnalysisResult(False, f"{location}{exc.msg if isinstance(exc, SyntaxError) else exc}",
                              (), 0, 0, 0, (), 0, 0)

    visitor = _StructureVisitor()
    visitor.visit(tree)
    functions: list[FunctionInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Visit this function's body only. Nested functions and classes are
            # analyzed separately as ast.walk reaches their own definitions.
            local = _StructureVisitor(skip_nested_scopes=True)
            for statement in node.body:
                local.visit(statement)
            functions.append(FunctionInfo(
                name=node.name,
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                loop_count=local.loop_count,
                max_loop_nesting=local.max_loop_nesting,
                condition_count=local.condition_count,
                function_calls=_function_calls(node),
                recursive_calls=_recursion_count(node),
            ))
    return AnalysisResult(True, None, tuple(functions), visitor.loop_count,
                          visitor.max_loop_nesting, visitor.condition_count,
                          tuple(visitor.function_calls),
                          sum(item.recursive_calls for item in functions),
                          sum(1 for _ in ast.walk(tree)))
