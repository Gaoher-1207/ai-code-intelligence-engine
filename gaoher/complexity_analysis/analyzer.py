"""Conservative AST-based time and auxiliary-space estimates for Python.

The analyzer recognizes common loop bounds and collection operations. Results
are symbolic estimates, not proofs of runtime complexity; calls, aliases,
mutation and dynamic dispatch can change actual behavior.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

from gaoher.code_analysis import analyze_source
from gaoher.feature_extraction.features import _halves

# A monomial is a sorted set of variable/logarithm factors with exponents.
# A polynomial is a set of monomials; this supports sequential addition and
# nested-loop multiplication without assuming that n and m are interchangeable.
_Monomial = tuple[tuple[str, int], ...]
_Polynomial = frozenset[_Monomial]
_ONE: _Polynomial = frozenset({()})
_UNKNOWN: None = None
_KNOWN_CALLS = {
    "all", "any", "append", "appendleft", "add", "Counter", "copy", "deque",
    "dict", "enumerate", "extend", "extendleft", "get", "heapify", "heappop",
    "heappush", "insert", "len", "list", "max", "min", "range", "reversed",
    "set", "sort", "sorted", "sum", "tuple", "update", "zip",
}


@dataclass(frozen=True)
class ComplexityEstimate:
    """Estimated complexity and the evidence and limitations behind it."""

    time_complexity: str
    space_complexity: str
    confidence: str
    loop_nesting: int
    recursion_detected: bool
    uncertain_patterns: tuple[str, ...]
    time_reasoning: tuple[str, ...] = ()
    space_reasoning: tuple[str, ...] = ()


@dataclass
class _AnalysisNotes:
    """Deduplicated explanations collected during one analysis."""

    time: list[str] = field(default_factory=list)
    space: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)

    def add(self, bucket: list[str], message: str) -> None:
        if message not in bucket:
            bucket.append(message)


def _add(left: _Polynomial | None, right: _Polynomial | None) -> _Polynomial | None:
    if left is None or right is None:
        return _UNKNOWN
    return _simplify(left | right)


def _multiply(left: _Polynomial | None, right: _Polynomial | None) -> _Polynomial | None:
    if left is None or right is None:
        return _UNKNOWN
    products: set[_Monomial] = set()
    for a in left:
        for b in right:
            powers = dict(a)
            for name, exponent in b:
                powers[name] = powers.get(name, 0) + exponent
            products.add(tuple(sorted(powers.items())))
    return _simplify(frozenset(products))


def _simplify(terms: Iterable[_Monomial]) -> _Polynomial:
    """Drop polynomial terms dominated by another term for large inputs."""
    unique = set(terms)

    def dominates(left: _Monomial, right: _Monomial) -> bool:
        left_powers = dict(left)
        right_powers = dict(right)
        names = left_powers.keys() | right_powers.keys()
        return (all(left_powers.get(name, 0) >= right_powers.get(name, 0) for name in names)
                and left != right)

    kept = {term for term in unique
            if not any(dominates(other, term) for other in unique)}
    return frozenset(kept or {()})


def _factor(name: str) -> _Polynomial:
    return frozenset({((name, 1),)})


def _format_polynomial(value: _Polynomial | None) -> str:
    if value is None:
        return "unknown"
    formatted: list[str] = []
    for term in sorted(value, key=lambda item: (len(item), item)):
        if not term:
            formatted.append("1")
            continue
        factors = []
        for name, exponent in term:
            factors.append(name if exponent == 1 else f"{name}^{exponent}")
        formatted.append("*".join(factors))
    return " + ".join(formatted)


def _combine_max(branches: Iterable[_Polynomial | None]) -> _Polynomial | None:
    """Represent the worst branch with a safe polynomial upper bound."""
    result: _Polynomial = frozenset()
    for branch in branches:
        if branch is None:
            return _UNKNOWN
        result |= branch
    return _simplify(result or _ONE)


def _size_expression(node: ast.AST) -> _Polynomial | None:
    """Estimate the size represented by a simple Python expression."""
    if isinstance(node, ast.Constant):
        return _ONE
    if isinstance(node, ast.Name):
        return _factor(node.id)
    if isinstance(node, ast.UnaryOp):
        return _size_expression(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = _size_expression(node.left), _size_expression(node.right)
        if isinstance(node.op, ast.Mult):
            return _multiply(left, right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _add(left, right)
        if isinstance(node.op, (ast.FloorDiv, ast.Div, ast.Mod)):
            if isinstance(node.right, ast.Constant) and node.right.value not in (0, None):
                return left
        return _UNKNOWN
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
            arg = node.args[0]
            return _factor(arg.id) if isinstance(arg, ast.Name) else _UNKNOWN
        if isinstance(node.func, ast.Name) and node.func.id in {"min", "max"} and node.args:
            result: _Polynomial = frozenset()
            for arg in node.args:
                part = _size_expression(arg)
                if part is None:
                    return _UNKNOWN
                result |= part
            return _simplify(result)
    return _UNKNOWN


def _literal_size(node: ast.AST) -> _Polynomial | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return _ONE
    if isinstance(node, ast.Dict):
        return _ONE
    return None


def _iterable_size(node: ast.AST) -> _Polynomial | None:
    """Estimate iteration count; dynamic/opaque producers remain unknown."""
    literal = _literal_size(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Name):
        return _factor(node.id)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            if not node.args:
                return _UNKNOWN
            if len(node.args) == 1:
                return _size_expression(node.args[0])
            parts = [_size_expression(arg) for arg in node.args[:2]]
            if any(part is None for part in parts):
                return _UNKNOWN
            result = _ONE
            for part in parts:
                result = _add(result, part)
            if len(node.args) == 3 and not isinstance(node.args[2], ast.Constant):
                return _UNKNOWN
            return result
        if node.func.id in {"enumerate", "reversed"} and node.args:
            return _iterable_size(node.args[0])
        if node.func.id == "zip" and node.args:
            result: _Polynomial = frozenset()
            for arg in node.args:
                size = _iterable_size(arg)
                if size is None:
                    return _UNKNOWN
                result |= size
            return _simplify(result)
    if isinstance(node, ast.GeneratorExp):
        return _comprehension_size(node)
    return _UNKNOWN


def _comprehension_size(node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> _Polynomial | None:
    size: _Polynomial = _ONE
    for generator in node.generators:
        size = _multiply(size, _iterable_size(generator.iter))
        if size is None:
            return _UNKNOWN
    return size


def _loop_bound(node: ast.For | ast.AsyncFor | ast.While,
                notes: _AnalysisNotes) -> _Polynomial | None:
    if isinstance(node, ast.While):
        if _halves(node):
            tested = sorted({child.id for child in ast.walk(node.test)
                             if isinstance(child, ast.Name)})
            if tested:
                name = f"log({tested[0]})"
                notes.add(notes.time, f"While-loop bound estimated logarithmic from repeated halving of {tested[0]}.")
                notes.add(notes.uncertain, "Logarithmic while-loop classification is a syntactic halving heuristic.")
                return _factor(name)
        countdown = _countdown_bound(node)
        if countdown is not None:
            notes.add(notes.time, f"While-loop bound estimated linear in {', '.join(sorted(countdown))} from unit-step updates.")
            notes.add(notes.uncertain, "While-loop termination was inferred from a comparison and update pattern.")
            result = _ONE
            for name in sorted(countdown):
                result = _multiply(result, _factor(name))
            return result
        notes.add(notes.time, "A while-loop has no recognized bound.")
        notes.add(notes.uncertain, "Dynamic while-loop termination or progress is unsupported.")
        return _UNKNOWN

    size = _iterable_size(node.iter)
    if size is None:
        notes.add(notes.time, "Loop iterable has an unknown or dynamic size.")
        notes.add(notes.uncertain, "Opaque iterable bounds such as get_items() are not guessed.")
    elif isinstance(node.iter, ast.Name):
        notes.add(notes.time, f"Loop over {node.iter.id} is estimated proportional to its length.")
    elif isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) \
            and node.iter.func.id == "range":
        notes.add(notes.time, "range() bound is estimated from its simple arithmetic arguments.")
    return size


def _countdown_bound(node: ast.While) -> set[str] | None:
    """Recognize basic unit-step loops such as ``while i < n: i += 1``."""
    test_names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
    if not test_names:
        return None
    directions: dict[str, int] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
            if isinstance(child.op, ast.Sub) and isinstance(child.value, ast.Constant) \
                    and isinstance(child.value.value, (int, float)) and child.value.value > 0:
                directions[child.target.id] = -1
            elif isinstance(child.op, ast.Add) and isinstance(child.value, ast.Constant) \
                    and isinstance(child.value.value, (int, float)) and child.value.value < 0:
                directions[child.target.id] = -1
            elif isinstance(child.op, ast.Add) and isinstance(child.value, ast.Constant) \
                    and child.value.value > 0:
                directions[child.target.id] = 1
    if not (directions.keys() & test_names):
        return None
    for compare in (child for child in ast.walk(node.test) if isinstance(child, ast.Compare)):
        left_names = {child.id for child in ast.walk(compare.left) if isinstance(child, ast.Name)}
        comparator_names = {child.id for comparator in compare.comparators
                             for child in ast.walk(comparator) if isinstance(child, ast.Name)}
        for target, direction in directions.items():
            if target not in left_names:
                continue
            if direction < 0 and isinstance(compare.ops[0], (ast.Gt, ast.GtE)):
                return (comparator_names - directions.keys()) or {target}
            if direction > 0 and isinstance(compare.ops[0], (ast.Lt, ast.LtE)):
                return (comparator_names - directions.keys()) or {target}
    if isinstance(node.test, ast.Name) and node.test.id in directions:
        return {node.test.id}
    if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) \
            and isinstance(node.test.operand, ast.Name) and node.test.operand.id in directions:
        return {node.test.operand.id}
    return None


def _logarithmic_term(value: _Polynomial | None) -> _Polynomial | None:
    if value is None:
        return None
    names = sorted({name for monomial in value for name, _ in monomial})
    if not names:
        return _ONE
    result: _Polynomial = frozenset()
    for name in names:
        result |= _factor(f"log({name})")
    return _simplify(result)


def _expr_cost(node: ast.AST, notes: _AnalysisNotes) -> _Polynomial | None:
    """Estimate local work for expressions and recognizable built-ins."""
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        child_cost: _Polynomial | None = _ONE
        for child in ast.iter_child_nodes(node):
            part = _expr_cost(child, notes)
            child_cost = _add(child_cost, part)
        size = _iterable_size(node.args[0]) if node.args else _UNKNOWN
        if name == "sorted" or (isinstance(node.func, ast.Attribute) and name == "sort"):
            if size is None:
                notes.add(notes.time, "Sorting input size is unknown.")
                notes.add(notes.uncertain, "Sorting an opaque iterable has unknown size.")
                operation = _UNKNOWN
            else:
                operation = _multiply(size, _logarithmic_term(size))
                notes.add(notes.time, "Recognized sorting as O(k log k) for input size k.")
            return _add(child_cost, operation)
        if name in {"sum", "min", "max", "any", "all"} and len(node.args) == 1:
            notes.add(notes.time, f"Recognized {name}() as a linear traversal of its iterable.")
            return _add(child_cost, size)
        if name in {"heappush", "heappop"}:
            heap_size = _iterable_size(node.args[0]) if node.args else _UNKNOWN
            if heap_size is None:
                notes.add(notes.time, f"{name}() heap size is unknown.")
                notes.add(notes.uncertain, "Heap operation cost depends on an opaque heap size.")
                operation = _UNKNOWN
            else:
                operation = _logarithmic_term(heap_size)
                notes.add(notes.time, f"Recognized {name}() as logarithmic in heap size.")
            return _add(child_cost, operation)
        if name == "heapify":
            heap_size = _iterable_size(node.args[0]) if node.args else _UNKNOWN
            notes.add(notes.time, "Recognized heapify() as linear in the input collection size.")
            return _add(child_cost, heap_size)
        if name in {"len", "range", "enumerate", "zip", "reversed", "append", "add", "get"}:
            return child_cost
        if name in {"extend", "update"} and node.args:
            return _add(child_cost, size)
        if name == "insert":
            receiver_size = _iterable_size(node.func.value) if isinstance(node.func, ast.Attribute) else _UNKNOWN
            return _add(child_cost, receiver_size)
        if name in {"list", "tuple", "set", "dict", "deque", "Counter", "defaultdict"}:
            if not node.args:
                operation = _ONE
            elif name == "defaultdict":
                operation = _iterable_size(node.args[1]) if len(node.args) > 1 else _ONE
            else:
                operation = size
            return _add(child_cost, operation)
        if name == "copy" and isinstance(node.func, ast.Attribute):
            return _add(child_cost, _iterable_size(node.func.value))
        if name:
            notes.add(notes.uncertain, f"Call to {name}() is not inlined; only local call overhead is counted.")
        else:
            notes.add(notes.uncertain, "Dynamic call target is not inlined.")
        return child_cost
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
        size = _comprehension_size(node)
        body: _Polynomial | None = _ONE
        for child in ast.iter_child_nodes(node):
            part = _expr_cost(child, notes)
            body = _add(body, part)
        notes.add(notes.time, "Comprehension work is estimated as the product of generator sizes.")
        return _multiply(size, body)
    if isinstance(node, ast.GeneratorExp):
        # Constructing a generator is constant work; consumption is charged by
        # the enclosing recognized consumer, or remains an uncertainty.
        return _ONE
    result: _Polynomial | None = _ONE
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            result = _add(result, _expr_cost(child, notes))
    return result


def _sequence_cost(statements: Iterable[ast.stmt], notes: _AnalysisNotes) -> _Polynomial | None:
    result: _Polynomial | None = _ONE
    for statement in statements:
        result = _add(result, _statement_cost(statement, notes))
    return result


def _statement_cost(statement: ast.stmt, notes: _AnalysisNotes) -> _Polynomial | None:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                              ast.Pass, ast.Break, ast.Continue, ast.Global, ast.Nonlocal,
                              ast.Import, ast.ImportFrom)):
        return _ONE
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        bound = _loop_bound(statement, notes)
        body = _sequence_cost(statement.body, notes)
        per_iteration = _add(_ONE, body)
        repeated = _multiply(bound, per_iteration)
        return _add(repeated, _sequence_cost(statement.orelse, notes))
    if isinstance(statement, ast.If):
        test = _expr_cost(statement.test, notes)
        branches = _combine_max((_sequence_cost(statement.body, notes),
                                 _sequence_cost(statement.orelse, notes)))
        return _add(test, branches)
    if isinstance(statement, ast.Match):
        case_costs = [_sequence_cost(case.body, notes) for case in statement.cases]
        return _add(_ONE, _combine_max(case_costs))
    if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        branches = [_sequence_cost(statement.body, notes)]
        branches.extend(_sequence_cost(handler.body, notes) for handler in statement.handlers)
        branch_cost = _combine_max(branches)
        return _add(branch_cost, _sequence_cost(statement.finalbody, notes))
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        context_cost: _Polynomial | None = _ONE
        for item in statement.items:
            context_cost = _add(context_cost, _expr_cost(item.context_expr, notes))
        return _add(context_cost, _sequence_cost(statement.body, notes))
    if isinstance(statement, (ast.Return, ast.Raise, ast.Assert, ast.Expr, ast.Assign,
                              ast.AnnAssign, ast.AugAssign, ast.Delete)):
        result: _Polynomial | None = _ONE
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.expr):
                result = _add(result, _expr_cost(child, notes))
        return result
    notes.add(notes.uncertain, f"Statement {type(statement).__name__} uses a conservative constant local-cost assumption.")
    return _ONE


def _scope_nodes(tree: ast.Module) -> list[tuple[str, list[ast.stmt], tuple[str, ...]]]:
    """Return module and function bodies, with nested scopes considered separately."""
    scopes: list[tuple[str, list[ast.stmt], tuple[str, ...]]] = []
    module_body = [node for node in tree.body
                   if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    scopes.append(("<module>", module_body, ()))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = tuple(arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                               *node.args.kwonlyargs))
            if node.args.vararg:
                params += (node.args.vararg.arg,)
            if node.args.kwarg:
                params += (node.args.kwarg.arg,)
            scopes.append((node.name, node.body, params))
        elif isinstance(node, ast.ClassDef):
            body = [child for child in node.body
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            if body:
                scopes.append((f"{node.name} class body", body, ()))
    return scopes


def _find_recursive_calls(body: list[ast.stmt], function_name: str) -> list[ast.Call]:
    found: list[ast.Call] = []

    class Finder(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if ((isinstance(func, ast.Name) and func.id == function_name)
                    or (isinstance(func, ast.Attribute) and func.attr == function_name
                        and isinstance(func.value, ast.Name)
                        and func.value.id in {"self", "cls"})):
                found.append(node)
            self.generic_visit(node)

    visitor = Finder()
    for statement in body:
        visitor.visit(statement)
    return found


def _has_base_case(body: list[ast.stmt], parameter: str,
                   recursive_calls: list[ast.Call]) -> bool:
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.If):
                names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
                if parameter not in names:
                    continue
                body_calls = {id(child) for branch in node.body for child in ast.walk(branch)}
                else_calls = {id(child) for branch in node.orelse for child in ast.walk(branch)}
                recursive_in_body = any(id(call) in body_calls for call in recursive_calls)
                recursive_in_else = any(id(call) in else_calls for call in recursive_calls)
                if recursive_in_body != recursive_in_else:
                    return True
                body_returns = any(isinstance(child, ast.Return)
                                   for branch in node.body for child in ast.walk(branch))
                else_returns = any(isinstance(child, ast.Return)
                                   for branch in node.orelse for child in ast.walk(branch))
                body_terminal = body_returns and not recursive_in_body
                else_terminal = else_returns and not recursive_in_else
                test = node.test
                if isinstance(test, ast.Name) and test.id == parameter:
                    if recursive_in_body and not node.orelse:
                        return True
                if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                        and isinstance(test.operand, ast.Name) and test.operand.id == parameter
                        and body_terminal):
                    return True
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) \
                        and test.left.id == parameter and len(test.ops) == 1 \
                        and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant):
                    op = test.ops[0]
                    if body_terminal and isinstance(op, (ast.Lt, ast.LtE)):
                        return True
                    if else_terminal and isinstance(op, (ast.Gt, ast.GtE)):
                        return True
    return False


def _recursive_call_is_repeated(body: list[ast.stmt], call: ast.Call) -> bool:
    """Return true when a recursive call site is nested in an iteration."""
    parents: dict[ast.AST, ast.AST] = {}
    for statement in body:
        for parent in ast.walk(statement):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
    current = call
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.For, ast.AsyncFor, ast.While,
                                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
    return False


def _recurrence_kind(call: ast.Call, parameters: tuple[str, ...]) -> tuple[str, str] | None:
    args = list(call.args)
    if not args:
        return None
    parameter = parameters[1] if parameters and parameters[0] in {"self", "cls"} \
        and len(parameters) > 1 else (parameters[0] if parameters else "")
    if not parameter:
        return None
    arg = args[0]
    if isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Name) and arg.left.id == parameter:
        if isinstance(arg.op, ast.Sub) and isinstance(arg.right, ast.Constant) \
                and isinstance(arg.right.value, (int, float)) and arg.right.value > 0:
            return "linear", parameter
        if isinstance(arg.op, (ast.FloorDiv, ast.Div, ast.RShift)) \
                and isinstance(arg.right, ast.Constant) and arg.right.value >= 1:
            return "logarithmic", parameter
    return None


def _recursion_time(body: list[ast.stmt], name: str, params: tuple[str, ...],
                    local: _Polynomial | None, notes: _AnalysisNotes) -> _Polynomial | None:
    recursive = _find_recursive_calls(body, name)
    if not recursive:
        return local
    if len(recursive) != 1:
        notes.add(notes.time, f"{name} has multiple recursive calls; recurrence is not solved.")
        notes.add(notes.uncertain, "Multiple recursive calls are unsupported for time-growth solving.")
        return _UNKNOWN
    if _recursive_call_is_repeated(body, recursive[0]):
        notes.add(notes.time, f"{name} calls itself from a repeated loop or comprehension.")
        notes.add(notes.uncertain, "Recursive calls nested in iterations have an unsupported recurrence.")
        return _UNKNOWN
    kind = _recurrence_kind(recursive[0], params)
    parameter = kind[1] if kind else ""
    if not kind or not _has_base_case(body, parameter, recursive):
        notes.add(notes.time, f"{name} recursion has no recognized decreasing argument and base case.")
        notes.add(notes.uncertain, "Recursive termination and recurrence are not established.")
        return _UNKNOWN
    mode, parameter = kind
    if local is None:
        return _UNKNOWN
    if mode == "linear":
        notes.add(notes.time, f"Single linear-decreasing recursion in {name} adds a factor of {parameter}.")
        notes.add(notes.uncertain, "Recursive complexity assumes one call, a recognized base case, and constant decrement.")
        return _multiply(local, _factor(parameter))
    # For T(n)=T(n/c)+f(n), retain polynomial terms containing n; multiply
    # terms independent of n by log(n). This is a conservative common recurrence.
    notes.add(notes.time, f"Single halving recursion in {name} is estimated using a logarithmic depth.")
    notes.add(notes.uncertain, "Recursive complexity assumes one call and a recognized halving recurrence.")
    result: _Polynomial = frozenset()
    for term in local:
        if any(name == parameter for name, _ in term):
            result |= frozenset({term})
        else:
            result |= _multiply(frozenset({term}), _factor(f"log({parameter})")) or frozenset()
    return _simplify(result)


def _recursion_space(local: _Polynomial | None, kind: tuple[str, str] | None) -> _Polynomial | None:
    """Estimate accumulated frame allocation plus stack depth for a simple recurrence."""
    if local is None or kind is None:
        return _UNKNOWN
    mode, parameter = kind
    result: _Polynomial = frozenset()
    for term in local:
        has_parameter = any(name == parameter and exponent > 0 for name, exponent in term)
        if mode == "linear":
            result |= _multiply(frozenset({term}), _factor(parameter)) or frozenset()
        elif has_parameter:
            # Geometrically decreasing per-frame allocations sum to the first
            # (largest) frame's size for ordinary polynomial size terms.
            result |= frozenset({term})
        else:
            result |= _multiply(frozenset({term}), _factor(f"log({parameter})")) or frozenset()
    return _simplify(result or {()})


def _iterable_element_size(node: ast.AST) -> _Polynomial | None:
    """Space bound for a value materialized from an iterable expression."""
    return _iterable_size(node)


def _collection_literal_size(node: ast.AST) -> _Polynomial | None:
    if isinstance(node, ast.GeneratorExp):
        return _ONE
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
        return _comprehension_size(node)
    literal = _literal_size(node)
    if literal is not None:
        return literal
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "defaultdict":
            return _ONE
        if node.func.id in {"list", "tuple", "set", "dict", "deque", "Counter"}:
            return _iterable_size(node.args[0]) if node.args else _ONE
    return None


def _assignment_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_assignment_targets(item) for item in node.elts))
    return set()


class _SpaceVisitor:
    """Track local collection lifetimes and auxiliary growth per scope."""

    _GROW_METHODS = {"append", "extend", "add", "update", "insert", "appendleft", "extendleft"}

    def __init__(self, parameters: tuple[str, ...], notes: _AnalysisNotes) -> None:
        self.parameters = set(parameters)
        self.notes = notes
        self.created: dict[str, int] = {}
        self.terms: list[_Polynomial | None] = []

    def analyze(self, statements: list[ast.stmt]) -> _Polynomial | None:
        self._scan(statements, [])
        result: _Polynomial = frozenset()
        for term in self.terms:
            if term is None:
                return _UNKNOWN
            result |= term
        return _simplify(result or _ONE)

    def _scan(self, statements: list[ast.stmt], loops: list[_Polynomial | None]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                bound = _loop_bound(statement, self.notes)
                self._scan(statement.body, [*loops, bound])
                self._scan(statement.orelse, loops.copy())
                continue
            if isinstance(statement, ast.If):
                self._scan(statement.body, loops.copy())
                self._scan(statement.orelse, loops.copy())
                continue
            if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                self._scan(statement.body, loops.copy())
                for handler in statement.handlers:
                    self._scan(handler.body, loops.copy())
                self._scan(statement.orelse, loops.copy())
                self._scan(statement.finalbody, loops.copy())
                continue
            if isinstance(statement, ast.Match):
                for case in statement.cases:
                    self._scan(case.body, loops.copy())
                continue
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                self._scan(statement.body, loops.copy())
                continue
            if isinstance(statement, ast.Assign):
                allocation = _collection_literal_size(statement.value)
                if allocation is not None:
                    for name in _assignment_targets(statement.targets[0]):
                        self.created[name] = len(loops)
                    self.terms.append(allocation)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                allocation = _collection_literal_size(statement.value)
                if allocation is not None:
                    for name in _assignment_targets(statement.target):
                        self.created[name] = len(loops)
                    self.terms.append(allocation)
            self._record_mutation(statement, loops)
            self._record_temporary_collections(statement)

    def _record_temporary_collections(self, statement: ast.stmt) -> None:
        for node in ast.walk(statement):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
                self.terms.append(_comprehension_size(node))
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name in {"sorted", "list", "tuple", "set", "dict", "deque", "Counter", "copy", "sort"}:
                    if name in {"sorted", "copy", "sort"} or node.args:
                        if name == "sort" and isinstance(node.func, ast.Attribute):
                            size = _iterable_size(node.func.value)
                        else:
                            size = _iterable_size(node.args[0]) if node.args else _UNKNOWN
                        self.terms.append(size)
                        if size is None:
                            self.notes.add(self.notes.space, f"Temporary result from {name}() has unknown size.")
                            self.notes.add(self.notes.uncertain, f"The size of {name}() input is dynamic or unsupported.")
                if name in {"sorted", "sort"}:
                    self.notes.add(self.notes.space, "Sorting may require temporary O(k) storage.")

    def _record_mutation(self, statement: ast.stmt, loops: list[_Polynomial | None]) -> None:
        if isinstance(statement, ast.Assign):
            for target_node in statement.targets:
                if isinstance(target_node, ast.Subscript) and isinstance(target_node.value, ast.Name):
                    self._record_growth(target_node.value.id, "key assignment", loops, _ONE)
        elif (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Subscript)
              and isinstance(statement.target.value, ast.Name)):
            self._record_growth(statement.target.value.id, "key assignment", loops, _ONE)
        calls = [node for node in ast.walk(statement) if isinstance(node, ast.Call)]
        for call in calls:
            func = call.func
            method = func.attr if isinstance(func, ast.Attribute) else ""
            target = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else None
            if (isinstance(func, ast.Name) and func.id == "heappush" and call.args
                    and isinstance(call.args[0], ast.Name)):
                target = call.args[0].id
                method = "heappush"
            if method not in self._GROW_METHODS or not target:
                continue
            if target not in self.created:
                if target in self.parameters:
                    self.notes.add(self.notes.space, f"Mutation of input parameter {target} is treated as input storage, not auxiliary allocation.")
                continue
            extra: _Polynomial | None = _ONE
            if method in {"extend", "extendleft", "update"} and call.args:
                extra = _iterable_size(call.args[0])
            elif method == "insert":
                extra = _factor(target)
            self._record_growth(target, method, loops, extra)

    def _record_growth(self, target: str, operation: str,
                       loops: list[_Polynomial | None], extra: _Polynomial | None) -> None:
        if target not in self.created:
            if target in self.parameters:
                self.notes.add(self.notes.space, f"Mutation of input parameter {target} is treated as input storage, not auxiliary allocation.")
            return
        growth: _Polynomial | None = _ONE
        for bound in loops[self.created[target]:]:
            growth = _multiply(growth, bound)
        growth = _multiply(growth, extra)
        self.terms.append(growth)
        self.notes.add(self.notes.space, f"Growth of local collection {target} via {operation} is counted across its active loops.")
        if growth is None:
            self.notes.add(self.notes.uncertain, f"Growth bound for {target} via {operation} is unknown.")


def _scope_space(body: list[ast.stmt], params: tuple[str, ...], notes: _AnalysisNotes) -> _Polynomial | None:
    return _SpaceVisitor(params, notes).analyze(body)


def _estimate_scope_time(name: str, body: list[ast.stmt], params: tuple[str, ...],
                         notes: _AnalysisNotes) -> _Polynomial | None:
    local = _sequence_cost(body, notes)
    return _recursion_time(body, name, params, local, notes)


def _format_estimate(value: _Polynomial | None) -> str:
    if value is None:
        return "unknown"
    return f"O({_format_polynomial(value)}) estimated"


def estimate_complexity(source: str) -> ComplexityEstimate:
    """Estimate symbolic time and auxiliary-space growth for Python source.

    Simple loops over names and ``range`` expressions preserve input dimensions
    (for example, nested loops over ``n`` and ``m`` become O(n*m)). Sequential
    work is added and simplified. Common sorting/traversal built-ins, basic
    halving/countdown loops, simple one-call recursion, and local collection
    growth are recognized. Unknown bounds stay unknown; user-defined callees
    are not inlined. Space estimates distinguish local allocations from input
    parameters and report the assumptions used.
    """
    result = analyze_source(source)
    if not result.valid:
        raise ValueError(f"Invalid Python source: {result.error}")
    tree = ast.parse(source)
    scopes = _scope_nodes(tree)
    notes = _AnalysisNotes()
    notes.add(notes.time, "Time results are symbolic static estimates, not proofs of runtime cost.")
    notes.add(notes.space, "Space estimates cover auxiliary allocations; input storage is excluded when identified.")
    time_total: _Polynomial | None = _ONE
    space_total: _Polynomial | None = _ONE
    recursive_space_total: _Polynomial | None = _ONE
    loop_nesting = result.max_loop_nesting
    for name, body, params in scopes:
        scope_time = _estimate_scope_time(name, body, params, notes)
        scope_space = _scope_space(body, params, notes)
        time_total = _add(time_total, scope_time)
        space_total = _add(space_total, scope_space)
        recursive_calls = _find_recursive_calls(body, name)
        if recursive_calls:
            kind = _recurrence_kind(recursive_calls[0], params) if len(recursive_calls) == 1 else None
            parameter = kind[1] if kind else ""
            if len(recursive_calls) != 1 or not kind or not _has_base_case(body, parameter, recursive_calls):
                recursive_space_total = _UNKNOWN
                notes.add(notes.space, f"Recursive stack depth in {name} is not recognized.")
                notes.add(notes.uncertain, "Recursive stack depth can depend on runtime values and call structure.")
            else:
                recursive_space_total = _add(recursive_space_total,
                                             _recursion_space(scope_space, kind))
        if scope_time is not None:
            notes.add(notes.time, f"Analyzed local work in scope {name} independently; callees are not expanded.")
        if scope_space is not None:
            notes.add(notes.space, f"Estimated peak auxiliary allocations in scope {name}; lifetimes are approximated.")

    if not result.functions:
        notes.add(notes.time, "Top-level executable statements were analyzed as one scope.")
    if result.recursive_call_count:
        notes.add(notes.space, "Recursive stack space is estimated from recognized single-call recurrences.")
        space_total = _add(space_total, recursive_space_total)

    unexpanded_calls = sorted(set(result.function_calls) - _KNOWN_CALLS)
    if unexpanded_calls:
        notes.add(notes.uncertain,
                  "Calls to user-defined or dynamic functions are not analyzed interprocedurally: "
                  + ", ".join(unexpanded_calls) + ".")
    if time_total is None:
        notes.add(notes.time, "At least one dynamic or unsupported loop/recurrence prevents a finite symbolic estimate.")
    if space_total is None:
        notes.add(notes.space, "At least one dynamic allocation or recursive stack bound is unknown.")
    confidence = "low" if notes.uncertain or time_total is None or space_total is None else "medium"
    return ComplexityEstimate(
        time_complexity=_format_estimate(time_total),
        space_complexity=_format_estimate(space_total),
        confidence=confidence,
        loop_nesting=loop_nesting,
        recursion_detected=bool(result.recursive_call_count),
        uncertain_patterns=tuple(notes.uncertain),
        time_reasoning=tuple(notes.time),
        space_reasoning=tuple(notes.space),
    )


