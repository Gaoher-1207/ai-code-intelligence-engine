"""Schema feature extractors (stand-in until Gaoher's extractor is ready).

They return dicts that follow feature_schema.py exactly, so his extractor can
replace these two functions later without changing the training code.
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from feature_schema import validate_complexity_features, validate_defect_features  # noqa: E402,F401


def _dup_ratio(code):
    lines = [l.strip() for l in code.splitlines() if len(l.strip()) > 3]
    return round(1 - len(set(lines)) / len(lines), 4) if lines else 0.0


# ---------------- Python (complexity model) ----------------
def _nest(node, depth=0):
    best = depth
    for c in ast.iter_child_nodes(node):
        d = depth + 1 if isinstance(c, (ast.For, ast.While)) else depth
        best = max(best, _nest(c, d))
    return best


def _halves(loop):
    """Heuristically recognize a shrinking loop bound or binary-search midpoint.

    This remains a static hint, not a proof of a logarithmic runtime. Unrelated
    divisions, multiplication, division by one, and nested-loop bodies are ignored.
    """
    test_names = {n.id for n in ast.walk(loop.test) if isinstance(n, ast.Name)}
    if not test_names:
        return False

    body_nodes = []
    stack = list(loop.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Lambda)):
            continue
        body_nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))

    def assigned_names(node):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            return set()
        return {target.id for target in targets if isinstance(target, ast.Name)}

    divisions = []
    for node in body_nodes:
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            divisor = node.value.value if isinstance(node.value, ast.Constant) else None
            minimum = 1 if isinstance(node.op, ast.RShift) else 2
            if (node.target.id in test_names and isinstance(divisor, (int, float))
                    and divisor >= minimum and isinstance(node.op, (ast.FloorDiv, ast.Div, ast.RShift))):
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Div, ast.RShift)) \
                and isinstance(node.right, ast.Constant):
            minimum = 1 if isinstance(node.op, ast.RShift) else 2
            names = {child.id for child in ast.walk(node.left) if isinstance(child, ast.Name)}
            if isinstance(node.right.value, (int, float)) and node.right.value >= minimum and names & test_names:
                divisions.append(node)

    midpoint_names = set()
    for division in divisions:
        for parent in body_nodes:
            if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is not None \
                    and any(child is division for child in ast.walk(parent.value)):
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


def _loop_bound(nodes):
    kinds = set()
    for n in nodes:
        if isinstance(n, ast.For):
            it = n.iter
            if isinstance(it, ast.Call) and getattr(it.func, "id", "") == "range":
                const = it.args and all(isinstance(a, ast.Constant) for a in it.args)
                kinds.add("constant" if const else "input_dependent")
            else:
                kinds.add("input_dependent")
        elif isinstance(n, ast.While):
            kinds.add("logarithmic" if _halves(n) else "unknown")
    if not kinds:
        return "constant"  # no loops: zero iterations is the real value
    for k in ("input_dependent", "logarithmic", "constant", "unknown"):
        if k in kinds:
            return k


def _call_name(c):
    return getattr(c.func, "id", None) or getattr(c.func, "attr", None)


def _function_body_nodes(function):
    """Walk one function body without attributing nested definitions to it."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def extract_complexity_features(code, language="python"):
    """Return schema features for the entire supplied Python source unit.

    Loop, branch, recursion, call, and AST-node counts use Python's AST. Input-size,
    loop-bound, and data-structure classifications are heuristics over that AST.
    """
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    names = {_call_name(c) for c in calls}
    functions = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    rec = 0
    for function in functions:
        for node in _function_body_nodes(function):
            if not isinstance(node, ast.Call):
                continue
            direct_call = isinstance(node.func, ast.Name) and node.func.id == function.name
            self_method_call = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == function.name
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
            )
            rec += int(direct_call or self_method_call)
    size_var = (
        bool(names & {"input", "len", "readline", "read", "readlines"})
        or any(isinstance(c.func, ast.Name) and c.func.id == "range"
               and any(not isinstance(a, ast.Constant) for a in c.args) for c in calls)
        or any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (n.args.posonlyargs or n.args.args or n.args.kwonlyargs or n.args.vararg or n.args.kwarg)
            for n in nodes
        )
        or any(isinstance(n, ast.Attribute) and n.attr == "stdin" for n in nodes)
    )
    structs = []
    if any(isinstance(n, (ast.List, ast.ListComp)) for n in nodes):
        structs.append("list")
    if any(isinstance(n, (ast.Dict, ast.DictComp)) for n in nodes) or names & {"defaultdict", "Counter"}:
        structs.append("dict")
    if any(isinstance(n, (ast.Set, ast.SetComp)) for n in nodes):
        structs.append("set")
    if names & {"heappush", "heappop", "heapify"}:
        structs.append("heap")
    if "deque" in names:
        structs.append("deque")
    return {
        "language": language,
        "function_length": len(code.splitlines()),
        "loop_count": sum(isinstance(n, (ast.For, ast.While)) for n in nodes),
        "max_loop_nesting": _nest(tree),
        "branch_count": sum(isinstance(n, (ast.If, ast.IfExp)) for n in nodes)
                        + sum(len(n.cases) for n in nodes if isinstance(n, ast.Match)),
        "recursion_count": rec,
        "function_call_count": len(calls),
        "ast_node_count": len(nodes),
        "input_size_variable": bool(size_var),
        "loop_bound_type": _loop_bound(nodes),
        "data_structure_types": structs,
    }


BOUNDS = ["constant", "input_dependent", "logarithmic", "unknown"]
STRUCTS = ["list", "dict", "set", "heap", "deque"]
CX_NUM = ["function_length", "loop_count", "max_loop_nesting", "branch_count",
          "recursion_count", "function_call_count", "ast_node_count"]


def complexity_vector(f):
    row = [f[k] for k in CX_NUM] + [int(f["input_size_variable"])]
    row += [int(f["loop_bound_type"] == b) for b in BOUNDS]
    row += [int(s in f["data_structure_types"]) for s in STRUCTS]
    return row


# ---------------- C (defect model) ----------------
DANGEROUS_C_APIS = ["strcpy", "strcat", "sprintf", "vsprintf", "gets", "scanf", "sscanf",
                    "strncpy", "strncat", "memcpy", "memmove", "system", "popen"]
_DANGER = re.compile(r"\b(?:%s)\s*\(" % "|".join(DANGEROUS_C_APIS))
_MEM = re.compile(r"\b(?:k?z?malloc|calloc|realloc|free|kfree|alloca|av_malloc|av_free|av_freep|g_malloc|g_free)\s*\(")
_CLEAN = re.compile(r"\b(?:fclose|close|closedir|pclose|free|kfree|av_free|av_freep|g_free|\w*unlock|\w*release|\w*destroy)\s*\(")
_PTR = re.compile(r"->|(?:\b(?:const|struct\s+\w+|unsigned|signed|char|int|long|short|void|float|double|size_t|\w+_t)\s*|[=(,;]\s*|^\s*)\*+\s*[A-Za-z_(]|(?<!&)&\s*[A-Za-z_]", re.M)
_NULL = re.compile(r"[!=]=\s*NULL|NULL\s*[!=]=|if\s*\(\s*!\s*[\w>.\[\]\-]+\s*\)")
_ERR = re.compile(r"\b(?:setjmp|longjmp|errno|goto)\b")
_UNINIT = re.compile(r"^\s*(?:const\s+|static\s+|unsigned\s+|signed\s+|struct\s+\w+\s+)*"
                     r"(?:int|char|long|short|float|double|size_t|ssize_t|bool|u?int\d+_t|\w+_t)\s+\**\s*\w+(?:\[[^\]]*\])?\s*;", re.M)
def _mask_c_noncode(code, mask_literals=True):
    """Blank C comments and optionally literals while preserving line positions.

    This is a small lexical mask for the regex estimates below, not a C parser.
    """
    chars = list(code)
    state = "code"
    quote = None
    i = 0

    def blank(pos):
        if chars[pos] not in "\r\n":
            chars[pos] = " "

    while i < len(code):
        current = code[i]
        following = code[i + 1] if i + 1 < len(code) else ""
        if state == "code":
            if current == "/" and following == "/":
                blank(i)
                blank(i + 1)
                state = "line_comment"
                i += 2
                continue
            if current == "/" and following == "*":
                blank(i)
                blank(i + 1)
                state = "block_comment"
                i += 2
                continue
            if current in ("\"", "'"):
                quote = current
                state = "string" if current == "\"" else "char"
                if mask_literals:
                    blank(i)
                i += 1
                continue
        elif state == "line_comment":
            if current in "\r\n":
                state = "code"
            else:
                blank(i)
        elif state == "block_comment":
            if current == "*" and following == "/":
                blank(i)
                blank(i + 1)
                state = "code"
                i += 2
                continue
            blank(i)
        else:  # string or character literal
            if current == "\\":
                if mask_literals:
                    blank(i)
                    if i + 1 < len(code):
                        blank(i + 1)
                i += 2
                continue
            if current == quote:
                state = "code"
                quote = None
            if mask_literals:
                blank(i)
        i += 1
    return "".join(chars)


def _c_token_count(comment_stripped_code):
    """Return the existing regex token-count proxy; this is not a C AST count."""
    return len(re.findall(r"[A-Za-z_]\w*|\d+|\S", comment_stripped_code))


def extract_defect_features(code, language="c"):
    """Return C defect-risk estimates using lexical regex heuristics, not a C AST."""
    s = _mask_c_noncode(code)
    code_without_comments = _mask_c_noncode(code, mask_literals=False)
    branches = len(re.findall(r"\bif\b|\bcase\b|\bswitch\b|\?", s))
    loops = len(re.findall(r"\bfor\b|\bwhile\b", s))
    return {
        "language": language,
        "source_code": code,
        "function_length": len(code.splitlines()),
        "branch_count": branches,
        "loop_count": loops,
        "pointer_usage": len(_PTR.findall(s)),
        "memory_operation_count": len(_MEM.findall(s)),
        "dangerous_api_count": len(_DANGER.findall(s)),
        "exception_handling_count": len(_ERR.findall(s)),
        "null_check_count": len(_NULL.findall(s)),
        "cyclomatic_complexity": 1 + branches + loops + s.count("&&") + s.count("||"),
        "duplicate_code_ratio": _dup_ratio(code_without_comments),
        "uninitialized_variable_count": len(_UNINIT.findall(s)),
        "resource_cleanup_count": len(_CLEAN.findall(s)),
        # No C parser is used: preserve the legacy regex token-count proxy here.
        "ast_node_count": _c_token_count(code_without_comments),
    }


DEFECT_NUM = ["function_length", "branch_count", "loop_count", "pointer_usage",
              "memory_operation_count", "dangerous_api_count", "exception_handling_count",
              "null_check_count", "cyclomatic_complexity", "duplicate_code_ratio",
              "uninitialized_variable_count", "resource_cleanup_count", "ast_node_count"]


def defect_vector(f):
    return [f[k] for k in DEFECT_NUM]


def prep(rows):
    """log1p tames the long tails of the count features."""
    return np.log1p(np.asarray(rows, dtype=float))
