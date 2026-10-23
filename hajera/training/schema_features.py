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
    for n in ast.walk(loop):
        if isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.FloorDiv, ast.Div, ast.RShift, ast.Mult)) \
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (int, float)) \
                and n.value.value >= 2:
            return True
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.FloorDiv, ast.RShift)) \
                and isinstance(n.right, ast.Constant):
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


def extract_complexity_features(code, language="python"):
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    names = {_call_name(c) for c in calls}
    rec = 0
    for f in nodes:
        if isinstance(f, ast.FunctionDef):
            rec += sum(1 for n in ast.walk(f) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == f.name)
    size_var = (
        bool(names & {"input", "len", "readline", "read", "readlines"})
        or any(isinstance(c.func, ast.Name) and c.func.id == "range"
               and any(not isinstance(a, ast.Constant) for a in c.args) for c in calls)
        or any(isinstance(n, ast.FunctionDef) and n.args.args for n in nodes)
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
        "branch_count": sum(isinstance(n, (ast.If, ast.IfExp)) for n in nodes),
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
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def extract_defect_features(code, language="c"):
    s = _COMMENT.sub("", code)
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
        "duplicate_code_ratio": _dup_ratio(code),
        "uninitialized_variable_count": len(_UNINIT.findall(s)),
        "resource_cleanup_count": len(_CLEAN.findall(s)),
        # no C parser yet: token count is used as a stand-in for AST node count
        "ast_node_count": len(re.findall(r"[A-Za-z_]\w*|\d+|\S", s)),
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
