import ast
from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

MODELS = Path(__file__).resolve().parent.parent / "models"
_cx = joblib.load(MODELS / "complexity_python_tfidf_lr.joblib")
_df = joblib.load(MODELS / "defect_c_tfidf_lr.joblib")

def _nest(node, depth=0):
    best = depth
    for c in ast.iter_child_nodes(node):
        d = depth + 1 if isinstance(c, (ast.For, ast.While)) else depth
        best = max(best, _nest(c, d))
    return best

def extract_features(code):
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    rec = 0
    for f in nodes:
        if isinstance(f, ast.FunctionDef):
            rec += sum(1 for n in ast.walk(f) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == f.name)
    bound = "unknown"
    for n in nodes:
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) \
           and getattr(n.iter.func, "id", "") == "range":
            arg = n.iter.args[-1] if n.iter.args else None
            bound = "constant" if isinstance(arg, ast.Constant) else "input_dependent"
    return {
        "function_length": len(code.splitlines()),
        "loop_count": sum(isinstance(n, (ast.For, ast.While)) for n in nodes),
        "max_loop_nesting": _nest(tree),
        "branch_count": sum(isinstance(n, (ast.If, ast.IfExp)) for n in nodes),
        "recursion_count": rec,
        "function_call_count": len(calls),
        "ast_node_count": len(nodes),
        "input_size_variable": int(any(getattr(c.func, "id", "") in ("input", "len") for c in calls)),
        "loop_bound_type": bound,
        "uses_list": int(any(isinstance(n, ast.List) for n in nodes)),
        "uses_dict": int(any(isinstance(n, ast.Dict) for n in nodes)),
        "uses_set": int(any(isinstance(n, ast.Set) for n in nodes)),
    }

def predict_complexity(python_code):
    feats = pd.get_dummies(pd.DataFrame([extract_features(python_code)]),
                           columns=["loop_bound_type"])
    feats = feats.reindex(columns=_cx["columns"], fill_value=0).astype(float)
    A = hstack([_cx["vectorizer"].transform([python_code]),
                csr_matrix(_cx["scaler"].transform(feats))]).tocsr()
    probs = _cx["model"].predict_proba(A)[0]
    i = probs.argmax()
    return {"complexity_class": str(_cx["model"].classes_[i]),
            "confidence": round(float(probs[i]), 3)}

def predict_defect(c_code):
    X = _df["vectorizer"].transform([c_code])
    return {"risk_probability": round(float(_df["model"].predict_proba(X)[0][1]), 3)}

if __name__ == "__main__":
    tests = {
        "constant": "def f(a, b):\n    return a + b\n",
        "linear": "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
        "quadratic": "def f(xs):\n    c = 0\n    for i in xs:\n        for j in xs:\n            c += i * j\n    return c\n",
    }
    for name, code in tests.items():
        print(name, "->", predict_complexity(code))
    print("defect:", predict_defect("void f(char *s){ char b[8]; strcpy(b, s); }"))