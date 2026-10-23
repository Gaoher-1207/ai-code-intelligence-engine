from pathlib import Path

import joblib
from scipy.sparse import hstack, csr_matrix

from schema_features import (complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, prep)

MODELS = Path(__file__).resolve().parents[1] / "models"
_cx = joblib.load(MODELS / "complexity_python_schema_lr.joblib")
_df = joblib.load(MODELS / "defect_c_schema_lr.joblib")

BIGO = {"constant": "O(1)", "logn": "O(log n)", "linear": "O(n)", "nlogn": "O(n log n)",
        "quadratic": "O(n^2)", "cubic": "O(n^3)", "np": "exponential / NP-hard"}


def predict_complexity(python_code):
    f = extract_complexity_features(python_code)
    num = _cx["scaler"].transform(prep([complexity_vector(f)]))
    A = hstack([_cx["vectorizer"].transform([python_code]), csr_matrix(num)]).tocsr()
    p = _cx["model"].predict_proba(A)[0]
    i = int(p.argmax())
    cls = str(_cx["model"].classes_[i])
    return {"predicted_complexity": BIGO.get(cls, cls), "complexity_class": cls,
            "confidence": round(float(p[i]), 3)}


def predict_defect(c_code):
    f = extract_defect_features(c_code)
    num = _df["scaler"].transform(prep([defect_vector(f)]))
    A = hstack([_df["vectorizer"].transform([c_code]), csr_matrix(num)]).tocsr()
    risk = float(_df["model"].predict_proba(A)[0][1])  # probability of target=True
    return {"risk_probability": round(risk, 3),
            "predicted_label": "vulnerable" if risk >= 0.5 else "non-vulnerable"}


if __name__ == "__main__":
    tests = {
        "constant": "def f(a, b):\n    return a + b\n",
        "linear": "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
        "quadratic": "def f(xs):\n    c = 0\n    for i in xs:\n        for j in xs:\n            c += i * j\n    return c\n",
        "logn": "def f(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n",
    }
    for name, code in tests.items():
        print(name, "->", predict_complexity(code))
    print("defect:", predict_defect("void f(char *s){ char b[8]; strcpy(b, s); }"))
