"""AI Code Intelligence Engine - Streamlit frontend.

Run from the repo root:  streamlit run frontend/app.py
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hajera" / "training"))

st.set_page_config(page_title="AI Code Intelligence Engine", page_icon="🧠", layout="wide")


@st.cache_resource
def load_models():
    import predict
    import schema_features
    return predict, schema_features


EXAMPLES = {
    "Python": {
        "Sum of a list": "def total(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
        "Nested loops": "def pairs(xs):\n    c = 0\n    for i in xs:\n        for j in xs:\n            c += i * j\n    return c\n",
        "Binary search": "def search(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n",
        "Broken code": "def f(xs):\n    for x in xs\n        print(x)\n",
    },
    "C": {
        "Unsafe copy": "void copy(char *s) {\n    char buf[8];\n    strcpy(buf, s);\n}\n",
        "Checked allocation": "int *make(int n) {\n    int *p = malloc(n * sizeof(int));\n    if (p == NULL) {\n        return NULL;\n    }\n    return p;\n}\n",
    },
}

st.title("AI Code Intelligence Engine")
st.caption("Estimates time complexity (Python) and defect risk (C) with ML models trained on public datasets.")

try:
    predict, sf = load_models()
except Exception as e:  # missing model files, etc.
    st.error(f"Could not load the models: {e}")
    st.stop()

left, right = st.columns([3, 2])

with left:
    language = st.radio("Language", ["Python", "C"], horizontal=True)
    example = st.selectbox("Load an example", list(EXAMPLES[language]))
    code = st.text_area("Code", value=EXAMPLES[language][example], height=320,
                        key=f"code_{language}_{example}")
    run = st.button("Analyze", type="primary")

with right:
    st.subheader("Result")
    if not run:
        st.info("Paste code and click Analyze.")
    elif not code.strip():
        st.warning("Please enter some code.")
    elif language == "Python":
        try:
            ast.parse(code)
        except SyntaxError as e:
            st.error(f"Syntax error on line {e.lineno}: {e.msg}")
            st.stop()
        res = predict.predict_complexity(code)
        c1, c2 = st.columns(2)
        c1.metric("Estimated time complexity", res["predicted_complexity"])
        c2.metric("Confidence", f"{res['confidence']:.0%}")
        if res["confidence"] < 0.5:
            st.warning("Low confidence: treat this as a rough guess.")
        feats = sf.extract_complexity_features(code)
        st.markdown("**Extracted features**")
        st.dataframe(pd.DataFrame({"feature": list(feats), "value": [str(v) for v in feats.values()]}),
                     hide_index=True, width="stretch")
    else:
        res = predict.predict_defect(code)
        risk = res["risk_probability"]
        st.metric("Defect risk", f"{risk:.0%}", res["predicted_label"])
        st.progress(min(max(risk, 0.0), 1.0))
        feats = {k: v for k, v in sf.extract_defect_features(code).items() if k != "source_code"}
        st.markdown("**Extracted features**")
        st.dataframe(pd.DataFrame({"feature": list(feats), "value": [str(v) for v in feats.values()]}),
                     hide_index=True, width="stretch")

st.divider()
st.caption("These are statistical estimates, not proof. Static metrics cannot guarantee exact Big-O "
           "or prove that a bug exists. The Python check only catches syntax errors.")