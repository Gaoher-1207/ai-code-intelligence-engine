"""Streamlit UI for Gaoher analysis and Hajera model predictions.

Run from the repository root with ``streamlit run frontend/app.py``.
"""

import pandas as pd
import streamlit as st

from gaoher.ml_integration import predict_c_defect, predict_python_complexity


st.set_page_config(page_title="AI Code Intelligence Engine", layout="wide")

EXAMPLES = {
    "Python": {
        "Sum of a list": "def total(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
        "Nested loops": "def pairs(xs):\n    c = 0\n    for i in xs:\n        for j in xs:\n            c += i * j\n    return c\n",
        "Binary search": (
            "def search(a, x):\n    lo, hi = 0, len(a)\n"
            "    while lo < hi:\n        mid = (lo + hi) // 2\n"
            "        if a[mid] < x:\n            lo = mid + 1\n"
            "        else:\n            hi = mid\n    return lo\n"
        ),
        "Broken code": "def f(xs):\n    for x in xs\n        print(x)\n",
    },
    "C": {
        "Unsafe copy": "void copy(char *s) {\n    char buf[8];\n    strcpy(buf, s);\n}\n",
        "Checked allocation": (
            "int *make(int n) {\n    int *p = malloc(n * sizeof(int));\n"
            "    if (p == NULL) {\n        return NULL;\n    }\n"
            "    return p;\n}\n"
        ),
    },
}


st.title("AI Code Intelligence Engine")
st.caption(
    "Static Python analysis with trained complexity estimates and trained C defect-risk predictions."
)

left, right = st.columns([3, 2])

with left:
    language = st.radio("Language", ["Python", "C"], horizontal=True)
    example = st.selectbox("Load an example", list(EXAMPLES[language]))
    code = st.text_area(
        "Code", value=EXAMPLES[language][example], height=320,
        key=f"code_{language}_{example}",
    )
    run = st.button("Analyze", type="primary")

with right:
    st.subheader("Result")
    if not run:
        st.info("Paste code and click Analyze.")
    elif not code.strip():
        st.warning("Please enter some code.")
    elif language == "Python":
        try:
            result = predict_python_complexity(code)
        except Exception as exc:
            st.error(f"Python analysis failed: {exc}")
        else:
            prediction = result["prediction"]
            analysis = result["analysis"]
            first, second = st.columns(2)
            first.metric("Estimated time complexity", prediction["predicted_complexity"])
            second.metric("Model confidence", f"{prediction['confidence']:.0%}")
            if prediction["confidence"] < 0.5:
                st.warning("Low confidence: treat this model output as a rough estimate.")
            st.caption(
                f"Parsed {len(analysis['function_names'])} function(s); "
                f"{analysis['loop_count']} loop(s), maximum nesting "
                f"{analysis['max_loop_nesting']}."
            )
            st.markdown("**Gaoher features checked against Hajera's contract**")
            st.dataframe(
                pd.DataFrame(
                    {"feature": list(result["features"]),
                     "value": [str(value) for value in result["features"].values()]}
                ),
                hide_index=True,
                width="stretch",
            )
    else:
        try:
            result = predict_c_defect(code)
        except Exception as exc:
            st.error(f"C defect prediction failed: {exc}")
        else:
            prediction = result["prediction"]
            risk = prediction["risk_probability"]
            st.metric("Defect risk", f"{risk:.0%}", prediction["predicted_label"])
            st.progress(min(max(risk, 0.0), 1.0))
            features = {
                key: value for key, value in result["features"].items()
                if key != "source_code"
            }
            st.markdown("**Hajera defect features**")
            st.dataframe(
                pd.DataFrame(
                    {"feature": list(features),
                     "value": [str(value) for value in features.values()]}
                ),
                hide_index=True,
                width="stretch",
            )

st.divider()
st.caption(
    "Model outputs are statistical estimates. Static analysis cannot prove a bug "
    "or guarantee exact runtime complexity."
)
