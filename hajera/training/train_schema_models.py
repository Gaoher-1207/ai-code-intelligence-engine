"""Train both models on the shared-schema features (+ code text) and compare with the first baselines."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from schema_features import (complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, prep, validate_complexity_features,
                             validate_defect_features)

HAJERA = Path(__file__).resolve().parents[1]
MODELS, EVAL = HAJERA / "models", HAJERA / "evaluation"
PY_DATA = Path("D:/projects/datasets/CodeComplex/data/python_data.jsonl")
OLD_CX, OLD_DEFECT = 0.534, 0.63  # accuracy of the first baselines


def train_complexity(cx):
    feats = [extract_complexity_features(s) for s in cx["src"]]
    for f in feats:
        validate_complexity_features(f)
    num = prep([complexity_vector(f) for f in feats])
    y, groups = cx["complexity"].reset_index(drop=True), cx["problem"].reset_index(drop=True)
    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(num, y, groups))

    schema_only = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    schema_only.fit(num[tr], y.iloc[tr])
    acc_schema = accuracy_score(y.iloc[te], schema_only.predict(num[te]))

    src = cx["src"].reset_index(drop=True)
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w*|[^\sA-Za-z_0-9]", ngram_range=(1, 3),
                          max_features=80000, sublinear_tf=True)
    sc = StandardScaler().fit(num[tr])
    A_tr = hstack([vec.fit_transform(src.iloc[tr]), csr_matrix(sc.transform(num[tr]))]).tocsr()
    A_te = hstack([vec.transform(src.iloc[te]), csr_matrix(sc.transform(num[te]))]).tocsr()
    model = LogisticRegression(max_iter=3000, C=5, class_weight="balanced").fit(A_tr, y.iloc[tr])
    pred = model.predict(A_te)
    acc = accuracy_score(y.iloc[te], pred)
    report = classification_report(y.iloc[te], pred)
    print(report)
    print(f"COMPLEXITY  schema features only: {acc_schema:.3f} | schema + code text: {acc:.3f} | first baseline: {OLD_CX}")
    (EVAL / "complexity_python_schema_report.txt").write_text(
        report + f"\nschema-only RandomForest accuracy: {acc_schema:.3f}\n")
    if acc >= OLD_CX:
        joblib.dump({"vectorizer": vec, "scaler": sc, "model": model, "version": "schema-v1"},
                    MODELS / "complexity_python_schema_lr.joblib")
        print("saved complexity schema model")
    else:
        print("not saved: not better than the first baseline")


def train_defect(train_df, test_df):
    def build(df):
        feats = [extract_defect_features(c) for c in df["func"]]
        for f in feats:
            validate_defect_features(f)
        return prep([defect_vector(f) for f in feats])

    n_tr, n_te = build(train_df), build(test_df)
    y_tr, y_te = train_df["target"].astype(bool), test_df["target"].astype(bool)

    gb = HistGradientBoostingClassifier(random_state=42).fit(n_tr, y_tr)
    acc_schema = accuracy_score(y_te, gb.predict(n_te))

    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w+|[^\sA-Za-z_0-9]", max_features=50000, ngram_range=(1, 2))
    sc = StandardScaler().fit(n_tr)
    A_tr = hstack([vec.fit_transform(train_df["func"]), csr_matrix(sc.transform(n_tr))]).tocsr()
    A_te = hstack([vec.transform(test_df["func"]), csr_matrix(sc.transform(n_te))]).tocsr()
    model = LogisticRegression(max_iter=3000).fit(A_tr, y_tr)
    pred = model.predict(A_te)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred)
    print(report)
    print(f"DEFECT  schema features only: {acc_schema:.3f} | schema + code text: {acc:.3f} | first baseline: {OLD_DEFECT}")
    (EVAL / "defect_c_schema_report.txt").write_text(
        report + f"\nschema-only gradient boosting accuracy: {acc_schema:.3f}\n")
    if acc >= OLD_DEFECT:
        joblib.dump({"vectorizer": vec, "scaler": sc, "model": model, "version": "schema-v1"},
                    MODELS / "defect_c_schema_lr.joblib")
        print("saved defect schema model")
    else:
        print("not saved: not better than the first baseline")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    EVAL.mkdir(exist_ok=True)
    train_complexity(pd.read_json(PY_DATA, lines=True))
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    train_defect(ds["train"].to_pandas(), ds["test"].to_pandas())
