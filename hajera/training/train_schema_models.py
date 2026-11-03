"""Train both models on the shared-schema features (+ code text) and compare with the first baselines."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from dataset_preparation import (
    complexity_quality_report,
    ensure_no_devign_exact_overlap,
    load_complexity_dataset,
    load_devign_splits,
    split_complexity_dataset,
)
from schema_features import (complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, prep, validate_complexity_features,
                             validate_defect_features)

HAJERA = Path(__file__).resolve().parents[1]
MODELS, EVAL = HAJERA / "models", HAJERA / "evaluation"


def train_complexity(cx):
    splits = split_complexity_dataset(cx)
    train_df, validation_df = splits["train"], splits["validation"]
    train_features = [extract_complexity_features(s) for s in train_df["src"]]
    validation_features = [extract_complexity_features(s) for s in validation_df["src"]]
    for f in train_features + validation_features:
        validate_complexity_features(f)
    num_train = prep([complexity_vector(f) for f in train_features])
    num_validation = prep([complexity_vector(f) for f in validation_features])
    y_train = train_df["complexity"].reset_index(drop=True)
    y_validation = validation_df["complexity"].reset_index(drop=True)

    majority = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_train), 1)), y_train
    )
    majority_acc = accuracy_score(y_validation, majority.predict(np.zeros((len(y_validation), 1))))
    schema_only = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    schema_only.fit(num_train, y_train)
    acc_schema = accuracy_score(y_validation, schema_only.predict(num_validation))

    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w*|[^\sA-Za-z_0-9]", ngram_range=(1, 3),
                          max_features=80000, sublinear_tf=True)
    sc = StandardScaler().fit(num_train)
    A_train = hstack([vec.fit_transform(train_df["src"]), csr_matrix(sc.transform(num_train))]).tocsr()
    A_validation = hstack([
        vec.transform(validation_df["src"]), csr_matrix(sc.transform(num_validation))
    ]).tocsr()
    model = LogisticRegression(max_iter=3000, C=5, class_weight="balanced").fit(A_train, y_train)
    pred = model.predict(A_validation)
    acc = accuracy_score(y_validation, pred)
    report = classification_report(y_validation, pred, zero_division=0)
    print(report)
    print(
        f"COMPLEXITY validation  schema features only: {acc_schema:.3f} | "
        f"schema + code text: {acc:.3f} | majority baseline: {majority_acc:.3f}"
    )
    (EVAL / "complexity_python_schema_report.txt").write_text(
        "Validation results only; the final test split was not used for model selection.\n\n"
        + report + f"\nmajority validation accuracy: {majority_acc:.3f}\n"
        + f"schema-only RandomForest validation accuracy: {acc_schema:.3f}\n")
    if acc >= majority_acc:
        # Refit the selected model on train+validation; leave the outer test split untouched.
        fit_df = pd.concat([train_df, validation_df], ignore_index=True)
        fit_features = train_features + validation_features
        fit_num = prep([complexity_vector(f) for f in fit_features])
        final_vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w*|[^\sA-Za-z_0-9]", ngram_range=(1, 3),
                                    max_features=80000, sublinear_tf=True)
        final_scaler = StandardScaler().fit(fit_num)
        final_matrix = hstack([
            final_vec.fit_transform(fit_df["src"]), csr_matrix(final_scaler.transform(fit_num))
        ]).tocsr()
        final_model = LogisticRegression(max_iter=3000, C=5, class_weight="balanced")
        final_model.fit(final_matrix, fit_df["complexity"])
        joblib.dump({"vectorizer": final_vec, "scaler": final_scaler, "model": final_model,
                     "version": "schema-v1", "dataset_split_version": "problem-grouped-80-20-v1"},
                    MODELS / "complexity_python_schema_lr.joblib")
        print("saved complexity schema model selected on validation data")
    else:
        print("not saved: validation accuracy did not meet the existing baseline threshold")


def train_defect(train_df, validation_df):
    def build(df):
        feats = [extract_defect_features(c) for c in df["func"]]
        for f in feats:
            validate_defect_features(f)
        return prep([defect_vector(f) for f in feats])

    n_tr, n_val = build(train_df), build(validation_df)
    y_tr, y_val = train_df["target"].astype(bool), validation_df["target"].astype(bool)

    majority = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_tr), 1)), y_tr
    )
    majority_acc = accuracy_score(y_val, majority.predict(np.zeros((len(y_val), 1))))
    gb = HistGradientBoostingClassifier(random_state=42).fit(n_tr, y_tr)
    acc_schema = accuracy_score(y_val, gb.predict(n_val))

    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w+|[^\sA-Za-z_0-9]", max_features=50000, ngram_range=(1, 2))
    sc = StandardScaler().fit(n_tr)
    A_tr = hstack([vec.fit_transform(train_df["func"]), csr_matrix(sc.transform(n_tr))]).tocsr()
    A_val = hstack([vec.transform(validation_df["func"]), csr_matrix(sc.transform(n_val))]).tocsr()
    model = LogisticRegression(max_iter=3000).fit(A_tr, y_tr)
    pred = model.predict(A_val)
    acc = accuracy_score(y_val, pred)
    report = classification_report(y_val, pred, zero_division=0)
    print(report)
    print(
        f"DEFECT validation  schema features only: {acc_schema:.3f} | "
        f"schema + code text: {acc:.3f} | majority baseline: {majority_acc:.3f}"
    )
    (EVAL / "defect_c_schema_report.txt").write_text(
        "Validation results only; the official test split was not used for model selection.\n\n"
        + report + f"\nmajority validation accuracy: {majority_acc:.3f}\n"
        + f"schema-only gradient boosting validation accuracy: {acc_schema:.3f}\n")
    if acc >= majority_acc:
        fit_df = pd.concat([train_df, validation_df], ignore_index=True)
        fit_num = np.vstack([n_tr, n_val])
        final_vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w+|[^\sA-Za-z_0-9]",
                                    max_features=50000, ngram_range=(1, 2))
        final_scaler = StandardScaler().fit(fit_num)
        final_matrix = hstack([
            final_vec.fit_transform(fit_df["func"]),
            csr_matrix(final_scaler.transform(fit_num)),
        ]).tocsr()
        final_model = LogisticRegression(max_iter=3000)
        final_model.fit(final_matrix, fit_df["target"].astype(bool))
        joblib.dump({"vectorizer": final_vec, "scaler": final_scaler, "model": final_model,
                     "version": "schema-v1", "dataset_split_version": "official-train-validation-v1"},
                    MODELS / "defect_c_schema_lr.joblib")
        print("saved defect schema model selected on validation data")
    else:
        print("not saved: validation accuracy did not meet the existing baseline threshold")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    EVAL.mkdir(exist_ok=True)
    complexity = load_complexity_dataset()
    print("CodeComplex data quality:", complexity_quality_report(complexity))
    split_complexity_dataset(complexity)  # Fail closed if exact source code crosses a split.
    train_complexity(complexity)

    defect = load_devign_splits()
    ensure_no_devign_exact_overlap(defect)
    train_defect(defect["train"], defect["validation"])
