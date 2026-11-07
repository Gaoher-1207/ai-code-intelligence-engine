"""Evaluate the saved models on the held-out test sets and write confusion matrices + RESULTS.md.

Run:  python make_report.py      (from hajera/training)
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from dataset_preparation import (
    devign_quality_report,
    ensure_no_devign_exact_overlap,
    load_complexity_dataset,
    load_devign_splits,
    prepare_complexity_dataset,
)
from schema_features import (complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, prep)

HAJERA = Path(__file__).resolve().parents[1]
MODELS, EVAL = HAJERA / "models", HAJERA / "evaluation"


def _plot(y_true, y_pred, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize="true",
                                            values_format=".2f", cmap="Blues",
                                            xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eval_complexity(cx):
    bundle = joblib.load(MODELS / "complexity_python_schema_lr.joblib")
    prepared = prepare_complexity_dataset(cx)
    test_df = prepared["splits"]["test"]
    feats = [extract_complexity_features(s) for s in test_df["src"]]
    num = prep([complexity_vector(f) for f in feats])
    y = test_df["complexity"].reset_index(drop=True)
    A = hstack([bundle["vectorizer"].transform(test_df["src"]),
                csr_matrix(bundle["scaler"].transform(num))]).tocsr()
    pred = bundle["model"].predict(A)
    _plot(y, pred, list(bundle["model"].classes_),
          "Complexity (Python): normalized confusion matrix", EVAL / "complexity_confusion.png")
    return {"n": len(y), "acc": accuracy_score(y, pred),
            "f1": f1_score(y, pred, average="macro"),
            "dataset_summary": prepared["summary"],
            "quarantined_group_ids": prepared["quarantined_group_ids"]}


def eval_defect(test_df):
    bundle = joblib.load(MODELS / "defect_c_schema_lr.joblib")
    feats = [extract_defect_features(c) for c in test_df["func"]]
    num = prep([defect_vector(f) for f in feats])
    A = hstack([bundle["vectorizer"].transform(test_df["func"]),
                csr_matrix(bundle["scaler"].transform(num))]).tocsr()
    pred = bundle["model"].predict(A)
    y = test_df["target"].astype(bool)
    _plot(y, pred, [False, True], "Defect (C): normalized confusion matrix (True = insecure)",
          EVAL / "defect_confusion.png")
    return {"n": len(y), "acc": accuracy_score(y, pred), "f1": f1_score(y, pred, average="macro")}


def write_results(c, d):
    text = f"""# Results

Both models are baselines. All numbers are on held-out test data.

| Model | Data | Test size | Accuracy | Macro F1 |
|---|---|---|---|---|
| Time complexity (Python, 7 classes) | CodeComplex, split by problem | {c['n']} | {c['acc']:.1%} | {c['f1']:.2f} |
| Defect risk (C, insecure vs secure) | CodeXGLUE / Devign test split | {d['n']} | {d['acc']:.1%} | {d['f1']:.2f} |

## Complexity quarantine

Original CodeComplex records: {c['dataset_summary']['original_record_count']}<br>
Quarantined conflicting groups: {c['dataset_summary']['quarantined_group_count']}<br>
Quarantined records: {c['dataset_summary']['quarantined_record_count']}<br>
Records used for modeling: {c['dataset_summary']['modeling_record_count']}<br>
Split sizes (train / validation / test): {c['dataset_summary']['split_record_counts']['train']} / {c['dataset_summary']['split_record_counts']['validation']} / {c['dataset_summary']['split_record_counts']['test']}

Confusion matrices: `complexity_confusion.png`, `defect_confusion.png`.

## What we learned
- The schema fields alone are weak predictors: 36.5% for complexity (7 classes) and 55.1% for defects
  (about the same as always guessing "secure"). The code text carries most of the signal.
- An earlier baseline that used only our own loop and nesting counts plus code text scored 53.4% (complexity)
  and 63.0% (defect), within noise of the schema-based models.
- Complexity: best on constant and quadratic programs, weakest on `np` and `logn`.
- Defect: `target=True` means insecure code (dataset card). Recall for insecure code is only about 54%.

## Limits
- Static features and text cannot prove a bug or guarantee exact Big-O.
- Defect labels come from the Devign dataset (C functions from open-source projects); results may not carry
  over to other code styles.
- For C, `ast_node_count` is a token count and several counts are pattern-matching estimates until a C parser
  is added.
"""
    (EVAL / "RESULTS.md").write_text(text, encoding="utf-8")
    summary = c["dataset_summary"]
    quarantine_text = (
        "CodeComplex quarantine summary\n\n"
        f"Original records: {summary['original_record_count']}\n"
        f"Exact conflicting groups: {summary['exact_conflicting_group_count']}\n"
        f"Exact-conflict records: {summary['exact_conflicting_record_count']}\n"
        f"Normalized conflicting groups: {summary['normalized_conflicting_group_count']}\n"
        f"Records in normalized-conflict groups: {summary['normalized_conflicting_record_count']}\n"
        f"Quarantined groups: {summary['quarantined_group_count']}\n"
        f"Quarantined records: {summary['quarantined_record_count']}\n"
        f"Modeling records: {summary['modeling_record_count']}\n\n"
        f"Original class counts: {summary['original_label_counts']}\n"
        f"Quarantined class counts: {summary['quarantined_label_counts']}\n"
        f"Modeling class counts: {summary['modeling_label_counts']}\n"
        f"Split record counts: {summary['split_record_counts']}\n"
        f"Split class counts: {summary['split_label_counts']}\n\n"
        "Quarantined normalized source group IDs:\n"
        + "\n".join(c["quarantined_group_ids"])
        + "\n"
    )
    (EVAL / "complexity_quarantine_report.txt").write_text(quarantine_text, encoding="utf-8")


if __name__ == "__main__":
    complexity = load_complexity_dataset()
    c = eval_complexity(complexity)
    print("complexity:", c)
    defect = load_devign_splits()
    print("Devign data quality:", devign_quality_report(defect))
    ensure_no_devign_exact_overlap(defect)
    d = eval_defect(defect["test"])
    print("defect:", d)
    write_results(c, d)
    print("wrote confusion matrices and RESULTS.md")
