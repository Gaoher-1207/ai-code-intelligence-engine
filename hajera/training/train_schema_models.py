"""Train both models on the shared-schema features (+ code text) and compare with the first baselines."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             average_precision_score, roc_auc_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from dataset_preparation import (
    complexity_quality_report,
    ensure_no_devign_exact_overlap,
    load_complexity_dataset,
    load_devign_splits,
    prepare_complexity_dataset,
)
from schema_features import (COMPLEXITY_VECTOR_FEATURE_NAMES, DEFECT_VECTOR_FEATURE_NAMES,
                             complexity_vector, defect_vector, extract_complexity_features,
                             extract_defect_features, prep, validate_complexity_features,
                             validate_defect_features)

HAJERA = Path(__file__).resolve().parents[1]
MODELS, EVAL = HAJERA / "models", HAJERA / "evaluation"
def _classification_metrics(y_true, predictions, labels, probabilities=None, positive_label=None):
    """Return JSON-serializable metrics shared by validation and final test reports."""
    report = classification_report(y_true, predictions, labels=labels,
                                   output_dict=True, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, labels=labels,
                                                  average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, labels=labels,
                                            average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class": {str(key): value for key, value in report.items()
                      if key not in {"accuracy", "macro avg", "weighted avg"}},
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "confusion_matrix_labels": [str(label) for label in labels],
    }
    if probabilities is not None and positive_label is not None:
        binary_truth = np.asarray(y_true) == positive_label
        positive_index = list(labels).index(positive_label)
        result["pr_auc"] = float(average_precision_score(binary_truth, probabilities[:, positive_index]))
        result["roc_auc"] = (float(roc_auc_score(binary_truth, probabilities[:, positive_index]))
                             if len(np.unique(binary_truth)) == 2 else None)
    return result


def select_random_forest(x_train, y_train, x_validation, y_validation):
    """Fit the fixed candidate set on train and choose by validation macro-F1."""
    labels = sorted(set(np.asarray(y_train).tolist()))
    if len(labels) < 2 or set(np.asarray(y_validation).tolist()) - set(labels):
        raise ValueError("Train and validation must share at least two known classes")
    candidates = []
    for tree_count in (200, 400):
        estimator = RandomForestClassifier(
            n_estimators=tree_count, class_weight="balanced", random_state=42,
            n_jobs=-1, max_features="sqrt", min_samples_leaf=1,
        )
        estimator.fit(x_train, y_train)
        metrics = _classification_metrics(
            y_validation, estimator.predict(x_validation), labels,
        )
        candidates.append((metrics["macro_f1"], metrics["accuracy"],
                           -tree_count, tree_count, estimator, metrics))
    _, _, _, tree_count, estimator, metrics = max(candidates, key=lambda item: item[:3])
    return estimator, tree_count, metrics


def train_schema_baseline(task_name, train_frame, validation_frame, test_frame,
                          source_column, target_column, feature_extractor, vectorizer,
                          model_path, report_path, positive_label=None, dataset_details=None,
                          vector_feature_names=None):
    """Select a schema-feature RandomForest on validation, then evaluate test once.

    Every candidate and the majority baseline are fitted on the training split only.
    The chosen estimator is saved as fitted on training data, so validation and test
    records never enter its fit.
    """
    def transform(frame):
        features = [feature_extractor(source) for source in frame[source_column].astype(str)]
        vectors = np.asarray([vectorizer(features_row) for features_row in features], dtype=float)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError(f"{task_name} features must form a finite two-dimensional matrix")
        return features, vectors

    train_features, x_train = transform(train_frame)
    _, x_validation = transform(validation_frame)
    feature_names = list(vector_feature_names or train_features[0].keys())
    if len(feature_names) != x_train.shape[1]:
        raise ValueError(
            f"{task_name} feature-name count {len(feature_names)} does not match "
            f"vector dimension {x_train.shape[1]}"
        )
    y_train = train_frame[target_column].to_numpy()
    y_validation = validation_frame[target_column].to_numpy()
    labels = sorted(set(y_train.tolist()))
    if len(labels) < 2:
        raise ValueError(f"{task_name} training split must contain at least two classes")
    if set(y_validation.tolist()) - set(labels):
        raise ValueError(f"{task_name} validation contains labels absent from training")

    majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    majority_validation = _classification_metrics(
        y_validation, majority.predict(x_validation), labels,
        (majority.predict_proba(x_validation) if positive_label is not None else None),
        positive_label,
    )
    chosen, chosen_tree_count, chosen_validation = select_random_forest(
        x_train, y_train, x_validation, y_validation
    )
    if positive_label is not None:
        chosen_validation = _classification_metrics(
            y_validation, chosen.predict(x_validation), labels,
            chosen.predict_proba(x_validation), positive_label,
        )

    if Path(model_path).exists() or Path(report_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline output for {task_name}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _, x_test = transform(test_frame)
    y_test = test_frame[target_column].to_numpy()
    if set(y_test.tolist()) - set(labels):
        raise ValueError(f"{task_name} test contains labels absent from training")
    train_probabilities = chosen.predict_proba(x_train)
    training_metrics = _classification_metrics(
        y_train, chosen.predict(x_train), labels,
        train_probabilities if positive_label is not None else None, positive_label,
    )
    majority_training = _classification_metrics(
        y_train, majority.predict(x_train), labels,
        (majority.predict_proba(x_train) if positive_label is not None else None),
        positive_label,
    )
    test_probabilities = chosen.predict_proba(x_test)
    test_metrics = _classification_metrics(
        y_test, chosen.predict(x_test), labels,
        test_probabilities if positive_label is not None else None, positive_label,
    )
    majority_test = _classification_metrics(
        y_test, majority.predict(x_test), labels,
        (majority.predict_proba(x_test) if positive_label is not None else None),
        positive_label,
    )
    bundle = {
        "model": chosen,
        "feature_names": feature_names,
        "feature_mode": "schema_numeric",
        "preprocessing": "schema_features.prep",
        "target_labels": labels,
        "version": "schema-random-forest-baseline-v1",
        "random_state": 42,
        "selected_parameters": {"n_estimators": chosen_tree_count,
                                 "class_weight": "balanced", "max_features": "sqrt",
                                 "min_samples_leaf": 1},
    }
    joblib.dump(bundle, model_path)
    report = {
        "task": task_name,
        "dataset": dataset_details or {},
        "method": {
            "features": bundle["feature_names"],
            "feature_dimensions": int(x_train.shape[1]),
            "random_state": 42,
            "candidate_tree_counts": [200, 400],
            "selection_metric": "validation macro-F1 (ties: accuracy, then fewer trees)",
            "selected_parameters": bundle["selected_parameters"],
            "fit_policy": "training split only; validation used for selection; test evaluated after selection",
        },
        "split_sizes": {"train": len(train_frame), "validation": len(validation_frame),
                        "test": len(test_frame)},
        "class_distributions": {
            name: {str(k): int(v) for k, v in frame[target_column].value_counts().items()}
            for name, frame in (("train", train_frame), ("validation", validation_frame),
                                ("test", test_frame))
        },
        "project_distributions": {
            name: {str(k): int(v) for k, v in frame["project"].value_counts().items()}
            for name, frame in (("train", train_frame), ("validation", validation_frame),
                                ("test", test_frame)) if "project" in frame.columns
        },
        "validation": {"random_forest": chosen_validation,
                       "majority_baseline": majority_validation},
        "training_in_sample": {"random_forest": training_metrics,
                               "majority_baseline": majority_training},
        "test": {"random_forest": test_metrics, "majority_baseline": majority_test},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def train_prepared_complexity_baseline():
    from schema_features import extract_complexity_features, complexity_vector

    complexity = prepare_complexity_dataset(load_complexity_dataset())
    cx_splits = complexity["splits"]
    cx_report = train_schema_baseline(
        "CodeComplex seven-class complexity", cx_splits["train"], cx_splits["validation"],
        cx_splits["test"], "src", "complexity", extract_complexity_features,
        complexity_vector, MODELS / "complexity_python_rf_baseline.joblib",
        EVAL / "complexity_python_rf_baseline_report.json",
        dataset_details=complexity["summary"],
        vector_feature_names=COMPLEXITY_VECTOR_FEATURE_NAMES,
    )

    return cx_report


def train_prepared_defect_baseline():
    from schema_features import extract_defect_features, defect_vector

    prepared_dir = HAJERA / "data" / "devign_prepared"
    preparation_report = json.loads(
        (prepared_dir / "preparation_report.json").read_text(encoding="utf-8")
    )
    leakage = preparation_report["leakage_verification"]
    if (not all(leakage.get(key) is True for key in (
            "both_labels_in_every_split", "both_projects_in_every_split",
            "no_connected_component_is_divided", "no_normalized_source_group_crosses_controlled_splits",
            "no_project_commit_pair_crosses_controlled_splits",
            "no_quarantined_record_in_controlled_splits"))
            or leakage.get("connected_component_crossings") != 0
            or leakage.get("quarantined_record_split_overlap_count") != 0):
        raise ValueError("Prepared Devign split leakage verification is not clean")
    defect_splits = {
        name: pd.read_json(prepared_dir / f"{name}.jsonl", lines=True)
        for name in ("train", "validation", "test")
    }
    defect_splits = load_devign_splits(defect_splits)
    expected_counts = preparation_report["prepared_artifact_counts"]
    if {name: len(frame) for name, frame in defect_splits.items()} != expected_counts:
        raise ValueError("Prepared Devign row counts do not match their preparation report")
    defect_report = train_schema_baseline(
        "Devign binary defect prediction", defect_splits["train"],
        defect_splits["validation"], defect_splits["test"], "func", "target",
        extract_defect_features, defect_vector, MODELS / "defect_c_rf_baseline.joblib",
        EVAL / "defect_c_rf_baseline_report.json", positive_label=True,
        dataset_details={"source": str(prepared_dir),
                         "evaluation_scope": "unseen commits within FFmpeg and qemu; not unseen projects",
                         "quarantined_records": preparation_report["quarantined_record_count"],
                         "quarantined_groups": preparation_report["quarantined_group_count"],
                         "leakage_verification": leakage},
        vector_feature_names=DEFECT_VECTOR_FEATURE_NAMES,
    )
    return defect_report


def train_prepared_baselines():
    """Train both authorized schema-feature baselines from prepared splits."""
    defect_report = train_prepared_defect_baseline()
    complexity_report = train_prepared_complexity_baseline()
    return complexity_report, defect_report


def _candidate_specs(task):
    """A small fixed candidate set; every estimator is fitted only on training rows."""
    if task == "complexity":
        return [
            ("rf_depth_8_leaf_5", RandomForestClassifier, {
                "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5,
                "class_weight": "balanced", "max_features": "sqrt", "n_jobs": -1,
                "random_state": 42,
            }),
            ("rf_depth_12_leaf_10", RandomForestClassifier, {
                "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10,
                "class_weight": "balanced", "max_features": "sqrt", "n_jobs": -1,
                "random_state": 42,
            }),
            ("extratrees_depth_12_leaf_5", ExtraTreesClassifier, {
                "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5,
                "class_weight": "balanced", "max_features": "sqrt", "n_jobs": -1,
                "random_state": 42,
            }),
            ("hist_gradient_boosting", HistGradientBoostingClassifier, {
                "max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15,
                "min_samples_leaf": 20, "l2_regularization": 1.0,
                "early_stopping": True, "random_state": 42,
            }),
        ]
    if task == "defect":
        return [
            ("rf_depth_8_leaf_20", RandomForestClassifier, {
                "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20,
                "class_weight": "balanced", "max_features": "sqrt", "n_jobs": -1,
                "random_state": 42,
            }),
            ("extratrees_depth_10_leaf_20", ExtraTreesClassifier, {
                "n_estimators": 400, "max_depth": 10, "min_samples_leaf": 20,
                "class_weight": "balanced", "max_features": "sqrt", "n_jobs": -1,
                "random_state": 42,
            }),
            ("hist_gradient_boosting", HistGradientBoostingClassifier, {
                "max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 7,
                "min_samples_leaf": 40, "l2_regularization": 2.0,
                "early_stopping": True, "random_state": 42,
            }),
            ("logistic_regression_l2", LogisticRegression, {
                "C": 0.3, "class_weight": "balanced", "max_iter": 2000,
                "random_state": 42,
            }),
        ]
    raise ValueError(f"Unknown experiment task: {task}")


def _select_validation_candidate(candidates):
    """Choose by validation macro-F1, then accuracy, retaining stable first tie."""
    if not candidates:
        raise ValueError("No experiment candidates were evaluated")
    return max(candidates, key=lambda candidate: (
        candidate["validation"]["macro_f1"], candidate["validation"]["accuracy"],
        -candidate["order"],
    ))


def _run_task_experiment(task, train_frame, validation_frame, test_frame, source_column,
                         target_column, feature_extractor, vectorizer, feature_names,
                         baseline_report_path, improved_model_path, experiment_report_path,
                         positive_label=None):
    if improved_model_path.exists() or experiment_report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {task} experiment outputs")

    def transform(frame):
        features = [feature_extractor(source) for source in frame[source_column].astype(str)]
        if task == "complexity":
            for feature in features:
                validate_complexity_features(feature)
        else:
            for feature in features:
                validate_defect_features(feature)
        matrix = np.asarray([vectorizer(feature) for feature in features], dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or matrix.shape[1] != len(feature_names):
            raise ValueError(f"{task} feature vectors do not match the stored feature contract")
        return matrix

    x_train = transform(train_frame)
    x_validation = transform(validation_frame)
    y_train = train_frame[target_column].to_numpy()
    y_validation = validation_frame[target_column].to_numpy()
    labels = sorted(set(y_train.tolist()))
    if set(y_validation.tolist()) - set(labels):
        raise ValueError(f"{task} validation contains a class absent from training")

    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    candidate_results = []
    for order, (name, estimator_type, parameters) in enumerate(_candidate_specs(task)):
        estimator = estimator_type(**parameters)
        if estimator_type is HistGradientBoostingClassifier:
            estimator.fit(x_train, y_train,
                          sample_weight=compute_sample_weight("balanced", y_train))
        elif estimator_type is LogisticRegression:
            scaler = StandardScaler().fit(x_train)
            estimator.fit(scaler.transform(x_train), y_train)
            # This numeric feature scaler is fitted only on the training rows.
            parameters = {**parameters, "preprocessing": "StandardScaler fitted on train"}
        else:
            estimator.fit(x_train, y_train)
            scaler = None
        if estimator_type is LogisticRegression:
            validation_input = scaler.transform(x_validation)
            candidate_train_input = scaler.transform(x_train)
        else:
            validation_input = x_validation
            candidate_train_input = x_train
        train_metrics = _classification_metrics(
            y_train, estimator.predict(candidate_train_input), labels,
            estimator.predict_proba(candidate_train_input)
            if positive_label is not None else None, positive_label,
        )
        val_predictions = estimator.predict(validation_input)
        val_probabilities = estimator.predict_proba(validation_input)
        val_metrics = _classification_metrics(
            y_validation, val_predictions, labels,
            val_probabilities if positive_label is not None else None, positive_label,
        )
        candidate_results.append({
            "name": name, "order": order, "estimator": estimator, "scaler": scaler,
            "parameters": parameters, "training": train_metrics, "validation": val_metrics,
        })

    selected = _select_validation_candidate(candidate_results)
    baseline_validation = baseline_report["validation"]["random_forest"]
    baseline_test = baseline_report["test"]["random_forest"]
    baseline_dummy_validation = baseline_report["validation"]["majority_baseline"]
    baseline_dummy_test = baseline_report["test"]["majority_baseline"]

    # Test source rows are first featurized after candidate selection is fixed.
    x_test = transform(test_frame)
    y_test = test_frame[target_column].to_numpy()
    selected_test_input = (selected["scaler"].transform(x_test)
                           if selected["scaler"] is not None else x_test)
    selected_test_metrics = _classification_metrics(
        y_test, selected["estimator"].predict(selected_test_input), labels,
        selected["estimator"].predict_proba(selected_test_input)
        if positive_label is not None else None, positive_label,
    )

    selected_train_input = (selected["scaler"].transform(x_train)
                            if selected["scaler"] is not None else x_train)
    train_metrics = _classification_metrics(
        y_train, selected["estimator"].predict(selected_train_input), labels,
        selected["estimator"].predict_proba(selected_train_input)
        if positive_label is not None else None, positive_label,
    )
    mutual_info = mutual_info_classif(x_train, y_train, random_state=42)
    feature_signal = {
        "computed_on": "training split only",
        "constant_features": [feature_names[i] for i in range(x_train.shape[1])
                              if np.unique(x_train[:, i]).size <= 1],
        "mutual_information_top_features": [
            {"feature": feature_names[i], "score": float(mutual_info[i])}
            for i in np.argsort(mutual_info)[::-1][:min(8, len(feature_names))]
        ],
    }

    bundle = {
        "model": selected["estimator"],
        "scaler": selected["scaler"],
        "feature_names": list(feature_names),
        "feature_mode": "schema_numeric",
        "preprocessing": ("StandardScaler fitted on train, then schema_features.prep"
                          if selected["scaler"] is not None else "schema_features.prep"),
        "target_labels": labels,
        "version": "schema-generalization-experiment-v1",
        "model_family": selected["name"],
        "selected_parameters": selected["parameters"],
        "random_state": 42,
    }
    improved_model_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, improved_model_path)

    baseline_val_f1 = baseline_validation["macro_f1"]
    selected_val_f1 = selected["validation"]["macro_f1"]
    baseline_test_f1 = baseline_test["macro_f1"]
    selected_test_f1 = selected_test_metrics["macro_f1"]
    report = {
        "task": task,
        "split_sizes": {"train": len(train_frame), "validation": len(validation_frame),
                        "test": len(test_frame)},
        "selection": {
            "primary_metric": "validation macro-F1",
            "tie_breaker": "validation accuracy, then fixed candidate order",
            "selected_candidate": selected["name"],
            "selected_parameters": selected["parameters"],
            "test_used_for_selection": False,
            "baseline_validation": baseline_validation,
            "baseline_training_in_sample": baseline_report.get("training_in_sample", {}).get(
                "random_forest"),
            "majority_baseline_validation": baseline_dummy_validation,
            "candidates_validation": [
                {"name": candidate["name"], "parameters": candidate["parameters"],
                 "training_in_sample": candidate["training"],
                 "metrics": candidate["validation"],
                 "delta_macro_f1_vs_saved_rf": candidate["validation"]["macro_f1"] - baseline_val_f1}
                for candidate in candidate_results
            ],
        },
        "training_in_sample": {"selected_candidate": train_metrics},
        "test_after_selection": {
            "selected_candidate": selected_test_metrics,
            "saved_random_forest_baseline": baseline_test,
            "majority_baseline": baseline_dummy_test,
            "delta_macro_f1_vs_saved_rf": selected_test_f1 - baseline_test_f1,
            "test_delta_is_descriptive_only": True,
        },
        "feature_signal_diagnostic": feature_signal,
        "interpretation": {
            "validation_macro_f1_delta_vs_baseline": selected_val_f1 - baseline_val_f1,
            "validation_gain_at_least_0_01": selected_val_f1 - baseline_val_f1 >= 0.01,
            "test_gain_at_least_0_01_descriptive_only": selected_test_f1 - baseline_test_f1 >= 0.01,
        },
        "saved_model": str(improved_model_path),
    }
    experiment_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    return report


def run_generalization_experiments():
    """Compare compact candidate sets and save one validation-selected model per task."""
    from schema_features import extract_complexity_features, extract_defect_features

    complexity = prepare_complexity_dataset(load_complexity_dataset())
    cx = complexity["splits"]
    cx_report = _run_task_experiment(
        "complexity", cx["train"], cx["validation"], cx["test"], "src", "complexity",
        extract_complexity_features, complexity_vector, COMPLEXITY_VECTOR_FEATURE_NAMES,
        EVAL / "complexity_python_rf_baseline_report.json",
        MODELS / "complexity_python_rf_improved.joblib",
        EVAL / "complexity_python_generalization_experiment.json",
    )

    prepared_dir = HAJERA / "data" / "devign_prepared"
    defect = {name: pd.read_json(prepared_dir / f"{name}.jsonl", lines=True)
              for name in ("train", "validation", "test")}
    defect_report = json.loads((prepared_dir / "preparation_report.json").read_text(encoding="utf-8"))
    leakage = defect_report["leakage_verification"]
    if (not leakage.get("no_normalized_source_group_crosses_controlled_splits")
            or not leakage.get("no_project_commit_pair_crosses_controlled_splits")
            or not leakage.get("no_quarantined_record_in_controlled_splits")):
        raise ValueError("Devign controlled split leakage checks are not clean")
    df_report = _run_task_experiment(
        "defect", defect["train"], defect["validation"], defect["test"], "func", "target",
        extract_defect_features, defect_vector, DEFECT_VECTOR_FEATURE_NAMES,
        EVAL / "defect_c_rf_baseline_report.json",
        MODELS / "defect_c_rf_improved.joblib",
        EVAL / "defect_c_generalization_experiment.json", positive_label=True,
    )
    df_report["dataset_validation"] = {
        "quarantined_record_count": defect_report["quarantined_record_count"],
        "quarantined_group_count": defect_report["quarantined_group_count"],
        "leakage_verification": leakage,
    }
    (EVAL / "defect_c_generalization_experiment.json").write_text(
        json.dumps(df_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cx_report, df_report


def train_complexity(cx, prepared=None):
    prepared = prepared or prepare_complexity_dataset(cx)
    splits = prepared["splits"]
    dataset_summary = prepared["summary"]
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
        + f"schema-only RandomForest validation accuracy: {acc_schema:.3f}\n"
        + "\nDataset quarantine and split summary:\n"
        + json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n")
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
    parser = argparse.ArgumentParser(description="Train schema-feature random-forest baselines")
    parser.add_argument("--task", choices=("all", "complexity", "defect", "experiments"),
                        default="all")
    selected_task = parser.parse_args().task
    if selected_task == "complexity":
        print(json.dumps(train_prepared_complexity_baseline(), indent=2))
    elif selected_task == "defect":
        print(json.dumps(train_prepared_defect_baseline(), indent=2))
    elif selected_task == "experiments":
        complexity_result, defect_result = run_generalization_experiments()
        print(json.dumps({"complexity": complexity_result, "defect": defect_result}, indent=2))
    else:
        complexity_result, defect_result = train_prepared_baselines()
        print("complexity baseline:", complexity_result["validation"], complexity_result["test"])
        print("defect baseline:", defect_result["validation"], defect_result["test"])
