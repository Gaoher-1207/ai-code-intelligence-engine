import numpy as np

from schema_features import (complexity_vector, defect_vector,
                             extract_complexity_features, extract_defect_features)
from train_schema_models import (_classification_metrics, _select_validation_candidate,
                                 select_random_forest)


def test_baseline_vectors_match_prediction_feature_dimensions():
    assert len(complexity_vector(extract_complexity_features("def f(n): return n"))) == 17
    assert len(defect_vector(extract_defect_features("int f(void) { return 0; }"))) == 13


def test_random_forest_selection_is_reproducible_and_uses_validation_metrics():
    x_train = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    y_train = np.asarray([0, 0, 1, 1, 2, 2])
    x_validation = np.asarray([[0, 0], [1, 0], [2, 0]])
    y_validation = np.asarray([0, 1, 2])

    first, first_trees, first_metrics = select_random_forest(
        x_train, y_train, x_validation, y_validation
    )
    second, second_trees, second_metrics = select_random_forest(
        x_train, y_train, x_validation, y_validation
    )

    assert first_trees == second_trees
    assert first_metrics == second_metrics
    assert np.array_equal(first.predict(x_validation), second.predict(x_validation))
    assert first_metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_defect_metrics_include_pr_auc_roc_auc_and_per_class_results():
    metrics = _classification_metrics(
        np.asarray([False, False, True, True]),
        np.asarray([False, True, True, True]), [False, True],
        np.asarray([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]), True,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert set(metrics["per_class"]) == {"False", "True"}


def test_experiment_selection_uses_validation_macro_f1_then_accuracy():
    candidates = [
        {"name": "accuracy_winner", "order": 0,
         "validation": {"macro_f1": 0.40, "accuracy": 0.70}},
        {"name": "f1_winner", "order": 1,
         "validation": {"macro_f1": 0.45, "accuracy": 0.60}},
        {"name": "tie_later", "order": 2,
         "validation": {"macro_f1": 0.45, "accuracy": 0.60}},
    ]

    assert _select_validation_candidate(candidates)["name"] == "f1_winner"
