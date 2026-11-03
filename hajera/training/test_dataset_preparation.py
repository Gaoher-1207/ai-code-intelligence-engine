"""Offline, in-memory checks for dataset preparation and split integrity."""
import json

import pandas as pd
import pytest

from dataset_preparation import (
    DatasetValidationError,
    complexity_quality_report,
    devign_quality_report,
    ensure_no_devign_exact_overlap,
    parse_complexity_jsonl,
    split_complexity_dataset,
    validate_devign_frame,
)


COMPLEXITY_LABELS = [
    "constant", "logn", "linear", "nlogn", "quadratic", "cubic", "np"
]


def complexity_records():
    return [
        {
            "src": f"print('program {index}')\n",
            "complexity": COMPLEXITY_LABELS[index % len(COMPLEXITY_LABELS)],
            "problem": f"problem-{index // 2}",
            "from": "CODEFORCES",
            "tags": "implementation",
        }
        for index in range(80)
    ]


def devign_frame(suffix, labels=(False, True, False, True), funcs=None):
    funcs = funcs or [f"void func_{suffix}_{i}(void) {{ return; }}" for i in range(len(labels))]
    return pd.DataFrame(
        {
            "id": list(range(len(labels))),
            "func": funcs,
            "target": list(labels),
            "project": ["project-a"] * len(labels),
            "commit_id": [f"commit-{suffix}-{i}" for i in range(len(labels))],
        }
    )


def test_complexity_jsonl_loader_preserves_all_seven_labels():
    lines = [json.dumps(record) for record in complexity_records()[:7]]
    frame = parse_complexity_jsonl(lines)

    assert frame["complexity"].tolist() == COMPLEXITY_LABELS
    assert set(frame.columns) == {"src", "complexity", "problem", "from", "tags"}


def test_complexity_jsonl_rejects_malformed_records_without_dropping_them():
    lines = [json.dumps(complexity_records()[0]), "{not json}"]

    with pytest.raises(DatasetValidationError, match="line 2"):
        parse_complexity_jsonl(lines)


def test_complexity_missing_required_value_fails_validation():
    lines = [json.dumps({"src": "x = 1", "problem": "p1"})]

    with pytest.raises(DatasetValidationError, match="complexity"):
        parse_complexity_jsonl(lines)


def test_complexity_duplicate_and_conflicting_labels_are_reported_separately():
    frame = pd.DataFrame(
        [
            {"src": "same code\n", "complexity": "linear", "problem": "p1"},
            {"src": "same code\r\n", "complexity": "np", "problem": "p2"},
            {"src": "unique code\n", "complexity": "constant", "problem": "p3"},
        ]
    )

    report = complexity_quality_report(frame)

    assert report["record_count"] == 3
    assert report["duplicate_source_groups"] == 1
    assert report["duplicate_source_records"] == 2
    assert report["conflicting_duplicate_source_groups"] == 1
    assert report["cross_problem_duplicate_source_groups"] == 1
    assert set(report["label_counts"]) == {"linear", "np", "constant"}


def test_complexity_split_is_reproducible_and_keeps_problems_and_code_disjoint():
    frame = parse_complexity_jsonl([json.dumps(record) for record in complexity_records()])

    splits = split_complexity_dataset(frame)
    repeated = split_complexity_dataset(frame)

    assert list(splits) == ["train", "validation", "test"]
    assert sum(len(part) for part in splits.values()) == len(frame)
    for name in splits:
        assert splits[name]["src"].tolist() == repeated[name]["src"].tolist()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(splits[left]["problem"]).isdisjoint(splits[right]["problem"])
        assert set(splits[left]["src"]).isdisjoint(splits[right]["src"])


def test_devign_validation_accepts_official_boolean_labels():
    frame = devign_frame("validation")

    validate_devign_frame(frame, "validation")

    report = devign_quality_report(
        {"train": devign_frame("train"), "validation": frame, "test": devign_frame("test")}
    )
    assert report["splits"]["validation"]["record_count"] == 4
    assert report["splits"]["validation"]["missing_counts"]["func"] == 0
    assert report["splits"]["validation"]["label_counts"] == {"False": 2, "True": 2}


def test_devign_missing_source_or_label_fails_validation():
    frame = devign_frame("bad").copy()
    frame.loc[0, "func"] = None

    with pytest.raises(DatasetValidationError, match="func"):
        validate_devign_frame(frame, "train")

    frame = devign_frame("bad-label")
    frame["target"] = frame["target"].astype(object)
    frame.loc[0, "target"] = "vulnerable"
    with pytest.raises(DatasetValidationError, match="target"):
        validate_devign_frame(frame, "train")


def test_devign_exact_duplicate_code_across_official_splits_is_reported_and_fails_closed():
    duplicate = "void shared(void) { return; }"
    splits = {
        "train": devign_frame("train", funcs=[duplicate, "void train1(){}", "void train2(){}", "void train3(){}"]),
        "validation": devign_frame("validation", funcs=[duplicate, "void val1(){}", "void val2(){}", "void val3(){}"]),
        "test": devign_frame("test"),
    }

    report = devign_quality_report(splits)
    assert report["cross_split_exact_source_overlap"]["train/validation"] == 1
    with pytest.raises(DatasetValidationError, match="Official rows were not changed"):
        ensure_no_devign_exact_overlap(splits)


def test_devign_formatting_or_comment_only_variants_are_reported_as_near_duplicates():
    splits = {
        "train": devign_frame(
            "train",
            funcs=["void shared(void) { return; }", "void train1(){}", "void train2(){}", "void train3(){}"],
        ),
        "validation": devign_frame(
            "validation",
            funcs=["void shared(void){ /* comment */ return; }", "void val1(){}", "void val2(){}", "void val3(){}"],
        ),
        "test": devign_frame("test"),
    }

    report = devign_quality_report(splits)
    assert report["cross_split_exact_source_overlap"]["train/validation"] == 0
    assert report["cross_split_formatting_normalized_overlap"]["train/validation"] == 1
    with pytest.raises(DatasetValidationError, match="Duplicate or formatting"):
        ensure_no_devign_exact_overlap(splits)


def test_devign_official_split_mapping_is_required():
    with pytest.raises(DatasetValidationError, match="official splits"):
        devign_quality_report({"train": devign_frame("train")})
