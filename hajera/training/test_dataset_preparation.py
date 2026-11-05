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
    prepare_complexity_dataset,
    split_complexity_dataset,
    validate_devign_frame,
    _complexity_conflict_partition,
    _normalized_source_hash,
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


def full_complexity_quarantine_records():
    records = []
    for group_index in range(59):
        if group_index == 57:
            labels = ["constant", "logn", "logn"]
        elif group_index == 58:
            labels = ["cubic", "cubic", "quadratic", "quadratic"]
        else:
            labels = [
                COMPLEXITY_LABELS[group_index % len(COMPLEXITY_LABELS)],
                COMPLEXITY_LABELS[(group_index + 1) % len(COMPLEXITY_LABELS)],
            ]
        source = f"def conflict_{group_index}():\n    return {group_index}\n"
        for record_index, label in enumerate(labels):
            records.append(
                {
                    "src": source,
                    "complexity": label,
                    "problem": f"conflict-{group_index}-{record_index}",
                    "from": "CODEFORCES",
                    "tags": "implementation",
                }
            )
        if group_index == 0:
            variant_sources = [source + "\n", source + "\n\n"]
        elif group_index == 1:
            variant_sources = [source + "  \n", source + " \n"]
        elif group_index == 2:
            variant_sources = [source + "\n", source + "\n\n"]
        else:
            variant_sources = []
        for variant_index, variant_source in enumerate(variant_sources):
            records.append(
                {
                    "src": variant_source,
                    "complexity": labels[-1],
                    "problem": f"conflict-{group_index}-variant-{variant_index}",
                    "from": "CODEFORCES",
                    "tags": "implementation",
                }
            )
    for record_index in range(4773):
        records.append(
            {
                "src": f"def clean_{record_index}():\n    return {record_index}\n",
                "complexity": COMPLEXITY_LABELS[record_index % len(COMPLEXITY_LABELS)],
                "problem": f"clean-{record_index}",
                "from": "CODEFORCES",
                "tags": "implementation",
            }
        )
    return pd.DataFrame(records)


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


def test_complexity_quarantine_detects_59_groups_and_preserves_original_rows():
    frame = full_complexity_quarantine_records()
    original = frame.copy(deep=True)

    prepared = prepare_complexity_dataset(frame)

    assert prepared["summary"]["original_record_count"] == 4900
    assert prepared["summary"]["exact_conflicting_group_count"] == 59
    assert prepared["summary"]["normalized_conflicting_group_count"] == 59
    assert prepared["summary"]["exact_conflicting_record_count"] == 121
    assert prepared["summary"]["quarantined_group_count"] == 59
    assert prepared["summary"]["normalized_conflicting_record_count"] == 127
    assert prepared["summary"]["quarantined_record_count"] == 127
    assert prepared["summary"]["modeling_record_count"] == 4773
    assert len(prepared["quarantined_group_ids"]) == 59
    assert set(prepared["modeling"]["complexity"]) == set(COMPLEXITY_LABELS)
    assert all(set(part["complexity"]) == set(COMPLEXITY_LABELS)
               for part in prepared["splits"].values())
    assert sum(prepared["summary"]["split_record_counts"].values()) == 4773

    quarantined_sources = set(prepared["quarantined"]["src"])
    quarantined_normalized = {
        _normalized_source_hash(source, "python") for source in quarantined_sources
    }
    for split in prepared["splits"].values():
        assert quarantined_sources.isdisjoint(split["src"])
        assert quarantined_normalized.isdisjoint(
            {_normalized_source_hash(source, "python") for source in split["src"]}
        )
    _, _, remaining_conflicts, _ = _complexity_conflict_partition(prepared["modeling"])
    assert not remaining_conflicts
    pd.testing.assert_frame_equal(frame, original)
    restored = pd.concat([prepared["modeling"], prepared["quarantined"]]).sort_index()
    pd.testing.assert_frame_equal(restored, original)


def test_complexity_quarantine_membership_and_splits_are_deterministic():
    frame = full_complexity_quarantine_records()

    first = prepare_complexity_dataset(frame)
    second = prepare_complexity_dataset(frame)

    assert first["quarantined_group_ids"] == second["quarantined_group_ids"]
    pd.testing.assert_frame_equal(first["quarantined"], second["quarantined"])
    for name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first["splits"][name], second["splits"][name])


def test_complexity_split_detects_normalized_conflicts_and_checks_final_overlap():
    records = complexity_records()
    records.extend(
        [
            {"src": "x = 1\n", "complexity": "constant", "problem": "normalized-shared"},
            {"src": "x=1  # formatting-only variant\n", "complexity": "logn", "problem": "normalized-shared"},
        ]
    )
    frame = pd.DataFrame(records)

    mask, exact_conflicts, normalized_conflicts, normalized_record_count = \
        _complexity_conflict_partition(frame)
    splits = split_complexity_dataset(frame)

    assert not exact_conflicts
    assert len(normalized_conflicts) == 1
    assert normalized_record_count == 2
    assert mask.sum() == 2
    assert set(splits) == {"train", "validation", "test"}
    assert sum(len(part) for part in splits.values()) == 80
    assert not any(
        _normalized_source_hash(source, "python") == next(iter(normalized_conflicts))
        for part in splits.values()
        for source in part["src"]
    )
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_hashes = {_normalized_source_hash(source, "python") for source in splits[left]["src"]}
        right_hashes = {_normalized_source_hash(source, "python") for source in splits[right]["src"]}
        assert left_hashes.isdisjoint(right_hashes)


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
