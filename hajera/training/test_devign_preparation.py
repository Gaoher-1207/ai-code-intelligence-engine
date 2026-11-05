"""Focused in-memory regression tests for controlled Devign preparation."""
from __future__ import annotations

import pandas as pd
import pytest

from dataset_preparation import DatasetValidationError, devign_whitespace_source_key
from devign_preparation import prepare_devign_dataset


DEVIGN_COLUMNS = ("id", "func", "target", "project", "commit_id", "metadata")


def synthetic_official_splits():
    records = []
    record_id = 0

    # Enough independent records in every label/project stratum for a three-way split.
    for project in ("FFmpeg", "qemu"):
        for target in (False, True):
            for index in range(36):
                records.append(
                    {
                        "id": record_id,
                        "func": f"int independent_{project}_{target}_{index}(void) {{ return {index}; }}",
                        "target": target,
                        "project": project,
                        "commit_id": f"commit-{project}-{target}-{index}",
                        "metadata": {"retained": True, "index": index},
                    }
                )
                record_id += 1

    # These two programs differ only in whitespace and have contradictory labels.
    records.extend(
        [
            {
                "id": record_id,
                "func": "int conflict(void) { return 7; }",
                "target": False,
                "project": "FFmpeg",
                "commit_id": "conflict-a",
                "metadata": {"origin": "first"},
            },
            {
                "id": record_id + 1,
                "func": "int conflict(void){return 7;}\n",
                "target": True,
                "project": "FFmpeg",
                "commit_id": "conflict-b",
                "metadata": {"origin": "second"},
            },
        ]
    )
    record_id += 2

    # A commit edge connects record A to B; a normalized-source edge connects B to C.
    records.extend(
        [
            {
                "id": record_id,
                "func": "void bridge_a(void) { return; }",
                "target": False,
                "project": "qemu",
                "commit_id": "shared-commit",
                "metadata": {"chain": "a"},
            },
            {
                "id": record_id + 1,
                "func": "void bridge_b(void) { return; }",
                "target": False,
                "project": "qemu",
                "commit_id": "shared-commit",
                "metadata": {"chain": "b"},
            },
            {
                "id": record_id + 2,
                "func": "void bridge_b(void){return;}\n",
                "target": False,
                "project": "qemu",
                "commit_id": "other-commit",
                "metadata": {"chain": "c"},
            },
        ]
    )

    frames = {name: [] for name in ("train", "validation", "test")}
    for index, record in enumerate(records):
        frames[("train", "validation", "test")[index % 3]].append(record)
    return {
        name: pd.DataFrame.from_records(rows, columns=DEVIGN_COLUMNS)
        for name, rows in frames.items()
    }


def test_devign_normalization_ignores_whitespace_but_preserves_comments_and_literals():
    assert devign_whitespace_source_key("int f(void){return 1;}") == devign_whitespace_source_key(
        "int f ( void ) { return 1 ; }\n"
    )
    assert devign_whitespace_source_key("int f(){ /* one */ return 1; }") != devign_whitespace_source_key(
        "int f(){ /* two */ return 1; }"
    )
    assert devign_whitespace_source_key('char *f(){return "a";}') != devign_whitespace_source_key(
        'char *f(){return "b";}'
    )


def test_devign_quarantine_preserves_original_records_and_retains_same_label_duplicates():
    official = synthetic_official_splits()
    snapshots = {split: frame.copy(deep=True) for split, frame in official.items()}
    original_by_id = {
        int(row["id"]): row
        for frame in official.values()
        for row in frame.to_dict(orient="records")
    }

    prepared = prepare_devign_dataset(official)

    assert prepared["summary"]["original_record_count"] == 149
    assert prepared["summary"]["quarantined_group_count"] == 1
    assert prepared["summary"]["quarantined_record_count"] == 2
    assert prepared["summary"]["retained_record_count"] == 147
    assert prepared["summary"]["connected_components"]["count"] < 147
    assert len(prepared["quarantined_group_ids"]) == 1

    quarantine = prepared["quarantine_records"]
    assert {item["quarantine_reason"] for item in quarantine} == {
        "normalized_source_group_has_conflicting_target_labels"
    }
    for item in quarantine:
        record_id = int(item["record"]["id"])
        assert item["record"] == original_by_id[record_id]
        assert item["group_original_labels"] == [0, 1]

    split_ids = {
        split: {int(record_id) for record_id in frame["id"]}
        for split, frame in prepared["splits"].items()
    }
    assert not ({int(item["record"]["id"]) for item in quarantine} & set.union(*split_ids.values()))

    retained_same_label_ids = {147, 148}
    containing_splits = [split for split, ids in split_ids.items() if retained_same_label_ids <= ids]
    assert len(containing_splits) == 1
    assert prepared["summary"]["leakage_verification"][
        "no_normalized_source_group_crosses_controlled_splits"
    ]
    assert prepared["summary"]["leakage_verification"][
        "no_project_commit_pair_crosses_controlled_splits"
    ]
    assert prepared["summary"]["leakage_verification"]["no_connected_component_is_divided"]
    assert prepared["summary"]["leakage_verification"][
        "no_quarantined_record_in_controlled_splits"
    ]

    for split in official:
        pd.testing.assert_frame_equal(official[split], snapshots[split])
    assert sum(len(frame) for frame in prepared["splits"].values()) == 147


def test_devign_commit_and_source_edges_form_transitive_components():
    prepared = prepare_devign_dataset(synthetic_official_splits())
    split_for_id = {
        int(record_id): split
        for split, frame in prepared["splits"].items()
        for record_id in frame["id"]
    }
    # The three bridge records are connected transitively by commit and source edges.
    bridge_ids = {146, 147, 148}
    assert len({split_for_id[value] for value in bridge_ids}) == 1


def test_devign_assignments_are_deterministic_and_preserve_labels_and_projects():
    official = synthetic_official_splits()
    first = prepare_devign_dataset(official)
    second = prepare_devign_dataset(official)

    assert first["quarantined_group_ids"] == second["quarantined_group_ids"]
    assert first["component_ids_by_split"] == second["component_ids_by_split"]
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first["splits"][split], second["splits"][split])
        assert set(first["splits"][split]["target"]) == {False, True}
        assert set(first["splits"][split]["project"]) == {"FFmpeg", "qemu"}


def test_devign_preparation_rejects_duplicate_record_ids():
    official = synthetic_official_splits()
    duplicate = official["validation"].iloc[[0]].copy()
    duplicate.loc[:, "id"] = official["train"].iloc[0]["id"]
    official["validation"] = pd.concat([official["validation"], duplicate], ignore_index=True)

    with pytest.raises(DatasetValidationError, match="IDs are not globally unique"):
        prepare_devign_dataset(official)
