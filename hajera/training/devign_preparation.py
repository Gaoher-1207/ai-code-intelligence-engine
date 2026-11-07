"""Prepare a leakage-controlled Devign dataset from cached Arrow IPC shards.

The Hugging Face cache is opened read-only with PyArrow. The original official
splits are never rewritten; prepared artifacts are written only under ``hajera``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from dataset_preparation import (
    DEVIGN_REQUIRED_COLUMNS,
    DEVIGN_SPLITS,
    DatasetValidationError,
    HAJERA_ROOT,
    devign_whitespace_source_key,
    validate_devign_frame,
)


DEVIGN_DATASET_CACHE_FOLDER = "google___code_x_glue_cc_defect_detection"
DEVIGN_ARROW_PREFIX = "code_x_glue_cc_defect_detection"
EXPECTED_OFFICIAL_SPLIT_COUNTS = {"train": 21854, "validation": 2732, "test": 2732}
EXPECTED_CONFLICT_GROUPS = 85
EXPECTED_QUARANTINE_RECORDS = 170
DEFAULT_OUTPUT_DIR = HAJERA_ROOT / "data" / "devign_prepared"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
QUARANTINE_REASON = "normalized_source_group_has_conflicting_target_labels"


def _arrow_filenames() -> dict[str, str]:
    return {
        split: f"{DEVIGN_ARROW_PREFIX}-{split}.arrow"
        for split in DEVIGN_SPLITS
    }


def _contains_devign_shards(path: Path) -> bool:
    return all((path / name).is_file() for name in _arrow_filenames().values())


def resolve_devign_arrow_dir(cache_dir: str | Path | None = None) -> Path:
    """Find the existing local Devign Arrow shards without invoking Hub loading."""
    if cache_dir is not None:
        requested = Path(cache_dir).expanduser().resolve()
        if _contains_devign_shards(requested):
            return requested
        roots = [requested]
    else:
        roots: list[Path] = []
        datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        hf_home = os.environ.get("HF_HOME")
        if datasets_cache:
            roots.append(Path(datasets_cache).expanduser())
        if hf_home:
            roots.append(Path(hf_home).expanduser() / "datasets")
        roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if _contains_devign_shards(root):
            return root

        dataset_root = root / DEVIGN_DATASET_CACHE_FOLDER / "default"
        if dataset_root.is_dir():
            for candidate in sorted(dataset_root.iterdir()):
                if candidate.is_dir() and _contains_devign_shards(candidate):
                    return candidate.resolve()

        # Also support a caller-supplied cache root with a different HF fingerprint.
        if root.is_dir():
            for candidate in root.rglob(_arrow_filenames()["train"]):
                if candidate.is_file() and _contains_devign_shards(candidate.parent):
                    return candidate.parent.resolve()

    searched = ", ".join(str(path) for path in roots)
    raise FileNotFoundError(
        "Could not find cached Devign Arrow IPC shards. Searched: " + searched
    )


def _cache_fingerprints(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash each cached source shard using read-only sequential reads."""
    fingerprints: dict[str, dict[str, Any]] = {}
    for filename in _arrow_filenames().values():
        path = cache_dir / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        fingerprints[filename] = {
            "size_bytes": stat.st_size,
            "sha256": digest.hexdigest(),
            "mtime_ns": stat.st_mtime_ns,
        }
    return fingerprints


def load_devign_arrow_splits(
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Read the cached Arrow IPC streams directly, without cache locks or downloads."""
    resolved_dir = resolve_devign_arrow_dir(cache_dir)
    frames: dict[str, pd.DataFrame] = {}
    for split, filename in _arrow_filenames().items():
        path = resolved_dir / filename
        try:
            with pa.memory_map(str(path), "r") as source:
                table = ipc.open_stream(source).read_all()
        except (pa.ArrowException, OSError) as exc:
            raise DatasetValidationError(
                f"Could not read cached Devign {split} Arrow IPC stream at {path}: {exc}"
            ) from exc
        missing = DEVIGN_REQUIRED_COLUMNS - set(table.schema.names)
        if missing:
            raise DatasetValidationError(
                f"Cached Devign {split} shard is missing columns: {', '.join(sorted(missing))}"
            )
        frame = table.to_pandas()
        validate_devign_frame(frame, split)
        frames[split] = frame
    return frames


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def _stable_group_id(source_key: tuple[str, ...]) -> str:
    token_json = json.dumps(source_key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(token_json.encode("utf-8")).hexdigest()


def _stable_component_id(record_ids: Sequence[int]) -> str:
    payload = json.dumps(sorted(int(record_id) for record_id in record_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _distribution(records: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    counts = Counter(str(record[column]) for record in records)
    return dict(sorted(counts.items()))


def _official_split_overlap(
    frames: Mapping[str, pd.DataFrame],
    key_fn,
) -> dict[str, int]:
    keys_by_split = {
        split: {key_fn(record) for record in frame.to_dict(orient="records")}
        for split, frame in frames.items()
    }
    return {
        f"{left}/{right}": len(keys_by_split[left] & keys_by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }


def _component_assignment(
    component_members: Sequence[Sequence[int]],
    retained_records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Greedily balance whole components, with stable ties and deterministic refinement."""
    project_names = sorted({str(record["project"]) for record in retained_records})
    metric_names = ["records", "secure", "insecure"] + [f"project:{name}" for name in project_names]
    metric_count = len(metric_names)
    component_vectors = np.zeros((len(component_members), metric_count), dtype=np.int64)
    tie_hashes: list[str] = []

    for component_index, member_indices in enumerate(component_members):
        ids: list[int] = []
        for member_index in member_indices:
            record = retained_records[member_index]
            component_vectors[component_index, 0] += 1
            component_vectors[component_index, 1] += int(not bool(record["target"]))
            component_vectors[component_index, 2] += int(bool(record["target"]))
            project_metric = 3 + project_names.index(str(record["project"]))
            component_vectors[component_index, project_metric] += 1
            ids.append(int(record["id"]))
        tie_hashes.append(hashlib.sha256("|".join(map(str, ids)).encode("ascii")).hexdigest())

    proportions = np.asarray([SPLIT_PROPORTIONS[name] for name in SPLIT_NAMES], dtype=float)
    totals = np.asarray(
        [
            len(retained_records),
            sum(not bool(record["target"]) for record in retained_records),
            sum(bool(record["target"]) for record in retained_records),
            *[sum(str(record["project"]) == project for record in retained_records)
              for project in project_names],
        ],
        dtype=float,
    )
    targets = proportions[:, None] * totals[None, :]
    scales = np.maximum(targets, 1.0)
    current = np.zeros_like(targets)
    assignments = np.full(len(component_members), -1, dtype=np.int8)

    order = sorted(
        range(len(component_members)),
        key=lambda index: (-int(component_vectors[index, 0]), tie_hashes[index]),
    )
    for component_index in order:
        candidate_deltas = []
        for split_index in range(len(SPLIT_NAMES)):
            after = current[split_index] + component_vectors[component_index]
            delta = np.sum(
                ((after - targets[split_index]) / scales[split_index]) ** 2
                - ((current[split_index] - targets[split_index]) / scales[split_index]) ** 2
            )
            candidate_deltas.append(float(delta))
        selected = int(np.argmin(candidate_deltas))
        assignments[component_index] = selected
        current[selected] += component_vectors[component_index]

    def objective(values: np.ndarray) -> float:
        return float(np.sum(((values - targets) / scales) ** 2))

    refinement_order = sorted(
        range(len(component_members)),
        key=lambda index: (int(component_vectors[index, 0]), tie_hashes[index]),
    )
    accepted_moves = 0
    for _ in range(4):
        for component_index in refinement_order:
            original_split = int(assignments[component_index])
            best_score = objective(current)
            best_split = original_split
            for candidate_split in range(len(SPLIT_NAMES)):
                if candidate_split == original_split:
                    continue
                trial = current.copy()
                trial[original_split] -= component_vectors[component_index]
                trial[candidate_split] += component_vectors[component_index]
                score = objective(trial)
                if score < best_score - 1e-12:
                    best_score = score
                    best_split = candidate_split
            if best_split != original_split:
                current[original_split] -= component_vectors[component_index]
                current[best_split] += component_vectors[component_index]
                assignments[component_index] = best_split
                accepted_moves += 1

    split_for_component = [SPLIT_NAMES[int(value)] for value in assignments]
    details = {
        "algorithm": (
            "Largest-component-first deterministic greedy assignment minimizing normalized "
            "squared deviation from split targets for total records, labels, and projects; "
            "then up to four deterministic improving-move passes."
        ),
        "target_proportions": dict(SPLIT_PROPORTIONS),
        "tie_breaker": "SHA-256 of the ordered original record IDs in each component",
        "random_seed": None,
        "randomization": "None; assignment is deterministic and uses no RNG.",
        "refinement_moves_accepted": accepted_moves,
        "balanced_metrics": metric_names,
        "normalized_balance_objective": objective(current),
    }
    return split_for_component, details


def prepare_devign_dataset(
    official_splits: Mapping[str, pd.DataFrame],
    *,
    expected_conflict_groups: int | None = None,
    expected_quarantine_records: int | None = None,
) -> dict[str, Any]:
    """Quarantine contradictory normalized-source groups and make a grouped split."""
    missing_splits = set(DEVIGN_SPLITS) - set(official_splits)
    if missing_splits:
        raise DatasetValidationError(
            "Devign data is missing official splits: " + ", ".join(sorted(missing_splits))
        )
    frames = {name: official_splits[name] for name in DEVIGN_SPLITS}
    for name, frame in frames.items():
        validate_devign_frame(frame, name)
    columns = list(frames["train"].columns)
    if any(list(frame.columns) != columns for frame in frames.values()):
        raise DatasetValidationError("Official Devign splits have inconsistent record columns")

    all_records: list[dict[str, Any]] = []
    original_split_by_position: list[str] = []
    for split in DEVIGN_SPLITS:
        for record in frames[split].to_dict(orient="records"):
            all_records.append(record)
            original_split_by_position.append(split)
    ids = [int(record["id"]) for record in all_records]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("Devign record IDs are not globally unique")

    source_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for position, record in enumerate(all_records):
        source_groups[devign_whitespace_source_key(record["func"])].append(position)
        exact_hash = hashlib.sha256(record["func"].encode("utf-8")).hexdigest()
        exact_groups[exact_hash].append(position)
    conflicting_groups = {
        key: positions
        for key, positions in source_groups.items()
        if len(positions) > 1
        and len({bool(all_records[position]["target"]) for position in positions}) > 1
    }
    exact_conflicting_groups = {
        key: positions
        for key, positions in exact_groups.items()
        if len(positions) > 1
        and len({bool(all_records[position]["target"]) for position in positions}) > 1
    }
    quarantine_positions = {
        position
        for positions in conflicting_groups.values()
        for position in positions
    }
    if expected_conflict_groups is not None and len(conflicting_groups) != expected_conflict_groups:
        raise DatasetValidationError(
            f"Expected {expected_conflict_groups} normalized conflict groups; "
            f"found {len(conflicting_groups)}. Review the source data before preparation."
        )
    if expected_quarantine_records is not None and len(quarantine_positions) != expected_quarantine_records:
        raise DatasetValidationError(
            f"Expected {expected_quarantine_records} quarantined records; "
            f"found {len(quarantine_positions)}. Review the source data before preparation."
        )

    quarantine_records: list[dict[str, Any]] = []
    for source_key, positions in sorted(
        conflicting_groups.items(), key=lambda item: _stable_group_id(item[0])
    ):
        group_id = _stable_group_id(source_key)
        source_labels = sorted({int(bool(all_records[position]["target"])) for position in positions})
        for position in positions:
            quarantine_records.append(
                {
                    "quarantine_group_id": group_id,
                    "quarantine_reason": QUARANTINE_REASON,
                    "group_original_labels": source_labels,
                    "original_split": original_split_by_position[position],
                    "record": dict(all_records[position]),
                }
            )

    retained_positions = [
        position for position in range(len(all_records)) if position not in quarantine_positions
    ]
    retained_records = [all_records[position] for position in retained_positions]
    retained_original_splits = [original_split_by_position[position] for position in retained_positions]
    retained_source_keys = [devign_whitespace_source_key(record["func"]) for record in retained_records]

    parent = list(range(len(retained_records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    source_members: dict[tuple[str, ...], list[int]] = defaultdict(list)
    commit_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(retained_records):
        source_members[retained_source_keys[index]].append(index)
        commit_members[(str(record["project"]), str(record["commit_id"]))].append(index)
    for groups in (source_members, commit_members):
        for member_indices in groups.values():
            for index in member_indices[1:]:
                union(member_indices[0], index)

    components_by_root: dict[int, list[int]] = {}
    for index in range(len(retained_records)):
        components_by_root.setdefault(find(index), []).append(index)
    components = list(components_by_root.values())
    component_ids = [
        _stable_component_id([int(retained_records[index]["id"]) for index in members])
        for members in components
    ]
    component_assignment, assignment_details = _component_assignment(components, retained_records)

    split_records: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_component_ids: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    component_split: dict[str, str] = {}
    for component_id, members, split in zip(component_ids, components, component_assignment):
        component_split[component_id] = split
        split_component_ids[split].append(component_id)
        split_records[split].extend(retained_records[index] for index in members)

    split_frames = {
        split: pd.DataFrame.from_records(records, columns=columns).reset_index(drop=True)
        for split, records in split_records.items()
    }

    def cross_split_overlap(key_function) -> dict[str, int]:
        key_sets = {
            split: {key_function(record) for record in records}
            for split, records in split_records.items()
        }
        return {
            f"{left}/{right}": len(key_sets[left] & key_sets[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }

    source_overlap = cross_split_overlap(
        lambda record: devign_whitespace_source_key(record["func"])
    )
    commit_overlap = cross_split_overlap(
        lambda record: (str(record["project"]), str(record["commit_id"]))
    )
    split_by_record_id = {
        int(record["id"]): split
        for split, records in split_records.items()
        for record in records
    }
    component_crossings = sum(
        len({split_by_record_id[int(retained_records[index]["id"])] for index in members}) > 1
        for members in components
    )
    # Independently verify source edges, commit edges, and full graph components.
    if any(source_overlap.values()) or any(commit_overlap.values()):
        raise DatasetValidationError("A source group or project/commit pair crosses controlled splits")
    if component_crossings:
        raise DatasetValidationError("A connected component was divided across controlled splits")

    quarantined_ids = {int(item["record"]["id"]) for item in quarantine_records}
    split_ids = {
        split: {int(record["id"]) for record in records}
        for split, records in split_records.items()
    }
    if quarantined_ids & set.union(*split_ids.values()):
        raise DatasetValidationError("A quarantined record entered a controlled modelling split")
    label_names = {"0": "Secure", "1": "Insecure"}
    required_labels = {False, True}
    projects = {str(record["project"]) for record in retained_records}
    if any(set(record["target"] for record in records) != required_labels for records in split_records.values()):
        raise DatasetValidationError("Both Devign labels must appear in every controlled split")
    if any({str(record["project"]) for record in records} != projects for records in split_records.values()):
        raise DatasetValidationError("Every retained project must appear in every controlled split")

    component_sizes = [len(members) for members in components]
    size_counts = Counter(component_sizes)
    component_has_both_labels = sum(
        len({bool(retained_records[index]["target"]) for index in members}) > 1
        for members in components
    )
    component_spans_original_splits = sum(
        len({retained_original_splits[index] for index in members}) > 1
        for members in components
    )

    def record_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        labels = Counter("1" if bool(record["target"]) else "0" for record in records)
        return {
            "record_count": len(records),
            "label_counts": dict(sorted(labels.items())),
            "label_names": {label_names[label]: count for label, count in sorted(labels.items())},
            "project_counts": _distribution(records, "project"),
        }

    official_summary = {
        split: record_summary(frames[split].to_dict(orient="records"))
        for split in DEVIGN_SPLITS
    }
    official_leakage = {
        "exact_source_overlap_groups": _official_split_overlap(
            frames, lambda record: hashlib.sha256(record["func"].encode("utf-8")).hexdigest()
        ),
        "conservative_whitespace_normalized_source_overlap_groups": _official_split_overlap(
            frames, lambda record: devign_whitespace_source_key(record["func"])
        ),
        "project_commit_pair_overlap_groups": _official_split_overlap(
            frames, lambda record: (str(record["project"]), str(record["commit_id"]))
        ),
    }

    summary = {
        "original_record_count": len(all_records),
        "original_official_split_counts": {
            split: len(frames[split]) for split in DEVIGN_SPLITS
        },
        "quarantined_group_count": len(conflicting_groups),
        "quarantined_record_count": len(quarantine_records),
        "exact_conflicting_group_count": len(exact_conflicting_groups),
        "exact_conflicting_record_count": sum(len(positions) for positions in exact_conflicting_groups.values()),
        "quarantined_label_counts": record_summary(
            [item["record"] for item in quarantine_records]
        )["label_counts"],
        "quarantined_project_counts": record_summary(
            [item["record"] for item in quarantine_records]
        )["project_counts"],
        "retained_record_count": len(retained_records),
        "retained_label_counts": record_summary(retained_records)["label_counts"],
        "retained_project_counts": record_summary(retained_records)["project_counts"],
        "retained_original_split_counts": {
            split: record_summary(
                [record for record, original_split in zip(retained_records, retained_original_splits)
                 if original_split == split]
            )
            for split in DEVIGN_SPLITS
        },
        "controlled_split_counts": {
            split: record_summary(records) for split, records in split_records.items()
        },
        "connected_components": {
            "count": len(components),
            "size_counts_for_1_to_10": {
                str(size): int(size_counts.get(size, 0)) for size in range(1, 11)
            },
            "components_size_11_to_20": sum(count for size, count in size_counts.items() if 11 <= size <= 20),
            "components_size_21_to_50": sum(count for size, count in size_counts.items() if 21 <= size <= 50),
            "components_size_51_to_100": sum(count for size, count in size_counts.items() if 51 <= size <= 100),
            "components_size_over_100": sum(count for size, count in size_counts.items() if size > 100),
            "records_in_components_over_100": sum(size for size in component_sizes if size > 100),
            "largest_component_size": max(component_sizes, default=0),
            "median_component_size": float(np.median(component_sizes)) if component_sizes else 0.0,
            "mean_component_size": float(np.mean(component_sizes)) if component_sizes else 0.0,
            "singleton_components": int(size_counts.get(1, 0)),
            "components_containing_both_labels": int(component_has_both_labels),
            "components_spanning_original_splits": int(component_spans_original_splits),
            "components_per_controlled_split": {
                split: len(component_ids_for_split)
                for split, component_ids_for_split in split_component_ids.items()
            },
        },
        "assignment": assignment_details,
        "quarantine_reason": QUARANTINE_REASON,
        "official_split_leakage_before_quarantine": official_leakage,
        "leakage_verification": {
            "no_normalized_source_group_crosses_controlled_splits": not any(source_overlap.values()),
            "normalized_source_overlap_groups": source_overlap,
            "no_project_commit_pair_crosses_controlled_splits": not any(commit_overlap.values()),
            "project_commit_pair_overlap_groups": commit_overlap,
            "no_connected_component_is_divided": component_crossings == 0,
            "connected_component_crossings": component_crossings,
            "no_quarantined_record_in_controlled_splits": not bool(quarantined_ids & set.union(*split_ids.values())),
            "quarantined_record_split_overlap_count": len(quarantined_ids & set.union(*split_ids.values())),
            "both_labels_in_every_split": all(
                set(record["target"] for record in records) == required_labels
                for records in split_records.values()
            ),
            "both_projects_in_every_split": all(
                {str(record["project"]) for record in records} == projects
                for records in split_records.values()
            ),
        },
        "label_mapping": {"0": "Secure", "1": "Insecure"},
        "dataset_limitations": [
            "The official train/validation/test split has source and project/commit overlap; see official_split_leakage_before_quarantine.",
            "The controlled split measures generalization to unseen commits within FFmpeg and qemu, not generalization to unseen projects.",
            "The audit flagged 753 structurally unusual C records. The lexical feature extractor can process them, but it is not a C parser and those warnings are not ground-truth syntax judgments.",
            "Quarantined records retain their original labels and metadata and are excluded from the controlled modelling split; they are not relabeled or merged.",
        ],
    }

    return {
        "official_splits": frames,
        "splits": split_frames,
        "quarantine_records": quarantine_records,
        "quarantined_group_ids": sorted(_stable_group_id(key) for key in conflicting_groups),
        "component_ids_by_split": split_component_ids,
        "summary": summary,
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_value,
                    allow_nan=False,
                )
            )
            stream.write("\n")


def _assignment_signature(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quarantined_group_ids": prepared["quarantined_group_ids"],
        "quarantined_ids": sorted(
            int(item["record"]["id"]) for item in prepared["quarantine_records"]
        ),
        "split_ids": {
            split: sorted(int(value) for value in frame["id"].tolist())
            for split, frame in prepared["splits"].items()
        },
        "component_ids_by_split": prepared["component_ids_by_split"],
    }


def _verify_input_frames_unchanged(
    before: Mapping[str, pd.DataFrame], after: Mapping[str, pd.DataFrame]
) -> None:
    for split in DEVIGN_SPLITS:
        try:
            pd.testing.assert_frame_equal(before[split], after[split], check_exact=True)
        except AssertionError as exc:
            raise DatasetValidationError(
                f"Preparation changed the in-memory original {split} records"
            ) from exc


def run_devign_preparation(
    *,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read, validate, reproducibly prepare, and write actual cached Devign data."""
    resolved_cache = resolve_devign_arrow_dir(cache_dir)
    resolved_output = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    if not resolved_output.is_relative_to(HAJERA_ROOT.resolve()):
        raise ValueError(f"Devign artifacts must be written inside {HAJERA_ROOT}")
    artifact_names = (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "quarantine.jsonl",
        "preparation_report.json",
    )
    if any((resolved_output / name).exists() for name in artifact_names):
        raise FileExistsError(
            f"Refusing to overwrite existing Devign artifacts in {resolved_output}"
        )

    cache_before = _cache_fingerprints(resolved_cache)
    official = load_devign_arrow_splits(resolved_cache)
    official_snapshots = {split: frame.copy(deep=True) for split, frame in official.items()}
    actual_counts = {split: len(frame) for split, frame in official.items()}
    if actual_counts != EXPECTED_OFFICIAL_SPLIT_COUNTS:
        raise DatasetValidationError(
            "Cached official Devign split counts changed: "
            + json.dumps(actual_counts, sort_keys=True)
        )

    prepared = prepare_devign_dataset(
        official,
        expected_conflict_groups=EXPECTED_CONFLICT_GROUPS,
        expected_quarantine_records=EXPECTED_QUARANTINE_RECORDS,
    )
    repeated = prepare_devign_dataset(
        official,
        expected_conflict_groups=EXPECTED_CONFLICT_GROUPS,
        expected_quarantine_records=EXPECTED_QUARANTINE_RECORDS,
    )
    if _assignment_signature(prepared) != _assignment_signature(repeated):
        raise DatasetValidationError("Repeated Devign preparation produced different assignments")
    _verify_input_frames_unchanged(official_snapshots, official)

    cache_after = _cache_fingerprints(resolved_cache)
    if cache_before != cache_after:
        raise DatasetValidationError("A cached Devign Arrow shard changed during preparation")

    resolved_output.mkdir(parents=True, exist_ok=True)
    for split, frame in prepared["splits"].items():
        _write_jsonl(resolved_output / f"{split}.jsonl", frame.to_dict(orient="records"))
    _write_jsonl(resolved_output / "quarantine.jsonl", prepared["quarantine_records"])

    # Re-read generated JSONL to verify row counts and record schema before publishing report.
    artifact_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        artifact_path = resolved_output / f"{split}.jsonl"
        with artifact_path.open("r", encoding="utf-8") as stream:
            artifact_rows = [json.loads(line) for line in stream]
        artifact_counts[split] = len(artifact_rows)
        if any(set(row) != set(official["train"].columns) for row in artifact_rows):
            raise DatasetValidationError(f"Generated {split} JSONL changed the original record schema")
    with (resolved_output / "quarantine.jsonl").open("r", encoding="utf-8") as stream:
        quarantine_rows = [json.loads(line) for line in stream]
    if len(quarantine_rows) != EXPECTED_QUARANTINE_RECORDS:
        raise DatasetValidationError("Generated quarantine JSONL has an unexpected record count")
    if any(
        not {"quarantine_group_id", "quarantine_reason", "group_original_labels", "original_split", "record"}
        <= set(item)
        for item in quarantine_rows
    ):
        raise DatasetValidationError("Generated quarantine JSONL is missing quarantine metadata")

    cache_after_artifacts = _cache_fingerprints(resolved_cache)
    if cache_before != cache_after_artifacts:
        raise DatasetValidationError("A cached Devign Arrow shard changed while writing artifacts")
    _verify_input_frames_unchanged(official_snapshots, official)

    report = dict(prepared["summary"])
    report.update(
        {
            "dataset": "google/code_x_glue_cc_defect_detection",
            "cache_directory": str(resolved_cache),
            "official_cache_files": cache_before,
            "official_cache_unchanged": cache_before == cache_after_artifacts,
            "in_memory_original_records_unchanged": True,
            "repeated_preparation_assignments_identical": True,
            "prepared_artifact_counts": artifact_counts,
            "quarantine_artifact_count": len(quarantine_rows),
            "artifacts": {
                split: str(resolved_output / f"{split}.jsonl") for split in SPLIT_NAMES
            }
            | {
                "quarantine": str(resolved_output / "quarantine.jsonl"),
                "report": str(resolved_output / "preparation_report.json"),
            },
            "official_split_benchmark": {
                "original_splits_preserved": True,
                "split_counts": prepared["summary"]["original_official_split_counts"],
                "distributions": {
                    split: {
                        "record_count": len(frame),
                        "label_counts": {
                            "0": int((frame["target"] == False).sum()),  # noqa: E712
                            "1": int((frame["target"] == True).sum()),  # noqa: E712
                        },
                        "project_counts": {
                            str(project): int(count)
                            for project, count in frame["project"].value_counts().sort_index().items()
                        },
                    }
                    for split, frame in official.items()
                },
            },
            "official_cache_hashes_after_preparation": cache_after_artifacts,
        }
    )
    with (resolved_output / "preparation_report.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")

    return {"prepared": prepared, "report": report, "output_dir": resolved_output}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", help="Existing directory containing all three cached Arrow shards")
    parser.add_argument("--output-dir", help="Output directory inside hajera/ (default: hajera/data/devign_prepared)")
    args = parser.parse_args(argv)
    result = run_devign_preparation(cache_dir=args.cache_dir, output_dir=args.output_dir)
    print(json.dumps({
        "output_dir": str(result["output_dir"]),
        "quarantined_groups": result["report"]["quarantined_group_count"],
        "quarantined_records": result["report"]["quarantined_record_count"],
        "splits": {
            name: details["record_count"]
            for name, details in result["report"]["controlled_split_counts"].items()
        },
        "leakage_verification": result["report"]["leakage_verification"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the preparation command
    raise SystemExit(main())
