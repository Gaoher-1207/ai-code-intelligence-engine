"""Read-only loading, validation, and split checks for the ML datasets.

This module never downloads data, writes prepared datasets, or changes labels.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from numbers import Integral
import re
import tokenize
from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


HAJERA_ROOT = Path(__file__).resolve().parents[1]
COMPLEXITY_PATH_ENV = "CODEINTEL_CODECOMPLEX_PATH"
DEVIGN_DATASET_ENV = "CODEINTEL_DEVIGN_DATASET"
DEFAULT_DEVIGN_DATASET = "google/code_x_glue_cc_defect_detection"
COMPLEXITY_REQUIRED_COLUMNS = frozenset({"src", "complexity", "problem"})
DEVIGN_REQUIRED_COLUMNS = frozenset({"id", "func", "target", "project", "commit_id"})
DEVIGN_SPLITS = ("train", "validation", "test")
_C_TOKENS = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|"
    r"(?P<string>\"(?:\\.|[^\"\\])*\")|"
    r"(?P<char>'(?:\\.|[^'\\])*')|"
    r"(?P<identifier>[A-Za-z_]\w*)|"
    r"(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|"
    r"(?P<operator>->|\+\+|--|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|[^\s])"
)
_DEVIGN_C_WHITESPACE_TOKENS = re.compile(
    r'//[^\r\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|[A-Za-z_]\w*|"
    r"(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?[uUlLfF]*)|"
    r">>=|<<=|\.\.\.|->|\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|"
    r"\+=|-=|\*=|/=|%=|&=|\|=|\^=|##|[^\s]",
    re.S,
)


class DatasetValidationError(ValueError):
    """Raised when an input dataset cannot be safely used as supplied."""


def resolve_complexity_path(path: str | Path | None = None) -> Path:
    """Resolve the JSONL path without assuming a machine-specific location."""
    selected = path or os.environ.get(COMPLEXITY_PATH_ENV)
    result = Path(selected).expanduser() if selected else HAJERA_ROOT / "data" / "python_data.jsonl"
    return result.resolve()


def _validate_nonempty_strings(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    missing_columns = columns - set(frame.columns)
    if missing_columns:
        raise DatasetValidationError(
            f"{context} is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    issues = []
    for column in sorted(columns):
        values = frame[column]
        null_count = int(values.isna().sum())
        bad_type_count = int((~values.map(lambda value: isinstance(value, str))).sum())
        empty_count = int(values.map(lambda value: isinstance(value, str) and not value.strip()).sum())
        if null_count or bad_type_count or empty_count:
            issues.append(
                f"{column}: null={null_count}, non-string={bad_type_count}, empty={empty_count}"
            )
    if issues:
        raise DatasetValidationError(f"Invalid {context} records: " + "; ".join(issues))


def validate_complexity_frame(frame: pd.DataFrame) -> None:
    """Validate required fields while preserving every source label verbatim."""
    if not isinstance(frame, pd.DataFrame):
        raise DatasetValidationError("Complexity data must be a pandas DataFrame")
    if frame.empty:
        raise DatasetValidationError("Complexity dataset contains no records")
    _validate_nonempty_strings(frame, set(COMPLEXITY_REQUIRED_COLUMNS), "complexity")


def load_complexity_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load CodeComplex JSONL; malformed lines fail with their line numbers."""
    source = resolve_complexity_path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"CodeComplex JSONL not found at {source}. Set {COMPLEXITY_PATH_ENV} "
            "or pass an explicit path. Raw datasets are not downloaded or copied."
        )
    with source.open("r", encoding="utf-8") as stream:
        frame = parse_complexity_jsonl(stream)
    return frame


def parse_complexity_jsonl(lines: Any) -> pd.DataFrame:
    """Parse JSONL lines in memory and reject malformed records without dropping them."""
    records: list[dict[str, Any]] = []
    malformed: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            malformed.append(f"line {line_number}: blank line")
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            malformed.append(f"line {line_number}: {exc.msg}")
            continue
        if not isinstance(record, dict):
            malformed.append(f"line {line_number}: expected a JSON object")
            continue
        records.append(record)
    if malformed:
        preview = "; ".join(malformed[:5])
        extra = f" (and {len(malformed) - 5} more)" if len(malformed) > 5 else ""
        raise DatasetValidationError(f"Malformed CodeComplex JSONL: {preview}{extra}")
    frame = pd.DataFrame.from_records(records)
    validate_complexity_frame(frame)
    return frame


def _source_hash(source: str) -> str:
    # Normalize only line endings. Whitespace and literals can affect code semantics.
    canonical = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_source_hash(source: str, language: str) -> str:
    """Hash token streams while ignoring formatting and comments, preserving literals/names."""
    if language == "python":
        ignored = {
            tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER,
        }
        tokens = [
            (item.type, item.string)
            for item in tokenize.generate_tokens(io.StringIO(source).readline)
            if item.type not in ignored
        ]
    elif language == "c":
        tokens = [match.group(0) for match in _C_TOKENS.finditer(source)
                  if match.lastgroup != "comment"]
    else:
        tokens = source.split()
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode("utf-8")).hexdigest()


def devign_whitespace_source_key(source: str) -> tuple[str, ...]:
    """Return conservative C tokens for Devign grouping.

    Inter-token whitespace is ignored, while comments and string/character literal
    contents are retained. This deliberately differs from the older report helper,
    which also ignores comments, and matches the audited Devign quarantine policy.
    """
    if not isinstance(source, str):
        raise DatasetValidationError("Devign source code must be a string")
    return tuple(_DEVIGN_C_WHITESPACE_TOKENS.findall(source))


def _duplicate_summary(frame: pd.DataFrame, source_column: str) -> dict[str, int]:
    by_hash: dict[str, list[int]] = {}
    for index, source in enumerate(frame[source_column].tolist()):
        by_hash.setdefault(_source_hash(source), []).append(index)
    duplicate_groups = [indices for indices in by_hash.values() if len(indices) > 1]
    conflict_groups = 0
    cross_problem_groups = 0
    if "complexity" in frame.columns:
        conflict_groups = sum(
            frame.iloc[indices]["complexity"].nunique(dropna=False) > 1
            for indices in duplicate_groups
        )
        cross_problem_groups = sum(
            frame.iloc[indices]["problem"].nunique(dropna=False) > 1
            for indices in duplicate_groups
        )
    return {
        "duplicate_source_groups": len(duplicate_groups),
        "duplicate_source_records": sum(map(len, duplicate_groups)),
        "conflicting_duplicate_source_groups": int(conflict_groups),
        "cross_problem_duplicate_source_groups": int(cross_problem_groups),
    }


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in frame.isna().sum().items()}


def complexity_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize labels and duplicate source without deleting or relabeling rows."""
    validate_complexity_frame(frame)
    labels = frame["complexity"].value_counts(dropna=False).sort_index()
    return {
        "record_count": len(frame),
        "columns": list(frame.columns),
        "missing_counts": _missing_counts(frame),
        "label_counts": {str(label): int(count) for label, count in labels.items()},
        "problem_count": int(frame["problem"].nunique()),
        **_duplicate_summary(frame, "src"),
    }


def _complexity_conflict_partition(
    frame: pd.DataFrame,
) -> tuple[pd.Series, set[str], set[str], int]:
    exact_groups: dict[str, list[int]] = {}
    normalized_groups: dict[str, list[int]] = {}
    for position, source in enumerate(frame["src"].tolist()):
        exact_groups.setdefault(_source_hash(source), []).append(position)
        normalized_groups.setdefault(_normalized_source_hash(source, "python"), []).append(position)

    exact_conflicts = {
        source_hash
        for source_hash, positions in exact_groups.items()
        if frame.iloc[positions]["complexity"].nunique(dropna=False) > 1
    }
    normalized_conflicts = {
        source_hash
        for source_hash, positions in normalized_groups.items()
        if frame.iloc[positions]["complexity"].nunique(dropna=False) > 1
    }
    quarantined_positions = {
        position
        for source_hash, positions in normalized_groups.items()
        if source_hash in normalized_conflicts
        for position in positions
    }
    mask = pd.Series(False, index=frame.index)
    if quarantined_positions:
        mask.iloc[list(quarantined_positions)] = True
    normalized_conflict_record_count = sum(
        len(positions)
        for source_hash, positions in normalized_groups.items()
        if source_hash in normalized_conflicts
    )
    return mask, exact_conflicts, normalized_conflicts, normalized_conflict_record_count


def prepare_complexity_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Quarantine contradictory normalized-source groups, then split clean records."""
    validate_complexity_frame(frame)
    quarantine_mask, exact_conflicts, normalized_conflicts, \
        normalized_conflict_record_count = _complexity_conflict_partition(frame)
    quarantined = frame.loc[quarantine_mask].copy()
    modeling = frame.loc[~quarantine_mask].copy()

    if len(exact_conflicts) != 59 or len(normalized_conflicts) != 59 \
            or len(quarantined) != 127 or len(modeling) != 4773:
        raise DatasetValidationError(
            "CodeComplex conflict profile changed: expected 59 exact groups, 59 normalized "
            "groups, 127 quarantined records, and 4773 modeling records; found "
            f"{len(exact_conflicts)}, {len(normalized_conflicts)}, {len(quarantined)}, "
            f"and {len(modeling)}. Review before training."
        )
    missing_labels = set(frame["complexity"]) - set(modeling["complexity"])
    if missing_labels:
        raise DatasetValidationError(
            "Quarantine removed the only records for complexity labels: "
            + ", ".join(sorted(missing_labels))
        )

    splits = _split_complexity_records(modeling)
    quarantine_hashes = {_source_hash(source) for source in quarantined["src"]}
    quarantine_normalized_hashes = {
        _normalized_source_hash(source, "python") for source in quarantined["src"]
    }
    for split_name, split_frame in splits.items():
        leaked = quarantine_hashes & {_source_hash(source) for source in split_frame["src"]}
        if leaked:
            raise DatasetValidationError(
                f"Quarantined source groups entered the {split_name} split ({len(leaked)} groups)."
            )
        normalized_leaked = quarantine_normalized_hashes & {
            _normalized_source_hash(source, "python") for source in split_frame["src"]
        }
        if normalized_leaked:
            raise DatasetValidationError(
                f"Quarantined normalized-source groups entered the {split_name} split "
                f"({len(normalized_leaked)} groups)."
            )
    _, _, remaining_normalized_conflicts, _ = _complexity_conflict_partition(modeling)
    if remaining_normalized_conflicts:
        raise DatasetValidationError(
            "Normalized-source label conflicts remain in the modeling pool "
            f"({len(remaining_normalized_conflicts)} groups)."
        )

    return {
        "splits": splits,
        "modeling": modeling,
        "quarantined": quarantined,
        "quarantined_group_ids": sorted(normalized_conflicts),
        "summary": {
            "original_record_count": len(frame),
            "exact_conflicting_group_count": len(exact_conflicts),
            "normalized_conflicting_group_count": len(normalized_conflicts),
            "normalized_conflicting_record_count": normalized_conflict_record_count,
            "exact_conflicting_record_count": int(
                frame["src"].map(lambda source: _source_hash(source) in exact_conflicts).sum()
            ),
            "quarantined_group_count": len(normalized_conflicts),
            "quarantined_record_count": len(quarantined),
            "modeling_record_count": len(modeling),
            "original_label_counts": {
                str(label): int(count)
                for label, count in frame["complexity"].value_counts().sort_index().items()
            },
            "modeling_label_counts": {
                str(label): int(count)
                for label, count in modeling["complexity"].value_counts().sort_index().items()
            },
            "quarantined_label_counts": {
                str(label): int(count)
                for label, count in quarantined["complexity"].value_counts().sort_index().items()
            },
            "split_record_counts": {name: len(part) for name, part in splits.items()},
            "split_label_counts": {
                name: {
                    str(label): int(count)
                    for label, count in part["complexity"].value_counts().sort_index().items()
                }
                for name, part in splits.items()
            },
        },
    }


def split_complexity_dataset(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    validation_fraction_of_train: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Quarantine conflicting normalized-source groups, then split by problem.

    The outer split deliberately matches the historical GroupShuffleSplit used by the
    saved complexity model. Exact and normalized source overlap is checked at both
    split boundaries; the function fails instead of allowing duplicate code across splits.
    """
    validate_complexity_frame(frame)
    if not 0 < test_size < 1 or not 0 < validation_fraction_of_train < 1:
        raise ValueError("Split fractions must be strictly between 0 and 1")
    quarantine_mask, _, _, _ = _complexity_conflict_partition(frame)
    modeling = frame.loc[~quarantine_mask]
    if modeling.empty:
        raise DatasetValidationError("No CodeComplex records remain after conflict quarantine")
    return _split_complexity_records(
        modeling,
        random_state=random_state,
        test_size=test_size,
        validation_fraction_of_train=validation_fraction_of_train,
    )


def _split_complexity_records(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    validation_fraction_of_train: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Split a conflict-free frame by problem and reject source overlap."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_validation_indices, test_indices = next(
        outer.split(frame, frame["complexity"], groups=frame["problem"])
    )
    train_validation = frame.iloc[train_validation_indices]
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction_of_train,
        random_state=random_state + 1,
    )
    train_relative, validation_relative = next(
        inner.split(
            train_validation,
            train_validation["complexity"],
            groups=train_validation["problem"],
        )
    )
    indices = {
        "train": train_validation_indices[train_relative],
        "validation": train_validation_indices[validation_relative],
        "test": test_indices,
    }
    hashes = [_source_hash(source) for source in frame["src"]]
    normalized_hashes = [_normalized_source_hash(source, "python") for source in frame["src"]]
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {hashes[i] for i in indices[left]} & {hashes[i] for i in indices[right]}
        if overlap:
            raise DatasetValidationError(
                f"Exact source-code duplicates cross the {left}/{right} split boundary "
                f"({len(overlap)} groups); no records were removed."
            )
        normalized_overlap = (
            {normalized_hashes[i] for i in indices[left]}
            & {normalized_hashes[i] for i in indices[right]}
        )
        if normalized_overlap:
            raise DatasetValidationError(
                f"Formatting/comment-normalized Python code crosses the {left}/{right} "
                f"split boundary ({len(normalized_overlap)} groups); no records were removed."
            )
    return {name: frame.iloc[index].reset_index(drop=True) for name, index in indices.items()}


def _to_frame(split: Any, name: str) -> pd.DataFrame:
    if isinstance(split, pd.DataFrame):
        return split.copy()
    if hasattr(split, "to_pandas"):
        return split.to_pandas()
    raise DatasetValidationError(f"Devign {name} split must support to_pandas()")


def validate_devign_frame(frame: pd.DataFrame, split_name: str = "dataset") -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DatasetValidationError(f"Devign {split_name} split must be a pandas DataFrame")
    if frame.empty:
        raise DatasetValidationError(f"Devign {split_name} split contains no records")
    missing = DEVIGN_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetValidationError(
            f"Devign {split_name} split is missing required columns: {', '.join(sorted(missing))}"
        )
    for column in ("func", "project", "commit_id"):
        _validate_nonempty_strings(frame, {column}, f"Devign {split_name}")
    labels = frame["target"]
    if labels.isna().any():
        raise DatasetValidationError(f"Devign {split_name} target contains missing labels")
    valid_label = labels.map(
        lambda value: isinstance(value, (bool,))
        or (isinstance(value, Integral) and not isinstance(value, bool) and value in (0, 1))
        or (type(value).__module__ == "numpy" and type(value).__name__ == "bool_")
    )
    if not bool(valid_label.all()):
        raise DatasetValidationError(
            f"Devign {split_name} target must contain only boolean/0/1 labels"
        )
    if "id" in frame.columns and frame["id"].isna().any():
        raise DatasetValidationError(f"Devign {split_name} id contains missing values")
    if "id" in frame.columns and not frame["id"].map(
        lambda value: isinstance(value, Integral) and not isinstance(value, bool)
    ).all():
        raise DatasetValidationError(f"Devign {split_name} id must contain integer identifiers")


def load_devign_splits(
    dataset: Mapping[str, Any] | None = None,
    *,
    dataset_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all official Devign splits, offline-only, or validate supplied split data.

    Supplying ``dataset`` keeps tests and local workflows independent of Hugging Face.
    The default loader is cache-only and never downloads the dataset.
    """
    if dataset is None:
        name = dataset_name or os.environ.get(DEVIGN_DATASET_ENV, DEFAULT_DEVIGN_DATASET)
        try:
            from datasets import DownloadConfig, load_dataset
        except ImportError as exc:
            raise ImportError(
                "Devign loading requires the 'datasets' package. Install dependencies "
                "from hajera/requirements.txt; this loader will not download data."
            ) from exc
        dataset = load_dataset(
            name,
            download_config=DownloadConfig(local_files_only=True),
        )
    missing_splits = set(DEVIGN_SPLITS) - set(dataset)
    if missing_splits:
        raise DatasetValidationError(
            f"Devign dataset is missing official splits: {', '.join(sorted(missing_splits))}"
        )
    frames = {name: _to_frame(dataset[name], name) for name in DEVIGN_SPLITS}
    for name, frame in frames.items():
        validate_devign_frame(frame, name)
    return frames


def devign_quality_report(splits: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Report per-split labels, duplicates, and cross-split exact-code overlap."""
    validated = load_devign_splits(splits)
    report: dict[str, Any] = {"splits": {}}
    hashes_by_split: dict[str, set[str]] = {}
    normalized_hashes_by_split: dict[str, set[str]] = {}
    for name, frame in validated.items():
        hashes = [_source_hash(code) for code in frame["func"]]
        normalized_hashes = [_normalized_source_hash(code, "c") for code in frame["func"]]
        hashes_by_split[name] = set(hashes)
        normalized_hashes_by_split[name] = set(normalized_hashes)
        counts = pd.Series(hashes).value_counts()
        normalized_counts = pd.Series(normalized_hashes).value_counts()
        duplicate_groups = int((counts > 1).sum())
        duplicate_records = int(counts[counts > 1].sum())
        report["splits"][name] = {
            "record_count": len(frame),
            "missing_counts": _missing_counts(frame),
            "label_counts": {
                str(label): int(count)
                for label, count in frame["target"].value_counts(dropna=False).sort_index().items()
            },
            "duplicate_source_groups": duplicate_groups,
            "duplicate_source_records": duplicate_records,
            "formatting_normalized_duplicate_groups": int((normalized_counts > 1).sum()),
            "formatting_normalized_duplicate_records": int(normalized_counts[normalized_counts > 1].sum()),
            "duplicate_id_records": int(frame["id"].duplicated(keep=False).sum()),
        }
    report["cross_split_exact_source_overlap"] = {
        f"{left}/{right}": len(hashes_by_split[left] & hashes_by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    report["cross_split_formatting_normalized_overlap"] = {
        f"{left}/{right}": len(normalized_hashes_by_split[left] & normalized_hashes_by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    return report


def ensure_no_devign_exact_overlap(splits: Mapping[str, pd.DataFrame]) -> None:
    """Fail training readiness on duplicate code across official splits; do not move rows."""
    report = devign_quality_report(splits)
    overlaps = {
        name: count
        for category in ("cross_split_exact_source_overlap", "cross_split_formatting_normalized_overlap")
        for name, count in report[category].items()
        if count
    }
    if overlaps:
        detail = ", ".join(f"{name}: {count}" for name, count in sorted(overlaps.items()))
        raise DatasetValidationError(
            "Duplicate or formatting/comment-normalized code crosses official Devign split boundaries ("
            f"{detail}). Official rows were not changed; review before training."
        )
