"""Import ADR-0002 ground truth from one answers CSV per dataset root.

The CSV has one row per sheet, typed exactly as printed::

    file,source_kind,split,customer,part_no,order_no,quantity
    sheet-001.jpg,scan,dev,Nidecマシンツール株式会社,HR237101021,023235-2025,1 PC

``file`` names a file in ``<root>/raw/``. Importing validates every row,
anonymizes the files with :func:`ocrbench.dataset.build_manifest`, and writes
``gt/<doc-id>.json`` as :class:`~ocrbench.edi_fields.EdiFields`. Error messages
name rows and columns but never echo a sheet's values.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ocrbench.dataset import DatasetError, Manifest, build_manifest, load_manifest
from ocrbench.edi_fields import EdiFields, ground_truth_problems
from ocrbench.normalize import normalize_num_pieces, strip_outer_whitespace

ANSWER_COLUMNS: Final[tuple[str, ...]] = (
    "file",
    "source_kind",
    "split",
    "customer",
    "part_no",
    "order_no",
    "quantity",
)
_SOURCE_KINDS: Final = frozenset({"scan", "photo"})
_SPLITS: Final = frozenset({"dev", "final"})


@dataclass(frozen=True)
class AnswerRow:
    """One validated answers CSV row; ``row_number`` counts the header as 1."""

    row_number: int
    file: str
    source_kind: str
    split: str
    fields: EdiFields


def _optional_text(value: str) -> str | None:
    return strip_outer_whitespace(value) or None


def _parse_row(row_number: int, values: dict[str, str]) -> tuple[AnswerRow | None, list[str]]:
    prefix = f"CSV row {row_number}"
    problems: list[str] = []
    if not values["file"].strip():
        problems.append(f"{prefix}: file: missing")
    if values["source_kind"] not in _SOURCE_KINDS:
        problems.append(f"{prefix}: source_kind must be 'scan' or 'photo'")
    if values["split"] not in _SPLITS:
        problems.append(f"{prefix}: split must be 'dev' or 'final'")

    quantity: int | None = None
    if values["quantity"].strip():
        quantity = normalize_num_pieces(values["quantity"])
        if quantity is None:
            problems.append(
                f"{prefix}: quantity must be a positive whole number, "
                "optionally with a unit such as PC"
            )
    fields = EdiFields(
        customer=_optional_text(values["customer"]),
        part_no=_optional_text(values["part_no"]),
        order_no=_optional_text(values["order_no"]),
        quantity=quantity,
    )
    for problem in ground_truth_problems(fields):
        if problem == "quantity: missing" and values["quantity"].strip():
            continue
        problems.append(f"{prefix}: {problem}")
    if problems:
        return None, problems
    return (
        AnswerRow(row_number, values["file"], values["source_kind"], values["split"], fields),
        [],
    )


def read_answers_csv(answers_csv: Path) -> list[AnswerRow]:
    """Read and validate every row, raising one :class:`DatasetError` listing all problems."""
    rows: list[AnswerRow] = []
    violations: list[str] = []
    try:
        with answers_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None:
                raise DatasetError("answers CSV is empty")
            duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
            missing = [name for name in ANSWER_COLUMNS if name not in header]
            extra = sorted(set(header) - set(ANSWER_COLUMNS))
            if duplicates:
                violations.append("answers CSV has duplicate columns: " + ", ".join(duplicates))
            if missing:
                violations.append("answers CSV is missing columns: " + ", ".join(missing))
            if extra:
                violations.append("answers CSV has unsupported columns: " + ", ".join(extra))
            if violations:
                raise DatasetError(violations)

            for row_number, values in enumerate(reader, start=2):
                if not any(value.strip() for value in values):
                    violations.append(f"CSV row {row_number}: blank row; delete it")
                    continue
                if len(values) != len(header):
                    violations.append(
                        f"CSV row {row_number}: expected {len(header)} values, found {len(values)}"
                    )
                    continue
                row, problems = _parse_row(row_number, dict(zip(header, values, strict=True)))
                violations.extend(problems)
                if row is not None:
                    rows.append(row)
    except (OSError, UnicodeError, csv.Error) as error:
        raise DatasetError(f"cannot read answers CSV: {error}") from error
    if not violations and not rows:
        violations.append("answers CSV has no data rows")
    if violations:
        raise DatasetError(violations)
    return rows


def ground_truth_bytes(fields: EdiFields) -> bytes:
    """Serialize ground truth deterministically for ``gt/<doc-id>.json``."""
    text = json.dumps(fields.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def import_answers(root: Path, answers_csv: Path) -> Manifest:
    """Validate the answers CSV, anonymize ``raw/``, and write EDI ground truth.

    Nothing is renamed or written unless every row is valid. The result is
    re-loaded with ``ground_truth="edi_fields"`` before it is returned.
    """
    rows = read_answers_csv(answers_csv)
    with tempfile.TemporaryDirectory(prefix="ocrbench-answers-") as temp_dir:
        # Every answers row is a data row, so row numbers match in both files.
        assignments = Path(temp_dir) / "assignments.csv"
        with assignments.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("file", "source_kind", "split"))
            for row in rows:
                writer.writerow((row.file, row.source_kind, row.split))
        build_manifest(
            root,
            assignments,
            ground_truth_by_row={row.row_number: ground_truth_bytes(row.fields) for row in rows},
        )
    return load_manifest(root, ground_truth="edi_fields")
