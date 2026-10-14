"""Tests for importing ADR-0002 ground truth from an answers CSV (synthetic data only)."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import ocrbench.dataset as dataset_module
from ocrbench.cli import main
from ocrbench.dataset import (
    DatasetError,
    DocEntry,
    Manifest,
    dataset_fingerprint,
    load_manifest,
    validate_split_minimums,
)
from ocrbench.edi_answers import ANSWER_COLUMNS, ground_truth_bytes, import_answers
from ocrbench.edi_fields import EdiFields

_CUSTOMER = "Nidecマシンツール株式会社"
_SECRET_PART = "ZZ-SECRET-PART"
_SECRET_ORDER = "SECRET-ORDER-1"

_ROWS: tuple[tuple[str, ...], ...] = (
    ("a.png", "scan", "dev", _CUSTOMER, "HR000000001", "000001-2025", "1 PC"),
    ("b.jpg", "photo", "dev", f" {_CUSTOMER} ", "HR000000002", "000002-2026", "4PC"),
    ("c.png", "scan", "final", _CUSTOMER, "HR000000003", "000003-2026", "\uff12"),
    ("d.png", "photo", "final", _CUSTOMER, "HR000000004", "000004-2025", "10"),
)


def _source_root(tmp_path: Path, names: Sequence[str]) -> Path:
    root = tmp_path / "dataset"
    (root / "raw").mkdir(parents=True)
    for index, name in enumerate(names):
        (root / "raw" / name).write_bytes(f"synthetic sheet {index}".encode())
    return root


def _write_answers(
    path: Path,
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = ANSWER_COLUMNS,
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _raw_names(root: Path) -> list[str]:
    return sorted(path.name for path in (root / "raw").iterdir())


def test_import_answers_anonymizes_and_writes_edi_ground_truth(tmp_path: Path) -> None:
    root = _source_root(tmp_path, [row[0] for row in _ROWS])
    answers = _write_answers(tmp_path / "answers.csv", _ROWS)

    manifest = import_answers(root, answers)

    assert len(manifest.docs) == 4
    assert all(name.startswith("doc-") for name in _raw_names(root))
    by_split = sorted((entry.split, entry.source_kind) for entry in manifest.docs)
    assert by_split == [("dev", "photo"), ("dev", "scan"), ("final", "photo"), ("final", "scan")]
    quantities = sorted(
        json.loads((root / "gt" / f"{entry.doc_id}.json").read_text(encoding="utf-8"))["quantity"]
        for entry in manifest.docs
    )
    assert quantities == [1, 2, 4, 10]
    orders = {
        json.loads((root / "gt" / f"{entry.doc_id}.json").read_text(encoding="utf-8"))["order_no"]
        for entry in manifest.docs
    }
    assert "000001-2025" in orders  # stored as printed; the year rule applies when scoring
    assert load_manifest(root, ground_truth="edi_fields") == manifest
    assert len(dataset_fingerprint(root, manifest)) == 64
    assert validate_split_minimums(manifest, minimum=2, strict=True) == []


def test_import_answers_reports_every_row_problem_without_values(tmp_path: Path) -> None:
    names = ["a.png", "b.png", "c.png", "d.png", "e.png"]
    root = _source_root(tmp_path, names)
    rows = [
        ("a.png", "fax", "selection", _CUSTOMER, _SECRET_PART, _SECRET_ORDER, "1"),
        ("b.png", "scan", "dev", "Unknown Corp", _SECRET_PART, _SECRET_ORDER, "1"),
        ("c.png", "scan", "dev", _CUSTOMER, "  ", _SECRET_ORDER, "0 PC"),
        ("d.png", "scan", "dev", _CUSTOMER, _SECRET_PART, "", ""),
        ("", "", "", "", "", "", ""),
        ("e.png", "scan", "dev"),
    ]
    answers = _write_answers(tmp_path / "answers.csv", rows)

    with pytest.raises(DatasetError) as caught:
        import_answers(root, answers)

    assert caught.value.violations == (
        "CSV row 2: source_kind must be 'scan' or 'photo'",
        "CSV row 2: split must be 'dev' or 'final'",
        "CSV row 3: customer: no EDI site is mapped for this name",
        "CSV row 4: quantity must be a positive whole number, optionally with a unit such as PC",
        "CSV row 4: part_no: missing",
        "CSV row 5: order_no: missing",
        "CSV row 5: quantity: missing",
        "CSV row 6: blank row; delete it",
        "CSV row 7: expected 7 values, found 3",
    )
    message = str(caught.value)
    for secret in (_SECRET_PART, _SECRET_ORDER, "Unknown Corp"):
        assert secret not in message
    assert _raw_names(root) == names
    assert not (root / "gt").exists()
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (
            ("file", "source_kind", "split", "customer", "part_no", "order_no"),
            "answers CSV is missing columns: quantity",
        ),
        (
            (*ANSWER_COLUMNS, "note"),
            "answers CSV has unsupported columns: note",
        ),
        (
            (*ANSWER_COLUMNS, "file"),
            "answers CSV has duplicate columns: file",
        ),
    ],
)
def test_import_answers_rejects_bad_headers(
    tmp_path: Path, header: tuple[str, ...], expected: str
) -> None:
    root = _source_root(tmp_path, ["a.png"])
    answers = _write_answers(tmp_path / "answers.csv", [], header=header)

    with pytest.raises(DatasetError) as caught:
        import_answers(root, answers)

    assert caught.value.violations == (expected,)


def test_import_answers_rejects_empty_and_header_only_files(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png"])
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    header_only = _write_answers(tmp_path / "header.csv", [])

    with pytest.raises(DatasetError, match="answers CSV is empty"):
        import_answers(root, empty)
    with pytest.raises(DatasetError, match="no data rows"):
        import_answers(root, header_only)
    with pytest.raises(DatasetError, match="cannot read answers CSV"):
        import_answers(root, tmp_path / "missing.csv")


def test_import_answers_writes_nothing_when_a_raw_file_is_missing(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png"])
    answers = _write_answers(tmp_path / "answers.csv", _ROWS[:2])

    with pytest.raises(DatasetError) as caught:
        import_answers(root, answers)

    assert any("raw/b.jpg is missing" in item for item in caught.value.violations)
    assert _raw_names(root) == ["a.png"]
    assert not (root / "gt").exists()


def test_import_answers_refuses_to_replace_different_ground_truth(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png"])
    answers = _write_answers(tmp_path / "answers.csv", _ROWS[:1])
    doc_id = "doc-" + dataset_module._sha256_file(root / "raw" / "a.png")[:12]
    (root / "gt").mkdir()
    (root / "gt" / f"{doc_id}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetError, match="already exists with different content"):
        import_answers(root, answers)

    assert _raw_names(root) == ["a.png"]
    assert (root / "gt" / f"{doc_id}.json").read_text(encoding="utf-8") == "{}"


def test_import_answers_keeps_identical_existing_ground_truth(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png"])
    answers = _write_answers(tmp_path / "answers.csv", _ROWS[:1])
    doc_id = "doc-" + dataset_module._sha256_file(root / "raw" / "a.png")[:12]
    expected = ground_truth_bytes(
        EdiFields(customer=_CUSTOMER, part_no="HR000000001", order_no="000001-2025", quantity=1)
    )
    (root / "gt").mkdir()
    (root / "gt" / f"{doc_id}.json").write_bytes(expected)

    manifest = import_answers(root, answers)

    assert [entry.doc_id for entry in manifest.docs] == [doc_id]
    assert (root / "gt" / f"{doc_id}.json").read_bytes() == expected


def test_import_answers_removes_created_ground_truth_on_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _source_root(tmp_path, [row[0] for row in _ROWS])
    answers = _write_answers(tmp_path / "answers.csv", _ROWS)

    def fail_link(source: Path, destination: Path) -> None:
        raise OSError("simulated link failure")

    monkeypatch.setattr(dataset_module, "_link_no_overwrite", fail_link)

    with pytest.raises(DatasetError, match="rolled back"):
        import_answers(root, answers)

    assert _raw_names(root) == sorted(row[0] for row in _ROWS)
    assert list((root / "gt").iterdir()) == []
    assert not (root / "manifest.json").exists()


def test_build_manifest_requires_ground_truth_for_every_row(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png", "b.png"])
    assignments = tmp_path / "assignments.csv"
    assignments.write_text("file,source_kind,split\na.png,scan,dev\nb.png,photo,dev\n")

    with pytest.raises(DatasetError) as caught:
        dataset_module.build_manifest(root, assignments, ground_truth_by_row={2: b"{}\n"})

    assert caught.value.violations == ("CSV row 3: no ground truth was supplied",)
    assert _raw_names(root) == ["a.png", "b.png"]


def test_load_manifest_checks_edi_ground_truth_is_scorable(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png"])
    import_answers(root, _write_answers(tmp_path / "answers.csv", _ROWS[:1]))
    (gt_path,) = (root / "gt").iterdir()

    gt_path.write_text(
        json.dumps({"customer": "Unknown", "part_no": "X", "order_no": "1", "quantity": 1}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="cannot be scored: customer: no EDI site"):
        load_manifest(root, ground_truth="edi_fields")

    gt_path.write_text(json.dumps({"customer": _CUSTOMER}), encoding="utf-8")
    with pytest.raises(DatasetError, match="is invalid"):
        load_manifest(root, ground_truth="edi_fields")


def test_default_load_manifest_still_expects_order_documents(tmp_path: Path) -> None:
    root = _source_root(tmp_path, ["a.png"])
    import_answers(root, _write_answers(tmp_path / "answers.csv", _ROWS[:1]))

    with pytest.raises(DatasetError, match="is invalid"):
        load_manifest(root)


def _entry(index: int, split: str, source_kind: str) -> DocEntry:
    sha256 = f"{index:064x}"
    doc_id = f"doc-{sha256[:12]}"
    return DocEntry.model_validate(
        {
            "doc_id": doc_id,
            "source_kind": source_kind,
            "file": f"{doc_id}.png",
            "sha256": sha256,
            "split": split,
        }
    )


def test_validate_split_minimums_counts_and_warnings() -> None:
    manifest = Manifest(
        schema_version=1,
        docs=[
            _entry(1, "dev", "scan"),
            _entry(2, "dev", "scan"),
            _entry(3, "final", "scan"),
            _entry(4, "final", "photo"),
            _entry(5, "selection", "photo"),
        ],
    )

    expected = [
        "split 'selection' is not used by ADR-0002, found 1",
        "split 'final': expected at least 3 documents, found 2",
        "split 'dev': expected at least 3 documents, found 2",
        "split 'dev' has no 'photo' documents",
    ]
    result = validate_split_minimums(manifest, minimum=3, strict=False)
    assert sorted(result) == sorted(expected)
    with pytest.raises(DatasetError) as caught:
        validate_split_minimums(manifest, minimum=3, strict=True)
    assert "split 'dev' has no 'photo' documents" not in caught.value.violations

    warnings_only = validate_split_minimums(
        manifest.model_copy(update={"docs": manifest.docs[:4]}), minimum=2, strict=True
    )
    assert warnings_only == ["split 'dev' has no 'photo' documents"]
    with pytest.raises(ValueError, match="at least 1"):
        validate_split_minimums(manifest, minimum=0, strict=False)


def test_cli_import_validate_and_fingerprint_edi_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _source_root(tmp_path, [row[0] for row in _ROWS])
    answers = _write_answers(tmp_path / "answers.csv", _ROWS)

    assert main(["dataset", "import-answers", "--root", str(root), "--answers", str(answers)]) == 0
    assert "Imported answers and built manifest for 4 documents." in capsys.readouterr().out

    validate = ["dataset", "validate", "--root", str(root), "--ground-truth", "edi_fields"]
    assert main([*validate, "--min-per-split", "2", "--strict-counts"]) == 0
    assert "Dataset is valid (4 documents)." in capsys.readouterr().out

    assert main([*validate, "--min-per-split", "3", "--strict-counts"]) != 0
    assert "expected at least 3 documents" in capsys.readouterr().err

    fingerprint = ["dataset", "fingerprint", "--root", str(root), "--ground-truth", "edi_fields"]
    assert main(fingerprint) == 0
    assert len(capsys.readouterr().out.strip()) == 64

    assert main([*validate, "--min-per-split", "0"]) != 0
    assert "must be at least 1" in capsys.readouterr().err
