"""Tests for the T25 synthetic sheet generator (all values invented)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.dataset import load_manifest, validate_split_minimums
from ocrbench.edi_answers import import_answers, read_answers_csv
from scripts.make_synthetic_sheets import (
    REPOSITORY_ROOT,
    SyntheticSheetError,
    find_font,
    generate,
    main,
    plan_sheets,
)


def _font_or_skip() -> Path:
    try:
        return find_font(None)
    except SyntheticSheetError:
        pytest.skip("no Japanese font installed")


def test_plan_is_deterministic_and_covers_both_splits_and_kinds() -> None:
    sheets = plan_sheets(10, seed=7)
    assert sheets == plan_sheets(10, seed=7)
    assert sheets != plan_sheets(10, seed=8)
    counts = Counter((sheet.split, sheet.source_kind) for sheet in sheets)
    assert counts == {
        ("dev", "scan"): 2,
        ("dev", "photo"): 2,
        ("final", "scan"): 3,
        ("final", "photo"): 3,
    }
    assert len({sheet.part_no for sheet in sheets}) == 10
    assert all(sheet.file.endswith(".jpg") == (sheet.source_kind == "photo") for sheet in sheets)


@pytest.mark.parametrize(
    ("count", "dev_share", "photo_share"), [(1, 0.4, 0.5), (4, 0.0, 0.5), (4, 0.5, 1.5)]
)
def test_plan_rejects_bad_arguments(count: int, dev_share: float, photo_share: float) -> None:
    with pytest.raises(SyntheticSheetError):
        plan_sheets(count, seed=1, dev_share=dev_share, photo_share=photo_share)


def test_output_must_be_outside_the_repository_and_empty(tmp_path: Path) -> None:
    with pytest.raises(SyntheticSheetError, match="outside the repository"):
        generate(REPOSITORY_ROOT / "synthetic-out", count=2)
    assert not (REPOSITORY_ROOT / "synthetic-out").exists()
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SyntheticSheetError, match="empty"):
        generate(tmp_path / "busy", count=2)


def test_missing_font_is_reported(tmp_path: Path) -> None:
    with pytest.raises(SyntheticSheetError, match="--font"):
        find_font(tmp_path / "missing.ttf")


def test_generated_sheets_import_as_a_valid_edi_dataset(tmp_path: Path) -> None:
    font = _font_or_skip()
    out = tmp_path / "synthetic"
    sheets = generate(out, count=6, font=font)

    rows = read_answers_csv(out / "answers.csv")
    assert [row.file for row in rows] == [sheet.file for sheet in sheets]
    assert {row.fields.quantity for row in rows} == {
        int("".join(ch for ch in sheet.quantity_printed if ch.isdigit())) for sheet in sheets
    }
    with Image.open(out / "raw" / sheets[0].file) as image:
        assert image.size == (1240, 1754)

    manifest = import_answers(out, out / "answers.csv")
    assert len(manifest.docs) == 6
    assert load_manifest(out, ground_truth="edi_fields") == manifest
    assert validate_split_minimums(manifest, minimum=2, strict=True) == []


def test_main_reports_success_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    font = _font_or_skip()
    assert main(["--out", str(tmp_path / "a"), "--count", "3", "--font", str(font)]) == 0
    assert "Wrote 3 synthetic sheets (1 dev, 2 final)" in capsys.readouterr().out
    assert main(["--out", str(tmp_path / "a"), "--count", "3", "--font", str(font)]) == 2
    assert "empty" in capsys.readouterr().err


@pytest.mark.ocr_engines
@pytest.mark.parametrize(("remove_lines", "expected"), [(False, False), (True, True)])
def test_tesseract_needs_line_removal_on_a_ruled_synthetic_scan(
    tmp_path: Path, remove_lines: bool, expected: bool
) -> None:
    """Tesseract skips text inside ruled table cells unless the lines are removed."""
    from ocrbench.ocr_engines import TesseractEngine
    from ocrbench.preprocess import preprocess

    out = tmp_path / "synthetic"
    sheets = generate(out, count=2, photo_share=0.0, font=_font_or_skip())
    engine = TesseractEngine(page_segmentation_mode=4, remove_lines=remove_lines)
    text = engine.transcribe(preprocess(out / "raw" / sheets[0].file)).text
    assert (sheets[0].part_no in text.replace(" ", "")) is expected
